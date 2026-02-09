import sys
import os
import time
import math
import torch
import modal
from pathlib import Path
from torch.utils.cpp_extension import load_inline
from concurrent.futures import ThreadPoolExecutor

app = modal.App("blackwell-attention-benchmark")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "clang")
    .pip_install("torch", "numpy", "ninja")
    .env({
        "TORCH_CUDA_ARCH_LIST": "9.0",
        "CUDA_HOME": "/usr/local/cuda"
    })
)

REMOTE_BENCHMARK_SCRIPT = r"""
import torch
import time
import math
from torch.utils.cpp_extension import load_inline
from concurrent.futures import ThreadPoolExecutor

# ------------------------------------------------------------------------------
# 1. KERNEL SOURCE V6
# ------------------------------------------------------------------------------
CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

constexpr int HEAD_DIM_CKV = 512;
constexpr int HEAD_DIM_KPE = 64;
constexpr int NUM_Q_HEADS = 16;
constexpr int TILE_SIZE = 64;
constexpr int WARP_SIZE = 32;
constexpr int ITEMS_PER_THREAD_CKV = HEAD_DIM_CKV / WARP_SIZE;
constexpr int ITEMS_PER_THREAD_KPE = HEAD_DIM_KPE / WARP_SIZE;

using bfloat16 = __nv_bfloat16;

__global__ void __launch_bounds__(256) dsa_attention_kernel(
    const bfloat16* __restrict__ q_nope,
    const bfloat16* __restrict__ q_pe,
    const bfloat16* __restrict__ ckv_cache,
    const bfloat16* __restrict__ kpe_cache,
    const int32_t* __restrict__ indices,
    bfloat16* __restrict__ output,
    float* __restrict__ lse_out,
    float sm_scale,
    int topk
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int heads_per_warp = 2;
    int h_start = warp_id * heads_per_warp;

    extern __shared__ char smem_buffer[];

    int32_t* smem_indices[2];
    bfloat16* smem_k_nope[2];
    bfloat16* smem_k_pe[2];

    smem_indices[0] = (int32_t*)smem_buffer;
    smem_k_nope[0] = (bfloat16*)(smem_indices[0] + TILE_SIZE);
    smem_k_pe[0] = smem_k_nope[0] + (TILE_SIZE * HEAD_DIM_CKV);

    smem_indices[1] = (int32_t*)(smem_k_pe[0] + (TILE_SIZE * HEAD_DIM_KPE));
    smem_k_nope[1] = (bfloat16*)(smem_indices[1] + TILE_SIZE);
    smem_k_pe[1] = smem_k_nope[1] + (TILE_SIZE * HEAD_DIM_CKV);

    bfloat16 qn_reg[2][ITEMS_PER_THREAD_CKV];
    bfloat16 qp_reg[2][ITEMS_PER_THREAD_KPE];

    #pragma unroll
    for(int h=0; h<heads_per_warp; ++h) {
        int global_head = h_start + h;
        long q_base_ckv = (long)token_idx * NUM_Q_HEADS * HEAD_DIM_CKV + global_head * HEAD_DIM_CKV;
        long q_base_kpe = (long)token_idx * NUM_Q_HEADS * HEAD_DIM_KPE + global_head * HEAD_DIM_KPE;
        for(int i=0; i<ITEMS_PER_THREAD_CKV; ++i)
            qn_reg[h][i] = q_nope[q_base_ckv + i * WARP_SIZE + lane_id];
        for(int i=0; i<ITEMS_PER_THREAD_KPE; ++i)
            qp_reg[h][i] = q_pe[q_base_kpe + i * WARP_SIZE + lane_id];
    }

    float m[2] = {-INFINITY, -INFINITY};
    float l[2] = {0.0f, 0.0f};
    float acc[2][ITEMS_PER_THREAD_CKV];
    #pragma unroll
    for(int h=0; h<2; ++h)
        for(int i=0; i<ITEMS_PER_THREAD_CKV; ++i) acc[h][i] = 0.0f;

    int num_tiles = (topk + TILE_SIZE - 1) / TILE_SIZE;
    int cur_stage = 0;
    int next_stage = 1;
    int total_int4_gather = (TILE_SIZE * (HEAD_DIM_CKV + HEAD_DIM_KPE)) * sizeof(bfloat16) / 16;

    {
        int t = 0;
        int base_k_idx = t * TILE_SIZE;
        if (tid < TILE_SIZE / 4) {
            int4* idx_src = (int4*)(indices + token_idx * topk + base_k_idx);
            int4* idx_dst = (int4*)smem_indices[cur_stage];
            if (base_k_idx + (tid + 1) * 4 <= topk) {
                idx_dst[tid] = idx_src[tid];
            } else {
                int32_t* s_src = (int32_t*)idx_src;
                int32_t* s_dst = (int32_t*)idx_dst;
                #pragma unroll
                for(int j=0; j<4; ++j)
                   if (base_k_idx + tid*4 + j < topk) s_dst[tid*4+j] = s_src[tid*4+j];
                   else s_dst[tid*4+j] = -1;
            }
        }
        __syncthreads();

        for (int i = tid; i < total_int4_gather; i += blockDim.x) {
            int stride = 72;
            int vec_idx = i / stride;
            int offset = i % stride;
            int global_k_id = smem_indices[cur_stage][vec_idx];
            if (global_k_id != -1) {
                void* src = (offset < 64)
                    ? (void*)(ckv_cache + (long)global_k_id * HEAD_DIM_CKV + offset * 8)
                    : (void*)(kpe_cache + (long)global_k_id * HEAD_DIM_KPE + (offset-64) * 8);
                void* dst = (offset < 64)
                    ? (void*)(smem_k_nope[cur_stage] + vec_idx * HEAD_DIM_CKV + offset * 8)
                    : (void*)(smem_k_pe[cur_stage] + vec_idx * HEAD_DIM_KPE + (offset-64) * 8);
                __pipeline_memcpy_async(dst, src, 16);
            }
        }
        __pipeline_commit();
    }

    for (int t = 0; t < num_tiles; ++t) {
        if (t + 1 < num_tiles) {
            int base_k_next = (t + 1) * TILE_SIZE;
            if (tid < TILE_SIZE / 4) {
                int4* idx_src = (int4*)(indices + token_idx * topk + base_k_next);
                int4* idx_dst = (int4*)smem_indices[next_stage];
                if (base_k_next + (tid + 1) * 4 <= topk) idx_dst[tid] = idx_src[tid];
                else {
                     int32_t* s_src = (int32_t*)idx_src;
                     int32_t* s_dst = (int32_t*)idx_dst;
                     #pragma unroll
                     for(int j=0; j<4; ++j)
                        if (base_k_next + tid*4 + j < topk) s_dst[tid*4+j] = s_src[tid*4+j];
                        else s_dst[tid*4+j] = -1;
                }
            }
            __syncthreads();

            for (int i = tid; i < total_int4_gather; i += blockDim.x) {
                int stride = 72;
                int vec_idx = i / stride;
                int offset = i % stride;
                int global_k_id = smem_indices[next_stage][vec_idx];
                if (global_k_id != -1) {
                    void* src = (offset < 64)
                        ? (void*)(ckv_cache + (long)global_k_id * HEAD_DIM_CKV + offset * 8)
                        : (void*)(kpe_cache + (long)global_k_id * HEAD_DIM_KPE + (offset-64) * 8);
                    void* dst = (offset < 64)
                        ? (void*)(smem_k_nope[next_stage] + vec_idx * HEAD_DIM_CKV + offset * 8)
                        : (void*)(smem_k_pe[next_stage] + vec_idx * HEAD_DIM_KPE + (offset-64) * 8);
                    __pipeline_memcpy_async(dst, src, 16);
                }
            }
            __pipeline_commit();
        }

        if (t + 1 < num_tiles) __pipeline_wait_prior(1);
        else __pipeline_wait_prior(0);

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            int global_id = smem_indices[cur_stage][k];
            if (global_id == -1) continue;

            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                float score = 0.0f;
                for (int i = 0; i < ITEMS_PER_THREAD_CKV; ++i)
                    score += __bfloat162float(qn_reg[h][i]) * __bfloat162float(smem_k_nope[cur_stage][k * HEAD_DIM_CKV + i * WARP_SIZE + lane_id]);
                for (int i = 0; i < ITEMS_PER_THREAD_KPE; ++i)
                    score += __bfloat162float(qp_reg[h][i]) * __bfloat162float(smem_k_pe[cur_stage][k * HEAD_DIM_KPE + i * WARP_SIZE + lane_id]);

                for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
                    score += __shfl_down_sync(0xffffffff, score, offset);
                score = __shfl_sync(0xffffffff, score, 0);

                score *= sm_scale;
                float m_new = fmaxf(m[h], score);
                float exp_score = expf(score - m_new);
                float correction = expf(m[h] - m_new);
                l[h] = l[h] * correction + exp_score;
                m[h] = m_new;

                for (int i = 0; i < ITEMS_PER_THREAD_CKV; ++i)
                    acc[h][i] = acc[h][i] * correction + exp_score * __bfloat162float(smem_k_nope[cur_stage][k * HEAD_DIM_CKV + i * WARP_SIZE + lane_id]);
            }
        }

        __syncthreads();
        cur_stage ^= 1;
        next_stage ^= 1;
    }

    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        int global_head = h_start + h;
        if (lane_id == 0) lse_out[token_idx * NUM_Q_HEADS + global_head] = (m[h] + logf(l[h])) / logf(2.0f);

        float inv_l = 1.0f / l[h];
        for (int i = 0; i < ITEMS_PER_THREAD_CKV; ++i) {
            int d = i * WARP_SIZE + lane_id;
            float val = acc[h][i] * inv_l;
            output[(long)token_idx * NUM_Q_HEADS * HEAD_DIM_CKV + global_head * HEAD_DIM_CKV + d] = __float2bfloat16(val);
        }
    }
}

void dsa_attention(torch::Tensor qn, torch::Tensor qp, torch::Tensor ckv, torch::Tensor kpe, torch::Tensor idx, torch::Tensor out, torch::Tensor lse, double scale) {
    int num_tokens = qn.size(0);
    int topk = idx.size(1);
    int smem_bytes = 150000;

    cudaFuncSetAttribute(dsa_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    dsa_attention_kernel<<<num_tokens, 256, smem_bytes>>>(
        (bfloat16*)qn.data_ptr(), (bfloat16*)qp.data_ptr(),
        (bfloat16*)ckv.data_ptr(), (bfloat16*)kpe.data_ptr(),
        (int32_t*)idx.data_ptr(), (bfloat16*)out.data_ptr(),
        (float*)lse.data_ptr(), (float)scale, topk
    );
}
'''

# ------------------------------------------------------------------------------
# 2. COMPILATION
# ------------------------------------------------------------------------------
def compile_kernel():
    return load_inline(
        name="dsa_attn_benchmark_v8",
        cpp_sources="#include <torch/extension.h>\nvoid dsa_attention(torch::Tensor qn, torch::Tensor qp, torch::Tensor ckv, torch::Tensor kpe, torch::Tensor idx, torch::Tensor out, torch::Tensor lse, double scale);",
        cuda_sources=CUDA_SOURCE,
        functions=["dsa_attention"],
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "--use_fast_math",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_90,code=sm_90",
            "-gencode=arch=compute_90,code=compute_90",
            "-U__CUDA_NO_HALF_OPERATORS__"
        ],
        with_cuda=True,
        verbose=False
    )

# ------------------------------------------------------------------------------
# 3. VERIFIED BENCHMARK RUNNER
# ------------------------------------------------------------------------------
def run_benchmark():
    if not torch.cuda.is_available():
        print("No CUDA.")
        return

    module = compile_kernel()
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs. Initializing Warmup...", flush=True)

    executor = ThreadPoolExecutor(max_workers=num_gpus)

    # --- CRITICAL WARMUP STEP ---
    # Forces all GPUs to initialize context BEFORE benchmarking starts
    print("Warming up all GPUs...", flush=True)
    def warmup_worker(gpu_id):
        with torch.cuda.device(gpu_id):
            # Run a dummy kernel to force context creation
            torch.zeros(1, device=gpu_id)
            torch.cuda.synchronize()

    list(executor.map(warmup_worker, range(num_gpus)))
    print("Warmup Complete.", flush=True)
    # ----------------------------

    configs = [16, 32, 64, 128, 256, 512, 1024]
    MIN_BATCH_FOR_MULTIGPU = 256

    print(f"\n{'BATCH':<10} | {'STRATEGY':<15} | {'LATENCY (ms)':<15} | {'THROUGHPUT':<15}")
    print("-" * 75)

    for b in configs:
        topk = 2048
        q_nope = torch.randn(b, 16, 512, dtype=torch.bfloat16)
        q_pe   = torch.randn(b, 16, 64, dtype=torch.bfloat16)
        indices = torch.randint(0, 1000*64, (b, topk), dtype=torch.int32)

        use_multi_gpu = (b >= MIN_BATCH_FOR_MULTIGPU) and (num_gpus > 1)
        strategy_name = f"Multi-GPU ({num_gpus})" if use_multi_gpu else "Single GPU"

        def _worker(gpu_id, qn_local, qp_local, idx_local):
            with torch.cuda.device(gpu_id):
                if qn_local.numel() == 0: return

                qn_d = qn_local.to(gpu_id, non_blocking=True)
                qp_d = qp_local.to(gpu_id, non_blocking=True)
                idx_d = idx_local.to(gpu_id, non_blocking=True)
                ckv_d = torch.randn(1000, 64, 512, dtype=torch.bfloat16, device=gpu_id)
                kpe_d = torch.randn(1000, 64, 64, dtype=torch.bfloat16, device=gpu_id)
                out_d = torch.empty_like(qn_d)
                lse_d = torch.empty((qn_d.size(0), 16), dtype=torch.float32, device=gpu_id)

                module.dsa_attention(qn_d, qp_d, ckv_d, kpe_d, idx_d, out_d, lse_d, 0.1)

        torch.cuda.synchronize()
        start = time.time()

        iterations = 20
        for _ in range(iterations):
            if not use_multi_gpu:
                _worker(0, q_nope, q_pe, indices)
            else:
                chunk_size = (b + num_gpus - 1) // num_gpus
                qn_splits = list(torch.split(q_nope, chunk_size))
                qp_splits = list(torch.split(q_pe, chunk_size))
                idx_splits = list(torch.split(indices, chunk_size))
                while len(qn_splits) < num_gpus:
                    qn_splits.append(torch.empty(0))
                    qp_splits.append(torch.empty(0))
                    idx_splits.append(torch.empty(0))

                futures = []
                for i in range(num_gpus):
                    futures.append(executor.submit(_worker, i, qn_splits[i], qp_splits[i], idx_splits[i]))
                for f in futures: f.result()

        torch.cuda.synchronize()
        end = time.time()

        avg_latency_ms = ((end - start) / iterations) * 1000
        throughput = b / (avg_latency_ms / 1000)

        print(f"{b:<10} | {strategy_name:<15} | {avg_latency_ms:<15.4f} | {throughput:<15.2f} toks/s")

if __name__ == "__main__":
    run_benchmark()
"""

@app.function(image=image, gpu="B200:5", timeout=1200)
def run_multi_gpu_test():
    Path("run_benchmark.py").write_text(REMOTE_BENCHMARK_SCRIPT)
    import subprocess
    print("Launching ADAPTIVE Attention Benchmark...", flush=True)
    subprocess.run(["python", "run_benchmark.py"], check=True)

@app.local_entrypoint()
def main():
    run_multi_gpu_test.remote()
