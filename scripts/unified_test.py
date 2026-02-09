import sys
import os
import time
import torch
import modal
from pathlib import Path
from torch.utils.cpp_extension import load_inline
from concurrent.futures import ThreadPoolExecutor

app = modal.App("blackwell-unified-test")

# UPDATE: Added CUTLASS installation to the image
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "clang")
    .pip_install("torch", "numpy", "ninja")
    # Clone CUTLASS for CuTe headers
    .run_commands("git clone https://github.com/NVIDIA/cutlass.git /root/cutlass")
    .env({
        "TORCH_CUDA_ARCH_LIST": "9.0",
        "CUDA_HOME": "/usr/local/cuda"
    })
)

REMOTE_SCRIPT = r"""
import torch
import time
from torch.utils.cpp_extension import load_inline
from concurrent.futures import ThreadPoolExecutor

# ==============================================================================
#  THE COMBINED KERNEL SOURCE
# ==============================================================================
CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

// -----------------------------------------------------------------------------
// [TRACK 1] ATTENTION KERNEL (V6)
// -----------------------------------------------------------------------------
constexpr int ATTN_HEAD_DIM_CKV = 512;
constexpr int ATTN_HEAD_DIM_KPE = 64;
constexpr int ATTN_NUM_Q_HEADS = 16;
constexpr int ATTN_TILE_SIZE = 64;
constexpr int ATTN_WARP_SIZE = 32;
constexpr int ATTN_ITEMS_PER_THREAD_CKV = ATTN_HEAD_DIM_CKV / ATTN_WARP_SIZE;
constexpr int ATTN_ITEMS_PER_THREAD_KPE = ATTN_HEAD_DIM_KPE / ATTN_WARP_SIZE;

using bfloat16 = __nv_bfloat16;

__global__ void __launch_bounds__(256) dsa_attention_kernel_v6(
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
    int warp_id = tid / ATTN_WARP_SIZE;
    int lane_id = tid % ATTN_WARP_SIZE;

    int heads_per_warp = 2;
    int h_start = warp_id * heads_per_warp;

    extern __shared__ char smem_buffer[];

    int32_t* smem_indices[2];
    bfloat16* smem_k_nope[2];
    bfloat16* smem_k_pe[2];

    smem_indices[0] = (int32_t*)smem_buffer;
    smem_k_nope[0] = (bfloat16*)(smem_indices[0] + ATTN_TILE_SIZE);
    smem_k_pe[0] = smem_k_nope[0] + (ATTN_TILE_SIZE * ATTN_HEAD_DIM_CKV);

    smem_indices[1] = (int32_t*)(smem_k_pe[0] + (ATTN_TILE_SIZE * ATTN_HEAD_DIM_KPE));
    smem_k_nope[1] = (bfloat16*)(smem_indices[1] + ATTN_TILE_SIZE);
    smem_k_pe[1] = smem_k_nope[1] + (ATTN_TILE_SIZE * ATTN_HEAD_DIM_CKV);

    bfloat16 qn_reg[2][ATTN_ITEMS_PER_THREAD_CKV];
    bfloat16 qp_reg[2][ATTN_ITEMS_PER_THREAD_KPE];

    #pragma unroll
    for(int h=0; h<heads_per_warp; ++h) {
        int global_head = h_start + h;
        long q_base_ckv = (long)token_idx * ATTN_NUM_Q_HEADS * ATTN_HEAD_DIM_CKV + global_head * ATTN_HEAD_DIM_CKV;
        long q_base_kpe = (long)token_idx * ATTN_NUM_Q_HEADS * ATTN_HEAD_DIM_KPE + global_head * ATTN_HEAD_DIM_KPE;
        for(int i=0; i<ATTN_ITEMS_PER_THREAD_CKV; ++i)
            qn_reg[h][i] = q_nope[q_base_ckv + i * ATTN_WARP_SIZE + lane_id];
        for(int i=0; i<ATTN_ITEMS_PER_THREAD_KPE; ++i)
            qp_reg[h][i] = q_pe[q_base_kpe + i * ATTN_WARP_SIZE + lane_id];
    }

    float m[2] = {-INFINITY, -INFINITY};
    float l[2] = {0.0f, 0.0f};
    float acc[2][ATTN_ITEMS_PER_THREAD_CKV];
    #pragma unroll
    for(int h=0; h<2; ++h)
        for(int i=0; i<ATTN_ITEMS_PER_THREAD_CKV; ++i) acc[h][i] = 0.0f;

    int num_tiles = (topk + ATTN_TILE_SIZE - 1) / ATTN_TILE_SIZE;
    int cur_stage = 0;
    int next_stage = 1;
    int total_int4_gather = (ATTN_TILE_SIZE * (ATTN_HEAD_DIM_CKV + ATTN_HEAD_DIM_KPE)) * sizeof(bfloat16) / 16;

    {
        int t = 0;
        int base_k_idx = t * ATTN_TILE_SIZE;
        if (tid < ATTN_TILE_SIZE / 4) {
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
                    ? (void*)(ckv_cache + (long)global_k_id * ATTN_HEAD_DIM_CKV + offset * 8)
                    : (void*)(kpe_cache + (long)global_k_id * ATTN_HEAD_DIM_KPE + (offset-64) * 8);
                void* dst = (offset < 64)
                    ? (void*)(smem_k_nope[cur_stage] + vec_idx * ATTN_HEAD_DIM_CKV + offset * 8)
                    : (void*)(smem_k_pe[cur_stage] + vec_idx * ATTN_HEAD_DIM_KPE + (offset-64) * 8);
                __pipeline_memcpy_async(dst, src, 16);
            }
        }
        __pipeline_commit();
    }

    for (int t = 0; t < num_tiles; ++t) {
        if (t + 1 < num_tiles) {
            int base_k_next = (t + 1) * ATTN_TILE_SIZE;
            if (tid < ATTN_TILE_SIZE / 4) {
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
                        ? (void*)(ckv_cache + (long)global_k_id * ATTN_HEAD_DIM_CKV + offset * 8)
                        : (void*)(kpe_cache + (long)global_k_id * ATTN_HEAD_DIM_KPE + (offset-64) * 8);
                    void* dst = (offset < 64)
                        ? (void*)(smem_k_nope[next_stage] + vec_idx * ATTN_HEAD_DIM_CKV + offset * 8)
                        : (void*)(smem_k_pe[next_stage] + vec_idx * ATTN_HEAD_DIM_KPE + (offset-64) * 8);
                    __pipeline_memcpy_async(dst, src, 16);
                }
            }
            __pipeline_commit();
        }

        if (t + 1 < num_tiles) __pipeline_wait_prior(1);
        else __pipeline_wait_prior(0);

        __syncthreads();

        for (int k = 0; k < ATTN_TILE_SIZE; ++k) {
            int global_id = smem_indices[cur_stage][k];
            if (global_id == -1) continue;

            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                float score = 0.0f;
                for (int i = 0; i < ATTN_ITEMS_PER_THREAD_CKV; ++i)
                    score += __bfloat162float(qn_reg[h][i]) * __bfloat162float(smem_k_nope[cur_stage][k * ATTN_HEAD_DIM_CKV + i * ATTN_WARP_SIZE + lane_id]);
                for (int i = 0; i < ATTN_ITEMS_PER_THREAD_KPE; ++i)
                    score += __bfloat162float(qp_reg[h][i]) * __bfloat162float(smem_k_pe[cur_stage][k * ATTN_HEAD_DIM_KPE + i * ATTN_WARP_SIZE + lane_id]);

                for (int offset = ATTN_WARP_SIZE/2; offset > 0; offset /= 2)
                    score += __shfl_down_sync(0xffffffff, score, offset);
                score = __shfl_sync(0xffffffff, score, 0);

                score *= sm_scale;
                float m_new = fmaxf(m[h], score);
                float exp_score = expf(score - m_new);
                float correction = expf(m[h] - m_new);
                l[h] = l[h] * correction + exp_score;
                m[h] = m_new;

                for (int i = 0; i < ATTN_ITEMS_PER_THREAD_CKV; ++i)
                    acc[h][i] = acc[h][i] * correction + exp_score * __bfloat162float(smem_k_nope[cur_stage][k * ATTN_HEAD_DIM_CKV + i * ATTN_WARP_SIZE + lane_id]);
            }
        }

        __syncthreads();
        cur_stage ^= 1;
        next_stage ^= 1;
    }

    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        int global_head = h_start + h;
        if (lane_id == 0) lse_out[token_idx * ATTN_NUM_Q_HEADS + global_head] = (m[h] + logf(l[h])) / logf(2.0f);

        float inv_l = 1.0f / l[h];
        for (int i = 0; i < ATTN_ITEMS_PER_THREAD_CKV; ++i) {
            int d = i * ATTN_WARP_SIZE + lane_id;
            float val = acc[h][i] * inv_l;
            output[(long)token_idx * ATTN_NUM_Q_HEADS * ATTN_HEAD_DIM_CKV + global_head * ATTN_HEAD_DIM_CKV + d] = __float2bfloat16(val);
        }
    }
}

// -----------------------------------------------------------------------------
// [TRACK 2] INDEXER KERNEL (CuTe FP8)
// -----------------------------------------------------------------------------
using namespace cute;
using T_FP8 = cutlass::float_e4m3_t;

__global__ void __launch_bounds__(128) dsa_cute_indexer_kernel(
    const T_FP8* __restrict__ q_ptr,
    const T_FP8* __restrict__ k_cache_ptr,
    const float* __restrict__ weights_ptr,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    float* __restrict__ output_scores,
    int num_heads,
    int head_dim,
    int page_size,
    int max_num_pages
) {
    int page_idx_in_seq = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;

    if (page_idx_in_seq * page_size >= seq_lens[b]) return;

    extern __shared__ uint8_t smem_k_bytes[];

    int physical_page_id = block_table[b * max_num_pages + page_idx_in_seq];
    const T_FP8* gmem_k_start = k_cache_ptr + physical_page_id * (64 * 128);
    const int4* k_in_int4 = reinterpret_cast<const int4*>(gmem_k_start);
    int4* k_smem_int4 = reinterpret_cast<int4*>(smem_k_bytes);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int offset = i * blockDim.x + tid;
        if (offset < (64 * 128 / 16)) {
            k_smem_int4[offset] = k_in_int4[offset];
        }
    }

    __syncthreads();

    int token_idx = tid % 64;
    int head_group = tid / 64;
    int heads_per_group = num_heads / 2;
    int start_head = head_group * heads_per_group;
    int end_head = start_head + heads_per_group;

    T_FP8* smem_k = reinterpret_cast<T_FP8*>(smem_k_bytes);
    const T_FP8* k_tok_ptr = smem_k + token_idx * 128;
    const T_FP8* gmem_q_batch = q_ptr + b * (num_heads * 128);
    const float* w_vec = weights_ptr + b * num_heads;

    float partial_score = 0.0f;

    #pragma unroll 4
    for (int h = start_head; h < end_head; ++h) {
        float dot = 0.0f;
        const T_FP8* q_head_ptr = gmem_q_batch + h * 128;

        const int4* q_int4 = reinterpret_cast<const int4*>(q_head_ptr);
        const int4* k_int4 = reinterpret_cast<const int4*>(k_tok_ptr);

        #pragma unroll
        for (int k_blk = 0; k_blk < 8; ++k_blk) {
            int4 q_pack = q_int4[k_blk];
            int4 k_pack = k_int4[k_blk];

            uint8_t q_bytes[16];
            uint8_t k_bytes[16];
            *(int4*)q_bytes = q_pack;
            *(int4*)k_bytes = k_pack;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                T_FP8 q_val; q_val.storage = q_bytes[i];
                T_FP8 k_val; k_val.storage = k_bytes[i];
                dot += float(q_val) * float(k_val);
            }
        }
        float val = fmaxf(0.0f, dot);
        partial_score += val * w_vec[h];
    }

    float other_half_score = __shfl_down_sync(0xFFFFFFFF, partial_score, 64);

    if (tid < 64) {
        float final_score = partial_score + other_half_score;
        int global_token_idx = page_idx_in_seq * page_size + token_idx;
        long out_idx = (long)b * (max_num_pages * page_size) + global_token_idx;
        output_scores[out_idx] = final_score;
    }
}

// =============================================================================
// LAUNCHERS
// =============================================================================

void run_attention(
    torch::Tensor q_nope, torch::Tensor q_pe, torch::Tensor ckv, torch::Tensor kpe,
    torch::Tensor idx, torch::Tensor out, torch::Tensor lse, double scale
) {
    int num_tokens = q_nope.size(0);
    int topk = idx.size(1);
    int smem_bytes = 150000;
    cudaFuncSetAttribute(dsa_attention_kernel_v6, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    dsa_attention_kernel_v6<<<num_tokens, 256, smem_bytes>>>(
        (bfloat16*)q_nope.data_ptr(), (bfloat16*)q_pe.data_ptr(),
        (bfloat16*)ckv.data_ptr(), (bfloat16*)kpe.data_ptr(),
        (int32_t*)idx.data_ptr(), (bfloat16*)out.data_ptr(),
        (float*)lse.data_ptr(), (float)scale, topk
    );
}

void run_indexer(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table,
    torch::Tensor topk_indices
) {
    int b = q.size(0);
    int num_heads = q.size(1);
    int head_dim = q.size(2);
    int page_size = k.size(1);
    int max_num_pages = block_table.size(1);
    int topk = topk_indices.size(1);

    int max_tokens = max_num_pages * page_size;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    torch::Tensor all_scores = torch::full({b, max_tokens}, -1e9, options);

    dim3 grid(max_num_pages, b);
    dim3 block(128);
    int smem_size = 8192;

    dsa_cute_indexer_kernel<<<grid, block, smem_size>>>(
        (const T_FP8*)q.data_ptr(),
        (const T_FP8*)k.data_ptr(),
        weights.data_ptr<float>(),
        block_table.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(),
        all_scores.data_ptr<float>(),
        num_heads,
        head_dim,
        page_size,
        max_num_pages
    );

    auto result = torch::topk(all_scores, topk, 1, true, true);
    topk_indices.copy_(std::get<1>(result).to(torch::kInt32));
}
'''

def compile_unified():
    return load_inline(
        name="blackwell_unified_test_lib",
        cpp_sources=r'''
#include <torch/extension.h>
void run_attention(torch::Tensor qn, torch::Tensor qp, torch::Tensor ckv, torch::Tensor kpe, torch::Tensor idx, torch::Tensor out, torch::Tensor lse, double scale);
void run_indexer(torch::Tensor q, torch::Tensor k, torch::Tensor w, torch::Tensor sl, torch::Tensor bt, torch::Tensor out);
        ''',
        cuda_sources=CUDA_SOURCE,
        functions=["run_attention", "run_indexer"],
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "--use_fast_math",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_90,code=sm_90",
            "-gencode=arch=compute_90,code=compute_90",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-I/root/cutlass/include",
            "-I/root/cutlass/tools/util/include"
        ],
        with_cuda=True,
        verbose=False
    )

def get_optimal_gpu_count(batch_size, max_gpus):
    if batch_size < 256: return 1
    elif batch_size < 384: return min(2, max_gpus)
    elif batch_size < 512: return min(3, max_gpus)
    elif batch_size < 768: return min(4, max_gpus)
    else: return max_gpus

def test_unified():
    if not torch.cuda.is_available(): return
    print("Compiling Unified Kernel...", flush=True)
    module = compile_unified()
    print("Compilation Success.", flush=True)

    num_total_gpus = torch.cuda.device_count()
    executor = ThreadPoolExecutor(max_workers=num_total_gpus)

    # Pre-alloc for attention
    gpu_buffers = {}
    print("Warming up Attention...", flush=True)
    def setup_gpu(gpu_id):
        with torch.cuda.device(gpu_id):
            ckv = torch.randn(1000, 64, 512, dtype=torch.bfloat16, device=gpu_id)
            kpe = torch.randn(1000, 64, 64, dtype=torch.bfloat16, device=gpu_id)
            return (ckv, kpe)
    results = list(executor.map(setup_gpu, range(num_total_gpus)))
    for i, bufs in enumerate(results): gpu_buffers[i] = bufs

    # --- TEST 1: ADAPTIVE ATTENTION ---
    print("\n[Track 1] Testing Adaptive Attention...", flush=True)
    configs = [128, 512, 1024]
    for b in configs:
        n_gpus = get_optimal_gpu_count(b, num_total_gpus)
        q_nope = torch.randn(b, 16, 512, dtype=torch.bfloat16, device="cuda:0")
        q_pe   = torch.randn(b, 16, 64, dtype=torch.bfloat16, device="cuda:0")
        indices = torch.randint(0, 1000*64, (b, 2048), dtype=torch.int32, device="cuda:0")

        # Helper for dispatch (Simplified for test)
        def _worker(gpu_id, qn_i, qp_i, idx_i):
            with torch.cuda.device(gpu_id):
                if qn_i.numel() == 0: return
                qn_d = qn_i.to(gpu_id, non_blocking=True)
                qp_d = qp_i.to(gpu_id, non_blocking=True)
                idx_d = idx_i.to(gpu_id, non_blocking=True)
                ckv_d, kpe_d = gpu_buffers[gpu_id]
                out_d = torch.empty_like(qn_d)
                lse_d = torch.empty((qn_d.size(0), 16), dtype=torch.float32, device=gpu_id)
                module.run_attention(qn_d, qp_d, ckv_d, kpe_d, idx_d, out_d, lse_d, 0.1)

        # Split logic
        chunk = (b + n_gpus - 1) // n_gpus
        qn_s = list(torch.split(q_nope, chunk))
        qp_s = list(torch.split(q_pe, chunk))
        idx_s = list(torch.split(indices, chunk))
        while len(qn_s) < n_gpus: qn_s.append(torch.empty(0)); qp_s.append(torch.empty(0)); idx_s.append(torch.empty(0))

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            if n_gpus == 1: # Direct
                with torch.cuda.device(0):
                    ckv0, kpe0 = gpu_buffers[0]
                    out0 = torch.empty_like(q_nope); lse0 = torch.empty((b,16), dtype=torch.float32, device="cuda:0")
                    module.run_attention(q_nope, q_pe, ckv0, kpe0, indices, out0, lse0, 0.1)
            else: # Multi
                futures = []
                for i in range(n_gpus):
                    futures.append(executor.submit(_worker, i, qn_s[i], qp_s[i], idx_s[i]))
                for f in futures: f.result()
        torch.cuda.synchronize()
        end = time.time()
        print(f"Batch {b} ({n_gpus} GPUs): {((end-start)/10)*1000:.2f} ms", flush=True)

    # --- TEST 2: INDEXER (CuTe) ---
    print("\n[Track 2] Testing Indexer...", flush=True)
    with torch.cuda.device(0):
        B = 128
        H = 64
        D = 128
        # FIX: Generate FP32 then Cast to FP8
        q = torch.randn(B, H, D, device="cuda:0", dtype=torch.float32).to(torch.float8_e4m3fn)
        k = torch.randn(100, 64, D, device="cuda:0", dtype=torch.float32).to(torch.float8_e4m3fn)
        w = torch.randn(B, H, dtype=torch.float32, device="cuda:0")
        bt = torch.randint(0, 100, (B, 32), dtype=torch.int32, device="cuda:0")
        sl = torch.full((B,), 32*64, dtype=torch.int32, device="cuda:0")
        out = torch.empty((B, 2048), dtype=torch.int32, device="cuda:0")

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            module.run_indexer(q, k, w, sl, bt, out)
        torch.cuda.synchronize()
        end = time.time()
        print(f"Indexer (Batch 128) Latency: {((end-start)/10)*1000:.2f} ms", flush=True)

    print("\nUnified Verification Complete.", flush=True)

if __name__ == "__main__":
    test_unified()
"""

@app.function(image=image, gpu="B200:5", timeout=1200)
def run_test():
    Path("run_test.py").write_text(REMOTE_SCRIPT)
    import subprocess
    print("Launching Unified Test...", flush=True)
    subprocess.run(["python", "run_test.py"], check=True)

@app.local_entrypoint()
def main():
    run_test.remote()
