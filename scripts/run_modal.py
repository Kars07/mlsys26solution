import sys
import os
import time
from pathlib import Path
import modal

app = modal.App("flashinfer-bench-verify")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

# Image
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "clang")
    .env({"CACHE_BUST": str(int(time.time()))})
    .run_commands(
        "rm -rf /usr/local/cutlass",
        "git clone https://github.com/NVIDIA/cutlass.git /usr/local/cutlass",
    )
    .pip_install("torch", "numpy", "ninja")
    .env({
        "TORCH_CUDA_ARCH_LIST": "9.0",
        "CUDA_HOME": "/usr/local/cuda"
    })
)

# --- INLINED VERIFICATION SCRIPT ---
VERIFY_SCRIPT_CONTENT = r"""
import torch
import time
from torch.utils.cpp_extension import load_inline

# Configuration
BATCH_SIZE = 2
NUM_HEADS = 64
HEAD_DIM = 128
PAGE_SIZE = 64
MAX_PAGES = 100
NUM_PAGES_TOTAL = 1000
TOPK = 50

def compile_kernel():
    print("Compiling CUDA Kernel...", flush=True)
    with open("kernel.cu", "r") as f:
        cuda_src = f.read()

    if "PYBIND11_MODULE" in cuda_src:
        cuda_src = cuda_src.split("PYBIND11_MODULE")[0]

    return load_inline(
        name="dsa_indexer_jit",
        cpp_sources="#include <torch/extension.h>\nvoid dsa_topk_indexer(torch::Tensor q, torch::Tensor k, torch::Tensor w, torch::Tensor sl, torch::Tensor bt, torch::Tensor out);",
        cuda_sources=cuda_src,
        functions=["dsa_topk_indexer"],
        extra_cuda_cflags=[
            "-O3", "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_90,code=sm_90",
            "-gencode=arch=compute_90,code=compute_90"
        ],
        extra_include_paths=[
            "/usr/local/cutlass/include",
            "/usr/local/cutlass/tools/util/include"
        ],
        with_cuda=True,
        verbose=True
    )

def ref_implementation(q_fp8, k_cache_fp8, weights, block_table, seq_lens):
    results = []
    q_f32 = q_fp8.float()
    k_cache_f32 = k_cache_fp8.float()

    for b in range(len(seq_lens)):
        sl = seq_lens[b].item()
        k_seq = []
        for i in range(len(block_table[b])):
            page_id = block_table[b, i].item()
            if page_id < 0: continue
            k_seq.append(k_cache_f32[page_id])

        if not k_seq:
            results.append(torch.zeros((TOPK,), dtype=torch.int32, device=q_fp8.device))
            continue

        k_concat = torch.cat(k_seq, dim=0)[:sl]
        scores = torch.matmul(q_f32[b], k_concat.t())
        scores = torch.relu(scores)
        scores = scores * weights[b].view(-1, 1)
        final_scores = scores.sum(dim=0)

        k_val = min(TOPK, sl)
        vals, indices = torch.topk(final_scores, k_val)
        if k_val < TOPK:
            indices = torch.cat([indices.int(), torch.zeros((TOPK - k_val,), dtype=torch.int32, device=q_fp8.device)])
        else:
            indices = indices.int()
        results.append(indices)
    return torch.stack(results)

def main():
    if not torch.cuda.is_available():
        print("No CUDA device found.")
        return

    try:
        module = compile_kernel()
    except Exception as e:
        print("COMPILATION FAILED:\n", e)
        return

    print("Compilation Success. Running Test...", flush=True)
    device = torch.device("cuda")

    q = torch.randn(BATCH_SIZE, NUM_HEADS, HEAD_DIM, device=device).to(torch.float8_e4m3fn)
    k_cache = torch.randn(NUM_PAGES_TOTAL, PAGE_SIZE, HEAD_DIM, device=device).to(torch.float8_e4m3fn)
    weights = torch.rand(BATCH_SIZE, NUM_HEADS, device=device)
    seq_lens = torch.randint(10, MAX_PAGES * PAGE_SIZE, (BATCH_SIZE,), device=device, dtype=torch.int32)
    block_table = torch.randint(0, NUM_PAGES_TOTAL, (BATCH_SIZE, MAX_PAGES), device=device, dtype=torch.int32)
    out_indices = torch.zeros(BATCH_SIZE, TOPK, device=device, dtype=torch.int32)

    torch.cuda.synchronize()
    start = time.time()
    module.dsa_topk_indexer(q, k_cache, weights, seq_lens, block_table, out_indices)
    torch.cuda.synchronize()
    print(f"Kernel finished in {(time.time()-start)*1000:.2f} ms")

    ref_indices = ref_implementation(q, k_cache, weights, block_table, seq_lens)

    print("\n--- RESULTS ---")
    passed = True
    for b in range(BATCH_SIZE):
        k_out = set(out_indices[b].cpu().numpy())
        k_ref = set(ref_indices[b].cpu().numpy())
        overlap = len(k_out & k_ref)
        print(f"Batch {b}: Overlap {overlap}/{TOPK}")
        if overlap < TOPK * 0.9:
            passed = False

    if passed:
        print("\n[PASS] Kernel Output Matches Reference")
    else:
        print("\n[FAIL] Significant Mismatch Detected")

if __name__ == "__main__":
    main()
"""

@app.function(image=image, gpu="B200:1", timeout=1200, volumes={"/data": trace_volume})
def run_verification(kernel_code: str):
    # Fix Tuple Access for C++17
    fixed_kernel_code = kernel_code.replace("std::get(1, result)", "std::get<1>(result)")
    fixed_kernel_code = fixed_kernel_code.replace("std::get(1, topk_result)", "std::get<1>(topk_result)")

    Path("kernel.cu").write_text(fixed_kernel_code)
    Path("verify_kernel.py").write_text(VERIFY_SCRIPT_CONTENT)

    import subprocess
    print("Launching Verification Script...", flush=True)

    result = subprocess.run(
        ["python", "verify_kernel.py"],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("\n--- STDERR ---")
        print(result.stderr)

@app.local_entrypoint()
def main():
    # --- HERE IS THE FIXED KERNEL LOGIC TO WRITE TO FILE ---
    # We define it here to ensure it overwrites whatever is local if needed,
    # or you can paste this into solution/cuda/kernel.cu manually.

    kernel_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

using namespace cute;
using T_FP8 = cutlass::float_e4m3_t;

__global__ void dsa_cute_kernel(
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

    int physical_page_id = block_table[b * max_num_pages + page_idx_in_seq];

    // Shared Memory for K Page (64x128 FP8 = 8KB)
    // FIX: Use byte array to avoid constructor issues
    __shared__ uint8_t smem_k_bytes[64 * 128];
    T_FP8* smem_k = reinterpret_cast<T_FP8*>(smem_k_bytes);

    // Global Pointers
    const T_FP8* gmem_k_start = k_cache_ptr + physical_page_id * (64 * 128);
    const T_FP8* gmem_q_start = q_ptr + b * (num_heads * 128);
    const float* w_vec = weights_ptr + b * num_heads;

    // Load K -> SMEM (Vectorized Copy)
    const int4* k_in_int4 = reinterpret_cast<const int4*>(gmem_k_start);
    int4* k_smem_int4 = reinterpret_cast<int4*>(smem_k);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int offset = i * blockDim.x + tid; // Stride 128
        if (offset < (64 * 128 / 16)) {
            k_smem_int4[offset] = k_in_int4[offset];
        }
    }

    __syncthreads();

    // Compute Dot Product
    // Each thread handles ONE token (row) of K
    if (tid < 64) {
        int token_idx = tid;
        float final_score = 0.0f;

        for (int h = 0; h < num_heads; ++h) {
            float dot = 0.0f;
            const T_FP8* q_head_ptr = gmem_q_start + h * 128;
            const T_FP8* k_tok_ptr = smem_k + token_idx * 128;

            const int4* q_int4 = reinterpret_cast<const int4*>(q_head_ptr);
            const int4* k_int4 = reinterpret_cast<const int4*>(k_tok_ptr);

            #pragma unroll
            for (int k_blk = 0; k_blk < (128/16); ++k_blk) {
                int4 q_pack = q_int4[k_blk];
                int4 k_pack = k_int4[k_blk];

                // Reinterpret as bytes
                uint8_t q_bytes[16];
                uint8_t k_bytes[16];
                *(int4*)q_bytes = q_pack;
                *(int4*)k_bytes = k_pack;

                for (int i = 0; i < 16; ++i) {
                    // FIX: Reinterpret bits as FP8, then convert to float
                    T_FP8 q_val; q_val.storage = q_bytes[i];
                    T_FP8 k_val; k_val.storage = k_bytes[i];
                    dot += float(q_val) * float(k_val);
                }
            }

            float val = fmaxf(0.0f, dot);
            final_score += val * w_vec[h];
        }

        int global_token_idx = page_idx_in_seq * page_size + token_idx;
        long out_idx = (long)b * (max_num_pages * page_size) + global_token_idx;
        output_scores[out_idx] = final_score;
    }
}

void dsa_topk_indexer(
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

    dsa_cute_kernel<<<grid, block>>>(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_topk_indexer", &dsa_topk_indexer, "DSA TopK Indexer");
}
"""
    # -----------------------------------------------

    print("Uploading and running verification on B200...")
    run_verification.remote(kernel_source)
