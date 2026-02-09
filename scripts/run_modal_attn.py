import sys
import os
import time
import math
import torch
import modal
from pathlib import Path
from torch.utils.cpp_extension import load_inline

app = modal.App("blackwell-attention-benchmark")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

# Image: Standard + Clang for JIT + CUTLASS
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

# ------------------------------------------------------------------------------
# 1. REFERENCE IMPLEMENTATION (Baseline)
# ------------------------------------------------------------------------------
@torch.no_grad()
def ref_attention(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    # Flatten paged KV cache
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)

    output = torch.zeros((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=q_nope.device)
    lse = torch.full((num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=q_nope.device)

    for t in range(num_tokens):
        indices = sparse_indices[t]
        valid_mask = indices != -1
        valid_indices = indices[valid_mask]

        if valid_indices.numel() == 0: continue

        tok_idx = valid_indices.to(torch.long)

        Kc = Kc_all[tok_idx] # Gather
        Kp = Kp_all[tok_idx] # Gather

        qn = q_nope[t].to(torch.float32)
        qp = q_pe[t].to(torch.float32)

        # Attention
        logits = (qn @ Kc.T) + (qp @ Kp.T)
        logits_scaled = logits * sm_scale

        lse[t] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
        attn = torch.softmax(logits_scaled, dim=-1)
        out = attn @ Kc
        output[t] = out.to(torch.bfloat16)

    return output, lse

# ------------------------------------------------------------------------------
# 2. JIT COMPILATION
# ------------------------------------------------------------------------------
def compile_kernel():
    print("Compiling CUDA Kernel...", flush=True)
    with open("kernel.cu", "r") as f:
        cuda_src = f.read()

    # --- STRIP PYBIND11 ---
    # Prevents "multiple definition" errors
    if "PYBIND11_MODULE" in cuda_src:
        cuda_src = cuda_src.split("PYBIND11_MODULE")[0]

    return load_inline(
        name="dsa_attn_jit",
        cpp_sources="#include <torch/extension.h>\nvoid dsa_attention(torch::Tensor qn, torch::Tensor qp, torch::Tensor ckv, torch::Tensor kpe, torch::Tensor idx, torch::Tensor out, torch::Tensor lse, double scale);",
        cuda_sources=cuda_src,
        functions=["dsa_attention"],
        extra_cuda_cflags=[
            "-O3", "-std=c++17",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",

            # --- FAT BINARY FLAGS (The Fix) ---
            "-gencode=arch=compute_80,code=sm_80",       # A100 (Fallbacks)
            "-gencode=arch=compute_90,code=sm_90",       # H100 (Native)
            "-gencode=arch=compute_90,code=compute_90"   # B200 (Forward Compat via PTX)
        ],
        with_cuda=True,
        verbose=False
    )

# ------------------------------------------------------------------------------
# 3. BENCHMARK RUNNER
# ------------------------------------------------------------------------------
def main():
    if not torch.cuda.is_available():
        print("No CUDA device found.")
        return

    try:
        module = compile_kernel()
    except Exception as e:
        print("COMPILATION FAILED:\n", e)
        return

    print("Compilation Success. Starting Benchmark...", flush=True)
    device = torch.device("cuda")

    # Workload Config
    configs = [
        (16, 2048),  # Batch 16, TopK 2048 (Decode)
        (32, 2048),  # Batch 32
        (64, 2048),  # Batch 64
        (1, 2048),   # Batch 1
    ]

    print(f"\n{'BATCH':<10} | {'LATENCY (ms)':<15} | {'BASELINE (ms)':<15} | {'SPEEDUP':<10} | {'STATUS':<10}")
    print("-" * 80)

    for b, topk in configs:
        # Tensors
        q_nope = torch.randn(b, 16, 512, device=device, dtype=torch.bfloat16)
        q_pe   = torch.randn(b, 16, 64, device=device, dtype=torch.bfloat16)

        # 1000 Pages of Cache
        num_pages = 1000
        ckv_cache = torch.randn(num_pages, 64, 512, device=device, dtype=torch.bfloat16)
        kpe_cache = torch.randn(num_pages, 64, 64, device=device, dtype=torch.bfloat16)

        # Random Indices
        total_tokens = num_pages * 64
        indices = torch.randint(0, total_tokens, (b, topk), device=device, dtype=torch.int32)

        # Outputs
        out_kernel = torch.empty_like(q_nope)
        lse_kernel = torch.empty((b, 16), device=device, dtype=torch.float32)
        sm_scale = 0.1

        # Warmup
        module.dsa_attention(q_nope, q_pe, ckv_cache, kpe_cache, indices, out_kernel, lse_kernel, sm_scale)

        # TIMING KERNEL
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
             module.dsa_attention(q_nope, q_pe, ckv_cache, kpe_cache, indices, out_kernel, lse_kernel, sm_scale)
        torch.cuda.synchronize()
        kernel_time = (time.time() - start) / 50 * 1000

        # TIMING BASELINE
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(5): # Fewer runs for slow baseline
            out_ref, _ = ref_attention(q_nope, q_pe, ckv_cache, kpe_cache, indices, sm_scale)
        torch.cuda.synchronize()
        base_time = (time.time() - start) / 5 * 1000

        # Correctness Check
        # Relaxed tolerance for BF16
        is_close = torch.allclose(out_kernel, out_ref, atol=1e-1, rtol=1e-2)
        status = "PASS" if is_close else "FAIL"

        print(f"{b:<10} | {kernel_time:<15.4f} | {base_time:<15.4f} | {base_time/kernel_time:<10.2f} | {status}")

if __name__ == "__main__":
    main()
"""

@app.function(image=image, gpu="B200:1", timeout=1200)
def run_attention_benchmark(kernel_code: str):
    Path("kernel.cu").write_text(kernel_code)
    Path("run_benchmark.py").write_text(REMOTE_BENCHMARK_SCRIPT)

    import subprocess
    print("Launching Attention Benchmark...", flush=True)

    result = subprocess.run(
        ["python", "run_benchmark.py"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print("--- STDERR ---")
        print(result.stderr)

@app.local_entrypoint()
def main():
    # Read the attention kernel (ensure this file exists locally!)
    with open("solution/cuda/attention_kernel.cu", "r") as f:
        kernel_code = f.read()

    run_attention_benchmark.remote(kernel_code)
