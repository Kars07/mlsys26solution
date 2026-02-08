"""
Standalone Verification Script for DSA TopK Indexer.
Compiles and runs the kernel against a pure PyTorch reference.
"""

import sys
import os
import torch
import random
import time
from pathlib import Path
from torch.utils.cpp_extension import load_inline

# --- 1. CONFIGURATION ---
BATCH_SIZE = 2
NUM_HEADS = 64
HEAD_DIM = 128
PAGE_SIZE = 64
MAX_PAGES = 100
NUM_PAGES_TOTAL = 1000
TOPK = 50

def compile_kernel():
    print("Compiling CUDA Kernel...")

    # Read kernel.cu
    with open("solution/cuda/kernel.cu", "r") as f:
        cuda_src = f.read()

    # PyTorch JIT Compile
    module = load_inline(
        name="dsa_indexer_jit",
        cpp_sources="""
        #include <torch/extension.h>
        void dsa_topk_indexer(torch::Tensor q, torch::Tensor k, torch::Tensor w, torch::Tensor sl, torch::Tensor bt, torch::Tensor out);
        """,
        cuda_sources=cuda_src,
        functions=["dsa_topk_indexer"],
        extra_cuda_cflags=[
            "-O3", "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--expt-relaxed-constexpr"
        ],
        with_cuda=True,
        verbose=True
    )
    return module

def ref_implementation(q_fp8, k_cache_fp8, weights, block_table, seq_lens):
    # Reference implementation in Float32
    results = []

    q_f32 = q_fp8.float()
    k_cache_f32 = k_cache_fp8.float()

    for b in range(len(seq_lens)):
        sl = seq_lens[b].item()

        # 1. Gather K for this sequence
        # block_table[b] -> list of page indices
        # We need to stitch them together
        k_seq = []
        for i in range(len(block_table[b])):
            page_id = block_table[b, i].item()
            if page_id < 0: continue # padding
            k_seq.append(k_cache_f32[page_id]) # [64, D]

        if not k_seq:
            results.append(torch.zeros((TOPK,), dtype=torch.int32, device=q_fp8.device))
            continue

        k_concat = torch.cat(k_seq, dim=0) # [Total_Pages * 64, D]
        k_concat = k_concat[:sl] # [Seq_Len, D]

        # 2. Compute Score
        # Q: [H, D]
        # Score = Sum(ReLU(Q @ K.T) * W)
        # Q @ K.T -> [H, Seq]
        scores = torch.matmul(q_f32[b], k_concat.t())
        scores = torch.relu(scores)

        # Apply weights: [H] -> [H, 1] broadcast
        w = weights[b].view(-1, 1)
        scores = scores * w

        # Sum over heads -> [Seq]
        final_scores = scores.sum(dim=0)

        # 3. TopK
        # If sequence is shorter than K, we pad with something small?
        # torch.topk on small tensor returns min(k, size)
        k_val = min(TOPK, sl)
        vals, indices = torch.topk(final_scores, k_val)

        # Pad indices to TOPK if needed
        if k_val < TOPK:
            pad = torch.zeros((TOPK - k_val,), dtype=torch.int32, device=q_fp8.device)
            indices = torch.cat([indices.int(), pad])
        else:
            indices = indices.int()

        results.append(indices)

    return torch.stack(results)

def main():
    if not torch.cuda.is_available():
        print("Skipping: No CUDA device found.")
        return

    # Compile
    try:
        module = compile_kernel()
    except Exception as e:
        print("COMPILATION FAILED")
        print(e)
        return

    print("Compilation Success. Generating Data...")
    device = torch.device("cuda")

    # Generate Inputs
    # Q: [B, H, D] (FP8)
    q = torch.randn(BATCH_SIZE, NUM_HEADS, HEAD_DIM, device=device).to(torch.float8_e4m3fn)

    # K Cache: [NumPages, PageSize, 1, D] - Flatten head dim
    k_cache = torch.randn(NUM_PAGES_TOTAL, PAGE_SIZE, HEAD_DIM, device=device).to(torch.float8_e4m3fn)

    # Weights: [B, H]
    weights = torch.rand(BATCH_SIZE, NUM_HEADS, device=device)

    # Sequence Lengths
    seq_lens = torch.randint(10, MAX_PAGES * PAGE_SIZE, (BATCH_SIZE,), device=device, dtype=torch.int32)

    # Block Table
    block_table = torch.randint(0, NUM_PAGES_TOTAL, (BATCH_SIZE, MAX_PAGES), device=device, dtype=torch.int32)

    # Output
    out_indices = torch.zeros(BATCH_SIZE, TOPK, device=device, dtype=torch.int32)

    # --- RUN KERNEL ---
    print("Running Kernel...")
    torch.cuda.synchronize()
    start = time.time()

    module.dsa_topk_indexer(q, k_cache, weights, seq_lens, block_table, out_indices)

    torch.cuda.synchronize()
    print(f"Kernel finished in {(time.time() - start)*1000:.3f} ms")

    # --- VERIFY ---
    print("Running Reference...")
    ref_indices = ref_implementation(q, k_cache, weights, block_table, seq_lens)

    # Compare
    # Note: TopK indices might differ if scores are identical, but with randn() collisions are rare.
    # We check intersection rate or exact match.

    print("\n--- RESULTS ---")
    for b in range(BATCH_SIZE):
        k_out = out_indices[b].cpu().numpy()
        k_ref = ref_indices[b].cpu().numpy()

        # Sort for easy comparison (set intersection)
        overlap = len(set(k_out) & set(k_ref))
        print(f"Batch {b}: Overlap {overlap}/{TOPK}")

        if overlap == TOPK:
            print("  [PASS]")
        else:
            print(f"  [FAIL] Ref: {k_ref[:5]}... vs Kernel: {k_out[:5]}...")

if __name__ == "__main__":
    main()
