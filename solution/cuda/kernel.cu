#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

using namespace cute;
using T_FP8 = cutlass::float_e4m3_t;

// -----------------------------------------------------------------------------
// Kernel: Vectorized & Shared Memory Optimized
// -----------------------------------------------------------------------------
__global__ void dsa_cute_kernel(
    const T_FP8* __restrict__ q_ptr,           // [B, H, D]
    const T_FP8* __restrict__ k_cache_ptr,     // [TotalPages, PageSize, D]
    const float* __restrict__ weights_ptr,     // [B, H]
    const int32_t* __restrict__ block_table,   // [B, MaxPages]
    const int32_t* __restrict__ seq_lens,      // [B]
    float* __restrict__ output_scores,         // [B, MaxPages * PageSize]
    int num_heads,
    int head_dim,
    int page_size,
    int max_num_pages
) {
    int page_idx_in_seq = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;

    // 1. Bounds Check
    if (page_idx_in_seq * page_size >= seq_lens[b]) return;

    int physical_page_id = block_table[b * max_num_pages + page_idx_in_seq];

    // 2. Shared Memory for K Page
    // We treat this as a byte array to avoid constructor overhead issues
    // Size: 64 tokens * 128 dim = 8192 bytes
    __shared__ uint8_t smem_k_bytes[64 * 128];

    // 3. Global Pointers
    const T_FP8* gmem_k_start = k_cache_ptr + physical_page_id * (64 * 128);
    const T_FP8* gmem_q_start = q_ptr + b * (num_heads * 128);
    const float* w_vec = weights_ptr + b * num_heads;

    // 4. Load K -> SMEM (Vectorized 128-bit Copy)
    // Each thread loads int4 (16 bytes) at a time.
    // 128 threads * 16 bytes = 2048 bytes per step.
    // 8192 bytes / 2048 = 4 steps.
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

    // 5. Compute Dot Product
    // Strategy: Each thread handles ONE token row of K (0..63)
    // Threads 64..127 are idle in this specific setup (could be used for double buffering later)
    if (tid < 64) {
        int token_idx = tid;
        float final_score = 0.0f;

        // Pointers for this token in SMEM
        T_FP8* smem_k = reinterpret_cast<T_FP8*>(smem_k_bytes);
        const T_FP8* k_tok_ptr = smem_k + token_idx * 128;

        for (int h = 0; h < num_heads; ++h) {
            float dot = 0.0f;
            const T_FP8* q_head_ptr = gmem_q_start + h * 128;

            // Vectorized Compute (16 elems at a time)
            const int4* q_int4 = reinterpret_cast<const int4*>(q_head_ptr);
            const int4* k_int4 = reinterpret_cast<const int4*>(k_tok_ptr);

            #pragma unroll
            for (int k_blk = 0; k_blk < (128/16); ++k_blk) {
                int4 q_pack = q_int4[k_blk];
                int4 k_pack = k_int4[k_blk];

                // Reinterpret as bytes to extract FP8 values
                uint8_t q_bytes[16];
                uint8_t k_bytes[16];
                *(int4*)q_bytes = q_pack;
                *(int4*)k_bytes = k_pack;

                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    // Convert FP8 (e4m3) bits -> float
                    T_FP8 q_val; q_val.storage = q_bytes[i];
                    T_FP8 k_val; k_val.storage = k_bytes[i];
                    dot += float(q_val) * float(k_val);
                }
            }

            // Fused Activation + Weighted Sum
            float val = fmaxf(0.0f, dot); // ReLU
            final_score += val * w_vec[h];
        }

        // 6. Write Result to Global Memory
        int global_token_idx = page_idx_in_seq * page_size + token_idx;
        long out_idx = (long)b * (max_num_pages * page_size) + global_token_idx;
        output_scores[out_idx] = final_score;
    }
}

// -----------------------------------------------------------------------------
// Launcher
// -----------------------------------------------------------------------------
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

    // Temp buffer for scores (B, MaxTokens)
    int max_tokens = max_num_pages * page_size;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    torch::Tensor all_scores = torch::full({b, max_tokens}, -1e9, options);

    // Launch Kernel
    // Grid covers all pages. Block size 128 fits standard GPU scheduling well.
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

    // Perform TopK Selection using PyTorch (Highly optimized)
    auto result = torch::topk(all_scores, topk, 1, true, true);

    // Extract indices (element 1 of tuple)
    topk_indices.copy_(std::get<1>(result).to(torch::kInt32));
}

// PyBind Definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_topk_indexer", &dsa_topk_indexer, "DSA TopK Indexer");
}
