#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

using namespace cute;
using T_FP8 = cutlass::float_e4m3_t;

// -----------------------------------------------------------------------------
// Kernel: Split-Head Parallelism
// -----------------------------------------------------------------------------
__global__ void __launch_bounds__(128) dsa_cute_kernel(
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

    // 2. Load K Page to Shared Memory
    extern __shared__ uint8_t smem_k_bytes[];

    int physical_page_id = block_table[b * max_num_pages + page_idx_in_seq];
    const T_FP8* gmem_k_start = k_cache_ptr + physical_page_id * (64 * 128);
    const int4* k_in_int4 = reinterpret_cast<const int4*>(gmem_k_start);
    int4* k_smem_int4 = reinterpret_cast<int4*>(smem_k_bytes);

    // All 128 threads load data (Vectorized)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int offset = i * blockDim.x + tid;
        if (offset < (64 * 128 / 16)) {
            k_smem_int4[offset] = k_in_int4[offset];
        }
    }

    __syncthreads();

    // 3. Compute Dot Product (Split-Head Optimization)
    // tid 0-63   -> Heads 0-31
    // tid 64-127 -> Heads 32-63

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

        // Vectorized Math (16 elems at a time)
        #pragma unroll
        for (int k_blk = 0; k_blk < 8; ++k_blk) {
            int4 q_pack = q_int4[k_blk];
            int4 k_pack = k_int4[k_blk];

            // Reinterpret bits as bytes
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

        // Fused Activation
        float val = fmaxf(0.0f, dot);
        partial_score += val * w_vec[h];
    }

    // 4. Parallel Reduction (Warp Shuffle - Faster than SMEM)
    // Reduce partial scores from the two thread groups
    float other_half_score = __shfl_down_sync(0xFFFFFFFF, partial_score, 64);

    if (tid < 64) {
        float final_score = partial_score + other_half_score;
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
    int smem_size = 8192;

    dsa_cute_kernel<<<grid, block, smem_size>>>(
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
