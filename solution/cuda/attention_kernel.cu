#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
constexpr int HEAD_DIM_CKV = 512;
constexpr int HEAD_DIM_KPE = 64;
constexpr int NUM_Q_HEADS = 16;
constexpr int TILE_SIZE = 64;
constexpr int WARP_SIZE = 32;

// Reduced registers to prevent spilling
constexpr int ITEMS_PER_THREAD_CKV = HEAD_DIM_CKV / WARP_SIZE; // 16
constexpr int ITEMS_PER_THREAD_KPE = HEAD_DIM_KPE / WARP_SIZE; // 2

using bfloat16 = __nv_bfloat16;

// -----------------------------------------------------------------------------
// Kernel: Double Buffered Async Gather
// -----------------------------------------------------------------------------
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

    // --- DOUBLE BUFFER SETUP ---
    // We need 2 stages of SMEM
    // Size per stage: 64 indices (256B) + 64*512*2 (64KB) + 64*64*2 (8KB) ~= 73KB
    // Total x2 = ~146KB
    extern __shared__ char smem_buffer[];

    // Pointers for Double Buffering (0 and 1)
    int32_t* smem_indices[2];
    bfloat16* smem_k_nope[2];
    bfloat16* smem_k_pe[2];

    // Stage 0 pointers
    smem_indices[0] = (int32_t*)smem_buffer;
    smem_k_nope[0] = (bfloat16*)(smem_indices[0] + TILE_SIZE);
    smem_k_pe[0] = smem_k_nope[0] + (TILE_SIZE * HEAD_DIM_CKV);

    // Stage 1 pointers
    smem_indices[1] = (int32_t*)(smem_k_pe[0] + (TILE_SIZE * HEAD_DIM_KPE));
    smem_k_nope[1] = (bfloat16*)(smem_indices[1] + TILE_SIZE);
    smem_k_pe[1] = smem_k_nope[1] + (TILE_SIZE * HEAD_DIM_CKV);

    // Compact Registers for Q
    bfloat16 qn_reg[2][ITEMS_PER_THREAD_CKV];
    bfloat16 qp_reg[2][ITEMS_PER_THREAD_KPE];

    // Load Q (Coalesced)
    #pragma unroll
    for(int h=0; h<heads_per_warp; ++h) {
        int global_head = h_start + h;
        for(int i=0; i<ITEMS_PER_THREAD_CKV; ++i) {
            int d = i * WARP_SIZE + lane_id;
            qn_reg[h][i] = q_nope[(long)token_idx * NUM_Q_HEADS * HEAD_DIM_CKV + global_head * HEAD_DIM_CKV + d];
        }
        for(int i=0; i<ITEMS_PER_THREAD_KPE; ++i) {
            int d = i * WARP_SIZE + lane_id;
            qp_reg[h][i] = q_pe[(long)token_idx * NUM_Q_HEADS * HEAD_DIM_KPE + global_head * HEAD_DIM_KPE + d];
        }
    }

    // Accumulators
    float m[2] = {-INFINITY, -INFINITY};
    float l[2] = {0.0f, 0.0f};
    float acc[2][ITEMS_PER_THREAD_CKV];
    #pragma unroll
    for(int h=0; h<2; ++h)
        for(int i=0; i<ITEMS_PER_THREAD_CKV; ++i) acc[h][i] = 0.0f;

    // --- PIPELINE LOOP ---
    int num_tiles = (topk + TILE_SIZE - 1) / TILE_SIZE;
    int cur_stage = 0;
    int next_stage = 1;
    int total_int4_gather = (TILE_SIZE * (HEAD_DIM_CKV + HEAD_DIM_KPE)) * sizeof(bfloat16) / 16;

    // PROLOGUE: Issue Load for Tile 0
    {
        int t = 0;
        int base_k_idx = t * TILE_SIZE;
        // 1. Load Indices
        if (tid < TILE_SIZE) {
            int k_ptr = base_k_idx + tid;
            smem_indices[cur_stage][tid] = (k_ptr < topk) ? indices[token_idx * topk + k_ptr] : -1;
        }
        __syncthreads(); // Wait for indices to be ready for gather logic

        // 2. Async Gather
        for (int i = tid; i < total_int4_gather; i += blockDim.x) {
            int stride = 72; // (512+64)*2/16
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

    // MAIN LOOP
    for (int t = 0; t < num_tiles; ++t) {
        // A. Issue Next Load (t+1)
        if (t + 1 < num_tiles) {
            int base_k_next = (t + 1) * TILE_SIZE;

            // Sync before overwriting next_stage buffer?
            // Yes, but we do it at end of loop.

            // 1. Load Indices for Next
            if (tid < TILE_SIZE) {
                int k_ptr = base_k_next + tid;
                smem_indices[next_stage][tid] = (k_ptr < topk) ? indices[token_idx * topk + k_ptr] : -1;
            }
            __syncthreads(); // Indices must be ready

            // 2. Gather for Next
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

        // B. Wait for Current Load (t)
        // If we issued Next, we have 2 batches in flight. Wait for oldest (1 left).
        // If we didn't issue Next (last tile), we have 1 batch. Wait for it (0 left).
        if (t + 1 < num_tiles) __pipeline_wait_prior(1);
        else __pipeline_wait_prior(0);

        __syncthreads(); // Data is now in SMEM

        // C. Compute Current (t)
        for (int k = 0; k < TILE_SIZE; ++k) {
            int global_id = smem_indices[cur_stage][k];
            if (global_id == -1) continue;

            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                float score = 0.0f;
                // Dot Product
                for (int i = 0; i < ITEMS_PER_THREAD_CKV; ++i)
                    score += __bfloat162float(qn_reg[h][i]) * __bfloat162float(smem_k_nope[cur_stage][k * HEAD_DIM_CKV + i * WARP_SIZE + lane_id]);
                for (int i = 0; i < ITEMS_PER_THREAD_KPE; ++i)
                    score += __bfloat162float(qp_reg[h][i]) * __bfloat162float(smem_k_pe[cur_stage][k * HEAD_DIM_KPE + i * WARP_SIZE + lane_id]);

                // Warp Reduce
                for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
                    score += __shfl_down_sync(0xffffffff, score, offset);
                score = __shfl_sync(0xffffffff, score, 0);

                // Softmax
                score *= sm_scale;
                float m_new = fmaxf(m[h], score);
                float exp_score = expf(score - m_new);
                float correction = expf(m[h] - m_new);
                l[h] = l[h] * correction + exp_score;
                m[h] = m_new;

                // Accumulate
                for (int i = 0; i < ITEMS_PER_THREAD_CKV; ++i)
                    acc[h][i] = acc[h][i] * correction + exp_score * __bfloat162float(smem_k_nope[cur_stage][k * HEAD_DIM_CKV + i * WARP_SIZE + lane_id]);
            }
        }

        // D. Swap Buffers
        __syncthreads(); // Ensure compute is done before next load overwrites
        cur_stage ^= 1;
        next_stage ^= 1;
    }

    // Final Write
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

// -----------------------------------------------------------------------------
// Launcher
// -----------------------------------------------------------------------------
void dsa_attention(
    torch::Tensor q_nope, torch::Tensor q_pe, torch::Tensor ckv, torch::Tensor kpe,
    torch::Tensor idx, torch::Tensor out, torch::Tensor lse, double scale
) {
    int num_tokens = q_nope.size(0);
    int topk = idx.size(1);

    // Double Buffer Size: ~150KB
    int smem_bytes = 150000;

    cudaFuncSetAttribute(dsa_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    dsa_attention_kernel<<<num_tokens, 256, smem_bytes>>>(
        (bfloat16*)q_nope.data_ptr(), (bfloat16*)q_pe.data_ptr(),
        (bfloat16*)ckv.data_ptr(), (bfloat16*)kpe.data_ptr(),
        (int32_t*)idx.data_ptr(), (bfloat16*)out.data_ptr(),
        (float*)lse.data_ptr(), (float)scale, topk
    );
}
