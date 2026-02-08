import ctypes
from tvm.ffi import register_func
import torch


@register_func("flashinfer.dsa_topk_indexer")
def dsa_topk_indexer(q, k_cache, weights, block_table, seq_lens, topk=2048):
    """
    Python binding for DSA Indexer.

    Args:
        q: [Batch, Heads, Dim] (FP8)
        k_cache: [NumPages, 64, 1, 132] (INT8 raw bytes)
        weights: [Batch, Heads] (FP32)
        block_table: [Batch, MaxPages] (INT32)
        seq_lens: [Batch] (INT32)
    """
    batch_size = q.shape[0]
    num_pages = k_cache.shape[0]
    max_pages = block_table.shape[1]

    # Allocate Output
    out_indices = torch.empty((batch_size, topk), dtype=torch.int32, device=q.device)

    # Get raw pointers
    q_ptr = q.data_ptr()
    k_ptr = k_cache.data_ptr()
    w_ptr = weights.data_ptr()
    table_ptr = block_table.data_ptr()
    lens_ptr = seq_lens.data_ptr()
    out_ptr = out_indices.data_ptr()

    # We need to find the C++ symbol.
    # In the contest environment, we often use a helper or load the specific .so
    # For now, we assume a JIT or pre-loaded symbol via TVM.

    # NOTE: The actual C++ launch logic typically needs to be exposed via TVM PackedFunc
    # or a ctypes CDLL load.
    # Since the starter kit creates a 'solution.json', the runner likely loads the source
    # and compiles it.

    # Placeholder for the actual ctypes call once the lib is built:
    # lib.run_dsa_indexer(
    #    ctypes.c_void_p(q_ptr),
    #    ctypes.c_void_p(k_ptr),
    #    ...
    # )

    return out_indices
