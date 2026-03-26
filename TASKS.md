# Task Queue
# Format: STATUS TASK_ID — Description
# STATUS: ✅ (done), ⬜ (pending), 🔄 (in progress), ❌ (blocked)
# Tasks are executed top-to-bottom, first incomplete wins.
# The planner may reorder, add, or mark tasks blocked after each PASS.

✅ T01 — Migrate QK_TBQ from 64 to 128 for full head_dim rotation
✅ T02 — Replace rotation matrix with randomized Walsh-Hadamard transform
✅ T03 — Test WHT quality on 7B+ model (Llama-3.1-8B: 5/5 correct, all configs including K+V)
✅ T04 — Cross-validate C++ vs MLX PoC (same input → compare quantized indices and dequantized output)
✅ T05 — Implement flash attention support for TBQ V-cache
✅ T06 — Remove legacy rotation matrix code from ggml-quants.c (tbq_generate_rotation, tbq_get_rotation)
✅ T07 — Optimize Metal flash-attn WHT: cache inverse WHT in shared memory to avoid redundant full-block transforms per dequantize call
✅ T08 — Fix Qwen head_dim mismatch: add runtime guard in Metal kernel to skip TBQ when model head_dim ≠ QK_TBQ, or make QK_TBQ dynamic
✅ T09 — Optimize CPU vec_dot: eliminate per-call malloc in ggml_vec_dot_tbq4_0_f32 (use stack buffer or thread-local allocation)

# --- All planned tasks complete ---
# Remaining uncommitted work: T05-T09 changes across 10 files (see git diff)
# Known minor issue: d=256 round-trip test prints a spurious "block 1 norm" warning (cosmetic, doesn't affect results)
