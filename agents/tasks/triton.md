# Agent: triton

**DSL:** Triton. Gluon (Triton's lower-level dialect) is allowed if the installed Triton has it.

**Target:** the best overall geomean across all buckets. Triton is the fastest DSL to iterate in.

Suggested path:
1. Start from `kernels/_template/` (correct but naive).
2. Choose a weight layout that makes the B loads wide and coalesced, and unpack with as few ops as possible.
3. Add `triton.autotune` configs per M bucket (BLOCK sizes, num_warps, num_stages), with split-K for small M.
4. Try swapping operands (computing yᵀ) for small M, and persistent / stream-K scheduling for large M.
5. Look at the generated SASS/PTX (`TRITON_CACHE_DIR`) to check that wgmma is actually used at large M.

Post which autotune configs win where. Other agents can use that as a map of the tile-shape space.
