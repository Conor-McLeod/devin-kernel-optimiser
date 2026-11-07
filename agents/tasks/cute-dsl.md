# Agent: cute-dsl

**DSL:** CUTLASS CuTe DSL (the Python DSL, `import cutlass.cute as cute`, package `nvidia-cutlass-dsl`).

**Target:** the mid and prefill buckets (M ≥ 64).

Suggested path:
1. Start from the CuTe DSL Hopper dense GEMM example (TMA + wgmma + warp-specialised pipeline).
2. Change the B operand path: TMA-load the packed int4 tile, convert to bf16 (in registers before
   staging to shared memory for wgmma, or using RS-wgmma with A in registers if you swap operands),
   and apply group scales.
3. Swapping operands (computing yᵀ = W·xᵀ) lets the large N dimension fill wgmma's M=64 side
   while the small M of x maps to wgmma's N. That matters a lot for M ≤ 64. Look into it early
   and post what you find.
4. Tune tile shapes and stage counts per M bucket.

Post any CuTe DSL gotchas you hit. The cuda-wgmma agent is doing the same thing in C++,
so trade notes with it.
