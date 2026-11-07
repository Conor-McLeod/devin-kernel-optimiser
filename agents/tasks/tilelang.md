# Agent: tilelang

**DSL:** TileLang (`import tilelang`). It ships dequantize-GEMM examples (BitBLAS lineage). Start there.

**Target:** the mid and prefill buckets.

Suggested path:
1. Get TileLang's dequant GEMM example running on our format: int4 sym, group 128, bf16 scales.
2. Make sure it lowers to wgmma + TMA on sm_90a, and try its pipelining and warp-specialisation knobs.
3. Tune tile configs per M bucket.

Post how TileLang's generated kernel differs from Marlin structurally. That's useful for everyone.
