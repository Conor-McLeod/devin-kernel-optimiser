# Agent: cuda-wgmma

**DSL:** CUDA C++ with inline PTX, compiled via `torch.utils.cpp_extension.load_inline`
(or `load` with a `.cu` file in your directory). `-arch=sm_90a` is required for wgmma.
CUTLASS/CuTe C++ headers are allowed (`/opt/hackathon/cutlass/include`).

**Target:** the mid and prefill buckets (M ≥ 64).

Suggested path:
1. TMA loads of the packed int4 B tile and the bf16 A tile into shared memory, with a multi-stage mbarrier pipeline.
2. A warp-specialised producer/consumer split: one producer warp issues TMA, and consumer warpgroups
   dequantise int4 → bf16 in registers or shared memory, then issue `wgmma.mma_async` (m64nNk16).
3. Fast dequant: lop3/prmt tricks (see Marlin's `dequant.h`, or the FasterTransformer
   int4→fp16 conversion) and scale application fused with it.
4. Tile and cluster tuning per M bucket. Consider a persistent kernel with a tile scheduler.

You choose the weight layout in `prepare()`, so lay it out for TMA and wgmma rather than copying GPTQ's.
