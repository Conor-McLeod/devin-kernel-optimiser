# Agent: decode

**DSL:** your choice (CUDA C++ is recommended for control).

**Target:** the decode bucket (M = 1..16), where Marlin is strongest because the problem is limited by
memory bandwidth. Beating it here means getting closer to H100 DRAM peak (~3.35 TB/s) than Marlin does.

Ideas:
- The ceiling is bytes moved: about N*K/2 + scales. Compute the % of peak bandwidth you reach per shape and post it.
- Split-K / stream-K to fill all 132 SMs when N is small (e.g. the expert shapes).
- Wide vectorised loads (128-bit), or TMA bulk copies, of a weight layout made for streaming.
- mma.sync with the activation broadcast, or even CUDA-core FMAs at M=1. Measure, don't assume.
- Use L2 persistence or prefetch hints sparingly. Remember that the bench flushes L2 before every call.

The scout agent will post Marlin's achieved bandwidth at decode. Your goal is to beat that number.
