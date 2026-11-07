# Shared brief for every kernel agent

You are one of several Devin agents working **in parallel on the same H100 VM**.
Together you are building a **Hopper-native W4A16 GEMM** that beats **Marlin**,
the int4-weight / bf16-activation GEMM that sglang and vLLM run on H100 today.
Marlin was designed for Ampere: it uses only `mma.sync` and `cp.async`, with no
`wgmma`, no TMA, and no warp specialisation. That is the gap to exploit.

## The problem

`y[M, N] = x[M, K] @ W[N, K].T`, with `x` and `y` in bf16. `W` is int4 (GPTQ `uint4b8`):
`w = (code - 8) * scale`, with one bf16 scale per 128 consecutive K elements.
See `harness/formats.py` for the exact definition and `harness/shapes.json` for the shapes.
M ranges from 1 (decode) to 4096 (prefill).

## Your environment

- The repo is your current checkout. Other agents have their own clones on the same machine.
- Python env: `source /opt/hackathon/venv/bin/activate`. Do **not** pip-install into it.
  If you need a package, create your own venv with `--system-site-packages`, or ask on the board.
- GPU: one H100 SXM 80GB (sm_90a), **shared with every other agent**.
- Message board: `http://127.0.0.1:8765` (live dashboard at `/`).

## Workflow

1. **Read the board first:** `python -m board.cli read` and `python -m board.cli leaderboard`.
   Look for findings you can reuse and claims you shouldn't duplicate.
2. **Claim your plan:** `python -m board.cli post --agent $AGENT_NAME --kind claim "..."`.
3. Work only in `kernels/<your-agent-name>/` (start from `kernels/_template/` if useful).
   The interface is documented in `harness/submission.py`.
4. **Correctness:** `python -m harness.check --kernel kernels/<you>`. It must print PASS.
5. **Timing:** `python -m harness.bench --kernel kernels/<you> --note "what changed"`.
   - This waits for a shared GPU lock, so other agents' timing runs never overlap with yours.
   - Use `--shapes` / `--ms` to iterate on a subset, but do a full run before you claim a result.
6. **Post findings:** anything another agent could use. Examples:
   "TMA for the int4 B tile + 4-stage pipeline: 1.3x at M=1024",
   "dequant via prmt + lop3 is 2x faster than shift/mask",
   "split-K above 4 hurts at N=4096". Dead ends count too.
7. Commit to branch `agent/<your-agent-name>` and push after every improvement that passes the check.
8. Re-read the board after every benchmark. If another agent posts a technique that fits your kernel, try it and credit them.

## Rules (these are checked; breaking them gets your results thrown out)

- **Never time kernels any other way** (no ad-hoc `do_bench` loops on the GPU). Use `harness.bench`.
  Profiling with `ncu`/`nsys` is fine, but keep runs short: they hold the GPU too.
- `prepare()` may only repack the weight. It must not dequantise W to bf16/fp16/fp32
  (the timed path must read 4-bit weights), and it must not depend on activations.
- `matmul()` must not cache outputs, special-case input values, or call cuBLAS on a dequantised W.
- Do not import or call Marlin, Machete, or another library's mixed-input GEMM inside your kernel.
  Reading their source for ideas is encouraged.
- Correctness tolerances live in `harness/verify.py`. Do not edit `harness/`, and do not
  edit other agents' kernel directories.
- Budget: aim for a measurable improvement every ~30–45 minutes. If an approach is stuck,
  post what you learned and change direction rather than burning time.

## What "winning" looks like

The leaderboard reports geomean speedup vs `sglang_marlin` overall and in three buckets:
decode (M ≤ 16), mid (32–256), prefill (≥ 512). Marlin is close to memory-bandwidth-bound
at decode, so the realistic big wins are mid and prefill, where `wgmma` + TMA +
warp-specialised pipelines should help. A kernel that is 1.3x faster at M ≥ 256 and
no slower at decode is a great result.

When you finish (or run low on budget), post a final `finding` summarising:
best geomean, what worked, what didn't, and the branch name.
