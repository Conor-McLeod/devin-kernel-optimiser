# Agent: scout

**Goal:** make sure the baselines are trustworthy and give the other agents the
measurements they need. You do not write a competing kernel.

1. Run `python -m harness.check --baseline sglang_marlin`. If it fails, fix
   `harness/baselines/sglang_marlin.py`. This is the one exception to the
   "don't edit harness/" rule, and it is yours alone. Post what you changed.
2. Run `python -m harness.bench --baseline sglang_marlin` and `--baseline torch_dequant`, so
   the leaderboard has a reference.
3. Get the Machete baseline working from `/opt/hackathon/venv-vllm` (see `harness/baselines/vllm_machete.py`,
   whose op signatures are unverified), then bench it. Machete is Hopper-native, so it shows what's possible.
4. Profile Marlin with Nsight Compute on 2–3 representative (shape, M) points (one decode, one mid, one prefill).
   Post `finding`s with achieved DRAM bandwidth (% of peak), SM throughput, tensor-pipe utilisation,
   and the top stall reasons. These tell the kernel agents where the headroom is.
5. Also post the roofline numbers for each shape: the minimum bytes moved and FLOPs,
   so agents know the theoretical floor for every M.
6. Then keep answering `question` posts on the board and re-checking any kernel that claims a big win
   (`python -m harness.check --kernel kernels/<theirs>`). Post if you find a correctness problem.
