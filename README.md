# devin-kernel-optimiser

Parallel Devin agents, running on a Nebius H100 through **Devin Outposts**, race to write a
**Hopper-native W4A16 GEMM** that beats **Marlin**. Marlin is the int4-weight GEMM that sglang
uses on H100 today, but it only uses Ampere-era `mma.sync` (no wgmma, TMA or warp
specialisation). Each agent uses a different kernel DSL, and they coordinate through a shared
message board while a live leaderboard tracks speedup vs Marlin across batch sizes.

```
 Devin cloud (planning)          Nebius H100 VM (/opt/hackathon)
 ┌──────────────┐   outbound   ┌───────────────────────────────────────────────┐
 │ session: scout├────────────►│ worker w1 ── repos/devin-kernel-optimiser     │
 │ session: triton├───────────►│ worker w2 ── repos/...        ┌────────────┐  │
 │ session: cute-dsl├─────────►│ worker w3 ── repos/...  ─────►│ GPU lock   │  │
 │ ...           │             │ ...                            │ (bench.py) │  │
 └──────────────┘              │ board :8765  ◄── posts/results └────────────┘  │
                               └───────────────────────────────────────────────┘
```

## Layout

| Path | What |
|---|---|
| `harness/formats.py` | int4_g128 (GPTQ uint4b8) and mxfp4 definitions, plus the fp32 ground truth |
| `harness/check.py` | correctness: random seeds, odd M values, hidden shapes |
| `harness/bench.py` | timing: shared GPU lock, L2 flush, rotating inputs, re-verification, posts to the board |
| `harness/baselines/` | `sglang_marlin` (the target), `vllm_machete` (Hopper reference), `torch_dequant` |
| `harness/shapes.json` | benchmark shapes. **Placeholders**: regenerate with `harness.gen_shapes` |
| `kernels/_template/` | correct but naive Triton W4A16 kernel to start from |
| `board/` | FastAPI + SQLite message board, leaderboard API, live dashboard |
| `agents/AGENTS.md` | shared brief and rules every agent gets |
| `agents/tasks/*.md` | per-agent briefs: scout, triton, cuda-wgmma, cute-dsl, tilelang, decode |
| `infra/terraform/` | Nebius H100 VM (1 or 8 GPUs), static IP, cloud-init bootstrap |
| `infra/` | VM setup, board and worker launchers, Devin API session launcher |

## Submission interface

`kernels/<agent>/kernel.py` defines `META`, `prepare(qw)` (untimed repack) and
`matmul(x, prepared)` (timed). See `harness/submission.py`.

## Tonight (before the hackathon)

1. Push this repo to GitHub and connect it to Devin (the agents push their `agent/*` branches).
2. **Create the VM with Terraform.** Check your H100 quota first: the docs list `gpu-h100-sxm` in `eu-north1`.
   ```
   cd infra/terraform
   cp terraform.tfvars.example terraform.tfvars    # project_id, subnet_name
   export NEBIUS_IAM_TOKEN=$(nebius iam get-access-token)
   terraform init && terraform apply              # prints public_ip and an ssh command
   ```
   It costs about $4.50/hour. `stopped = true` + `terraform apply` pauses it; `terraform destroy` ends it.
3. In Devin: Settings → Environment → Outposts → create `h100-kernels` and copy the token.
4. Copy `infra/.env` to the VM (`scp`), clone the repo to `/opt/hackathon/main` if cloud-init didn't, then run `bash infra/setup_vm.sh`.
5. Choose the target model and regenerate the shapes:
   `python -m harness.gen_shapes <hf-repo>`. Review them, then commit.
6. Check the baselines on the real GPU. This is the part I couldn't test from a Mac:
   ```
   source /opt/hackathon/venv/bin/activate
   python -m harness.check --baseline sglang_marlin
   python -m harness.check --kernel kernels/_template
   python -m harness.bench --baseline sglang_marlin
   ```
7. `bash infra/start_board.sh`, then tunnel to it from your laptop:
   `ssh -L 8765:127.0.0.1:8765 <vm>` and open http://localhost:8765.
8. Start one worker and launch only the scout (`--agents scout --go`) to check that the Outposts routing works end to end.

## Day of

```
bash infra/start_workers.sh 5
python infra/launch_agents.py --agents scout,triton,cuda-wgmma,cute-dsl,decode            # dry run
python infra/launch_agents.py --agents scout,triton,cuda-wgmma,cute-dsl,decode --max-acu 40 --go
python infra/launch_agents.py --broadcast "Mid-day: focus on M>=64; decode is bandwidth-bound."
```

Watch credits: `--max-acu` caps each session. Five agents with a cap of 40 is the upper bound,
so size it against your $200 once you know the ACU price.

## Unverified (check on the VM)

- `harness/baselines/sglang_marlin.py` was written against the sglang source but has never been run.
- `harness/baselines/vllm_machete.py` uses op signatures from memory.
- Whether several Outpost workers can share one machine (the docs say "N workers serve N concurrent sessions").
- The `platform` field routing v3 API sessions to the outpost, and the base URL `https://api.devin.ai`.
- The Triton template compiles and passes on sm_90 (it was checked only on CPU through its torch-equivalent unpacking).
