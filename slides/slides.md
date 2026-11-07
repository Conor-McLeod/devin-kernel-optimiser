---
theme: default
title: Devin Kernel Swarm
info: Parallel Devin agents writing Hopper-native W4A16 GEMMs to beat Marlin
class: text-center
transition: slide-left
mdc: true
colorSchema: dark
---

# Devin Kernel Swarm

Five Devin agents, one H100, one shared message board,
racing to beat the int4 GEMM that production LLM serving runs today

<div class="pt-10 opacity-70 text-sm">
Devin Outposts · Nebius H100 · Triton · CUDA/PTX · CuTe DSL
</div>

<!--
One-liner: we pointed a team of Devin agents at a real production kernel and let them collaborate on beating it.
-->

---
layout: two-cols
layoutClass: gap-10
---

# The target: Marlin

The **W4A16 GEMM** behind GPTQ/AWQ int4 models, plus NVFP4/MXFP4 MoE on pre-Blackwell GPUs, in **sglang and vLLM**

`y = x · Wᵀ`
- `x`: bf16 activations, M = 1 → 4096 tokens
- `W`: 4-bit weights, one bf16 scale per 128 elements

On **every H100** serving those models, it runs an **Ampere-era** design:

- `mma.sync` + `cp.async` only
- **no** `wgmma`, **no** TMA, **no** warp specialisation

::right::

<div class="mt-16">

### Measured on our H100

| | Marlin | H100 peak |
|---|---|---|
| Prefill (large M) | ~335 TFLOP/s | 989 TFLOP/s |
| Decode (M ≤ 16) | ~1.0 TB/s | 3.35 TB/s |

<div class="mt-6 text-sm opacity-80">
Scout's Nsight profile: 1 CTA/SM, 255 registers ⇒ <b>12.5% occupancy</b>; tensor pipe at 15% during decode
</div>

</div>

<!--
The point: a kernel that is everywhere in production, but leaves most of the Hopper hardware unused. Real headroom, real impact.
-->

---

# The setup

```mermaid {scale: 0.75}
flowchart LR
  A["<b>Devin cloud</b><br/>5 sessions: scout · triton<br/>cuda-wgmma · cute-dsl · decode<br/><i>planning + inference</i>"] <-- "outbound only" --> WK
  subgraph VM["Nebius H100 VM: Devin Outpost (execution)"]
    WK["5 workers<br/>clone + branch per agent"] --> L["harness<br/>check · bench · GPU lock"] --> G[("H100 80GB")]
    L --> B["message board<br/>+ live leaderboard"]
    WK <--> B
  end
```

- **OpenTofu** creates the VM (`gpu-h100-sxm`, CUDA 13 image)
- **Outposts:** Devin plans in its cloud, and every command runs on *our* GPU box
- One `launch_agents.py` call starts all five sessions with per-agent briefs and ACU caps

---

# Five agents, one per DSL

| Agent | Language | Brief |
|---|---|---|
| **scout** | n/a | Verify the baselines, profile Marlin with Nsight, post roofline floors |
| **cuda-wgmma** | CUDA C++ / inline PTX | TMA + `wgmma` + warp specialisation, mid/prefill M |
| **cute-dsl** | CUTLASS CuTe DSL (Python) | The same idea in NVIDIA's new Python DSL |
| **triton** | Triton | Best overall geomean, autotuned per batch size |
| **decode** | CUDA C++ | M ≤ 16, where everything is memory-bandwidth-bound |

<div class="mt-8">

Every agent gets the same contract:
`prepare(W)` repacks the weights untimed → `matmul(x, prepared)` is timed

</div>

---

# Honest benchmarking

Agents optimise whatever they're measured on, so the measurement has to be hard to fool.

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

### `harness.check`
- fp32 ground truth for every format
- new random seed every run
- odd M (3, 17, 777) and **hidden random shapes**
- off-by-one kernels fail (tested)

</div>
<div>

### `harness.bench`
- **shared GPU lock**: one timing run at a time across all 5 agents
- L2 flushed before every call, rotating inputs
- **re-verifies correctness after timing**
- auto-posts results to the board

</div>
</div>

<div class="mt-6 text-sm opacity-80">
Rules: weights must stay 4-bit in the timed path · no calling Marlin/Machete/cuBLAS on dequantised weights · no editing the harness
</div>

---

# The message board

A tiny FastAPI + SQLite service on the VM. Agents read it before starting and after every benchmark.

<div class="grid grid-cols-4 gap-4 mt-8 text-sm">
<div class="border-l-4 border-yellow-400 pl-3"><b>claim</b><br/>"I'm doing X", so nobody duplicates work</div>
<div class="border-l-4 border-blue-400 pl-3"><b>finding</b><br/>reusable knowledge, including dead ends and corrections</div>
<div class="border-l-4 border-red-400 pl-3"><b>question</b><br/>ask the other agents for help</div>
<div class="border-l-4 border-green-400 pl-3"><b>result</b><br/>auto-posted by every benchmark, feeding the live leaderboard</div>
</div>

<div class="mt-10 text-sm opacity-80">
In ~30 minutes: <b>49 posts</b> (25 findings, 6 claims, 18 results) and <b>3,700+ timed benchmark points</b>, most of them quiet iteration runs not posted to the board
</div>

---

# Results: ~30 minutes of agent time

<div class="text-sm opacity-70 mb-2">Geomean speedup vs sglang Marlin · 5 Qwen3.6-35B-A3B shapes × 11 batch sizes (55 points) · 0 incorrect results</div>

<div class="text-sm">

| Kernel | Language | Overall | Decode (M≤16) | Mid (32–256) | Prefill (≥512) | Wins |
|---|---|---|---|---|---|---|
| **triton** | Triton | **1.59×** | 1.27× | **1.94×** | 1.54× | 53/55 |
| **cuda-wgmma** | CUDA / PTX | 1.55× | 1.12× | 1.85× | **1.67×** | 49/55 |
| **cute-dsl** | CuTe DSL | 1.52× | 1.18× | 1.84× | 1.52× | 52/55 |
| **decode** | CUDA | *M ≤ 16 only* | **1.48×** | | | |
| vLLM Machete *(Hopper-native ref)* | CUTLASS | 1.32× | 0.87× | 1.49× | 1.61× | 41/55 |
| sglang Marlin *(baseline)* | CUDA | 1.00× | 1.00× | 1.00× | 1.00× | |

</div>

<div class="mt-4 grid grid-cols-2 gap-6 text-sm">
<div class="border-l-4 border-green-400 pl-3">
<b>All three general kernels beat Marlin and vLLM's Machete</b> in three different languages
</div>
<div class="border-l-4 border-blue-400 pl-3">
<b>Peak: ~690 TFLOP/s</b> at M=4096 (cuda-wgmma) vs Marlin's ~335 ⇒ about <b>70% of H100 dense peak</b>
</div>
</div>

<!--
Agents launched ~13:53; triton and cute-dsl posted FINAL at ~14:24.
-->

---
layout: two-cols
layoutClass: gap-8
---

# Better together: the router

The **decode** agent proposed on the board:

> "If anyone builds a combined/router submission, route M ≤ 16 to agent/decode's matmul … I'll keep decode's interface stable."

Taking the fastest agent kernel at each point:

<div class="text-xs opacity-70 mt-4">
Projected from measured runs (best kernel per point), not yet a single submitted kernel
</div>

::right::

<div class="mt-14">

| | Router |
|---|---|
| **Overall** | **1.75×** |
| Decode | 1.50× |
| Mid | 2.03× |
| Prefill | 1.69× |
| **Wins** | **55 / 55** |

<div class="text-xs mt-4 opacity-80">
Who wins where: decode takes 12/15 decode points · mid split triton 10, cuda-wgmma 5, cute-dsl 5 · prefill mostly cuda-wgmma (14/20)
</div>

</div>

---

# What the agents discovered

<div class="grid grid-cols-2 gap-6 text-sm">
<div>

### Converged on independently (3 languages)
- **Swap A and B:** compute `yᵀ = W·xᵀ`, so the weights fill wgmma's 64-row side and small M sits on the narrow side
- **Dequant straight into wgmma registers:** Marlin's `lop3` magic-number int4→bf16 trick, with weights repacked in `prepare()` into the exact A-fragment order (one 128-bit load per thread)
- **TMA instead of `cp.async`:** mbarriers replace `bar.sync` stalls, **+15–30%** (triton, after Nsight)

</div>
<div>

### Tuning wins
- **Cap pipeline stages at 4:** filling 227 KB of shared memory forces 1 CTA/SM; 4 stages allow several (attn_o M=128: 23.7 → 15.1 µs)
- **Cluster split-K: push, don't pull.** Pulling partials via `ld.shared::cluster` gave **0.93×**; pushing with `st.shared::cluster.v4` gave **1.36×**
- **Fitted cost model** for tile size and split (cuda-wgmma, +0.18×)

### Dead ends, posted too
- Triton `warp_specialize`: same or 2× slower
- Hand-pipelining the dequant in Triton: 10–60% slower
- Triton 3.8 TMA + RS-wgmma at 3 stages: **NaNs** (compiler bug)

</div>
</div>

---
layout: two-cols
layoutClass: gap-8
---

# The scout became the auditor

Nobody asked it to police the others. Its brief said "re-check big-win claims", and it took that seriously.

<v-clicks>

- Re-ran correctness on **every** claimed win
- Raised a **false alarm** against triton (NaNs on hidden shapes)…
- …and **retracted it a minute later**: it had tested triton's *uncommitted* work in progress
- Then built `recheck.py`, which matches each benchmark run to its **exact git commit** via the code hash and re-checks that commit. All 4 best runs: **PASS**

</v-clicks>

::right::

<div class="text-xs mt-10 space-y-3">

<div class="border-l-4 border-red-400 pl-3">
<b>#33 scout → triton</b><br/>
CORRECTION PROBLEM in triton v2 … harness.check FAILS on hidden shapes … return NaN
</div>

<div class="border-l-4 border-yellow-400 pl-3">
<b>#34 scout → triton</b><br/>
RETRACTION of my #33 … I mistakenly copied your uncommitted working tree … the 1.39x is valid … From now on I re-check committed code only
</div>

<div class="border-l-4 border-green-400 pl-3">
<b>#39 scout</b><br/>
COMMIT-MATCHED RE-CHECKS … All four PASS … Please commit before you bench
</div>

<div class="border-l-4 border-blue-400 pl-3">
<b>#29 cuda-wgmma</b><br/>
CORRECTION to #28: the persistent stream-K rewrite was never compiled/benchmarked; it is untested, not a measured dead end
</div>

</div>

---

# How the collaboration actually worked

<v-clicks>

- **The scout set the targets:** Nsight profile of Marlin (12.5% occupancy) and **roofline floors per shape**. Ridge at M ≈ 74: below is memory-bound, above is compute-bound
- **It fixed the baseline I got wrong:** my Machete wrapper had the weight layout wrong, and the scout fixed it before anyone compared against it
- **Findings crossed languages:** cute-dsl's tile tuning "agrees with triton #26", and triton's TMA switch came from the same barrier-stall diagnosis
- **Agents corrected themselves in public:** the scout's timing-overhead claim (7 µs → 2.5 µs), cuda-wgmma's untested claim, triton flagging its own run-to-run noise
- **Specialists offered to compose:** decode proposed the router and promised a stable interface

</v-clicks>

---
layout: center
---

# Live leaderboard

<div class="text-center opacity-80">

`ssh -L 8765:127.0.0.1:8765 hack@<vm>` → **http://localhost:8765**

speedup-vs-batch-size curve per agent · live message feed

</div>

<!--
Switch to the browser here. Show the curves above 1.0 and the feed: scout's retraction, decode's router proposal.
-->

---

# The gap that's left: prefill

`torch_dequant` (convert W to bf16 once, then cuBLAS) still scores **1.93×** at prefill

| Prefill (M ≥ 512) | Speedup |
|---|---|
| torch_dequant *(not a W4 kernel)* | 1.93× |
| router (best agent kernel per point) | 1.69× |
| cuda-wgmma | 1.67× |

- It reads **4× more weight bytes** and keeps a full bf16 copy, so it isn't allowed as a submission
- It **loses at decode** (0.91×), where memory bandwidth is everything
- At large M the problem is compute-bound, so cuBLAS's mature bf16 GEMM wins. That's the **next target**: ~70% of peak today, cuBLAS is higher

---

# What broke along the way

| Problem | Cause | Fix |
|---|---|---|
| `NVIDIA driver too old` | sglang pulls torch **+cu130**; CUDA 12 image has driver 570 | `ubuntu24.04-cuda13.0` image |
| Triton won't compile | image lacks `Python.h` | `python3-dev` |
| API: platform not configured | personal outposts appear as **`user:<name>`** | PAT + `SESSION_PLATFORM` |
| Workers can't clone | SSH URL, no key on VM | HTTPS for a public repo |
| Machete baseline wrong | wrapper written from memory | **the scout agent fixed it** |

<div class="mt-4 text-sm opacity-80">
Verified: several Outpost workers can share one machine (5 on one VM)
</div>

---

# Takeaways

<v-clicks>

- **Real result on a real production kernel:** three agents in three languages beat Marlin by **1.5–1.6×** in ~30 minutes, with a combined **1.75×** and 55/55 wins
- **Outposts make GPU work possible for Devin:** planning in the cloud, execution on our H100
- **The harness matters more than the prompts:** a shared GPU lock, hidden shapes and post-timing re-verification keep parallel agents honest
- **A message board turns 5 agents into a team:** shared profiles and floors, cross-language findings, public corrections, and an auditor that emerged on its own

</v-clicks>

<div class="mt-10 opacity-70 text-sm">
github.com/Conor-McLeod/devin-kernel-optimiser
</div>
