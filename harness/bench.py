"""The only sanctioned way to time a kernel.

    uv run python -m harness.bench --kernel kernels/<your-dir>
    uv run python -m harness.bench --baseline sglang_marlin
    uv run python -m harness.bench --kernel kernels/<dir> --shapes attn_o --ms 1,64,1024   # quick iteration

What it does:
  * takes the shared GPU lock (one timing run at a time across all agents)
  * L2 flush before every timed call, rotating activation buffers, median of N
  * re-verifies correctness on fresh inputs after timing; incorrect runs are recorded as failed
  * appends results to <shared>/results.jsonl and posts them to the message board
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import torch

from . import shapes as shp
from .config import BOARD_URL, PRIMARY_BASELINE, REPO_ROOT, shared_dir
from .formats import random_weight
from .gpulock import gpu_lock
from .submission import load
from .verify import check_once

L2_FLUSH_BYTES = 256 * 1024 * 1024  # H100 L2 is 50 MB
N_ACT_BUFFERS = 4


def _code_hash(path: str) -> str:
    p = REPO_ROOT / path
    d = p.parent
    h = hashlib.sha1()
    for f in sorted(d.rglob("*")):
        if f.is_file() and f.suffix in {".py", ".cu", ".cuh", ".h", ".cpp"} and "__pycache__" not in f.parts:
            h.update(f.read_bytes())
    return h.hexdigest()[:10]


def time_kernel(fn, xs, prepared, iters: int, warmup: int) -> float:
    flush = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    for i in range(warmup):
        fn(xs[i % len(xs)], prepared)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flush.zero_()
        starts[i].record()
        fn(xs[i % len(xs)], prepared)
        ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends))


def baseline_times(name: str) -> dict:
    """Latest correct timing per (N, K, M) for a baseline, from results.jsonl."""
    out = {}
    f = shared_dir() / "results.jsonl"
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if r["submission"] == name and r["correct"]:
            out[(r["N"], r["K"], r["M"])] = r["ms"]
    return out


def post_results(rows: list[dict]) -> None:
    try:
        req = urllib.request.Request(
            f"{BOARD_URL}/api/results",
            data=json.dumps(rows).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5).read()
    except Exception as e:  # noqa: BLE001
        print(f"(could not post to board at {BOARD_URL}: {e})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel")
    ap.add_argument("--baseline")
    ap.add_argument("--shapes", help="comma-separated shape names (default: all)")
    ap.add_argument("--ms", help="comma-separated M values (default: shapes.json)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--note", default="", help="what changed in this version (shown on the board)")
    ap.add_argument("--no-post", action="store_true")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("bench needs a CUDA GPU (use harness.check --device cpu for smoke tests)")

    sub = load(args.kernel, args.baseline)
    fmt, ms, cases = shp.load()
    fmt = sub.fmt
    if args.shapes:
        wanted = set(args.shapes.split(","))
        cases = [c for c in cases if c.name in wanted]
    if args.ms:
        ms = [int(m) for m in args.ms.split(",")]

    run_id = uuid.uuid4().hex[:8]
    code_hash = _code_hash(sub.path) if not sub.is_baseline else "baseline"
    gpu = torch.cuda.get_device_name()
    gen = torch.Generator().manual_seed(int.from_bytes(os.urandom(4), "little"))
    base = baseline_times(PRIMARY_BASELINE) if sub.name != PRIMARY_BASELINE else {}

    rows = []
    with gpu_lock(holder=f"{sub.agent}/{sub.name}"):
        print(f"bench {sub.name} ({sub.dsl}, {fmt}) run={run_id} code={code_hash} on {gpu}")
        hdr = f"{'shape':12s} {'N':>6s} {'K':>6s} {'M':>5s} {'ms':>9s} {'TFLOP/s':>8s} {'GB/s':>7s} {'vs ' + PRIMARY_BASELINE:>18s}"
        print(hdr)
        for s in cases:
            qw = random_weight(fmt, s.N, s.K, gen, device="cuda")
            prepared = sub.prepare(qw)
            torch.cuda.synchronize()
            w = qw.dequant_fp32()
            for M in ms:
                xs = [torch.randn(M, s.K, generator=gen).to(torch.bfloat16).cuda() for _ in range(N_ACT_BUFFERS)]
                err = ""
                try:
                    t = time_kernel(sub.matmul, xs, prepared, args.iters, args.warmup)
                    v = check_once(sub, qw, prepared, M, gen, w_fp32=w)
                    correct, err = v.ok, v.msg
                except Exception as e:  # noqa: BLE001
                    t, correct, err = float("nan"), False, f"{type(e).__name__}: {e}"
                flops = 2 * M * s.N * s.K
                bytes_ = qw.nbytes_packed() + 2 * M * s.K + 2 * M * s.N
                b = base.get((s.N, s.K, M))
                speed = f"{b / t:6.2f}x" if (b and correct and t == t) else "-"
                flag = "" if correct else f"  INCORRECT {err}"
                print(f"{s.name:12s} {s.N:6d} {s.K:6d} {M:5d} {t * 1e3:8.1f}us {flops / t / 1e9:8.1f} {bytes_ / t / 1e6:7.0f} {speed:>18s}{flag}")
                rows.append({
                    "ts": time.time(), "run_id": run_id, "submission": sub.name, "agent": sub.agent,
                    "dsl": sub.dsl, "fmt": fmt, "is_baseline": sub.is_baseline, "code_hash": code_hash,
                    "shape": s.name, "N": s.N, "K": s.K, "M": M, "ms": t, "correct": correct,
                    "error": err, "gpu": gpu, "note": args.note,
                })
            del prepared, qw, w
            torch.cuda.empty_cache()

    with open(shared_dir() / "results.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if not args.no_post:
        post_results(rows)
    bad = sum(not r["correct"] for r in rows)
    print(f"done: {len(rows)} points, {bad} incorrect. results -> {shared_dir() / 'results.jsonl'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
