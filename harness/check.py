"""Correctness check. Run this before every benchmark.

    uv run python -m harness.check --kernel kernels/<your-dir>
    uv run python -m harness.check --baseline torch_dequant --device cpu --quick

Uses fresh random seeds every run, odd M values, and hidden random shapes.
Exit code 0 = all passed.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import torch

from . import shapes as shp
from .formats import random_weight
from .submission import load
from .verify import check_once


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel")
    ap.add_argument("--baseline")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=None, help="default: random each run")
    ap.add_argument("--quick", action="store_true", help="small shapes only (CPU smoke test)")
    args = ap.parse_args(argv)

    sub = load(args.kernel, args.baseline)
    fmt, _, public = shp.load()
    if sub.fmt != fmt:
        print(f"note: submission format {sub.fmt} != shapes.json format {fmt}; checking {sub.fmt}")
        fmt = sub.fmt

    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)

    if args.quick:
        cases = [shp.Shape("quick0", 256, 512), shp.Shape("quick1", 192, 256)]
        ms = [1, 3, 17, 64]
    else:
        cases = public + shp.hidden_shapes(rng)
        ms = shp.CHECK_MS

    print(f"checking {sub.name} ({sub.dsl}, {fmt}) seed={seed} on {args.device}")
    failures = 0
    for s in cases:
        qw = random_weight(fmt, s.N, s.K, gen, device=args.device)
        try:
            prepared = sub.prepare(qw)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {s.name:12s} N={s.N} K={s.K}: prepare raised {type(e).__name__}: {e}")
            failures += len(ms)
            continue
        w = qw.dequant_fp32()
        for M in ms:
            v = check_once(sub, qw, prepared, M, gen, w_fp32=w)
            tag = "ok  " if v.ok else "FAIL"
            print(f"  {tag} {s.name:12s} N={s.N:6d} K={s.K:6d} M={M:5d}  rel_fro={v.rel_fro:.2e} max={v.max_err:.2e} {v.msg}")
            failures += not v.ok

    print("PASS" if failures == 0 else f"{failures} FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
