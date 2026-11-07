"""Benchmark problem definitions."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from .config import SHAPES_FILE

# Legality constraints every kernel must handle (same as Marlin).
N_MULTIPLE = 64
K_MULTIPLE = 128


@dataclass(frozen=True)
class Shape:
    name: str
    N: int
    K: int


def load(path=SHAPES_FILE) -> tuple[str, list[int], list[Shape]]:
    cfg = json.loads(open(path).read())
    shapes = [Shape(s["name"], int(s["N"]), int(s["K"])) for s in cfg["shapes"]]
    for s in shapes:
        assert s.N % N_MULTIPLE == 0 and s.K % K_MULTIPLE == 0, f"illegal shape {s}"
    return cfg["format"], [int(m) for m in cfg["ms"]], shapes


def hidden_shapes(rng: random.Random, n: int = 3) -> list[Shape]:
    """Random legal shapes not in shapes.json, so kernels can't special-case the public set."""
    out = []
    for i in range(n):
        N = N_MULTIPLE * rng.randint(4, 96)
        K = K_MULTIPLE * rng.randint(2, 48)
        out.append(Shape(f"hidden{i}", N, K))
    return out


# M values used by the correctness check: odd sizes catch tail-handling bugs.
CHECK_MS = [1, 3, 16, 17, 64, 100, 256, 777, 2048]
