"""Loading kernels (agent submissions and baselines) behind one interface.

A submission is a directory containing `kernel.py` that defines:

    META = {"name": "alice-triton", "agent": "alice", "dsl": "triton", "format": "int4_g128"}

    def prepare(qw: harness.formats.QuantWeight):
        # Untimed. Repack codes/scales into whatever layout your kernel wants.
        # Must only depend on the weight, never on activations.
        return anything

    def matmul(x: torch.Tensor, prepared) -> torch.Tensor:
        # Timed. x: [M, K] bf16 contiguous, returns y = x @ W.T as [M, N] bf16.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from .config import REPO_ROOT


@dataclass
class Submission:
    name: str
    agent: str
    dsl: str
    fmt: str
    is_baseline: bool
    path: str
    prepare: Callable[[Any], Any]
    matmul: Callable[[Any, Any], Any]


def _from_module(mod: ModuleType, path: str, is_baseline: bool) -> Submission:
    meta = getattr(mod, "META", {})
    for attr in ("prepare", "matmul"):
        if not callable(getattr(mod, attr, None)):
            raise TypeError(f"{path}: missing `{attr}()`")
    name = meta.get("name") or Path(path).stem
    return Submission(
        name=name,
        agent=meta.get("agent", "baseline" if is_baseline else "unknown"),
        dsl=meta.get("dsl", "unknown"),
        fmt=meta.get("format", "int4_g128"),
        is_baseline=is_baseline,
        path=path,
        prepare=mod.prepare,
        matmul=mod.matmul,
    )


def load_kernel_dir(path: str | Path) -> Submission:
    p = Path(path).resolve()
    f = p / "kernel.py" if p.is_dir() else p
    if not f.exists():
        raise FileNotFoundError(f)
    mod_name = f"submission_{f.parent.name.replace('-', '_')}"
    # Make sibling helper modules importable from kernel.py.
    sys.path.insert(0, str(f.parent))
    spec = importlib.util.spec_from_file_location(mod_name, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return _from_module(mod, str(f.relative_to(REPO_ROOT) if f.is_relative_to(REPO_ROOT) else f), False)


def load_baseline(name: str) -> Submission:
    mod = importlib.import_module(f"harness.baselines.{name}")
    sub = _from_module(mod, f"harness/baselines/{name}.py", True)
    sub.name = name
    return sub


def load(kernel: str | None, baseline: str | None) -> Submission:
    if (kernel is None) == (baseline is None):
        raise SystemExit("pass exactly one of --kernel DIR or --baseline NAME")
    return load_kernel_dir(kernel) if kernel else load_baseline(baseline)
