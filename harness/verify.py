"""Correctness comparison against the fp32 ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .formats import QuantWeight

# bf16 output + bf16 dequantised weights => ~3 significant digits.
REL_FRO_TOL = 1e-2
MAX_ERR_TOL = 4e-2  # max |y - ref| relative to max |ref|


@dataclass
class Verdict:
    ok: bool
    rel_fro: float
    max_err: float
    msg: str = ""


def reference(x: torch.Tensor, w_fp32: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32) @ w_fp32.t()


def compare(y: torch.Tensor, ref: torch.Tensor, M: int, N: int) -> Verdict:
    if not isinstance(y, torch.Tensor):
        return Verdict(False, float("nan"), float("nan"), f"returned {type(y).__name__}, not a tensor")
    if tuple(y.shape) != (M, N):
        return Verdict(False, float("nan"), float("nan"), f"shape {tuple(y.shape)} != {(M, N)}")
    if y.dtype != torch.bfloat16:
        return Verdict(False, float("nan"), float("nan"), f"dtype {y.dtype} != bfloat16")
    yf = y.to(torch.float32)
    if not torch.isfinite(yf).all():
        return Verdict(False, float("nan"), float("nan"), "non-finite values in output")
    diff = yf - ref
    rel_fro = (diff.norm() / ref.norm().clamp_min(1e-12)).item()
    max_err = (diff.abs().max() / ref.abs().max().clamp_min(1e-12)).item()
    ok = rel_fro < REL_FRO_TOL and max_err < MAX_ERR_TOL
    return Verdict(ok, rel_fro, max_err, "" if ok else "error above tolerance")


def check_once(sub, qw: QuantWeight, prepared, M: int, gen: torch.Generator, w_fp32=None) -> Verdict:
    dev = qw.device
    x = torch.randn(M, qw.K, generator=gen, dtype=torch.float32).to(torch.bfloat16).to(dev)
    w = w_fp32 if w_fp32 is not None else qw.dequant_fp32()
    ref = reference(x, w)
    try:
        y = sub.matmul(x, prepared)
    except Exception as e:  # noqa: BLE001
        return Verdict(False, float("nan"), float("nan"), f"{type(e).__name__}: {e}")
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return compare(y, ref, M, qw.N)
