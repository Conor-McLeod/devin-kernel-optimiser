"""Naive submission: dequantise inside matmul() every call, then torch matmul.

Exists to exercise the submission loader and as the slowest sane W4A16 kernel.
Runs on CPU, so `harness.check --kernel kernels/reference_torch --device cpu --quick` works on a laptop.
"""

import torch

META = {"name": "reference_torch", "agent": "human", "dsl": "torch", "format": "int4_g128"}


def prepare(qw):
    return {"codes": qw.codes, "scales": qw.scales, "group": qw.group}


def matmul(x, p):
    w = (p["codes"].to(torch.bfloat16) - 8) * p["scales"].repeat_interleave(p["group"], dim=1)
    return x @ w.t()
