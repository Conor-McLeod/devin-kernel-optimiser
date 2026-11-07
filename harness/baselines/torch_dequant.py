"""Baseline: dequantise the whole weight to bf16 once, then cuBLAS.

prepare() materialises the bf16 weight, so this is an *upper bound on what
skipping the dequant costs* at large M, and a memory-bandwidth lower bound
to beat at small M (it reads 4x more weight bytes than a W4 kernel).
Also used for CPU smoke tests of the harness.
"""

import torch

META = {"name": "torch_dequant", "dsl": "torch", "format": "int4_g128"}


def prepare(qw):
    return qw.dequant_fp32().to(torch.bfloat16).contiguous()


def matmul(x, w_bf16):
    return x @ w_bf16.t()
