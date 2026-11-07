"""Baseline: vLLM Machete, a Hopper-native (CUTLASS wgmma) mixed-input GEMM.

This is the "someone already did a Hopper Marlin" reference. Run it from the
separate vLLM venv so it doesn't fight sglang over torch versions:

    /opt/hackathon/venv-vllm/bin/python -m harness.bench --baseline vllm_machete

Verified by scout against the installed vllm (MacheteLinearKernel):
machete_prepack_B wants the [K/8, N] GPTQ qweight in column-major strides
(stride(1) == 1 fails; vLLM passes `w.t().contiguous().t()`), and requires
N % 128 == 0. For other N we pad with zero-weight columns (code 8) and return a
view of the first N output columns.
"""

import torch

from harness.formats import pack_int4_along_k

META = {"name": "vllm_machete", "dsl": "cutlass (machete)", "format": "int4_g128"}


def prepare(qw):
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types

    assert qw.fmt == "int4_g128"
    codes, scales = qw.codes, qw.scales
    pad = (-qw.N) % 128
    if pad:
        codes = torch.cat([codes, torch.full((pad, qw.K), 8, dtype=codes.dtype, device=codes.device)])
        scales = torch.cat([scales, torch.ones((pad, scales.shape[1]), dtype=scales.dtype, device=scales.device)])
    qweight = pack_int4_along_k(codes)  # [K/8, N] int32, same as GPTQ
    qweight = qweight.t().contiguous().t()  # column-major, as vLLM's MacheteLinearKernel does
    b = ops.machete_prepack_B(qweight, torch.bfloat16, scalar_types.uint4b8, torch.bfloat16)
    s = scales.t().contiguous()  # [K/group, N]
    return {"b": b, "s": s, "group": qw.group, "type": scalar_types.uint4b8, "N": qw.N, "pad": pad}


def matmul(x, p):
    from vllm import _custom_ops as ops

    y = ops.machete_mm(a=x, b_q=p["b"], b_type=p["type"], b_group_scales=p["s"], b_group_size=p["group"])
    return y[:, : p["N"]] if p["pad"] else y
