"""Baseline: vLLM Machete, a Hopper-native (CUTLASS wgmma) mixed-input GEMM.

This is the "someone already did a Hopper Marlin" reference. Run it from the
separate vLLM venv so it doesn't fight sglang over torch versions:

    /opt/hackathon/venv-vllm/bin/python -m harness.bench --baseline vllm_machete

UNVERIFIED: the machete op signatures below are from memory of vLLM's
_custom_ops.py. If this fails, look up `machete_prepack_B` / `machete_mm` in the
installed vllm and fix the call; it is a good first task for a scout agent.
"""

import torch

from harness.formats import pack_int4_along_k, scales_kn

META = {"name": "vllm_machete", "dsl": "cutlass (machete)", "format": "int4_g128"}


def prepare(qw):
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types

    assert qw.fmt == "int4_g128"
    qweight = pack_int4_along_k(qw.codes)  # [K/8, N] int32, same as GPTQ
    b = ops.machete_prepack_B(qweight, torch.bfloat16, scalar_types.uint4b8, torch.bfloat16)
    return {"b": b, "s": scales_kn(qw), "group": qw.group, "type": scalar_types.uint4b8}


def matmul(x, p):
    from vllm import _custom_ops as ops

    return ops.machete_mm(a=x, b_q=p["b"], b_type=p["type"], b_group_scales=p["s"], b_group_size=p["group"])
