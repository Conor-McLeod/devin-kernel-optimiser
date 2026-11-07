"""Baseline: sglang's Marlin W4A16 GEMM (Ampere-style mma.sync, no wgmma/TMA).

This is the kernel to beat. Requires `sglang` installed (infra/setup_vm.sh).
Marlin is JIT-compiled on first use, so the first run is slow.

Format: int4_g128 == GPTQ uint4b8, group 128, no act-order, no zero points.
NOTE: written against sglang's python/sglang/srt/layers/quantization/marlin_utils.py;
verify it on the VM with `python -m harness.check --baseline sglang_marlin` before
trusting any numbers.
"""

import torch

from harness.formats import pack_int4_along_k, scales_kn

META = {"name": "sglang_marlin", "dsl": "cuda (marlin)", "format": "int4_g128"}

_sg = None


def _import():
    global _sg
    if _sg is None:
        from sglang.kernels.ops.quantization import gptq_marlin_repack
        from sglang.srt.layers.quantization import marlin_utils as mu

        _sg = (gptq_marlin_repack, mu)
    return _sg


def prepare(qw):
    gptq_marlin_repack, mu = _import()
    assert qw.fmt == "int4_g128"
    dev = qw.device
    K, N = qw.K, qw.N
    qweight = pack_int4_along_k(qw.codes)  # int32 [K/8, N]
    perm = torch.empty(0, dtype=torch.int, device=dev)
    marlin_q = gptq_marlin_repack(qweight, perm, K, N, 4)
    s = mu.marlin_permute_scales(scales_kn(qw), size_k=K, size_n=N, group_size=qw.group)
    return {
        "q": marlin_q,
        "s": s,
        "zp": mu.marlin_make_empty_g_idx(dev),
        "g_idx": mu.marlin_make_empty_g_idx(dev),
        "sort": mu.marlin_make_empty_g_idx(dev),
        "ws": mu.marlin_make_workspace(dev),
        "wtype": mu.scalar_types.uint4b8,
        "N": N,
        "K": K,
    }


def matmul(x, p):
    _, mu = _import()
    return mu.apply_gptq_marlin_linear(
        input=x,
        weight=p["q"],
        weight_scale=p["s"],
        weight_zp=p["zp"],
        g_idx=p["g_idx"],
        g_idx_sort_indices=p["sort"],
        workspace=p["ws"],
        wtype=p["wtype"],
        output_size_per_partition=p["N"],
        input_size_per_partition=p["K"],
        is_k_full=True,
    )
