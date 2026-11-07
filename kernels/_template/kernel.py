"""Starter W4A16 kernel in Triton. Copy this directory to kernels/<agent>-<dsl>/ and go.

    cp -r kernels/_template kernels/alice-triton
    uv run python -m harness.check --kernel kernels/alice-triton
    uv run python -m harness.bench --kernel kernels/alice-triton --note "baseline template"

Deliberately simple: no pipelining tuning, no split-K, no autotune, dequant in
registers right before tl.dot. Expect it to lose to Marlin everywhere. That's
the point: it is a correct starting layout, not a competitor.

Layout chosen in prepare():
  qweight int32 [K/8, N]  (GPTQ packing, 8 nibbles along K per word, low nibble first)
  scales  bf16  [K/128, N]
"""

import torch
import triton
import triton.language as tl

from harness.formats import pack_int4_along_k, scales_kn

META = {"name": "template-triton", "agent": "template", "dsl": "triton", "format": "int4_g128"}


@triton.jit
def _w4a16_kernel(
    x_ptr, qw_ptr, s_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_qk, stride_qn,
    stride_sg, stride_sn,
    stride_ym, stride_yn,
    GROUP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        x = tl.load(x_ptr + rm[:, None] * stride_xm + kk[None, :] * stride_xk, mask=rm[:, None] < M, other=0.0)
        q = tl.load(qw_ptr + (kk[:, None] // 8) * stride_qk + rn[None, :] * stride_qn)  # [BK, BN] int32
        code = (q >> ((kk[:, None] % 8) * 4)) & 0xF
        s = tl.load(s_ptr + (kk[:, None] // GROUP) * stride_sg + rn[None, :] * stride_sn)
        w = ((code - 8).to(tl.float32) * s.to(tl.float32)).to(tl.bfloat16)
        acc += tl.dot(x, w)
    tl.store(y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn, acc.to(tl.bfloat16), mask=rm[:, None] < M)


def prepare(qw):
    assert qw.fmt == "int4_g128"
    return {"q": pack_int4_along_k(qw.codes), "s": scales_kn(qw), "N": qw.N, "K": qw.K, "group": qw.group}


def matmul(x, p):
    M, K = x.shape
    N = p["N"]
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    BM = 16 if M <= 16 else 64
    BN, BK = 64, 64
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _w4a16_kernel[grid](
        x, p["q"], p["s"], y, M, N, K,
        x.stride(0), x.stride(1),
        p["q"].stride(0), p["q"].stride(1),
        p["s"].stride(0), p["s"].stride(1),
        y.stride(0), y.stride(1),
        GROUP=p["group"], BM=BM, BN=BN, BK=BK,
    )
    return y
