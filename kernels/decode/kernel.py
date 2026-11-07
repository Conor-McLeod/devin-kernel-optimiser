"""decode: Hopper W4A16 GEMM specialised for small M (decode).

Swap-AB mma.sync (A = dequantised int4 W tile, B = x), weight repacked in prepare()
into exact A-fragment order so every lane streams 128-bit coalesced loads, all loads
issued up front, split-K over warps (smem) and over CTAs of a thread-block cluster
(DSMEM reduction, no atomics / workspace / extra kernels). See w4a16_decode.cu.
"""

import os
from pathlib import Path

import torch

META = {"name": "decode", "agent": "decode", "dsl": "cuda", "format": "int4_g128"}

_HERE = Path(__file__).resolve().parent
_ext = None
_cfgs = None


def _load():
    global _ext, _cfgs
    if _ext is None:
        from torch.utils.cpp_extension import load

        build = _HERE / "build"
        build.mkdir(exist_ok=True)
        _ext = load(
            name="decode_w4a16_ext",
            sources=[str(_HERE / "w4a16_decode.cu")],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math", "-std=c++20"],
            build_directory=str(build),
            verbose=False,
        )
        _cfgs = {tuple(c) for c in _ext.configs()}
    return _ext


def prepare(qw):
    assert qw.fmt == "int4_g128"
    _load()
    N, K = qw.N, qw.K
    assert N % 16 == 0 and K % 128 == 0
    NT, KB = N // 16, K // 64
    c = qw.codes.to(torch.int32)
    # row = 16*nt + 8*h + g ; k = 64*kb + 16*t + 4*j + 2*q + e ; nibble = h + 2q + 4e
    c = c.view(NT, 2, 8, KB, 4, 4, 2, 2)  # nt h g kb t j q e
    c = c.permute(0, 3, 2, 4, 5, 7, 6, 1).contiguous()  # nt kb g t j e q h
    c = c.view(NT, KB, 32, 4, 8)
    shifts = (torch.arange(8, device=c.device, dtype=torch.int32) * 4)
    words = (c << shifts).sum(dim=-1, dtype=torch.int64).to(torch.int64)
    words = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32).contiguous()
    # scales [N, G] -> [NT, G, 8] packed bf16x2 (row g, row g+8)
    G = K // 128
    s = qw.scales.view(NT, 2, 8, G).permute(0, 3, 2, 1).contiguous()  # nt G g h
    s = s.view(torch.int32).view(NT, G, 8).contiguous()
    return {"w": words, "s": s, "N": N, "K": K}


def _pick(M, N, K):
    env = os.environ.get("DECODE_CFG")
    MB = 1 if M <= 8 else 2
    if env:
        T, U, WC, S = (int(v) for v in env.split(","))
        return T, U, MB, WC, S
    NT, KB = N // 16, K // 64
    best = None
    for (T, U, mb, WC, S) in _cfgs:
        if mb != MB or T != 1:
            continue
        cover = S * WC * U
        if cover < KB:
            continue
        ctas = NT // T * S
        # prefer >= 2 waves-worth of CTAs, then U close to 2, then fewer idle warp-steps, then WC=8
        score = ((cover - KB) >= WC * U, min(ctas, 264) * -1, abs(U - 2), cover - KB, -WC, S)
        if best is None or score < best[0]:
            best = (score, (T, U, MB, WC, S))
    if best is None:
        raise RuntimeError(f"no config for N={N} K={K}")
    return best[1]


def matmul(x, p):
    ext = _load()
    M, K = x.shape
    N = p["N"]
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    for m0 in range(0, M, 16):
        mc = min(16, M - m0)
        T, U, MB, WC, S = _pick(mc, N, K)
        ext.run(p["w"], p["s"], x[m0:m0 + mc], y[m0:m0 + mc], mc, N, K, T, U, MB, WC, S)
    return y
