"""decode: Hopper W4A16 GEMM specialised for small M (decode).

Default path (tma.cu): persistent, warp-specialised. One CTA per SM streams a contiguous
range of the repacked weight with cp.async.bulk + mbarrier into an smem ring; consumer
warps dequantise (lop3) and run swap-AB mma.sync (x is the 8-wide B operand).
Fallback path (w4a16_decode.cu, DECODE_IMPL=oneshot): one-shot register loads.
"""

import os
from pathlib import Path

import torch

META = {"name": "decode", "agent": "decode", "dsl": "cuda", "format": "int4_g128"}

_HERE = Path(__file__).resolve().parent
_FLAGS = ["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math", "-std=c++20"]
_mods = {}
_cfgs = None
_nsm = None


def _ext(name):
    global _nsm
    if name not in _mods:
        from torch.utils.cpp_extension import load

        pol = os.environ.get("DECODE_POLICY", "0")
        build = _HERE / f"build_{name}_p{pol}"
        build.mkdir(exist_ok=True)
        src = {"tma": "tma.cu", "oneshot": "w4a16_decode.cu"}[name]
        _mods[name] = load(name=f"decode_{name}_p{pol}_ext", sources=[str(_HERE / src)],
                           extra_cuda_cflags=_FLAGS + [f"-DSTREAM_POLICY={pol}"],
                           build_directory=str(build), verbose=False)
        _nsm = torch.cuda.get_device_properties(0).multi_processor_count
    return _mods[name]


def _impl():
    return os.environ.get("DECODE_IMPL", "oneshot")


def prepare(qw):
    assert qw.fmt == "int4_g128"
    N, K = qw.N, qw.K
    assert N % 16 == 0 and K % 128 == 0
    NT, KB, G = N // 16, K // 64, K // 128
    c = qw.codes.to(torch.int32)
    # row = 16*nt + 8*h + g ; k = 64*kb + 16*t + 4*j + 2*q + e ; nibble = h + 2q + 4e
    c = c.view(NT, 2, 8, KB, 4, 4, 2, 2)  # nt h g kb t j q e
    c = c.permute(0, 3, 2, 4, 5, 7, 6, 1).contiguous()  # nt kb g t j e q h
    c = c.view(NT, KB, 32, 4, 8)
    shifts = torch.arange(8, device=c.device, dtype=torch.int32) * 4
    words = (c << shifts).sum(dim=-1, dtype=torch.int64)
    words = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32).contiguous()  # [NT, KB, 32, 4]
    # scales [N, G] -> [NT, G, 8] packed bf16x2 (row g, row g+8)
    s = qw.scales.view(NT, 2, 8, G).permute(0, 3, 2, 1).contiguous().view(torch.int32).view(NT, G, 8)
    # units: [NT, G, 2 blocks * 128 words + 8 scale words] = 1056 B each
    units = torch.cat([words.view(NT, G, 256), s], dim=-1).contiguous()
    p = {"w": words, "s": s.contiguous(), "u": units, "N": N, "K": K,
         "ws": torch.zeros(16, N, dtype=torch.float32, device=c.device),
         "cnt": torch.zeros(NT, dtype=torch.int32, device=c.device)}
    _ext(_impl())
    return p


def _tma_cfg(M, N, K):
    MB = 1 if M <= 8 else 2
    env = os.environ.get("DECODE_TMA")
    if env:
        NC, U, NS, ctas = (int(v) for v in env.split(","))
    else:
        NC, U, NS, ctas = 8, 2, 8, 0
    total = (N // 16) * (K // 128)
    ctas = min(total, ctas if ctas > 0 else _nsm)
    return MB, NC, U, NS, ctas


def _pick_oneshot(M, N, K):
    global _cfgs
    ext = _ext("oneshot")
    if _cfgs is None:
        _cfgs = {tuple(c) for c in ext.configs()}
    MB = 1 if M <= 8 else 2
    NT, KB = N // 16, K // 64
    best = None
    for (T, U, mb, WC, S) in _cfgs:
        if mb != MB or T != 1 or S * WC * U < KB:
            continue
        cover = S * WC * U
        score = ((cover - KB) >= WC * U, S, -min(NT * S, 264), abs(U - 4), cover - KB, -WC)
        if best is None or score < best[0]:
            best = (score, (T, U, MB, WC, S))
    return best[1]


def matmul(x, p):
    M, K = x.shape
    N = p["N"]
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    impl = _impl()
    ext = _ext(impl)
    for m0 in range(0, M, 16):
        mc = min(16, M - m0)
        xs, ys = x[m0:m0 + mc], y[m0:m0 + mc]
        if impl == "tma":
            MB, NC, U, NS, ctas = _tma_cfg(mc, N, K)
            ext.run(p["u"], xs, ys, p["ws"], p["cnt"], mc, N, K, MB, NC, U, NS, ctas)
        else:
            T, U, MB, WC, S = _pick_oneshot(mc, N, K)
            ext.run(p["w"], p["s"], xs, ys, mc, N, K, T, U, MB, WC, S)
    return y
