"""decode: Hopper W4A16 GEMM. M<=16: swap-AB mma.sync streaming kernel (w4a16_decode.cu).
M>16: swap-AB RS-wgmma + TMA warp-specialised kernel (wgmma.cu, mainloop from agent/cuda-wgmma)
with a 'tail split' schedule picked by simulating the CTA scheduler over 132 SMs.

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
        occ = os.environ.get("DECODE_OCC2", "1")
        tag = f"p{pol}o{occ}" if name == "wgmma" else f"p{pol}"
        build = _HERE / f"build_{name}_{tag}"
        build.mkdir(exist_ok=True)
        src = {"tma": "tma.cu", "oneshot": "w4a16_decode.cu", "wgmma": "wgmma.cu"}[name]
        flags = _FLAGS + [f"-DSTREAM_POLICY={pol}", f"-DWG_OCC2={occ}", f"-I{_HERE}"]
        _mods[name] = load(name=f"decode_{name}_{tag}_ext", sources=[str(_HERE / src)],
                           extra_cuda_cflags=flags, extra_ldflags=["-lcuda"] if name == "wgmma" else [],
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
    p = _prepare_wgmma(qw)
    p.update({"w": words, "s": s.contiguous(), "u": units, "N": N, "K": K,
         "ws": torch.zeros(16, N, dtype=torch.float32, device=c.device),
         "cnt": torch.zeros(NT, dtype=torch.int32, device=c.device)})
    _ext(_impl())
    _ext("wgmma")
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
    env = os.environ.get("DECODE_CFG")
    if env:
        T, U, WC, S = (int(v) for v in env.split(","))
        return T, U, MB, WC, S
    NT, KB = N // 16, K // 64
    best = None
    for (T, U, mb, WC, S) in _cfgs:
        if mb != MB or T != 1 or S * WC * U < KB:
            continue
        cover = S * WC * U
        score = ((cover - KB) >= U, U > 2, S, -WC, U)
        if best is None or score < best[0]:
            best = (score, (T, U, MB, WC, S))
    return best[1]


def _prepare_wgmma(qw):
    # agent/cuda-wgmma layout: wq [Np/128, K/64, 4096] B (m64k16 A fragments), scales [Np/128, K/128, 128]
    N, K = qw.N, qw.K
    Np = (N + 127) // 128 * 128
    codes, scales = qw.codes, qw.scales
    if Np != N:
        codes = torch.cat([codes, torch.full((Np - N, K), 8, dtype=codes.dtype, device=codes.device)])
        scales = torch.cat([scales, torch.zeros(Np - N, K // 128, dtype=scales.dtype, device=scales.device)])
    NT, KB = Np // 128, K // 64
    c = codes.to(torch.int64).view(NT, 2, 4, 2, 8, KB, 4, 2, 4, 2)
    c = c.permute(0, 5, 1, 2, 4, 8, 6, 9, 7, 3).reshape(NT, KB, 2, 4, 8, 4, 4, 8)
    shifts = torch.arange(8, device=codes.device, dtype=torch.int64) * 4
    words = (c << shifts).sum(-1)
    words = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
    wq = words.contiguous().view(torch.uint8).view(NT, KB, 4096)
    sc = scales.view(NT, 128, K // 128).permute(0, 2, 1).contiguous()
    return {"wq": wq, "wsc": sc}


_PERSIST = os.environ.get("DECODE_P", "1") == "1"
_BMS = (16, 32, 64, 96, 128, 192, 256)
_SPLITS = (1, 2, 3, 4, 6, 8)
_sched = {}
_bufs = {}


def _kb_cost(bm):
    return max(0.15, 0.2 + 0.0016 * bm)  # us per 128x(bm)x64 block, fit to harness sweeps


def _sim(M, N, K, bm, D, S):
    import heapq
    nsm = _nsm or 132
    KB = K // 64
    T = -(-N // 128) * -(-M // bm)
    R = T - D
    kbps = -(-KB // S)
    c = _kb_cost(bm)
    durs = [0.3 + KB * c] * D + [0.3 + kbps * c] * (R * S)
    if len(durs) <= nsm:
        t = max(durs)
    else:
        h = [0.0] * nsm
        for d in durs:
            heapq.heapreplace(h, h[0] + d)
        t = max(h)
    if R > 0 and S > 1:
        t += 1.0 + R * S * bm * 128 * 8 / 4e6
    return t


def _wg_cfg(M, N, K):
    key = (M, N, K)
    if key in _sched:
        return _sched[key]
    env = os.environ.get("DECODE_WG")
    if env:  # "bm,mode,S": mode -1 = all tiles split, 0 = T%nsm tail tiles split, 1 = T%nsm+nsm
        bm, mode, S = (int(v) for v in env.split(","))
        T = -(-N // 128) * -(-M // bm)
        nsm = _nsm or 132
        R = T if mode < 0 or S == 1 else min(T, T % nsm + mode * nsm)
        cfg = (bm, T - R, S)
    else:
        nsm = _nsm or 132
        KB = K // 64
        best = None
        for bm in _BMS:
            if not (bm <= 16 or bm // 2 < M):
                continue
            T = -(-N // 128) * -(-M // bm)
            for S in _SPLITS:
                if S > KB:
                    continue
                Rs = {0} if S == 1 else {T, T % nsm, T % nsm + nsm}
                for R in Rs:
                    if R > T or (S > 1 and R == 0):
                        continue
                    t = _sim(M, N, K, bm, T - R, S)
                    if best is None or t < best[0]:
                        best = (t, (bm, T - R, S))
        cfg = best[1]
    _sched[key] = cfg
    return cfg


def _wg_bufs(dev, nfloat):
    b = _bufs.get(dev)
    if b is None or b[1].numel() < nfloat:
        cnt = torch.zeros(1 << 16, dtype=torch.int32, device=dev) if b is None else b[0]
        b = (cnt, torch.empty(max(nfloat, 1 << 20), dtype=torch.float32, device=dev))
        _bufs[dev] = b
    return b


def matmul(x, p):
    M, K = x.shape
    N = p["N"]
    if M > 16:
        bm, D, S = _wg_cfg(M, N, K)
        T = -(-N // 128) * -(-M // bm)
        cnt, part = _wg_bufs(x.device, (T - D) * S * bm * 128)
        ctas = (_nsm or 132) if _PERSIST else 0
        return _ext("wgmma").w4a16_mm(x, p["wq"], p["wsc"], cnt, part, N, bm, D, S, ctas)
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
