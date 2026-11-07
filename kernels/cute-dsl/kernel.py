"""cute-dsl: Hopper W4A16 GEMM written in the CUTLASS CuTe Python DSL. See w4a16_sm90.py."""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_tensor
from cutlass.runtime import make_fake_stream

from w4a16_sm90 import W4A16Sm90

META = {"name": "cute-dsl", "agent": "cute-dsl", "dsl": "cute-dsl", "format": "int4_g128"}


def pack_weight(codes: torch.Tensor, N_pad: int) -> torch.Tensor:
    """codes [N, K] uint8 -> int32 [Kt64, Nb, 128 thr, 4 kblk] in RS-wgmma A-fragment order.

    thread t = 32w + 4g + tq, k16-block b; nibble (h + 2jj + 4e) holds
    W[nb*64 + 16w + 8h + g, kt*64 + 16b + 8jj + 2tq + e].
    """
    N, K = codes.shape
    c = torch.full((N_pad, K), 8, dtype=torch.int64, device=codes.device)
    c[:N] = codes.to(torch.int64)
    Nb, Kt = N_pad // 64, K // 64
    c = c.view(Nb, 4, 2, 8, Kt, 4, 2, 4, 2)  # nb w h g kt b jj tq e
    c = c.permute(4, 0, 1, 3, 7, 5, 8, 6, 2).reshape(Kt, Nb, 128, 4, 8)  # kt nb (w g tq) b (e jj h)
    shifts = (torch.arange(8, device=codes.device, dtype=torch.int64) * 4)
    words = (c << shifts).sum(-1)
    words = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
    return words.contiguous()


def pack_scales(scales: torch.Tensor, N_pad: int) -> torch.Tensor:
    """scales [N, G] bf16 -> int32 [G, Nb, 32]; word (w*8+g) = (lo: row 16w+g, hi: row 16w+g+8)."""
    N, G = scales.shape
    s = torch.zeros((N_pad, G), dtype=torch.bfloat16, device=scales.device)
    s[:N] = scales
    s = s.view(N_pad // 64, 4, 2, 8, G).permute(4, 0, 1, 3, 2).contiguous()
    return s.view(torch.int32).view(G, N_pad // 64, 32)


def prepare(qw):
    assert qw.fmt == "int4_g128" and qw.group == 128
    N_pad = (qw.N + 63) // 64 * 64
    w = pack_weight(qw.codes, N_pad)
    s = pack_scales(qw.scales, N_pad)
    Kt, Nb = w.shape[0], w.shape[1]
    G = s.shape[0]
    tw = w.view(Kt, Nb, 128, 4).permute(3, 2, 1, 0)
    ts = s.permute(2, 1, 0)
    return {"w": w, "s": s, "tw": tw, "ts": ts, "N": qw.N, "K": qw.K, "Nb": Nb, "G": G}


_compiled = {}


def _ctas(N, M, nwg, bm):
    return -(-N // (64 * nwg)) * -(-M // bm)


def _prefill_config(M, N, K, bk):
    # 1 CTA/SM for the 2-WG tiles: pick by wave count; a 128x256 tile costs ~1.72x a 128x128 one.
    c128, c256 = _ctas(N, M, 2, 128), _ctas(N, M, 2, 256)
    if c128 >= 128:
        if -(-c256 // 132) * 1.72 < -(-c128 // 132):
            return (2, 256, 64, 1, 0)
        return (2, 128, bk, 1, 0)
    if c128 * 2 >= 128 and (K // bk) % 4 == 0 and K // bk >= 16:
        return (2, 128, bk, 2, 0)
    for nwg, bm in ((1, 128), (1, 64)):
        if _ctas(N, M, nwg, bm) >= 128:
            break
    return (nwg, bm, bk, 1, 4)


def _config(M, N, K):
    """(num_wg, BM, BK, split, stages): largest tile giving >= 64 CTAs, then split-K
    towards ~132 CTAs. Fitted to ncu sweeps on 132-SM H100."""
    bk = 128 if (K // 128) % 2 == 0 else 64
    if M >= 512:
        return _prefill_config(M, N, K, bk)
    tiles = [t for t in ((2, 128, 0), (1, 64, 4), (1, 32, 4), (1, 16, 4)) if t[1] <= max(16, M)]
    for nwg, bm, stages in tiles:
        if _ctas(N, M, nwg, bm) >= 64:
            break
    ctas = _ctas(N, M, nwg, bm)
    max_split = 4 if bm == 16 else 2
    split = 1
    while (split < max_split and ctas * split * 2 <= 132 and (K // bk) % (4 * split) == 0
           and (2 * split - 1) * 64 * nwg * bm * 4 <= 72 * 1024):
        split *= 2
    return (nwg, bm, bk, split, stages)


def _compile(cfg, N, K, Nb, G):
    m = cute.sym_int()
    fx = make_fake_compact_tensor(cutlass.BFloat16, (m, K), stride_order=(1, 0), assumed_align=16)
    fy = make_fake_compact_tensor(cutlass.BFloat16, (m, N), stride_order=(1, 0), assumed_align=16)
    Kt = K // 64
    fw = make_fake_tensor(cutlass.Int32, (4, 128, Nb, Kt), (1, 4, 512, 512 * Nb), assumed_align=16)
    fs = make_fake_tensor(cutlass.Int32, (32, Nb, G), (1, 32, 32 * Nb), assumed_align=16)
    return cute.compile(W4A16Sm90(*cfg), fx, fw, fs, fy, make_fake_stream(use_tvm_ffi_env_stream=True),
                        options="--enable-tvm-ffi")


def matmul(x, p):
    M, K = x.shape
    N = p["N"]
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    cfg = _config(M, N, K)
    key = (cfg, N, K)
    fn = _compiled.get(key)
    if fn is None:
        fn = _compile(cfg, N, K, p["Nb"], p["G"])
        _compiled[key] = fn
    fn(x, p["tw"], p["ts"], y)
    return y
