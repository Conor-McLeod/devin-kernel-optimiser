"""Hopper W4A16 GEMM (CUDA C++ / inline PTX): TMA + mbarrier pipeline, warp-specialised,
int4 weights dequantised in registers and fed to wgmma as the A operand (swap-AB).

Weight layout from prepare():
  wq     uint8 [Np/128, K/64, 4096]: per (128-row, 64-K) tile, 256 threads x 16 B. Each 16 B is
         the four m64k16 A fragments (k16 steps) a thread needs, nibbles ordered for lop3 dequant.
  scales bf16  [Np/128, K/128, 128]
"""

import os
import tempfile

import torch

META = {"name": "cuda-wgmma", "agent": "cuda-wgmma", "dsl": "cuda (wgmma+tma)", "format": "int4_g128"}

_HERE = os.path.dirname(os.path.abspath(__file__))
_ext = None


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load

        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
        build = os.path.join(tempfile.gettempdir(), f"cuda_wgmma_build_{os.getuid()}")
        os.makedirs(build, exist_ok=True)
        _ext = load(
            name="cuda_wgmma_w4a16",
            sources=[os.path.join(_HERE, "w4a16.cu")],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-std=c++20", "-lineinfo",
                               f"-I{_HERE}"],
            extra_ldflags=["-lcuda"],
            build_directory=build,
            verbose=False,
        )
    return _ext


_counters = {}


def _counter_buf(dev):
    buf = _counters.get(dev)
    if buf is None:
        buf = torch.zeros(1 << 16, dtype=torch.int32, device=dev)
        _counters[dev] = buf
    return buf


def prepare(qw):
    assert qw.fmt == "int4_g128" and qw.group == 128
    _load()
    N, K = qw.N, qw.K
    Np = (N + 127) // 128 * 128
    codes = qw.codes
    scales = qw.scales
    if Np != N:
        codes = torch.cat([codes, torch.full((Np - N, K), 8, dtype=codes.dtype, device=codes.device)])
        scales = torch.cat([scales, torch.zeros(Np - N, K // 128, dtype=scales.dtype, device=scales.device)])
    NT, KB = Np // 128, K // 64
    # row n = T*128 + wg*64 + w*16 + h*8 + g ; col k = kb*64 + j*16 + c*8 + t*2 + e
    c = codes.to(torch.int64).view(NT, 2, 4, 2, 8, KB, 4, 2, 4, 2)
    #            dims:          T  wg w  h  g  kb  j  c  t  e
    c = c.permute(0, 5, 1, 2, 4, 8, 6, 9, 7, 3)  # T kb wg w g t j e c h ; nibble = e*4 + c*2 + h
    c = c.reshape(NT, KB, 2, 4, 8, 4, 4, 8)
    shifts = torch.arange(8, device=codes.device, dtype=torch.int64) * 4
    words = (c << shifts).sum(-1)  # [..., 4] int64 in [0, 2^32)
    words = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
    wq = words.contiguous().view(torch.uint8).view(NT, KB, 4096)
    s = scales.view(NT, 128, K // 128).permute(0, 2, 1).contiguous()
    return {"wq": wq, "s": s, "N": N, "K": K}


def _config(M, N, K):
    nt = (N + 127) // 128
    if M <= 16:
        bm = 16
    elif M <= 32:
        bm = 32
    elif M <= 64:
        bm = 64
    elif M <= 512:
        bm = 128
    else:
        bm = 256
    tiles = nt * ((M + bm - 1) // bm)
    kb = K // 64
    split = 1
    while tiles * split * 2 <= 132 and kb // (split * 2) >= 4:
        split *= 2
    return bm, split


def matmul(x, p):
    M, K = x.shape
    bm, split = _config(M, p["N"], K)
    return _ext.w4a16_mm(x, p["wq"], p["s"], _counter_buf(x.device), p["N"], bm, split)
