"""W4A16 GEMM in Triton (agent: triton).

y[M, N] = x[M, K] @ W[N, K].T computed as yT = W @ xT ("swap-AB"): the dequantised
W tile is the MMA A operand (lives in registers), x is the B operand.

prepare() layout:
  q  int32 [N, K // 8]: word j of row n holds k = 8j .. 8j+7, nibbles interleaved as
     [k0, k2, k4, k6, k1, k3, k5, k7] (low nibble first) so that
     (word >> 4i) & 0x000F000F gives the bf16x2 pair (k_{2i}, k_{2i+1}).
  s  bf16  [N, K // 128]
"""

import os

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

META = {"name": "triton", "agent": "triton", "dsl": "triton", "format": "int4_g128"}


_ASM_UNPACK = """
{
.reg .b32 t<4>, u<3>, m;
mov.b32 m, 0x43084308;
shr.u32 u0, $8, 4;
shr.u32 u1, $8, 8;
shr.u32 u2, $8, 12;
lop3.b32 t0, $8, 0x000F000F, 0x43004300, 0xEA;
lop3.b32 t1, u0, 0x000F000F, 0x43004300, 0xEA;
lop3.b32 t2, u1, 0x000F000F, 0x43004300, 0xEA;
lop3.b32 t3, u2, 0x000F000F, 0x43004300, 0xEA;
sub.bf16x2 t0, t0, m;
sub.bf16x2 t1, t1, m;
sub.bf16x2 t2, t2, m;
sub.bf16x2 t3, t3, m;
%SCALE%
mov.b32 {$0, $1}, t0;
mov.b32 {$2, $3}, t1;
mov.b32 {$4, $5}, t2;
mov.b32 {$6, $7}, t3;
}
"""

_SCALE = """
.reg .b32 s2;
mov.b32 s2, {$9, $9};
mul.rn.bf16x2 t0, t0, s2;
mul.rn.bf16x2 t1, t1, s2;
mul.rn.bf16x2 t2, t2, s2;
mul.rn.bf16x2 t3, t3, s2;
"""

_ASM_PLAIN = tl.constexpr(_ASM_UNPACK.replace("%SCALE%", ""))
_ASM_SCALED = tl.constexpr(_ASM_UNPACK.replace("%SCALE%", _SCALE))


@triton.jit
def _assemble(o, BN: tl.constexpr, BK: tl.constexpr):
    # o[2g+e][n, 4c+t] holds k = 32c + 8g + 2t + e: the wgmma A-fragment order
    # (thread t of a quad owns k pairs 2t, 2t+8, 2t+16, 2t+24 of every 32).
    p0 = tl.join(o[0], o[1])
    p1 = tl.join(o[2], o[3])
    p2 = tl.join(o[4], o[5])
    p3 = tl.join(o[6], o[7])
    w = tl.join(tl.join(p0, p2), tl.join(p1, p3))  # [BN, BK/8, e, ., .]
    w = tl.reshape(w, (BN, BK // 32, 4, 2, 2, 2))
    w = tl.permute(w, (0, 1, 4, 5, 2, 3))
    return tl.reshape(w, (BN, BK))


@triton.jit
def _dequant(q, BN: tl.constexpr, BK: tl.constexpr):
    o = tl.inline_asm_elementwise(
        _ASM_PLAIN, "=h,=h,=h,=h,=h,=h,=h,=h,r", [q],
        dtype=(tl.bfloat16,) * 8, is_pure=True, pack=1)
    return _assemble(o, BN, BK)


@triton.jit
def _dequant_scaled(q, s, BN: tl.constexpr, BK: tl.constexpr):
    o = tl.inline_asm_elementwise(
        _ASM_SCALED, "=h,=h,=h,=h,=h,=h,=h,=h,r,h", [q, s],
        dtype=(tl.bfloat16,) * 8, is_pure=True, pack=1)
    return _assemble(o, BN, BK)


@triton.jit(do_not_specialize=["M"])
def _w4a16_kernel(
    x_ptr, q_ptr, s_ptr, y_ptr, ws_ptr, cnt_ptr, x_desc, q_desc,
    M, N, K,
    GROUP: tl.constexpr, BN: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
    SPLIT_K: tl.constexpr, GROUP_M: tl.constexpr, SCALE_AFTER: tl.constexpr,
    TMA: tl.constexpr, NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_m = tl.cdiv(M, BM)
    # grouped ordering: walk N within a group of GROUP_M m-tiles
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_m = group_id * GROUP_M
    gsz = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % num_pid_in_group) % gsz
    pid_n = (pid % num_pid_in_group) // gsz

    rn = pid_n * BN + tl.arange(0, BN)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = rn % N
    rm_c = tl.where(rm < M, rm, 0)

    n_iters = K // BK
    it0 = (pid_k * n_iters) // SPLIT_K
    it1 = ((pid_k + 1) * n_iters) // SPLIT_K
    rkw = tl.arange(0, BK // 8)
    rk = tl.arange(0, BK)

    acc = tl.zeros((BN, BM), dtype=tl.float32)
    q_off = rn[:, None] * 16 + (rkw // 16)[None, :] * (N * 16) + (rkw % 16)[None, :]
    for it in tl.range(it0, it1, num_stages=NUM_STAGES):
        k0 = it * BK
        if TMA:
            q = q_desc.load([k0 // 128, pid_n * BN, (k0 % 128) // 8]).reshape(BN, BK // 8)
            xt = tl.trans(x_desc.load([pid_m * BM, k0]))
        else:
            q = tl.load(q_ptr + (k0 // 128) * (N * 16) + (k0 % 128) // 8 + q_off)  # [BN, BK/8] int32
            xt = tl.load(x_ptr + rm_c[None, :] * K + (k0 + rk)[:, None], mask=(rm < M)[None, :], other=0.0)  # [BK, BM]
        if SCALE_AFTER:
            wb = _dequant(q, BN, BK)
            s = tl.load(s_ptr + rn * (K // GROUP) + k0 // GROUP).to(tl.float32)  # [BN]
            acc += tl.dot(wb, xt) * s[:, None]
        else:
            if BK <= GROUP:
                s = tl.load(s_ptr + rn * (K // GROUP) + k0 // GROUP)  # [BN]
                wb = _dequant_scaled(q, tl.broadcast_to(s[:, None], (BN, BK // 8)), BN, BK)
            else:
                s = tl.load(s_ptr + rn[:, None] * (K // GROUP) + k0 // GROUP + (rkw // (GROUP // 8))[None, :])
                wb = _dequant_scaled(q, s, BN, BK)
            acc = tl.dot(wb, xt, acc)

    rn2 = pid_n * BN + tl.arange(0, BN)
    y_ptrs = y_ptr + rm[None, :] * N + rn2[:, None]
    mask = (rm[None, :] < M) & (rn2[:, None] < N)
    if SPLIT_K == 1:
        tl.store(y_ptrs, acc.to(tl.bfloat16), mask=mask)
    else:
        # Serial-free split-K: every split writes an fp32 partial, the last one to
        # arrive (per output tile) reduces and writes bf16, then resets the counter.
        tile = pid
        loc = tl.arange(0, BN)[:, None] * BM + tl.arange(0, BM)[None, :]
        ws_tile = ws_ptr + tile * (SPLIT_K * BN * BM)
        tl.store(ws_tile + pid_k * (BN * BM) + loc, acc, cache_modifier=".cg")
        prev = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel", scope="gpu")
        if prev == SPLIT_K - 1:
            tot = tl.zeros((BN, BM), dtype=tl.float32)
            for s_i in range(SPLIT_K):
                tot += tl.load(ws_tile + s_i * (BN * BM) + loc, cache_modifier=".cg")
            tl.store(y_ptrs, tot.to(tl.bfloat16), mask=mask)
            tl.atomic_xchg(cnt_ptr + tile, 0, sem="relaxed", scope="gpu")


@triton.jit(do_not_specialize=["M"])
def _w4a16_persistent_kernel(
    x_desc, q_desc, y_desc, s_ptr, M, N, K,
    GROUP: tl.constexpr, BN: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_STAGES: tl.constexpr, NUM_PROGS: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    # persistent data-parallel tiles; flatten=True lets the pipeliner prefetch the next
    # tile's k-blocks while the current tile's epilogue runs
    num_pid_n = tl.cdiv(N, BN)
    num_pid_m = tl.cdiv(M, BM)
    num_tiles = num_pid_n * num_pid_m
    k_tiles = K // BK
    num_pid_in_group = GROUP_M * num_pid_n
    for tile in tl.range(tl.program_id(0), num_tiles, NUM_PROGS, flatten=FLATTEN, num_stages=NUM_STAGES):
        group_id = tile // num_pid_in_group
        first_m = group_id * GROUP_M
        gsz = min(num_pid_m - first_m, GROUP_M)
        pid_m = first_m + (tile % num_pid_in_group) % gsz
        pid_n = (tile % num_pid_in_group) // gsz
        n0 = pid_n * BN
        m0 = pid_m * BM
        rn = (n0 + tl.arange(0, BN)) % N
        acc = tl.zeros((BN, BM), dtype=tl.float32)
        for it in range(k_tiles):
            k0 = it * BK
            q = q_desc.load([k0 // 128, n0, (k0 % 128) // 8]).reshape(BN, BK // 8)
            xt = tl.trans(x_desc.load([m0, k0]))
            s = tl.load(s_ptr + rn * (K // GROUP) + k0 // GROUP)
            wb = _dequant_scaled(q, tl.broadcast_to(s[:, None], (BN, BK // 8)), BN, BK)
            acc = tl.dot(wb, xt, acc)
        y_desc.store([m0, n0], tl.trans(acc).to(tl.bfloat16))


def _interleave(codes: torch.Tensor) -> torch.Tensor:
    """codes [N, K] -> int32 [N, K/8]; word 4c+t of each 128-k block holds
    k = 32c + 8g + 2t + e at nibble g + 4e."""
    N, K = codes.shape
    c = codes.to(torch.int32).view(N, K // 32, 4, 4, 2)  # [n, c, g, t, e]
    out = torch.zeros(N, K // 32, 4, dtype=torch.int32, device=codes.device)  # [n, c, t]
    for g in range(4):
        for e in range(2):
            out |= c[:, :, g, :, e] << (4 * (g + 4 * e))
    return out.view(N, K // 8)


def prepare(qw):
    assert qw.fmt == "int4_g128" and qw.group == 128
    q = _interleave(qw.codes)  # [N, K/8]
    q = q.view(qw.N, qw.K // 128, 16).permute(1, 0, 2).contiguous()  # [K/128, N, 16]
    return {"q": q, "s": qw.scales.contiguous(), "N": qw.N, "K": qw.K}


def _config(M, N, K):
    env = os.environ.get("TRITON_W4_CFG")
    if env:
        v = list(map(int, env.split(",")))
        return tuple(v) if len(v) == 7 else tuple(v) + (int(M <= 64),)
    cd = lambda a, b: (a + b - 1) // b  # noqa: E731
    if M <= 16:
        return 64, 16, 128, -264, 4, 4, 1
    if M <= 256:
        if M >= 128 and cd(N, 64) * cd(M, 128) >= 132:
            return 64, 128, 64, 1, 4, 4, 0
        if cd(N, 64) * cd(M, 32) <= 640:
            return 64, 32, 128, -264, 4, 4, 1
        return 128, 64, 128, -264, 8, 4, 0
    # prefill: persistent kernel (flatten=True), last field = programs per SM
    if cd(N, 64) * cd(M, 128) <= 132:
        return 64, 128, 128, 1, 4, 4, 0, 2
    t128 = cd(N, 128) * cd(M, 128)
    if t128 < 150:
        return 64, 128, 64, 1, 4, 4, 0
    # wave-quantisation model: a 128x256 tile costs ~1.76x a 128x128 tile
    waves = lambda t: cd(t, 132)  # noqa: E731
    if waves(cd(N, 128) * cd(M, 256)) * 1.76 < waves(t128):
        return 128, 256, 64, 1, 8, 3, 0, 1
    return 128, 128, 128, 1, 8, 4, 0, 1


_bufs = {}


def _buf(name, numel, dtype, device):
    b = _bufs.get(name)
    if b is None or b.numel() < numel or b.device != device:
        b = torch.zeros(max(numel, 1 << 16), dtype=dtype, device=device)
        _bufs[name] = b
    return b


_sms = {}


def _num_sms(device):
    if device not in _sms:
        _sms[device] = torch.cuda.get_device_properties(device).multi_processor_count
    return _sms[device]


def matmul(x, p):
    M, K = x.shape
    N = p["N"]
    bn, bm, bk, sk, nw, ns, sa, *rest = _config(M, N, K)
    tiles = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    if sk < 0:  # auto: aim for ~|sk| CTAs in total
        sk = max(1, min(K // bk, round(-sk / tiles)))
    sa = sa and bk <= 128
    while K % bk:
        bk //= 2
    sk = min(sk, K // bk)
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    pers = int(os.environ.get("TRITON_W4_PERSIST", rest[0] if rest else 0))
    if pers and sk == 1 and bk <= 128:
        nprog = min(tiles, _num_sms(x.device) * pers)
        xd = TensorDescriptor.from_tensor(x, [bm, bk])
        qd = TensorDescriptor.from_tensor(p["q"], [1, bn, bk // 8])
        yd = TensorDescriptor.from_tensor(y, [bm, bn])
        _w4a16_persistent_kernel[(nprog,)](
            xd, qd, yd, p["s"], M, N, K,
            GROUP=128, BN=bn, BM=bm, BK=bk, GROUP_M=8, NUM_STAGES=ns, NUM_PROGS=nprog,
            FLATTEN=bool(int(os.environ.get("TRITON_W4_FLAT", "1"))),
            num_warps=nw, num_stages=ns,
        )
        return y
    if sk > 1:
        ws = _buf("ws", tiles * sk * bn * bm, torch.float32, x.device)
        cnt = _buf("cnt", tiles, torch.int32, x.device)
    else:
        ws = cnt = y
    tma = int(os.environ.get("TRITON_W4_TMA", "1"))
    if tma:
        xd = TensorDescriptor.from_tensor(x, [bm, bk])
        qd = TensorDescriptor.from_tensor(p["q"], [1, bn, bk // 8])
    else:
        xd = qd = None
    _w4a16_kernel[(tiles, sk)](
        x, p["q"], p["s"], y, ws, cnt, xd, qd, M, N, K,
        GROUP=128, BN=bn, BM=bm, BK=bk, SPLIT_K=sk, GROUP_M=8, SCALE_AFTER=sa, TMA=bool(tma), NUM_STAGES=ns,
        num_warps=nw, num_stages=ns,
    )
    return y
