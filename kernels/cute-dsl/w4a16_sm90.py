"""Hopper W4A16 GEMM in CuTe DSL.

Swapped problem: D[N, M] = W[N, K] . x[M, K]^T, so the large N fills wgmma's M=64 side and the
token count M maps to wgmma's N.  W is the RS-wgmma A operand: packed int4 is TMA'd to smem
in exact A-fragment order (see pack_weight), each thread does one 16B ld.shared per 64-K tile,
dequantises in registers (lop3 magic-number trick + bf16x2 fma for the group scale) and feeds
wgmma straight from registers.  x is TMA'd (128B swizzle) as the K-major B operand.
Warp-specialised: WG0 = TMA producer, WG1..n = consumers.
"""

import math

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Int32, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait


@dsl_user_op
def _asm1(asm: str, a, *, loc=None, ip=None):
    return Int32(llvm.inline_asm(Int32.mlir_type, [Int32(a).ir_value(loc=loc, ip=ip)], asm, "=r,r",
                                 has_side_effects=False, loc=loc, ip=ip))


@dsl_user_op
def _asm2(asm: str, a, b, *, loc=None, ip=None):
    return Int32(llvm.inline_asm(Int32.mlir_type,
                                 [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
                                 asm, "=r,r,r", has_side_effects=False, loc=loc, ip=ip))


@dsl_user_op
def st_dsmem_v4f32(addr, a, b, c, d, *, loc=None, ip=None):
    vals = [Int32(addr).ir_value(loc=loc, ip=ip)] + [cutlass.Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)]
    llvm.inline_asm(None, vals, "st.shared::cluster.v4.f32 [$0], {$1, $2, $3, $4};", "r,f,f,f,f",
                    has_side_effects=True, loc=loc, ip=ip)


def dq_pair(q, s2, j: int):
    """bf16x2 = ((nibble j, nibble j+4) of q  - 8) * s2."""
    asm = (
        "{.reg .b32 t, c1, c2, cz;\n"
        f"shr.u32 t, $1, {4 * j};\n"
        "lop3.b32 t, t, 0x000F000F, 0x43004300, 0xEA;\n"
        "mov.b32 c1, 0x3F803F80;\n"
        "mov.b32 c2, 0xC308C308;\n"
        "fma.rn.bf16x2 t, t, c1, c2;\n"
        "mov.b32 cz, 0x80008000;\n"
        "fma.rn.bf16x2 $0, t, $2, cz;}\n"
    )
    return _asm2(asm, q, s2)


def bcast_lo(s):
    return _asm1("prmt.b32 $0, $1, 0, 0x1010;", s)


def bcast_hi(s):
    return _asm1("prmt.b32 $0, $1, 0, 0x3232;", s)


class W4A16Sm90:
    def __init__(self, num_wg: int, bm: int, bk: int = 64, split: int = 1, stages: int = 0):
        self.num_wg = num_wg
        self.split = split  # split-K across a (1,1,split) cluster, reduced through DSMEM
        self.bn = 64 * num_wg  # tile over N (wgmma M side)
        self.bm = bm  # tile over tokens (wgmma N side)
        self.bk = bk
        self.kc = bk // 64
        self.stages_req = stages
        self.threads = 128 * (num_wg + 1)
        self.acc_dtype = cutlass.Float32

    @cute.jit
    def __call__(self, mX2: cute.Tensor, mW: cute.Tensor, mS: cute.Tensor, mY2: cute.Tensor,
                 stream):
        # mX2 (M, K) bf16 row-major ; mW (4, 128, Nb, Kt64) i32 ; mS (32, Nb, G) i32 ; mY2 (M, N) bf16 row-major
        M = mX2.shape[0]
        K = mX2.shape[1]
        N = mY2.shape[1]
        mX = cute.make_tensor(mX2.iterator, cute.make_layout((M, K, 1), stride=(K, 1, K)))
        mY = cute.make_tensor(mY2.iterator, cute.make_layout((N, M, 1), stride=(1, N, N)))
        self.x_dtype = mX.element_type
        self.y_dtype = mY.element_type
        self.x_layout = utils.LayoutEnum.from_tensor(mX)
        self.y_layout = utils.LayoutEnum.from_tensor(mY)
        tile_mnk = (self.bn, self.bm, self.bk)

        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.warpgroup.OperandMajorMode.K,
            self.x_layout.sm90_mma_major_mode(),
            self.acc_dtype, (self.num_wg, 1, 1), tiler_mn=(64, self.bm),
            a_source=cute.nvgpu.warpgroup.OperandSource.RMEM,
        )
        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            tile_mnk, self.y_dtype, is_cooperative=self.num_wg > 1)
        self.epi_stage = 2

        x_bytes = self.bm * self.bk * 2
        w_bytes = 2048 * self.num_wg * self.kc
        s_bytes = 128 * self.num_wg
        stage_bytes = x_bytes + w_bytes + s_bytes
        epi_bytes = self.epi_tile[0] * self.epi_tile[1] * 2 * self.epi_stage
        self.red_elems = (self.split - 1) * self.bn * self.bm if self.split > 1 else 4
        epi_bytes += self.red_elems * 4
        smem_cap = utils.get_smem_capacity_in_bytes("sm_90")
        st = (smem_cap - epi_bytes - 4096) // (stage_bytes + 3072)
        if cutlass.const_expr(self.stages_req > 0):
            st = min(st, self.stages_req)
        self.stages = st

        x_smem_staged = sm90_utils.make_smem_layout_b(self.x_layout, tile_mnk, self.x_dtype, self.stages)
        w_smem_staged = cute.make_layout((4, 128, self.num_wg, self.kc, self.stages))
        s_smem_staged = cute.make_layout((32, self.num_wg, 1, self.stages))
        epi_smem_staged = sm90_utils.make_smem_layout_epi(self.y_dtype, self.y_layout, self.epi_tile, self.epi_stage)

        G2S = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_x, tX = cute.nvgpu.cpasync.make_tiled_tma_atom(
            G2S, mX, cute.slice_(x_smem_staged, (None, None, 0)), (self.bm, self.bk))
        tma_w, tW = cute.nvgpu.cpasync.make_tiled_tma_atom(
            G2S, mW, cute.slice_(w_smem_staged, (None, None, None, None, 0)), (4, 128, self.num_wg, self.kc))
        tma_s, tS = cute.nvgpu.cpasync.make_tiled_tma_atom(
            G2S, mS, cute.slice_(s_smem_staged, (None, None, None, 0)), (32, self.num_wg, 1))
        tma_y, tY = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), mY,
            cute.slice_(epi_smem_staged, (None, None, 0)), self.epi_tile)

        self.tx_bytes = stage_bytes

        @cute.struct
        class SharedStorage:
            bars: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            sX: cute.struct.Align[cute.struct.MemRange[self.x_dtype, cute.cosize(x_smem_staged)], 1024]
            sW: cute.struct.Align[cute.struct.MemRange[Int32, cute.cosize(w_smem_staged)], 1024]
            sS: cute.struct.Align[cute.struct.MemRange[Int32, cute.cosize(s_smem_staged)], 128]
            sY: cute.struct.Align[cute.struct.MemRange[self.y_dtype, cute.cosize(epi_smem_staged)], 1024]
            sRed: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.red_elems], 128]

        self.shared_storage = SharedStorage

        grid = (cute.ceil_div(cute.size(mY, mode=[0]), self.bn),
                cute.ceil_div(cute.size(mY, mode=[1]), self.bm), self.split)
        self.kernel(tma_x, tX, tma_w, tW, tma_s, tS, tma_y, tY, self.tiled_mma,
                    x_smem_staged, w_smem_staged, s_smem_staged, epi_smem_staged).launch(
            grid=grid, block=[self.threads, 1, 1], cluster=(1, 1, self.split), stream=stream)

    def dequant(self, A, sS, sW, srow, wg, t, idx):
        A32 = cute.recast_tensor(A, Int32)
        s32 = Int32(sS[srow, wg, 0, idx])
        slo = bcast_lo(s32)
        shi = bcast_hi(s32)
        wv = cute.make_rmem_tensor((4, self.kc), Int32)
        for c in range(self.kc):
            src = sW[(None, t, wg, c, idx)]
            for i in range(4):
                wv[i, c] = Int32(src[i])
        for c in range(self.kc):
            for b in range(4):
                q = wv[b, c]
                base = (c * 4 + b) * 4
                A32[base + 0] = dq_pair(q, slo, 0)
                A32[base + 1] = dq_pair(q, shi, 1)
                A32[base + 2] = dq_pair(q, slo, 2)
                A32[base + 3] = dq_pair(q, shi, 3)

    def mma(self, tiled_mma, acc, A, tCrB, idx, num_kb, first=False):
        cute.nvgpu.warpgroup.fence()
        for kb in range(num_kb):
            cute.gemm(tiled_mma, acc, A[(None, None, kb)], tCrB[(None, None, kb, idx)], acc)
            if first:
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        cute.nvgpu.warpgroup.commit_group()

    @cute.kernel
    def kernel(self, tma_x: cute.CopyAtom, mX: cute.Tensor, tma_w: cute.CopyAtom, mW: cute.Tensor,
               tma_s: cute.CopyAtom, mS: cute.Tensor, tma_y: cute.CopyAtom, mY: cute.Tensor,
               tiled_mma: cute.TiledMma, x_smem_staged: cute.ComposedLayout, w_smem_staged: cute.Layout,
               s_smem_staged: cute.Layout, epi_smem_staged: cute.ComposedLayout):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, bidz = cute.arch.block_idx()

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_x)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_w)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_s)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.bars.data_ptr(),
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4 * self.num_wg),
            tx_count=self.tx_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

        sX = storage.sX.get_tensor(x_smem_staged.outer, swizzle=x_smem_staged.inner)
        sW = storage.sW.get_tensor(w_smem_staged)
        sS = storage.sS.get_tensor(s_smem_staged)
        sY = storage.sY.get_tensor(epi_smem_staged.outer, swizzle=epi_smem_staged.inner)

        # global tiles
        gX = cute.local_tile(mX, (self.bm, self.bk), (bidy, None, 0))  # (bm, bk, kt)
        gW = cute.local_tile(mW, (4, 128, self.num_wg, self.kc), (0, 0, bidx, None))  # (4,128,nwg,kc,kt)
        gS = cute.local_tile(mS, (32, self.num_wg, 1), (0, bidx, None))  # (32,nwg,1,G)
        gY = cute.local_tile(mY, (self.bn, self.bm), (bidx, bidy, 0))

        k_tiles = cute.size(gX, mode=[2]) // self.split
        kbeg = bidz * k_tiles
        pipeline_init_wait(cluster_shape_mn=(1, 1))

        if warp_idx < 4:
            cute.arch.warpgroup_reg_dealloc(40)
            if warp_idx == 0:
                tXs, tXg = cute.nvgpu.cpasync.tma_partition(
                    tma_x, 0, cute.make_layout(1), cute.group_modes(sX, 0, 2), cute.group_modes(gX, 0, 2))
                tWs, tWg = cute.nvgpu.cpasync.tma_partition(
                    tma_w, 0, cute.make_layout(1), cute.group_modes(sW, 0, 4), cute.group_modes(gW, 0, 4))
                tSs, tSg = cute.nvgpu.cpasync.tma_partition(
                    tma_s, 0, cute.make_layout(1), cute.group_modes(sS, 0, 3), cute.group_modes(gS, 0, 3))
                pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
                for kt in cutlass.range(kbeg, kbeg + k_tiles, unroll=1):
                    mainloop.producer_acquire(pstate)
                    bar = mainloop.producer_get_barrier(pstate)
                    cute.copy(tma_x, tXg[(None, kt)], tXs[(None, pstate.index)], tma_bar_ptr=bar)
                    cute.copy(tma_w, tWg[(None, kt)], tWs[(None, pstate.index)], tma_bar_ptr=bar)
                    cute.copy(tma_s, tSg[(None, kt * self.bk // 128)], tSs[(None, pstate.index)], tma_bar_ptr=bar)
                    mainloop.producer_commit(pstate)
                    pstate.advance()
                mainloop.producer_tail(pstate)
            if cutlass.const_expr(self.split > 1):
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
        else:
            cute.arch.warpgroup_reg_alloc(232)
            cidx = tidx - 128
            wg = cidx // 128
            t = cidx % 128
            srow = (t // 32) * 8 + (t % 32) // 4

            thr_mma = tiled_mma.get_slice(cidx)
            tCsB = thr_mma.partition_B(sX)
            tCrB = tiled_mma.make_fragment_B(tCsB)
            tCgC = thr_mma.partition_C(gY)
            acc = cute.make_rmem_tensor(tCgC.shape, self.acc_dtype)
            a_shape = thr_mma.partition_shape_A((self.bn, self.bk))
            A0 = cute.make_rmem_tensor(a_shape, cutlass.BFloat16)
            A1 = cute.make_rmem_tensor(a_shape, cutlass.BFloat16)
            num_kb = cute.size(A0, mode=[2])

            rstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            lstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
            # k_tiles (K/64) is always even since K % 128 == 0: unroll by 2 to ping-pong A0/A1.
            mainloop.consumer_wait(rstate)
            self.dequant(A0, sS, sW, srow, wg, t, rstate.index)
            # first tile
            self.mma(tiled_mma, acc, A0, tCrB, rstate.index, num_kb, first=True)
            rstate.advance()
            mainloop.consumer_wait(rstate)
            self.dequant(A1, sS, sW, srow, wg, t, rstate.index)
            self.mma(tiled_mma, acc, A1, tCrB, rstate.index, num_kb)
            for kp in cutlass.range(k_tiles // 2 - 1, unroll=1):
                cute.nvgpu.warpgroup.wait_group(1)
                mainloop.consumer_release(lstate)
                lstate.advance()
                rstate.advance()
                mainloop.consumer_wait(rstate)
                self.dequant(A0, sS, sW, srow, wg, t, rstate.index)
                self.mma(tiled_mma, acc, A0, tCrB, rstate.index, num_kb)
                cute.nvgpu.warpgroup.wait_group(1)
                mainloop.consumer_release(lstate)
                lstate.advance()
                rstate.advance()
                mainloop.consumer_wait(rstate)
                self.dequant(A1, sS, sW, srow, wg, t, rstate.index)
                self.mma(tiled_mma, acc, A1, tCrB, rstate.index, num_kb)
            cute.nvgpu.warpgroup.wait_group(0)
            mainloop.consumer_release(lstate)
            lstate.advance()
            mainloop.consumer_release(lstate)

            nthr = 128 * self.num_wg
            if cutlass.const_expr(self.split > 1):
                nv = cute.size(acc)
                red_layout = cute.make_layout((4, nthr, nv // 4), stride=(1, 4, 4 * nthr))
                sRed = storage.sRed.get_tensor(red_layout)
                # peers push their fp32 partials into rank 0's smem (fire-and-forget remote stores)
                if bidz > 0:
                    rbase = Int32(cute.arch.map_dsmem_ptr(sRed.iterator, 0).toint()) + cidx * 16 + (bidz - 1) * (nv * nthr * 4)
                    for j in cutlass.range_constexpr(nv // 4):
                        st_dsmem_v4f32(rbase + 16 * nthr * j, acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3])
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                if bidz == 0:
                    for peer in cutlass.range_constexpr(self.split - 1):
                        part = cute.make_tensor(sRed.iterator + peer * nv * nthr, red_layout)
                        for j in cutlass.range_constexpr(nv // 4):
                            for i in cutlass.range_constexpr(4):
                                acc[4 * j + i] = acc[4 * j + i] + part[i, cidx, j]
                    self.epilogue(tiled_mma, acc, sY, gY, tma_y, mY, cidx, warp_idx, nthr)
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
            else:
                self.epilogue(tiled_mma, acc, sY, gY, tma_y, mY, cidx, warp_idx, nthr)

    @cute.jit
    def epilogue(self, tiled_mma: cute.TiledMma, acc: cute.Tensor, sY: cute.Tensor, gY: cute.Tensor,
                 tma_y: cute.CopyAtom, mY: cute.Tensor, cidx: Int32, warp_idx: Int32, nthr: cutlass.Constexpr):
        # ---------------- epilogue: acc -> bf16 -> smem (stmatrix) -> TMA store
        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            self.y_layout, elem_ty_d=self.y_dtype, elem_ty_acc=self.acc_dtype)
        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(self.y_layout.is_m_major_c(), 4), self.y_dtype)
        tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_atom)
        thr_copy_r2s = tiled_copy_r2s.get_slice(cidx)
        tRS_sD = thr_copy_r2s.partition_D(sY)
        tRS_rAcc = tiled_copy_r2s.retile(acc)
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sY))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor_like(tRS_rD_layout, self.acc_dtype)
        size_tRS_rD = cute.size(tRS_rD)

        gY_epi = cute.zipped_divide(gY, self.epi_tile)
        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
            tma_y, 0, cute.make_layout(1), cute.group_modes(sY, 0, 2), gY_epi)
        epi_tile_num = cute.size(gY_epi, mode=[1])
        epi_tile_shape = gY_epi.shape[1]
        epi_tile_layout = cute.make_layout(epi_tile_shape, stride=(epi_tile_shape[1], 1))
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, nthr))
        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            for v in cutlass.range_constexpr(size_tRS_rD):
                tRS_rD[v] = tRS_rAcc[epi_idx * size_tRS_rD + v]
            tRS_rD_out = cute.make_rmem_tensor_like(tRS_rD_layout, self.y_dtype)
            tRS_rD_out.store(tRS_rD.load().to(self.y_dtype))
            epi_buffer = epi_idx % self.epi_stage
            cute.copy(tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)])
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier(barrier_id=1, number_of_threads=nthr)
            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            if warp_idx == 4:
                cute.copy(tma_y, bSG_sD[(None, epi_buffer)], bSG_gD[(None, gmem_coord)])
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
            cute.arch.barrier(barrier_id=1, number_of_threads=nthr)
        if warp_idx == 4:
            c_pipeline.producer_tail()
