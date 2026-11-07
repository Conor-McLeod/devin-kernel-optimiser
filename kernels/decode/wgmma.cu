// Hopper W4A16 GEMM: y[M,N] = x[M,K] @ W[N,K]^T, W int4 (uint4b8) with bf16 scale per 128 K.
// Swap-AB: W (dequantised in registers) is the wgmma A operand, x (TMA, 128B swizzle) is B.
// Mainloop ported from agent/cuda-wgmma (commit 5701156, credit: cuda-wgmma agent).
// decode change: 'tail split' schedule. The first D tiles are data-parallel (full K); the remaining
// tiles are split-K S ways, all in one launch, so the split pieces fill the last partial wave.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "wgmma_ops.cuh"

#define DEVI __device__ __forceinline__

namespace {

constexpr int BN = 128;   // weight rows per CTA (2 consumer warpgroups x 64)
constexpr int BK = 64;    // K per pipeline stage
constexpr int W_BYTES = BN * BK / 2;
constexpr int S_BYTES = BN * 2;

DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

DEVI void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
DEVI void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
DEVI void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
DEVI void mbar_wait(uint64_t* bar, uint32_t phase) {
  asm volatile(
      "{\n.reg .pred P1;\nLAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra DONE;\nbra LAB_WAIT;\nDONE:\n}" ::"r"(smem_u32(bar)),
      "r"(phase)
      : "memory");
}
DEVI void tma_load_2d(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(
          smem_u32(dst)),
      "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1)
      : "memory");
}
DEVI void bulk_load(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   smem_u32(dst)),
               "l"(src), "r"(bytes), "r"(smem_u32(bar))
               : "memory");
}
DEVI void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
DEVI void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
DEVI void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory"); }
DEVI void fence_reg(float& r) { asm volatile("" : "+f"(r)::"memory"); }
DEVI void fence_reg(uint32_t& r) { asm volatile("" : "+r"(r)::"memory"); }
DEVI void consumer_sync() { asm volatile("bar.sync 1, 256;" ::: "memory"); }

DEVI uint4 lds128(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
  return v;
}
DEVI uint32_t lds_bf16x2_bcast(uint32_t addr) {
  uint16_t h;
  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(h) : "r"(addr));
  return static_cast<uint32_t>(h) * 0x10001u;
}
DEVI uint64_t make_desc_sw128(uint32_t saddr) {
  uint64_t d = 0;
  d |= static_cast<uint64_t>((saddr & 0x3FFFF) >> 4);
  d |= static_cast<uint64_t>(1) << 16;
  d |= static_cast<uint64_t>(1024 >> 4) << 32;
  d |= static_cast<uint64_t>(1) << 62;
  return d;
}

DEVI uint32_t lop3_and_or(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
DEVI uint32_t bf2_sub_mul(uint32_t t, uint32_t off, uint32_t s) {
  uint32_t r;
  asm("sub.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(t), "r"(off));
  asm("mul.rn.bf16x2 %0, %0, %1;" : "+r"(r) : "r"(s));
  return r;
}
// q: 8 nibbles packed by prepare() so the outputs are exactly the m64k16 A fragment regs.
DEVI void dequant8(uint32_t q, uint32_t s0, uint32_t s1, uint32_t* a) {
  constexpr uint32_t MASK = 0x000f000f, EX = 0x43004300, OFF = 0x43084308;  // 128+c ; 136
  a[0] = bf2_sub_mul(lop3_and_or(q, MASK, EX), OFF, s0);
  a[1] = bf2_sub_mul(lop3_and_or(q >> 4, MASK, EX), OFF, s1);
  a[2] = bf2_sub_mul(lop3_and_or(q >> 8, MASK, EX), OFF, s0);
  a[3] = bf2_sub_mul(lop3_and_or(q >> 12, MASK, EX), OFF, s1);
}

template <int BM>
struct Cfg {
  static constexpr int X_BYTES = BM * BK * 2;
  static constexpr int STAGE_BYTES = ((X_BYTES + W_BYTES + S_BYTES + 1023) / 1024) * 1024;
  static constexpr int EPI_BYTES = 2 * BM * 128;  // per consumer WG: [BM][64] bf16, 128B-swizzled for TMA store
  static constexpr int OCC = BM <= 64 ? 2 : 1;  // small tiles: 2 CTAs/SM hide each other's latency
  static constexpr int STAGES_RAW = ((OCC == 2 ? 112 : 225) * 1024 - EPI_BYTES) / STAGE_BYTES;
  static constexpr int STAGES_CAP = OCC == 2 ? 4 : 8;
  static constexpr int STAGES = STAGES_RAW > STAGES_CAP ? STAGES_CAP : STAGES_RAW;
  static constexpr int REG_P = OCC == 2 ? 24 : 40;
  static constexpr int REG_C = OCC == 2 ? 104 : 232;  // 128*24 + 256*104 <= 384*80
  static constexpr int PIPE_BYTES = STAGES * STAGE_BYTES;
  static constexpr int BAR_OFF = PIPE_BYTES + EPI_BYTES;
  static constexpr int SMEM = 1024 + BAR_OFF + 2 * STAGES * 8 + 16;
  static constexpr int NACC = BM / 2;
};

// Persistent: CTA b handles work items b, b+gridDim.x, ... Items [0,D) are full-K tiles, items
// [D, D+R*S) are split-K pieces of the remaining R tiles. The TMA producer runs ahead across items;
// the epilogue has its own smem so the next item's loads overlap it.
template <int BM>
__global__ void __launch_bounds__(384, Cfg<BM>::OCC)
    w4a16_kernel(const __grid_constant__ CUtensorMap tmx, const __grid_constant__ CUtensorMap tmy, const uint8_t* __restrict__ wq,
                 const __nv_bfloat16* __restrict__ ws, __nv_bfloat16* __restrict__ y, float* __restrict__ part,
                 int* __restrict__ counters, int M, int N, int K, int kb_per_split, int ntn, int D, int S,
                 int nitems) {
  using C = Cfg<BM>;
  constexpr int STAGES = C::STAGES;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  __nv_bfloat16* epi = reinterpret_cast<__nv_bfloat16*>(smem + C::PIPE_BYTES);
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + C::BAR_OFF);
  uint64_t* empty = full + STAGES;
  int* flag = reinterpret_cast<int*>(empty + STAGES);

  const int KB = K / BK, G = K / 128;
  const int wg = threadIdx.x / 128, tid = threadIdx.x % 128;
  auto item = [&](int w, int& tile, int& split, int& nsplit, int& kb0, int& kb1) {
    if (w < D) {
      tile = w; split = 0; nsplit = 1; kb0 = 0; kb1 = KB;
    } else {
      tile = D + (w - D) / S; split = (w - D) % S; nsplit = S;
      kb0 = split * kb_per_split; kb1 = min(KB, kb0 + kb_per_split);
    }
  };

  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], 8);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (wg == 0) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(C::REG_P));
    if (tid == 0) {
      int it = 0;
      for (int w = blockIdx.x; w < nitems; w += gridDim.x) {
        int tile, split, nsplit, kb0, kb1;
        item(w, tile, split, nsplit, kb0, kb1);
        const int tile_n = tile % ntn, tile_m = tile / ntn;
        const uint8_t* wsrc = wq + (static_cast<size_t>(tile_n) * KB + kb0) * W_BYTES;
        const __nv_bfloat16* ssrc = ws + static_cast<size_t>(tile_n) * G * BN;
        for (int i = 0; i < kb1 - kb0; ++i, ++it) {
          const int s = it % STAGES;
          if (it >= STAGES) mbar_wait(&empty[s], ((it / STAGES) - 1) & 1);
          uint8_t* st = smem + s * C::STAGE_BYTES;
          mbar_expect_tx(&full[s], C::X_BYTES + W_BYTES + S_BYTES);
          tma_load_2d(st, &tmx, &full[s], (kb0 + i) * BK, tile_m * BM);
          bulk_load(st + C::X_BYTES, wsrc + static_cast<size_t>(i) * W_BYTES, W_BYTES, &full[s]);
          bulk_load(st + C::X_BYTES + W_BYTES, ssrc + ((kb0 + i) >> 1) * BN, S_BYTES, &full[s]);
        }
      }
    }
    return;
  }

  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(C::REG_C));
  const int c = wg - 1, warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
  const int row = c * 64 + warp * 16 + g;
  float acc[C::NACC];
  int it = 0;
  for (int w = blockIdx.x; w < nitems; w += gridDim.x) {
    int tile, split, nsplit, kb0, kb1;
    item(w, tile, split, nsplit, kb0, kb1);
    const int tile_n = tile % ntn, tile_m = tile / ntn;
    const int nk = kb1 - kb0;
#pragma unroll
    for (int r = 0; r < C::NACC; ++r) acc[r] = 0.f;
    const int it0 = it;
    // Stage i: wait for TMA, dequantise into a, issue 4 wgmma, keep one wgmma group in flight.
    auto step = [&](int i, uint32_t (&a)[4][4]) {
      const int s = i % STAGES;
      mbar_wait(&full[s], (i / STAGES) & 1);
      const uint8_t* st = smem + s * C::STAGE_BYTES;
      const uint32_t sts = smem_u32(st);
      const uint4 q = lds128(sts + C::X_BYTES + (c * 128 + tid) * 16);
      const uint32_t s0 = lds_bf16x2_bcast(sts + C::X_BYTES + W_BYTES + row * 2);
      const uint32_t s1 = lds_bf16x2_bcast(sts + C::X_BYTES + W_BYTES + (row + 8) * 2);
      dequant8(q.x, s0, s1, a[0]);
      dequant8(q.y, s0, s1, a[1]);
      dequant8(q.z, s0, s1, a[2]);
      dequant8(q.w, s0, s1, a[3]);
      const uint64_t desc = make_desc_sw128(smem_u32(st));
#pragma unroll
      for (int r = 0; r < C::NACC; ++r) fence_reg(acc[r]);
      wg_fence();
#pragma unroll
      for (int j = 0; j < 4; ++j) wgmma_rs<BM>(acc, a[j], desc + 2 * j, 1);
      wg_commit();
#pragma unroll
      for (int r = 0; r < C::NACC; ++r) fence_reg(acc[r]);
      wg_wait<1>();
      if (i > it0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[(i - 1) % STAGES]);
      }
    };
    uint32_t a0[4][4], a1[4][4];
    const int it1 = it0 + nk;
    for (; it + 1 < it1; it += 2) {
      step(it, a0);
      step(it + 1, a1);
    }
    if (it < it1) { step(it, a0); ++it; }
    wg_wait<0>();
#pragma unroll
    for (int r = 0; r < C::NACC; ++r) fence_reg(acc[r]);
    __syncwarp();
    if (lane == 0) mbar_arrive(&empty[(it1 - 1) % STAGES]);

    if (nsplit > 1) {
      // Split-K: publish fp32 partial; the last CTA of this tile reduces and stores.
      const int tile_lin = tile - D;
      const int ctid = c * 128 + tid;
      float4* mine = reinterpret_cast<float4*>(part + (static_cast<size_t>(tile_lin) * nsplit + split) * BM * BN);
#pragma unroll
      for (int r = 0; r < C::NACC / 4; ++r)
        mine[r * 256 + ctid] = make_float4(acc[4 * r], acc[4 * r + 1], acc[4 * r + 2], acc[4 * r + 3]);
      __threadfence();
      consumer_sync();
      if (ctid == 0) {
        const int old = atomicAdd(&counters[tile_lin], 1);
        const int last = (old == nsplit - 1);
        if (last) counters[tile_lin] = 0;
        *flag = last;
      }
      consumer_sync();
      const int last = *flag;
      consumer_sync();
      if (!last) continue;
      __threadfence();
      for (int sp = 0; sp < nsplit; ++sp) {
        if (sp == split) continue;
        const float4* other =
            reinterpret_cast<const float4*>(part + (static_cast<size_t>(tile_lin) * nsplit + sp) * BM * BN);
#pragma unroll
        for (int r = 0; r < C::NACC / 4; ++r) {
          const float4 v = __ldcg(&other[r * 256 + ctid]);
          acc[4 * r] += v.x;
          acc[4 * r + 1] += v.y;
          acc[4 * r + 2] += v.z;
          acc[4 * r + 3] += v.w;
        }
      }
    }

    // Epilogue (per warpgroup): stmatrix.trans into a swizzled [m][64n] tile, then one TMA store.
    uint8_t* epi_c = reinterpret_cast<uint8_t*>(epi) + c * BM * 128;
    if (tid == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
    asm volatile("bar.sync %0, 128;" ::"r"(2 + c) : "memory");
    {
      const int jj = lane >> 3, r = lane & 7;
      const int nch = warp * 2 + (jj & 1);
#pragma unroll
      for (int i = 0; i < BM / 8; i += 2) {
        const int m = 8 * (i + (jj >> 1)) + r;
        const uint32_t addr = smem_u32(epi_c + m * 128 + ((nch ^ (m & 7)) << 4));
        __nv_bfloat162 p0 = __floats2bfloat162_rn(acc[4 * i], acc[4 * i + 1]);
        __nv_bfloat162 p1 = __floats2bfloat162_rn(acc[4 * i + 2], acc[4 * i + 3]);
        __nv_bfloat162 p2 = __floats2bfloat162_rn(acc[4 * i + 4], acc[4 * i + 5]);
        __nv_bfloat162 p3 = __floats2bfloat162_rn(acc[4 * i + 6], acc[4 * i + 7]);
        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr),
                     "r"(*reinterpret_cast<uint32_t*>(&p0)), "r"(*reinterpret_cast<uint32_t*>(&p1)),
                     "r"(*reinterpret_cast<uint32_t*>(&p2)), "r"(*reinterpret_cast<uint32_t*>(&p3))
                     : "memory");
      }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("bar.sync %0, 128;" ::"r"(2 + c) : "memory");
    const int ny = tile_n * BN + c * 64;
    if (tid == 0 && ny < N) {
      asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];" ::"l"(&tmy),
                   "r"(smem_u32(epi_c)), "r"(ny), "r"(tile_m * BM)
                   : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    }
  }
  if (tid == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

CUtensorMap make_y_map(void* ptr, int M, int N, int box_m) {
  CUtensorMap map;
  cuuint64_t dims[2] = {static_cast<cuuint64_t>(N), static_cast<cuuint64_t>(M)};
  cuuint64_t strides[1] = {static_cast<cuuint64_t>(N) * 2};
  cuuint32_t box[2] = {64, static_cast<cuuint32_t>(box_m)};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box, estr,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(y) failed: ", static_cast<int>(r));
  return map;
}

CUtensorMap make_x_map(const void* ptr, int M, int K, int box_m) {
  CUtensorMap map;
  cuuint64_t dims[2] = {static_cast<cuuint64_t>(K), static_cast<cuuint64_t>(M)};
  cuuint64_t strides[1] = {static_cast<cuuint64_t>(K) * 2};
  cuuint32_t box[2] = {BK, static_cast<cuuint32_t>(box_m)};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides,
                                      box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return map;
}

template <int BM>
void launch(const torch::Tensor& x, const torch::Tensor& wq, const torch::Tensor& ws, torch::Tensor& y,
            torch::Tensor& counters, torch::Tensor& part, int N, int D, int S, int ctas, cudaStream_t stream) {
  using C = Cfg<BM>;
  const int M = x.size(0), K = x.size(1);
  const int KB = K / BK;
  const int ntn = (N + BN - 1) / BN, mtiles = (M + BM - 1) / BM;
  const int T = ntn * mtiles;
  S = std::max(1, std::min(S, KB));
  const int kbps = (KB + S - 1) / S;
  S = (KB + kbps - 1) / kbps;
  D = std::max(0, std::min(D, T));
  if (S == 1) D = T;
  const int R = T - D;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(w4a16_kernel<BM>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    attr_set = true;
  }
  float* part_ptr = nullptr;
  if (R > 0) {
    TORCH_CHECK(counters.numel() >= R, "counter buffer too small");
    TORCH_CHECK(part.numel() >= static_cast<int64_t>(R) * S * BM * BN, "partial buffer too small");
    part_ptr = part.data_ptr<float>();
  }
  // Tensor maps only encode address/shape; cache the last one per BM (allocator reuses buffers).
  static const void* xk[3] = {nullptr, nullptr, nullptr};
  static const void* yk[3] = {nullptr, nullptr, nullptr};
  static CUtensorMap map, ymap;
  if (xk[0] != x.data_ptr() || xk[1] != reinterpret_cast<void*>(static_cast<intptr_t>(M)) ||
      xk[2] != reinterpret_cast<void*>(static_cast<intptr_t>(K))) {
    map = make_x_map(x.data_ptr(), M, K, BM);
    xk[0] = x.data_ptr(); xk[1] = reinterpret_cast<void*>(static_cast<intptr_t>(M));
    xk[2] = reinterpret_cast<void*>(static_cast<intptr_t>(K));
  }
  if (yk[0] != y.data_ptr() || yk[1] != reinterpret_cast<void*>(static_cast<intptr_t>(M)) ||
      yk[2] != reinterpret_cast<void*>(static_cast<intptr_t>(N))) {
    ymap = make_y_map(y.data_ptr(), M, N, BM);
    yk[0] = y.data_ptr(); yk[1] = reinterpret_cast<void*>(static_cast<intptr_t>(M));
    yk[2] = reinterpret_cast<void*>(static_cast<intptr_t>(N));
  }
  const int nitems = D + R * S;
  dim3 grid(ctas > 0 ? std::min(ctas * C::OCC, nitems) : nitems);
  w4a16_kernel<BM><<<grid, 384, C::SMEM, stream>>>(
      map, ymap, wq.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(ws.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), part_ptr, counters.data_ptr<int>(), M, N, K, kbps, ntn, D, S, nitems);
}

}  // namespace

torch::Tensor w4a16_mm(torch::Tensor x, torch::Tensor wq, torch::Tensor ws, torch::Tensor counters, torch::Tensor part,
                       int64_t N, int64_t bm, int64_t D, int64_t S, int64_t ctas) {
  TORCH_CHECK(x.is_contiguous() && x.scalar_type() == torch::kBFloat16);
  const at::cuda::CUDAGuard guard(x.device());
  auto y = torch::empty({x.size(0), N}, x.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (bm) {
    case 16: launch<16>(x, wq, ws, y, counters, part, N, D, S, ctas, stream); break;
    case 32: launch<32>(x, wq, ws, y, counters, part, N, D, S, ctas, stream); break;
    case 64: launch<64>(x, wq, ws, y, counters, part, N, D, S, ctas, stream); break;
    case 96: launch<96>(x, wq, ws, y, counters, part, N, D, S, ctas, stream); break;
    case 128: launch<128>(x, wq, ws, y, counters, part, N, D, S, ctas, stream); break;
    case 192: launch<192>(x, wq, ws, y, counters, part, N, D, S, ctas, stream); break;
    case 256: launch<256>(x, wq, ws, y, counters, part, N, D, S, ctas, stream); break;
    default: TORCH_CHECK(false, "bad bm");
  }
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("w4a16_mm", &w4a16_mm); }
