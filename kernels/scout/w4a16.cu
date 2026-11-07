// Hopper W4A16 GEMM: y[M,N] = x[M,K] @ W[N,K]^T, W int4 (uint4b8) with bf16 scale per 128 K.
// Swap-AB: W (dequantised in registers) is the wgmma A operand, x (TMA, 128B swizzle) is B.
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

// PERSIST: grid-stride over (n,m) tiles with a separate epilogue buffer, so the producer keeps
// prefetching the next tile while consumers run the epilogue (split-K must be 1).
template <int BM, bool PERSIST>
struct Cfg {
  static constexpr int X_BYTES = BM * BK * 2;
  static constexpr int STAGE_BYTES = ((X_BYTES + W_BYTES + S_BYTES + 1023) / 1024) * 1024;
  static constexpr int EPI_STRIDE = BN + 8;
  static constexpr int EPI_BYTES = ((BM * EPI_STRIDE * 2 + 1023) / 1024) * 1024;
  static constexpr int AVAIL = PERSIST ? (227 * 1024 - 1024 - 256 - EPI_BYTES) : 200 * 1024;
  static constexpr int STAGES_RAW = AVAIL / STAGE_BYTES;
  static constexpr int STAGES = STAGES_RAW > 8 ? 8 : STAGES_RAW;
  static constexpr int PIPE_BYTES = STAGES * STAGE_BYTES;
  static constexpr int EPI_OFF = PERSIST ? PIPE_BYTES : 0;
  static constexpr int BAR_OFF = PERSIST ? PIPE_BYTES + EPI_BYTES : (PIPE_BYTES > EPI_BYTES ? PIPE_BYTES : EPI_BYTES);
  static constexpr int SMEM = 1024 + BAR_OFF + 2 * STAGES * 8 + 16;
  static constexpr int NACC = BM / 2;
};

template <int BM, bool PERSIST>
__global__ void __launch_bounds__(384, 1)
    w4a16_kernel(const __grid_constant__ CUtensorMap tmx, const uint8_t* __restrict__ wq,
                 const __nv_bfloat16* __restrict__ ws, __nv_bfloat16* __restrict__ y, float* __restrict__ part,
                 int* __restrict__ counters, int M, int N, int K, int kb_per_split) {
  using C = Cfg<BM, PERSIST>;
  constexpr int STAGES = C::STAGES;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + C::BAR_OFF);
  uint64_t* empty = full + STAGES;
  int* flag = reinterpret_cast<int*>(empty + STAGES);

  // PERSIST = stream-K: CTA b owns work iterations [b*per, (b+1)*per) of the flattened (tile, k-block)
  // space; tiles split across CTAs are reduced by the last arriving CTA (fp32 partials in `part`).
  const int split = PERSIST ? 0 : blockIdx.z, nsplit = PERSIST ? 1 : gridDim.z;
  const int ntn = (N + BN - 1) / BN;
  const int KB = K / BK, G = K / 128;
  const int kb0 = PERSIST ? 0 : split * kb_per_split;
  const int kb1 = PERSIST ? KB : min(KB, kb0 + kb_per_split);
  const int nk = kb1 - kb0;
  const int per = kb_per_split;
  const int work = PERSIST ? ntn * ((M + BM - 1) / BM) * KB : nk;
  const int beg = PERSIST ? blockIdx.x * per : 0;
  const int end = PERSIST ? min(work, beg + per) : nk;
  const int wg = threadIdx.x / 128, tid = threadIdx.x % 128;
  auto segment = [&](int cur, int& tile_n, int& tile_m, int& kbeg, int& kend) {
    if (PERSIST) {
      const int tile = cur / KB;
      kbeg = cur % KB;
      kend = min(KB, kbeg + (end - cur));
      tile_n = tile % ntn;
      tile_m = tile / ntn;
    } else {
      tile_n = blockIdx.x;
      tile_m = blockIdx.y;
      kbeg = kb0;
      kend = kb1;
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
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (tid == 0) {
      int it = 0;
      for (int cur = beg; cur < end;) {
        int tile_n, tile_m, kbeg, kend;
        segment(cur, tile_n, tile_m, kbeg, kend);
        cur += kend - kbeg;
        const uint8_t* wsrc = wq + static_cast<size_t>(tile_n) * KB * W_BYTES;
        const __nv_bfloat16* ssrc = ws + static_cast<size_t>(tile_n) * G * BN;
        for (int kb = kbeg; kb < kend; ++kb, ++it) {
          const int s = it % STAGES;
          if (it >= STAGES) mbar_wait(&empty[s], ((it / STAGES) - 1) & 1);
          uint8_t* st = smem + s * C::STAGE_BYTES;
          mbar_expect_tx(&full[s], C::X_BYTES + W_BYTES + S_BYTES);
          tma_load_2d(st, &tmx, &full[s], kb * BK, tile_m * BM);
          bulk_load(st + C::X_BYTES, wsrc + static_cast<size_t>(kb) * W_BYTES, W_BYTES, &full[s]);
          bulk_load(st + C::X_BYTES + W_BYTES, ssrc + (kb >> 1) * BN, S_BYTES, &full[s]);
        }
      }
    }
    return;
  }

  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  const int c = wg - 1, warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
  float acc[C::NACC];
  const int row = c * 64 + warp * 16 + g;
  // Stage i: wait for TMA, dequantise into a, issue 4 wgmma, keep one wgmma group in flight.
  int it = 0;
  auto step = [&](int li, uint32_t (&a)[4][4]) {
    const int i = it + li;
    const int s = i % STAGES;
    mbar_wait(&full[s], (i / STAGES) & 1);
    const uint8_t* st = smem + s * C::STAGE_BYTES;
    const uint4 q = reinterpret_cast<const uint4*>(st + C::X_BYTES)[c * 128 + tid];
    const __nv_bfloat16* sc = reinterpret_cast<const __nv_bfloat16*>(st + C::X_BYTES + W_BYTES);
    const __nv_bfloat162 s0b = __bfloat162bfloat162(sc[row]);
    const __nv_bfloat162 s1b = __bfloat162bfloat162(sc[row + 8]);
    const uint32_t s0 = *reinterpret_cast<const uint32_t*>(&s0b);
    const uint32_t s1 = *reinterpret_cast<const uint32_t*>(&s1b);
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
    if (li > 0) {
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[(i - 1) % STAGES]);
    }
  };
  uint32_t a0[4][4], a1[4][4];
  const int ctid = c * 128 + tid;
  for (int cur = beg; cur < end;) {
  int tile_n, tile_m, kbeg, kend;
  segment(cur, tile_n, tile_m, kbeg, kend);
  const int nseg = kend - kbeg;
  const bool first_seg = (cur == beg);
  cur += nseg;
#pragma unroll
  for (int r = 0; r < C::NACC; ++r) acc[r] = 0.f;
  int i = 0;
  for (; i + 1 < nseg; i += 2) {
    step(i, a0);
    step(i + 1, a1);
  }
  if (i < nseg) step(i, a0);
  wg_wait<0>();
#pragma unroll
  for (int r = 0; r < C::NACC; ++r) fence_reg(acc[r]);
  if (PERSIST) {
    __syncwarp();
    if (lane == 0) mbar_arrive(&empty[(it + nseg - 1) % STAGES]);
  }
  it += nseg;

  const bool partial = PERSIST ? (nseg < KB) : (nsplit > 1);
  if (partial) {
    // Publish fp32 partial; the last CTA to arrive for this tile reduces and stores.
    const int tile_lin = tile_m * ntn + tile_n;
    const size_t tstart = static_cast<size_t>(tile_lin) * KB;
    const int g_first = PERSIST ? static_cast<int>(tstart / per) : 0;
    const int g_last = PERSIST ? static_cast<int>((tstart + KB - 1) / per) : nsplit - 1;
    const int me = PERSIST ? static_cast<int>(blockIdx.x) : split;
    auto slot = [&](int gq) -> size_t {
      if (PERSIST) return static_cast<size_t>(gq) * 2 + (static_cast<size_t>(gq) * per >= tstart ? 0 : 1);
      return static_cast<size_t>(tile_lin) * nsplit + gq;
    };
    float4* mine = reinterpret_cast<float4*>(part + slot(me) * BM * BN);
#pragma unroll
    for (int r = 0; r < C::NACC / 4; ++r)
      mine[r * 256 + ctid] = make_float4(acc[4 * r], acc[4 * r + 1], acc[4 * r + 2], acc[4 * r + 3]);
    __threadfence();
    consumer_sync();
    if (ctid == 0) {
      const int old = atomicAdd(&counters[tile_lin], 1);
      const int last = (old == g_last - g_first);
      if (last) counters[tile_lin] = 0;
      *flag = last;
    }
    consumer_sync();
    if (!*flag) continue;
    __threadfence();
    for (int gq = g_first; gq <= g_last; ++gq) {
      if (gq == me) continue;
      const float4* other = reinterpret_cast<const float4*>(part + slot(gq) * BM * BN);
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

  // Epilogue: transpose through smem ([m][n] bf16), then 16B coalesced stores.
  consumer_sync();
  __nv_bfloat16* epi = reinterpret_cast<__nv_bfloat16*>(smem + C::EPI_OFF);
  {
    // acc fragment (row n=g/g+8, cols m=2t,2t+1) == stmatrix fragment; .trans writes epi[m][n].
    const int nbase = c * 64 + warp * 16, mk = lane >> 3, mc = lane & 7;
#pragma unroll
    for (int i = 0; i < BM / 8; i += 2) {
      uint32_t r[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int i2 = i + (k >> 1), h = k & 1;
        __nv_bfloat162 v = __floats2bfloat162_rn(acc[4 * i2 + 2 * h], acc[4 * i2 + 2 * h + 1]);
        r[k] = *reinterpret_cast<uint32_t*>(&v);
      }
      const int m = 8 * (i + (mk >> 1)) + mc;
      const uint32_t addr = smem_u32(epi + m * C::EPI_STRIDE + nbase + 8 * (mk & 1));
      asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(r[0]),
                   "r"(r[1]), "r"(r[2]), "r"(r[3])
                   : "memory");
    }
  }
  consumer_sync();
  const int n0 = tile_n * BN;
  for (int idx = ctid; idx < BM * 16; idx += 256) {
    const int m = idx / 16, ch = idx % 16;
    const int gm = tile_m * BM + m, gn = n0 + ch * 8;
    if (gm < M && gn < N) {
      const uint4 v = *reinterpret_cast<const uint4*>(epi + m * C::EPI_STRIDE + ch * 8);
      *reinterpret_cast<uint4*>(y + static_cast<size_t>(gm) * N + gn) = v;
    }
  }
  }  // tile loop
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

int num_sms() {
  static int n = 0;
  if (!n) {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
  }
  return n;
}

template <int BM, bool PERSIST>
void launch(const torch::Tensor& x, const torch::Tensor& wq, const torch::Tensor& ws, torch::Tensor& y,
            torch::Tensor& counters, int N, int split, cudaStream_t stream) {
  // PERSIST: `split` is the minimum number of k-blocks per CTA (stream-K granularity).
  using C = Cfg<BM, PERSIST>;
  const int M = x.size(0), K = x.size(1);
  const int KB = K / BK;
  const int ntiles = (N + BN - 1) / BN, mtiles = (M + BM - 1) / BM;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(w4a16_kernel<BM, PERSIST>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    attr_set = true;
  }
  TORCH_CHECK(counters.numel() >= ntiles * mtiles, "counter buffer too small");
  torch::Tensor part;
  float* part_ptr = nullptr;
  int grid_x, kbps;
  if (PERSIST) {
    const int work = ntiles * mtiles * KB;
    const int min_per = std::max(1, split);
    int g = std::min(num_sms(), std::max(1, work / min_per));
    kbps = (work + g - 1) / g;
    grid_x = (work + kbps - 1) / kbps;
    if (kbps % KB != 0 || work % kbps != 0) {
      part = torch::empty({static_cast<int64_t>(grid_x) * 2 * BM * BN}, x.options().dtype(torch::kFloat32));
      part_ptr = part.data_ptr<float>();
    }
  } else {
    split = std::max(1, std::min(split, KB));
    kbps = (KB + split - 1) / split;
    split = (KB + kbps - 1) / kbps;
    grid_x = ntiles;
    if (split > 1) {
      part = torch::empty({static_cast<int64_t>(ntiles) * mtiles * split * BM * BN},
                          x.options().dtype(torch::kFloat32));
      part_ptr = part.data_ptr<float>();
    }
  }
  CUtensorMap map = make_x_map(x.data_ptr(), M, K, BM);
  dim3 grid = PERSIST ? dim3(grid_x) : dim3(ntiles, mtiles, split);
  w4a16_kernel<BM, PERSIST><<<grid, 384, C::SMEM, stream>>>(
      map, wq.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(ws.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), part_ptr, counters.data_ptr<int>(), M, N, K, kbps);
}

}  // namespace

torch::Tensor w4a16_mm(torch::Tensor x, torch::Tensor wq, torch::Tensor ws, torch::Tensor counters, int64_t N,
                       int64_t bm, int64_t split, int64_t persist) {
  TORCH_CHECK(x.is_contiguous() && x.scalar_type() == torch::kBFloat16);
  const at::cuda::CUDAGuard guard(x.device());
  auto y = torch::empty({x.size(0), N}, x.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (persist) {
    // persist > 0: stream-K, `split` = minimum k-blocks per CTA.
    switch (bm) {
      case 64: launch<64, true>(x, wq, ws, y, counters, N, split, stream); break;
      case 128: launch<128, true>(x, wq, ws, y, counters, N, split, stream); break;
      case 256: launch<256, true>(x, wq, ws, y, counters, N, split, stream); break;
      default: TORCH_CHECK(false, "bad bm for persist");
    }
    return y;
  }
  switch (bm) {
    case 16: launch<16, false>(x, wq, ws, y, counters, N, split, stream); break;
    case 32: launch<32, false>(x, wq, ws, y, counters, N, split, stream); break;
    case 64: launch<64, false>(x, wq, ws, y, counters, N, split, stream); break;
    case 128: launch<128, false>(x, wq, ws, y, counters, N, split, stream); break;
    case 256: launch<256, false>(x, wq, ws, y, counters, N, split, stream); break;
    default: TORCH_CHECK(false, "bad bm");
  }
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("w4a16_mm", &w4a16_mm); }
