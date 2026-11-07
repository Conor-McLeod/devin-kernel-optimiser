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

template <int BM>
struct Cfg {
  static constexpr int X_BYTES = BM * BK * 2;
  static constexpr int STAGE_BYTES = ((X_BYTES + W_BYTES + S_BYTES + 1023) / 1024) * 1024;
  static constexpr int STAGES_RAW = 200 * 1024 / STAGE_BYTES;
  static constexpr int STAGES = STAGES_RAW > 8 ? 8 : STAGES_RAW;
  static constexpr int EPI_STRIDE = BN + 8;
  static constexpr int EPI_BYTES = BM * EPI_STRIDE * 2;
  static constexpr int PIPE_BYTES = STAGES * STAGE_BYTES;
  static constexpr int BAR_OFF = PIPE_BYTES > EPI_BYTES ? PIPE_BYTES : EPI_BYTES;
  static constexpr int SMEM = 1024 + BAR_OFF + 2 * STAGES * 8 + 16;
  static constexpr int NACC = BM / 2;
};

template <int BM>
__global__ void __launch_bounds__(384, 1)
    w4a16_kernel(const __grid_constant__ CUtensorMap tmx, const uint8_t* __restrict__ wq,
                 const __nv_bfloat16* __restrict__ ws, __nv_bfloat16* __restrict__ y, float* __restrict__ part,
                 int* __restrict__ counters, int M, int N, int K, int kb_per_split) {
  using C = Cfg<BM>;
  constexpr int STAGES = C::STAGES;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + C::BAR_OFF);
  uint64_t* empty = full + STAGES;
  int* flag = reinterpret_cast<int*>(empty + STAGES);

  const int tile_n = blockIdx.x, tile_m = blockIdx.y, split = blockIdx.z, nsplit = gridDim.z;
  const int KB = K / BK, G = K / 128;
  const int kb0 = split * kb_per_split;
  const int kb1 = min(KB, kb0 + kb_per_split);
  const int nk = kb1 - kb0;
  const int wg = threadIdx.x / 128, tid = threadIdx.x % 128;

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
      const uint8_t* wsrc = wq + (static_cast<size_t>(tile_n) * KB + kb0) * W_BYTES;
      const __nv_bfloat16* ssrc = ws + static_cast<size_t>(tile_n) * G * BN;
      for (int i = 0; i < nk; ++i) {
        const int s = i % STAGES;
        if (i >= STAGES) mbar_wait(&empty[s], ((i / STAGES) - 1) & 1);
        uint8_t* st = smem + s * C::STAGE_BYTES;
        mbar_expect_tx(&full[s], C::X_BYTES + W_BYTES + S_BYTES);
        tma_load_2d(st, &tmx, &full[s], (kb0 + i) * BK, tile_m * BM);
        bulk_load(st + C::X_BYTES, wsrc + static_cast<size_t>(i) * W_BYTES, W_BYTES, &full[s]);
        bulk_load(st + C::X_BYTES + W_BYTES, ssrc + ((kb0 + i) >> 1) * BN, S_BYTES, &full[s]);
      }
    }
    return;
  }

  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  const int c = wg - 1, warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
  float acc[C::NACC];
#pragma unroll
  for (int r = 0; r < C::NACC; ++r) acc[r] = 0.f;

  for (int i = 0; i < nk; ++i) {
    const int s = i % STAGES;
    mbar_wait(&full[s], (i / STAGES) & 1);
    const uint8_t* st = smem + s * C::STAGE_BYTES;
    const uint4 q = reinterpret_cast<const uint4*>(st + C::X_BYTES)[c * 128 + tid];
    const __nv_bfloat16* sc = reinterpret_cast<const __nv_bfloat16*>(st + C::X_BYTES + W_BYTES);
    const int row = c * 64 + warp * 16 + g;
    const __nv_bfloat162 s0b = __bfloat162bfloat162(sc[row]);
    const __nv_bfloat162 s1b = __bfloat162bfloat162(sc[row + 8]);
    const uint32_t s0 = *reinterpret_cast<const uint32_t*>(&s0b);
    const uint32_t s1 = *reinterpret_cast<const uint32_t*>(&s1b);
    uint32_t a[4][4];
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
    wg_wait<0>();
#pragma unroll
    for (int r = 0; r < C::NACC; ++r) fence_reg(acc[r]);
    __syncwarp();
    if (lane == 0) mbar_arrive(&empty[s]);
  }

  if (nsplit > 1) {
    // Split-K: publish fp32 partial; the last CTA of this tile reduces and stores.
    const int tile_lin = tile_m * gridDim.x + tile_n;
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
    if (!*flag) return;
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

  // Epilogue: transpose through smem ([m][n] bf16), then 16B coalesced stores.
  consumer_sync();
  __nv_bfloat16* epi = reinterpret_cast<__nv_bfloat16*>(smem);
  const int nl = c * 64 + warp * 16 + g;
#pragma unroll
  for (int i = 0; i < BM / 8; ++i) {
    const int m = 8 * i + 2 * t;
    epi[m * C::EPI_STRIDE + nl] = __float2bfloat16_rn(acc[4 * i]);
    epi[(m + 1) * C::EPI_STRIDE + nl] = __float2bfloat16_rn(acc[4 * i + 1]);
    epi[m * C::EPI_STRIDE + nl + 8] = __float2bfloat16_rn(acc[4 * i + 2]);
    epi[(m + 1) * C::EPI_STRIDE + nl + 8] = __float2bfloat16_rn(acc[4 * i + 3]);
  }
  consumer_sync();
  const int ctid = c * 128 + tid;
  const int n0 = tile_n * BN;
  for (int idx = ctid; idx < BM * 16; idx += 256) {
    const int m = idx / 16, ch = idx % 16;
    const int gm = tile_m * BM + m, gn = n0 + ch * 8;
    if (gm < M && gn < N) {
      const uint4 v = *reinterpret_cast<const uint4*>(epi + m * C::EPI_STRIDE + ch * 8);
      *reinterpret_cast<uint4*>(y + static_cast<size_t>(gm) * N + gn) = v;
    }
  }
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
            torch::Tensor& counters, int N, int split, cudaStream_t stream) {
  using C = Cfg<BM>;
  const int M = x.size(0), K = x.size(1);
  const int KB = K / BK;
  const int ntiles = (N + BN - 1) / BN, mtiles = (M + BM - 1) / BM;
  split = std::max(1, std::min(split, KB));
  const int kbps = (KB + split - 1) / split;
  split = (KB + kbps - 1) / kbps;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(w4a16_kernel<BM>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    attr_set = true;
  }
  torch::Tensor part;
  float* part_ptr = nullptr;
  if (split > 1) {
    TORCH_CHECK(counters.numel() >= ntiles * mtiles, "counter buffer too small");
    part = torch::empty({static_cast<int64_t>(ntiles) * mtiles * split * BM * BN},
                        x.options().dtype(torch::kFloat32));
    part_ptr = part.data_ptr<float>();
  }
  CUtensorMap map = make_x_map(x.data_ptr(), M, K, BM);
  dim3 grid(ntiles, mtiles, split);
  w4a16_kernel<BM><<<grid, 384, C::SMEM, stream>>>(
      map, wq.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(ws.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), part_ptr, counters.data_ptr<int>(), M, N, K, kbps);
}

}  // namespace

torch::Tensor w4a16_mm(torch::Tensor x, torch::Tensor wq, torch::Tensor ws, torch::Tensor counters, int64_t N,
                       int64_t bm, int64_t split) {
  TORCH_CHECK(x.is_contiguous() && x.scalar_type() == torch::kBFloat16);
  const at::cuda::CUDAGuard guard(x.device());
  auto y = torch::empty({x.size(0), N}, x.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (bm) {
    case 16: launch<16>(x, wq, ws, y, counters, N, split, stream); break;
    case 32: launch<32>(x, wq, ws, y, counters, N, split, stream); break;
    case 64: launch<64>(x, wq, ws, y, counters, N, split, stream); break;
    case 128: launch<128>(x, wq, ws, y, counters, N, split, stream); break;
    case 256: launch<256>(x, wq, ws, y, counters, N, split, stream); break;
    default: TORCH_CHECK(false, "bad bm");
  }
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("w4a16_mm", &w4a16_mm); }
