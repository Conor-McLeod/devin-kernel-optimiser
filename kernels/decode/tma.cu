// Persistent, warp-specialised W4A16 decode GEMM for Hopper (sm_90a).
//
// Weight is repacked into "units": one 16(n) x 128(k) scale group = 2 x 512B of int4
// already in mma.sync A-fragment order + 32B of bf16 scales = 1056B, laid out
// [N/16][K/128] so the whole weight is one linear stream.  Each CTA (one per SM)
// owns a contiguous range of that stream.  A producer warp streams it into a
// shared-memory ring with cp.async.bulk (TMA bulk copy) + mbarrier; NC consumer
// warps each take one unit per stage, dequantise with lop3, run swap-AB
// mma.sync m16n8k16 (x is the 8-wide B operand), and reduce per tile in smem.
// Tiles that straddle CTA ranges are reduced through a self-cleaning fp32
// workspace (red.add + per-tile counter; the last contributor writes y).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>

constexpr int UNIT_BYTES = 1056;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(
          smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(policy)
      : "memory");
}
__device__ __forceinline__ uint4 ld_x(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ void red_add(float* p, float v) {
  asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory");
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t lop3_and_or(uint32_t a, uint32_t mask, uint32_t ex) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(mask), "r"(ex));
  return r;
}
__device__ __forceinline__ void dequant(uint32_t q, uint32_t (&r)[4]) {
  const uint32_t MASK = 0x000F000F, EX = 0x43004300;  // bf16 128 + code
  r[0] = lop3_and_or(q, MASK, EX);
  r[1] = lop3_and_or(q >> 4, MASK, EX);
  r[2] = lop3_and_or(q >> 8, MASK, EX);
  r[3] = lop3_and_or(q >> 12, MASK, EX);
  const __nv_bfloat162 off = __halves2bfloat162(__ushort_as_bfloat16(0x4308), __ushort_as_bfloat16(0x4308));  // 136
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    __nv_bfloat162 v = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&r[i]), off);
    r[i] = *reinterpret_cast<uint32_t*>(&v);
  }
}

template <int MB>
__device__ __forceinline__ void load_x(uint4 (&xr)[2][MB][2], const __nv_bfloat16* __restrict__ X, int M, int K,
                                       int grp, int g, int t) {
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int mb = 0; mb < MB; ++mb) {
      const int m = mb * 8 + g;
      if (m < M) {
        const uint4* p = reinterpret_cast<const uint4*>(X + (size_t)m * K + grp * 128 + h * 64 + 16 * t);
        xr[h][mb][0] = ld_x(p);
        xr[h][mb][1] = ld_x(p + 1);
      } else {
        xr[h][mb][0] = xr[h][mb][1] = make_uint4(0, 0, 0, 0);
      }
    }
}

template <int MB, int NC, int U, int NS>
__global__ void __launch_bounds__((NC + 1) * 32, 1)
tma_kernel(const uint8_t* __restrict__ Wu, const __nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ Y,
           float* __restrict__ ws, int* __restrict__ cnt, int M, int N, int K) {
  constexpr int CH = NC * U;  // units per stage
  constexpr int MR = MB * 8;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* ring = smem;
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + NS * CH * UNIT_BYTES);
  uint64_t* empty = full + NS;
  float* tacc = reinterpret_cast<float*>(empty + NS);  // [ntl][MR][16]

  const int G = K >> 7;
  const long long total = (long long)(N >> 4) * G;
  const int u0 = (int)(total * blockIdx.x / gridDim.x);
  const int u1 = (int)(total * (blockIdx.x + 1) / gridDim.x);
  const int nunits = u1 - u0;
  const int nstages = (nunits + CH - 1) / CH;
  const int tile0 = u0 / G;
  const int ntl = (u1 - 1) / G - tile0 + 1;

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    for (int s = 0; s < NS; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], NC);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = threadIdx.x; i < ntl * MR * 16; i += blockDim.x) tacc[i] = 0.f;
  __syncthreads();

  if (warp == NC) {
    // producer
    if (lane == 0) {
      uint64_t policy;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
      for (int i = 0; i < nstages; ++i) {
        const int s = i % NS;
        if (i >= NS) mbar_wait(&empty[s], ((i / NS) & 1) ^ 1);
        const int ub = u0 + i * CH;
        const int nu = min(CH, u1 - ub);
        const uint32_t bytes = nu * UNIT_BYTES;
        mbar_expect_tx(&full[s], bytes);
        bulk_g2s(ring + s * CH * UNIT_BYTES, Wu + (size_t)ub * UNIT_BYTES, bytes, &full[s], policy);
      }
    }
    return;
  }

  // consumers
  const int g = lane >> 2, t = lane & 3;
  float acc[MB][4];
#pragma unroll
  for (int mb = 0; mb < MB; ++mb)
#pragma unroll
    for (int c = 0; c < 4; ++c) acc[mb][c] = 0.f;
  int cur_tile = -1;

  auto flush = [&]() {
    if (cur_tile >= 0) {
      float* base = tacc + (cur_tile - tile0) * MR * 16;
#pragma unroll
      for (int mb = 0; mb < MB; ++mb)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int m = mb * 8 + 2 * t + (c & 1);
          const int n = g + 8 * (c >> 1);
          if (m < M) atomicAdd(base + m * 16 + n, acc[mb][c]);
          acc[mb][c] = 0.f;
        }
    }
  };

  // x fragments for this warp's U units of the current stage, prefetched one stage ahead
  uint4 xr[U][2][MB][2];
#pragma unroll
  for (int v = 0; v < U; ++v) {
    const int u = u0 + warp * U + v;
    load_x<MB>(xr[v], X, M, K, (u < u1 ? u : u0) % G, g, t);
  }
  for (int i = 0; i < nstages; ++i) {
    const int s = i % NS;
    const int ubase = u0 + i * CH + warp * U;
    uint4 xn[U][2][MB][2];
#pragma unroll
    for (int v = 0; v < U; ++v) {
      const int un = ubase + CH + v;
      load_x<MB>(xn[v], X, M, K, (un < u1 ? un : u0) % G, g, t);
    }
    mbar_wait(&full[s], (i / NS) & 1);
    uint4 w[U][2];
    uint32_t sc[U];
#pragma unroll
    for (int v = 0; v < U; ++v) {
      if (ubase + v < u1) {
        const uint8_t* up = ring + (s * CH + warp * U + v) * UNIT_BYTES;
        w[v][0] = *reinterpret_cast<const uint4*>(up + lane * 16);
        w[v][1] = *reinterpret_cast<const uint4*>(up + 512 + lane * 16);
        sc[v] = *reinterpret_cast<const uint32_t*>(up + 1024 + g * 4);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(&empty[s]);
    float p[U][2][MB][4];
#pragma unroll
    for (int v = 0; v < U; ++v)
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int mb = 0; mb < MB; ++mb)
#pragma unroll
          for (int c = 0; c < 4; ++c) p[v][h][mb][c] = 0.f;
#pragma unroll
    for (int v = 0; v < U; ++v) {
      if (ubase + v < u1) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t a[4][4];
          dequant(w[v][h].x, a[0]);
          dequant(w[v][h].y, a[1]);
          dequant(w[v][h].z, a[2]);
          dequant(w[v][h].w, a[3]);
#pragma unroll
          for (int mb = 0; mb < MB; ++mb) {
            mma_bf16(p[v][h][mb], a[0], xr[v][h][mb][0].x, xr[v][h][mb][0].y);
            mma_bf16(p[v][h][mb], a[1], xr[v][h][mb][0].z, xr[v][h][mb][0].w);
            mma_bf16(p[v][h][mb], a[2], xr[v][h][mb][1].x, xr[v][h][mb][1].y);
            mma_bf16(p[v][h][mb], a[3], xr[v][h][mb][1].z, xr[v][h][mb][1].w);
          }
        }
      }
    }
#pragma unroll
    for (int v = 0; v < U; ++v) {
      const int u = ubase + v;
      if (u < u1) {
        const int tile = u / G;
        if (tile != cur_tile) {
          flush();
          cur_tile = tile;
        }
        __nv_bfloat162 s2 = *reinterpret_cast<__nv_bfloat162*>(&sc[v]);
        const float slo = __low2float(s2), shi = __high2float(s2);
#pragma unroll
        for (int mb = 0; mb < MB; ++mb) {
          acc[mb][0] = fmaf(slo, p[v][0][mb][0] + p[v][1][mb][0], acc[mb][0]);
          acc[mb][1] = fmaf(slo, p[v][0][mb][1] + p[v][1][mb][1], acc[mb][1]);
          acc[mb][2] = fmaf(shi, p[v][0][mb][2] + p[v][1][mb][2], acc[mb][2]);
          acc[mb][3] = fmaf(shi, p[v][0][mb][3] + p[v][1][mb][3], acc[mb][3]);
        }
      }
    }
#pragma unroll
    for (int v = 0; v < U; ++v)
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int mb = 0; mb < MB; ++mb) {
          xr[v][h][mb][0] = xn[v][h][mb][0];
          xr[v][h][mb][1] = xn[v][h][mb][1];
        }
  }
  flush();
  asm volatile("bar.sync 1, %0;" ::"r"(NC * 32) : "memory");

  // epilogue: one consumer warp per local tile
  M = min(M, MR);
  for (int tl = warp; tl < ntl; tl += NC) {
    const int tile = tile0 + tl;
    const int ts = max(tile * G, u0), te = min((tile + 1) * G, u1);
    const int seg = te - ts;
    const float* base = tacc + tl * MR * 16;
    if (seg == G) {
      for (int e = lane; e < M * 16; e += 32) {
        const int m = e >> 4, n = e & 15;
        Y[(size_t)m * N + tile * 16 + n] = __float2bfloat16(base[e]);
      }
    } else {
      for (int e = lane; e < M * 16; e += 32) {
        const int m = e >> 4, n = e & 15;
        red_add(ws + (size_t)m * N + tile * 16 + n, base[e]);
      }
      __threadfence();
      __syncwarp();
      int old = 0;
      if (lane == 0) old = atomicAdd(cnt + tile, seg);
      old = __shfl_sync(0xffffffffu, old, 0);
      if (old + seg == G) {
        __threadfence();
        for (int e = lane; e < M * 16; e += 32) {
          const int m = e >> 4, n = e & 15;
          float* p = ws + (size_t)m * N + tile * 16 + n;
          Y[(size_t)m * N + tile * 16 + n] = __float2bfloat16(__ldcg(p));
          __stcg(p, 0.f);
        }
        if (lane == 0) cnt[tile] = 0;
      }
    }
  }
}

template <int MB, int NC, int U, int NS>
void launch(const torch::Tensor& Wu, const torch::Tensor& X, torch::Tensor& Y, torch::Tensor& ws, torch::Tensor& cnt,
            int M, int N, int K, int ctas, cudaStream_t stream) {
  auto kern = tma_kernel<MB, NC, U, NS>;
  const int G = K / 128;
  const long long total = (long long)(N / 16) * G;
  const int max_range = (int)((total + ctas - 1) / ctas) + 1;
  const int ntl_max = max_range / G + 2;
  const size_t smem = (size_t)NS * NC * U * UNIT_BYTES + 2 * NS * sizeof(uint64_t) + (size_t)ntl_max * MB * 8 * 16 * 4;
  TORCH_CHECK(smem <= 227 * 1024, "smem too large: ", smem);
  static size_t configured = 0;
  if (smem > configured) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    configured = 227 * 1024;
  }
  kern<<<ctas, (NC + 1) * 32, smem, stream>>>(reinterpret_cast<const uint8_t*>(Wu.data_ptr()),
                                              reinterpret_cast<const __nv_bfloat16*>(X.data_ptr()),
                                              reinterpret_cast<__nv_bfloat16*>(Y.data_ptr()), ws.data_ptr<float>(),
                                              cnt.data_ptr<int>(), M, N, K);
}

#define CFG_LIST(X) \
  X(1, 8, 1, 16) X(1, 16, 1, 8) X(1, 8, 2, 8) X(1, 8, 2, 12) X(1, 16, 2, 4) \
  X(1, 16, 2, 6) X(1, 4, 4, 8) X(1, 8, 4, 4) X(1, 12, 2, 6) X(1, 16, 1, 12) \
  X(2, 8, 1, 16) X(2, 16, 1, 8) X(2, 8, 2, 8) X(2, 8, 2, 12) X(2, 16, 2, 4) \
  X(2, 16, 2, 6) X(2, 4, 4, 8) X(2, 8, 4, 4) X(2, 12, 2, 6) X(2, 16, 1, 12)

void run(torch::Tensor Wu, torch::Tensor X, torch::Tensor Y, torch::Tensor ws, torch::Tensor cnt, int64_t M,
         int64_t N, int64_t K, int64_t MB, int64_t NC, int64_t U, int64_t NS, int64_t ctas) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define DISPATCH(mb, nc, u, ns)                                                        \
  if (MB == mb && NC == nc && U == u && NS == ns) {                                    \
    launch<mb, nc, u, ns>(Wu, X, Y, ws, cnt, (int)M, (int)N, (int)K, (int)ctas, stream);  \
    return;                                                                            \
  }
  CFG_LIST(DISPATCH)
#undef DISPATCH
  TORCH_CHECK(false, "unsupported tma config MB=", MB, " NC=", NC, " U=", U, " NS=", NS);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
