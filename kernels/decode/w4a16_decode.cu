// Swap-AB W4A16 decode GEMM for Hopper (sm_90a).
// y^T[N, M] = W[N, K] * x^T[K, M] with mma.sync m16n8k16 (A = dequantised W, B = x).
// Weight is pre-permuted so each lane streams its A fragments with one 128-bit load
// per 16(n) x 64(k) block; K is permuted inside each 64-block so each lane's
// B fragment is 16 contiguous bf16 of x.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ uint4 ld_x(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ uint32_t ld_scale(const uint32_t* p) {
  uint32_t r;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
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

// word holds 8 nibbles; nibble r + 4e is element e of A register r. Output: bf16x2 of (code - 8).
__device__ __forceinline__ void dequant(uint32_t q, uint32_t (&r)[4]) {
  const uint32_t MASK = 0x000F000F, EX = 0x43004300;  // 0x4300 = bf16 128.0
  r[0] = lop3_and_or(q, MASK, EX);
  r[1] = lop3_and_or(q >> 4, MASK, EX);
  r[2] = lop3_and_or(q >> 8, MASK, EX);
  r[3] = lop3_and_or(q >> 12, MASK, EX);
  const __nv_bfloat162 off = __halves2bfloat162(__ushort_as_bfloat16(0x4308), __ushort_as_bfloat16(0x4308));  // 136
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&r[i]);
    v = __hsub2(v, off);
    r[i] = *reinterpret_cast<uint32_t*>(&v);
  }
}

template <int T, int U, int MB, int WC, int S>
__global__ void __launch_bounds__(WC * 32)
w4a16_decode_kernel(const uint4* __restrict__ W, const uint32_t* __restrict__ Sc,
                    const __nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ Y,
                    int M, int N, int K) {
  constexpr int OUT = T * MB * 4 * 32;  // fp32 outputs per warp tile
  __shared__ float red[WC][OUT];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int KB = K >> 6, G = K >> 7;
  const int nt0 = blockIdx.x * T;
  const int split = blockIdx.y;
  const int kb0 = (split * WC + warp) * U;

  uint4 wq[T][U];
#pragma unroll
  for (int u = 0; u < U; ++u) {
    if (kb0 + u < KB) {
#pragma unroll
      for (int i = 0; i < T; ++i) wq[i][u] = ld_stream(W + ((size_t)(nt0 + i) * KB + kb0 + u) * 32 + lane);
    }
  }
  uint4 xr[U][MB][2];
#pragma unroll
  for (int u = 0; u < U; ++u) {
#pragma unroll
    for (int mb = 0; mb < MB; ++mb) {
      const int m = mb * 8 + g;
      if (m < M && kb0 + u < KB) {
        const uint4* p = reinterpret_cast<const uint4*>(X + (size_t)m * K + (kb0 + u) * 64 + 16 * t);
        xr[u][mb][0] = ld_x(p);
        xr[u][mb][1] = ld_x(p + 1);
      } else {
        xr[u][mb][0] = make_uint4(0, 0, 0, 0);
        xr[u][mb][1] = make_uint4(0, 0, 0, 0);
      }
    }
  }
  uint32_t sc[T][U];
#pragma unroll
  for (int u = 0; u < U; ++u) {
    if (kb0 + u < KB) {
#pragma unroll
      for (int i = 0; i < T; ++i) sc[i][u] = ld_scale(Sc + ((size_t)(nt0 + i) * G + ((kb0 + u) >> 1)) * 8 + g);
    }
  }

  float acc[T][MB][4];
#pragma unroll
  for (int i = 0; i < T; ++i)
#pragma unroll
    for (int mb = 0; mb < MB; ++mb)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[i][mb][c] = 0.f;

#pragma unroll
  for (int u = 0; u < U; ++u) {
    if (kb0 + u < KB) {
#pragma unroll
      for (int i = 0; i < T; ++i) {
        uint32_t a[4][4];
        dequant(wq[i][u].x, a[0]);
        dequant(wq[i][u].y, a[1]);
        dequant(wq[i][u].z, a[2]);
        dequant(wq[i][u].w, a[3]);
        __nv_bfloat162 s2 = *reinterpret_cast<__nv_bfloat162*>(&sc[i][u]);
        const float slo = __low2float(s2), shi = __high2float(s2);
#pragma unroll
        for (int mb = 0; mb < MB; ++mb) {
          float p[4] = {0.f, 0.f, 0.f, 0.f};
          mma_bf16(p, a[0], xr[u][mb][0].x, xr[u][mb][0].y);
          mma_bf16(p, a[1], xr[u][mb][0].z, xr[u][mb][0].w);
          mma_bf16(p, a[2], xr[u][mb][1].x, xr[u][mb][1].y);
          mma_bf16(p, a[3], xr[u][mb][1].z, xr[u][mb][1].w);
          acc[i][mb][0] = fmaf(slo, p[0], acc[i][mb][0]);
          acc[i][mb][1] = fmaf(slo, p[1], acc[i][mb][1]);
          acc[i][mb][2] = fmaf(shi, p[2], acc[i][mb][2]);
          acc[i][mb][3] = fmaf(shi, p[3], acc[i][mb][3]);
        }
      }
    }
  }

#pragma unroll
  for (int i = 0; i < T; ++i)
#pragma unroll
    for (int mb = 0; mb < MB; ++mb)
#pragma unroll
      for (int c = 0; c < 4; ++c) red[warp][((i * MB + mb) * 4 + c) * 32 + lane] = acc[i][mb][c];
  __syncthreads();

  if constexpr (S == 1) {
    for (int idx = threadIdx.x; idx < OUT; idx += WC * 32) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WC; ++w) v += red[w][idx];
      const int l = idx & 31, c = (idx >> 5) & 3, imb = idx >> 7;
      const int i = imb / MB, mb = imb % MB;
      const int n = (nt0 + i) * 16 + (l >> 2) + 8 * (c >> 1);
      const int m = mb * 8 + 2 * (l & 3) + (c & 1);
      if (m < M) Y[(size_t)m * N + n] = __float2bfloat16(v);
    }
  } else {
    __shared__ float part[OUT];
    for (int idx = threadIdx.x; idx < OUT; idx += WC * 32) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WC; ++w) v += red[w][idx];
      part[idx] = v;
    }
    cg::cluster_group cluster = cg::this_cluster();
    cluster.sync();
    const int rank = cluster.block_rank();
    float* peers[S];
#pragma unroll
    for (int r = 0; r < S; ++r) peers[r] = cluster.map_shared_rank(part, r);
    for (int idx = rank * WC * 32 + threadIdx.x; idx < OUT; idx += S * WC * 32) {
      const int l = idx & 31, c = (idx >> 5) & 3, imb = idx >> 7;
      const int i = imb / MB, mb = imb % MB;
      const int m = mb * 8 + 2 * (l & 3) + (c & 1);
      if (m < M) {
        float v = 0.f;
#pragma unroll
        for (int r = 0; r < S; ++r) v += peers[r][idx];
        const int n = (nt0 + i) * 16 + (l >> 2) + 8 * (c >> 1);
        Y[(size_t)m * N + n] = __float2bfloat16(v);
      }
    }
    cluster.sync();
  }
}

template <int T, int U, int MB, int WC, int S>
void launch(const torch::Tensor& W, const torch::Tensor& Sc, const torch::Tensor& X, torch::Tensor& Y, int M, int N,
            int K, cudaStream_t stream) {
  auto kern = w4a16_decode_kernel<T, U, MB, WC, S>;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / 16 / T, S, 1);
  cfg.blockDim = dim3(WC * 32, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 1;
  attr[0].val.clusterDim.y = S;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, kern, reinterpret_cast<const uint4*>(W.data_ptr()),
                     reinterpret_cast<const uint32_t*>(Sc.data_ptr()),
                     reinterpret_cast<const __nv_bfloat16*>(X.data_ptr()),
                     reinterpret_cast<__nv_bfloat16*>(Y.data_ptr()), M, N, K);
}

#define CFG_LIST(X) \
  X(1, 1, 1, 4, 1) X(1, 1, 1, 8, 1) X(1, 2, 1, 4, 1) X(1, 2, 1, 8, 1) X(1, 4, 1, 4, 1) X(1, 4, 1, 8, 1) \
  X(1, 8, 1, 4, 1) X(1, 8, 1, 8, 1) X(1, 1, 1, 4, 2) X(1, 1, 1, 8, 2) X(1, 2, 1, 4, 2) X(1, 2, 1, 8, 2) \
  X(1, 4, 1, 4, 2) X(1, 4, 1, 8, 2) X(1, 8, 1, 4, 2) X(1, 8, 1, 8, 2) X(1, 1, 1, 4, 4) X(1, 1, 1, 8, 4) \
  X(1, 2, 1, 4, 4) X(1, 2, 1, 8, 4) X(1, 4, 1, 4, 4) X(1, 4, 1, 8, 4) X(1, 8, 1, 4, 4) X(1, 8, 1, 8, 4) \
  X(1, 1, 1, 4, 8) X(1, 1, 1, 8, 8) X(1, 2, 1, 4, 8) X(1, 2, 1, 8, 8) X(1, 4, 1, 4, 8) X(1, 4, 1, 8, 8) \
  X(1, 8, 1, 4, 8) X(1, 8, 1, 8, 8) X(1, 1, 2, 4, 1) X(1, 1, 2, 8, 1) X(1, 2, 2, 4, 1) X(1, 2, 2, 8, 1) \
  X(1, 4, 2, 4, 1) X(1, 4, 2, 8, 1) X(1, 8, 2, 4, 1) X(1, 8, 2, 8, 1) X(1, 1, 2, 4, 2) X(1, 1, 2, 8, 2) \
  X(1, 2, 2, 4, 2) X(1, 2, 2, 8, 2) X(1, 4, 2, 4, 2) X(1, 4, 2, 8, 2) X(1, 8, 2, 4, 2) X(1, 8, 2, 8, 2) \
  X(1, 1, 2, 4, 4) X(1, 1, 2, 8, 4) X(1, 2, 2, 4, 4) X(1, 2, 2, 8, 4) X(1, 4, 2, 4, 4) X(1, 4, 2, 8, 4) \
  X(1, 8, 2, 4, 4) X(1, 8, 2, 8, 4) X(1, 1, 2, 4, 8) X(1, 1, 2, 8, 8) X(1, 2, 2, 4, 8) X(1, 2, 2, 8, 8) \
  X(1, 4, 2, 4, 8) X(1, 4, 2, 8, 8) X(1, 8, 2, 4, 8) X(1, 8, 2, 8, 8) X(2, 2, 2, 4, 1) X(2, 2, 2, 8, 1) \
  X(2, 4, 2, 8, 1) X(2, 2, 2, 8, 2) X(2, 4, 2, 4, 2)

void run(torch::Tensor W, torch::Tensor Sc, torch::Tensor X, torch::Tensor Y, int64_t M, int64_t N, int64_t K,
         int64_t T, int64_t U, int64_t MB, int64_t WC, int64_t S) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define DISPATCH(t, u, mb, wc, s)                                                    \
  if (T == t && U == u && MB == mb && WC == wc && S == s) {                          \
    launch<t, u, mb, wc, s>(W, Sc, X, Y, (int)M, (int)N, (int)K, stream);            \
    return;                                                                          \
  }
  CFG_LIST(DISPATCH)
#undef DISPATCH
  TORCH_CHECK(false, "unsupported config T=", T, " U=", U, " MB=", MB, " WC=", WC, " S=", S);
}

std::vector<std::vector<int64_t>> configs() {
  std::vector<std::vector<int64_t>> out;
#define LIST(t, u, mb, wc, s) out.push_back({t, u, mb, wc, s});
  CFG_LIST(LIST)
#undef LIST
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run);
  m.def("configs", &configs);
}
