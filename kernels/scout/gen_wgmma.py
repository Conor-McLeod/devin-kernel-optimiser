"""Generates wgmma_ops.cuh: register-A wgmma wrappers (m64nNk16, bf16 -> f32)."""
NS = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]

out = ["#pragma once", "#include <cstdint>", ""]
for n in NS:
    nd = n // 2
    dregs = ", ".join(f"%{i}" for i in range(nd))
    a0 = nd
    asm = (f'"{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{nd + 5}, 0;\\n'
           f'wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 '
           f'{{{dregs}}}, {{%{a0}, %{a0 + 1}, %{a0 + 2}, %{a0 + 3}}}, %{a0 + 4}, p, 1, 1, 0;\\n}}\\n"')
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nd))
    out.append(f"__device__ __forceinline__ void wgmma_rs_{n}(float* d, const uint32_t* a, uint64_t desc, int scale_d) {{")
    out.append(f"  asm volatile({asm}")
    out.append(f"    : {outs}")
    out.append(f'    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));')
    out.append("}")
    out.append("")
out.append("template <int N> __device__ __forceinline__ void wgmma_rs(float* d, const uint32_t* a, uint64_t desc, int scale_d);")
for n in NS:
    out.append(f"template <> __device__ __forceinline__ void wgmma_rs<{n}>(float* d, const uint32_t* a, uint64_t desc, int scale_d) {{ wgmma_rs_{n}(d, a, desc, scale_d); }}")
open(__file__.replace("gen_wgmma.py", "wgmma_ops.cuh"), "w").write("\n".join(out) + "\n")
