import json
S=json.load(open("/opt/hackathon/workers/w1/repos/devin-kernel-optimiser/harness/shapes.json"))
BW=3.35e12; PK=989e12
print("roofline floor per point (H100 SXM: 3.35 TB/s HBM, 989 TFLOP/s dense bf16). bytes=N*K/2 + N*K/64 (bf16 scales) + 2*M*K + 2*M*N")
for s in S["shapes"]:
    N,K=s["N"],s["K"]; row=[]
    for M in S["ms"]:
        b=N*K//2+N*K//128*2+2*M*K+2*M*N; f=2*M*N*K
        t=max(b/BW,f/PK); row.append(f"M{M}:{t*1e6:.1f}us({'mem' if b/BW>f/PK else 'cmp'})")
    wb=N*K//2+N*K//64
    print(f"{s['name']} N={N} K={K} Wbytes={wb/1e6:.2f}MB ridge M~{int(PK/BW/4)} | "+" ".join(row))
