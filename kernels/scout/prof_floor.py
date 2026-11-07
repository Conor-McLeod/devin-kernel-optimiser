# nsys: real kernel durations for a few points, using the harness timing loop structure (flush then call)
import sys, torch
sys.path.insert(0, "/opt/hackathon/workers/w1/repos/devin-kernel-optimiser")
from harness.gpulock import gpu_lock
from harness.formats import random_weight
from harness.submission import load
sub = load(None, sys.argv[1])
pts = [tuple(int(v) for v in p.split("x")) for p in sys.argv[2:]]
gen = torch.Generator().manual_seed(0)
flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
with gpu_lock(holder="scout/nsys-" + sys.argv[1]):
    for N, K, M in pts:
        qw = random_weight("int4_g128", N, K, gen, device="cuda")
        p = sub.prepare(qw)
        x = torch.randn(M, K, generator=gen).to(torch.bfloat16).cuda()
        for _ in range(3): sub.matmul(x, p)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"N{N}K{K}M{M}")
        for _ in range(10):
            flush.zero_(); sub.matmul(x, p)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
