import sys, torch
sys.path.insert(0, "/opt/hackathon/workers/w1/repos/devin-kernel-optimiser")
from harness.formats import random_weight
from harness.submission import load_kernel_dir
from harness.verify import check_once
sub = load_kernel_dir(sys.argv[1])
for pt in sys.argv[2:]:
    N, K, M = map(int, pt.split("x"))
    for seed in range(3):
        gen = torch.Generator().manual_seed(seed)
        qw = random_weight("int4_g128", N, K, gen, device="cuda")
        p = sub.prepare(qw)
        v = check_once(sub, qw, p, M, gen)
        print(pt, "seed", seed, "ok" if v.ok else "FAIL", v.rel_fro, v.msg, flush=True)
