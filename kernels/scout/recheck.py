"""Re-check the exact code behind a submission's best benchmark run.
usage: recheck.py <submission_name>
Finds the best-geomean... simply: latest all-correct full run for that submission, its code_hash,
then searches every worker clone's git history for a commit whose kernel dir hashes to it,
exports that tree with `git archive` and runs harness.check on it."""
import hashlib, io, json, subprocess, sys, tarfile, tempfile
from pathlib import Path
name = sys.argv[1]
runs = {}
for l in open("/opt/hackathon/shared/results.jsonl"):
    r = json.loads(l)
    if r["submission"] == name:
        runs.setdefault(r["run_id"], []).append(r)
import math
base = {}
for l in open("/opt/hackathon/shared/results.jsonl"):
    r = json.loads(l)
    if r["submission"] == "sglang_marlin" and r["correct"]:
        base[(r["N"], r["K"], r["M"])] = r["ms"]
def gm(v):
    if not all(r["correct"] for r in v): return -1
    xs = [base[(r["N"], r["K"], r["M"])] / r["ms"] for r in v if (r["N"], r["K"], r["M"]) in base]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else -1
npts = max(len(v) for v in runs.values())
if len(sys.argv) > 2:
    rid = sys.argv[2]; rows = runs[rid]
else:
    _, rid, rows = max((gm(v), k, v) for k, v in runs.items() if len(v) == npts)
print(f"geomean {gm(rows):.3f}")
want = rows[0]["code_hash"]
print(f"{name}: best full run {rid} code_hash={want} correct={all(r['correct'] for r in rows)}")
def h_of(tf, prefix):
    h = hashlib.sha1()
    mem = sorted((m for m in tf.getmembers() if m.isfile()), key=lambda m: m.name)
    for m in mem:
        p = Path(m.name)
        if p.suffix in {".py", ".cu", ".cuh", ".h", ".cpp"} and "__pycache__" not in p.parts:
            h.update(tf.extractfile(m).read())
    return h.hexdigest()[:10]
for repo in sorted(Path("/opt/hackathon/workers").glob("w*/repos/devin-kernel-optimiser")):
    sub = f"kernels/{name}"
    if not (repo / sub).exists():
        continue
    commits = subprocess.run(["git", "-C", repo, "log", "--all", "--format=%h", "-30", "--", sub], capture_output=True, text=True).stdout.split()
    for c in commits:
        data = subprocess.run(["git", "-C", repo, "archive", c, sub], capture_output=True).stdout
        tf = tarfile.open(fileobj=io.BytesIO(data))
        if h_of(tf, sub) == want:
            d = Path(tempfile.mkdtemp(prefix=f"recheck_{name}_", dir="/opt/hackathon/workers/w1/scout/recheck"))
            tf.extractall(d)
            print(f"matched commit {c} in {repo}; exported to {d / sub}")
            r = subprocess.run([sys.executable, "-m", "harness.check", "--kernel", str(d / sub)], cwd="/opt/hackathon/workers/w1/repos/devin-kernel-optimiser", capture_output=True, text=True)
            print("\n".join(l for l in r.stdout.splitlines() if " ok " not in l)[-2000:], r.stderr[-500:] if r.returncode else "")
            sys.exit(r.returncode)
print("NO COMMIT MATCHES the benchmarked code_hash (benched from uncommitted code?)")
sys.exit(2)
