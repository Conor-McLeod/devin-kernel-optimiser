"""Launch the parallel Devin sessions on the outpost via the v3 API.

Dry run (default) prints the prompts and the payload; nothing is sent and no credits are spent:
    python infra/launch_agents.py --agents scout,triton,cuda-wgmma,cute-dsl,decode

Actually launch:
    python infra/launch_agents.py --agents scout,triton,cuda-wgmma,cute-dsl,decode --go

Message every launched session (e.g. a mid-day redirect):
    python infra/launch_agents.py --broadcast "Scout found Marlin hits 92% DRAM BW at M=1; focus on M>=64."

Reads infra/.env (see env.example). Launched sessions are recorded in infra/launched.json.
API per docs.devin.ai v3: POST /v3/organizations/{org_id}/sessions with `platform` = outpost name.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAUNCHED = ROOT / "infra" / "launched.json"


def load_env() -> dict:
    env = dict(os.environ)
    f = ROOT / "infra" / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    return env


def build_prompt(agent: str, repo_name: str) -> str:
    shared = (ROOT / "agents" / "AGENTS.md").read_text()
    task = (ROOT / "agents" / "tasks" / f"{agent}.md").read_text()
    return f"""Your agent name is `{agent}`. Before anything else run `export AGENT_NAME={agent}`.

The repository is already checked out in this outpost at `repos/{repo_name}`. Work there:
`cd repos/{repo_name} && git checkout -b agent/{agent}`. Your kernel directory is `kernels/{agent}/`,
and its META name must be "{agent}".

{shared}

---

{task}
"""


def api(env: dict, method: str, path: str, body: dict | None = None) -> dict:
    url = f"{env.get('DEVIN_API_BASE', 'https://api.devin.ai')}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"Authorization": f"Bearer {env['DEVIN_API_KEY']}", "Content-Type": "application/json"},
    )
    try:
        return json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {url} -> {e.code}: {e.read().decode()[:1000]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="scout,triton,cuda-wgmma,cute-dsl,decode")
    ap.add_argument("--max-acu", type=int, default=None, help="per-session ACU cap (max_acu_limit)")
    ap.add_argument("--go", action="store_true", help="actually create sessions (spends credits)")
    ap.add_argument("--broadcast", help="send this message to every session in launched.json")
    args = ap.parse_args()
    env = load_env()

    if args.broadcast:
        sessions = json.loads(LAUNCHED.read_text())
        for s in sessions:
            sid = s["session_id"]
            devin_id = sid if sid.startswith("devin-") else f"devin-{sid}"
            api(env, "POST", f"/v3/organizations/{env['DEVIN_ORG_ID']}/sessions/{devin_id}/messages", {"message": args.broadcast})
            print(f"sent to {s['agent']} ({sid})")
        return

    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    for a in agents:
        if not (ROOT / "agents" / "tasks" / f"{a}.md").exists():
            raise SystemExit(f"no task brief agents/tasks/{a}.md")

    launched = json.loads(LAUNCHED.read_text()) if LAUNCHED.exists() else []
    for a in agents:
        body = {
            "prompt": build_prompt(a, env.get("REPO_NAME", "devin-kernel-optimiser")),
            "title": f"kernel agent: {a}",
            "tags": ["hackathon", "kernels", a],
            # API pool name can differ from the worker's outpost name (personal pools are "user:<name>").
            "platform": env.get("SESSION_PLATFORM") or env.get("OUTPOST", "h100-kernels"),
        }
        if args.max_acu:
            body["max_acu_limit"] = args.max_acu
        if not args.go:
            print(f"===== {a} (dry run) =====")
            print(json.dumps({k: v for k, v in body.items() if k != "prompt"}, indent=2))
            print(body["prompt"][:600] + "\n...\n")
            continue
        res = api(env, "POST", f"/v3/organizations/{env['DEVIN_ORG_ID']}/sessions", body)
        launched.append({"agent": a, "session_id": res["session_id"], "url": res.get("url")})
        print(f"launched {a}: {res.get('url')}")
        LAUNCHED.write_text(json.dumps(launched, indent=2))

    if not args.go:
        print("dry run only. Re-run with --go to create sessions.", file=sys.stderr)


if __name__ == "__main__":
    main()
