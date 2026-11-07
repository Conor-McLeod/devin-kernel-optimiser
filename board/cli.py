"""Tiny client for the board, for agents that prefer it over curl.

    python -m board.cli read [--since ID] [--kind finding]
    python -m board.cli post --agent alice --kind finding "TMA multicast on B halves L2 traffic at M>=512"
    python -m board.cli claims
    python -m board.cli leaderboard
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import urllib.parse
import urllib.request

from harness.config import BOARD_URL


def _req(method: str, path: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{BOARD_URL}{path}", data=data, method=method, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def _fmt(p: dict) -> str:
    t = dt.datetime.fromtimestamp(p["ts"]).strftime("%H:%M:%S")
    rt = f" (re #{p['reply_to']})" if p.get("reply_to") else ""
    return f"#{p['id']:<4d} {t} [{p['kind']:8s}] {p['agent']}{rt}: {p['text']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("read")
    r.add_argument("--since", type=int, default=0)
    r.add_argument("--kind")
    p = sp.add_parser("post")
    p.add_argument("--agent", default=os.environ.get("AGENT_NAME", "unknown"))
    p.add_argument("--kind", default="note")
    p.add_argument("--reply-to", type=int)
    p.add_argument("text")
    sp.add_parser("claims")
    sp.add_parser("leaderboard")
    a = ap.parse_args()

    if a.cmd == "read":
        q = {"since": a.since} | ({"kind": a.kind} if a.kind else {})
        for post in _req("GET", "/api/posts?" + urllib.parse.urlencode(q)):
            print(_fmt(post))
    elif a.cmd == "post":
        res = _req("POST", "/api/posts", {"agent": a.agent, "kind": a.kind, "text": a.text, "reply_to": a.reply_to})
        print(f"posted #{res['id']}")
    elif a.cmd == "claims":
        for post in _req("GET", "/api/posts?kind=claim"):
            print(_fmt(post))
    elif a.cmd == "leaderboard":
        lb = _req("GET", "/api/leaderboard")
        print(f"speedup vs {lb['baseline']} (geomean over shapes x M; >1 is faster)")
        for e in lb["entries"]:
            g = f"{e['geomean']:.2f}x" if e["geomean"] else "  -  "
            bk = "  ".join(f"{k}: {v:.2f}x" for k, v in sorted(e["buckets"].items()) if v)
            print(f"{g:>7s}  {e['submission']:24s} {e['dsl']:14s} wins {e['wins']}/{e['compared']}  {bk}")


if __name__ == "__main__":
    main()
