"""Message board + leaderboard for the agents. Runs on the VM next to the workers.

    uv run uvicorn board.server:app --host 0.0.0.0 --port 8765

Agents talk to it over plain HTTP (curl or `python -m board.cli`). Humans
watch http://<vm>:8765/ for the live leaderboard and feed.

Post kinds:
  claim    "I'm working on X" (so others don't duplicate it)
  finding  something learned that others can reuse (a technique, a measurement, a dead end)
  question asking the other agents for help
  result   auto-posted summary when harness.bench finishes
  note     anything else
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from harness.config import PRIMARY_BASELINE, shared_dir

DB_PATH = Path(os.environ.get("BOARD_DB", shared_dir() / "board.sqlite3"))
KINDS = {"claim", "finding", "question", "result", "note"}
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="kernel board")
_lock = threading.Lock()


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


with _db() as _c:
    _c.executescript(
        """
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, agent TEXT, kind TEXT,
            text TEXT, reply_to INTEGER, tags TEXT);
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, run_id TEXT, submission TEXT, agent TEXT,
            dsl TEXT, is_baseline INTEGER, code_hash TEXT, shape TEXT, N INTEGER, K INTEGER, M INTEGER,
            ms REAL, correct INTEGER, error TEXT, note TEXT);
        """
    )


class Post(BaseModel):
    agent: str
    kind: str = "note"
    text: str
    reply_to: int | None = None
    tags: list[str] = []


@app.post("/api/posts")
def create_post(p: Post):
    if p.kind not in KINDS:
        raise HTTPException(400, f"kind must be one of {sorted(KINDS)}")
    with _lock, _db() as con:
        cur = con.execute(
            "INSERT INTO posts (ts, agent, kind, text, reply_to, tags) VALUES (?,?,?,?,?,?)",
            (time.time(), p.agent, p.kind, p.text, p.reply_to, json.dumps(p.tags)),
        )
        return {"id": cur.lastrowid}


@app.get("/api/posts")
def list_posts(since: int = 0, kind: str | None = None, agent: str | None = None, limit: int = 200):
    q, args = "SELECT * FROM posts WHERE id > ?", [since]
    if kind:
        q += " AND kind = ?"
        args.append(kind)
    if agent:
        q += " AND agent = ?"
        args.append(agent)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with _db() as con:
        rows = [dict(r) | {"tags": json.loads(r["tags"] or "[]")} for r in con.execute(q, args)]
    return list(reversed(rows))


@app.post("/api/results")
def add_results(rows: list[dict]):
    if not rows:
        return {"added": 0}
    with _lock, _db() as con:
        con.executemany(
            "INSERT INTO results (ts, run_id, submission, agent, dsl, is_baseline, code_hash, shape, N, K, M, ms, correct, error, note)"
            " VALUES (:ts,:run_id,:submission,:agent,:dsl,:is_baseline,:code_hash,:shape,:N,:K,:M,:ms,:correct,:error,:note)",
            [{**r, "ms": r["ms"] if r["ms"] == r["ms"] else None} for r in rows],
        )
        con.commit()  # so leaderboard() below sees these rows
        r0 = rows[0]
        lb = {e["submission"]: e for e in leaderboard()["entries"]}.get(r0["submission"])
        bad = sum(not r["correct"] for r in rows)
        summary = f"benchmarked {r0['submission']} ({r0['dsl']}) run {r0['run_id']}: {len(rows)} points"
        if bad:
            summary += f", {bad} INCORRECT"
        if lb and lb.get("geomean") and not r0["is_baseline"]:
            summary += f", geomean {lb['geomean']:.2f}x vs {PRIMARY_BASELINE}"
        if r0.get("note"):
            summary += f" — {r0['note']}"
        con.execute(
            "INSERT INTO posts (ts, agent, kind, text, reply_to, tags) VALUES (?,?,?,?,?,?)",
            (time.time(), r0["agent"], "result", summary, None, "[]"),
        )
    return {"added": len(rows)}


def _bucket(m: int) -> str:
    return "decode (M≤16)" if m <= 16 else "mid (32–256)" if m <= 256 else "prefill (≥512)"


@app.get("/api/leaderboard")
def leaderboard():
    """Latest run per submission, speedup vs the latest run of the primary baseline."""
    with _db() as con:
        rows = [dict(r) for r in con.execute("SELECT * FROM results ORDER BY id")]
    latest_run: dict[str, str] = {}
    for r in rows:
        latest_run[r["submission"]] = r["run_id"]
    by_sub: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if latest_run[r["submission"]] == r["run_id"]:
            by_sub[r["submission"]].append(r)

    base = {(r["N"], r["K"], r["M"]): r["ms"] for r in by_sub.get(PRIMARY_BASELINE, []) if r["correct"] and r["ms"]}
    entries = []
    for sub, rs in by_sub.items():
        speedups, per_m, per_bucket = [], defaultdict(list), defaultdict(list)
        for r in rs:
            b = base.get((r["N"], r["K"], r["M"]))
            if b and r["correct"] and r["ms"]:
                s = b / r["ms"]
                speedups.append(s)
                per_m[r["M"]].append(s)
                per_bucket[_bucket(r["M"])].append(s)
        gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None  # noqa: E731
        entries.append({
            "submission": sub, "agent": rs[0]["agent"], "dsl": rs[0]["dsl"],
            "is_baseline": bool(rs[0]["is_baseline"]), "run_id": rs[0]["run_id"], "ts": rs[-1]["ts"],
            "code_hash": rs[0]["code_hash"], "note": rs[0]["note"],
            "points": len(rs), "incorrect": sum(not r["correct"] for r in rs),
            "geomean": gm(speedups),
            "buckets": {k: gm(v) for k, v in per_bucket.items()},
            "curve": sorted([(m, gm(v)) for m, v in per_m.items()]),
            "wins": sum(s > 1.0 for s in speedups), "compared": len(speedups),
        })
    entries.sort(key=lambda e: (e["geomean"] is None, -(e["geomean"] or 0)))
    return {"baseline": PRIMARY_BASELINE, "entries": entries}


@app.get("/api/results")
def list_results(submission: str | None = None, latest: bool = True):
    with _db() as con:
        if submission:
            rows = [dict(r) for r in con.execute("SELECT * FROM results WHERE submission = ? ORDER BY id", (submission,))]
        else:
            rows = [dict(r) for r in con.execute("SELECT * FROM results ORDER BY id")]
    if latest and rows:
        last = {}
        for r in rows:
            last[r["submission"]] = r["run_id"]
        rows = [r for r in rows if last[r["submission"]] == r["run_id"]]
    return rows


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
