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
import re
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


VIEWS = ("latest", "best", "final")
_RUN_RE = re.compile(r"\brun[\s:=#]*([0-9a-f]{8})\b")


def _gm(xs: list[float]) -> float | None:
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def _run_stats(rs: list[dict], base: dict) -> dict:
    speedups, per_m, per_bucket = [], defaultdict(list), defaultdict(list)
    for r in rs:
        b = base.get((r["N"], r["K"], r["M"]))
        if b and r["correct"] and r["ms"]:
            s = b / r["ms"]
            speedups.append(s)
            per_m[r["M"]].append(s)
            per_bucket[_bucket(r["M"])].append(s)
    return {
        "submission": rs[0]["submission"], "agent": rs[0]["agent"], "dsl": rs[0]["dsl"],
        "is_baseline": bool(rs[0]["is_baseline"]), "run_id": rs[0]["run_id"], "ts": rs[-1]["ts"],
        "code_hash": rs[0]["code_hash"], "note": rs[0]["note"],
        "points": len(rs), "incorrect": sum(not r["correct"] for r in rs),
        "geomean": _gm(speedups),
        "buckets": {k: _gm(v) for k, v in per_bucket.items()},
        "curve": sorted([(m, _gm(v)) for m, v in per_m.items()]),
        "wins": sum(s > 1.0 for s in speedups), "compared": len(speedups),
    }


def _final_runs(con: sqlite3.Connection, run_owner: dict[str, tuple[str, str]]) -> dict[str, str]:
    """submission -> run_id named in that agent's most recent FINAL post."""
    out: dict[str, str] = {}
    for p in con.execute("SELECT agent, text FROM posts WHERE text LIKE '%FINAL%' ORDER BY id"):
        if "FINAL" not in p["text"]:  # LIKE is case-insensitive
            continue
        for run_id in _RUN_RE.findall(p["text"]):
            if run_id in run_owner and run_owner[run_id][1] == p["agent"]:
                out[run_owner[run_id][0]] = run_id
                break
    return out


@app.get("/api/leaderboard")
def leaderboard(view: str = "latest"):
    """One run per submission, speedup vs the latest run of the primary baseline.

    view=latest  the submission's most recent run
    view=best    its highest-geomean run among full-coverage runs with no incorrect points
    view=final   the run named in its agent's latest FINAL post (falls back to latest)
    Baselines always use their latest run.
    """
    if view not in VIEWS:
        raise HTTPException(400, f"view must be one of {VIEWS}")
    with _db() as con:
        rows = [dict(r) for r in con.execute("SELECT * FROM results ORDER BY id")]
        runs: dict[str, dict[str, list[dict]]] = defaultdict(dict)  # submission -> run_id -> rows, in run order
        for r in rows:
            runs[r["submission"]].setdefault(r["run_id"], []).append(r)
        run_owner = {rid: (sub, rs[0]["agent"]) for sub, by_run in runs.items() for rid, rs in by_run.items()}
        finals = _final_runs(con, run_owner) if view == "final" else {}

    latest = {sub: list(by_run.values())[-1] for sub, by_run in runs.items()}
    base = {(r["N"], r["K"], r["M"]): r["ms"] for r in latest.get(PRIMARY_BASELINE, []) if r["correct"] and r["ms"]}
    entries = []
    for sub, by_run in runs.items():
        e = _run_stats(latest[sub], base)
        selection = "latest"
        if view == "best" and not e["is_baseline"]:
            cands = [_run_stats(rs, base) for rs in by_run.values()]
            full = max(c["compared"] for c in cands)
            ok = [c for c in cands if c["compared"] == full and not c["incorrect"] and c["geomean"]]
            if ok:
                e, selection = max(ok, key=lambda c: c["geomean"]), "best"
        elif view == "final" and not e["is_baseline"]:
            if sub in finals:
                e, selection = _run_stats(by_run[finals[sub]], base), "final"
            else:
                selection = "latest (no FINAL)"
        e["selection"], e["runs"] = selection, len(by_run)
        entries.append(e)
    entries.sort(key=lambda e: (e["geomean"] is None, -(e["geomean"] or 0)))
    return {"baseline": PRIMARY_BASELINE, "view": view, "entries": entries}


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
