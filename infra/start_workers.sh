#!/usr/bin/env bash
# Start N Devin Outpost workers on this VM, one tmux session each.
# Each worker serves one Devin session at a time, so N workers = N parallel agents.
#   bash infra/start_workers.sh 5
#
# UNVERIFIED: the docs say "N workers serve N concurrent sessions" but don't say whether
# several workers may share one machine. Each gets its own directory and acceptor id.
# If sessions misbehave, fall back to one worker per container.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/.env"
N="${1:-5}"
H=/opt/hackathon
for i in $(seq 1 "$N"); do
  d="$H/workers/w$i"
  mkdir -p "$d/repos"
  # Pre-clone so session start is fast (workers reuse repos/<name> if present).
  [[ -d "$d/repos/$REPO_NAME" ]] || git clone "$REPO_URL" "$d/repos/$REPO_NAME"
  if tmux has-session -t "w$i" 2>/dev/null; then echo "w$i already running"; continue; fi
  tmux new-session -d -s "w$i" \
    "cd $d && DEVIN_WORKER_ACCEPTOR_ID=$(hostname)-w$i HACK_SHARED=$H/shared \
     devin worker start --outpost=$OUTPOST --token=$DEVIN_OUTPOSTS_TOKEN; bash"
  echo "started worker w$i in $d"
done
echo "tmux ls to see workers; tmux attach -t w1 to watch one."
