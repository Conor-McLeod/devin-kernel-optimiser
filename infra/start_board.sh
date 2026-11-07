#!/usr/bin/env bash
# Start the message board + leaderboard on :8765 in a tmux session.
# View from your laptop with:  ssh -L 8765:127.0.0.1:8765 <vm>  then open http://localhost:8765
set -euo pipefail
H=/opt/hackathon
tmux has-session -t board 2>/dev/null && { echo "board already running (tmux attach -t board)"; exit 0; }
tmux new-session -d -s board "cd $H/main && $H/venv/bin/uvicorn board.server:app --host 127.0.0.1 --port 8765"
sleep 2 && curl -s localhost:8765/api/leaderboard | head -c 200 && echo
