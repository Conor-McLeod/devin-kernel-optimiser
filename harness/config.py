"""Paths and endpoints shared by every agent on the VM."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Shared across all Devin workers on the VM: GPU lock, results log.
# infra/setup_vm.sh creates /opt/hackathon/shared; locally we fall back to ./shared.
if "HACK_SHARED" in os.environ:
    SHARED_DIR = Path(os.environ["HACK_SHARED"])
elif Path("/opt/hackathon").is_dir():
    SHARED_DIR = Path("/opt/hackathon/shared")
else:
    SHARED_DIR = REPO_ROOT / "shared"

BOARD_URL = os.environ.get("BOARD_URL", "http://127.0.0.1:8765")

# Baseline that speedups on the leaderboard are measured against.
PRIMARY_BASELINE = os.environ.get("PRIMARY_BASELINE", "sglang_marlin")

SHAPES_FILE = REPO_ROOT / "harness" / "shapes.json"


def shared_dir() -> Path:
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    return SHARED_DIR
