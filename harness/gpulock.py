"""One benchmark at a time on the GPU.

All agents share one H100. Compiling and correctness checks can run in parallel,
but timing runs must not overlap or every number is noise. `harness.bench`
takes this lock automatically; never time kernels any other way.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import socket
import time

from .config import shared_dir


@contextlib.contextmanager
def gpu_lock(holder: str, timeout_s: float = 1800):
    d = shared_dir()
    lock_path = d / "bench.lock"
    owner_path = d / "bench.lock.owner"
    f = open(lock_path, "a+")
    t0 = time.time()
    announced = False
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if not announced:
                try:
                    who = owner_path.read_text().strip()
                except OSError:
                    who = "?"
                print(f"[gpulock] waiting for GPU (held by {who}) ...", flush=True)
                announced = True
            if time.time() - t0 > timeout_s:
                raise TimeoutError("timed out waiting for the GPU benchmark lock")
            time.sleep(0.5)
    owner_path.write_text(json.dumps({"holder": holder, "pid": os.getpid(), "host": socket.gethostname(), "since": time.time()}))
    if announced:
        print(f"[gpulock] acquired after {time.time() - t0:.0f}s", flush=True)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            owner_path.write_text("")
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()
