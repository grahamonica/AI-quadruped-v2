"""Entry point: starts the training server and the React frontend together.

Usage (any python):
    python main.py

Ctrl-C shuts down both processes cleanly.
Existing processes on ports 8000 and 5173 are killed before starting.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.resolve()
FRONTEND = PROJECT / "frontend"

# Use python3.13 (which has uvicorn/fastapi/numpy installed system-wide).
# Fall back to the current interpreter if python3.13 isn't on PATH.
_PY = "python3.13" if subprocess.run(
    ["python3.13", "--version"], capture_output=True
).returncode == 0 else sys.executable


def _kill_port(port: int) -> None:
    """Kill any process currently listening on the given port."""
    result = subprocess.run(
        ["lsof", "-ti", f":{port}"],
        capture_output=True, text=True,
    )
    pids = result.stdout.strip().split()
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except (ProcessLookupError, ValueError):
            pass
    if pids:
        time.sleep(0.5)  # let the OS reclaim the port


def _start_server() -> subprocess.Popen:
    return subprocess.Popen(
        [_PY, "-m", "uvicorn", "server:app",
         "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"],
        cwd=PROJECT,
    )


def _start_frontend() -> subprocess.Popen:
    return subprocess.Popen(
        ["npm", "run", "dev", "--", "--port", "5173"],
        cwd=FRONTEND,
    )


def main() -> None:
    print("Clearing ports 8000 and 5173…")
    _kill_port(8000)
    _kill_port(5173)

    procs: list[subprocess.Popen] = []

    def _shutdown(sig, frame):
        print("\nShutting down…")
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server = _start_server()
    procs.append(server)
    time.sleep(1)

    frontend = _start_frontend()
    procs.append(frontend)

    print(f"Python          : {_PY}")
    print("Training server : http://localhost:8000")
    print("Frontend        : http://localhost:5173")
    print("Press Ctrl-C to stop.\n")

    while True:
        for p in procs:
            if p.poll() is not None:
                print(f"Process {p.args[0]} exited with code {p.returncode}")
                _shutdown(None, None)
        time.sleep(1)


if __name__ == "__main__":
    main()
