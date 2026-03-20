"""Run the single-robot viewer (no training).

Loads the best checkpoint and shows the robot moving toward a goal in the
React frontend.  Uses the same ports as main.py — do not run both at once.

Usage:
    ./venv/bin/python run_single.py

Ctrl-C shuts down cleanly.
"""
from __future__ import annotations

import atexit
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.resolve()
FRONTEND = PROJECT / "frontend"
PID_FILE = PROJECT / ".single_pids"
VENV_PY = PROJECT / "venv" / "bin" / "python"


def _resolve_python() -> str:
    if VENV_PY.exists():
        return str(VENV_PY)
    python313 = shutil.which("python3.13")
    if python313:
        return python313
    return sys.executable


_PY = _resolve_python()


def _save_pids(*pids: int) -> None:
    PID_FILE.write_text("\n".join(str(p) for p in pids if p))


def _kill_from_pid_file() -> None:
    if not PID_FILE.exists():
        return
    pids = PID_FILE.read_text().split()
    killed = 0
    for pid_str in pids:
        try:
            pid = int(pid_str)
            subprocess.run(["pkill", "-KILL", "-P", str(pid)], capture_output=True)
            os.kill(pid, signal.SIGKILL)
            killed += 1
        except (ProcessLookupError, ValueError, OSError):
            pass
    PID_FILE.unlink(missing_ok=True)
    if killed:
        print(f"  cleaned up {killed} process(es) from previous run")
        time.sleep(0.4)


def _kill_port(port: int) -> None:
    result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
    for pid_str in result.stdout.strip().split():
        try:
            os.kill(int(pid_str), signal.SIGKILL)
        except (ProcessLookupError, ValueError):
            pass
    if result.stdout.strip():
        time.sleep(0.5)


def main() -> None:
    print("Clearing previous run…")
    _kill_from_pid_file()
    _kill_port(8000)
    _kill_port(5173)

    procs: list[subprocess.Popen] = []

    def _shutdown(sig=None, frame=None) -> None:
        print("\nShutting down…")
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        time.sleep(1)
        for p in procs:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass
        PID_FILE.unlink(missing_ok=True)
        sys.exit(0)

    atexit.register(_shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server = subprocess.Popen(
        [_PY, "-m", "uvicorn", "server_single:app",
         "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"],
        cwd=PROJECT,
    )
    procs.append(server)
    time.sleep(1)

    frontend = subprocess.Popen(
        ["npm", "run", "dev", "--", "--port", "5173"],
        cwd=FRONTEND,
    )
    procs.append(frontend)

    _save_pids(server.pid, frontend.pid)

    print(f"Python        : {_PY}")
    print("Single viewer : http://localhost:5173")
    print("Press Ctrl-C to stop.\n")

    while True:
        for p in procs:
            if p.poll() is not None:
                print(f"Process {p.args[0]} exited with code {p.returncode}")
                _shutdown()
        time.sleep(1)


if __name__ == "__main__":
    main()
