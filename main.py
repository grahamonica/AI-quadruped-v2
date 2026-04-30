"""Start the unified viewer API and frontend together."""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from http.client import HTTPConnection
from pathlib import Path

from ai.config import DEFAULT_CONFIG_PATH


PROJECT = Path(__file__).parent.resolve()
FRONTEND = PROJECT / "frontend"
PID_FILE = PROJECT / ".server_pids"
VENV_PY = PROJECT / "venv" / "bin" / "python"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the quadruped viewer API and frontend.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Runtime config for the API process.")
    parser.add_argument("--seed", type=int, default=42, help="Trainer seed for the API process.")
    parser.add_argument("--api-port", type=int, default=8000, help="Port for the FastAPI websocket service.")
    parser.add_argument("--frontend-port", type=int, default=5173, help="Port for the Vite frontend.")
    return parser.parse_args()


def _resolve_python() -> str:
    if Path(sys.executable).exists():
        return sys.executable
    if VENV_PY.exists():
        return str(VENV_PY)
    python313 = shutil.which("python3.13")
    if python313:
        return python313
    python3 = shutil.which("python3")
    if python3:
        return python3
    return sys.executable


_PY = _resolve_python()


def _ensure_runtime_files() -> None:
    missing: list[str] = []
    for relative_path in ("ai/api/live.py", "frontend/package.json", "frontend/index.html"):
        if not (PROJECT / relative_path).exists():
            missing.append(relative_path)
    if missing:
        raise SystemExit("Cannot start the UI stack because required runtime files are missing: " + ", ".join(missing))


def _save_pids(*pids: int) -> None:
    PID_FILE.write_text("\n".join(str(pid) for pid in pids if pid), encoding="utf-8")


def _kill_from_pid_file() -> None:
    if not PID_FILE.exists():
        return
    killed = 0
    for pid_str in PID_FILE.read_text(encoding="utf-8").split():
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
    for pid in result.stdout.strip().split():
        try:
            os.kill(int(pid), signal.SIGKILL)
        except (ProcessLookupError, ValueError):
            pass
    if result.stdout.strip():
        time.sleep(0.4)


def _start_server(api_port: int, config: Path, seed: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["QUADRUPED_CONFIG"] = str(config.resolve())
    env["QUADRUPED_SEED"] = str(seed)
    return subprocess.Popen(
        [_PY, "-m", "uvicorn", "ai.api.live:app", "--host", "0.0.0.0", "--port", str(api_port), "--log-level", "warning"],
        cwd=PROJECT,
        env=env,
    )


def _start_frontend(frontend_port: int, api_port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["VITE_API_PORT"] = str(api_port)
    return subprocess.Popen(
        ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", str(frontend_port)],
        cwd=FRONTEND,
        env=env,
    )


def _wait_for_port(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.8):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def _wait_for_http_ok(host: str, port: int, path: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        connection = None
        try:
            connection = HTTPConnection(host, port, timeout=1.5)
            connection.request("GET", path)
            response = connection.getresponse()
            response.read()
            if 200 <= int(response.status) < 500:
                return True
        except OSError:
            pass
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
        time.sleep(0.35)
    return False


def _wait_until_ready(
    server: subprocess.Popen,
    frontend: subprocess.Popen,
    api_port: int,
    frontend_port: int,
) -> None:
    print("Waiting for viewer API health endpoint...", flush=True)
    if not _wait_for_http_ok("127.0.0.1", api_port, "/healthz", timeout_s=45.0):
        raise RuntimeError(
            f"Viewer API did not become ready on http://127.0.0.1:{api_port}/healthz "
            f"(server return code: {server.poll()})."
        )

    print("Waiting for frontend dev server...", flush=True)
    if not _wait_for_port("127.0.0.1", frontend_port, timeout_s=75.0):
        raise RuntimeError(
            f"Frontend did not open port {frontend_port} "
            f"(frontend return code: {frontend.poll()})."
        )
    if not _wait_for_http_ok("127.0.0.1", frontend_port, "/", timeout_s=20.0):
        raise RuntimeError(
            f"Frontend port {frontend_port} opened but HTTP root did not respond "
            f"(frontend return code: {frontend.poll()})."
        )


def main() -> None:
    args = _parse_args()
    _ensure_runtime_files()

    print("Clearing previous run…", flush=True)
    _kill_from_pid_file()
    _kill_port(args.api_port)
    _kill_port(args.frontend_port)

    processes: list[subprocess.Popen] = []

    def _shutdown(_sig=None, _frame=None, *, exit_process: bool = True) -> None:
        print("\nShutting down…", flush=True)
        for process in processes:
            try:
                process.terminate()
            except Exception:
                pass
        time.sleep(1)
        for process in processes:
            try:
                if process.poll() is None:
                    process.kill()
            except Exception:
                pass
        PID_FILE.unlink(missing_ok=True)
        if exit_process:
            raise SystemExit(0)

    atexit.register(lambda: _shutdown(exit_process=False))
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server = _start_server(args.api_port, args.config, args.seed)
    processes.append(server)
    time.sleep(1)

    frontend = _start_frontend(args.frontend_port, args.api_port)
    processes.append(frontend)
    try:
        _wait_until_ready(server, frontend, args.api_port, args.frontend_port)
    except Exception as exc:
        print(f"Startup failed: {exc}", flush=True)
        _shutdown()
        return

    _save_pids(server.pid, frontend.pid)

    print(f"Python          : {_PY}", flush=True)
    print(f"Config          : {args.config.resolve()}", flush=True)
    print(f"Viewer API      : http://localhost:{args.api_port}", flush=True)
    print(f"Frontend        : http://localhost:{args.frontend_port}", flush=True)
    print("Press Ctrl-C to stop.\n", flush=True)

    while True:
        for process in processes:
            if process.poll() is not None:
                print(f"Process {process.args[0]} exited with code {process.returncode}", flush=True)
                _shutdown()
        time.sleep(1)


if __name__ == "__main__":
    main()
