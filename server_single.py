"""Single-robot replay server — no training.

Loads the best saved checkpoint and replays continuous episodes so the
robot can be watched moving toward a goal.  Streams the same step/goal
message format as server.py so the existing React frontend works unchanged.
"""
from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

PROJECT_DIR = str(Path(__file__).parent.resolve())

_mp_queue: mp.Queue | None = None
_clients: set[WebSocket] = set()


# ── Replay process ─────────────────────────────────────────────────────────────

def _replay_process(queue: mp.Queue, project_dir: str) -> None:
    """Runs in a child process.  Replays the best checkpoint indefinitely."""
    import sys
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    os.chdir(project_dir)

    import jax
    from ai.jax_trainer import BRAIN_DT, SINGLE_VIEW_EPISODE_S
    from ai.trainer import ESTrainer

    trainer = ESTrainer(seed=42)

    checkpoints_dir = Path(project_dir) / "checkpoints"
    for candidate in ("best.npz", "latest.npz"):
        path = checkpoints_dir / candidate
        if path.exists():
            trainer.load_checkpoint(path)
            print(f"[single] Loaded {path}  gen={trainer.state.generation}", flush=True)
            break
    else:
        print("[single] No checkpoint found, using random weights.", flush=True)

    _last_step_wall: list[float] = [time.perf_counter()]

    def _put(msg: dict[str, Any]) -> None:
        # Pace output to real-time: sleep until BRAIN_DT seconds have elapsed
        # since the last step was sent.
        now = time.perf_counter()
        elapsed = now - _last_step_wall[0]
        remaining = BRAIN_DT - elapsed
        if remaining > 0.0:
            time.sleep(remaining)
        _last_step_wall[0] = time.perf_counter()
        try:
            queue.put_nowait(msg)
        except Exception:
            pass

    while True:
        goal_xyz = trainer._random_goal()
        trainer._key, episode_key = jax.random.split(trainer._key)
        trainer._run_logged_episode(
            trainer._params, goal_xyz, episode_key, _put,
            steps=int(SINGLE_VIEW_EPISODE_S / BRAIN_DT),
        )


# ── Asyncio broadcast ──────────────────────────────────────────────────────────

async def _drain_and_broadcast(queue: mp.Queue) -> None:
    loop = asyncio.get_running_loop()
    while True:
        msg = await loop.run_in_executor(None, _safe_get, queue)
        if msg is None:
            await asyncio.sleep(0.005)
            continue
        text = json.dumps(msg)
        dead: set[WebSocket] = set()
        for ws in list(_clients):
            try:
                await ws.send_text(text)
            except Exception:
                dead.add(ws)
        _clients.difference_update(dead)


def _safe_get(queue: mp.Queue) -> dict[str, Any] | None:
    try:
        return queue.get(timeout=0.02)
    except Exception:
        return None


# ── FastAPI lifespan ───────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _mp_queue
    ctx = mp.get_context("spawn")
    _mp_queue = ctx.Queue(maxsize=2048)
    proc = ctx.Process(
        target=_replay_process,
        args=(_mp_queue, PROJECT_DIR),
        daemon=True,
    )
    proc.start()
    task = asyncio.create_task(_drain_and_broadcast(_mp_queue))
    yield
    task.cancel()
    proc.terminate()


app = FastAPI(lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── WebSocket endpoint ─────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    _clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        _clients.discard(ws)
