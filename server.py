"""WebSocket server: streams live training state to the React frontend.

Training runs in a separate *process* to avoid GIL starvation of the
uvicorn event loop.  The subprocess receives the project directory so it
can find the `quadruped` and `ai` packages regardless of
the working directory when the process is spawned.
"""
from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

PROJECT_DIR = str(Path(__file__).parent.resolve())

# ── Globals ────────────────────────────────────────────────────────────────────
_mp_queue: mp.Queue | None = None
_clients: set[WebSocket] = set()
_latest_gen: dict[str, Any] = {}


# ── Training process ───────────────────────────────────────────────────────────

def _training_process(queue: mp.Queue, seed: int, project_dir: str) -> None:
    """Runs in a child process.  Puts step/generation dicts into queue."""
    import sys
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    os.chdir(project_dir)

    from ai.trainer import ESTrainer

    trainer = ESTrainer(seed=seed)

    def _put(msg: dict[str, Any]) -> None:
        try:
            queue.put_nowait(msg)
        except Exception:
            pass  # drop if queue full — keeps frontend responsive

    while True:
        trainer.run_generation(on_step=_put, on_gen_done=_put)


# ── Asyncio broadcast ──────────────────────────────────────────────────────────

async def _drain_and_broadcast(queue: mp.Queue) -> None:
    loop = asyncio.get_running_loop()
    while True:
        msg = await loop.run_in_executor(None, _safe_get, queue)
        if msg is None:
            await asyncio.sleep(0.005)
            continue
        if msg.get("type") == "generation":
            _latest_gen.update(msg)
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
        target=_training_process,
        args=(_mp_queue, 7, PROJECT_DIR),
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
    if _latest_gen:
        try:
            await ws.send_text(json.dumps(_latest_gen))
        except Exception:
            pass
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        _clients.discard(ws)
