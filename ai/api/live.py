"""Live training websocket service."""

from __future__ import annotations

import asyncio
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from ai.config import DEFAULT_CONFIG_PATH, load_runtime_spec
from ai.trainer import ESTrainer, apply_runtime_spec
from ai.runtime import resolve_training_resume_checkpoint

from .common import BroadcastHub, build_viewer_metadata


def _load_service_config() -> tuple[Path, int]:
    config_path = Path(os.environ.get("QUADRUPED_CONFIG", str(DEFAULT_CONFIG_PATH)))
    seed = int(os.environ.get("QUADRUPED_SEED", "42"))
    return config_path, seed


def _trainer_thread(hub: BroadcastHub, config_path: Path, seed: int) -> None:
    spec = load_runtime_spec(config_path)
    apply_runtime_spec(spec)
    trainer = ESTrainer(seed=seed, spec=spec)
    resume_path = resolve_training_resume_checkpoint()
    if resume_path is not None:
        try:
            trainer.load_checkpoint(resume_path)
        except Exception:
            pass

    hub.publish(build_viewer_metadata(spec, mode="live").to_message())
    if trainer.state.rewards_history:
        hub.publish(
            {
                "type": "generation",
                "generation": trainer.state.generation,
                "mean_reward": trainer.state.mean_reward,
                "best_reward": trainer.state.best_reward,
                "top_rewards": trainer.top_rewards.tolist(),
                "rewards_history": trainer.state.rewards_history[-100:],
                "goal": list(trainer.state.goal_xyz),
            }
        )
    trainer.run_continuously(on_swarm_step=hub.publish, on_gen_done=hub.publish)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path, seed = _load_service_config()
    hub = BroadcastHub()
    hub.attach_loop(asyncio.get_running_loop())
    thread = threading.Thread(target=_trainer_thread, args=(hub, config_path, seed), daemon=True, name="live-trainer")
    app.state.hub = hub
    app.state.thread = thread
    thread.start()
    yield


app = FastAPI(title="Quadruped Live Training API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "quadruped-live", "status": "ok"}


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    hub: BroadcastHub = websocket.app.state.hub
    await hub.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        hub.disconnect(websocket)
    except Exception:
        hub.disconnect(websocket)
