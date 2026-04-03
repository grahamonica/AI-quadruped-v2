"""Single-bot checkpoint viewer websocket service."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import jax

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from ai.config import DEFAULT_CONFIG_PATH, load_runtime_spec
from ai.runtime import resolve_single_view_checkpoint
from ai.trainer import ESTrainer, apply_runtime_spec

from .common import BroadcastHub, build_viewer_metadata, single_step_to_swarm


def _load_service_config() -> tuple[Path, int]:
    config_path = Path(os.environ.get("QUADRUPED_CONFIG", str(DEFAULT_CONFIG_PATH)))
    seed = int(os.environ.get("QUADRUPED_SEED", "42"))
    return config_path, seed


def _single_view_thread(hub: BroadcastHub, config_path: Path, seed: int) -> None:
    spec = load_runtime_spec(config_path)
    apply_runtime_spec(spec)
    trainer = ESTrainer(seed=seed, spec=spec)
    checkpoint_path = resolve_single_view_checkpoint()
    if checkpoint_path is not None:
        try:
            trainer.load_checkpoint(checkpoint_path)
        except Exception:
            checkpoint_path = None

    hub.publish(build_viewer_metadata(spec, mode="single").to_message())
    hub.publish(
        {
            "type": "generation",
            "generation": trainer.state.generation,
            "mean_reward": trainer.state.mean_reward,
            "best_reward": trainer.state.best_reward,
            "top_rewards": trainer.top_rewards.tolist(),
            "rewards_history": trainer.state.rewards_history[-100:],
            "goal": list(trainer.state.goal_xyz),
            "checkpoint_loaded": str(checkpoint_path) if checkpoint_path is not None else None,
        }
    )

    while True:
        trainer._key, episode_key = jax.random.split(trainer._key)
        goal_xyz = trainer._random_goal()
        trainer.state.goal_xyz = tuple(float(value) for value in goal_xyz.tolist())
        params = trainer.top_params[0] if trainer.top_params.size else trainer.checkpoint_dict()["params"]

        def _emit(step_message: dict) -> None:
            hub.publish(single_step_to_swarm(step_message, generation=trainer.state.generation, spec=spec))

        trainer._run_logged_episode(params, goal_xyz, episode_key, _emit, steps=int(trainer.spec.episode.single_view_episode_s / trainer.spec.episode.brain_dt_s))
        time.sleep(0.2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path, seed = _load_service_config()
    hub = BroadcastHub()
    hub.attach_loop(asyncio.get_running_loop())
    thread = threading.Thread(target=_single_view_thread, args=(hub, config_path, seed), daemon=True, name="single-view")
    app.state.hub = hub
    app.state.thread = thread
    thread.start()
    yield


app = FastAPI(title="Quadruped Single Viewer API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "quadruped-single", "status": "ok"}


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
