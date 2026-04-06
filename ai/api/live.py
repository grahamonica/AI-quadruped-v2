"""Single-model viewer websocket service."""

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
from ai.runtime import viewer_checkpoint_candidates
from ai.trainer import ESTrainer, apply_runtime_spec

from .common import (
    BroadcastHub,
    build_viewer_metadata,
    current_policy_params,
    load_first_compatible_checkpoint,
    single_step_to_frame,
    viewer_reset_steps,
)


def _load_service_config() -> tuple[Path, int]:
    config_path = Path(os.environ.get("QUADRUPED_CONFIG", str(DEFAULT_CONFIG_PATH)))
    seed = int(os.environ.get("QUADRUPED_SEED", "42"))
    return config_path, seed


def _candidate_signatures(paths: tuple[Path, ...]) -> tuple[tuple[str, int], ...]:
    signatures: list[tuple[str, int]] = []
    for path in paths:
        try:
            signatures.append((str(path.resolve()), path.stat().st_mtime_ns))
        except OSError:
            continue
    return tuple(signatures)


def _viewer_thread(hub: BroadcastHub, config_path: Path, seed: int) -> None:
    spec = load_runtime_spec(config_path)
    apply_runtime_spec(spec)
    trainer = ESTrainer(seed=seed, spec=spec)

    hub.publish(build_viewer_metadata(spec, mode="viewer").to_message())

    candidate_state: tuple[tuple[str, int], ...] | None = None
    loaded_checkpoint: Path | None = None
    skipped = ()
    replay_steps = viewer_reset_steps(spec)

    while True:
        candidates = viewer_checkpoint_candidates()
        next_candidate_state = _candidate_signatures(candidates)
        if next_candidate_state != candidate_state:
            loaded_checkpoint, skipped = load_first_compatible_checkpoint(trainer, spec, candidates)
            candidate_state = next_candidate_state

        trainer._key, episode_key = jax.random.split(trainer._key)
        goal_xyz = trainer._random_goal()
        trainer.state.goal_xyz = tuple(float(value) for value in goal_xyz.tolist())
        hub.publish(
            {
                "type": "generation",
                "generation": trainer.state.generation,
                "mean_reward": trainer.state.mean_reward,
                "best_reward": trainer.state.best_reward,
                "top_rewards": trainer.top_rewards.tolist(),
                "rewards_history": trainer.state.rewards_history[-100:],
                "goal": list(trainer.state.goal_xyz),
                "checkpoint_loaded": str(loaded_checkpoint) if loaded_checkpoint is not None else None,
                "playback_only": True,
                "simulator_backend": spec.simulator.backend,
                "skipped_checkpoints": [
                    {"path": str(item.path), "reason": item.reason}
                    for item in skipped
                ],
            }
        )

        def _emit(step_message: dict) -> None:
            hub.publish(single_step_to_frame(step_message, generation=trainer.state.generation, spec=spec))

        trainer._run_logged_episode(current_policy_params(trainer), goal_xyz, episode_key, _emit, steps=replay_steps)
        time.sleep(0.2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path, seed = _load_service_config()
    hub = BroadcastHub()
    hub.attach_loop(asyncio.get_running_loop())
    thread = threading.Thread(target=_viewer_thread, args=(hub, config_path, seed), daemon=True, name="viewer")
    app.state.hub = hub
    app.state.thread = thread
    thread.start()
    yield


app = FastAPI(title="Quadruped Viewer API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "quadruped-viewer", "status": "ok"}


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
