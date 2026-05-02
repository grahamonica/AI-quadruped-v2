"""Native MuJoCo viewer for the playback pipeline."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from brains.config import DEFAULT_CONFIG_PATH
from brains.runtime.model_store import discover_model_artifacts
from brains.runtime.playback import load_playback_for_selection


GLFW_KEY_R = 82
GLFW_KEY_G = 71
GLFW_KEY_RIGHT = 262
GLFW_KEY_LEFT = 263
GLFW_KEY_DOWN = 264
GLFW_KEY_UP = 265

GOAL_NUDGE_M = 0.25
DEFAULT_CHECKPOINT_ROOT = Path("checkpoints")
CONTINUOUS_VIEWER_STEPS = 2_000_000_000

CAMERA_DISTANCE_M = 3.8
CAMERA_AZIMUTH_DEG = 135.0
CAMERA_ELEVATION_DEG = -22.0


class _EpisodeRestart(Exception):
    """Raised inside on_step to break out of the current episode."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native MuJoCo viewer for trained models.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", default=None, help="Model id to play back (omit for static scene).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--goal", default=None, help="Goal x,y override.")
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--list", action="store_true", help="List available models and exit.")
    return parser.parse_args()


def _parse_goal_arg(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    parts = [chunk.strip() for chunk in text.replace(";", ",").split(",") if chunk.strip()]
    if len(parts) < 2:
        raise SystemExit(f"--goal expects 'x,y' (got {text!r}).")
    return float(parts[0]), float(parts[1])


def _list_models(checkpoint_root: Path) -> None:
    artifacts = discover_model_artifacts(checkpoint_root)
    if not artifacts:
        print(f"No models found under {checkpoint_root}.")
        return
    for artifact in artifacts:
        message = artifact.to_message()
        model_id = str(message.get("id", "?"))
        generation = message.get("generation", 0)
        best_reward = message.get("best_reward")
        best_text = f"{best_reward:.2f}" if isinstance(best_reward, (int, float)) else "n/a"
        print(f"{model_id:40s}  gen={generation}  best_reward={best_text}")


def _resolve_goal(playback, override_xy: tuple[float, float] | None) -> np.ndarray:
    height_m = float(playback.spec.goals.height_m)
    if override_xy is not None:
        return np.asarray([override_xy[0], override_xy[1], height_m], dtype=np.float32)
    goal = np.asarray(playback.random_goal(), dtype=np.float32)
    if goal.shape[-1] == 2:
        return np.asarray([goal[0], goal[1], height_m], dtype=np.float32)
    return goal


def main() -> None:
    args = _parse_args()
    if args.list:
        _list_models(args.checkpoints)
        return

    initial_goal_xy = _parse_goal_arg(args.goal)
    result = load_playback_for_selection(
        config_path=args.config,
        seed=args.seed,
        selected_model_id=args.model,
        checkpoint_root=args.checkpoints,
    )
    playback = result.playback
    for entry in result.skipped:
        print(f"skipped: {entry.get('path')}: {entry.get('reason')}")

    backend = playback.rollout_backend
    model = backend.model
    data = backend.make_data()
    spec = playback.spec
    brain_dt_s = float(spec.episode.brain_dt_s)
    is_static = bool(getattr(playback, "static", False))

    state_lock = threading.Lock()
    goal_xy: tuple[float, float] | None = initial_goal_xy
    restart_requested = False

    def _key_callback(keycode: int) -> None:
        nonlocal goal_xy, restart_requested
        with state_lock:
            current = goal_xy
            if current is None:
                fallback = np.asarray(playback.random_goal(), dtype=np.float32)
                current = (float(fallback[0]), float(fallback[1])) if fallback.shape[-1] >= 2 else (0.0, 0.0)

            if keycode == GLFW_KEY_R:
                restart_requested = True
                print("[viewer] restart episode", flush=True)
                return
            if keycode == GLFW_KEY_G:
                print(f"[viewer] goal x,y = ({current[0]:.2f}, {current[1]:.2f})", flush=True)
                return
            if keycode == GLFW_KEY_RIGHT:
                goal_xy = (current[0] + GOAL_NUDGE_M, current[1])
            elif keycode == GLFW_KEY_LEFT:
                goal_xy = (current[0] - GOAL_NUDGE_M, current[1])
            elif keycode == GLFW_KEY_UP:
                goal_xy = (current[0], current[1] + GOAL_NUDGE_M)
            elif keycode == GLFW_KEY_DOWN:
                goal_xy = (current[0], current[1] - GOAL_NUDGE_M)
            else:
                return
            restart_requested = True
            print(f"[viewer] goal -> ({goal_xy[0]:.2f}, {goal_xy[1]:.2f}); restarting", flush=True)

    print(f"Loaded model: {args.model or '(static scene)'}")
    if result.loaded_checkpoint is not None:
        print(f"Checkpoint: {result.loaded_checkpoint}")
    print("Keys: arrows nudge goal | R restart | G print goal | close window to exit")

    with mujoco.viewer.launch_passive(model, data, key_callback=_key_callback) as viewer:
        torso_id = getattr(backend, "_torso_body_id", None)
        if torso_id is not None:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = int(torso_id)
        viewer.cam.distance = CAMERA_DISTANCE_M
        viewer.cam.azimuth = CAMERA_AZIMUTH_DEG
        viewer.cam.elevation = CAMERA_ELEVATION_DEG

        def _emit(step_message: dict) -> None:
            nonlocal restart_requested
            with state_lock:
                if restart_requested:
                    restart_requested = False
                    raise _EpisodeRestart()
            qpos = np.asarray(step_message["qpos"], dtype=np.float64)
            qvel = np.asarray(step_message["qvel"], dtype=np.float64)
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            if data.act.size:
                data.act[:] = 0.0
            data.time = float(step_message.get("time_s", 0.0))
            mujoco.mj_forward(model, data)
            viewer.sync()
            if not viewer.is_running():
                raise _EpisodeRestart()
            time.sleep(brain_dt_s)

        while viewer.is_running():
            with state_lock:
                current_goal_xy = goal_xy
            goal_xyz = _resolve_goal(playback, current_goal_xy)
            playback.state.goal_xyz = (float(goal_xyz[0]), float(goal_xyz[1]), float(goal_xyz[2]))
            episode_key = playback.advance_key()
            spawn_xy = None if is_static else np.zeros((2,), dtype=np.float32)
            replay_steps = 1 if is_static else CONTINUOUS_VIEWER_STEPS
            try:
                playback.run_logged_episode(
                    goal_xyz, episode_key, _emit, steps=replay_steps, spawn_xy=spawn_xy
                )
            except _EpisodeRestart:
                continue
            except Exception as exc:
                print(f"[viewer] episode error: {exc}", flush=True)
                time.sleep(0.5)
                continue

            if is_static:
                while viewer.is_running():
                    with state_lock:
                        if restart_requested:
                            restart_requested = False
                            break
                    viewer.sync()
                    time.sleep(0.05)


if __name__ == "__main__":
    main()
