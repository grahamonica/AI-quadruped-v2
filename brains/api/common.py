"""Shared websocket API helpers for live viewers."""

from __future__ import annotations

import asyncio
import base64
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import WebSocket

from brains.config import RuntimeSpec


@dataclass(frozen=True)
class ViewerMetadata:
    mode: str
    config_name: str
    control: dict[str, Any]
    terrain: dict[str, Any]
    robot: dict[str, Any]
    model: dict[str, Any]
    goal: dict[str, Any]
    training: dict[str, Any]
    simulator: dict[str, Any]

    def to_message(self) -> dict[str, Any]:
        return {
            "type": "metadata",
            "mode": self.mode,
            "config_name": self.config_name,
            "control": self.control,
            "terrain": self.terrain,
            "robot": self.robot,
            "model": self.model,
            "goal": self.goal,
            "training": self.training,
            "simulator": self.simulator,
        }


class BroadcastHub:
    """Caches the latest messages and fans them out to websocket clients."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._latest_messages: dict[str, dict[str, Any]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._dirty = False
        self._drain_scheduled = False

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> bool:
        await websocket.accept()
        with self._lock:
            latest_messages = list(self._latest_messages.values())
        for message in latest_messages:
            try:
                await websocket.send_json(message)
            except Exception:
                return False
        self._connections.add(websocket)
        return True

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    def publish(self, message: dict[str, Any]) -> None:
        message_type = str(message.get("type", "event"))
        should_schedule = False
        with self._lock:
            self._latest_messages[message_type] = message
            self._dirty = True
            should_schedule = self._loop is not None and not self._drain_scheduled
            if should_schedule:
                self._drain_scheduled = True
        if should_schedule:
            assert self._loop is not None
            self._loop.call_soon_threadsafe(lambda: asyncio.create_task(self._drain_latest()))

    async def _broadcast(self, message: dict[str, Any]) -> None:
        stale: list[WebSocket] = []
        for websocket in list(self._connections):
            try:
                await websocket.send_json(message)
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            self.disconnect(websocket)

    async def _drain_latest(self) -> None:
        while True:
            with self._lock:
                latest_messages = list(self._latest_messages.values())
                self._dirty = False
            for message in latest_messages:
                await self._broadcast(message)
            with self._lock:
                if not self._dirty:
                    self._drain_scheduled = False
                    return


def build_viewer_metadata(spec: RuntimeSpec, mode: str) -> ViewerMetadata:
    # Lazy import avoids pulling the full model registry (and JAX dependencies)
    # during API module import, so /healthz can come up quickly.
    from brains.models import list_model_definitions
    from brains.sim.mujoco_assets import LEG_NAMES, LEG_ROTATION_AXIS_BODY, load_mujoco_model, robot_geometry_from_model

    terrain = {
        "kind": spec.terrain.kind,
        "field_half_m": spec.terrain.field_half_m,
        "floor_height_m": spec.terrain.floor_height_m,
    }
    control = {
        "mode": spec.control.mode,
        "command_vocabulary": list(spec.control.command_vocabulary),
        "default_command_speed": spec.control.default_command_speed,
        "command_update_interval_s": spec.control.command_update_interval_s,
        "command_default_duration_s": spec.control.command_default_duration_s,
        "command_max_duration_s": spec.control.command_max_duration_s,
    }
    geometry = robot_geometry_from_model(load_mujoco_model())
    rotation_axes = [list(LEG_ROTATION_AXIS_BODY) for _ in LEG_NAMES]
    robot = {
        "body_length_m": geometry.body_size_m[0],
        "body_width_m": geometry.body_size_m[1],
        "body_height_m": geometry.body_size_m[2],
        "leg_length_m": geometry.leg_length_m,
        "leg_radius_m": geometry.leg_radius_m,
        "foot_radius_m": geometry.leg_radius_m,
        "mount_points_body": [list(point) for point in geometry.mount_points_body],
        "rotation_axes_body": rotation_axes,
        "leg_names": list(LEG_NAMES),
    }
    goal = {
        "strategy": spec.goals.strategy,
        "radius_m": spec.goals.radius_m,
        "height_m": spec.goals.height_m,
        "fixed_goal_xyz": list(spec.goals.fixed_goal_xyz) if spec.goals.fixed_goal_xyz is not None else None,
    }
    model = {
        "active": spec.model.type,
        "architecture": spec.model.architecture,
        "trainer": spec.model.trainer,
        "description": spec.model.description,
        "positional_encoding": spec.model.positional_encoding,
        "positional_encoding_gain": spec.model.positional_encoding_gain,
        "registered": [definition.to_dict() for definition in list_model_definitions()],
    }
    training = {
        "population_size": spec.training.population_size,
        "episode_s": spec.episode.episode_s,
        "selection_interval_s": spec.episode.selection_interval_s,
        "brain_dt_s": spec.episode.brain_dt_s,
    }
    simulator = {
        "backend": spec.simulator.backend,
        "render": spec.simulator.render,
        "deterministic_mode": spec.simulator.deterministic_mode,
    }
    return ViewerMetadata(
        mode=mode,
        config_name=spec.name,
        control=control,
        terrain=terrain,
        robot=robot,
        model=model,
        goal=goal,
        training=training,
        simulator=simulator,
    )


def current_policy_params(trainer: Any) -> np.ndarray:
    return np.asarray(trainer.params, dtype=np.float32)


def build_tracking_camera(
    mujoco_module: Any,
    torso_body_id: int,
    *,
    distance_m: float,
    azimuth_deg: float,
    elevation_deg: float,
) -> Any:
    camera = mujoco_module.MjvCamera()
    camera.type = mujoco_module.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = int(torso_body_id)
    camera.distance = float(distance_m)
    camera.azimuth = float(azimuth_deg)
    camera.elevation = float(elevation_deg)
    return camera


def encode_rgb_frame(rgb: np.ndarray) -> dict[str, Any]:
    encoded_rgb = base64.b64encode(np.ascontiguousarray(rgb).tobytes()).decode("ascii")
    return {
        "rgb": encoded_rgb,
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "channels": int(rgb.shape[2]),
    }
