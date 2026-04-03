"""Shared websocket API helpers for live viewers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from fastapi import WebSocket

from ai.config import RuntimeSpec
from ai.sim import single_step_to_swarm as translate_single_step_to_swarm
from quadruped import QuadrupedRobot, SimulationEnvironment


@dataclass(frozen=True)
class ViewerMetadata:
    mode: str
    config_name: str
    terrain: dict[str, Any]
    robot: dict[str, Any]
    goal: dict[str, Any]
    training: dict[str, Any]
    simulator: dict[str, Any]

    def to_message(self) -> dict[str, Any]:
        return {
            "type": "metadata",
            "mode": self.mode,
            "config_name": self.config_name,
            "terrain": self.terrain,
            "robot": self.robot,
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

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)
        for message in self._latest_messages.values():
            await websocket.send_json(message)

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    def publish(self, message: dict[str, Any]) -> None:
        message_type = str(message.get("type", "event"))
        self._latest_messages[message_type] = message
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self._loop)

    async def _broadcast(self, message: dict[str, Any]) -> None:
        stale: list[WebSocket] = []
        for websocket in list(self._connections):
            try:
                await websocket.send_json(message)
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            self.disconnect(websocket)


def build_viewer_metadata(spec: RuntimeSpec, mode: str) -> ViewerMetadata:
    robot_model = QuadrupedRobot.from_runtime_spec(spec)
    environment_model = SimulationEnvironment.from_runtime_spec(spec)
    terrain = {
        "kind": environment_model.terrain.kind,
        "field_half_m": environment_model.terrain.field_half_m,
        "center_half_m": environment_model.terrain.center_half_m,
        "step_count": environment_model.terrain.step_count,
        "step_width_m": environment_model.terrain.step_width_m,
        "step_height_m": environment_model.terrain.step_height_m,
        "floor_height_m": environment_model.terrain.floor_height_m,
    }
    robot = {
        "body_length_m": robot_model.body.length_m,
        "body_width_m": robot_model.body.width_m,
        "body_height_m": robot_model.body.height_m,
        "leg_length_m": robot_model.legs[0].length_m,
        "leg_names": list(robot_model.leg_names),
    }
    goal = {
        "strategy": environment_model.task.goal_strategy,
        "radius_m": environment_model.task.goal_radius_m,
        "height_m": environment_model.task.goal_height_m,
        "fixed_goal_xyz": list(environment_model.task.fixed_goal_xyz) if environment_model.task.fixed_goal_xyz is not None else None,
    }
    training = {
        "population_size": environment_model.training.population_size,
        "episode_s": environment_model.episode.episode_s,
        "selection_interval_s": environment_model.episode.selection_interval_s,
    }
    simulator = {
        "backend": spec.simulator.backend,
        "render": spec.simulator.render,
        "deterministic_mode": spec.simulator.deterministic_mode,
    }
    return ViewerMetadata(
        mode=mode,
        config_name=spec.name,
        terrain=terrain,
        robot=robot,
        goal=goal,
        training=training,
        simulator=simulator,
    )


def single_step_to_swarm(step_message: dict[str, Any], generation: int, spec: RuntimeSpec) -> dict[str, Any]:
    return translate_single_step_to_swarm(step_message, generation=generation, spec=spec)
