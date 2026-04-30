"""Backend-agnostic terrain and viewer translation helpers."""

from __future__ import annotations

from typing import Any

from ai.config import RuntimeSpec


def terrain_height_at(spec: RuntimeSpec, xy: tuple[float, float] | list[float]) -> float:
    x_value = float(xy[0])
    y_value = float(xy[1])
    terrain = spec.terrain
    if terrain.kind == "flat":
        return float(terrain.floor_height_m)
    radius = max(abs(x_value), abs(y_value))
    if radius <= terrain.center_half_m:
        return float(terrain.floor_height_m)
    raw_step = (radius - terrain.center_half_m) / terrain.step_width_m
    step_index = min(max(int(raw_step + 1e-6), 0), terrain.step_count)
    return float(terrain.floor_height_m + step_index * terrain.step_height_m)


def step_level_at(spec: RuntimeSpec, xy: tuple[float, float] | list[float]) -> int:
    x_value = float(xy[0])
    y_value = float(xy[1])
    terrain = spec.terrain
    if terrain.kind == "flat":
        return 0
    radius = max(abs(x_value), abs(y_value))
    if radius <= terrain.center_half_m:
        return 0
    raw_step = (radius - terrain.center_half_m) / terrain.step_width_m
    return int(min(max(int(raw_step + 1e-6), 0), terrain.step_count))


def single_step_to_viewer_frame(step_message: dict[str, Any], generation: int, spec: RuntimeSpec) -> dict[str, Any]:
    com = step_message["com"]
    body_pos = step_message["body"]["pos"]
    body_rot = step_message["body"]["rot"]
    body = step_message.get("body", {})
    legs = step_message.get("legs", [])
    leg_angles = [float(leg.get("angle_rad", 0.0)) for leg in legs]
    level = int(step_message.get("level", step_level_at(spec, com[:2])))
    return {
        "type": "frame",
        "pos": [float(value) for value in body_pos],
        "rot": [float(value) for value in body_rot],
        "leg": leg_angles,
        "body": body,
        "legs": legs,
        "level": [level],
        "n": 1,
        "gen": generation,
        "step": int(step_message.get("step", 0)),
        "total_steps": int(step_message.get("total_steps", 0)),
        "time_s": float(step_message.get("time_s", 0.0)),
        "goal": [float(value) for value in step_message.get("goal", [])],
    }
