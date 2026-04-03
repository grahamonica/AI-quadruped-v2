"""Backend-selecting trainer exports."""

from __future__ import annotations

from typing import Any

from .jax_trainer import (
    EPISODE_S,
    JaxESTrainer,
    POP_SIZE,
    SINGLE_VIEW_EPISODE_S,
    TrainingState,
    apply_runtime_spec,
    current_environment_model,
    current_robot_model,
    current_runtime_spec,
)
from .mujoco_trainer import MuJoCoESTrainer


def ESTrainer(*args: Any, **kwargs: Any) -> JaxESTrainer | MuJoCoESTrainer:
    spec = kwargs.get("spec")
    if spec is None and len(args) >= 2:
        spec = args[1]
    resolved = current_runtime_spec() if spec is None else spec
    if resolved.simulator.backend == "mujoco":
        return MuJoCoESTrainer(*args, **kwargs)
    return JaxESTrainer(*args, **kwargs)


__all__ = [
    "EPISODE_S",
    "SINGLE_VIEW_EPISODE_S",
    "POP_SIZE",
    "TrainingState",
    "ESTrainer",
    "JaxESTrainer",
    "MuJoCoESTrainer",
    "apply_runtime_spec",
    "current_environment_model",
    "current_robot_model",
    "current_runtime_spec",
]
