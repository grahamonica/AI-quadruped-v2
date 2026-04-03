"""Shared simulator backend contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import jax
import numpy as np

from ai.config import RuntimeSpec


LoggedStepCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    batched_rollout_support: bool
    realtime_viewer_support: bool
    differentiable: bool
    deterministic_mode_supported: bool
    max_parallel_envs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "batched_rollout_support": self.batched_rollout_support,
            "realtime_viewer_support": self.realtime_viewer_support,
            "differentiable": self.differentiable,
            "deterministic_mode_supported": self.deterministic_mode_supported,
            "max_parallel_envs": self.max_parallel_envs,
        }


class RolloutBackend(Protocol):
    spec: RuntimeSpec
    capabilities: BackendCapabilities

    def run_population(
        self,
        params_batch: np.ndarray | jax.Array,
        goal_xyz: np.ndarray | jax.Array,
        keys: np.ndarray | jax.Array,
        steps: int,
        spawn_xys: np.ndarray | jax.Array | None = None,
    ) -> np.ndarray: ...

    def run_logged_episode(
        self,
        params_flat: np.ndarray | jax.Array,
        goal_xyz: np.ndarray | jax.Array,
        key: np.ndarray | jax.Array,
        on_step: LoggedStepCallback,
        steps: int,
        spawn_xy: np.ndarray | jax.Array | None = None,
    ) -> float: ...

