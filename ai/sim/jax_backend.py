"""JAX simulator wrapper behind the rollout backend interface."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import ai.jax_trainer as trainer_module
from ai.config import RuntimeSpec

from .interfaces import BackendCapabilities, LoggedStepCallback


class JaxSimBackend:
    """Expose the existing functional JAX simulator behind a stable interface."""

    def __init__(self, spec: RuntimeSpec) -> None:
        self.spec = trainer_module.apply_runtime_spec(spec)
        self.capabilities = BackendCapabilities(
            name="jax",
            batched_rollout_support=True,
            realtime_viewer_support=True,
            differentiable=False,
            deterministic_mode_supported=True,
            max_parallel_envs=max(1, self.spec.training.population_size),
        )

    def run_population(
        self,
        params_batch: np.ndarray | jax.Array,
        goal_xyz: np.ndarray | jax.Array,
        keys: np.ndarray | jax.Array,
        steps: int,
        spawn_xys: np.ndarray | jax.Array | None = None,
    ) -> np.ndarray:
        params_batch_jax = jnp.asarray(params_batch, dtype=jnp.float32)
        goal_xyz_jax = jnp.asarray(goal_xyz, dtype=jnp.float32)
        keys_jax = jnp.asarray(keys, dtype=jnp.uint32)
        spawn_xys_jax = None if spawn_xys is None else jnp.asarray(spawn_xys, dtype=jnp.float32)
        returns = trainer_module._run_episode_batch_flat(
            params_batch_jax,
            goal_xyz_jax,
            keys_jax,
            int(steps),
            spawn_xys_jax,
        )
        return np.asarray(returns, dtype=np.float32)

    def run_logged_episode(
        self,
        params_flat: np.ndarray | jax.Array,
        goal_xyz: np.ndarray | jax.Array,
        key: np.ndarray | jax.Array,
        on_step: LoggedStepCallback,
        steps: int,
        spawn_xy: np.ndarray | jax.Array | None = None,
    ) -> float:
        params_flat_jax = jnp.asarray(params_flat, dtype=jnp.float32)
        goal_xyz_jax = jnp.asarray(goal_xyz, dtype=jnp.float32)
        key_jax = jnp.asarray(key, dtype=jnp.uint32)
        spawn_xy_jax = None if spawn_xy is None else jnp.asarray(spawn_xy, dtype=jnp.float32)
        carry = trainer_module._episode_init(goal_xyz_jax, key_jax, spawn_xy_jax)
        for step_index in range(int(steps)):
            carry = trainer_module._episode_step_logged(params_flat_jax, carry, goal_xyz_jax)
            on_step(trainer_module._step_snapshot(carry, goal_xyz_jax, step_index, int(steps)))
        return float(carry.total_reward)

