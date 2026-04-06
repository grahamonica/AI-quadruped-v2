"""Evolution-strategy trainer that evaluates policies on the MuJoCo backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import ai.jax_trainer as trainer_module
from ai.config import RuntimeSpec, canonical_config_json
from ai.sim.mujoco_backend import MuJoCoBackend


class MuJoCoESTrainer:
    """ES optimizer that keeps the existing policy network and swaps the simulator backend."""

    def __init__(self, seed: int = 42, spec: RuntimeSpec | None = None) -> None:
        self.spec = trainer_module.apply_runtime_spec(spec or trainer_module.current_runtime_spec())
        self.seed = seed
        self.state = trainer_module.TrainingState()
        if self.spec.goals.strategy == "fixed" and self.spec.goals.fixed_goal_xyz is not None:
            self.state.goal_xyz = tuple(float(value) for value in self.spec.goals.fixed_goal_xyz)
        else:
            self.state.goal_xyz = (float(self.spec.goals.radius_m), 0.0, float(self.spec.goals.height_m))
        self._key = jax.random.PRNGKey(seed)
        self._key, init_key = jax.random.split(self._key)
        self._params = trainer_module._init_param_vector(init_key)
        self._top_params = np.zeros((0, trainer_module.PARAM_COUNT), dtype=np.float32)
        self._top_rewards = np.zeros((0,), dtype=np.float32)
        self._top_indices = np.zeros((0,), dtype=np.int32)
        self._top_generations = np.zeros((0,), dtype=np.int32)
        self._backend = MuJoCoBackend(self.spec)

    @property
    def backend(self) -> str:
        return self._backend.capabilities.name

    @property
    def device_summary(self) -> str:
        return f"MuJoCo 3.6.0 rollout + JAX {jax.default_backend()} policy"

    @property
    def param_count(self) -> int:
        return trainer_module.PARAM_COUNT

    @property
    def params(self) -> np.ndarray:
        return np.asarray(self._params, dtype=np.float32).copy()

    @property
    def top_params(self) -> np.ndarray:
        return self._top_params.copy()

    @property
    def top_rewards(self) -> np.ndarray:
        return self._top_rewards.copy()

    @property
    def top_indices(self) -> np.ndarray:
        return self._top_indices.copy()

    @property
    def top_generations(self) -> np.ndarray:
        return self._top_generations.copy()

    def _random_goal(self) -> jax.Array:
        if self.spec.goals.strategy == "fixed" and self.spec.goals.fixed_goal_xyz is not None:
            return jnp.asarray(self.spec.goals.fixed_goal_xyz, dtype=jnp.float32)
        self._key, angle_key = jax.random.split(self._key, 2)
        angle = jax.random.uniform(angle_key, (), minval=0.0, maxval=2.0 * jnp.pi)
        radius = float(self.spec.goals.radius_m)
        return jnp.array(
            [radius * jnp.cos(angle), radius * jnp.sin(angle), float(self.spec.goals.height_m)],
            dtype=jnp.float32,
        )

    def _evaluate_population(
        self,
        params_batch: np.ndarray,
        goal_xyz: jax.Array,
        eval_keys: jax.Array,
        spawn_xys: jax.Array,
        on_step: Any = None,
    ) -> np.ndarray:
        if on_step is None or params_batch.shape[0] == 0:
            return self._backend.run_population(
                params_batch,
                goal_xyz,
                eval_keys,
                int(trainer_module.EPISODE_S / trainer_module.BRAIN_DT),
                spawn_xys=spawn_xys,
            )

        returns = np.zeros((params_batch.shape[0],), dtype=np.float32)
        returns[0] = self._backend.run_logged_episode(
            params_batch[0],
            goal_xyz,
            eval_keys[0],
            on_step,
            int(trainer_module.EPISODE_S / trainer_module.BRAIN_DT),
            spawn_xy=spawn_xys[0],
        )
        if params_batch.shape[0] > 1:
            returns[1:] = self._backend.run_population(
                params_batch[1:],
                goal_xyz,
                eval_keys[1:],
                int(trainer_module.EPISODE_S / trainer_module.BRAIN_DT),
                spawn_xys=spawn_xys[1:],
            )
        return returns

    def _run_logged_episode(
        self,
        params_flat: jax.Array,
        goal_xyz: jax.Array,
        key: jax.Array,
        on_step: Any,
        steps: int | None = None,
    ) -> float:
        if steps is None:
            steps = int(self.spec.episode.single_view_episode_s / self.spec.episode.brain_dt_s)
        return self._backend.run_logged_episode(params_flat, goal_xyz, key, on_step, int(steps))

    def run_generation(self, on_step: Any = None, on_gen_done: Any = None) -> None:
        goal_xyz = self._random_goal()
        self.state.goal_xyz = tuple(float(v) for v in np.asarray(goal_xyz).tolist())

        population_size = int(self.spec.training.population_size)
        sigma = float(self.spec.training.sigma)
        learning_rate = float(self.spec.training.learning_rate)
        elite_count_target = int(self.spec.training.parent_elite_count)

        self._key, noise_key, eval_key, center_key, spawn_key = jax.random.split(self._key, 5)
        noise = jax.random.normal(noise_key, (population_size, trainer_module.PARAM_COUNT), dtype=jnp.float32) * jnp.float32(sigma)
        params_batch = np.asarray(self._params[None, :] + noise, dtype=np.float32)
        eval_keys = jax.random.split(eval_key, population_size)
        spawn_xys = trainer_module._sample_spawn_batch(spawn_key, population_size)

        returns_np = self._evaluate_population(
            params_batch,
            goal_xyz,
            eval_keys,
            spawn_xys,
            on_step=on_step,
        )

        returns_std = returns_np.std()
        if returns_std > 1e-8:
            normalized_returns = (returns_np - returns_np.mean()) / returns_std
        else:
            normalized_returns = returns_np - returns_np.mean()
        noise_np = np.asarray(noise, dtype=np.float32)
        gradient = (normalized_returns[:, None] * noise_np).sum(axis=0) / (population_size * sigma)
        self._params = self._params + jnp.float32(learning_rate) * jnp.asarray(gradient, dtype=jnp.float32)

        elite_count = min(elite_count_target, returns_np.shape[0])
        top_indices = np.argpartition(returns_np, -elite_count)[-elite_count:]
        top_indices = top_indices[np.argsort(returns_np[top_indices])[::-1]]
        top_params = params_batch[top_indices].astype(np.float32)
        top_rewards = returns_np[top_indices].astype(np.float32)
        top_generations = np.full((elite_count,), self.state.generation + 1, dtype=np.int32)

        if self._top_rewards.size > 0:
            combined_params = np.concatenate([self._top_params, top_params], axis=0)
            combined_rewards = np.concatenate([self._top_rewards, top_rewards], axis=0)
            combined_indices = np.concatenate([self._top_indices, top_indices.astype(np.int32)], axis=0)
            combined_generations = np.concatenate([self._top_generations, top_generations], axis=0)
        else:
            combined_params = top_params
            combined_rewards = top_rewards
            combined_indices = top_indices.astype(np.int32)
            combined_generations = top_generations

        leaderboard_count = min(elite_count_target, combined_rewards.shape[0])
        leaderboard_indices = np.argpartition(combined_rewards, -leaderboard_count)[-leaderboard_count:]
        leaderboard_indices = leaderboard_indices[np.argsort(combined_rewards[leaderboard_indices])[::-1]]
        self._top_params = combined_params[leaderboard_indices].astype(np.float32)
        self._top_rewards = combined_rewards[leaderboard_indices].astype(np.float32)
        self._top_indices = combined_indices[leaderboard_indices].astype(np.int32)
        self._top_generations = combined_generations[leaderboard_indices].astype(np.int32)

        center_return = float(
            self._backend.run_episode(
                np.asarray(self._params, dtype=np.float32),
                goal_xyz,
                center_key,
                int(trainer_module.EPISODE_S / trainer_module.BRAIN_DT),
            )
        )

        self.state.generation += 1
        self.state.mean_reward = float(returns_np.mean())
        self.state.episode_reward = float(self._top_rewards.mean()) if self._top_rewards.size else 0.0
        self.state.best_reward = max(self.state.best_reward, center_return)
        self.state.best_single_reward = max(
            self.state.best_single_reward,
            float(self._top_rewards[0]) if self._top_rewards.size else -1e9,
        )
        self.state.rewards_history.append(self.state.mean_reward)

        if on_gen_done is not None:
            on_gen_done(
                {
                    "type": "generation",
                    "generation": self.state.generation,
                    "mean_reward": self.state.mean_reward,
                    "best_reward": self.state.best_reward,
                    "top_rewards": self._top_rewards.tolist(),
                    "rewards_history": self.state.rewards_history[-100:],
                    "goal": list(self.state.goal_xyz),
                }
            )

    def checkpoint_dict(self) -> dict[str, Any]:
        return {
            "params": np.asarray(self._params, dtype=np.float32),
            "top_params": self._top_params.astype(np.float32),
            "top_rewards": self._top_rewards.astype(np.float32),
            "top_indices": self._top_indices.astype(np.int32),
            "top_generations": self._top_generations.astype(np.int32),
            "generation": np.int32(self.state.generation),
            "best_reward": np.float32(self.state.best_reward),
            "best_single_reward": np.float32(self.state.best_single_reward),
            "mean_reward": np.float32(self.state.mean_reward),
            "episode_reward": np.float32(self.state.episode_reward),
            "goal_xyz": np.asarray(self.state.goal_xyz, dtype=np.float32),
            "rewards_history": np.asarray(self.state.rewards_history, dtype=np.float32),
            "rng_key": np.asarray(self._key, dtype=np.uint32),
            "seed": np.int32(self.seed),
            "backend": np.array(self.backend),
            "compute_backend": np.array(jax.default_backend()),
            "config_json": np.array(canonical_config_json(self.spec)),
            "config_name": np.array(self.spec.name),
            "simulator_backend": np.array(self.spec.simulator.backend),
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(checkpoint_path, **self.checkpoint_dict())
        return checkpoint_path

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            if "config_json" in checkpoint.files:
                checkpoint_config_json = str(checkpoint["config_json"].item())
                if checkpoint_config_json != canonical_config_json(self.spec):
                    raise ValueError(
                        "Checkpoint config does not match the active runtime spec. "
                        "Use the same config file to resume training."
                    )
            if "simulator_backend" in checkpoint.files:
                checkpoint_backend = str(checkpoint["simulator_backend"].item())
                if checkpoint_backend != self.spec.simulator.backend:
                    raise ValueError(
                        f"Checkpoint simulator backend '{checkpoint_backend}' does not match active backend '{self.spec.simulator.backend}'."
                    )
            params = checkpoint["params"]
            if params.shape != (trainer_module.PARAM_COUNT,):
                raise ValueError(
                    f"Checkpoint parameter shape {params.shape} does not match active model shape {(trainer_module.PARAM_COUNT,)}."
                )
            self._params = jnp.asarray(checkpoint["params"], dtype=jnp.float32)
            if "top_params" in checkpoint.files:
                top_params = checkpoint["top_params"].astype(np.float32)
                if top_params.ndim != 2 or top_params.shape[1] != trainer_module.PARAM_COUNT:
                    raise ValueError(
                        f"Checkpoint top_params shape {top_params.shape} does not match active model width {trainer_module.PARAM_COUNT}."
                    )
                self._top_params = top_params
            else:
                self._top_params = np.zeros((0, trainer_module.PARAM_COUNT), dtype=np.float32)
            self._top_rewards = (
                checkpoint["top_rewards"].astype(np.float32)
                if "top_rewards" in checkpoint.files
                else np.zeros((0,), dtype=np.float32)
            )
            self._top_indices = (
                checkpoint["top_indices"].astype(np.int32)
                if "top_indices" in checkpoint.files
                else np.zeros((0,), dtype=np.int32)
            )
            self.state.generation = int(checkpoint["generation"])
            if "top_generations" in checkpoint.files:
                self._top_generations = checkpoint["top_generations"].astype(np.int32)
            else:
                self._top_generations = np.full((self._top_rewards.shape[0],), self.state.generation, dtype=np.int32)
            self.state.best_reward = float(checkpoint["best_reward"])
            self.state.best_single_reward = (
                float(checkpoint["best_single_reward"])
                if "best_single_reward" in checkpoint.files
                else float(checkpoint["best_reward"])
            )
            self.state.mean_reward = float(checkpoint["mean_reward"])
            self.state.episode_reward = float(checkpoint["episode_reward"])
            self.state.goal_xyz = tuple(float(v) for v in checkpoint["goal_xyz"].tolist())
            self.state.rewards_history = [float(v) for v in checkpoint["rewards_history"].tolist()]
            self._key = jnp.asarray(checkpoint["rng_key"], dtype=jnp.uint32)
