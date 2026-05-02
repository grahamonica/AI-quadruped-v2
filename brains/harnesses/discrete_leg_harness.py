"""Discrete leg-step harness for local quadruped training.

Each step the policy chooses one of 81 actions: each of the four legs can stay,
step +45 deg, or step -45 deg during a single brain dt. The harness drives the
static-MJCF MuJoCo backend with a velocity command sized to land at the
requested joint angle within the brain dt window. Reward is the per-step
reduction in distance from the body to the goal. Episodes terminate early when
the robot tips over for too long or the body comes within
``goal_reached_radius_m`` of the goal (small but non-zero, to absorb numerical
offsets).

The harness is self-contained: observations, encoding, and reward live here so
notebooks can train tiny policies locally without going through the full
ESTrainer / spec-registry machinery.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from brains.config import DEFAULT_CONFIG_PATH, RuntimeSpec, load_runtime_spec


NUM_LEGS = 4
NUM_ACTIONS = 3 ** NUM_LEGS  # 81
LEG_DELTA_RAD = math.radians(45.0)
DEFAULT_BRAIN_DT_S = 0.20


def _build_action_table() -> np.ndarray:
    options = list(itertools.product((-1, 0, 1), repeat=NUM_LEGS))
    return np.asarray(options, dtype=np.float32)


ACTION_DELTAS = _build_action_table()  # shape (81, 4); entries in {-1, 0, +1}


@dataclass(frozen=True)
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]


@dataclass(frozen=True)
class EpisodeResult:
    total_reward: float
    steps: int
    reached_goal: bool
    tipped: bool
    final_distance_m: float
    initial_distance_m: float
    action_counts: np.ndarray
    frames: tuple[dict[str, Any], ...]


def positional_encoding(obs: jax.Array, *, gain: float) -> jax.Array:
    """Sinusoidal expansion of every observation feature.

    Output stacks ``[obs, gain*sin(pi*obs), gain*sin(2*pi*obs),
    gain*cos(pi*obs), gain*cos(2*pi*obs)]``, so encoded width is base * 5.
    Setting ``gain=0`` returns the raw observation unchanged.
    """

    base = jnp.asarray(obs, dtype=jnp.float32)
    if gain <= 0.0:
        return base
    g = jnp.float32(gain)
    sin_1 = jnp.sin(jnp.pi * base)
    sin_2 = jnp.sin(2.0 * jnp.pi * base)
    cos_1 = jnp.cos(jnp.pi * base)
    cos_2 = jnp.cos(2.0 * jnp.pi * base)
    return jnp.concatenate([base, g * sin_1, g * sin_2, g * cos_1, g * cos_2], axis=-1)


def positional_encoding_size(base_dim: int, *, gain: float) -> int:
    return int(base_dim * (5 if gain > 0.0 else 1))


class DiscreteLegHarness:
    """Gym-style harness for discrete 81-action leg control."""

    BASE_OBS_DIM = 10  # see _build_obs

    def __init__(
        self,
        spec: RuntimeSpec | str | Path | None = None,
        *,
        brain_dt_s: float = DEFAULT_BRAIN_DT_S,
        goal_reached_radius_m: float = 0.30,
        tip_kill_depth: float = 0.7,
        tip_kill_steps: int = 2,
        positional_encoding_gain: float = 0.35,
        goal_reached_bonus: float = 5.0,
        tipped_penalty: float = 0.0,
    ) -> None:
        if spec is None:
            base_spec = load_runtime_spec(DEFAULT_CONFIG_PATH)
        elif isinstance(spec, RuntimeSpec):
            base_spec = spec
        else:
            base_spec = load_runtime_spec(Path(spec))

        episode = replace(base_spec.episode, brain_dt_s=float(brain_dt_s))
        self.spec = replace(base_spec, episode=episode)
        self.spec.validate()

        self.brain_dt_s = float(brain_dt_s)
        self.goal_reached_radius_m = float(goal_reached_radius_m)
        self.tip_kill_depth = float(tip_kill_depth)
        self.tip_kill_steps = int(tip_kill_steps)
        self.positional_encoding_gain = float(positional_encoding_gain)
        self.goal_reached_bonus = float(goal_reached_bonus)
        self.tipped_penalty = float(tipped_penalty)
        self._side_band_half_rad = math.radians(float(self.spec.reward.side_tip_band_half_width_deg))

        self._backend = None
        self._data = None
        self._goal = np.zeros((3,), dtype=np.float32)
        self._initial_dist = 0.0
        self._prev_dist = 0.0
        self._tipped_streak = 0
        self._step_index = 0

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    @property
    def encoded_obs_dim(self) -> int:
        return positional_encoding_size(self.BASE_OBS_DIM, gain=self.positional_encoding_gain)

    @staticmethod
    def action_deltas(action_index: int) -> np.ndarray:
        return ACTION_DELTAS[int(action_index)].copy()

    def _make_backend(self):
        from brains.sim.mujoco_backend import MuJoCoBackend

        return MuJoCoBackend(self.spec)

    def reset(
        self,
        *,
        spawn_xy: np.ndarray | None = None,
        goal_xyz: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._backend is None:
            self._backend = self._make_backend()
        self._data = self._backend.reset_data(spawn_xy=spawn_xy)
        if goal_xyz is None:
            self._goal = np.asarray(
                [self.spec.goals.radius_m, 0.0, self.spec.goals.height_m],
                dtype=np.float32,
            )
        else:
            self._goal = np.asarray(goal_xyz, dtype=np.float32)
        com = self._backend.center_of_mass(self._data)
        self._initial_dist = float(np.linalg.norm(com[:2] - self._goal[:2]))
        self._prev_dist = self._initial_dist
        self._tipped_streak = 0
        self._step_index = 0
        return self._build_obs()

    def step(self, action_index: int) -> StepResult:
        if self._backend is None or self._data is None:
            raise RuntimeError("Call reset() before step().")
        deltas = ACTION_DELTAS[int(action_index)] * np.float32(LEG_DELTA_RAD)
        current_angles = self._backend.leg_angles(self._data)
        target_angles = (current_angles + deltas).astype(np.float32)
        max_v = float(self.spec.robot.max_motor_rad_s)
        target_velocity = np.clip(
            (target_angles - current_angles) / np.float32(self.brain_dt_s),
            -max_v,
            max_v,
        ).astype(np.float32)

        self._backend._advance(self._data, target_velocity)

        com = self._backend.center_of_mass(self._data)
        dist = float(np.linalg.norm(com[:2] - self._goal[:2]))
        progress = self._prev_dist - dist
        self._prev_dist = dist

        roll = float(self._backend.body_rotation(self._data)[0])
        wrapped = ((roll + math.pi) % (2.0 * math.pi)) - math.pi
        side_center_error = abs(abs(wrapped) - (math.pi / 2.0))
        depth = max(self._side_band_half_rad - side_center_error, 0.0) / max(
            self._side_band_half_rad, 1e-6
        )
        is_tipping = depth > self.tip_kill_depth
        self._tipped_streak = (self._tipped_streak + 1) if is_tipping else 0
        died = self._tipped_streak >= self.tip_kill_steps

        reached = dist <= self.goal_reached_radius_m

        reward = float(progress)
        if reached:
            reward += float(self.goal_reached_bonus)
        if died:
            reward -= float(self.tipped_penalty)

        self._step_index += 1
        info = {
            "distance_m": dist,
            "tipped": bool(died),
            "reached_goal": bool(reached),
            "step": int(self._step_index),
            "target_velocity_rad_s": target_velocity.tolist(),
            "deltas": ACTION_DELTAS[int(action_index)].tolist(),
        }
        done = bool(died or reached)
        return StepResult(self._build_obs(), reward, done, info)

    def _build_obs(self) -> np.ndarray:
        body_pos = self._backend.body_position(self._data)
        body_rot = self._backend.body_rotation(self._data)
        leg_angles = self._backend.leg_angles(self._data)
        field_half = max(float(self.spec.terrain.field_half_m), 1e-6)
        rel_xy = (self._goal[:2] - body_pos[:2]) / np.float32(field_half)
        dist_norm = np.float32(min(self._prev_dist / field_half, 1.0))
        yaw_norm = np.float32(body_rot[2] / math.pi)
        roll_norm = np.float32(body_rot[0] / math.pi)
        pitch_norm = np.float32(body_rot[1] / math.pi)
        # Continuous joints have no hard angle range; squash by pi-scale instead.
        legs_norm = np.tanh(leg_angles / np.float32(math.pi)).astype(np.float32)
        obs = np.concatenate(
            [
                rel_xy.astype(np.float32),                                  # 2
                np.asarray([dist_norm], dtype=np.float32),                  # 1
                np.asarray([yaw_norm, roll_norm, pitch_norm], dtype=np.float32),  # 3
                legs_norm,                                                  # 4
            ],
            axis=0,
        ).astype(np.float32)
        return obs

    def encode_obs(self, obs: np.ndarray | jax.Array) -> jax.Array:
        return positional_encoding(
            jnp.asarray(obs, dtype=jnp.float32),
            gain=self.positional_encoding_gain,
        )

    def rollout(
        self,
        choose_action: Callable[[np.ndarray, int], int],
        *,
        max_steps: int,
        spawn_xy: np.ndarray | None = None,
        goal_xyz: np.ndarray | None = None,
        record_frames: bool = False,
    ) -> EpisodeResult:
        obs = self.reset(spawn_xy=spawn_xy, goal_xyz=goal_xyz)
        total_reward = 0.0
        action_counts = np.zeros((NUM_ACTIONS,), dtype=np.int64)
        frames: list[dict[str, Any]] = []
        reached = False
        tipped = False
        steps_taken = 0

        for step_index in range(int(max_steps)):
            action = int(choose_action(obs, step_index))
            if not 0 <= action < NUM_ACTIONS:
                raise ValueError(f"choose_action returned {action!r}; expected int in [0, {NUM_ACTIONS}).")
            action_counts[action] += 1
            result = self.step(action)
            total_reward += float(result.reward)
            steps_taken = step_index + 1

            if record_frames:
                body_pos = self._backend.body_position(self._data)
                body_rot = self._backend.body_rotation(self._data)
                leg_angles = self._backend.leg_angles(self._data)
                foot_positions = self._backend.foot_positions(self._data)
                frames.append(
                    {
                        "step": step_index,
                        "time_s": float(self._step_index * self.brain_dt_s),
                        "action_index": action,
                        "deltas": ACTION_DELTAS[action].tolist(),
                        "reward": float(result.reward),
                        "distance_m": float(result.info["distance_m"]),
                        "body_pos": body_pos.tolist(),
                        "body_rot": body_rot.tolist(),
                        "leg_angles": leg_angles.tolist(),
                        "foot_positions": foot_positions.tolist(),
                        "goal": self._goal.tolist(),
                    }
                )

            obs = result.obs
            if result.done:
                tipped = bool(result.info.get("tipped", False))
                reached = bool(result.info.get("reached_goal", False))
                break

        return EpisodeResult(
            total_reward=float(total_reward),
            steps=int(steps_taken),
            reached_goal=bool(reached),
            tipped=bool(tipped),
            final_distance_m=float(self._prev_dist),
            initial_distance_m=float(self._initial_dist),
            action_counts=action_counts,
            frames=tuple(frames),
        )
