"""Episode runner and ES trainer for the SNN brain.

Uses OpenAI-ES style (Salimans et al. 2017):
  Each generation perturbs weights with Gaussian noise, evaluates N candidates,
  estimates gradient from returns, updates parameters.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ai.brain import SNNBrain, N_IN
from quadruped.environment import QuadrupedEnvironment
from quadruped.robot import Quadruped

# ── Training hyper-parameters ──────────────────────────────────────────────
EPISODE_S   = 30.0    # simulated seconds per episode — plenty of time to explore
BRAIN_DT    = 0.010   # brain & motor update interval (10 ms = 100 Hz)
MOTOR_SCALE = 6.0     # map [-1,1] → rad/s
GOAL_HEIGHT_M = 0.16  # = leg_length_m
FIELD_HALF  = 15.0    # field half-extent (m) — large open space
POP_SIZE    = 8       # ES population per generation
SIGMA       = 0.05    # ES noise std
LR          = 0.03    # ES learning rate
MAX_MOTOR_RAD_S = 8.0


@dataclass
class TrainingState:
    generation: int = 0
    best_reward: float = -1e9
    mean_reward: float = 0.0
    episode_reward: float = 0.0
    goal_xyz: tuple[float, float, float] = (1.0, 0.0, GOAL_HEIGHT_M)
    robot_state: dict[str, Any] = field(default_factory=dict)
    rewards_history: list[float] = field(default_factory=list)


def _make_env() -> QuadrupedEnvironment:
    robot = Quadruped.create_kt2_style()
    return QuadrupedEnvironment(robot=robot)


def _build_obs(env: QuadrupedEnvironment, goal: tuple[float, float, float]) -> np.ndarray:
    """Assemble the 48-element observation vector."""
    robot = env.robot
    body = robot.body

    # goal coords
    v_goal = list(goal)

    # total COM
    com = env.center_of_mass_xyz_m()
    v_com = list(com)

    # body COM (= body position)
    v_body_com = list(body.position_xyz_m)

    # leg foot world positions (4×3 = 12)
    v_feet: list[float] = []
    for leg in robot.legs:
        state = env.leg_force_states.get(leg.name)
        if state is not None:
            v_feet.extend(state.foot_position_xyz_m)
        else:
            # fallback: compute directly
            mount = env._body_point_world(leg.mount_point_xyz_m)
            foot_offset = env._body_vector_world(leg.foot_offset_from_mount_m())
            v_feet.extend([mount[0] + foot_offset[0],
                            mount[1] + foot_offset[1],
                            mount[2] + foot_offset[2]])

    # leg COM world positions (4×3 = 12)
    v_leg_com: list[float] = []
    for leg in robot.legs:
        mount = env._body_point_world(leg.mount_point_xyz_m)
        com_off = env._body_vector_world(leg.com_offset_from_mount_m())
        v_leg_com.extend([mount[0] + com_off[0],
                           mount[1] + com_off[1],
                           mount[2] + com_off[2]])

    # body IMU angles (roll, pitch, yaw)
    v_body_imu = list(body.imu.rotation_xyz_rad)

    # leg IMU angles (4×3 = 12)
    v_leg_imu: list[float] = []
    for leg in robot.legs:
        v_leg_imu.extend(leg.imu.rotation_xyz_rad)

    obs = np.array(
        v_goal + v_com + v_body_com + v_feet + v_leg_com + v_body_imu + v_leg_imu,
        dtype=np.float32,
    )
    assert obs.shape == (N_IN,), f"obs shape {obs.shape} != {N_IN}"

    # Normalize: scale positions by 1/FIELD_HALF, angles already in rad (≲π)
    obs[:33] /= FIELD_HALF
    obs[33:] /= math.pi
    return obs


def _run_episode(
    brain: SNNBrain,
    goal: tuple[float, float, float],
    on_step: Any = None,
) -> float:
    """Run one episode; return total reward. Calls on_step(state_dict) each physics step."""
    env = _make_env()
    brain.reset()
    robot = env.robot
    leg_names = [leg.name for leg in robot.legs]

    goal_xy = np.array([goal[0], goal[1]])
    prev_dist = None
    total_reward = 0.0

    steps = int(EPISODE_S / BRAIN_DT)        # 400 steps at 10 ms
    report_every = max(1, steps // 200)      # ~200 frontend updates

    for step_i in range(steps):
        obs = _build_obs(env, goal)
        motor_cmds = brain.step(obs, BRAIN_DT)   # (4,)

        for i, name in enumerate(leg_names):
            vel = float(np.clip(motor_cmds[i] * MOTOR_SCALE, -MAX_MOTOR_RAD_S, MAX_MOTOR_RAD_S))
            robot.set_leg_motor_velocity(name, vel)

        env.advance(BRAIN_DT)

        com = env.center_of_mass_xyz_m()
        com_xy = np.array([com[0], com[1]])
        dist = float(np.linalg.norm(com_xy - goal_xy))

        if prev_dist is None:
            prev_dist = dist

        # Dense reward: negative distance each step — maximising this = reaching goal.
        # Using -dist directly avoids any sign confusion; ES normalises across
        # candidates so the absolute scale doesn't matter.
        reward = -dist

        # Extra bonus for closing distance this step (shapes gradient toward goal)
        reward += (prev_dist - dist) * 5.0

        total_reward += reward
        prev_dist = dist

        if on_step is not None and step_i % report_every == 0:
            on_step(_snapshot(env, goal, step_i, steps, total_reward))

    return total_reward


def _snapshot(
    env: QuadrupedEnvironment,
    goal: tuple[float, float, float],
    step: int,
    total_steps: int,
    reward_so_far: float,
) -> dict[str, Any]:
    robot = env.robot
    body = robot.body
    legs_data = []
    for leg in robot.legs:
        state = env.leg_force_states.get(leg.name)
        foot = state.foot_position_xyz_m if state else (0.0, 0.0, 0.0)
        mount_w = env._body_point_world(leg.mount_point_xyz_m)
        com_off = env._body_vector_world(leg.com_offset_from_mount_m())
        leg_com = (mount_w[0] + com_off[0], mount_w[1] + com_off[1], mount_w[2] + com_off[2])
        legs_data.append({
            "name": leg.name,
            "mount": list(mount_w),
            "foot": list(foot),
            "com": list(leg_com),
            "angle_rad": leg.angle_rad,
            "contact_mode": state.contact_mode if state else "airborne",
        })
    com = env.center_of_mass_xyz_m()
    return {
        "type": "step",
        "step": step,
        "total_steps": total_steps,
        "reward": reward_so_far,
        "goal": list(goal),
        "body": {
            "pos": list(body.position_xyz_m),
            "rot": list(body.rotation_xyz_rad),
            "com": list(body.position_xyz_m),
        },
        "com": list(com),
        "legs": legs_data,
        "time_s": env.time_s,
    }


class ESTrainer:
    """OpenAI-ES trainer."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.brain = SNNBrain(seed=seed)
        self.state = TrainingState()
        self._params = self.brain.get_params()

    def _random_goal(self) -> tuple[float, float, float]:
        angle = self.rng.uniform(0, 2 * math.pi)
        radius = self.rng.uniform(0.5, FIELD_HALF * 0.8)
        return (radius * math.cos(angle), radius * math.sin(angle), GOAL_HEIGHT_M)

    def run_generation(self, on_step: Any = None, on_gen_done: Any = None) -> None:
        goal = self._random_goal()
        self.state.goal_xyz = goal

        noise = self.rng.normal(0.0, SIGMA, (POP_SIZE, len(self._params))).astype(np.float32)
        returns = np.zeros(POP_SIZE, dtype=np.float32)

        for k in range(POP_SIZE):
            candidate = SNNBrain()
            candidate.set_params(self._params + noise[k])
            # Only stream steps for the first candidate of each generation
            cb = on_step if k == 0 else None
            returns[k] = _run_episode(candidate, goal, cb)

        # Normalize returns
        std = returns.std()
        if std > 1e-6:
            normalized = (returns - returns.mean()) / std
        else:
            normalized = np.zeros_like(returns)

        # Gradient estimate
        grad = (noise.T @ normalized) / (POP_SIZE * SIGMA)
        self._params = self._params + LR * grad

        self.brain.set_params(self._params)
        self.state.generation += 1
        self.state.mean_reward = float(returns.mean())
        self.state.best_reward = max(self.state.best_reward, float(returns.max()))
        self.state.rewards_history.append(self.state.mean_reward)

        if on_gen_done is not None:
            on_gen_done({
                "type": "generation",
                "generation": self.state.generation,
                "mean_reward": self.state.mean_reward,
                "best_reward": self.state.best_reward,
                "rewards_history": self.state.rewards_history[-100:],
                "goal": list(goal),
            })
