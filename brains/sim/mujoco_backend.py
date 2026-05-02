"""MuJoCo simulator backend for single-environment rollouts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

import brains.jax_trainer as trainer_module
from brains.config import RuntimeSpec

from .action_layer import ActionProjector
from .mujoco_assets import (
    ACTUATOR_NAMES,
    FOOT_SITE_NAMES,
    LEG_BODY_NAMES,
    LEG_JOINT_NAMES,
    LEG_NAMES,
    TORSO_BODY_NAME,
    load_mujoco_model,
    robot_geometry_from_model,
)


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

@dataclass
class _RolloutMetrics:
    total_reward: float
    prev_dist: float
    noise_scale: float
    fast_closing_rate: float
    slow_closing_rate: float
    closing_rate: float
    progress_drop_ratio: float
    prev_side_tip_depth_norm: float
    goal_reached: bool
    max_step_reached: float
    tipped_time: float
    dead: bool


class MuJoCoBackend:
    """Higher-fidelity simulator wrapper that preserves the existing rollout contract."""

    def __init__(self, spec: RuntimeSpec) -> None:
        self.spec = trainer_module.apply_runtime_spec(spec)
        self.leg_names = LEG_NAMES
        self.leg_count = len(self.leg_names)
        self.model = load_mujoco_model()
        self.geometry = robot_geometry_from_model(self.model)
        self.body_half_extents_m = self.geometry.body_half_extents_m
        self.body_height_m = float(self.geometry.body_size_m[2])
        self.leg_length_m = float(self.geometry.leg_length_m)
        self.foot_radius_m = float(self.geometry.leg_radius_m)
        self.floor_height_m = float(self.geometry.floor_height_m)
        self.contact_margin_m = float(np.max(self.model.geom_margin)) if self.model.ngeom else 0.0
        self.capabilities = BackendCapabilities(
            name="mujoco",
            batched_rollout_support=False,
            realtime_viewer_support=True,
            differentiable=False,
            deterministic_mode_supported=True,
            max_parallel_envs=1,
        )
        control_ratio = self.spec.episode.brain_dt_s / float(self.model.opt.timestep)
        rounded = round(control_ratio)
        if rounded <= 0 or abs(control_ratio - rounded) > 1e-8:
            raise ValueError("MuJoCo XML timestep must divide episode.brain_dt_s exactly for stable control updates.")
        self.control_substeps = int(rounded)
        self._torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY_NAME)
        self._leg_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in LEG_BODY_NAMES
        ]
        self._foot_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in FOOT_SITE_NAMES
        ]
        self._joint_qpos_adr = [
            int(self.model.joint(name).qposadr[0])
            for name in LEG_JOINT_NAMES
        ]
        self._joint_dof_adr = [
            int(self.model.joint(name).dofadr[0])
            for name in LEG_JOINT_NAMES
        ]
        self._actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ACTUATOR_NAMES
        ]
        self._action_projector = ActionProjector(
            mode=self.spec.control.mode,
            command_vocabulary=self.spec.control.command_vocabulary,
            default_command_speed=self.spec.control.default_command_speed,
            command_update_interval_s=self.spec.control.command_update_interval_s,
            command_default_duration_s=self.spec.control.command_default_duration_s,
            command_max_duration_s=self.spec.control.command_max_duration_s,
        )

    def make_data(self) -> mujoco.MjData:
        return mujoco.MjData(self.model)

    def reset_data(self, spawn_xy: np.ndarray | jax.Array | None = None) -> mujoco.MjData:
        data = self.make_data()
        self._reset_into(data, spawn_xy=spawn_xy)
        return data

    def _reset_into(self, data: mujoco.MjData, spawn_xy: np.ndarray | jax.Array | None = None) -> None:
        self._action_projector.reset()
        if spawn_xy is None:
            spawn_xy_np = np.zeros((2,), dtype=np.float32)
        else:
            spawn_xy_np = np.asarray(spawn_xy, dtype=np.float32)
        spawn_floor = self.floor_height_at(spawn_xy_np)
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        if data.act.size:
            data.act[:] = 0.0
        data.ctrl[:] = 0.0
        data.qpos[0] = float(spawn_xy_np[0])
        data.qpos[1] = float(spawn_xy_np[1])
        data.qpos[2] = float(spawn_floor + self.geometry.reset_height_m)
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        for address in self._joint_qpos_adr:
            data.qpos[address] = 0.0
        mujoco.mj_forward(self.model, data)

    def _rotmat_to_euler_xyz(self, rotation_matrix: np.ndarray) -> np.ndarray:
        pitch = math.asin(float(-rotation_matrix[2, 0]))
        cos_pitch = math.cos(pitch)
        if abs(cos_pitch) > 1e-6:
            roll = math.atan2(float(rotation_matrix[2, 1]), float(rotation_matrix[2, 2]))
            yaw = math.atan2(float(rotation_matrix[1, 0]), float(rotation_matrix[0, 0]))
        else:
            roll = math.atan2(float(-rotation_matrix[1, 2]), float(rotation_matrix[1, 1]))
            yaw = 0.0
        return np.asarray([roll, pitch, yaw], dtype=np.float32)

    def body_position(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(data.xpos[self._torso_body_id], dtype=np.float32).copy()

    def body_rotation(self, data: mujoco.MjData) -> np.ndarray:
        rotation_matrix = np.asarray(data.xmat[self._torso_body_id], dtype=np.float32).reshape(3, 3)
        return self._rotmat_to_euler_xyz(rotation_matrix)

    def leg_angles(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray([data.qpos[address] for address in self._joint_qpos_adr], dtype=np.float32)

    def leg_velocities(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray([data.qvel[address] for address in self._joint_dof_adr], dtype=np.float32)

    def foot_positions(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(data.site_xpos[self._foot_site_ids], dtype=np.float32).copy()

    def leg_com_positions(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(data.xipos[self._leg_body_ids], dtype=np.float32).copy()

    def center_of_mass(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(data.subtree_com[self._torso_body_id], dtype=np.float32).copy()

    def build_obs(self, data: mujoco.MjData, goal_xyz: np.ndarray | jax.Array) -> jax.Array:
        goal_np = np.asarray(goal_xyz, dtype=np.float32)
        com_np = self.center_of_mass(data)
        body_pos_np = self.body_position(data)
        body_rot_np = self.body_rotation(data)
        foot_np = self.foot_positions(data)
        leg_com_np = self.leg_com_positions(data)
        leg_angles_np = self.leg_angles(data)
        leg_imu = np.stack(
            [
                np.full((self.leg_count,), body_rot_np[0], dtype=np.float32),
                leg_angles_np,
                np.full((self.leg_count,), body_rot_np[2], dtype=np.float32),
            ],
            axis=1,
        )
        obs = np.concatenate(
            [
                goal_np,
                com_np,
                body_pos_np,
                foot_np.reshape(-1),
                leg_com_np.reshape(-1),
                body_rot_np,
                leg_imu.reshape(-1),
            ],
            axis=0,
        ).astype(np.float32)
        obs[:33] /= np.float32(trainer_module.FIELD_HALF)
        obs[33:] /= np.float32(math.pi)
        obs = trainer_module.apply_positional_encoding_np(obs)
        return jnp.asarray(obs, dtype=jnp.float32)

    def initial_metrics(self, data: mujoco.MjData, goal_xyz: np.ndarray | jax.Array) -> _RolloutMetrics:
        com = self.center_of_mass(data)
        goal_np = np.asarray(goal_xyz, dtype=np.float32)
        initial_dist = float(np.linalg.norm(com[:2] - goal_np[:2]))
        body_rot = self.body_rotation(data)
        initial_roll = self._wrap_angle_pi(float(body_rot[0]))
        return _RolloutMetrics(
            total_reward=0.0,
            prev_dist=initial_dist,
            noise_scale=float(trainer_module.DEFAULT_MOTOR_NOISE_SCALE),
            fast_closing_rate=0.0,
            slow_closing_rate=0.0,
            closing_rate=0.0,
            progress_drop_ratio=0.0,
            prev_side_tip_depth_norm=self._side_tip_depth_norm(initial_roll),
            goal_reached=False,
            max_step_reached=0.0,
            tipped_time=0.0,
            dead=False,
        )

    def _wrap_angle_pi(self, angle_rad: float) -> float:
        return ((angle_rad + math.pi) % (2.0 * math.pi)) - math.pi

    def _side_tip_depth_norm(self, imu_roll_rad: float) -> float:
        abs_roll_rad = abs(self._wrap_angle_pi(imu_roll_rad))
        side_center_error_rad = abs(abs_roll_rad - (math.pi / 2.0))
        side_band_depth_rad = max(trainer_module.SIDE_TIP_BAND_HALF_WIDTH_RAD - side_center_error_rad, 0.0)
        return side_band_depth_rad / max(trainer_module.SIDE_TIP_BAND_HALF_WIDTH_RAD, 1e-6)

    def _ema_alpha(self, dt_s: float, tau_s: float) -> float:
        return float(1.0 - math.exp(-dt_s / max(tau_s, 1e-6)))

    def _step_metrics(
        self,
        data: mujoco.MjData,
        metrics: _RolloutMetrics,
        goal_xyz: np.ndarray | jax.Array,
    ) -> _RolloutMetrics:
        com = self.center_of_mass(data)
        goal_np = np.asarray(goal_xyz, dtype=np.float32)
        dist = float(np.linalg.norm(com[:2] - goal_np[:2]))
        closing_rate = max(metrics.prev_dist - dist, 0.0) / float(trainer_module.BRAIN_DT)

        fast_alpha = self._ema_alpha(float(trainer_module.BRAIN_DT), float(trainer_module.FAST_PROGRESS_TAU_S))
        slow_alpha = self._ema_alpha(float(trainer_module.BRAIN_DT), float(trainer_module.SLOW_PROGRESS_TAU_S))
        fast_closing_rate = metrics.fast_closing_rate + (fast_alpha * (closing_rate - metrics.fast_closing_rate))
        slow_closing_rate = metrics.slow_closing_rate + (slow_alpha * (closing_rate - metrics.slow_closing_rate))
        if slow_closing_rate > 1e-6:
            progress_drop_ratio = float(np.clip((slow_closing_rate - fast_closing_rate) / slow_closing_rate, 0.0, 1.0))
        else:
            progress_drop_ratio = 0.0
        dramatic_progress_drop = max(
            0.0,
            (progress_drop_ratio - float(trainer_module.DRAMATIC_PROGRESS_DROP_RATIO))
            / max(1.0 - float(trainer_module.DRAMATIC_PROGRESS_DROP_RATIO), 1e-6),
        )
        target_noise_scale = float(trainer_module.DEFAULT_MOTOR_NOISE_SCALE) + (
            dramatic_progress_drop
            * (float(trainer_module.MAX_MOTOR_NOISE_SCALE) - float(trainer_module.DEFAULT_MOTOR_NOISE_SCALE))
        )
        noise_alpha = self._ema_alpha(float(trainer_module.BRAIN_DT), float(trainer_module.NOISE_ATTACK_TAU_S))
        release_alpha = self._ema_alpha(float(trainer_module.BRAIN_DT), float(trainer_module.NOISE_RELEASE_TAU_S))
        if target_noise_scale > metrics.noise_scale:
            noise_scale = metrics.noise_scale + noise_alpha * (target_noise_scale - metrics.noise_scale)
        else:
            noise_scale = metrics.noise_scale + release_alpha * (target_noise_scale - metrics.noise_scale)

        progress = metrics.prev_dist - dist
        reward = progress * float(trainer_module.PROGRESS_REWARD_SCALE)
        reached_goal_now = (not metrics.goal_reached) and (dist <= float(trainer_module.GOAL_REACHED_RADIUS_M))
        reward += float(trainer_module.GOAL_REACHED_BONUS) if reached_goal_now else 0.0

        body_rot = self.body_rotation(data)
        imu_roll_rad = self._wrap_angle_pi(float(body_rot[0]))
        side_tip_depth_norm = self._side_tip_depth_norm(imu_roll_rad)
        side_tip_depth_delta = metrics.prev_side_tip_depth_norm - side_tip_depth_norm
        reward -= (side_tip_depth_norm * side_tip_depth_norm) * float(trainer_module.SIDE_TIP_DEPTH_PENALTY_SCALE)
        reward += side_tip_depth_delta * float(trainer_module.SIDE_TIP_ESCAPE_DELTA_SCALE)
        exited_side_band = metrics.prev_side_tip_depth_norm > 0.0 and side_tip_depth_norm <= 0.0
        if exited_side_band:
            reward += float(trainer_module.SIDE_TIP_EXIT_BONUS)

        tipped = side_tip_depth_norm > 0.7
        tipped_time = metrics.tipped_time + float(trainer_module.BRAIN_DT) if tipped else 0.0
        dead = metrics.dead or tipped_time >= float(trainer_module.TIPPED_KILL_TIME_S)
        if dead:
            reward = 0.0

        return _RolloutMetrics(
            total_reward=metrics.total_reward + reward,
            prev_dist=dist,
            noise_scale=noise_scale,
            fast_closing_rate=fast_closing_rate,
            slow_closing_rate=slow_closing_rate,
            closing_rate=closing_rate,
            progress_drop_ratio=progress_drop_ratio,
            prev_side_tip_depth_norm=side_tip_depth_norm,
            goal_reached=metrics.goal_reached or reached_goal_now,
            max_step_reached=metrics.max_step_reached,
            tipped_time=tipped_time,
            dead=dead,
        )

    def _contact_mode_name(self, foot_position: np.ndarray) -> str:
        floor_height = self.floor_height_at(foot_position[:2])
        threshold = floor_height + self.contact_margin_m
        return "static" if float(foot_position[2]) <= threshold else "airborne"

    def floor_height_at(self, _xy: np.ndarray | list[float] | tuple[float, float]) -> float:
        return self.floor_height_m

    def foot_clearance_m(self, foot_position: np.ndarray) -> float:
        return float(foot_position[2] - self.floor_height_at(foot_position[:2]))

    def _body_corners_world(self, data: mujoco.MjData) -> list[list[float]]:
        half_extents = np.asarray(self.body_half_extents_m, dtype=np.float32)
        local_corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    local_corners.append(
                        [
                            float(sx * half_extents[0]),
                            float(sy * half_extents[1]),
                            float(sz * half_extents[2]),
                        ]
                    )
        local_corners_np = np.asarray(local_corners, dtype=np.float32)
        body_pos = self.body_position(data)
        body_rotation = np.asarray(data.xmat[self._torso_body_id], dtype=np.float32).reshape(3, 3)
        world_corners = body_pos[None, :] + np.matmul(local_corners_np, body_rotation.T)
        return world_corners.tolist()

    def _snapshot(
        self,
        data: mujoco.MjData,
        metrics: _RolloutMetrics,
        goal_xyz: np.ndarray | jax.Array,
        step_index: int,
        total_steps: int,
        policy_output: np.ndarray | None = None,
        selected_command: str | None = None,
    ) -> dict[str, Any]:
        body_pos = self.body_position(data)
        body_rot = self.body_rotation(data)
        com = self.center_of_mass(data)
        foot_positions = self.foot_positions(data)
        leg_com_positions = self.leg_com_positions(data)
        leg_angles = self.leg_angles(data)
        leg_velocities = self.leg_velocities(data)

        legs = []
        motors = []
        for index, leg_name in enumerate(self.leg_names):
            mount = np.asarray(data.xpos[self._leg_body_ids[index]], dtype=np.float32)
            foot_position = foot_positions[index]
            legs.append(
                {
                    "name": leg_name,
                    "mount": mount.tolist(),
                    "foot": foot_position.tolist(),
                    "com": leg_com_positions[index].tolist(),
                    "angle_rad": float(leg_angles[index]),
                    "contact_mode": self._contact_mode_name(foot_position),
                }
            )
            motors.append(
                {
                    "name": leg_name,
                    "target_velocity_rad_s": float(leg_velocities[index]),
                }
            )

        return {
            "type": "step",
            "step": int(step_index),
            "total_steps": int(total_steps),
            "reward": float(metrics.total_reward),
            "qpos": np.asarray(data.qpos, dtype=np.float64).tolist(),
            "qvel": np.asarray(data.qvel, dtype=np.float64).tolist(),
            "goal": np.asarray(goal_xyz, dtype=np.float32).tolist(),
            "body": {
                "pos": body_pos.tolist(),
                "rot": body_rot.tolist(),
                "com": body_pos.tolist(),
                "corners": self._body_corners_world(data),
            },
            "com": com.tolist(),
            "legs": legs,
            "motors": motors,
            "noise_scale": float(metrics.noise_scale),
            "closing_rate_m_s": float(metrics.closing_rate),
            "progress_drop_ratio": float(metrics.progress_drop_ratio),
            "time_s": float(data.time),
            "action_mode": self.spec.control.mode,
            "policy_output": np.asarray(policy_output, dtype=np.float32).tolist() if policy_output is not None else None,
            "selected_command": selected_command,
        }

    def _advance(self, data: mujoco.MjData, target_velocity: np.ndarray) -> None:
        kp = float(self.spec.simulator.mujoco.velocity_servo_gain)
        torque_limit = float(self.spec.simulator.mujoco.actuator_force_limit)
        for _ in range(self.control_substeps):
            leg_velocity = self.leg_velocities(data)
            ctrl = np.clip(kp * (target_velocity - leg_velocity), -torque_limit, torque_limit)
            data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, data)

    def run_episode(
        self,
        params_flat: np.ndarray | jax.Array,
        goal_xyz: np.ndarray | jax.Array,
        key: np.ndarray | jax.Array,
        steps: int,
        spawn_xy: np.ndarray | jax.Array | None = None,
    ) -> float:
        return self.run_logged_episode(params_flat, goal_xyz, key, lambda _message: None, steps, spawn_xy=spawn_xy)

    def run_population(
        self,
        params_batch: np.ndarray | jax.Array,
        goal_xyz: np.ndarray | jax.Array,
        keys: np.ndarray | jax.Array,
        steps: int,
        spawn_xys: np.ndarray | jax.Array | None = None,
    ) -> np.ndarray:
        params_np = np.asarray(params_batch, dtype=np.float32)
        keys_jax = jnp.asarray(keys, dtype=jnp.uint32)
        if spawn_xys is None:
            spawn_np = np.zeros((params_np.shape[0], 2), dtype=np.float32)
        else:
            spawn_np = np.asarray(spawn_xys, dtype=np.float32)
        returns = np.zeros((params_np.shape[0],), dtype=np.float32)
        for index in range(params_np.shape[0]):
            returns[index] = self.run_episode(
                params_np[index],
                goal_xyz,
                keys_jax[index],
                int(steps),
                spawn_xy=spawn_np[index],
            )
        return returns

    def run_logged_episode(
        self,
        params_flat: np.ndarray | jax.Array,
        goal_xyz: np.ndarray | jax.Array,
        key: np.ndarray | jax.Array,
        on_step: LoggedStepCallback,
        steps: int,
        spawn_xy: np.ndarray | jax.Array | None = None,
    ) -> float:
        policy_runtime = trainer_module._require_policy_runtime()
        params = policy_runtime.unflatten_params(jnp.asarray(params_flat, dtype=jnp.float32))
        data = self.reset_data(spawn_xy=spawn_xy)
        goal_xyz_jax = jnp.asarray(goal_xyz, dtype=jnp.float32)
        key_jax = jnp.asarray(key, dtype=jnp.uint32)
        brain_state = policy_runtime.zero_state()
        metrics = self.initial_metrics(data, goal_xyz_jax)

        for step_index in range(int(steps)):
            obs = self.build_obs(data, goal_xyz_jax)
            brain_state, motor_cmds, key_jax = policy_runtime.step(
                params,
                brain_state,
                obs,
                key_jax,
                jnp.float32(metrics.noise_scale),
            )
            projection = self._action_projector.project(
                np.asarray(motor_cmds, dtype=np.float32),
                time_s=float(data.time),
                max_motor_rad_s=float(trainer_module.MAX_MOTOR_RAD_S),
                motor_scale=float(trainer_module.MOTOR_SCALE),
            )
            target_velocity = np.asarray(projection.target_velocity_rad_s, dtype=np.float32)
            self._advance(data, target_velocity)
            metrics = self._step_metrics(data, metrics, goal_xyz_jax)
            on_step(
                self._snapshot(
                    data,
                    metrics,
                    goal_xyz_jax,
                    step_index,
                    int(steps),
                    policy_output=projection.policy_output,
                    selected_command=projection.selected_command,
                )
            )

        return float(metrics.total_reward)
