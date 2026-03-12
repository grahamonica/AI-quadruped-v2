"""JAX trainer and functional simulator backend."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


# Robot / physics constants mirror the current KT2-style defaults.
N_IN = 48
N_OUT = 4
N_HIDDEN_LAYERS = 4
SHARED_TRUNK_WIDTH = 64
MOTOR_LANE_WIDTH = 64
N_LEGS = 4
N_BODY_CORNERS = 8
LEG_NAMES = ("front_left", "front_right", "rear_left", "rear_right")

BODY_LENGTH_M = 0.28
BODY_WIDTH_M = 0.12
BODY_HEIGHT_M = 0.02
BODY_MASS_KG = 2.4
LEG_LENGTH_M = 0.16
LEG_MASS_KG = 0.6
FOOT_STATIC_FRICTION = 0.9
FOOT_KINETIC_FRICTION = 0.65

TAU_MEM = 0.020
V_THRESH = 0.01
V_RESET = 0.0
DT = 0.010
TRACE_DECAY = 0.70
DEFAULT_MOTOR_NOISE_SCALE = 0.60

EPISODE_S = 300.0
BRAIN_DT = 0.050
MOTOR_SCALE = 6.0
GOAL_HEIGHT_M = 0.16
FIELD_HALF = 15.0
POP_SIZE = 256
SIGMA = 0.05
LR = 0.05
PARENT_ELITE_COUNT = 5
MAX_MOTOR_RAD_S = 8.0
MAX_MOTOR_NOISE_SCALE = 1.20
FAST_PROGRESS_TAU_S = 0.20
SLOW_PROGRESS_TAU_S = 0.80
DRAMATIC_PROGRESS_DROP_RATIO = 0.55
NOISE_ATTACK_TAU_S = 0.15
NOISE_RELEASE_TAU_S = 0.90
SIDE_TIP_BAND_HALF_WIDTH_RAD = math.radians(60.0)
SIDE_TIP_DEPTH_PENALTY_SCALE = 20.0
SIDE_TIP_ESCAPE_DELTA_SCALE = 20.0
SIDE_TIP_EXIT_BONUS = 8.0
PROGRESS_REWARD_SCALE = 25.0
GOAL_REACHED_RADIUS_M = 0.25
GOAL_REACHED_BONUS = 25.0

GRAVITY_M_S2 = 9.81
FLOOR_HEIGHT_M = 0.0
NORMAL_STIFFNESS_N_M = 20000.0
NORMAL_DAMPING_N_S_M = 3500.0
TANGENTIAL_STIFFNESS_N_M = 7000.0
TANGENTIAL_DAMPING_N_S_M = 450.0
BODY_CONTACT_FRICTION = 0.35
ANGULAR_DAMPING_N_M_S = 8.0
LINEAR_DAMPING_N_S_M = 80.0
AIRBORNE_LINEAR_DAMPING_N_S_M = 3.0
AIRBORNE_ANGULAR_DAMPING_N_M_S = 1.0
MAX_CONTACT_FORCE_N = 120.0
MAX_SUBSTEP_S = 1.0 / 4000.0
UNLOADING_STIFFNESS_SCALE = 0.4
SLEEP_LINEAR_SPEED_THRESHOLD_M_S = 0.01
SLEEP_ANGULAR_SPEED_THRESHOLD_RAD_S = 0.06

MOTOR_MAX_ANGULAR_ACCELERATION_RAD_S2 = 18.0
MOTOR_VISCOUS_DAMPING_PER_S = 8.0
MOTOR_VELOCITY_FILTER_TAU_S = 0.05

TOTAL_MASS_KG = BODY_MASS_KG + (N_LEGS * LEG_MASS_KG)
BODY_HALF_EXTENTS = jnp.array([BODY_LENGTH_M / 2.0, BODY_WIDTH_M / 2.0, BODY_HEIGHT_M / 2.0], dtype=jnp.float32)
BODY_PRINCIPAL_INERTIA = jnp.array(
    [
        (BODY_MASS_KG / 12.0) * ((BODY_WIDTH_M * BODY_WIDTH_M) + (BODY_HEIGHT_M * BODY_HEIGHT_M)),
        (BODY_MASS_KG / 12.0) * ((BODY_LENGTH_M * BODY_LENGTH_M) + (BODY_HEIGHT_M * BODY_HEIGHT_M)),
        (BODY_MASS_KG / 12.0) * ((BODY_LENGTH_M * BODY_LENGTH_M) + (BODY_WIDTH_M * BODY_WIDTH_M)),
    ],
    dtype=jnp.float32,
)
LEG_INERTIA_ABOUT_MOUNT = jnp.full((N_LEGS,), LEG_MASS_KG * (LEG_LENGTH_M**2) / 3.0, dtype=jnp.float32)
LEG_ROT_AXIS_BODY = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
REST_CONTACT_BUFFER_M = (TOTAL_MASS_KG * GRAVITY_M_S2) / (max(N_LEGS, 1) * NORMAL_STIFFNESS_N_M)
SUBSTEP_COUNT = int(math.ceil(BRAIN_DT / MAX_SUBSTEP_S))
SUBSTEP_DT_S = BRAIN_DT / SUBSTEP_COUNT

MOUNT_POINTS_BODY = jnp.array(
    [
        [BODY_LENGTH_M / 2.0, BODY_WIDTH_M / 2.0, 0.0],
        [BODY_LENGTH_M / 2.0, -BODY_WIDTH_M / 2.0, 0.0],
        [-BODY_LENGTH_M / 2.0, BODY_WIDTH_M / 2.0, 0.0],
        [-BODY_LENGTH_M / 2.0, -BODY_WIDTH_M / 2.0, 0.0],
    ],
    dtype=jnp.float32,
)
BODY_CORNERS_BODY = jnp.array(
    [
        [sx * BODY_HALF_EXTENTS[0], sy * BODY_HALF_EXTENTS[1], sz * BODY_HALF_EXTENTS[2]]
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ],
    dtype=jnp.float32,
)


class BrainState(NamedTuple):
    v_shared: jax.Array
    trace_shared: jax.Array
    v_motor: jax.Array
    trace_motor: jax.Array


class EnvState(NamedTuple):
    body_pos: jax.Array
    body_vel: jax.Array
    body_rot: jax.Array
    body_ang_vel: jax.Array
    leg_angle: jax.Array
    leg_ang_vel: jax.Array
    motor_target: jax.Array
    motor_current: jax.Array
    motor_smoothed_target: jax.Array
    leg_contact_mode: jax.Array
    leg_contact_in: jax.Array
    leg_anchor_xy: jax.Array
    body_contact_in: jax.Array
    body_anchor_xy: jax.Array
    time_s: jax.Array


class EpisodeCarry(NamedTuple):
    env_state: EnvState
    brain_state: BrainState
    key: jax.Array
    total_reward: jax.Array
    initial_dist: jax.Array
    prev_dist: jax.Array
    noise_scale: jax.Array
    fast_closing_rate: jax.Array
    slow_closing_rate: jax.Array
    closing_rate: jax.Array
    progress_drop_ratio: jax.Array
    prev_side_tip_depth_norm: jax.Array
    goal_reached: jax.Array


@dataclass
class TrainingState:
    generation: int = 0
    best_reward: float = -1e9
    best_single_reward: float = -1e9
    mean_reward: float = 0.0
    episode_reward: float = 0.0
    goal_xyz: tuple[float, float, float] = (1.0, 0.0, GOAL_HEIGHT_M)
    robot_state: dict[str, Any] = field(default_factory=dict)
    rewards_history: list[float] = field(default_factory=list)


PARAM_SPECS = [
    ("w_in_shared", (SHARED_TRUNK_WIDTH, N_IN)),
    ("w_in_motor", (N_OUT, MOTOR_LANE_WIDTH, N_IN)),
    ("w_shared_from_shared", (N_HIDDEN_LAYERS - 1, SHARED_TRUNK_WIDTH, SHARED_TRUNK_WIDTH)),
    ("w_shared_from_motors", (N_HIDDEN_LAYERS - 1, SHARED_TRUNK_WIDTH, N_OUT * MOTOR_LANE_WIDTH)),
    ("w_motor_from_motor", (N_HIDDEN_LAYERS - 1, N_OUT, MOTOR_LANE_WIDTH, MOTOR_LANE_WIDTH)),
    ("w_motor_from_shared", (N_HIDDEN_LAYERS - 1, N_OUT, MOTOR_LANE_WIDTH, SHARED_TRUNK_WIDTH)),
    ("b_shared", (N_HIDDEN_LAYERS, SHARED_TRUNK_WIDTH)),
    ("b_motor", (N_HIDDEN_LAYERS, N_OUT, MOTOR_LANE_WIDTH)),
    ("w_out_shared", (N_OUT, SHARED_TRUNK_WIDTH)),
    ("w_out_motor", (N_OUT, MOTOR_LANE_WIDTH)),
    ("b_out", (N_OUT,)),
]


def _param_size(shape: tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        size *= dim
    return size


PARAM_OFFSETS: dict[str, tuple[int, int, tuple[int, ...]]] = {}
_offset = 0
for _name, _shape in PARAM_SPECS:
    _size = _param_size(_shape)
    PARAM_OFFSETS[_name] = (_offset, _offset + _size, _shape)
    _offset += _size
PARAM_COUNT = _offset


def _rotation_matrix_xyz(rotation_xyz_rad: jax.Array) -> jax.Array:
    roll_rad, pitch_rad, yaw_rad = rotation_xyz_rad
    cr = jnp.cos(roll_rad)
    sr = jnp.sin(roll_rad)
    cp = jnp.cos(pitch_rad)
    sp = jnp.sin(pitch_rad)
    cy = jnp.cos(yaw_rad)
    sy = jnp.sin(yaw_rad)
    return jnp.array(
        [
            [cy * cp, (cy * sp * sr) - (sy * cr), (cy * sp * cr) + (sy * sr)],
            [sy * cp, (sy * sp * sr) + (cy * cr), (sy * sp * cr) - (cy * sr)],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=jnp.float32,
    )


def _body_vectors_world(rotation_xyz_rad: jax.Array, vectors_body: jax.Array) -> jax.Array:
    rotation = _rotation_matrix_xyz(rotation_xyz_rad)
    return jnp.matmul(vectors_body, rotation.T)


def _body_points_world(body_pos: jax.Array, rotation_xyz_rad: jax.Array, points_body: jax.Array) -> jax.Array:
    return body_pos + _body_vectors_world(rotation_xyz_rad, points_body)


def _point_velocity_world(body_vel: jax.Array, body_ang_vel: jax.Array, r_world: jax.Array) -> jax.Array:
    return body_vel + jnp.cross(body_ang_vel, r_world)


def _wrap_angle_pi(angle_rad: jax.Array) -> jax.Array:
    return jnp.mod(angle_rad + jnp.pi, 2.0 * jnp.pi) - jnp.pi


def _ema_alpha(dt_s: float, tau_s: float) -> jax.Array:
    return jnp.float32(1.0 - math.exp(-dt_s / max(tau_s, 1e-6)))


def _contact_mode_name(mode: int) -> str:
    if mode == 1:
        return "static"
    if mode == 2:
        return "kinetic"
    return "airborne"


def _side_tip_depth_norm(imu_roll_rad: jax.Array) -> jax.Array:
    abs_roll_rad = jnp.abs(_wrap_angle_pi(imu_roll_rad))
    side_center_error_rad = jnp.abs(abs_roll_rad - jnp.float32(math.pi / 2.0))
    side_band_depth_rad = jnp.maximum(jnp.float32(SIDE_TIP_BAND_HALF_WIDTH_RAD) - side_center_error_rad, 0.0)
    return side_band_depth_rad / jnp.float32(SIDE_TIP_BAND_HALF_WIDTH_RAD)


def _brain_zero_state() -> BrainState:
    return BrainState(
        v_shared=jnp.zeros((N_HIDDEN_LAYERS, SHARED_TRUNK_WIDTH), dtype=jnp.float32),
        trace_shared=jnp.zeros((N_HIDDEN_LAYERS, SHARED_TRUNK_WIDTH), dtype=jnp.float32),
        v_motor=jnp.zeros((N_HIDDEN_LAYERS, N_OUT, MOTOR_LANE_WIDTH), dtype=jnp.float32),
        trace_motor=jnp.zeros((N_HIDDEN_LAYERS, N_OUT, MOTOR_LANE_WIDTH), dtype=jnp.float32),
    )


def _init_param_vector(key: jax.Array) -> jax.Array:
    keys = jax.random.split(key, len(PARAM_SPECS))
    parts: list[jax.Array] = []
    for subkey, (name, shape) in zip(keys, PARAM_SPECS, strict=True):
        if name in {"w_in_shared", "w_in_motor"}:
            scale = 1.0 / math.sqrt(N_IN)
            part = jax.random.normal(subkey, shape, dtype=jnp.float32) * jnp.float32(scale)
        elif name == "w_shared_from_shared":
            scale = 1.0 / math.sqrt(SHARED_TRUNK_WIDTH)
            part = jax.random.normal(subkey, shape, dtype=jnp.float32) * jnp.float32(scale)
        elif name == "w_shared_from_motors":
            scale = 1.0 / math.sqrt(N_OUT * MOTOR_LANE_WIDTH)
            part = jax.random.normal(subkey, shape, dtype=jnp.float32) * jnp.float32(scale)
        elif name == "w_motor_from_motor":
            scale = 1.0 / math.sqrt(MOTOR_LANE_WIDTH)
            part = jax.random.normal(subkey, shape, dtype=jnp.float32) * jnp.float32(scale)
        elif name == "w_motor_from_shared":
            scale = 1.0 / math.sqrt(SHARED_TRUNK_WIDTH)
            part = jax.random.normal(subkey, shape, dtype=jnp.float32) * jnp.float32(scale)
        elif name in {"b_shared", "b_motor"}:
            part = jax.random.uniform(subkey, shape, dtype=jnp.float32, minval=-0.2, maxval=0.2)
        else:
            part = jnp.zeros(shape, dtype=jnp.float32)
        parts.append(part.reshape(-1))
    return jnp.concatenate(parts, axis=0)


def _unflatten_params(flat_params: jax.Array) -> dict[str, jax.Array]:
    return {
        name: flat_params[..., start:end].reshape(flat_params.shape[:-1] + shape)
        for name, (start, end, shape) in PARAM_OFFSETS.items()
    }


def _brain_step(
    params: dict[str, jax.Array],
    state: BrainState,
    obs: jax.Array,
    key: jax.Array,
    noise_scale: jax.Array,
) -> tuple[BrainState, jax.Array, jax.Array]:
    scale = jnp.float32(DT / TAU_MEM)
    decay = jnp.float32(1.0) - scale

    def _update_hidden(
        layer_index: int,
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        v_shared, trace_shared, v_motor, trace_motor = carry

        def _layer0_inputs() -> tuple[jax.Array, jax.Array]:
            shared_input = jnp.einsum("hi,i->h", params["w_in_shared"], obs) + params["b_shared"][0]
            motor_input = jnp.einsum("mhi,i->mh", params["w_in_motor"], obs) + params["b_motor"][0]
            return shared_input, motor_input

        def _later_inputs() -> tuple[jax.Array, jax.Array]:
            prev_shared = trace_shared[layer_index - 1]
            prev_motor = trace_motor[layer_index - 1]
            prev_motor_flat = prev_motor.reshape(-1)
            shared_input = (
                jnp.einsum("hj,j->h", params["w_shared_from_shared"][layer_index - 1], prev_shared)
                + jnp.einsum("hj,j->h", params["w_shared_from_motors"][layer_index - 1], prev_motor_flat)
                + params["b_shared"][layer_index]
            )
            motor_input = (
                jnp.einsum("mhj,mj->mh", params["w_motor_from_motor"][layer_index - 1], prev_motor)
                + jnp.einsum("mhj,j->mh", params["w_motor_from_shared"][layer_index - 1], prev_shared)
                + params["b_motor"][layer_index]
            )
            return shared_input, motor_input

        shared_input, motor_input = jax.lax.cond(layer_index == 0, _layer0_inputs, _later_inputs)

        new_v_shared = (v_shared[layer_index] * decay) + (shared_input * scale)
        shared_spikes = (new_v_shared >= V_THRESH).astype(jnp.float32)
        new_v_shared = jnp.where(shared_spikes > 0.0, jnp.float32(V_RESET), new_v_shared)
        new_trace_shared = (trace_shared[layer_index] * TRACE_DECAY) + (shared_spikes * (1.0 - TRACE_DECAY))

        new_v_motor = (v_motor[layer_index] * decay) + (motor_input * scale)
        motor_spikes = (new_v_motor >= V_THRESH).astype(jnp.float32)
        new_v_motor = jnp.where(motor_spikes > 0.0, jnp.float32(V_RESET), new_v_motor)
        new_trace_motor = (trace_motor[layer_index] * TRACE_DECAY) + (motor_spikes * (1.0 - TRACE_DECAY))

        v_shared = v_shared.at[layer_index].set(new_v_shared)
        trace_shared = trace_shared.at[layer_index].set(new_trace_shared)
        v_motor = v_motor.at[layer_index].set(new_v_motor)
        trace_motor = trace_motor.at[layer_index].set(new_trace_motor)
        return v_shared, trace_shared, v_motor, trace_motor

    v_shared, trace_shared, v_motor, trace_motor = jax.lax.fori_loop(
        0,
        N_HIDDEN_LAYERS,
        _update_hidden,
        (state.v_shared, state.trace_shared, state.v_motor, state.trace_motor),
    )
    out = jnp.tanh(
        jnp.einsum("mh,mh->m", params["w_out_motor"], trace_motor[-1])
        + jnp.einsum("mh,h->m", params["w_out_shared"], trace_shared[-1])
        + params["b_out"]
    )
    key, noise_key = jax.random.split(key)
    out = out + (jax.random.normal(noise_key, (N_OUT,), dtype=jnp.float32) * jnp.maximum(noise_scale, 0.0))
    out = jnp.clip(out, -1.0, 1.0)
    return BrainState(v_shared, trace_shared, v_motor, trace_motor), out, key


def _leg_offsets(angle_rad: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    sin_angle = jnp.sin(angle_rad)
    cos_angle = jnp.cos(angle_rad)
    foot_offset = jnp.stack([LEG_LENGTH_M * sin_angle, jnp.zeros_like(angle_rad), -LEG_LENGTH_M * cos_angle], axis=1)
    foot_vel = jnp.stack(
        [LEG_LENGTH_M * cos_angle * angle_rad * 0.0, jnp.zeros_like(angle_rad), LEG_LENGTH_M * sin_angle * angle_rad * 0.0],
        axis=1,
    )
    com_offset = jnp.stack(
        [(LEG_LENGTH_M * 0.5) * sin_angle, jnp.zeros_like(angle_rad), -(LEG_LENGTH_M * 0.5) * cos_angle],
        axis=1,
    )
    return sin_angle, cos_angle, foot_offset, com_offset


def _compute_leg_kinematics(
    body_pos: jax.Array,
    body_vel: jax.Array,
    body_rot: jax.Array,
    body_ang_vel: jax.Array,
    leg_angle: jax.Array,
    leg_ang_vel: jax.Array,
    leg_ang_acc: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    sin_angle = jnp.sin(leg_angle)
    cos_angle = jnp.cos(leg_angle)
    foot_offset_body = jnp.stack([LEG_LENGTH_M * sin_angle, jnp.zeros_like(leg_angle), -LEG_LENGTH_M * cos_angle], axis=1)
    foot_vel_body = jnp.stack(
        [LEG_LENGTH_M * cos_angle * leg_ang_vel, jnp.zeros_like(leg_angle), LEG_LENGTH_M * sin_angle * leg_ang_vel],
        axis=1,
    )
    com_offset_body = jnp.stack(
        [(LEG_LENGTH_M * 0.5) * sin_angle, jnp.zeros_like(leg_angle), -(LEG_LENGTH_M * 0.5) * cos_angle],
        axis=1,
    )
    com_acc_body = jnp.stack(
        [
            (-(LEG_LENGTH_M * 0.5) * sin_angle * (leg_ang_vel**2)) + ((LEG_LENGTH_M * 0.5) * cos_angle * leg_ang_acc),
            jnp.zeros_like(leg_angle),
            ((LEG_LENGTH_M * 0.5) * cos_angle * (leg_ang_vel**2)) + ((LEG_LENGTH_M * 0.5) * sin_angle * leg_ang_acc),
        ],
        axis=1,
    )
    mount_r_world = _body_vectors_world(body_rot, MOUNT_POINTS_BODY)
    mount_world = body_pos + mount_r_world
    foot_offset_world = _body_vectors_world(body_rot, foot_offset_body)
    foot_position = mount_world + foot_offset_world
    foot_velocity = _point_velocity_world(body_vel, body_ang_vel, mount_r_world) + _body_vectors_world(body_rot, foot_vel_body)
    leg_com_world = mount_world + _body_vectors_world(body_rot, com_offset_body)
    leg_com_acc_world = _body_vectors_world(body_rot, com_acc_body)
    return mount_world, foot_position, foot_velocity, foot_position - body_pos, leg_com_world, leg_com_acc_world


def _contact_force_batch(
    position: jax.Array,
    velocity: jax.Array,
    static_mu: jax.Array,
    kinetic_mu: jax.Array,
    memory_in_contact: jax.Array,
    memory_anchor_xy: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    support_gap = jnp.where(memory_in_contact, REST_CONTACT_BUFFER_M, 0.0)
    effective_penetration = jnp.maximum((FLOOR_HEIGHT_M + support_gap) - position[:, 2], 0.0)
    unloading_scale = jnp.where(velocity[:, 2] > 0.0, UNLOADING_STIFFNESS_SCALE, 1.0)
    normal_force = (
        (NORMAL_STIFFNESS_N_M * unloading_scale * effective_penetration)
        + (NORMAL_DAMPING_N_S_M * jnp.maximum(-velocity[:, 2], 0.0))
    )
    normal_force = jnp.clip(normal_force, 0.0, MAX_CONTACT_FORCE_N)
    active = (effective_penetration > 0.0) | ((position[:, 2] <= FLOOR_HEIGHT_M + 1e-5) & (velocity[:, 2] <= 0.0))
    normal_force = jnp.where(active, normal_force, 0.0)

    anchor_xy = jnp.where(memory_in_contact[:, None], memory_anchor_xy, position[:, :2])
    tangential_deflection = position[:, :2] - anchor_xy
    static_candidate = (
        (-TANGENTIAL_STIFFNESS_N_M * tangential_deflection)
        - (TANGENTIAL_DAMPING_N_S_M * velocity[:, :2])
    )
    static_mag = jnp.linalg.norm(static_candidate, axis=1)
    static_limit = static_mu * normal_force
    static_mask = active & (static_mag <= static_limit)

    tangential_speed = jnp.linalg.norm(velocity[:, :2], axis=1)
    safe_speed = jnp.where(tangential_speed > 1e-6, tangential_speed, 1.0)
    friction_from_velocity = -kinetic_mu[:, None] * normal_force[:, None] * velocity[:, :2] / safe_speed[:, None]
    friction_from_sign = -kinetic_mu[:, None] * normal_force[:, None] * jnp.sign(static_candidate)
    kinetic_xy = jnp.where((tangential_speed > 1e-6)[:, None], friction_from_velocity, friction_from_sign)

    static_force = jnp.concatenate([static_candidate, normal_force[:, None]], axis=1)
    kinetic_force = jnp.concatenate([kinetic_xy, normal_force[:, None]], axis=1)
    force = jnp.where(static_mask[:, None], static_force, kinetic_force)
    force = jnp.where(active[:, None], force, jnp.zeros_like(force))

    new_anchor_xy = jnp.where(
        (~active)[:, None],
        position[:, :2],
        jnp.where(static_mask[:, None], anchor_xy, position[:, :2]),
    )
    new_in_contact = active
    mode = jnp.where(~active, 0, jnp.where(static_mask, 1, 2))
    return force, mode, new_in_contact, new_anchor_xy


def _env_reset() -> EnvState:
    initial_body_pos = jnp.array([0.0, 0.0, LEG_LENGTH_M], dtype=jnp.float32)
    initial_body_rot = jnp.zeros((3,), dtype=jnp.float32)
    initial_body_vel = jnp.zeros((3,), dtype=jnp.float32)
    initial_body_ang_vel = jnp.zeros((3,), dtype=jnp.float32)
    initial_leg_angle = jnp.zeros((N_LEGS,), dtype=jnp.float32)
    initial_leg_ang_vel = jnp.zeros((N_LEGS,), dtype=jnp.float32)
    initial_motor_target = jnp.zeros((N_LEGS,), dtype=jnp.float32)
    initial_motor_current = jnp.zeros((N_LEGS,), dtype=jnp.float32)
    initial_motor_smoothed = jnp.zeros((N_LEGS,), dtype=jnp.float32)

    mount_world, foot_position, _, _, _, _ = _compute_leg_kinematics(
        initial_body_pos,
        initial_body_vel,
        initial_body_rot,
        initial_body_ang_vel,
        initial_leg_angle,
        initial_leg_ang_vel,
        jnp.zeros((N_LEGS,), dtype=jnp.float32),
    )
    del mount_world
    body_corner_world = _body_points_world(initial_body_pos, initial_body_rot, BODY_CORNERS_BODY)
    return EnvState(
        body_pos=initial_body_pos,
        body_vel=initial_body_vel,
        body_rot=initial_body_rot,
        body_ang_vel=initial_body_ang_vel,
        leg_angle=initial_leg_angle,
        leg_ang_vel=initial_leg_ang_vel,
        motor_target=initial_motor_target,
        motor_current=initial_motor_current,
        motor_smoothed_target=initial_motor_smoothed,
        leg_contact_mode=jnp.ones((N_LEGS,), dtype=jnp.int32),
        leg_contact_in=jnp.ones((N_LEGS,), dtype=bool),
        leg_anchor_xy=foot_position[:, :2],
        body_contact_in=body_corner_world[:, 2] <= FLOOR_HEIGHT_M + 1e-5,
        body_anchor_xy=body_corner_world[:, :2],
        time_s=jnp.float32(0.0),
    )


def _env_substep(state: EnvState, motor_target: jax.Array) -> EnvState:
    smooth_alpha = jnp.float32(1.0 - math.exp(-SUBSTEP_DT_S / max(MOTOR_VELOCITY_FILTER_TAU_S, 1e-6)))
    smoothed_target = state.motor_smoothed_target + smooth_alpha * (motor_target - state.motor_smoothed_target)
    prev_motor_current = state.motor_current
    damped_motor_current = state.motor_current * jnp.float32(math.exp(-MOTOR_VISCOUS_DAMPING_PER_S * SUBSTEP_DT_S))
    velocity_error = smoothed_target - damped_motor_current
    max_velocity_change = jnp.float32(MOTOR_MAX_ANGULAR_ACCELERATION_RAD_S2 * SUBSTEP_DT_S)
    motor_current = jnp.where(
        jnp.abs(velocity_error) <= max_velocity_change,
        smoothed_target,
        damped_motor_current + (jnp.sign(velocity_error) * max_velocity_change),
    )
    leg_ang_acc = (motor_current - prev_motor_current) / SUBSTEP_DT_S
    leg_ang_vel = motor_current
    leg_angle = state.leg_angle + (leg_ang_vel * SUBSTEP_DT_S)

    mount_world, foot_position, foot_velocity, foot_r_world, _, leg_com_acc_world = _compute_leg_kinematics(
        state.body_pos,
        state.body_vel,
        state.body_rot,
        state.body_ang_vel,
        leg_angle,
        leg_ang_vel,
        leg_ang_acc,
    )
    del mount_world
    leg_force, leg_contact_mode, leg_contact_in, leg_anchor_xy = _contact_force_batch(
        foot_position,
        foot_velocity,
        jnp.full((N_LEGS,), FOOT_STATIC_FRICTION, dtype=jnp.float32),
        jnp.full((N_LEGS,), FOOT_KINETIC_FRICTION, dtype=jnp.float32),
        state.leg_contact_in,
        state.leg_anchor_xy,
    )

    body_corner_world = _body_points_world(state.body_pos, state.body_rot, BODY_CORNERS_BODY)
    body_corner_r_world = body_corner_world - state.body_pos
    body_corner_velocity = _point_velocity_world(state.body_vel, state.body_ang_vel, body_corner_r_world)
    body_force, _, body_contact_in, body_anchor_xy = _contact_force_batch(
        body_corner_world,
        body_corner_velocity,
        jnp.full((N_BODY_CORNERS,), BODY_CONTACT_FRICTION, dtype=jnp.float32),
        jnp.full((N_BODY_CORNERS,), BODY_CONTACT_FRICTION, dtype=jnp.float32),
        state.body_contact_in,
        state.body_anchor_xy,
    )

    total_force = jnp.array([0.0, 0.0, -TOTAL_MASS_KG * GRAVITY_M_S2], dtype=jnp.float32)
    total_torque = jnp.zeros((3,), dtype=jnp.float32)

    inertia_force = -LEG_MASS_KG * leg_com_acc_world
    total_force = total_force + jnp.sum(inertia_force, axis=0)

    leg_rot_axis_world = _body_vectors_world(state.body_rot, LEG_ROT_AXIS_BODY)
    rot_reaction = leg_rot_axis_world * jnp.sum(-LEG_INERTIA_ABOUT_MOUNT * leg_ang_acc)
    total_torque = total_torque + rot_reaction + jnp.sum(jnp.cross(_body_vectors_world(state.body_rot, MOUNT_POINTS_BODY), inertia_force), axis=0)
    total_torque = total_torque + jnp.sum(jnp.cross(foot_r_world, leg_force), axis=0) + jnp.sum(jnp.cross(body_corner_r_world, body_force), axis=0)
    total_force = total_force + jnp.sum(leg_force, axis=0) + jnp.sum(body_force, axis=0)

    any_grounded = jnp.any(leg_force[:, 2] > 0.0) | jnp.any(body_force[:, 2] > 0.0)
    active_linear_damping = jnp.where(any_grounded, LINEAR_DAMPING_N_S_M, AIRBORNE_LINEAR_DAMPING_N_S_M)
    active_angular_damping = jnp.where(any_grounded, ANGULAR_DAMPING_N_M_S, AIRBORNE_ANGULAR_DAMPING_N_M_S)

    linear_acceleration_undamped = total_force / TOTAL_MASS_KG
    linear_damp_divisor = 1.0 + (active_linear_damping * SUBSTEP_DT_S / TOTAL_MASS_KG)
    body_vel = (state.body_vel + (linear_acceleration_undamped * SUBSTEP_DT_S)) / linear_damp_divisor
    body_pos = state.body_pos + (body_vel * SUBSTEP_DT_S)

    angular_acceleration_undamped = total_torque / jnp.maximum(BODY_PRINCIPAL_INERTIA, 1e-6)
    ang_dt = active_angular_damping * SUBSTEP_DT_S
    body_ang_vel = (state.body_ang_vel + (angular_acceleration_undamped * SUBSTEP_DT_S)) / (
        1.0 + (ang_dt / jnp.maximum(BODY_PRINCIPAL_INERTIA, 1e-6))
    )
    body_rot = state.body_rot + (body_ang_vel * SUBSTEP_DT_S)

    linear_speed = jnp.linalg.norm(body_vel)
    angular_speed = jnp.linalg.norm(body_ang_vel)
    sleep_mask = any_grounded & (linear_speed <= SLEEP_LINEAR_SPEED_THRESHOLD_M_S) & (angular_speed <= SLEEP_ANGULAR_SPEED_THRESHOLD_RAD_S)
    body_vel = jnp.where(sleep_mask, jnp.zeros_like(body_vel), body_vel)
    body_ang_vel = jnp.where(sleep_mask, jnp.zeros_like(body_ang_vel), body_ang_vel)

    return EnvState(
        body_pos=body_pos,
        body_vel=body_vel,
        body_rot=body_rot,
        body_ang_vel=body_ang_vel,
        leg_angle=leg_angle,
        leg_ang_vel=leg_ang_vel,
        motor_target=motor_target,
        motor_current=motor_current,
        motor_smoothed_target=smoothed_target,
        leg_contact_mode=leg_contact_mode,
        leg_contact_in=leg_contact_in,
        leg_anchor_xy=leg_anchor_xy,
        body_contact_in=body_contact_in,
        body_anchor_xy=body_anchor_xy,
        time_s=state.time_s + SUBSTEP_DT_S,
    )


def _env_advance(state: EnvState, motor_target: jax.Array) -> EnvState:
    def _step_fn(_index: int, loop_state: EnvState) -> EnvState:
        return _env_substep(loop_state, motor_target)

    return jax.lax.fori_loop(0, SUBSTEP_COUNT, _step_fn, state)


def _center_of_mass(state: EnvState) -> jax.Array:
    _, _, _, _, leg_com_world, _ = _compute_leg_kinematics(
        state.body_pos,
        state.body_vel,
        state.body_rot,
        state.body_ang_vel,
        state.leg_angle,
        state.leg_ang_vel,
        jnp.zeros((N_LEGS,), dtype=jnp.float32),
    )
    weighted = (state.body_pos * BODY_MASS_KG) + jnp.sum(leg_com_world * LEG_MASS_KG, axis=0)
    return weighted / TOTAL_MASS_KG


def _build_obs(state: EnvState, goal_xyz: jax.Array) -> jax.Array:
    _, foot_position, _, _, leg_com_world, _ = _compute_leg_kinematics(
        state.body_pos,
        state.body_vel,
        state.body_rot,
        state.body_ang_vel,
        state.leg_angle,
        state.leg_ang_vel,
        jnp.zeros((N_LEGS,), dtype=jnp.float32),
    )
    com = _center_of_mass(state)
    leg_imu = jnp.stack(
        [
            jnp.full((N_LEGS,), state.body_rot[0], dtype=jnp.float32),
            state.leg_angle,
            jnp.full((N_LEGS,), state.body_rot[2], dtype=jnp.float32),
        ],
        axis=1,
    )
    obs = jnp.concatenate(
        [
            goal_xyz,
            com,
            state.body_pos,
            foot_position.reshape(-1),
            leg_com_world.reshape(-1),
            state.body_rot,
            leg_imu.reshape(-1),
        ],
        axis=0,
    )
    obs = obs.astype(jnp.float32)
    obs = obs.at[:33].divide(jnp.float32(FIELD_HALF))
    obs = obs.at[33:].divide(jnp.float32(math.pi))
    return obs


def _episode_init(goal_xyz: jax.Array, key: jax.Array) -> EpisodeCarry:
    env_state = _env_reset()
    com = _center_of_mass(env_state)
    initial_dist = jnp.linalg.norm(com[:2] - goal_xyz[:2])
    initial_roll_rad = _wrap_angle_pi(env_state.body_rot[0])
    return EpisodeCarry(
        env_state=env_state,
        brain_state=_brain_zero_state(),
        key=key,
        total_reward=jnp.float32(0.0),
        initial_dist=initial_dist,
        prev_dist=initial_dist,
        noise_scale=jnp.float32(DEFAULT_MOTOR_NOISE_SCALE),
        fast_closing_rate=jnp.float32(0.0),
        slow_closing_rate=jnp.float32(0.0),
        closing_rate=jnp.float32(0.0),
        progress_drop_ratio=jnp.float32(0.0),
        prev_side_tip_depth_norm=_side_tip_depth_norm(initial_roll_rad),
        goal_reached=jnp.array(False),
    )


def _episode_step(params: dict[str, jax.Array], carry: EpisodeCarry, goal_xyz: jax.Array) -> EpisodeCarry:
    obs = _build_obs(carry.env_state, goal_xyz)
    brain_state, motor_cmds, key = _brain_step(params, carry.brain_state, obs, carry.key, carry.noise_scale)
    motor_target = jnp.clip(motor_cmds * MOTOR_SCALE, -MAX_MOTOR_RAD_S, MAX_MOTOR_RAD_S)
    env_state = _env_advance(carry.env_state, motor_target)

    com = _center_of_mass(env_state)
    dist = jnp.linalg.norm(com[:2] - goal_xyz[:2])
    closing_rate = jnp.maximum(carry.prev_dist - dist, 0.0) / BRAIN_DT

    fast_alpha = _ema_alpha(BRAIN_DT, FAST_PROGRESS_TAU_S)
    slow_alpha = _ema_alpha(BRAIN_DT, SLOW_PROGRESS_TAU_S)
    fast_closing_rate = carry.fast_closing_rate + (fast_alpha * (closing_rate - carry.fast_closing_rate))
    slow_closing_rate = carry.slow_closing_rate + (slow_alpha * (closing_rate - carry.slow_closing_rate))
    progress_drop_ratio = jnp.where(
        slow_closing_rate > 1e-6,
        jnp.clip((slow_closing_rate - fast_closing_rate) / slow_closing_rate, 0.0, 1.0),
        0.0,
    )
    dramatic_progress_drop = jnp.maximum(
        0.0,
        (progress_drop_ratio - DRAMATIC_PROGRESS_DROP_RATIO) / max(1.0 - DRAMATIC_PROGRESS_DROP_RATIO, 1e-6),
    )
    target_noise_scale = DEFAULT_MOTOR_NOISE_SCALE + (dramatic_progress_drop * (MAX_MOTOR_NOISE_SCALE - DEFAULT_MOTOR_NOISE_SCALE))
    noise_alpha = _ema_alpha(BRAIN_DT, NOISE_ATTACK_TAU_S)
    release_alpha = _ema_alpha(BRAIN_DT, NOISE_RELEASE_TAU_S)
    noise_scale = carry.noise_scale + (
        jnp.where(target_noise_scale > carry.noise_scale, noise_alpha, release_alpha)
        * (target_noise_scale - carry.noise_scale)
    )

    # reward is proportional to the absolute reduction in horizontal distance
    # from the previous step.  using carry.prev_dist directly ensures that
    # starting close to the goal does not inflate the score.
    progress = carry.prev_dist - dist
    reward = progress * PROGRESS_REWARD_SCALE
    reached_goal_now = (~carry.goal_reached) & (dist <= GOAL_REACHED_RADIUS_M)
    reward = reward + jnp.where(reached_goal_now, GOAL_REACHED_BONUS, 0.0)

    imu_roll_rad = _wrap_angle_pi(env_state.body_rot[0])
    side_tip_depth_norm = _side_tip_depth_norm(imu_roll_rad)
    side_tip_depth_delta = carry.prev_side_tip_depth_norm - side_tip_depth_norm
    reward = reward - ((side_tip_depth_norm * side_tip_depth_norm) * SIDE_TIP_DEPTH_PENALTY_SCALE)
    reward = reward + (side_tip_depth_delta * SIDE_TIP_ESCAPE_DELTA_SCALE)
    exited_side_band = (carry.prev_side_tip_depth_norm > 0.0) & (side_tip_depth_norm <= 0.0)
    reward = reward + jnp.where(exited_side_band, SIDE_TIP_EXIT_BONUS, 0.0)

    return EpisodeCarry(
        env_state=env_state,
        brain_state=brain_state,
        key=key,
        total_reward=carry.total_reward + reward,
        initial_dist=carry.initial_dist,
        prev_dist=dist,
        noise_scale=noise_scale,
        fast_closing_rate=fast_closing_rate,
        slow_closing_rate=slow_closing_rate,
        closing_rate=closing_rate,
        progress_drop_ratio=progress_drop_ratio,
        prev_side_tip_depth_norm=side_tip_depth_norm,
        goal_reached=carry.goal_reached | reached_goal_now,
    )


@partial(jax.jit, static_argnames=("steps",))
def _run_episode_flat(params_flat: jax.Array, goal_xyz: jax.Array, key: jax.Array, steps: int) -> jax.Array:
    params = _unflatten_params(params_flat)
    carry = _episode_init(goal_xyz, key)

    def _scan_step(loop_carry: EpisodeCarry, _unused: None) -> tuple[EpisodeCarry, jax.Array]:
        new_carry = _episode_step(params, loop_carry, goal_xyz)
        return new_carry, new_carry.total_reward

    carry, _ = jax.lax.scan(_scan_step, carry, xs=None, length=steps)
    return carry.total_reward


@partial(jax.jit, static_argnames=("steps",))
def _run_episode_batch_flat(params_batch: jax.Array, goal_xyz: jax.Array, keys: jax.Array, steps: int) -> jax.Array:
    return jax.vmap(lambda p, k: _run_episode_flat(p, goal_xyz, k, steps))(params_batch, keys)


@jax.jit
def _episode_step_logged(params_flat: jax.Array, carry: EpisodeCarry, goal_xyz: jax.Array) -> EpisodeCarry:
    return _episode_step(_unflatten_params(params_flat), carry, goal_xyz)


def _step_snapshot(carry: EpisodeCarry, goal_xyz: jax.Array, step: int, total_steps: int) -> dict[str, Any]:
    state = carry.env_state
    mount_world, foot_position, _, _, leg_com_world, _ = _compute_leg_kinematics(
        state.body_pos,
        state.body_vel,
        state.body_rot,
        state.body_ang_vel,
        state.leg_angle,
        state.leg_ang_vel,
        jnp.zeros((N_LEGS,), dtype=jnp.float32),
    )
    body_corners = _body_points_world(state.body_pos, state.body_rot, BODY_CORNERS_BODY)
    com = _center_of_mass(state)

    goal_np = np.asarray(goal_xyz, dtype=np.float32)
    body_pos_np = np.asarray(state.body_pos, dtype=np.float32)
    body_rot_np = np.asarray(state.body_rot, dtype=np.float32)
    body_corners_np = np.asarray(body_corners, dtype=np.float32)
    com_np = np.asarray(com, dtype=np.float32)
    mount_np = np.asarray(mount_world, dtype=np.float32)
    foot_np = np.asarray(foot_position, dtype=np.float32)
    leg_com_np = np.asarray(leg_com_world, dtype=np.float32)
    leg_angle_np = np.asarray(state.leg_angle, dtype=np.float32)
    leg_mode_np = np.asarray(state.leg_contact_mode, dtype=np.int32)
    motor_target_np = np.asarray(state.motor_target, dtype=np.float32)

    legs = []
    for index, name in enumerate(LEG_NAMES):
        legs.append(
            {
                "name": name,
                "mount": mount_np[index].tolist(),
                "foot": foot_np[index].tolist(),
                "com": leg_com_np[index].tolist(),
                "angle_rad": float(leg_angle_np[index]),
                "contact_mode": _contact_mode_name(int(leg_mode_np[index])),
            }
        )

    motors = []
    for index, name in enumerate(LEG_NAMES):
        motors.append(
            {
                "name": name,
                "target_velocity_rad_s": float(motor_target_np[index]),
            }
        )

    return {
        "type": "step",
        "step": step,
        "total_steps": total_steps,
        "reward": float(carry.total_reward),
        "goal": goal_np.tolist(),
        "body": {
            "pos": body_pos_np.tolist(),
            "rot": body_rot_np.tolist(),
            "com": body_pos_np.tolist(),
            "corners": body_corners_np.tolist(),
        },
        "com": com_np.tolist(),
        "legs": legs,
        "motors": motors,
        "noise_scale": float(carry.noise_scale),
        "closing_rate_m_s": float(carry.closing_rate),
        "progress_drop_ratio": float(carry.progress_drop_ratio),
        "time_s": float(state.time_s),
    }


class JaxESTrainer:
    """Headless ES trainer running the network and simulator on a JAX backend."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.state = TrainingState()
        self._key = jax.random.PRNGKey(seed)
        self._key, init_key = jax.random.split(self._key)
        self._params = _init_param_vector(init_key)
        self._top_params = np.zeros((0, PARAM_COUNT), dtype=np.float32)
        self._top_rewards = np.zeros((0,), dtype=np.float32)
        self._top_indices = np.zeros((0,), dtype=np.int32)
        self._top_generations = np.zeros((0,), dtype=np.int32)

    @property
    def backend(self) -> str:
        return jax.default_backend()

    @property
    def device_summary(self) -> str:
        return ", ".join(str(device) for device in jax.devices())

    @property
    def param_count(self) -> int:
        return PARAM_COUNT

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
        self._key, angle_key, radius_key = jax.random.split(self._key, 3)
        angle = jax.random.uniform(angle_key, (), minval=0.0, maxval=2.0 * jnp.pi)
        radius = jax.random.uniform(radius_key, (), minval=0.5, maxval=FIELD_HALF * 0.8)
        return jnp.array(
            [radius * jnp.cos(angle), radius * jnp.sin(angle), GOAL_HEIGHT_M],
            dtype=jnp.float32,
        )

    def _run_logged_episode(self, params_flat: jax.Array, goal_xyz: jax.Array, key: jax.Array, on_step: Any) -> float:
        carry = _episode_init(goal_xyz, key)
        steps = int(EPISODE_S / BRAIN_DT)
        for step_i in range(steps):
            carry = _episode_step_logged(params_flat, carry, goal_xyz)
            on_step(_step_snapshot(carry, goal_xyz, step_i, steps))
        return float(carry.total_reward)

    def run_generation(self, on_step: Any = None, on_gen_done: Any = None) -> None:
        goal_xyz = self._random_goal()
        self.state.goal_xyz = tuple(float(v) for v in np.asarray(goal_xyz).tolist())

        self._key, noise_key, eval_key, display_key = jax.random.split(self._key, 4)
        noise = jax.random.normal(noise_key, (POP_SIZE, PARAM_COUNT), dtype=jnp.float32) * jnp.float32(SIGMA)
        params_batch = self._params[None, :] + noise
        eval_keys = jax.random.split(eval_key, POP_SIZE)

        if on_step is not None:
            self._run_logged_episode(self._params, goal_xyz, display_key, on_step)

        returns = _run_episode_batch_flat(params_batch, goal_xyz, eval_keys, int(EPISODE_S / BRAIN_DT))

        returns_np = np.asarray(returns, dtype=np.float32)
        elite_count = min(PARENT_ELITE_COUNT, returns_np.shape[0])
        top_indices = np.argpartition(returns_np, -elite_count)[-elite_count:]
        top_indices = top_indices[np.argsort(returns_np[top_indices])[::-1]]
        top_params = np.asarray(jnp.take(params_batch, jnp.asarray(top_indices), axis=0), dtype=np.float32)
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

        leaderboard_count = min(PARENT_ELITE_COUNT, combined_rewards.shape[0])
        leaderboard_indices = np.argpartition(combined_rewards, -leaderboard_count)[-leaderboard_count:]
        leaderboard_indices = leaderboard_indices[np.argsort(combined_rewards[leaderboard_indices])[::-1]]
        self._top_params = combined_params[leaderboard_indices].astype(np.float32)
        self._top_rewards = combined_rewards[leaderboard_indices].astype(np.float32)
        self._top_indices = combined_indices[leaderboard_indices].astype(np.int32)
        self._top_generations = combined_generations[leaderboard_indices].astype(np.int32)
        self._params = jnp.asarray(self._top_params.mean(axis=0), dtype=jnp.float32)

        self.state.generation += 1
        self.state.mean_reward = float(returns_np.mean())
        self.state.episode_reward = float(self._top_rewards.mean()) if self._top_rewards.size else 0.0
        self.state.best_reward = max(self.state.best_reward, self.state.episode_reward)
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
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(checkpoint_path, **self.checkpoint_dict())
        return checkpoint_path

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            self._params = jnp.asarray(checkpoint["params"], dtype=jnp.float32)
            self._top_params = checkpoint["top_params"].astype(np.float32) if "top_params" in checkpoint.files else np.zeros((0, PARAM_COUNT), dtype=np.float32)
            self._top_rewards = checkpoint["top_rewards"].astype(np.float32) if "top_rewards" in checkpoint.files else np.zeros((0,), dtype=np.float32)
            self._top_indices = checkpoint["top_indices"].astype(np.int32) if "top_indices" in checkpoint.files else np.zeros((0,), dtype=np.int32)
            self.state.generation = int(checkpoint["generation"])
            if "top_generations" in checkpoint.files:
                self._top_generations = checkpoint["top_generations"].astype(np.int32)
            else:
                self._top_generations = np.full(
                    (self._top_rewards.shape[0],),
                    self.state.generation,
                    dtype=np.int32,
                )
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


ESTrainer = JaxESTrainer

__all__ = [
    "EPISODE_S",
    "POP_SIZE",
    "TrainingState",
    "JaxESTrainer",
    "ESTrainer",
]
