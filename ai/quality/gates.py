"""Runtime validation and quality gates."""

from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import ai.jax_trainer as trainer_module
from ai.config import RuntimeSpec, load_runtime_spec
from ai.sim.jax_backend import JaxSimBackend
from ai.sim.mujoco_backend import MuJoCoBackend
from ai.sim.translators import terrain_height_at


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    details: dict[str, Any]


@dataclass(frozen=True)
class QualityReport:
    spec_name: str
    results: list[GateResult]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_name": self.spec_name,
            "passed": self.passed,
            "results": [asdict(result) for result in self.results],
        }


def _first_goal(spec: RuntimeSpec) -> jax.Array:
    if spec.goals.strategy == "fixed" and spec.goals.fixed_goal_xyz is not None:
        return jnp.asarray(spec.goals.fixed_goal_xyz, dtype=jnp.float32)
    return jnp.asarray([spec.goals.radius_m, 0.0, spec.goals.height_m], dtype=jnp.float32)


def _first_spawn(spec: RuntimeSpec) -> jax.Array:
    if spec.spawn_policy.strategy == "origin":
        return jnp.zeros((2,), dtype=jnp.float32)
    if spec.spawn_policy.strategy == "fixed_points" and spec.spawn_policy.fixed_points:
        return jnp.asarray(spec.spawn_policy.fixed_points[0], dtype=jnp.float32)
    x_min, x_max = spec.spawn_policy.x_range_m
    y_min, y_max = spec.spawn_policy.y_range_m
    return jnp.asarray([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5], dtype=jnp.float32)


def _sample_spawn_points(spec: RuntimeSpec, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    rng = np.random.default_rng(20260403)
    if spec.spawn_policy.strategy == "origin":
        return np.zeros((count, 2), dtype=np.float32)
    if spec.spawn_policy.strategy == "fixed_points":
        points = np.asarray(spec.spawn_policy.fixed_points, dtype=np.float32)
        repeats = math.ceil(count / len(points))
        return np.tile(points, (repeats, 1))[:count]
    x_min, x_max = spec.spawn_policy.x_range_m
    y_min, y_max = spec.spawn_policy.y_range_m
    x_values = rng.uniform(x_min, x_max, size=(count, 1))
    y_values = rng.uniform(y_min, y_max, size=(count, 1))
    return np.concatenate([x_values, y_values], axis=1).astype(np.float32)


def _tree_max_abs_diff(left: Any, right: Any) -> float:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    max_diff = 0.0
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_np = np.asarray(left_leaf)
        right_np = np.asarray(right_leaf)
        if left_np.dtype == np.bool_ or right_np.dtype == np.bool_:
            if not np.array_equal(left_np, right_np):
                return float("inf")
            continue
        diff = float(np.max(np.abs(left_np - right_np))) if left_np.size else 0.0
        max_diff = max(max_diff, diff)
    return max_diff


class QualityGateRunner:
    """Fast validation suite used before long-running training."""

    def __init__(self, spec: RuntimeSpec) -> None:
        self.spec = spec

    def run(self) -> QualityReport:
        if self.spec.quality_gates.profile == "runtime":
            return self._run_mujoco()
        trainer_module.apply_runtime_spec(self.spec)
        results = [
            self._spawn_validity_gate(),
            self._collision_sanity_gate(),
            self._determinism_gate(),
            self._unstable_state_gate(),
            self._performance_gate(),
        ]
        return QualityReport(spec_name=self.spec.name, results=results)

    def _run_mujoco(self) -> QualityReport:
        backend = MuJoCoBackend(self.spec)
        results = [
            self._mujoco_model_compile_gate(backend),
            self._mujoco_reset_pose_gate(backend),
            self._mujoco_zero_action_gate(backend),
            self._mujoco_determinism_gate(backend),
            self._mujoco_cross_backend_gate(backend),
            self._mujoco_performance_gate(backend),
        ]
        return QualityReport(spec_name=self.spec.name, results=results)

    def _collision_sanity_gate(self) -> GateResult:
        spawn_xy = _first_spawn(self.spec)
        state = trainer_module._env_reset(spawn_xy)
        zero_target = jnp.zeros((trainer_module.N_LEGS,), dtype=jnp.float32)
        for _ in range(self.spec.quality_gates.collision_sanity_steps):
            state = trainer_module._env_advance(state, zero_target)

        _, foot_position, _, _, _, _ = trainer_module._compute_leg_kinematics(
            state.body_pos,
            state.body_vel,
            state.body_rot,
            state.body_ang_vel,
            state.leg_angle,
            state.leg_ang_vel,
            jnp.zeros((trainer_module.N_LEGS,), dtype=jnp.float32),
        )
        body_corners = trainer_module._body_points_world(state.body_pos, state.body_rot, trainer_module.BODY_CORNERS_BODY)
        foot_floor = np.asarray(jax.vmap(trainer_module._terrain_height_at)(foot_position[:, :2]), dtype=np.float32)
        body_floor = np.asarray(jax.vmap(trainer_module._terrain_height_at)(body_corners[:, :2]), dtype=np.float32)

        foot_clearance = np.asarray(foot_position[:, 2], dtype=np.float32) - foot_floor - trainer_module.FOOT_RADIUS_M
        body_penetration = body_floor - np.asarray(body_corners[:, 2], dtype=np.float32)

        max_body_penetration = float(np.max(body_penetration))
        max_abs_foot_clearance = float(np.max(np.abs(foot_clearance)))
        finite = np.isfinite(np.asarray(state.body_pos)).all() and np.isfinite(np.asarray(foot_position)).all()
        passed = finite and max_body_penetration <= 5e-3 and max_abs_foot_clearance <= 3e-2
        return GateResult(
            name="collision_sanity",
            passed=passed,
            details={
                "finite": bool(finite),
                "max_body_penetration_m": max_body_penetration,
                "max_abs_foot_clearance_m": max_abs_foot_clearance,
            },
        )

    def _determinism_gate(self) -> GateResult:
        steps = min(
            self.spec.quality_gates.determinism_steps,
            max(1, int(self.spec.episode.episode_s / self.spec.episode.brain_dt_s)),
        )
        key = jax.random.PRNGKey(101)
        params = trainer_module._init_param_vector(key)
        goal = _first_goal(self.spec)
        spawn_xy = _first_spawn(self.spec)
        run_key = jax.random.PRNGKey(202)

        carry_a = trainer_module._episode_init(goal, run_key, spawn_xy)
        carry_b = trainer_module._episode_init(goal, run_key, spawn_xy)
        for _ in range(steps):
            carry_a = trainer_module._episode_step_logged(params, carry_a, goal)
            carry_b = trainer_module._episode_step_logged(params, carry_b, goal)

        max_diff = _tree_max_abs_diff(carry_a, carry_b)
        passed = max_diff <= self.spec.quality_gates.determinism_tolerance
        return GateResult(
            name="determinism",
            passed=passed,
            details={
                "steps": steps,
                "max_abs_diff": max_diff,
                "tolerance": self.spec.quality_gates.determinism_tolerance,
            },
        )

    def _unstable_state_gate(self) -> GateResult:
        steps = self.spec.quality_gates.unstable_state_steps
        key = jax.random.PRNGKey(303)
        params = trainer_module._init_param_vector(key)
        goal = _first_goal(self.spec)
        spawn_xy = _first_spawn(self.spec)
        carry = trainer_module._episode_init(goal, jax.random.PRNGKey(404), spawn_xy)

        max_height = float(np.asarray(carry.env_state.body_pos[2]))
        max_rotation = float(np.max(np.abs(np.asarray(carry.env_state.body_rot))))
        finite = True
        for _ in range(steps):
            carry = trainer_module._episode_step_logged(params, carry, goal)
            body_pos = np.asarray(carry.env_state.body_pos, dtype=np.float32)
            body_rot = np.asarray(carry.env_state.body_rot, dtype=np.float32)
            finite = finite and np.isfinite(body_pos).all() and np.isfinite(body_rot).all()
            max_height = max(max_height, float(body_pos[2]))
            max_rotation = max(max_rotation, float(np.max(np.abs(body_rot))))

        passed = (
            finite
            and max_height <= self.spec.quality_gates.max_body_height_m
            and max_rotation <= self.spec.quality_gates.max_abs_body_rotation_rad
        )
        return GateResult(
            name="unstable_state_detection",
            passed=passed,
            details={
                "steps": steps,
                "finite": bool(finite),
                "max_body_height_m": max_height,
                "max_abs_body_rotation_rad": max_rotation,
                "height_limit_m": self.spec.quality_gates.max_body_height_m,
                "rotation_limit_rad": self.spec.quality_gates.max_abs_body_rotation_rad,
            },
        )

    def _performance_gate(self) -> GateResult:
        eval_runs = self.spec.quality_gates.performance_eval_runs
        steps = min(
            self.spec.quality_gates.performance_steps,
            max(1, int(self.spec.episode.episode_s / self.spec.episode.brain_dt_s)),
        )
        params = trainer_module._init_param_vector(jax.random.PRNGKey(505))
        goal = _first_goal(self.spec)
        spawn_xy = _first_spawn(self.spec)

        for warmup_index in range(self.spec.quality_gates.performance_warmup_runs):
            warmup_key = jax.random.PRNGKey(600 + warmup_index)
            trainer_module._run_episode_flat(params, goal, warmup_key, steps, spawn_xy).block_until_ready()

        start = time.perf_counter()
        last_reward = 0.0
        for eval_index in range(max(eval_runs, 1)):
            eval_key = jax.random.PRNGKey(700 + eval_index)
            last_reward = float(trainer_module._run_episode_flat(params, goal, eval_key, steps, spawn_xy).block_until_ready())
        elapsed = time.perf_counter() - start
        budget = self.spec.quality_gates.performance_budget_seconds
        passed = eval_runs == 0 or elapsed <= budget
        per_run = elapsed / max(eval_runs, 1)
        return GateResult(
            name="performance_budget",
            passed=passed,
            details={
                "steps": steps,
                "eval_runs": eval_runs,
                "elapsed_s": elapsed,
                "per_run_s": per_run,
                "budget_s": budget,
                "last_reward": last_reward,
            },
        )

    def _spawn_validity_gate(self) -> GateResult:
        samples = _sample_spawn_points(self.spec, self.spec.quality_gates.spawn_samples)
        valid = True
        max_body_penetration = 0.0
        for spawn_point in samples:
            if abs(float(spawn_point[0])) > self.spec.terrain.field_half_m or abs(float(spawn_point[1])) > self.spec.terrain.field_half_m:
                valid = False
                break
            state = trainer_module._env_reset(jnp.asarray(spawn_point, dtype=jnp.float32))
            body_corners = trainer_module._body_points_world(state.body_pos, state.body_rot, trainer_module.BODY_CORNERS_BODY)
            body_floor = np.asarray(jax.vmap(trainer_module._terrain_height_at)(body_corners[:, :2]), dtype=np.float32)
            body_penetration = np.max(body_floor - np.asarray(body_corners[:, 2], dtype=np.float32))
            max_body_penetration = max(max_body_penetration, float(body_penetration))
            if not np.isfinite(np.asarray(state.body_pos)).all():
                valid = False
                break
        passed = valid and max_body_penetration <= 5e-3
        return GateResult(
            name="spawn_validity",
            passed=passed,
            details={
                "samples": int(samples.shape[0]),
                "max_body_penetration_m": max_body_penetration,
            },
        )

    def _mujoco_model_compile_gate(self, backend: MuJoCoBackend) -> GateResult:
        body_count = int(backend.model.nbody)
        geom_count = int(backend.model.ngeom)
        actuator_count = int(backend.model.nu)
        passed = body_count > 0 and geom_count > 0 and actuator_count == len(backend.robot.legs)
        return GateResult(
            name="mujoco_model_compile",
            passed=passed,
            details={
                "body_count": body_count,
                "geom_count": geom_count,
                "actuator_count": actuator_count,
                "expected_actuators": len(backend.robot.legs),
            },
        )

    def _mujoco_reset_pose_gate(self, backend: MuJoCoBackend) -> GateResult:
        spawn_xy = np.asarray(_first_spawn(self.spec), dtype=np.float32)
        data = backend.reset_data(spawn_xy=spawn_xy)
        body_pos = backend.body_position(data)
        foot_positions = backend.foot_positions(data)
        spawn_floor = terrain_height_at(self.spec, spawn_xy.tolist())
        body_bottom = float(body_pos[2] - backend.robot.body.height_m * 0.5)
        foot_clearances = [
            float(foot_position[2] - terrain_height_at(self.spec, foot_position[:2].tolist()) - backend.robot.legs[0].foot_radius_m)
            for foot_position in foot_positions
        ]
        max_abs_foot_clearance = max(abs(value) for value in foot_clearances)
        passed = body_bottom >= (spawn_floor - 5e-3) and max_abs_foot_clearance <= 6e-2
        return GateResult(
            name="mujoco_reset_pose",
            passed=passed,
            details={
                "spawn_floor_m": spawn_floor,
                "body_bottom_m": body_bottom,
                "max_abs_foot_clearance_m": max_abs_foot_clearance,
            },
        )

    def _mujoco_zero_action_gate(self, backend: MuJoCoBackend) -> GateResult:
        data = backend.reset_data(spawn_xy=np.asarray(_first_spawn(self.spec), dtype=np.float32))
        max_height = float(backend.body_position(data)[2])
        max_rotation = float(np.max(np.abs(backend.body_rotation(data))))
        finite = True
        zero_target = np.zeros((len(backend.robot.legs),), dtype=np.float32)
        for _ in range(self.spec.quality_gates.unstable_state_steps):
            backend._advance(data, zero_target)
            body_pos = backend.body_position(data)
            body_rot = backend.body_rotation(data)
            finite = finite and np.isfinite(body_pos).all() and np.isfinite(body_rot).all()
            max_height = max(max_height, float(body_pos[2]))
            max_rotation = max(max_rotation, float(np.max(np.abs(body_rot))))
        passed = (
            finite
            and max_height <= self.spec.quality_gates.max_body_height_m
            and max_rotation <= self.spec.quality_gates.max_abs_body_rotation_rad
        )
        return GateResult(
            name="mujoco_zero_action_stability",
            passed=passed,
            details={
                "finite": bool(finite),
                "max_body_height_m": max_height,
                "max_abs_body_rotation_rad": max_rotation,
                "height_limit_m": self.spec.quality_gates.max_body_height_m,
                "rotation_limit_rad": self.spec.quality_gates.max_abs_body_rotation_rad,
            },
        )

    def _mujoco_determinism_gate(self, backend: MuJoCoBackend) -> GateResult:
        steps = min(
            self.spec.quality_gates.determinism_steps,
            max(1, int(self.spec.episode.episode_s / self.spec.episode.brain_dt_s)),
        )
        params = trainer_module._init_param_vector(jax.random.PRNGKey(101))
        goal = _first_goal(self.spec)
        spawn_xy = _first_spawn(self.spec)
        run_key = jax.random.PRNGKey(202)
        snapshots_a: list[np.ndarray] = []
        snapshots_b: list[np.ndarray] = []

        def _capture_a(message: dict[str, Any]) -> None:
            snapshots_a.append(
                np.asarray(
                    message["body"]["pos"] + message["body"]["rot"] + [message["reward"]],
                    dtype=np.float32,
                )
            )

        def _capture_b(message: dict[str, Any]) -> None:
            snapshots_b.append(
                np.asarray(
                    message["body"]["pos"] + message["body"]["rot"] + [message["reward"]],
                    dtype=np.float32,
                )
            )

        reward_a = backend.run_logged_episode(params, goal, run_key, _capture_a, steps, spawn_xy=spawn_xy)
        reward_b = backend.run_logged_episode(params, goal, run_key, _capture_b, steps, spawn_xy=spawn_xy)
        max_diff = abs(float(reward_a) - float(reward_b))
        if len(snapshots_a) != len(snapshots_b):
            max_diff = float("inf")
        else:
            for left, right in zip(snapshots_a, snapshots_b, strict=True):
                max_diff = max(max_diff, float(np.max(np.abs(left - right))))
        passed = max_diff <= self.spec.quality_gates.determinism_tolerance
        return GateResult(
            name="mujoco_determinism",
            passed=passed,
            details={
                "steps": steps,
                "max_abs_diff": max_diff,
                "tolerance": self.spec.quality_gates.determinism_tolerance,
            },
        )

    def _mujoco_cross_backend_gate(self, backend: MuJoCoBackend) -> GateResult:
        steps = min(
            self.spec.quality_gates.collision_sanity_steps,
            max(1, int(self.spec.episode.episode_s / self.spec.episode.brain_dt_s)),
        )
        goal = _first_goal(self.spec)
        spawn_xy = _first_spawn(self.spec)
        params = trainer_module._init_param_vector(jax.random.PRNGKey(303))
        key = jax.random.PRNGKey(404)
        jax_backend = JaxSimBackend(self.spec)
        jax_trace: list[np.ndarray] = []
        mujoco_trace: list[np.ndarray] = []

        def _capture_jax(message: dict[str, Any]) -> None:
            jax_trace.append(np.asarray(message["com"] + [message["reward"]], dtype=np.float32))

        def _capture_mujoco(message: dict[str, Any]) -> None:
            mujoco_trace.append(np.asarray(message["com"] + [message["reward"]], dtype=np.float32))

        jax_reward = jax_backend.run_logged_episode(params, goal, key, _capture_jax, steps, spawn_xy=spawn_xy)
        mujoco_reward = backend.run_logged_episode(params, goal, key, _capture_mujoco, steps, spawn_xy=spawn_xy)
        reward_delta = abs(float(jax_reward) - float(mujoco_reward))
        com_delta = float("inf")
        if jax_trace and mujoco_trace and len(jax_trace) == len(mujoco_trace):
            com_delta = max(
                float(np.max(np.abs(left[:3] - right[:3])))
                for left, right in zip(jax_trace, mujoco_trace, strict=True)
            )
        passed = np.isfinite(reward_delta) and np.isfinite(com_delta) and reward_delta <= 40.0 and com_delta <= 0.35
        return GateResult(
            name="cross_backend_smoke",
            passed=passed,
            details={
                "steps": steps,
                "reward_delta": reward_delta,
                "max_com_delta_m": com_delta,
                "reward_delta_limit": 40.0,
                "com_delta_limit_m": 0.35,
            },
        )

    def _mujoco_performance_gate(self, backend: MuJoCoBackend) -> GateResult:
        eval_runs = self.spec.quality_gates.performance_eval_runs
        steps = min(
            self.spec.quality_gates.performance_steps,
            max(1, int(self.spec.episode.episode_s / self.spec.episode.brain_dt_s)),
        )
        params = trainer_module._init_param_vector(jax.random.PRNGKey(505))
        goal = _first_goal(self.spec)
        spawn_xy = _first_spawn(self.spec)

        for warmup_index in range(self.spec.quality_gates.performance_warmup_runs):
            warmup_key = jax.random.PRNGKey(600 + warmup_index)
            backend.run_episode(params, goal, warmup_key, steps, spawn_xy=spawn_xy)

        start = time.perf_counter()
        last_reward = 0.0
        for eval_index in range(max(eval_runs, 1)):
            eval_key = jax.random.PRNGKey(700 + eval_index)
            last_reward = float(backend.run_episode(params, goal, eval_key, steps, spawn_xy=spawn_xy))
        elapsed = time.perf_counter() - start
        budget = self.spec.quality_gates.performance_budget_seconds
        passed = eval_runs == 0 or elapsed <= budget
        per_run = elapsed / max(eval_runs, 1)
        return GateResult(
            name="mujoco_performance_budget",
            passed=passed,
            details={
                "steps": steps,
                "eval_runs": eval_runs,
                "elapsed_s": elapsed,
                "per_run_s": per_run,
                "budget_s": budget,
                "last_reward": last_reward,
            },
        )


def collect_regression_metrics(spec: RuntimeSpec, seeds: list[int], generations: int = 1) -> dict[str, Any]:
    trainer_module.apply_runtime_spec(spec)
    metrics: dict[str, Any] = {"spec_name": spec.name, "generations": generations, "seeds": {}}
    for seed in seeds:
        trainer = trainer_module.JaxESTrainer(seed=seed, spec=spec)
        for _ in range(generations):
            trainer.run_generation()
        top_reward = float(trainer.top_rewards[0]) if trainer.top_rewards.size else float("-inf")
        metrics["seeds"][str(seed)] = {
            "generation": trainer.state.generation,
            "mean_reward": trainer.state.mean_reward,
            "best_reward": trainer.state.best_reward,
            "best_single_reward": trainer.state.best_single_reward,
            "top_reward": top_reward,
            "goal_xyz": list(trainer.state.goal_xyz),
        }
    return metrics


def write_regression_baseline(path: str | Path, metrics: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return target


def _select_regression_baseline_variant(baseline: dict[str, Any]) -> dict[str, Any]:
    variants = baseline.get("variants")
    if not isinstance(variants, dict):
        return baseline

    system_key = platform.system().lower()
    machine_key = platform.machine().lower()
    candidate_keys = (
        f"{system_key}-{machine_key}",
        system_key,
        "default",
    )
    for key in candidate_keys:
        value = variants.get(key)
        if isinstance(value, dict):
            return value
    raise ValueError(
        "Regression baseline file defines platform variants but none matched the current platform. "
        f"Tried: {', '.join(candidate_keys)}"
    )


def compare_regression_to_baseline(
    spec: RuntimeSpec,
    baseline_path: str | Path,
    seeds: list[int],
    generations: int = 1,
    atol: float = 1e-5,
) -> QualityReport:
    baseline_raw = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    baseline = _select_regression_baseline_variant(baseline_raw)
    current = collect_regression_metrics(spec, seeds=seeds, generations=generations)
    results: list[GateResult] = []
    for seed in seeds:
        expected = baseline["seeds"][str(seed)]
        actual = current["seeds"][str(seed)]
        max_abs_diff = 0.0
        for key in ("mean_reward", "best_reward", "best_single_reward", "top_reward"):
            max_abs_diff = max(max_abs_diff, abs(float(actual[key]) - float(expected[key])))
        goal_diff = max(abs(float(a) - float(b)) for a, b in zip(actual["goal_xyz"], expected["goal_xyz"], strict=True))
        passed = (
            int(actual["generation"]) == int(expected["generation"])
            and max_abs_diff <= atol
            and goal_diff <= atol
        )
        results.append(
            GateResult(
                name=f"regression_seed_{seed}",
                passed=passed,
                details={
                    "seed": seed,
                    "atol": atol,
                    "expected": expected,
                    "actual": actual,
                    "max_abs_metric_diff": max_abs_diff,
                    "max_abs_goal_diff": goal_diff,
                },
            )
        )
    return QualityReport(spec_name=spec.name, results=results)


def load_spec_and_run_quality(path: str | Path | None = None) -> QualityReport:
    return QualityGateRunner(load_runtime_spec(path)).run()
