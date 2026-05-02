"""Runtime validation and quality gates."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import brains.jax_trainer as trainer_module
from brains.config import RuntimeSpec
from brains.sim.mujoco_backend import MuJoCoBackend


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


class QualityGateRunner:
    """Fast validation suite used before long-running training."""

    def __init__(self, spec: RuntimeSpec) -> None:
        self.spec = spec

    def run(self) -> QualityReport:
        return self._run_mujoco()

    def _run_mujoco(self) -> QualityReport:
        backend = MuJoCoBackend(self.spec)
        results = [
            self._mujoco_model_compile_gate(backend),
            self._mujoco_reset_pose_gate(backend),
            self._mujoco_zero_action_gate(backend),
            self._mujoco_determinism_gate(backend),
            self._mujoco_performance_gate(backend),
        ]
        return QualityReport(spec_name=self.spec.name, results=results)

    def _mujoco_model_compile_gate(self, backend: MuJoCoBackend) -> GateResult:
        body_count = int(backend.model.nbody)
        geom_count = int(backend.model.ngeom)
        actuator_count = int(backend.model.nu)
        passed = body_count > 0 and geom_count > 0 and actuator_count == backend.leg_count
        return GateResult(
            name="mujoco_model_compile",
            passed=passed,
            details={
                "body_count": body_count,
                "geom_count": geom_count,
                "actuator_count": actuator_count,
                "expected_actuators": backend.leg_count,
            },
        )

    def _mujoco_reset_pose_gate(self, backend: MuJoCoBackend) -> GateResult:
        spawn_xy = np.asarray(_first_spawn(self.spec), dtype=np.float32)
        data = backend.reset_data(spawn_xy=spawn_xy)
        body_pos = backend.body_position(data)
        foot_positions = backend.foot_positions(data)
        spawn_floor = backend.floor_height_at(spawn_xy)
        body_bottom = min(float(corner[2]) for corner in backend._body_corners_world(data))
        foot_clearances = [backend.foot_clearance_m(foot_position) for foot_position in foot_positions]
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
        zero_target = np.zeros((backend.leg_count,), dtype=np.float32)
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
        params = trainer_module._require_policy_runtime().init_param_vector(jax.random.PRNGKey(101))
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

    def _mujoco_performance_gate(self, backend: MuJoCoBackend) -> GateResult:
        eval_runs = self.spec.quality_gates.performance_eval_runs
        steps = min(
            self.spec.quality_gates.performance_steps,
            max(1, int(self.spec.episode.episode_s / self.spec.episode.brain_dt_s)),
        )
        params = trainer_module._require_policy_runtime().init_param_vector(jax.random.PRNGKey(505))
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


def _select_regression_baseline_variant(baseline: dict[str, Any]) -> tuple[dict[str, Any], str]:
    variants = baseline.get("variants")
    if not isinstance(variants, dict):
        return baseline, "root"

    system_key = platform.system().lower()
    machine_key = platform.machine().lower()
    exact_key = f"{system_key}-{machine_key}"
    value = variants.get(exact_key)
    if isinstance(value, dict):
        return value, exact_key

    system_key_value = variants.get(system_key)
    if isinstance(system_key_value, dict):
        return system_key_value, system_key

    # If there is no exact arch match, prefer a same-system variant (e.g., linux-x86_64).
    # This keeps fallback behavior closer to platform-specific dynamics than "default".
    system_variants = [
        key for key, maybe_variant in variants.items()
        if key.startswith(f"{system_key}-") and isinstance(maybe_variant, dict)
    ]
    if system_variants:
        selected_key = sorted(system_variants)[0]
        selected_variant = variants[selected_key]
        if isinstance(selected_variant, dict):
            return selected_variant, selected_key

    default_value = variants.get("default")
    if isinstance(default_value, dict):
        return default_value, "default"

    raise ValueError(
        "Regression baseline file defines platform variants but none matched the current platform. "
        f"Tried exact='{exact_key}', system='{system_key}', and default/system-prefix fallbacks."
    )


def compare_regression_to_baseline(
    spec: RuntimeSpec,
    baseline_path: str | Path,
    seeds: list[int],
    generations: int = 1,
    atol: float = 1e-5,
) -> QualityReport:
    baseline_raw = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    baseline, baseline_variant_key = _select_regression_baseline_variant(baseline_raw)
    current = collect_regression_metrics(spec, seeds=seeds, generations=generations)
    system_key = platform.system().lower()
    machine_key = platform.machine().lower()
    exact_variant_key = f"{system_key}-{machine_key}"
    exact_variant_match = baseline_variant_key in {"root", exact_variant_key}

    # Allow a minimal cross-platform slack only when the baseline is a fallback variant.
    effective_atol = atol if exact_variant_match else max(atol, 1e-4)
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
            and max_abs_diff <= effective_atol
            and goal_diff <= effective_atol
        )
        results.append(
            GateResult(
                name=f"regression_seed_{seed}",
                passed=passed,
                details={
                    "seed": seed,
                    "atol": atol,
                    "effective_atol": effective_atol,
                    "baseline_variant_key": baseline_variant_key,
                    "exact_variant_match": exact_variant_match,
                    "expected": expected,
                    "actual": actual,
                    "max_abs_metric_diff": max_abs_diff,
                    "max_abs_goal_diff": goal_diff,
                },
            )
        )
    return QualityReport(spec_name=spec.name, results=results)
