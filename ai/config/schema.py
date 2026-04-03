"""Typed runtime configuration for the quadruped trainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


def _as_float_pair(value: Any, field_name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must be a 2-item list or tuple.")
    return float(value[0]), float(value[1])


def _as_points(value: Any, field_name: str) -> tuple[tuple[float, float], ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of [x, y] points.")
    points: list[tuple[float, float]] = []
    for index, point in enumerate(value):
        points.append(_as_float_pair(point, f"{field_name}[{index}]"))
    return tuple(points)


def _as_float_triple(value: Any, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must be a 3-item list or tuple.")
    return float(value[0]), float(value[1]), float(value[2])


def _merge_section(default_value: Any, override: Any) -> Any:
    if override is None:
        return default_value
    if isinstance(default_value, Mapping) and isinstance(override, Mapping):
        merged = dict(default_value)
        for key, value in override.items():
            merged[key] = _merge_section(default_value.get(key), value)
        return merged
    return override


@dataclass(frozen=True)
class TerrainSpec:
    kind: str = "stepped_arena"
    field_half_m: float = 15.0
    center_half_m: float = 2.5
    step_count: int = 5
    step_width_m: float = 2.0
    step_height_m: float = 0.15
    floor_height_m: float = 0.0

    def validate(self) -> None:
        if self.kind not in {"stepped_arena", "flat"}:
            raise ValueError("terrain.kind must be 'stepped_arena' or 'flat'.")
        if self.field_half_m <= 0.0:
            raise ValueError("terrain.field_half_m must be > 0.")
        if self.kind == "stepped_arena":
            if self.center_half_m <= 0.0:
                raise ValueError("terrain.center_half_m must be > 0 for stepped arenas.")
            if self.step_count < 0:
                raise ValueError("terrain.step_count must be >= 0.")
            if self.step_width_m <= 0.0:
                raise ValueError("terrain.step_width_m must be > 0.")
            if self.step_height_m < 0.0:
                raise ValueError("terrain.step_height_m must be >= 0.")


@dataclass(frozen=True)
class GoalSpec:
    strategy: str = "radial_random"
    radius_m: float = 10.0
    height_m: float = 0.16
    fixed_goal_xyz: tuple[float, float, float] | None = None

    def validate(self) -> None:
        if self.strategy not in {"radial_random", "fixed"}:
            raise ValueError("goals.strategy must be 'radial_random' or 'fixed'.")
        if self.radius_m < 0.0:
            raise ValueError("goals.radius_m must be >= 0.")
        if self.strategy == "fixed":
            if self.fixed_goal_xyz is None or len(self.fixed_goal_xyz) != 3:
                raise ValueError("goals.fixed_goal_xyz must be set when goals.strategy == 'fixed'.")


@dataclass(frozen=True)
class SpawnPolicySpec:
    strategy: str = "uniform_box"
    x_range_m: tuple[float, float] = (-2.0, 2.0)
    y_range_m: tuple[float, float] = (-2.0, 2.0)
    fixed_points: tuple[tuple[float, float], ...] = ()

    def validate(self, terrain: TerrainSpec) -> None:
        if self.strategy not in {"uniform_box", "fixed_points", "origin"}:
            raise ValueError("spawn_policy.strategy must be 'uniform_box', 'fixed_points', or 'origin'.")
        if self.x_range_m[0] > self.x_range_m[1]:
            raise ValueError("spawn_policy.x_range_m must be an ordered [min, max] pair.")
        if self.y_range_m[0] > self.y_range_m[1]:
            raise ValueError("spawn_policy.y_range_m must be an ordered [min, max] pair.")
        for axis_name, axis_range in (("x", self.x_range_m), ("y", self.y_range_m)):
            if max(abs(axis_range[0]), abs(axis_range[1])) > terrain.field_half_m:
                raise ValueError(f"spawn_policy.{axis_name}_range_m must fit inside terrain.field_half_m.")
        if self.strategy == "fixed_points":
            if not self.fixed_points:
                raise ValueError("spawn_policy.fixed_points must be non-empty for fixed_points strategy.")
            for x_value, y_value in self.fixed_points:
                if abs(x_value) > terrain.field_half_m or abs(y_value) > terrain.field_half_m:
                    raise ValueError("spawn_policy.fixed_points must fit inside terrain.field_half_m.")


@dataclass(frozen=True)
class FrictionSpec:
    foot_static: float = 0.9
    foot_kinetic: float = 0.65
    body: float = 0.35

    def validate(self) -> None:
        if self.foot_static < 0.0 or self.foot_kinetic < 0.0 or self.body < 0.0:
            raise ValueError("friction coefficients must be >= 0.")
        if self.foot_kinetic > self.foot_static:
            raise ValueError("friction.foot_kinetic should not exceed friction.foot_static.")


@dataclass(frozen=True)
class RobotSpec:
    body_length_m: float = 0.28
    body_width_m: float = 0.12
    body_height_m: float = 0.02
    body_mass_kg: float = 2.4
    leg_length_m: float = 0.16
    leg_mass_kg: float = 0.6
    leg_radius_m: float = 0.010
    foot_radius_m: float = 0.010
    elastic_deformation_m: float = 0.002
    leg_body_samples: int = 3
    motor_scale: float = 6.0
    max_motor_rad_s: float = 8.0
    motor_max_angular_acceleration_rad_s2: float = 18.0
    motor_viscous_damping_per_s: float = 8.0
    motor_velocity_filter_tau_s: float = 0.05

    def validate(self) -> None:
        positive_fields = {
            "body_length_m": self.body_length_m,
            "body_width_m": self.body_width_m,
            "body_height_m": self.body_height_m,
            "body_mass_kg": self.body_mass_kg,
            "leg_length_m": self.leg_length_m,
            "leg_mass_kg": self.leg_mass_kg,
            "leg_radius_m": self.leg_radius_m,
            "foot_radius_m": self.foot_radius_m,
            "motor_scale": self.motor_scale,
            "max_motor_rad_s": self.max_motor_rad_s,
            "motor_max_angular_acceleration_rad_s2": self.motor_max_angular_acceleration_rad_s2,
            "motor_viscous_damping_per_s": self.motor_viscous_damping_per_s,
            "motor_velocity_filter_tau_s": self.motor_velocity_filter_tau_s,
        }
        for field_name, value in positive_fields.items():
            if value <= 0.0:
                raise ValueError(f"robot.{field_name} must be > 0.")
        if self.elastic_deformation_m < 0.0:
            raise ValueError("robot.elastic_deformation_m must be >= 0.")
        if self.leg_body_samples <= 0:
            raise ValueError("robot.leg_body_samples must be > 0.")


@dataclass(frozen=True)
class PhysicsSpec:
    gravity_m_s2: float = 9.81
    normal_stiffness_n_m: float = 20000.0
    normal_damping_n_s_m: float = 3500.0
    tangential_stiffness_n_m: float = 7000.0
    tangential_damping_n_s_m: float = 450.0
    angular_damping_n_m_s: float = 8.0
    linear_damping_n_s_m: float = 80.0
    airborne_linear_damping_n_s_m: float = 3.0
    airborne_angular_damping_n_m_s: float = 1.0
    max_contact_force_n: float = 120.0
    max_substep_s: float = 1.0 / 4000.0
    unloading_stiffness_scale: float = 0.4
    sleep_linear_speed_threshold_m_s: float = 0.01
    sleep_angular_speed_threshold_rad_s: float = 0.06

    def validate(self) -> None:
        positive_fields = {
            "gravity_m_s2": self.gravity_m_s2,
            "normal_stiffness_n_m": self.normal_stiffness_n_m,
            "normal_damping_n_s_m": self.normal_damping_n_s_m,
            "tangential_stiffness_n_m": self.tangential_stiffness_n_m,
            "tangential_damping_n_s_m": self.tangential_damping_n_s_m,
            "angular_damping_n_m_s": self.angular_damping_n_m_s,
            "linear_damping_n_s_m": self.linear_damping_n_s_m,
            "airborne_linear_damping_n_s_m": self.airborne_linear_damping_n_s_m,
            "airborne_angular_damping_n_m_s": self.airborne_angular_damping_n_m_s,
            "max_contact_force_n": self.max_contact_force_n,
            "max_substep_s": self.max_substep_s,
            "sleep_linear_speed_threshold_m_s": self.sleep_linear_speed_threshold_m_s,
            "sleep_angular_speed_threshold_rad_s": self.sleep_angular_speed_threshold_rad_s,
        }
        for field_name, value in positive_fields.items():
            if value <= 0.0:
                raise ValueError(f"physics.{field_name} must be > 0.")
        if not 0.0 < self.unloading_stiffness_scale <= 1.0:
            raise ValueError("physics.unloading_stiffness_scale must be in (0, 1].")


@dataclass(frozen=True)
class EpisodeRulesSpec:
    neuron_dt_s: float = 0.010
    brain_dt_s: float = 0.050
    episode_s: float = 30.0
    single_view_episode_s: float = 120.0
    default_lifespan_s: float = 30.0
    tipped_kill_time_s: float = 5.0
    selection_interval_s: float = 15.0
    lifespan_bonus_s: float = 20.0
    selection_top_frac: float = 0.10
    selection_bot_frac: float = 0.10
    goal_reached_radius_m: float = 0.5

    def validate(self) -> None:
        positive_fields = {
            "neuron_dt_s": self.neuron_dt_s,
            "brain_dt_s": self.brain_dt_s,
            "episode_s": self.episode_s,
            "single_view_episode_s": self.single_view_episode_s,
            "default_lifespan_s": self.default_lifespan_s,
            "tipped_kill_time_s": self.tipped_kill_time_s,
            "selection_interval_s": self.selection_interval_s,
            "lifespan_bonus_s": self.lifespan_bonus_s,
            "goal_reached_radius_m": self.goal_reached_radius_m,
        }
        for field_name, value in positive_fields.items():
            if value <= 0.0:
                raise ValueError(f"episode.{field_name} must be > 0.")
        if self.brain_dt_s < self.neuron_dt_s:
            raise ValueError("episode.brain_dt_s must be >= episode.neuron_dt_s.")
        for field_name, value in (
            ("selection_top_frac", self.selection_top_frac),
            ("selection_bot_frac", self.selection_bot_frac),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"episode.{field_name} must be in (0, 1).")


@dataclass(frozen=True)
class RewardSpec:
    default_motor_noise_scale: float = 0.40
    max_motor_noise_scale: float = 1.20
    fast_progress_tau_s: float = 0.20
    slow_progress_tau_s: float = 0.80
    dramatic_progress_drop_ratio: float = 0.55
    noise_attack_tau_s: float = 0.15
    noise_release_tau_s: float = 0.90
    side_tip_band_half_width_deg: float = 60.0
    side_tip_depth_penalty_scale: float = 10.0
    side_tip_escape_delta_scale: float = 10.0
    side_tip_exit_bonus: float = 8.0
    progress_reward_scale: float = 50.0
    goal_reached_bonus: float = 50.0
    foot_level_reward_scale: float = 1.0
    step_climb_bonus: float = 30.0
    escape_bonus: float = 100.0

    def validate(self) -> None:
        positive_or_zero_fields = {
            "default_motor_noise_scale": self.default_motor_noise_scale,
            "max_motor_noise_scale": self.max_motor_noise_scale,
            "side_tip_depth_penalty_scale": self.side_tip_depth_penalty_scale,
            "side_tip_escape_delta_scale": self.side_tip_escape_delta_scale,
            "side_tip_exit_bonus": self.side_tip_exit_bonus,
            "progress_reward_scale": self.progress_reward_scale,
            "goal_reached_bonus": self.goal_reached_bonus,
            "foot_level_reward_scale": self.foot_level_reward_scale,
            "step_climb_bonus": self.step_climb_bonus,
            "escape_bonus": self.escape_bonus,
        }
        for field_name, value in positive_or_zero_fields.items():
            if value < 0.0:
                raise ValueError(f"reward.{field_name} must be >= 0.")
        positive_fields = {
            "fast_progress_tau_s": self.fast_progress_tau_s,
            "slow_progress_tau_s": self.slow_progress_tau_s,
            "noise_attack_tau_s": self.noise_attack_tau_s,
            "noise_release_tau_s": self.noise_release_tau_s,
        }
        for field_name, value in positive_fields.items():
            if value <= 0.0:
                raise ValueError(f"reward.{field_name} must be > 0.")
        if self.max_motor_noise_scale < self.default_motor_noise_scale:
            raise ValueError("reward.max_motor_noise_scale must be >= reward.default_motor_noise_scale.")
        if not 0.0 < self.dramatic_progress_drop_ratio < 1.0:
            raise ValueError("reward.dramatic_progress_drop_ratio must be in (0, 1).")
        if not 0.0 < self.side_tip_band_half_width_deg <= 180.0:
            raise ValueError("reward.side_tip_band_half_width_deg must be in (0, 180].")


@dataclass(frozen=True)
class TrainingSpec:
    population_size: int = 32
    sigma: float = 0.08
    learning_rate: float = 0.05
    parent_elite_count: int = 5

    def validate(self) -> None:
        if self.population_size <= 0:
            raise ValueError("training.population_size must be > 0.")
        if self.sigma <= 0.0:
            raise ValueError("training.sigma must be > 0.")
        if self.learning_rate <= 0.0:
            raise ValueError("training.learning_rate must be > 0.")
        if self.parent_elite_count <= 0:
            raise ValueError("training.parent_elite_count must be > 0.")
        if self.parent_elite_count > self.population_size:
            raise ValueError("training.parent_elite_count cannot exceed training.population_size.")


@dataclass(frozen=True)
class MujocoSpec:
    timestep_s: float = 0.0025
    integrator: str = "implicitfast"
    solver: str = "Newton"
    solver_iterations: int = 100
    line_search_iterations: int = 50
    noslip_iterations: int = 4
    contact_margin_m: float = 0.002
    actuator_force_limit: float = 12.0
    velocity_servo_gain: float = 6.0
    joint_range_rad: tuple[float, float] = (-1.1, 1.1)

    def validate(self) -> None:
        if self.timestep_s <= 0.0:
            raise ValueError("simulator.mujoco.timestep_s must be > 0.")
        if self.integrator not in {"Euler", "RK4", "implicit", "implicitfast"}:
            raise ValueError("simulator.mujoco.integrator must be Euler, RK4, implicit, or implicitfast.")
        if self.solver not in {"PGS", "CG", "Newton"}:
            raise ValueError("simulator.mujoco.solver must be PGS, CG, or Newton.")
        for field_name, value in (
            ("solver_iterations", self.solver_iterations),
            ("line_search_iterations", self.line_search_iterations),
            ("noslip_iterations", self.noslip_iterations),
        ):
            if value < 0:
                raise ValueError(f"simulator.mujoco.{field_name} must be >= 0.")
        for field_name, value in (
            ("contact_margin_m", self.contact_margin_m),
            ("actuator_force_limit", self.actuator_force_limit),
            ("velocity_servo_gain", self.velocity_servo_gain),
        ):
            if value <= 0.0:
                raise ValueError(f"simulator.mujoco.{field_name} must be > 0.")
        if self.joint_range_rad[0] >= self.joint_range_rad[1]:
            raise ValueError("simulator.mujoco.joint_range_rad must be an ordered [min, max] pair.")


@dataclass(frozen=True)
class SimulatorSpec:
    backend: str = "jax"
    render: bool = False
    deterministic_mode: bool = True
    mujoco: MujocoSpec = field(default_factory=MujocoSpec)

    def validate(self) -> None:
        if self.backend not in {"jax", "mujoco"}:
            raise ValueError("simulator.backend must be 'jax' or 'mujoco'.")
        self.mujoco.validate()


@dataclass(frozen=True)
class QualityGateSpec:
    enabled: bool = True
    run_on_startup: bool = True
    collision_sanity_steps: int = 12
    unstable_state_steps: int = 80
    spawn_samples: int = 32
    determinism_steps: int = 40
    performance_budget_seconds: float = 3.0
    performance_warmup_runs: int = 1
    performance_eval_runs: int = 3
    performance_steps: int = 40
    max_body_height_m: float = 3.0
    max_abs_body_rotation_rad: float = 6.5
    determinism_tolerance: float = 1e-6

    def validate(self) -> None:
        non_negative_ints = {
            "collision_sanity_steps": self.collision_sanity_steps,
            "unstable_state_steps": self.unstable_state_steps,
            "spawn_samples": self.spawn_samples,
            "determinism_steps": self.determinism_steps,
            "performance_warmup_runs": self.performance_warmup_runs,
            "performance_eval_runs": self.performance_eval_runs,
            "performance_steps": self.performance_steps,
        }
        for field_name, value in non_negative_ints.items():
            if value < 0:
                raise ValueError(f"quality_gates.{field_name} must be >= 0.")
        positive_fields = {
            "performance_budget_seconds": self.performance_budget_seconds,
            "max_body_height_m": self.max_body_height_m,
            "max_abs_body_rotation_rad": self.max_abs_body_rotation_rad,
            "determinism_tolerance": self.determinism_tolerance,
        }
        for field_name, value in positive_fields.items():
            if value <= 0.0:
                raise ValueError(f"quality_gates.{field_name} must be > 0.")


@dataclass(frozen=True)
class LoggingSpec:
    level: str = "INFO"
    root_dir: str = "logs"
    console: bool = True
    events_filename: str = "events.jsonl"
    metrics_filename: str = "metrics.jsonl"

    def validate(self) -> None:
        if self.level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("logging.level must be DEBUG, INFO, WARNING, or ERROR.")
        if not self.root_dir:
            raise ValueError("logging.root_dir must be non-empty.")
        if not self.events_filename:
            raise ValueError("logging.events_filename must be non-empty.")
        if not self.metrics_filename:
            raise ValueError("logging.metrics_filename must be non-empty.")


@dataclass(frozen=True)
class RuntimeSpec:
    name: str = "default"
    simulator: SimulatorSpec = field(default_factory=SimulatorSpec)
    terrain: TerrainSpec = field(default_factory=TerrainSpec)
    goals: GoalSpec = field(default_factory=GoalSpec)
    spawn_policy: SpawnPolicySpec = field(default_factory=SpawnPolicySpec)
    friction: FrictionSpec = field(default_factory=FrictionSpec)
    robot: RobotSpec = field(default_factory=RobotSpec)
    physics: PhysicsSpec = field(default_factory=PhysicsSpec)
    episode: EpisodeRulesSpec = field(default_factory=EpisodeRulesSpec)
    reward: RewardSpec = field(default_factory=RewardSpec)
    training: TrainingSpec = field(default_factory=TrainingSpec)
    quality_gates: QualityGateSpec = field(default_factory=QualityGateSpec)
    logging: LoggingSpec = field(default_factory=LoggingSpec)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("config.name must be non-empty.")
        self.simulator.validate()
        self.terrain.validate()
        self.goals.validate()
        self.spawn_policy.validate(self.terrain)
        self.friction.validate()
        self.robot.validate()
        self.physics.validate()
        self.episode.validate()
        self.reward.validate()
        self.training.validate()
        self.quality_gates.validate()
        self.logging.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_SPEC = RuntimeSpec()


def runtime_spec_from_dict(raw_data: Mapping[str, Any] | None) -> RuntimeSpec:
    data = _merge_section(DEFAULT_SPEC.to_dict(), dict(raw_data or {}))
    goals_fixed = data["goals"].get("fixed_goal_xyz")
    fixed_goal = None if goals_fixed is None else tuple(float(value) for value in goals_fixed)

    spec = RuntimeSpec(
        name=str(data["name"]),
        simulator=SimulatorSpec(
            backend=str(data["simulator"]["backend"]),
            render=bool(data["simulator"]["render"]),
            deterministic_mode=bool(data["simulator"]["deterministic_mode"]),
            mujoco=MujocoSpec(
                timestep_s=float(data["simulator"]["mujoco"]["timestep_s"]),
                integrator=str(data["simulator"]["mujoco"]["integrator"]),
                solver=str(data["simulator"]["mujoco"]["solver"]),
                solver_iterations=int(data["simulator"]["mujoco"]["solver_iterations"]),
                line_search_iterations=int(data["simulator"]["mujoco"]["line_search_iterations"]),
                noslip_iterations=int(data["simulator"]["mujoco"]["noslip_iterations"]),
                contact_margin_m=float(data["simulator"]["mujoco"]["contact_margin_m"]),
                actuator_force_limit=float(data["simulator"]["mujoco"]["actuator_force_limit"]),
                velocity_servo_gain=float(data["simulator"]["mujoco"]["velocity_servo_gain"]),
                joint_range_rad=_as_float_pair(
                    data["simulator"]["mujoco"]["joint_range_rad"],
                    "simulator.mujoco.joint_range_rad",
                ),
            ),
        ),
        terrain=TerrainSpec(**data["terrain"]),
        goals=GoalSpec(
            strategy=str(data["goals"]["strategy"]),
            radius_m=float(data["goals"]["radius_m"]),
            height_m=float(data["goals"]["height_m"]),
            fixed_goal_xyz=fixed_goal,
        ),
        spawn_policy=SpawnPolicySpec(
            strategy=str(data["spawn_policy"]["strategy"]),
            x_range_m=_as_float_pair(data["spawn_policy"]["x_range_m"], "spawn_policy.x_range_m"),
            y_range_m=_as_float_pair(data["spawn_policy"]["y_range_m"], "spawn_policy.y_range_m"),
            fixed_points=_as_points(data["spawn_policy"].get("fixed_points"), "spawn_policy.fixed_points"),
        ),
        friction=FrictionSpec(**data["friction"]),
        robot=RobotSpec(**data["robot"]),
        physics=PhysicsSpec(**data["physics"]),
        episode=EpisodeRulesSpec(**data["episode"]),
        reward=RewardSpec(**data["reward"]),
        training=TrainingSpec(**data["training"]),
        quality_gates=QualityGateSpec(**data["quality_gates"]),
        logging=LoggingSpec(**data["logging"]),
    )
    spec.validate()
    return spec
