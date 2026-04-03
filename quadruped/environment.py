"""Environment and task-level domain model for the quadruped runtime."""

from __future__ import annotations

from dataclasses import dataclass

from ai.config import RuntimeSpec


@dataclass(frozen=True)
class TerrainModel:
    kind: str
    field_half_m: float
    center_half_m: float
    step_count: int
    step_width_m: float
    step_height_m: float
    floor_height_m: float


@dataclass(frozen=True)
class TaskModel:
    goal_strategy: str
    goal_radius_m: float
    goal_height_m: float
    fixed_goal_xyz: tuple[float, float, float] | None
    spawn_strategy: str
    spawn_x_range_m: tuple[float, float]
    spawn_y_range_m: tuple[float, float]
    spawn_points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class PhysicsModel:
    gravity_m_s2: float
    normal_stiffness_n_m: float
    normal_damping_n_s_m: float
    tangential_stiffness_n_m: float
    tangential_damping_n_s_m: float
    angular_damping_n_m_s: float
    linear_damping_n_s_m: float
    airborne_linear_damping_n_s_m: float
    airborne_angular_damping_n_m_s: float
    max_contact_force_n: float
    max_substep_s: float
    unloading_stiffness_scale: float
    sleep_linear_speed_threshold_m_s: float
    sleep_angular_speed_threshold_rad_s: float


@dataclass(frozen=True)
class EpisodeModel:
    neuron_dt_s: float
    brain_dt_s: float
    episode_s: float
    single_view_episode_s: float
    default_lifespan_s: float
    tipped_kill_time_s: float
    selection_interval_s: float
    lifespan_bonus_s: float
    selection_top_frac: float
    selection_bot_frac: float
    goal_reached_radius_m: float


@dataclass(frozen=True)
class RewardModel:
    default_motor_noise_scale: float
    max_motor_noise_scale: float
    fast_progress_tau_s: float
    slow_progress_tau_s: float
    dramatic_progress_drop_ratio: float
    noise_attack_tau_s: float
    noise_release_tau_s: float
    side_tip_band_half_width_deg: float
    side_tip_depth_penalty_scale: float
    side_tip_escape_delta_scale: float
    side_tip_exit_bonus: float
    progress_reward_scale: float
    goal_reached_bonus: float
    foot_level_reward_scale: float
    step_climb_bonus: float
    escape_bonus: float


@dataclass(frozen=True)
class TrainingModel:
    population_size: int
    sigma: float
    learning_rate: float
    parent_elite_count: int


@dataclass(frozen=True)
class SimulationEnvironment:
    config_name: str
    terrain: TerrainModel
    task: TaskModel
    physics: PhysicsModel
    episode: EpisodeModel
    reward: RewardModel
    training: TrainingModel

    @classmethod
    def from_runtime_spec(cls, spec: RuntimeSpec) -> "SimulationEnvironment":
        return cls(
            config_name=spec.name,
            terrain=TerrainModel(
                kind=spec.terrain.kind,
                field_half_m=spec.terrain.field_half_m,
                center_half_m=spec.terrain.center_half_m,
                step_count=spec.terrain.step_count,
                step_width_m=spec.terrain.step_width_m,
                step_height_m=spec.terrain.step_height_m,
                floor_height_m=spec.terrain.floor_height_m,
            ),
            task=TaskModel(
                goal_strategy=spec.goals.strategy,
                goal_radius_m=spec.goals.radius_m,
                goal_height_m=spec.goals.height_m,
                fixed_goal_xyz=spec.goals.fixed_goal_xyz,
                spawn_strategy=spec.spawn_policy.strategy,
                spawn_x_range_m=spec.spawn_policy.x_range_m,
                spawn_y_range_m=spec.spawn_policy.y_range_m,
                spawn_points=spec.spawn_policy.fixed_points,
            ),
            physics=PhysicsModel(
                gravity_m_s2=spec.physics.gravity_m_s2,
                normal_stiffness_n_m=spec.physics.normal_stiffness_n_m,
                normal_damping_n_s_m=spec.physics.normal_damping_n_s_m,
                tangential_stiffness_n_m=spec.physics.tangential_stiffness_n_m,
                tangential_damping_n_s_m=spec.physics.tangential_damping_n_s_m,
                angular_damping_n_m_s=spec.physics.angular_damping_n_m_s,
                linear_damping_n_s_m=spec.physics.linear_damping_n_s_m,
                airborne_linear_damping_n_s_m=spec.physics.airborne_linear_damping_n_s_m,
                airborne_angular_damping_n_m_s=spec.physics.airborne_angular_damping_n_m_s,
                max_contact_force_n=spec.physics.max_contact_force_n,
                max_substep_s=spec.physics.max_substep_s,
                unloading_stiffness_scale=spec.physics.unloading_stiffness_scale,
                sleep_linear_speed_threshold_m_s=spec.physics.sleep_linear_speed_threshold_m_s,
                sleep_angular_speed_threshold_rad_s=spec.physics.sleep_angular_speed_threshold_rad_s,
            ),
            episode=EpisodeModel(
                neuron_dt_s=spec.episode.neuron_dt_s,
                brain_dt_s=spec.episode.brain_dt_s,
                episode_s=spec.episode.episode_s,
                single_view_episode_s=spec.episode.single_view_episode_s,
                default_lifespan_s=spec.episode.default_lifespan_s,
                tipped_kill_time_s=spec.episode.tipped_kill_time_s,
                selection_interval_s=spec.episode.selection_interval_s,
                lifespan_bonus_s=spec.episode.lifespan_bonus_s,
                selection_top_frac=spec.episode.selection_top_frac,
                selection_bot_frac=spec.episode.selection_bot_frac,
                goal_reached_radius_m=spec.episode.goal_reached_radius_m,
            ),
            reward=RewardModel(
                default_motor_noise_scale=spec.reward.default_motor_noise_scale,
                max_motor_noise_scale=spec.reward.max_motor_noise_scale,
                fast_progress_tau_s=spec.reward.fast_progress_tau_s,
                slow_progress_tau_s=spec.reward.slow_progress_tau_s,
                dramatic_progress_drop_ratio=spec.reward.dramatic_progress_drop_ratio,
                noise_attack_tau_s=spec.reward.noise_attack_tau_s,
                noise_release_tau_s=spec.reward.noise_release_tau_s,
                side_tip_band_half_width_deg=spec.reward.side_tip_band_half_width_deg,
                side_tip_depth_penalty_scale=spec.reward.side_tip_depth_penalty_scale,
                side_tip_escape_delta_scale=spec.reward.side_tip_escape_delta_scale,
                side_tip_exit_bonus=spec.reward.side_tip_exit_bonus,
                progress_reward_scale=spec.reward.progress_reward_scale,
                goal_reached_bonus=spec.reward.goal_reached_bonus,
                foot_level_reward_scale=spec.reward.foot_level_reward_scale,
                step_climb_bonus=spec.reward.step_climb_bonus,
                escape_bonus=spec.reward.escape_bonus,
            ),
            training=TrainingModel(
                population_size=spec.training.population_size,
                sigma=spec.training.sigma,
                learning_rate=spec.training.learning_rate,
                parent_elite_count=spec.training.parent_elite_count,
            ),
        )
