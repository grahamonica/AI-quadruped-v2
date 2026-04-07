from __future__ import annotations

import unittest
from pathlib import Path

from ai.config import load_runtime_spec
from ai.jax_trainer import apply_runtime_spec, current_environment_model, current_robot_model
from quadruped import QuadrupedRobot, SimulationEnvironment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DomainModelTests(unittest.TestCase):
    def test_robot_model_is_built_from_runtime_spec(self) -> None:
        spec = load_runtime_spec(PROJECT_ROOT / "configs" / "default.yaml")
        robot = QuadrupedRobot.from_runtime_spec(spec)
        self.assertEqual(robot.leg_names, ("front_left", "front_right", "rear_left", "rear_right"))
        self.assertEqual(robot.mount_points_body[0], (0.14, 0.06, 0.0))
        self.assertEqual(robot.legs[0].body_sample_fractions, (0.25, 0.5, 0.75))
        self.assertAlmostEqual(robot.total_mass_kg, 4.8)

    def test_environment_model_is_built_from_runtime_spec(self) -> None:
        spec = load_runtime_spec(PROJECT_ROOT / "configs" / "smoke.yaml")
        environment = SimulationEnvironment.from_runtime_spec(spec)
        self.assertEqual(environment.config_name, "smoke")
        self.assertEqual(environment.terrain.step_count, 2)
        self.assertEqual(environment.task.spawn_strategy, "uniform_box")
        self.assertEqual(environment.training.population_size, 8)

    def test_jax_runtime_uses_domain_models(self) -> None:
        spec = load_runtime_spec(PROJECT_ROOT / "configs" / "smoke.yaml")
        apply_runtime_spec(spec)
        robot = current_robot_model()
        environment = current_environment_model()
        self.assertEqual(robot.motor.max_velocity_rad_s, spec.robot.max_motor_rad_s)
        self.assertEqual(environment.task.goal_height_m, spec.goals.height_m)
        self.assertEqual(environment.episode.episode_s, spec.episode.episode_s)


if __name__ == "__main__":
    unittest.main()
