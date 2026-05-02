from __future__ import annotations

import unittest
from pathlib import Path

from brains.config import load_runtime_spec
import brains.jax_trainer as trainer_module
from brains.sim.mujoco_assets import LEG_NAMES, load_mujoco_model, robot_geometry_from_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DomainModelTests(unittest.TestCase):
    def test_mujoco_static_assets_are_canonical(self) -> None:
        model = load_mujoco_model()
        geometry = robot_geometry_from_model(model)

        self.assertEqual(LEG_NAMES, ("front_left", "front_right", "rear_left", "rear_right"))
        self.assertEqual(model.ncam, 0)
        self.assertAlmostEqual(geometry.leg_length_m, 0.16)
        self.assertAlmostEqual(geometry.leg_radius_m, 0.01)
        self.assertAlmostEqual(geometry.body_size_m[0], geometry.leg_length_m * 2.5)
        self.assertAlmostEqual(geometry.body_size_m[1], geometry.body_size_m[0] * 0.5)
        self.assertAlmostEqual(geometry.body_size_m[2], geometry.leg_radius_m * 2.0 * 1.5)
        self.assertEqual(geometry.mount_points_body[0], (0.2, 0.1, 0.0))

    def test_runtime_spec_contains_environment_fields(self) -> None:
        spec = load_runtime_spec(PROJECT_ROOT / "configs" / "smoke.yaml")
        self.assertEqual(spec.name, "smoke")
        self.assertEqual(spec.terrain.kind, "flat")
        self.assertEqual(spec.model.positional_encoding, "sinusoidal")
        self.assertEqual(spec.spawn_policy.strategy, "uniform_box")
        self.assertEqual(spec.training.population_size, 8)

    def test_jax_runtime_uses_runtime_spec_values(self) -> None:
        spec = load_runtime_spec(PROJECT_ROOT / "configs" / "smoke.yaml")
        trainer_module.apply_runtime_spec(spec)
        self.assertEqual(trainer_module.MAX_MOTOR_RAD_S, spec.robot.max_motor_rad_s)
        self.assertEqual(trainer_module.GOAL_HEIGHT_M, spec.goals.height_m)
        self.assertEqual(trainer_module.EPISODE_S, spec.episode.episode_s)


if __name__ == "__main__":
    unittest.main()
