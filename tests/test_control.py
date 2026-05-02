from __future__ import annotations

import unittest

import numpy as np

from brains.config import load_runtime_spec, runtime_spec_from_dict
from brains.sim.action_layer import ActionProjector, command_target_velocity


class ControlLayerTests(unittest.TestCase):
    def test_default_config_uses_motor_targets(self) -> None:
        spec = load_runtime_spec("configs/smoke.yaml")
        self.assertEqual(spec.control.mode, "motor_targets")

    def test_command_mode_requires_non_empty_vocabulary(self) -> None:
        config = load_runtime_spec("configs/smoke.yaml").to_dict()
        config["control"]["mode"] = "command_primitives"
        config["control"]["command_vocabulary"] = []
        with self.assertRaises(ValueError):
            runtime_spec_from_dict(config)

    def test_command_mode_rejects_invalid_timing_window(self) -> None:
        config = load_runtime_spec("configs/smoke.yaml").to_dict()
        config["control"]["mode"] = "command_primitives"
        config["control"]["command_default_duration_s"] = 0.05
        config["control"]["command_update_interval_s"] = 0.10
        with self.assertRaises(ValueError):
            runtime_spec_from_dict(config)

    def test_motor_projection_scales_direct_outputs(self) -> None:
        projector = ActionProjector(mode="motor_targets")
        projected = projector.project(
            np.asarray([1.0, -1.0, 0.5, -0.5], dtype=np.float32),
            time_s=0.0,
            max_motor_rad_s=8.0,
            motor_scale=6.0,
        )
        self.assertEqual(projected.selected_command, None)
        self.assertTrue(np.allclose(projected.target_velocity_rad_s, np.asarray([6.0, -6.0, 3.0, -3.0], dtype=np.float32)))

    def test_command_projection_selects_primitive(self) -> None:
        projector = ActionProjector(
            mode="command_primitives",
            command_vocabulary=("stand", "trot", "turn_left", "turn_right"),
            default_command_speed=0.55,
        )
        projected = projector.project(
            np.asarray([0.1, 0.9, 0.2, 0.1], dtype=np.float32),
            time_s=0.1,
            max_motor_rad_s=8.0,
            motor_scale=6.0,
        )
        self.assertEqual(projected.selected_command, "trot")
        self.assertEqual(projected.target_velocity_rad_s.shape, (4,))

    def test_command_projection_holds_active_command_until_hold_expires(self) -> None:
        projector = ActionProjector(
            mode="command_primitives",
            command_vocabulary=("stand", "trot", "turn_left", "turn_right"),
            default_command_speed=0.55,
            command_update_interval_s=0.10,
            command_default_duration_s=0.25,
            command_max_duration_s=0.80,
        )
        first = projector.project(
            np.asarray([0.1, 0.9, 0.2, 0.1], dtype=np.float32),
            time_s=0.00,
            max_motor_rad_s=8.0,
            motor_scale=6.0,
        )
        held = projector.project(
            np.asarray([0.1, 0.1, 0.9, 0.2], dtype=np.float32),
            time_s=0.10,
            max_motor_rad_s=8.0,
            motor_scale=6.0,
        )
        updated = projector.project(
            np.asarray([0.1, 0.1, 0.9, 0.2], dtype=np.float32),
            time_s=0.30,
            max_motor_rad_s=8.0,
            motor_scale=6.0,
        )
        self.assertEqual(first.selected_command, "trot")
        self.assertEqual(held.selected_command, "trot")
        self.assertEqual(updated.selected_command, "turn_left")

    def test_command_duration_head_extends_hold_window(self) -> None:
        projector = ActionProjector(
            mode="command_primitives",
            command_vocabulary=("stand", "trot", "turn_left", "turn_right"),
            default_command_speed=0.55,
            command_update_interval_s=0.10,
            command_default_duration_s=0.25,
            command_max_duration_s=0.80,
        )
        first = projector.project(
            np.asarray([0.1, 0.9, 0.2, 0.1, 1.0], dtype=np.float32),
            time_s=0.00,
            max_motor_rad_s=8.0,
            motor_scale=6.0,
        )
        held = projector.project(
            np.asarray([0.1, 0.1, 0.9, 0.2, -1.0], dtype=np.float32),
            time_s=0.50,
            max_motor_rad_s=8.0,
            motor_scale=6.0,
        )
        updated = projector.project(
            np.asarray([0.1, 0.1, 0.9, 0.2, -1.0], dtype=np.float32),
            time_s=0.90,
            max_motor_rad_s=8.0,
            motor_scale=6.0,
        )
        self.assertEqual(first.selected_command, "trot")
        self.assertEqual(held.selected_command, "trot")
        self.assertEqual(updated.selected_command, "turn_left")

    def test_walk_command_drives_all_legs_forward_constantly(self) -> None:
        max_motor = 8.0
        speed = 1.0
        magnitude = np.float32(0.35) * np.float32(max_motor * speed)

        at_start = command_target_velocity("walk", time_s=0.00, speed=speed, max_motor_rad_s=max_motor)
        mid_cycle = command_target_velocity("walk", time_s=0.55, speed=speed, max_motor_rad_s=max_motor)
        later = command_target_velocity("walk", time_s=1.85, speed=speed, max_motor_rad_s=max_motor)
        expected = np.asarray([magnitude, magnitude, magnitude, magnitude], dtype=np.float32)

        self.assertTrue(np.allclose(at_start, expected))
        self.assertTrue(np.allclose(mid_cycle, expected))
        self.assertTrue(np.allclose(later, expected))


if __name__ == "__main__":
    unittest.main()
