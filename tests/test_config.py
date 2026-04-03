from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai.config import DEFAULT_CONFIG_PATH, load_runtime_spec, runtime_spec_from_dict, save_runtime_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RuntimeConfigTests(unittest.TestCase):
    def test_default_config_loads(self) -> None:
        spec = load_runtime_spec(DEFAULT_CONFIG_PATH)
        self.assertEqual(spec.name, "default")
        self.assertEqual(spec.training.population_size, 32)
        self.assertEqual(spec.terrain.kind, "stepped_arena")

    def test_invalid_spawn_policy_is_rejected(self) -> None:
        bad_config = load_runtime_spec(DEFAULT_CONFIG_PATH).to_dict()
        bad_config["spawn_policy"]["x_range_m"] = [-99.0, 99.0]
        with self.assertRaises(ValueError):
            runtime_spec_from_dict(bad_config)

    def test_json_round_trip_preserves_config(self) -> None:
        spec = load_runtime_spec(PROJECT_ROOT / "configs" / "smoke.yaml")
        with TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "smoke.json"
            save_runtime_spec(target, spec)
            reloaded = load_runtime_spec(target)
        self.assertEqual(spec.to_dict(), reloaded.to_dict())


if __name__ == "__main__":
    unittest.main()
