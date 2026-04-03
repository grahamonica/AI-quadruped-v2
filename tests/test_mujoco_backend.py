from __future__ import annotations

import json
import unittest
from pathlib import Path

from ai.config import load_runtime_spec
from ai.quality import QualityGateRunner
from ai.sim.mujoco_backend import MuJoCoBackend
from ai.trainer import ESTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_SMOKE_CONFIG = PROJECT_ROOT / "configs" / "smoke_mujoco.yaml"


class MujocoBackendTests(unittest.TestCase):
    def test_smoke_mujoco_backend_builds(self) -> None:
        spec = load_runtime_spec(MUJOCO_SMOKE_CONFIG)
        backend = MuJoCoBackend(spec)
        self.assertEqual(spec.simulator.backend, "mujoco")
        self.assertEqual(backend.capabilities.name, "mujoco")
        self.assertEqual(backend.model.nu, 4)
        self.assertEqual(backend.control_substeps, 20)

    def test_est_trainer_factory_selects_mujoco(self) -> None:
        spec = load_runtime_spec(MUJOCO_SMOKE_CONFIG)
        trainer = ESTrainer(seed=7, spec=spec)
        self.assertEqual(trainer.backend, "mujoco")

    def test_smoke_mujoco_quality_gates_pass(self) -> None:
        report = QualityGateRunner(load_runtime_spec(MUJOCO_SMOKE_CONFIG)).run()
        self.assertTrue(report.passed, json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
