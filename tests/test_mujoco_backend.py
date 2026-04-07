from __future__ import annotations

import json
import unittest
from pathlib import Path

from ai.config import load_runtime_spec
from ai.quality import QualityGateRunner
from ai.sim.mujoco_backend import MuJoCoBackend
from ai.jax_trainer import ESTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = PROJECT_ROOT / "configs" / "smoke.yaml"


class RuntimeBackendTests(unittest.TestCase):
    def test_smoke_runtime_backend_builds(self) -> None:
        spec = load_runtime_spec(SMOKE_CONFIG)
        backend = MuJoCoBackend(spec)
        self.assertEqual(spec.simulator.backend, "unified")
        self.assertEqual(backend.capabilities.name, "mujoco")
        self.assertEqual(backend.model.nu, 4)
        self.assertEqual(backend.control_substeps, 20)

    def test_single_trainer_uses_unified_runtime_backend(self) -> None:
        spec = load_runtime_spec(SMOKE_CONFIG)
        trainer = ESTrainer(seed=7, spec=spec)
        self.assertEqual(trainer.backend, "unified")

    def test_smoke_runtime_quality_gates_pass(self) -> None:
        report = QualityGateRunner(load_runtime_spec(SMOKE_CONFIG)).run()
        self.assertTrue(report.passed, json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
