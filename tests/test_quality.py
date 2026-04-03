from __future__ import annotations

import json
import unittest
from pathlib import Path

from ai.config import load_runtime_spec
from ai.quality import QualityGateRunner, compare_regression_to_baseline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = PROJECT_ROOT / "configs" / "smoke.yaml"
REGRESSION_BASELINE = PROJECT_ROOT / "tests" / "fixtures" / "smoke_regression_baseline.json"


class QualityGateTests(unittest.TestCase):
    def test_smoke_quality_gates_pass(self) -> None:
        report = QualityGateRunner(load_runtime_spec(SMOKE_CONFIG)).run()
        self.assertTrue(report.passed, json.dumps(report.to_dict(), indent=2, sort_keys=True))

    def test_fixed_seed_regression_matches_baseline(self) -> None:
        report = compare_regression_to_baseline(
            load_runtime_spec(SMOKE_CONFIG),
            baseline_path=REGRESSION_BASELINE,
            seeds=[7, 17],
            generations=1,
            atol=1e-5,
        )
        self.assertTrue(report.passed, json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
