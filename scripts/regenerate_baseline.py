#!/usr/bin/env python3
"""Regenerate fixed-seed regression baseline for smoke config."""

import json
import platform
from pathlib import Path

from ai.config import load_runtime_spec
from ai.quality import collect_regression_metrics


def regenerate_baseline():
    project_root = Path(__file__).resolve().parents[1]
    smoke_config = project_root / "configs" / "smoke.yaml"
    baseline_path = project_root / "tests" / "fixtures" / "smoke_regression_baseline.json"

    print(f"Regenerating baseline from {smoke_config}")

    # Collect actual results
    spec = load_runtime_spec(smoke_config)
    metrics = collect_regression_metrics(spec, seeds=[7, 17], generations=1)

    # Load existing baseline or create new
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        baseline = {"variants": {}}

    # Determine variant key
    system_key = platform.system().lower()
    machine_key = platform.machine().lower()
    variant_key = f"{system_key}-{machine_key}"

    # Update the variant
    baseline["variants"][variant_key] = metrics

    # Save baseline
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2, sort_keys=True)

    print(f"Baseline saved for variant {variant_key}: {baseline_path}")

    # Note: Verification would require reloading, but since we just saved, it should pass


if __name__ == "__main__":
    regenerate_baseline()
