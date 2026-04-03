"""Run runtime quality gates and optional fixed-seed regression checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai.config import DEFAULT_CONFIG_PATH, load_runtime_spec
from ai.quality import QualityGateRunner, compare_regression_to_baseline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quadruped quality gates.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Runtime spec to validate.")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional regression baseline JSON. If provided, fixed-seed regression checks are executed too.",
    )
    parser.add_argument(
        "--regression-seeds",
        type=int,
        nargs="*",
        default=[7, 17],
        help="Seeds used when comparing against a regression baseline.",
    )
    parser.add_argument(
        "--regression-generations",
        type=int,
        default=1,
        help="Number of generations to run per seed for the regression check.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    spec = load_runtime_spec(args.config)
    runtime_report = QualityGateRunner(spec).run()
    print(json.dumps(runtime_report.to_dict(), indent=2, sort_keys=True))

    if not runtime_report.passed:
        return 2

    if args.baseline is not None:
        regression_report = compare_regression_to_baseline(
            spec,
            baseline_path=args.baseline,
            seeds=list(args.regression_seeds),
            generations=args.regression_generations,
        )
        print(json.dumps(regression_report.to_dict(), indent=2, sort_keys=True))
        if not regression_report.passed:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
