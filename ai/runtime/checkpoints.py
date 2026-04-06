"""Checkpoint resolution helpers for runtime entrypoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai.config import RuntimeSpec, canonical_config_json


@dataclass(frozen=True)
class CheckpointCompatibility:
    path: Path
    compatible: bool
    reason: str | None = None


def _existing_candidates(root: str | Path, names: tuple[str, ...]) -> tuple[Path, ...]:
    checkpoint_root = Path(root)
    return tuple((checkpoint_root / name) for name in names if (checkpoint_root / name).exists())


def viewer_checkpoint_candidates(root: str | Path = "checkpoints") -> tuple[Path, ...]:
    return _existing_candidates(root, ("latest.npz", "best.npz"))


def _checkpoint_backend(path: Path) -> str | None:
    with np.load(path, allow_pickle=False) as checkpoint:
        if "simulator_backend" in checkpoint.files:
            return str(checkpoint["simulator_backend"].item())
        if "config_json" not in checkpoint.files:
            return None
        try:
            config = json.loads(str(checkpoint["config_json"].item()))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    simulator = config.get("simulator")
    if not isinstance(simulator, dict):
        return None
    backend = simulator.get("backend")
    return None if backend is None else str(backend)


def checkpoint_matches_spec(path: str | Path, spec: RuntimeSpec) -> CheckpointCompatibility:
    checkpoint_path = Path(path)
    try:
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            if "config_json" in checkpoint.files:
                checkpoint_config_json = str(checkpoint["config_json"].item())
                if checkpoint_config_json != canonical_config_json(spec):
                    return CheckpointCompatibility(
                        path=checkpoint_path,
                        compatible=False,
                        reason="checkpoint runtime spec does not match the active config",
                    )
            else:
                checkpoint_backend = _checkpoint_backend(checkpoint_path)
                if checkpoint_backend is None:
                    return CheckpointCompatibility(
                        path=checkpoint_path,
                        compatible=False,
                        reason="checkpoint is missing runtime metadata",
                    )
                if checkpoint_backend != spec.simulator.backend:
                    return CheckpointCompatibility(
                        path=checkpoint_path,
                        compatible=False,
                        reason=(
                            f"checkpoint backend '{checkpoint_backend}' does not match "
                            f"active backend '{spec.simulator.backend}'"
                        ),
                    )
    except Exception as exc:
        return CheckpointCompatibility(
            path=checkpoint_path,
            compatible=False,
            reason=f"failed to inspect checkpoint: {exc}",
        )
    return CheckpointCompatibility(path=checkpoint_path, compatible=True)


def resolve_viewer_checkpoint(root: str | Path = "checkpoints", spec: RuntimeSpec | None = None) -> Path | None:
    for path in viewer_checkpoint_candidates(root):
        if spec is None or checkpoint_matches_spec(path, spec).compatible:
            return path
    return None
