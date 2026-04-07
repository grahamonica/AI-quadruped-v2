"""Checkpoint resolution helpers for runtime entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai.config import RuntimeSpec, config_json_matches_checkpoint


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


def checkpoint_matches_spec(path: str | Path, spec: RuntimeSpec) -> CheckpointCompatibility:
    checkpoint_path = Path(path)
    try:
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            if "config_json" in checkpoint.files:
                checkpoint_config_json = str(checkpoint["config_json"].item())
                if not config_json_matches_checkpoint(spec, checkpoint_config_json):
                    return CheckpointCompatibility(
                        path=checkpoint_path,
                        compatible=False,
                        reason="checkpoint runtime spec does not match the active config aside from backend selection",
                    )
            else:
                return CheckpointCompatibility(
                    path=checkpoint_path,
                    compatible=False,
                    reason="checkpoint is missing runtime metadata",
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
