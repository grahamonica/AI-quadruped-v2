"""Checkpoint resolution helpers for runtime entrypoints."""

from __future__ import annotations

from pathlib import Path


def resolve_training_resume_checkpoint(root: str | Path = "checkpoints") -> Path | None:
    checkpoint_root = Path(root)
    for candidate in ("latest.npz", "best.npz"):
        path = checkpoint_root / candidate
        if path.exists():
            return path
    return None


def resolve_single_view_checkpoint(root: str | Path = "checkpoints") -> Path | None:
    checkpoint_root = Path(root)
    for candidate in ("best_single.npz", "best.npz", "latest.npz"):
        path = checkpoint_root / candidate
        if path.exists():
            return path
    return None
