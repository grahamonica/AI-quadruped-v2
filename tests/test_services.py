from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ai.config import canonical_config_json, load_runtime_spec
from ai.api.live import app as viewer_app
from ai.runtime import resolve_viewer_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_checkpoint(path: Path, config_path: Path) -> None:
    spec = load_runtime_spec(config_path)
    np.savez_compressed(
        path,
        config_json=np.array(canonical_config_json(spec)),
        simulator_backend=np.array(spec.simulator.backend),
    )


class ServiceImportTests(unittest.TestCase):
    def test_viewer_app_routes_exist(self) -> None:
        paths = {route.path for route in viewer_app.routes}
        self.assertIn("/", paths)
        self.assertIn("/healthz", paths)
        self.assertIn("/ws", paths)

    def test_checkpoint_resolution_order(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "best.npz").write_text("best", encoding="utf-8")
            (root / "latest.npz").write_text("latest", encoding="utf-8")

            self.assertEqual(resolve_viewer_checkpoint(root), root / "latest.npz")

    def test_checkpoint_resolution_skips_incompatible_candidates(self) -> None:
        jax_spec = load_runtime_spec(PROJECT_ROOT / "configs" / "smoke.yaml")
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_checkpoint(root / "latest.npz", PROJECT_ROOT / "configs" / "smoke_mujoco.yaml")
            _write_checkpoint(root / "best.npz", PROJECT_ROOT / "configs" / "smoke.yaml")

            self.assertEqual(resolve_viewer_checkpoint(root, spec=jax_spec), root / "best.npz")


if __name__ == "__main__":
    unittest.main()
