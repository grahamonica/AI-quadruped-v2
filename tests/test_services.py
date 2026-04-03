from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai.runtime import resolve_single_view_checkpoint, resolve_training_resume_checkpoint
from server import app as live_app
from server_single import app as single_app


class ServiceImportTests(unittest.TestCase):
    def test_live_app_routes_exist(self) -> None:
        paths = {route.path for route in live_app.routes}
        self.assertIn("/", paths)
        self.assertIn("/healthz", paths)
        self.assertIn("/ws", paths)

    def test_single_app_routes_exist(self) -> None:
        paths = {route.path for route in single_app.routes}
        self.assertIn("/", paths)
        self.assertIn("/healthz", paths)
        self.assertIn("/ws", paths)

    def test_checkpoint_resolution_order(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "best.npz").write_text("best", encoding="utf-8")
            (root / "latest.npz").write_text("latest", encoding="utf-8")
            (root / "best_single.npz").write_text("single", encoding="utf-8")

            self.assertEqual(resolve_training_resume_checkpoint(root), root / "latest.npz")
            self.assertEqual(resolve_single_view_checkpoint(root), root / "best_single.npz")


if __name__ == "__main__":
    unittest.main()
