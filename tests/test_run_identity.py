"""Tests for the released-run identity guard used by query evaluation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from refergaussian.run_identity import (
    validate_query_ready_baseline_4dgs_run,
    validate_query_ready_refergaussian_run,
    validate_refergaussian_run,
)


class RunIdentityTest(unittest.TestCase):
    def test_refergaussian_config_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "config.yaml").write_text(
                "phase: refergaussian\n"
                "temporal_warp_type: refergaussian\n"
                "warp_enabled: true\n",
                encoding="utf-8",
            )
            self.assertEqual(validate_refergaussian_run(run_dir), [])

    def test_non_refergaussian_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "config.yaml").write_text(
                "phase: baseline\n"
                "temporal_warp_type: identity\n"
                "warp_enabled: false\n",
                encoding="utf-8",
            )
            errors = validate_refergaussian_run(run_dir)

        self.assertEqual(len(errors), 3)
        self.assertIn("phase='baseline', expected 'refergaussian'", errors)

    def test_query_ready_run_requires_renderer_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "config.yaml").write_text(
                "phase: refergaussian\n"
                "temporal_warp_type: refergaussian\n"
                "warp_enabled: true\n",
                encoding="utf-8",
            )
            self.assertEqual(
                validate_query_ready_refergaussian_run(run_dir),
                [
                    f"missing query-render artifact: {run_dir / 'point_cloud'}",
                    f"missing query-render artifact: {run_dir / 'test'}",
                ],
            )

            (run_dir / "point_cloud").mkdir()
            (run_dir / "test").mkdir()
            self.assertEqual(validate_query_ready_refergaussian_run(run_dir), [])

    def test_explicit_baseline_contract_requires_identity_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "config.yaml").write_text(
                "phase: baseline\nwarp_enabled: false\n",
                encoding="utf-8",
            )
            (run_dir / "point_cloud").mkdir()
            (run_dir / "test").mkdir()

            self.assertEqual(validate_query_ready_baseline_4dgs_run(run_dir), [])
            self.assertNotEqual(validate_refergaussian_run(run_dir), [])


if __name__ == "__main__":
    unittest.main()
