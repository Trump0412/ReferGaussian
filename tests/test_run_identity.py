"""Tests for the frozen upstream 4DGS input guard."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from refergaussian.run_identity import validate_query_ready_4dgs_run


class RunIdentityTest(unittest.TestCase):
    def test_standard_upstream_layout_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "cfg_args").write_text("Namespace(model_path='x')\n", encoding="utf-8")
            point_cloud = run_dir / "point_cloud" / "iteration_14000"
            point_cloud.mkdir(parents=True)
            (point_cloud / "point_cloud.ply").write_text("ply\n", encoding="utf-8")
            renders = run_dir / "test" / "ours_14000" / "renders"
            renders.mkdir(parents=True)
            (renders / "00000.png").write_bytes(b"png")

            self.assertEqual(validate_query_ready_4dgs_run(run_dir), [])

    def test_missing_input_artifacts_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            errors = validate_query_ready_4dgs_run(run_dir)

        self.assertEqual(len(errors), 3)
        self.assertIn("missing upstream 4DGS arguments", errors[0])
        self.assertIn("missing upstream 4DGS checkpoint", errors[1])
        self.assertIn("missing upstream 4DGS test renders", errors[2])


if __name__ == "__main__":
    unittest.main()
