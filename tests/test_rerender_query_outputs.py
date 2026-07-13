"""Small contracts for the standalone query re-render utility."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.rerender_query_outputs import _paths_for_row


class RerenderQueryOutputsTest(unittest.TestCase):
    def test_manifest_root_layout_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {
                "query_id": "keyboard_q1",
                "output_root": str(root / "source"),
                "dataset_dir": str(root / "dataset"),
            }
            source_root, source_run, selection, target = _paths_for_row(
                row,
                None,
                root / "target",
            )

            self.assertEqual(source_root, root / "source" / "keyboard_q1")
            self.assertEqual(source_run, source_root / "query_worldtube_run")
            self.assertEqual(selection, source_run / "entitybank" / "selected_query_qwen.json")
            self.assertEqual(target, root / "target" / "keyboard_q1" / "final_query_render_sourcebg")

    def test_source_root_override_is_used_for_every_query(self) -> None:
        row = {
            "query_id": "cut_lemon_q2",
            "output_root": "/ignored",
            "dataset_dir": "/dataset",
        }
        source_root, _source_run, _selection, _target = _paths_for_row(
            row,
            Path("/fixed/source"),
            Path("/fresh/output"),
        )
        self.assertEqual(source_root, Path("/fixed/source/cut_lemon_q2"))


if __name__ == "__main__":
    unittest.main()
