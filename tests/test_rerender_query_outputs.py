"""Small contracts for the standalone query re-render utility."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.rerender_query_outputs import _benchmark_frame_ids_by_query, _benchmark_image_ids, _paths_for_row


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

    def test_benchmark_camera_export_reads_only_frame_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = root / "benchmark.json"
            benchmark.write_text(
                """{
                  "queries": [{
                    "query_id": "keyboard_q1",
                    "ground_truth": {
                      "frames": [
                        {"frame_id": 10, "masks": [{"segmentation": "must_not_be_read"}]},
                        {"frame_id": 20, "masks": [{"segmentation": "must_not_be_read"}]}
                      ]
                    }
                  }]
                }""",
                encoding="utf-8",
            )
            frame_ids = _benchmark_frame_ids_by_query(benchmark)
            self.assertEqual(frame_ids, {"keyboard_q1": [10, 20]})

            hypernerf = root / "hypernerf"
            hypernerf.mkdir()
            (hypernerf / "metadata.json").write_text("{}", encoding="utf-8")
            self.assertEqual(_benchmark_image_ids(frame_ids["keyboard_q1"], hypernerf), ["000010", "000020"])
            self.assertEqual(_benchmark_image_ids([10, 20], root / "dynerf"), ["0010", "0020"])


if __name__ == "__main__":
    unittest.main()
