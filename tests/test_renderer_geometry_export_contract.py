"""Contracts for exact-camera renderer geometry exports."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.export_renderer_geometry import _frame_requests_from_image_ids


class RendererGeometryExportContractTest(unittest.TestCase):
    def test_image_id_protocol_filters_without_reordering_dataset_frames(self) -> None:
        entries = [
            {"image_id": "000001", "frame_index": 1},
            {"image_id": "000003", "frame_index": 3},
            {"image_id": "000005", "frame_index": 5},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image_ids.json"
            path.write_text(json.dumps({"image_ids": ["000005", "000001"]}), encoding="utf-8")
            with patch("scripts.export_renderer_geometry._source_entries", return_value=entries):
                result = _frame_requests_from_image_ids(path, Path(tmp))
        self.assertEqual([row["image_id"] for row in result], ["000001", "000005"])

    def test_missing_requested_camera_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image_ids.json"
            path.write_text(json.dumps(["000099"]), encoding="utf-8")
            with patch("scripts.export_renderer_geometry._source_entries", return_value=[]):
                with self.assertRaisesRegex(FileNotFoundError, "000099"):
                    _frame_requests_from_image_ids(path, Path(tmp))


if __name__ == "__main__":
    unittest.main()
