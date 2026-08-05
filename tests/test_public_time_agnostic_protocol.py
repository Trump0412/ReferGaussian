"""Tests for the Public time-agnostic protocol and metrics."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load(
    "refergaussian_time_agnostic_builder",
    SCRIPTS_ROOT / "build_4dlangsplat_time_agnostic_protocol.py",
)
EVALUATOR = _load(
    "refergaussian_time_agnostic_evaluator",
    SCRIPTS_ROOT / "evaluate_public_time_agnostic.py",
)


class PublicTimeAgnosticProtocolTest(unittest.TestCase):
    def test_builder_emits_every_used_coco_category(self) -> None:
        payload = {
            "images": [
                {"id": 1, "file_name": "000010_png.rf.x.jpg"},
                {"id": 2, "file_name": "000020_png.rf.y.jpg"},
            ],
            "categories": [
                {"id": 4, "name": "glass cup"},
                {"id": 7, "name": "unused"},
            ],
            "annotations": [{"image_id": 1, "category_id": 4}],
        }

        rows = BUILDER.build_scene_queries("espresso", payload)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["query_slug"], "espresso__time_agnostic__glass_cup")
        self.assertEqual(rows[0]["evaluation_image_ids"], ["000010", "000020"])
        self.assertEqual(rows[0]["annotated_frame_count"], 1)

    def test_metrics_include_false_positives_on_absent_test_frames(self) -> None:
        gt = np.zeros((4, 5), dtype=bool)
        gt[1:3, 1:4] = True
        false_positive = np.zeros_like(gt)
        false_positive[0, 0] = True
        row = {
            "query_slug": "scene__time_agnostic__target",
            "scene": "HyperNeRF/misc/scene",
            "query": "target",
            "target_category_id": 3,
            "target_category_name": "target",
            "evaluation_image_ids": ["000010", "000020"],
        }

        result = EVALUATOR.evaluate_query(
            row,
            predictions={"000010": gt, "000020": false_positive},
            gt_masks={"000010": gt},
            all_image_ids=["000010", "000020"],
            prediction_meta={},
        )

        self.assertEqual(result["mAcc"], 1.0)
        self.assertEqual(result["mIoU"], 1.0)
        self.assertEqual(result["reference_present_frame_mean_iou"], 1.0)
        self.assertAlmostEqual(result["pooled_mask_iou_all_test_frames"], 6.0 / 7.0)
        self.assertLess(result["binary_pixel_accuracy_all_test_frames"], 1.0)
        self.assertTrue(result["coverage_complete"])


if __name__ == "__main__":
    unittest.main()
