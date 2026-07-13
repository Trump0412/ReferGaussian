"""Regression tests for the public time-sensitive query evaluator."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = REPO_ROOT / "scripts" / "evaluate_public_query_protocol.py"
SPEC = importlib.util.spec_from_file_location("refergaussian_public_evaluator", EVALUATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def _query(target_ranges: list[list[int]]) -> dict:
    return {
        "query_slug": "target_q1",
        "query": "the target object",
        "targets": [{"target_ranges": target_ranges}],
    }


def _validation(mask_dir: Path, active: list[bool]) -> dict:
    return {
        "frame_exports": {"binary_masks": str(mask_dir)},
        "frames": [
            {"image_id": f"{index:04d}", "frame_index": index, "query_active": value}
            for index, value in enumerate(active)
        ],
    }


class PublicProtocolEvaluatorTest(unittest.TestCase):
    def test_coverage_marks_missing_protocol_outputs_incomplete(self) -> None:
        coverage = EVALUATOR.coverage_summary(
            ["query_a", "query_b"],
            [
                {"query_slug": "query_a", "Acc": 1.0},
                {"query_slug": "query_b", "Acc": None},
            ],
        )

        self.assertFalse(coverage["complete"])
        self.assertEqual(coverage["missing_query_ids"], ["query_b"])

    def test_summary_separates_empty_target_outcomes(self) -> None:
        nonempty = {
            "Acc": 0.25,
            "vIoU": 0.40,
            "temporal_tIoU": 0.30,
            "temporal_precision": 0.50,
            "temporal_recall": 0.20,
            "temporal_f1": 0.29,
            "temporal_gt_active_count": 4,
            "temporal_pred_active_count": 2,
            "empty_query_correct": False,
            "score_warnings": [],
        }
        empty = {
            "Acc": 1.0,
            "vIoU": 1.0,
            "temporal_tIoU": 1.0,
            "temporal_precision": 1.0,
            "temporal_recall": 1.0,
            "temporal_f1": 1.0,
            "temporal_gt_active_count": 0,
            "temporal_pred_active_count": 0,
            "empty_query_correct": True,
            "score_warnings": [],
        }

        summary = EVALUATOR.summarize_query_results([nonempty, empty], query_count=2)

        self.assertEqual(summary["zero_target_queries"], 1)
        self.assertEqual(summary["zero_target_correct"], 1)
        self.assertEqual(summary["nonempty_queries"], 1)
        self.assertEqual(summary["nonempty_only"]["vIoU"], 0.40)

    def test_empty_query_scores_perfectly_only_when_prediction_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = EVALUATOR.evaluate_query(
                query_item=_query([]),
                validation_payload=_validation(Path(temp_dir), [False, False]),
                metadata_payload={"0000": {"time_id": 0}, "0001": {"time_id": 1}},
                gt_masks_by_object={"target": {}},
                top_level_objects=["target"],
            )

        self.assertEqual(result["Acc"], 1.0)
        self.assertEqual(result["vIoU"], 1.0)
        self.assertEqual(result["temporal_tIoU"], 1.0)

    def test_empty_query_false_positive_scores_zero_spatiotemporally(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = EVALUATOR.evaluate_query(
                query_item=_query([]),
                validation_payload=_validation(Path(temp_dir), [True, False]),
                metadata_payload={"0000": {"time_id": 0}, "0001": {"time_id": 1}},
                gt_masks_by_object={"target": {}},
                top_level_objects=["target"],
            )

        self.assertEqual(result["Acc"], 0.5)
        self.assertEqual(result["vIoU"], 0.0)
        self.assertEqual(result["temporal_tIoU"], 0.0)

    def test_matching_active_mask_scores_perfectly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mask_dir = Path(temp_dir) / "binary_masks"
            mask_dir.mkdir()
            target_mask = np.zeros((5, 6), dtype=np.uint8)
            target_mask[1:4, 2:5] = 255
            Image.fromarray(target_mask, mode="L").save(mask_dir / "00000.png")
            result = EVALUATOR.evaluate_query(
                query_item=_query([[0, 0]]),
                validation_payload=_validation(mask_dir, [True]),
                metadata_payload={"0000": {"time_id": 0}},
                gt_masks_by_object={"target": {"0000": target_mask > 0}},
                top_level_objects=["target"],
            )

        self.assertEqual(result["Acc"], 1.0)
        self.assertEqual(result["vIoU"], 1.0)
        self.assertEqual(result["temporal_tIoU"], 1.0)


if __name__ == "__main__":
    unittest.main()
