"""Regression tests for the public R4D-Bench-QA evaluator."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = REPO_ROOT / "scripts" / "evaluate_ours_benchmark.py"
SPEC = importlib.util.spec_from_file_location("refergaussian_benchmark_evaluator", EVALUATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def _write_validation(query_root: Path, frames: list[dict]) -> Path:
    render_root = query_root / "final_query_render_sourcebg"
    render_root.mkdir(parents=True)
    (render_root / "validation.json").write_text(
        json.dumps(
            {
                "frames": frames,
                "frame_exports": {"binary_masks": "binary_masks"},
            }
        ),
        encoding="utf-8",
    )
    return render_root


class BenchmarkEvaluatorTest(unittest.TestCase):
    def test_complete_report_rejects_unresolved_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            query_root = Path(temp_dir)
            render_root = query_root / "final_query_render_sourcebg"
            render_root.mkdir(parents=True)
            (render_root / "validation.json").write_text(
                json.dumps({"selection_status": "unresolved", "frames": []}),
                encoding="utf-8",
            )

            result = EVALUATOR.evaluate_query(
                {"query_id": "scene_q1", "question": "an absent target"},
                query_root,
                require_resolved_selection=True,
            )

        self.assertEqual(result["status"], "unresolved_selection")
        self.assertIsNone(result["Acc"])
        self.assertFalse(result["selection_resolution_complete"])

    def test_file_identity_records_evaluator_input_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "benchmark.json"
            path.write_text("[]\n", encoding="utf-8")
            identity = EVALUATOR._file_identity(path)

        self.assertIsNotNone(identity)
        assert identity is not None
        self.assertEqual(identity["bytes"], 3)
        self.assertEqual(len(str(identity["sha256"])), 64)

    def test_sparse_temporal_predictions_are_densified_without_changing_sample_labels(self) -> None:
        dense = EVALUATOR._densify_sparse_temporal_prediction(
            {2: False, 6: True, 10: True},
            total_frames=13,
        )

        self.assertEqual(dense.astype(int).tolist(), [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertFalse(dense[2])
        self.assertTrue(dense[6])
        self.assertTrue(dense[10])

    def test_coverage_marks_missing_manifest_outputs_incomplete(self) -> None:
        coverage = EVALUATOR.coverage_summary(
            ["query_a", "query_b"],
            [
                {"query_id": "query_a", "Acc": 1.0},
                {"query_id": "query_b", "Acc": None},
            ],
        )

        self.assertFalse(coverage["complete"])
        self.assertEqual(coverage["missing_query_ids"], ["query_b"])

    def test_summary_separates_empty_target_outcomes(self) -> None:
        nonempty = {
            "Acc": 0.25,
            "vIoU": 0.40,
            "tIoU": 0.30,
            "temporal_precision": 0.50,
            "temporal_recall": 0.20,
            "temporal_f1": 0.29,
            "gt_active_count": 4,
            "pred_active_count": 2,
            "empty_query_correct": False,
            "score_warnings": [],
        }
        empty = {
            "Acc": 1.0,
            "vIoU": 1.0,
            "tIoU": 1.0,
            "temporal_precision": 1.0,
            "temporal_recall": 1.0,
            "temporal_f1": 1.0,
            "gt_active_count": 0,
            "pred_active_count": 0,
            "empty_query_correct": True,
            "score_warnings": [],
        }

        summary = EVALUATOR.summarize_query_results([nonempty, empty], total_queries=2)

        self.assertEqual(summary["zero_target_queries"], 1)
        self.assertEqual(summary["zero_target_correct"], 1)
        self.assertEqual(summary["nonempty_queries"], 1)
        self.assertEqual(summary["nonempty_only"]["vIoU"], 0.40)

    def test_empty_query_scores_perfectly_only_when_prediction_is_empty(self) -> None:
        query_item = {
            "query_id": "empty_q1",
            "ground_truth": {"existence_frames": [], "frames": []},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            query_root = Path(temp_dir) / "empty_q1"
            _write_validation(
                query_root,
                [
                    {"image_id": "0000", "frame_index": 0, "query_active": False},
                    {"image_id": "0001", "frame_index": 1, "query_active": False},
                ],
            )
            result = EVALUATOR.evaluate_query(query_item, query_root)

        self.assertEqual(result["Acc"], 1.0)
        self.assertEqual(result["tIoU"], 1.0)
        self.assertEqual(result["vIoU"], 1.0)
        self.assertEqual(result["temporal_frame_accuracy"], result["Acc"])
        self.assertEqual(result["mean_annotated_frame_iou"], result["vIoU"])
        self.assertEqual(result["annotated_volume_iou"], 1.0)
        self.assertIsNone(result["paper_exact_set_accuracy"])
        self.assertIsNone(result["paper_full_volume_iou"])
        self.assertEqual(result["vIoU_count"], 0)

    def test_empty_query_false_positive_is_not_awarded(self) -> None:
        query_item = {
            "query_id": "empty_q1",
            "ground_truth": {"existence_frames": [], "frames": []},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            query_root = Path(temp_dir) / "empty_q1"
            _write_validation(
                query_root,
                [
                    {"image_id": "0000", "frame_index": 0, "query_active": True},
                    {"image_id": "0001", "frame_index": 1, "query_active": False},
                ],
            )
            result = EVALUATOR.evaluate_query(query_item, query_root)

        self.assertEqual(result["Acc"], 0.5)
        self.assertEqual(result["tIoU"], 0.0)
        self.assertEqual(result["vIoU"], 0.0)
        self.assertEqual(result["annotated_volume_iou"], 0.0)

    def test_mean_frame_iou_is_not_relabelled_as_volume_iou(self) -> None:
        small_polygon = [[1, 1, 2, 1, 2, 2, 1, 2]]
        large_polygon = [[1, 1, 6, 1, 6, 4, 1, 4]]
        query_item = {
            "query_id": "volume_q1",
            "ground_truth": {
                "existence_frames": [0, 1],
                "frames": [
                    {"frame_id": 0, "masks": [{"segmentation": small_polygon}]},
                    {"frame_id": 1, "masks": [{"segmentation": large_polygon}]},
                ],
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            query_root = Path(temp_dir) / "volume_q1"
            render_root = _write_validation(
                query_root,
                [
                    {"image_id": "0000", "frame_index": 0, "query_active": True},
                    {"image_id": "0001", "frame_index": 1, "query_active": False},
                ],
            )
            mask_dir = render_root / "binary_masks"
            mask_dir.mkdir()
            image = Image.fromarray(np.zeros((6, 8), dtype=np.uint8), mode="L")
            ImageDraw.Draw(image).polygon(
                [(small_polygon[0][idx], small_polygon[0][idx + 1]) for idx in range(0, 8, 2)],
                fill=255,
            )
            image.save(mask_dir / "00000.png")
            result = EVALUATOR.evaluate_query(query_item, query_root)

        self.assertEqual(result["vIoU"], 0.5)
        self.assertLess(result["annotated_volume_iou"], result["vIoU"])
        self.assertAlmostEqual(
            result["annotated_volume_iou"],
            result["spatial_intersection_pixels"] / result["spatial_union_pixels"],
        )

    def test_polygon_annotation_uses_the_prediction_canvas(self) -> None:
        polygon = [[2, 1, 5, 1, 5, 4, 2, 4]]
        query_item = {
            "query_id": "polygon_q1",
            "ground_truth": {
                "existence_frames": [0],
                "frames": [{"frame_id": 0, "masks": [{"segmentation": polygon}]}],
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            query_root = Path(temp_dir) / "polygon_q1"
            render_root = _write_validation(
                query_root,
                [{"image_id": "0000", "frame_index": 0, "query_active": True}],
            )
            mask_dir = render_root / "binary_masks"
            mask_dir.mkdir()
            image = Image.fromarray(np.zeros((6, 8), dtype=np.uint8), mode="L")
            ImageDraw.Draw(image).polygon(
                [(polygon[0][idx], polygon[0][idx + 1]) for idx in range(0, len(polygon[0]), 2)],
                fill=255,
            )
            image.save(mask_dir / "00000.png")
            result = EVALUATOR.evaluate_query(query_item, query_root)

        self.assertEqual(result["Acc"], 1.0)
        self.assertEqual(result["tIoU"], 1.0)
        self.assertEqual(result["vIoU"], 1.0)

    def test_source_image_canvas_precedes_camera_intrinsics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            source_dir = dataset_dir / "rgb" / "2x"
            source_dir.mkdir(parents=True)
            Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8), mode="RGB").save(source_dir / "000001.png")
            camera_dir = dataset_dir / "camera"
            camera_dir.mkdir()
            (camera_dir / "0000.json").write_text(
                json.dumps({"image_size": [16, 12]}), encoding="utf-8"
            )

            canvas = EVALUATOR._infer_annotation_image_size([], dataset_dir, (12, 16))

        self.assertEqual(canvas, (6, 8))

    def test_exact_source_camera_output_is_preferred_over_nearest_time(self) -> None:
        polygon = [[2, 1, 5, 1, 5, 4, 2, 4]]
        query_item = {
            "query_id": "direct_camera_q1",
            "ground_truth": {
                "existence_frames": [0],
                "frames": [{"frame_id": 0, "masks": [{"segmentation": polygon}]}],
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.json").write_text(
                json.dumps({"000000": {"time_id": 0}}), encoding="utf-8"
            )
            query_root = root / "direct_camera_q1"
            render_root = _write_validation(
                query_root,
                [
                    {"image_id": "000000", "frame_index": 0, "query_active": True},
                    {"image_id": "000010", "frame_index": 1, "query_active": True},
                ],
            )
            mask_dir = render_root / "binary_masks"
            mask_dir.mkdir()
            exact = Image.fromarray(np.zeros((6, 8), dtype=np.uint8), mode="L")
            ImageDraw.Draw(exact).polygon(
                [(polygon[0][idx], polygon[0][idx + 1]) for idx in range(0, len(polygon[0]), 2)],
                fill=255,
            )
            exact.save(mask_dir / "00000.png")
            Image.fromarray(np.zeros((6, 8), dtype=np.uint8), mode="L").save(mask_dir / "00001.png")

            result = EVALUATOR.evaluate_query(query_item, query_root, dataset_dir=dataset_dir)

        self.assertEqual(result["vIoU"], 1.0)
        self.assertEqual(result["spatial_direct_camera_match_frames"], 1)

    def test_missing_spatial_prediction_counts_as_zero_and_fails_spatial_coverage(self) -> None:
        polygon = [[2, 1, 5, 1, 5, 4, 2, 4]]
        query_item = {
            "query_id": "missing_frame_q1",
            "ground_truth": {
                "existence_frames": [0, 10],
                "frames": [
                    {"frame_id": 0, "masks": [{"segmentation": polygon}]},
                    {"frame_id": 10, "masks": [{"segmentation": polygon}]},
                ],
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            query_root = Path(temp_dir) / "missing_frame_q1"
            render_root = _write_validation(
                query_root,
                [{"image_id": "0000", "frame_index": 0, "query_active": True}],
            )
            mask_dir = render_root / "binary_masks"
            mask_dir.mkdir()
            image = Image.fromarray(np.zeros((6, 8), dtype=np.uint8), mode="L")
            ImageDraw.Draw(image).polygon(
                [(polygon[0][idx], polygon[0][idx + 1]) for idx in range(0, len(polygon[0]), 2)],
                fill=255,
            )
            image.save(mask_dir / "00000.png")
            result = EVALUATOR.evaluate_query(query_item, query_root)

        self.assertEqual(result["spatial_gt_mask_frames"], 2)
        self.assertEqual(result["spatial_matched_render_frames"], 1)
        self.assertEqual(result["mask_missing"], 1)
        self.assertFalse(result["spatial_coverage_complete"])
        self.assertEqual(result["vIoU"], 0.5)
        coverage = EVALUATOR.coverage_summary(["missing_frame_q1"], [result])
        self.assertFalse(coverage["complete"])
        self.assertEqual(coverage["spatial_incomplete_query_ids"], ["missing_frame_q1"])


if __name__ == "__main__":
    unittest.main()
