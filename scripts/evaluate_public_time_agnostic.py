#!/usr/bin/env python3
"""Evaluate ReferGaussian on the 4D LangSplat time-agnostic COCO protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _polygon_to_mask(size: tuple[int, int], segmentation: object) -> np.ndarray:
    width, height = size
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    polygons = segmentation if isinstance(segmentation, list) else []
    # The released 4D LangSplat evaluator reads segmentation[0]. Preserve that
    # behavior for direct parity and record the policy in the output protocol.
    polygons = polygons[:1]
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
            continue
        points = [
            (float(polygon[index]), float(polygon[index + 1]))
            for index in range(0, len(polygon), 2)
        ]
        draw.polygon(points, fill=255)
    return np.asarray(image, dtype=np.uint8) > 0


def _coco_scene_masks(
    coco_path: Path,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, str], list[str]]:
    payload = _read_json(coco_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid COCO payload: {coco_path}")
    images = {int(item["id"]): item for item in payload.get("images", [])}
    category_names = {
        int(item["id"]): str(item["name"])
        for item in payload.get("categories", [])
    }
    all_image_ids = [str(item["file_name"]).split("_")[0] for item in payload.get("images", [])]
    masks: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    for annotation in payload.get("annotations", []):
        category_id = int(annotation["category_id"])
        image_meta = images[int(annotation["image_id"])]
        image_id = str(image_meta["file_name"]).split("_")[0]
        size = (int(image_meta["width"]), int(image_meta["height"]))
        mask = _polygon_to_mask(size, annotation.get("segmentation", []))
        previous = masks[category_id].get(image_id)
        masks[category_id][image_id] = mask if previous is None else np.logical_or(previous, mask)
    return dict(masks), category_names, all_image_ids


def _manifest_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id", "")).strip()
            if not query_id or query_id in rows:
                raise ValueError(f"Invalid or duplicate query_id at {path}:{line_number}")
            rows[query_id] = row
    return rows


def _load_prediction(validation_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    validation = _read_json(validation_path)
    if not isinstance(validation, dict):
        raise ValueError(f"Invalid validation payload: {validation_path}")
    binary_mask_dir = Path(str(validation["frame_exports"]["binary_masks"]))
    predictions: dict[str, np.ndarray] = {}
    duplicate_ids: list[str] = []
    for row in validation.get("frames", []):
        image_id = str(row["image_id"])
        if image_id in predictions:
            duplicate_ids.append(image_id)
            continue
        mask_path = binary_mask_dir / f"{int(row['frame_index']):05d}.png"
        if mask_path.is_file():
            with Image.open(mask_path) as image:
                predictions[image_id] = np.asarray(image.convert("L"), dtype=np.uint8) > 0
        else:
            predictions[image_id] = np.zeros((0, 0), dtype=bool)
    return predictions, {
        "selection_status": str(validation.get("selection_status", "")),
        "duplicate_image_ids": sorted(set(duplicate_ids)),
        "validation_path": str(validation_path),
    }


def _resize_prediction(prediction: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if prediction.shape == shape:
        return prediction
    if prediction.size == 0:
        return np.zeros(shape, dtype=bool)
    image = Image.fromarray(prediction.astype(np.uint8) * 255)
    image = image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 0


def _safe_div(numerator: int | float, denominator: int | float, *, empty: float) -> float:
    return float(numerator) / float(denominator) if denominator else float(empty)


def evaluate_query(
    protocol_row: dict,
    *,
    predictions: dict[str, np.ndarray],
    gt_masks: dict[str, np.ndarray],
    all_image_ids: list[str],
    prediction_meta: dict,
) -> dict:
    category_name = str(protocol_row["target_category_name"])
    expected_ids = [str(value) for value in protocol_row.get("evaluation_image_ids", all_image_ids)]
    if set(expected_ids) != set(all_image_ids):
        raise ValueError(f"Protocol test-frame ids differ from COCO for {protocol_row['query_slug']}")

    reference_shape = next((mask.shape for mask in gt_masks.values()), None)
    if reference_shape is None:
        raise ValueError(f"No ground-truth masks for category {category_name}")

    frame_rows = []
    total_intersection = 0
    total_union = 0
    total_gt = 0
    total_correct_pixels = 0
    total_pixels = 0
    present_frame_ious = []
    present_frame_accuracies = []
    all_frame_ious = []
    missing_prediction_ids = []
    for image_id in expected_ids:
        gt = gt_masks.get(image_id)
        if gt is None:
            gt = np.zeros(reference_shape, dtype=bool)
        prediction = predictions.get(image_id)
        if prediction is None:
            missing_prediction_ids.append(image_id)
            prediction = np.zeros(reference_shape, dtype=bool)
        prediction = _resize_prediction(prediction, gt.shape)
        intersection = int(np.logical_and(prediction, gt).sum())
        union = int(np.logical_or(prediction, gt).sum())
        gt_pixels = int(gt.sum())
        correct_pixels = int((prediction == gt).sum())
        frame_iou = _safe_div(intersection, union, empty=1.0)
        frame_accuracy = _safe_div(correct_pixels, int(gt.size), empty=1.0)
        total_intersection += intersection
        total_union += union
        total_gt += gt_pixels
        total_correct_pixels += correct_pixels
        total_pixels += int(gt.size)
        all_frame_ious.append(frame_iou)
        if gt_pixels:
            present_frame_ious.append(frame_iou)
            present_frame_accuracies.append(frame_accuracy)
        frame_rows.append(
            {
                "image_id": image_id,
                "gt_present": bool(gt_pixels),
                "pred_present": bool(prediction.any()),
                "intersection_pixels": intersection,
                "union_pixels": union,
                "gt_pixels": gt_pixels,
                "iou": frame_iou,
                "pixel_accuracy": frame_accuracy,
            }
        )

    return {
        "query_id": str(protocol_row["query_slug"]),
        "scene": str(protocol_row["scene"]).rsplit("/", 1)[-1],
        "query": str(protocol_row["query"]),
        "target_category_id": int(protocol_row["target_category_id"]),
        "target_category_name": category_name,
        "test_frames": len(expected_ids),
        "category_present_frames": len(present_frame_ious),
        "mAcc": float(np.mean(present_frame_accuracies)),
        "mIoU": float(np.mean(present_frame_ious)),
        "reference_present_frame_mean_iou": float(np.mean(present_frame_ious)),
        "foreground_recall_all_test_frames": _safe_div(
            total_intersection, total_gt, empty=1.0
        ),
        "pooled_mask_iou_all_test_frames": _safe_div(
            total_intersection, total_union, empty=1.0
        ),
        "all_test_frame_mean_iou": float(np.mean(all_frame_ious)),
        "binary_pixel_accuracy_all_test_frames": _safe_div(
            total_correct_pixels, total_pixels, empty=1.0
        ),
        "intersection_pixels": total_intersection,
        "union_pixels": total_union,
        "gt_pixels": total_gt,
        "missing_prediction_image_ids": missing_prediction_ids,
        "coverage_complete": not missing_prediction_ids,
        **prediction_meta,
        "frames": frame_rows,
    }


def _macro(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _summary(rows: list[dict]) -> dict:
    return {
        "queries": len(rows),
        "mAcc": _macro(rows, "mAcc"),
        "mIoU": _macro(rows, "mIoU"),
        "reference_present_frame_mean_iou": _macro(rows, "reference_present_frame_mean_iou"),
        "foreground_recall_all_test_frames": _macro(
            rows, "foreground_recall_all_test_frames"
        ),
        "pooled_mask_iou_all_test_frames": _macro(
            rows, "pooled_mask_iou_all_test_frames"
        ),
        "all_test_frame_mean_iou": _macro(rows, "all_test_frame_mean_iou"),
        "binary_pixel_accuracy_all_test_frames": _macro(
            rows, "binary_pixel_accuracy_all_test_frames"
        ),
        "coverage_complete": all(bool(row["coverage_complete"]) for row in rows),
        "missing_prediction_frames": sum(len(row["missing_prediction_image_ids"]) for row in rows),
    }


def _write_markdown(path: Path, payload: dict) -> None:
    summary = payload["summary"]
    lines = [
        "# Public Time-Agnostic Evaluation",
        "",
        f"- mAcc: {100.0 * summary['mAcc']:.2f}%",
        f"- mIoU: {100.0 * summary['mIoU']:.2f}%",
        f"- Pooled all-test-frame mask IoU (audit): {100.0 * summary['pooled_mask_iou_all_test_frames']:.2f}%",
        f"- Queries: {summary['queries']}",
        f"- Coverage complete: {summary['coverage_complete']}",
        "",
        "| Scene | Query | mAcc | mIoU | Pooled all-frame IoU | Coverage |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["per_query"]:
        lines.append(
            f"| {row['scene']} | {row['target_category_name']} | "
            f"{100.0 * row['mAcc']:.2f}% | {100.0 * row['mIoU']:.2f}% | "
            f"{100.0 * row['pooled_mask_iou_all_test_frames']:.2f}% | "
            f"{row['coverage_complete']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-json", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--query-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--scene", action="append", dest="scenes")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    protocol_path = Path(args.protocol_json).resolve()
    manifest_path = Path(args.manifest).resolve()
    query_root = Path(args.query_root).resolve()
    annotation_root = Path(args.annotation_root).resolve()
    protocol = _read_json(protocol_path)
    if not isinstance(protocol, dict) or not isinstance(protocol.get("queries"), list):
        raise ValueError(f"Invalid protocol: {protocol_path}")
    manifest_rows = _manifest_rows(manifest_path)
    requested_scenes = None if not args.scenes else set(args.scenes)

    scene_cache: dict[str, tuple[dict[int, dict[str, np.ndarray]], dict[int, str], list[str]]] = {}
    per_query = []
    for row in protocol["queries"]:
        if not isinstance(row, dict) or row.get("category") != "time_agnostic_reference":
            continue
        scene = str(row["scene"]).rsplit("/", 1)[-1]
        if requested_scenes is not None and scene not in requested_scenes:
            continue
        query_id = str(row["query_slug"])
        if query_id not in manifest_rows:
            continue
        if scene not in scene_cache:
            coco_path = annotation_root / scene / "train" / "_annotations.coco.json"
            scene_cache[scene] = _coco_scene_masks(coco_path)
        scene_masks, category_names, all_image_ids = scene_cache[scene]
        category_id = int(row["target_category_id"])
        if category_names.get(category_id) != str(row["target_category_name"]):
            raise ValueError(f"Category identity mismatch for {query_id}")
        validation_path = query_root / query_id / "final_query_render_sourcebg" / "validation.json"
        predictions, prediction_meta = _load_prediction(validation_path)
        per_query.append(
            evaluate_query(
                row,
                predictions=predictions,
                gt_masks=scene_masks.get(category_id, {}),
                all_image_ids=all_image_ids,
                prediction_meta=prediction_meta,
            )
        )
    if not per_query:
        raise ValueError("No time-agnostic protocol queries matched the manifest")

    scene_summaries = {
        scene: _summary([row for row in per_query if row["scene"] == scene])
        for scene in sorted({row["scene"] for row in per_query})
    }
    payload = {
        "metric_protocol": {
            "id": "4dlangsplat_time_agnostic_v1",
            "mAcc": (
                "macro category mean of per-frame binary pixel accuracy "
                "(TP+TN)/(H*W) on category-present test frames"
            ),
            "mIoU": (
                "macro category mean of per-frame mask IoU on category-present "
                "test frames, matching the released 4D LangSplat eval loop"
            ),
            "reference_present_frame_mean_iou": (
                "compatibility alias of mIoU"
            ),
            "foreground_recall_all_test_frames": "sum TP / sum GT pixels across all declared test frames",
            "pooled_mask_iou_all_test_frames": "sum intersection / sum union across all declared test frames",
            "binary_pixel_accuracy_all_test_frames": (
                "sum correct foreground/background pixels / all pixels across all declared test frames"
            ),
            "polygon_policy": "first_polygon_per_annotation_for_4dlangsplat_code_parity",
            "empty_mask_rule": "empty_gt_and_prediction_scores_one; empty_gt_with_prediction_scores_zero",
            "camera_matching": "exact_source_image_id_required",
        },
        "source_files": {
            "protocol_json": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
            "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        },
        "query_root": str(query_root),
        "summary": _summary(per_query),
        "per_scene": scene_summaries,
        "per_query": per_query,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(output_md, payload)
    print(json.dumps(payload["summary"], indent=2))
    if args.require_complete and not payload["summary"]["coverage_complete"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
