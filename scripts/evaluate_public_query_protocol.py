import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


METRIC_PROTOCOL = {
    "id": "legacy_public_evaluator_v1_with_audit_fields",
    "legacy_aliases": {
        "Acc": "temporal_frame_accuracy",
        "vIoU": "mean_annotated_frame_iou",
        "temporal_tIoU": "temporal_iou",
    },
    "audit_fields": {
        "annotated_volume_iou": (
            "pixel intersection sum divided by pixel union sum over available "
            "annotation-mask frames in the temporal union"
        ),
        "paper_exact_set_accuracy": "unavailable_without_auditable_predicted_and_gt_instance_sets",
        "paper_full_volume_iou": "not_asserted_unless_annotation-frame_coverage_is_exhaustive",
    },
    "empty_set_rule": {
        "empty_gt_empty_prediction": 1.0,
        "empty_gt_nonempty_prediction": 0.0,
    },
}


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _file_identity(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def _read_manifest_query_ids(path: Path) -> set[str]:
    query_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL row {line_number} in {path}: {exc}") from exc
            query_id = str(row.get("query_id", "")).strip()
            if not query_id:
                raise ValueError(f"Manifest row {line_number} has no query_id: {path}")
            query_ids.add(query_id)
    if not query_ids:
        raise ValueError(f"Manifest contains no query ids: {path}")
    return query_ids


def _norm(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("-", " ").replace("_", " ").split())


def _frames_from_ranges(ranges: list[list[int]]) -> set[int]:
    frames: set[int] = set()
    for start, end in ranges:
        start_i = int(start)
        end_i = int(end)
        if end_i < start_i:
            continue
        frames.update(range(start_i, end_i + 1))
    return frames


def _polygon_to_mask(size: tuple[int, int], segmentation) -> np.ndarray:
    width, height = int(size[0]), int(size[1])
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    if isinstance(segmentation, list):
        for polygon in segmentation:
            if not polygon:
                continue
            xy = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
            if len(xy) >= 3:
                draw.polygon(xy, fill=255)
    return np.asarray(mask, dtype=np.uint8) > 0


def _build_gt_masks(annotation_dir: Path) -> tuple[dict[str, tuple[int, int]], dict[str, dict[int, np.ndarray]], list[str]]:
    coco_path = annotation_dir / "train" / "_annotations.coco.json"
    video_annotation_path = annotation_dir / "video_annotations.json"
    coco_payload = _read_json(coco_path)
    video_annotations = _read_json(video_annotation_path)
    top_level_objects = [_norm(key) for key in video_annotations.keys()]

    category_name_by_id = {int(item["id"]): _norm(item["name"]) for item in coco_payload.get("categories", [])}
    image_meta = {
        int(item["id"]): (str(item["file_name"]), (int(item["width"]), int(item["height"])))
        for item in coco_payload.get("images", [])
    }
    masks_by_object: dict[str, dict[int, np.ndarray]] = {name: {} for name in top_level_objects}
    name_to_category_ids = defaultdict(list)
    for category_id, category_name in category_name_by_id.items():
        name_to_category_ids[category_name].append(category_id)

    annotations_by_image = defaultdict(list)
    for ann in coco_payload.get("annotations", []):
        annotations_by_image[int(ann["image_id"])].append(ann)

    for image_id, (file_name, size) in image_meta.items():
        image_key = file_name.split("_")[0]
        ann_list = annotations_by_image.get(image_id, [])
        if not ann_list:
            continue
        for object_name in top_level_objects:
            category_ids = set(name_to_category_ids.get(object_name, []))
            if not category_ids:
                continue
            merged_mask = np.zeros((int(size[1]), int(size[0])), dtype=bool)
            for ann in ann_list:
                if int(ann["category_id"]) not in category_ids:
                    continue
                merged_mask |= _polygon_to_mask(size, ann.get("segmentation", []))
            if merged_mask.any():
                masks_by_object[object_name][image_key] = merged_mask
    return image_meta, masks_by_object, top_level_objects


def _object_for_query(query_text: str, top_level_objects: list[str]) -> str:
    query_norm = _norm(query_text)
    matches = [name for name in top_level_objects if name in query_norm]
    if matches:
        matches.sort(key=len, reverse=True)
        return matches[0]
    if len(top_level_objects) == 1:
        return top_level_objects[0]
    raise ValueError(f"Unable to infer target object for query: {query_text}")


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return _safe_div(inter, union)


def _temporal_prf(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float, float, int, int, int]:
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    inter = int(np.logical_and(pred, gt).sum())
    pred_count = int(pred.sum())
    gt_count = int(gt.sum())
    precision = 1.0 if pred_count == 0 and gt_count == 0 else _safe_div(inter, pred_count)
    recall = 1.0 if pred_count == 0 and gt_count == 0 else _safe_div(inter, gt_count)
    f1 = _safe_div(2.0 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, inter, pred_count, gt_count


def _ranges_from_bool_mask(mask: np.ndarray) -> list[list[int]]:
    indices = np.where(np.asarray(mask, dtype=bool))[0]
    if indices.size == 0:
        return []
    ranges: list[list[int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        ranges.append([start, prev])
        start = value
        prev = value
    ranges.append([start, prev])
    return ranges


def _represented_intervals(time_ids: list[int], total_frames: int) -> list[list[int]]:
    if not time_ids:
        return []
    sorted_ids = sorted(int(value) for value in time_ids)
    intervals: list[list[int]] = []
    for index, current in enumerate(sorted_ids):
        if index == 0:
            start = 0
        else:
            start = (sorted_ids[index - 1] + current) // 2 + 1
        if index == len(sorted_ids) - 1:
            end = int(total_frames - 1)
        else:
            end = (current + sorted_ids[index + 1]) // 2
        intervals.append([int(start), int(end)])
    return intervals


def _rendered_activity_mask(frame_rows: list[dict], metadata_payload: dict, total_frames: int) -> tuple[np.ndarray, list[dict]]:
    rows = []
    for frame_row in frame_rows:
        image_id = str(frame_row["image_id"])
        time_id = int(metadata_payload[image_id]["time_id"])
        rows.append(
            {
                "frame_index": int(frame_row["frame_index"]),
                "image_id": image_id,
                "time_id": time_id,
                "query_active": bool(frame_row["query_active"]),
            }
        )
    rows.sort(key=lambda item: item["time_id"])
    intervals = _represented_intervals([row["time_id"] for row in rows], total_frames=total_frames)
    full_mask = np.zeros((int(total_frames),), dtype=bool)
    for row, interval in zip(rows, intervals):
        row["represented_interval"] = interval
        if not row["query_active"]:
            continue
        start, end = int(interval[0]), int(interval[1])
        if end < start:
            continue
        full_mask[start : end + 1] = True
    return full_mask, rows


def _nearest_render_row(time_id: int, rendered_rows: list[dict], max_distance: int) -> dict | None:
    if not rendered_rows:
        return None
    best = min(rendered_rows, key=lambda row: abs(int(row["time_id"]) - int(time_id)))
    if abs(int(best["time_id"]) - int(time_id)) > int(max_distance):
        return None
    return best


def evaluate_query(
    query_item: dict,
    validation_payload: dict,
    metadata_payload: dict,
    gt_masks_by_object: dict[str, dict[str, np.ndarray]],
    top_level_objects: list[str],
) -> dict:
    query_text = str(query_item["query"])
    query_slug = str(query_item["query_slug"])
    target_object = _object_for_query(query_text, top_level_objects)
    gt_frames = _frames_from_ranges(query_item["targets"][0]["target_ranges"])
    total_frames = max(int(meta["time_id"]) for meta in metadata_payload.values()) + 1

    binary_mask_dir = Path(validation_payload["frame_exports"]["binary_masks"])
    frame_rows = validation_payload.get("frames", [])
    if not frame_rows:
        raise ValueError(f"No frames in validation payload for {query_slug}")

    iou_sum = 0.0
    spatial_intersection_pixels = 0
    spatial_union_pixels = 0
    union_count = 0
    overlap_count = 0
    gt_mask_frames = 0
    matched_render_frames = 0
    missing_render_frames = 0
    missing_binary_mask_frames = 0
    unmapped_gt_mask_frames = 0

    gt_object_masks = gt_masks_by_object.get(target_object, {})
    pred_full_mask, rendered_rows = _rendered_activity_mask(frame_rows, metadata_payload=metadata_payload, total_frames=total_frames)
    gt_full_mask = np.zeros((int(total_frames),), dtype=bool)
    if gt_frames:
        valid_gt_frames = sorted(
            {
                int(frame_index)
                for frame_index in gt_frames
                if 0 <= int(frame_index) < int(total_frames)
            }
        )
        if valid_gt_frames:
            gt_full_mask[valid_gt_frames] = True
    acc = float(np.mean(pred_full_mask == gt_full_mask)) if total_frames > 0 else 0.0
    per_frame_rows = []
    for row in rendered_rows:
        image_id = str(row["image_id"])
        time_id = int(row["time_id"])
        pred_active = bool(row["query_active"])
        gt_active = 0 <= int(time_id) < int(total_frames) and bool(gt_full_mask[time_id])
        per_frame_rows.append(
            {
                "frame_index": int(row["frame_index"]),
                "image_id": image_id,
                "time_id": time_id,
                "pred_active": pred_active,
                "gt_active": gt_active,
                "represented_interval": row["represented_interval"],
            }
        )
    rendered_time_ids = [int(row["time_id"]) for row in rendered_rows]
    time_diffs = np.diff(np.asarray(rendered_time_ids, dtype=np.int32)) if len(rendered_time_ids) >= 2 else np.asarray([], dtype=np.int32)
    max_distance = int(max(2, int(np.median(time_diffs)) // 2 + 1)) if time_diffs.size else 2
    for image_id, gt_mask in gt_object_masks.items():
        if image_id not in metadata_payload:
            unmapped_gt_mask_frames += 1
            continue
        time_id = int(metadata_payload[image_id]["time_id"])
        pred_active = bool(pred_full_mask[time_id])
        gt_active = bool(gt_full_mask[time_id])
        if not (pred_active or gt_active):
            continue
        gt_mask_frames += 1
        union_count += 1
        row = _nearest_render_row(time_id=time_id, rendered_rows=rendered_rows, max_distance=max_distance)
        if row is None:
            missing_render_frames += 1
            if gt_active:
                spatial_union_pixels += int(np.asarray(gt_mask, dtype=bool).sum())
            continue
        matched_render_frames += 1
        pred_mask = np.zeros_like(gt_mask, dtype=bool)
        if pred_active:
            pred_mask_path = binary_mask_dir / f"{int(row['frame_index']):05d}.png"
            if pred_mask_path.exists():
                with Image.open(pred_mask_path) as image:
                    pred_mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
                if pred_mask.shape != gt_mask.shape:
                    pred_image = Image.fromarray(pred_mask.astype(np.uint8) * 255)
                    pred_image = pred_image.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
                    pred_mask = np.asarray(pred_image) > 0
            else:
                missing_binary_mask_frames += 1
        eval_gt_mask = np.asarray(gt_mask, dtype=bool) if gt_active else np.zeros_like(gt_mask, dtype=bool)
        frame_intersection = int(np.logical_and(pred_mask, eval_gt_mask).sum())
        frame_union = int(np.logical_or(pred_mask, eval_gt_mask).sum())
        spatial_intersection_pixels += frame_intersection
        spatial_union_pixels += frame_union
        if pred_active and gt_active:
            overlap_count += 1
            iou_sum += 1.0 if frame_union == 0 else _safe_div(frame_intersection, frame_union)

    predicted_ranges = validation_payload.get("active_segments", [])
    predicted_time_ranges = _ranges_from_bool_mask(pred_full_mask)
    temporal_union = int(np.logical_or(pred_full_mask, gt_full_mask).sum())
    temporal_inter = int(np.logical_and(pred_full_mask, gt_full_mask).sum())
    temporal_iou = 1.0 if temporal_union == 0 else _safe_div(temporal_inter, temporal_union)
    temporal_precision, temporal_recall, temporal_f1, _, pred_active_count, gt_active_count = _temporal_prf(
        pred_full_mask,
        gt_full_mask,
    )
    viou = 1.0 if union_count == 0 and temporal_union == 0 else _safe_div(iou_sum, union_count)
    if spatial_union_pixels == 0:
        annotated_volume_iou = 1.0 if temporal_union == 0 else 0.0
    else:
        annotated_volume_iou = _safe_div(spatial_intersection_pixels, spatial_union_pixels)
    warnings = []
    if acc >= 0.90 and temporal_iou < 0.50:
        warnings.append("high_acc_low_tiou")
    if viou < 0.10 and union_count > 0:
        warnings.append("viou_below_10_check_entity_match")
    if missing_render_frames or missing_binary_mask_frames or unmapped_gt_mask_frames:
        warnings.append("missing_spatial_prediction")

    return {
        "query_slug": query_slug,
        "query": query_text,
        "target_object": target_object,
        "frames_evaluated": int(len(frame_rows)),
        "timeline_frames_evaluated": int(total_frames),
        "gt_mask_frames": int(gt_mask_frames),
        "spatial_matched_render_frames": int(matched_render_frames),
        "spatial_missing_render_frames": int(missing_render_frames),
        "spatial_missing_binary_mask_frames": int(missing_binary_mask_frames),
        "spatial_unmapped_gt_mask_frames": int(unmapped_gt_mask_frames),
        "spatial_coverage_complete": bool(
            missing_render_frames == 0
            and missing_binary_mask_frames == 0
            and unmapped_gt_mask_frames == 0
        ),
        "Acc": acc,
        "vIoU": viou,
        "temporal_tIoU": temporal_iou,
        "temporal_frame_accuracy": acc,
        "mean_annotated_frame_iou": viou,
        "annotated_volume_iou": annotated_volume_iou,
        "paper_exact_set_accuracy": None,
        "paper_full_volume_iou": None,
        "temporal_precision": temporal_precision,
        "temporal_recall": temporal_recall,
        "temporal_f1": temporal_f1,
        "temporal_pred_active_count": int(pred_active_count),
        "temporal_gt_active_count": int(gt_active_count),
        "empty_query_correct": bool(gt_active_count == 0 and pred_active_count == 0),
        "temporal_inter": int(temporal_inter),
        "temporal_union": int(temporal_union),
        "mask_union_frames": int(union_count),
        "mask_overlap_frames": int(overlap_count),
        "spatial_intersection_pixels": spatial_intersection_pixels,
        "spatial_union_pixels": spatial_union_pixels,
        "score_warnings": warnings,
        "predicted_render_frame_segments": predicted_ranges,
        "predicted_time_segments": predicted_time_ranges,
        "gt_time_segments": query_item["targets"][0]["target_ranges"],
        "validation_path": validation_payload.get("_validation_path", ""),
        "frame_rows": per_frame_rows,
    }


_METRIC_KEYS = (
    "Acc",
    "vIoU",
    "temporal_tIoU",
    "temporal_frame_accuracy",
    "mean_annotated_frame_iou",
    "annotated_volume_iou",
    "temporal_precision",
    "temporal_recall",
    "temporal_f1",
)


def _mean_metrics(rows: list[dict]) -> dict[str, float | None]:
    result = {}
    for key in _METRIC_KEYS:
        values = [row.get(key) for row in rows if row.get(key) is not None]
        result[key] = float(np.mean(values)) if values else None
    return result


def summarize_query_results(valid: list[dict], query_count: int) -> dict:
    """Expose empty-target outcomes separately from ordinary referring queries."""

    nonempty = [row for row in valid if int(row.get("temporal_gt_active_count", 0)) > 0]
    empty = [row for row in valid if int(row.get("temporal_gt_active_count", 0)) == 0]
    return {
        "query_count": int(query_count),
        "valid_queries": int(len(valid)),
        **_mean_metrics(valid),
        "nonempty_queries": int(len(nonempty)),
        "nonempty_only": _mean_metrics(nonempty),
        "zero_target_queries": int(len(empty)),
        "zero_target_correct": int(sum(bool(row.get("empty_query_correct")) for row in empty)),
        "zero_target_false_positive": int(sum(not bool(row.get("empty_query_correct")) for row in empty)),
        "spatial_gt_mask_frames": int(sum(int(row.get("gt_mask_frames", 0)) for row in valid)),
        "spatial_matched_render_frames": int(
            sum(int(row.get("spatial_matched_render_frames", 0)) for row in valid)
        ),
        "spatial_missing_render_frames": int(
            sum(int(row.get("spatial_missing_render_frames", 0)) for row in valid)
        ),
        "spatial_missing_binary_mask_frames": int(
            sum(int(row.get("spatial_missing_binary_mask_frames", 0)) for row in valid)
        ),
        "spatial_unmapped_gt_mask_frames": int(
            sum(int(row.get("spatial_unmapped_gt_mask_frames", 0)) for row in valid)
        ),
        "warning_count": int(sum(len(item.get("score_warnings", [])) for item in valid)),
    }


def coverage_summary(expected_query_ids: list[str], rows: list[dict]) -> dict:
    """Describe whether a report covers every query selected by its protocol."""
    id_to_row = {str(row.get("query_slug", "")): row for row in rows}
    missing = [
        query_id
        for query_id in expected_query_ids
        if id_to_row.get(query_id, {}).get("Acc") is None
    ]
    spatial_incomplete = [
        query_id
        for query_id in expected_query_ids
        if id_to_row.get(query_id, {}).get("Acc") is not None
        and not bool(id_to_row.get(query_id, {}).get("spatial_coverage_complete", True))
    ]
    return {
        "expected_queries": int(len(expected_query_ids)),
        "valid_queries": int(len(expected_query_ids) - len(missing)),
        "missing_query_ids": missing,
        "spatial_incomplete_query_ids": spatial_incomplete,
        "complete": not missing and not spatial_incomplete,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ReferGaussian query outputs on public 4DLangSplat-style protocols.")
    parser.add_argument("--protocol-json", required=True)
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--query-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument(
        "--category",
        default=None,
        help="Optional protocol category to evaluate, e.g. temporal_state_reference.",
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Optional scene basename to evaluate from a multi-scene protocol.",
    )
    parser.add_argument(
        "--query-manifest",
        default=None,
        help="Optional JSONL manifest. Only protocol queries listed by query_id are evaluated.",
    )
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless every protocol/manifest query has a valid evaluation result.",
    )
    args = parser.parse_args()

    protocol_payload = _read_json(Path(args.protocol_json))
    metadata_payload = _read_json(Path(args.dataset_dir) / "metadata.json")
    _, gt_masks_by_object, top_level_objects = _build_gt_masks(Path(args.annotation_dir))

    manifest_query_ids = (
        _read_manifest_query_ids(Path(args.query_manifest))
        if args.query_manifest
        else None
    )

    protocol_items = list(protocol_payload.get("queries", []))
    protocol_ids = {str(query_item.get("query_slug", "")) for query_item in protocol_items}
    if manifest_query_ids is not None:
        unknown_manifest_ids = sorted(manifest_query_ids - protocol_ids)
        if unknown_manifest_ids:
            raise ValueError(
                "Query manifest contains ids absent from the public protocol: "
                + ", ".join(unknown_manifest_ids[:10])
            )

    selected_items = [
        query_item
        for query_item in protocol_items
        if (args.category is None or str(query_item.get("category", "")) == args.category)
        and (args.scene is None or str(query_item.get("scene", "")).rsplit("/", 1)[-1] == args.scene)
        and (manifest_query_ids is None or str(query_item["query_slug"]) in manifest_query_ids)
    ]
    if not selected_items:
        raise ValueError("No public protocol queries matched the requested category/scene/manifest")
    expected_query_ids = [str(query_item["query_slug"]) for query_item in selected_items]

    per_query = []
    for query_item in selected_items:
        query_slug = str(query_item["query_slug"])
        validation_path = Path(args.query_root) / query_slug / "final_query_render_sourcebg" / "validation.json"
        if not validation_path.exists():
            if args.skip_missing:
                per_query.append(
                    {
                        "query_slug": query_slug,
                        "query": str(query_item.get("query", "")),
                        "status": "missing_validation",
                        "Acc": None,
                        "vIoU": None,
                        "temporal_tIoU": None,
                        "temporal_precision": None,
                        "temporal_recall": None,
                        "score_warnings": ["missing_validation"],
                        "validation_path": str(validation_path),
                    }
                )
                continue
            raise FileNotFoundError(f"Missing validation for {query_slug}: {validation_path}")
        validation_payload = _read_json(validation_path)
        validation_payload["_validation_path"] = str(validation_path)
        result = evaluate_query(
            query_item=query_item,
            validation_payload=validation_payload,
            metadata_payload=metadata_payload,
            gt_masks_by_object=gt_masks_by_object,
            top_level_objects=top_level_objects,
        )
        result["status"] = "ok"
        per_query.append(result)

    valid = [item for item in per_query if item.get("Acc") is not None]
    summary = summarize_query_results(valid, len(per_query))
    coverage = coverage_summary(expected_query_ids, per_query)
    payload = {
        "metric_protocol": METRIC_PROTOCOL,
        "protocol_json": str(Path(args.protocol_json)),
        "annotation_dir": str(Path(args.annotation_dir)),
        "dataset_dir": str(Path(args.dataset_dir)),
        "query_root": str(Path(args.query_root)),
        "query_manifest": str(Path(args.query_manifest)) if args.query_manifest else None,
        "source_files": {
            "protocol": _file_identity(Path(args.protocol_json)),
            "video_annotations": _file_identity(
                Path(args.annotation_dir) / "video_annotations.json"
            ),
            "coco_annotations": _file_identity(
                Path(args.annotation_dir) / "train" / "_annotations.coco.json"
            ),
            "query_manifest": (
                _file_identity(Path(args.query_manifest)) if args.query_manifest else None
            ),
            "dataset_metadata": _file_identity(Path(args.dataset_dir) / "metadata.json"),
        },
        "category": args.category,
        "scene": args.scene,
        "summary": summary,
        "coverage": coverage,
        "queries": per_query,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    if args.output_md:
        def summary_pct(value: float | None) -> str:
            return f"{value * 100.0:.2f}" if value is not None else "n/a"

        lines = [
            "# Public Query Benchmark",
            "",
            f"- Metric protocol: `{METRIC_PROTOCOL['id']}`",
            "- Legacy `Acc` is temporal-frame accuracy; legacy `vIoU` is mean annotated-frame mask IoU.",
            "- Paper exact-set Acc and exhaustive full-volume vIoU are not inferred from unavailable identities/masks.",
            f"- Category: `{args.category or 'all'}`",
            f"- Scene: `{args.scene or 'all'}`",
            f"- Queries: `{summary['query_count']}`",
            f"- Valid queries: `{summary['valid_queries']}`",
            f"- Complete coverage: `{coverage['complete']}` ({coverage['valid_queries']} / {coverage['expected_queries']})",
            f"- Acc(%): `{summary_pct(summary['Acc'])}`",
            f"- vIoU(%): `{summary_pct(summary['vIoU'])}`",
            f"- annotated volume IoU(%): `{summary_pct(summary['annotated_volume_iou'])}`",
            f"- temporal tIoU(%): `{summary_pct(summary['temporal_tIoU'])}`",
            f"- temporal precision/recall(%): "
            f"`{summary_pct(summary['temporal_precision'])}` / "
            f"`{summary_pct(summary['temporal_recall'])}`",
            f"- non-empty queries: `{summary['nonempty_queries']}`",
            f"- non-empty Acc/vIoU/tIoU(%): "
            f"`{summary_pct(summary['nonempty_only']['Acc'])}` / "
            f"`{summary_pct(summary['nonempty_only']['vIoU'])}` / "
            f"`{summary_pct(summary['nonempty_only']['temporal_tIoU'])}`",
            f"- zero-target correctness: `{summary['zero_target_correct']} / {summary['zero_target_queries']}`",
            f"- spatial frames: `{summary['spatial_matched_render_frames']} / {summary['spatial_gt_mask_frames']}` matched "
            f"(missing render/mask/unmapped: `{summary['spatial_missing_render_frames']}` / "
            f"`{summary['spatial_missing_binary_mask_frames']}` / `{summary['spatial_unmapped_gt_mask_frames']}`)",
            f"- warnings: `{summary['warning_count']}`",
            "",
            "| Query | Acc(%) | mean-frame IoU(%) | annotated-volume IoU(%) | tIoU(%) | tPrec(%) | tRec(%) | Target | Warnings | Pred Segments | GT Segments |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
        ]
        for row in per_query:
            def fmt(value):
                return f"{value * 100.0:.2f}" if value is not None else "n/a"

            lines.append(
                f"| {row['query_slug']} | {fmt(row.get('Acc'))} | {fmt(row.get('vIoU'))} | {fmt(row.get('annotated_volume_iou'))} | {fmt(row.get('temporal_tIoU'))} | {fmt(row.get('temporal_precision'))} | {fmt(row.get('temporal_recall'))} | {row.get('target_object', '')} | {','.join(row.get('score_warnings', []))} | {row.get('predicted_time_segments', [])} | {row.get('gt_time_segments', [])} |"
            )
        Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.require_complete and not coverage["complete"]:
        raise SystemExit(
            "Incomplete public evaluation: "
            f"{coverage['valid_queries']} / {coverage['expected_queries']} valid. "
            f"Missing queries: {', '.join(coverage['missing_query_ids'][:10]) or 'none'}; "
            f"spatially incomplete: {', '.join(coverage['spatial_incomplete_query_ids'][:10]) or 'none'}"
        )


if __name__ == "__main__":
    main()
