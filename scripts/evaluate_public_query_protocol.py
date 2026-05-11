import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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
    return _safe_div(inter, union)


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
    union_count = 0
    overlap_count = 0
    gt_mask_frames = 0

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
            continue
        time_id = int(metadata_payload[image_id]["time_id"])
        row = _nearest_render_row(time_id=time_id, rendered_rows=rendered_rows, max_distance=max_distance)
        if row is None:
            continue
        gt_mask_frames += 1
        pred_active = bool(pred_full_mask[time_id])
        gt_active = bool(gt_full_mask[time_id])
        if pred_active or gt_active:
            union_count += 1
        if pred_active and gt_active:
            overlap_count += 1
            pred_mask_path = binary_mask_dir / f"{int(row['frame_index']):05d}.png"
            if pred_mask_path.exists():
                with Image.open(pred_mask_path) as image:
                    pred_mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
            else:
                pred_mask = np.zeros_like(gt_mask)
            iou_sum += _mask_iou(pred_mask, gt_mask)

    predicted_ranges = validation_payload.get("active_segments", [])
    predicted_time_ranges = _ranges_from_bool_mask(pred_full_mask)
    temporal_union = int(np.logical_or(pred_full_mask, gt_full_mask).sum())
    temporal_inter = int(np.logical_and(pred_full_mask, gt_full_mask).sum())
    temporal_iou = _safe_div(temporal_inter, temporal_union)

    return {
        "query_slug": query_slug,
        "query": query_text,
        "target_object": target_object,
        "frames_evaluated": int(len(frame_rows)),
        "timeline_frames_evaluated": int(total_frames),
        "gt_mask_frames": int(gt_mask_frames),
        "Acc": acc,
        "vIoU": _safe_div(iou_sum, union_count),
        "temporal_tIoU": temporal_iou,
        "mask_union_frames": int(union_count),
        "mask_overlap_frames": int(overlap_count),
        "predicted_render_frame_segments": predicted_ranges,
        "predicted_time_segments": predicted_time_ranges,
        "gt_time_segments": query_item["targets"][0]["target_ranges"],
        "validation_path": validation_payload.get("_validation_path", ""),
        "frame_rows": per_frame_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ReferGaussian query outputs on public 4DLangSplat-style protocols.")
    parser.add_argument("--protocol-json", required=True)
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--query-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    protocol_payload = _read_json(Path(args.protocol_json))
    metadata_payload = _read_json(Path(args.dataset_dir) / "metadata.json")
    _, gt_masks_by_object, top_level_objects = _build_gt_masks(Path(args.annotation_dir))

    per_query = []
    for query_item in protocol_payload.get("queries", []):
        query_slug = str(query_item["query_slug"])
        validation_path = Path(args.query_root) / query_slug / "final_query_render_sourcebg" / "validation.json"
        if not validation_path.exists():
            raise FileNotFoundError(f"Missing validation for {query_slug}: {validation_path}")
        validation_payload = _read_json(validation_path)
        validation_payload["_validation_path"] = str(validation_path)
        per_query.append(
            evaluate_query(
                query_item=query_item,
                validation_payload=validation_payload,
                metadata_payload=metadata_payload,
                gt_masks_by_object=gt_masks_by_object,
                top_level_objects=top_level_objects,
            )
        )

    summary = {
        "query_count": int(len(per_query)),
        "Acc": float(np.mean([item["Acc"] for item in per_query])) if per_query else 0.0,
        "vIoU": float(np.mean([item["vIoU"] for item in per_query])) if per_query else 0.0,
        "temporal_tIoU": float(np.mean([item["temporal_tIoU"] for item in per_query])) if per_query else 0.0,
    }
    payload = {
        "protocol_json": str(Path(args.protocol_json)),
        "annotation_dir": str(Path(args.annotation_dir)),
        "dataset_dir": str(Path(args.dataset_dir)),
        "query_root": str(Path(args.query_root)),
        "summary": summary,
        "queries": per_query,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    if args.output_md:
        lines = [
            "# Public Query Benchmark",
            "",
            f"- Queries: `{summary['query_count']}`",
            f"- Acc(%): `{summary['Acc'] * 100.0:.2f}`",
            f"- vIoU(%): `{summary['vIoU'] * 100.0:.2f}`",
            f"- temporal tIoU(%): `{summary['temporal_tIoU'] * 100.0:.2f}`",
            "",
            "| Query | Acc(%) | vIoU(%) | tIoU(%) | Target | Pred Segments | GT Segments |",
            "| --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
        for row in per_query:
            lines.append(
                f"| {row['query_slug']} | {row['Acc'] * 100.0:.2f} | {row['vIoU'] * 100.0:.2f} | {row['temporal_tIoU'] * 100.0:.2f} | {row['target_object']} | {row['predicted_time_segments']} | {row['gt_time_segments']} |"
            )
        Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
