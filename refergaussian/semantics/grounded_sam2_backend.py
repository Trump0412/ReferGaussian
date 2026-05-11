from __future__ import annotations

import json
import os
import shutil
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_GSAM2_ROOT = _PROJECT_ROOT / "external" / "Grounded-SAM-2"
for _candidate in (_PROJECT_ROOT, _GSAM2_ROOT):
    candidate_str = str(_candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from utils.track_utils import sample_points_from_masks

from .source_images import resolve_dataset_image_entries


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _materialize_jpeg_frame_dir(image_entries: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in image_entries:
        source_path = Path(entry["image_path"])
        target_path = output_dir / f"{int(entry['frame_index']):05d}.jpg"
        if target_path.exists() or target_path.is_symlink():
            continue
        os.symlink(source_path, target_path)
    return output_dir


def _materialize_local_jpeg_frame_dir(
    image_entries: list[dict[str, Any]],
    output_dir: Path,
    start_frame_index: int,
    end_frame_index: int,
) -> tuple[Path, list[dict[str, Any]]]:
    local_entries = [
        entry
        for entry in image_entries
        if int(start_frame_index) <= int(entry["frame_index"]) <= int(end_frame_index)
    ]
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for local_index, entry in enumerate(local_entries):
        source_path = Path(entry["image_path"])
        target_path = output_dir / f"{int(local_index):05d}.jpg"
        os.symlink(source_path, target_path)
    return output_dir, local_entries


def _sample_search_entries(image_entries: list[dict[str, Any]], frame_stride: int, max_frames: int) -> list[dict[str, Any]]:
    sampled = image_entries[:: max(int(frame_stride), 1)]
    if max_frames > 0 and len(sampled) > int(max_frames):
        indices = np.linspace(0, len(sampled) - 1, num=int(max_frames), dtype=np.int32)
        sampled = [sampled[int(index)] for index in indices.tolist()]
    return sampled


def _normalize_query_text(text: str) -> str:
    normalized = " ".join(str(text).strip().lower().split())
    if not normalized:
        raise ValueError("detector phrase must be non-empty")
    if not normalized.endswith("."):
        normalized = f"{normalized}."
    return normalized


def _draw_box_preview(image: Image.Image, box: list[float], label: str, score: float) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    left, top, right, bottom = [float(v) for v in box]
    top = max(0.0, top)  # clamp top to canvas boundary
    color = (255, 96, 32)
    draw.rectangle((left, top, right, bottom), fill=color + (56,), outline=color + (255,), width=5)
    caption = f"{label} {score:.2f}"
    label_y0 = max(0.0, top - 26.0)
    label_y1 = max(label_y0 + 1.0, top)  # ensure y1 > y0
    draw.rectangle((left, label_y0, min(canvas.width - 1.0, left + 220.0), label_y1), fill=(18, 18, 18, 220))
    draw.text((left + 4.0, max(0.0, top - 20.0)), caption, fill=(240, 240, 240, 255))
    return canvas


def _mask_component_bboxes(mask: np.ndarray, min_area: int = 128, min_area_ratio: float = 0.12) -> list[list[int]]:
    try:
        from scipy import ndimage  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for GSAM2 mask component analysis.") from exc
    binary = np.asarray(mask > 0, dtype=np.uint8)
    if binary.max() <= 0:
        return []
    labels, num = ndimage.label(binary)
    min_keep_area = max(int(min_area), int(binary.sum() * float(min_area_ratio)))
    boxes: list[list[int]] = []
    for component_id in range(1, int(num) + 1):
        ys, xs = np.where(labels == component_id)
        if ys.size == 0 or xs.size == 0:
            continue
        if ys.size < min_keep_area:
            continue
        boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
    return boxes


def _draw_mask_preview(image: Image.Image, mask: np.ndarray, label: str, score: float) -> Image.Image:
    canvas = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    binary = np.asarray(mask > 0, dtype=bool)
    color = np.asarray([255, 96, 32], dtype=np.uint8)
    if binary.any():
        canvas[binary] = (0.55 * canvas[binary] + 0.45 * color).astype(np.uint8)
    preview = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(preview, "RGBA")
    for left, top, right, bottom in _mask_component_bboxes(mask, min_area=64, min_area_ratio=0.05):
        draw.rectangle((left, top, right, bottom), outline=(255, 96, 32, 255), width=4)
    bbox = _mask_bbox(mask)
    if bbox is not None:
        left, top, right, bottom = bbox
        draw.rectangle((left, top, right, bottom), outline=(255, 224, 96, 255), width=2)
        draw.rectangle((left, max(0, top - 26), min(preview.width - 1, left + 280), top), fill=(18, 18, 18, 220))
        draw.text((left + 4, max(0, top - 20)), f"{label} {score:.2f}", fill=(240, 240, 240, 255))
    return preview


def _stable_split_frames(frame_rows: list[dict[str, Any]], min_components: int = 2, min_stable: int = 2) -> list[int]:
    active_rows = [
        row
        for row in sorted(frame_rows, key=lambda item: int(item["frame_index"]))
        if int(row.get("component_count", 0)) >= int(min_components)
    ]
    if not active_rows:
        return []
    stable: list[int] = []
    run: list[int] = []
    prev = None
    for row in active_rows:
        frame_index = int(row["frame_index"])
        if prev is None or frame_index == prev + 1:
            run.append(frame_index)
        else:
            if len(run) >= int(min_stable):
                stable.append(run[0])
            run = [frame_index]
        prev = frame_index
    if len(run) >= int(min_stable):
        stable.append(run[0])
    return stable


def _run_phrase_detection(
    phrase: str,
    sampled_entries: list[dict[str, Any]],
    processor: Any,
    grounding_model: Any,
    device: str,
    box_threshold: float,
    text_threshold: float,
    top_k: int,
) -> list[dict[str, Any]]:
    query_text = _normalize_query_text(phrase)
    detections: list[dict[str, Any]] = []
    for entry in sampled_entries:
        image = Image.open(entry["image_path"]).convert("RGB")
        inputs = processor(images=image, text=query_text, return_tensors="pt")
        moved_inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.no_grad():
            outputs = grounding_model(**moved_inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            moved_inputs["input_ids"],
            threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            target_sizes=[image.size[::-1]],
        )[0]
        for detection_index, (box_tensor, score_tensor, label) in enumerate(
            zip(results.get("boxes", []), results.get("scores", []), results.get("labels", []))
        ):
            if detection_index >= int(top_k):
                break
            box = [float(value) for value in box_tensor.detach().cpu().tolist()]
            detections.append(
                {
                    "phrase": phrase,
                    "query_text": query_text,
                    "frame_index": int(entry["frame_index"]),
                    "image_id": str(entry["image_id"]),
                    "image_path": str(entry["image_path"]),
                    "time_value": float(entry["time_value"]),
                    "label": str(label),
                    "score": float(score_tensor.detach().cpu().item()),
                    "bbox_xyxy": box,
                }
            )
    detections.sort(key=lambda item: item["score"], reverse=True)
    return detections


def _bbox_center(box: list[float]) -> np.ndarray:
    left, top, right, bottom = [float(value) for value in box]
    return np.asarray([0.5 * (left + right), 0.5 * (top + bottom)], dtype=np.float32)


def _bbox_diag(box: list[float]) -> float:
    left, top, right, bottom = [float(value) for value in box]
    width = max(right - left, 1.0)
    height = max(bottom - top, 1.0)
    return float(np.hypot(width, height))


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(value) for value in box_a]
    bx0, by0, bx1, by1 = [float(value) for value in box_b]
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(inter_x1 - inter_x0, 0.0)
    inter_h = max(inter_y1 - inter_y0, 0.0)
    inter_area = inter_w * inter_h
    area_a = max(ax1 - ax0, 1.0) * max(ay1 - ay0, 1.0)
    area_b = max(bx1 - bx0, 1.0) * max(by1 - by0, 1.0)
    union = max(area_a + area_b - inter_area, 1.0e-6)
    return float(inter_area / union)


def _bbox_proximity(box_a: list[float], box_b: list[float]) -> float:
    center_a = _bbox_center(box_a)
    center_b = _bbox_center(box_b)
    distance = float(np.linalg.norm(center_a - center_b))
    scale = max(0.5 * (_bbox_diag(box_a) + _bbox_diag(box_b)), 1.0)
    return float(np.exp(-0.5 * (distance / scale) ** 2))


def _combo_box_cohesion(combo: tuple[dict[str, Any], ...]) -> float:
    if len(combo) <= 1:
        return 0.0
    values: list[float] = []
    for first_index in range(len(combo)):
        for second_index in range(first_index + 1, len(combo)):
            box_a = combo[first_index]["bbox_xyxy"]
            box_b = combo[second_index]["bbox_xyxy"]
            iou = _bbox_iou(box_a, box_b)
            proximity = _bbox_proximity(box_a, box_b)
            values.append(0.45 * iou + 0.55 * proximity)
    return float(np.mean(values)) if values else 0.0


def _select_anchor_detections(
    detections_by_phrase: dict[str, list[dict[str, Any]]],
    must_track_phrases: list[str],
    total_frames: int,
    top_n: int = 3,
    successor_phrases: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    required = [phrase for phrase in must_track_phrases if detections_by_phrase.get(phrase)]
    successor_set = {str(item).strip() for item in successor_phrases or [] if str(item).strip()}
    if not required:
        required = [phrase for phrase, detections in detections_by_phrase.items() if detections]
    if len(required) <= 1:
        for phrase, detections in detections_by_phrase.items():
            if detections:
                selected[phrase] = detections[0]
        return selected

    frame_norm = max(int(total_frames) - 1, 1)
    best_combo: tuple[float, tuple[dict[str, Any], ...]] | None = None

    best_per_frame: dict[str, dict[int, dict[str, Any]]] = {}
    for phrase in required:
        frame_rows: dict[int, dict[str, Any]] = {}
        for detection in detections_by_phrase[phrase]:
            frame_index = int(detection["frame_index"])
            if frame_index not in frame_rows or float(detection["score"]) > float(frame_rows[frame_index]["score"]):
                frame_rows[frame_index] = detection
        best_per_frame[phrase] = frame_rows

    common_frames = sorted(set.intersection(*(set(rows.keys()) for rows in best_per_frame.values()))) if best_per_frame else []
    for frame_index in common_frames:
        combo = tuple(best_per_frame[phrase][frame_index] for phrase in required)
        mean_score = float(np.mean([float(item["score"]) for item in combo]))
        cohesion = _combo_box_cohesion(combo)
        combo_score = mean_score + 0.38 * cohesion
        if best_combo is None or combo_score > best_combo[0]:
            best_combo = (combo_score, combo)

    if best_combo is None:
        candidate_lists = [detections_by_phrase[phrase][: max(1, int(top_n))] for phrase in required]
        for combo in product(*candidate_lists):
            frames = [int(item["frame_index"]) for item in combo]
            scores = [float(item["score"]) for item in combo]
            frame_span = (max(frames) - min(frames)) / frame_norm
            cohesion = _combo_box_cohesion(combo)
            combo_score = float(np.mean(scores) - 0.35 * frame_span + 0.30 * cohesion)
            if best_combo is None or combo_score > best_combo[0]:
                best_combo = (combo_score, combo)

    if best_combo is not None:
        _, combo = best_combo
        for phrase, detection in zip(required, combo):
            selected[phrase] = detection

    if selected:
        anchor_center = float(np.mean([int(item["frame_index"]) for item in selected.values()]))
    else:
        anchor_center = 0.0
    for phrase, detections in detections_by_phrase.items():
        if not detections:
            continue
        if phrase in selected:
            continue
        candidate_rows = detections[: max(1, int(top_n) * 4)]
        if phrase in successor_set:
            later_rows = [item for item in candidate_rows if float(item["frame_index"]) >= anchor_center]
            if later_rows:
                candidate_rows = later_rows
        selected[phrase] = max(
            candidate_rows,
            key=lambda item: float(item["score"])
            - 0.18 * abs(float(item["frame_index"]) - anchor_center) / frame_norm
            + (0.08 if phrase in successor_set and float(item["frame_index"]) >= anchor_center else 0.0),
        )
    return selected


def _select_multi_anchor_detections(
    detections_by_phrase: dict[str, list[dict[str, Any]]],
    detector_phrases: list[str],
    query_plan: dict[str, Any],
    total_frames: int,
    max_anchors_per_phrase: int = 3,
    top_n_per_anchor: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    context_frames = sorted(
        int(frame.get("frame_index", 0))
        for frame in query_plan.get("context_frames", [])
        if isinstance(frame, dict)
    )
    if not context_frames:
        context_frames = [0, max(int(total_frames) // 2, 0), max(int(total_frames) - 1, 0)]
    frame_norm = max(int(total_frames) - 1, 1)
    anchor_map: dict[str, list[dict[str, Any]]] = {}
    for phrase in detector_phrases:
        detections = detections_by_phrase.get(phrase, [])
        if not detections:
            anchor_map[phrase] = []
            continue
        chosen: list[dict[str, Any]] = []
        used_frames: list[int] = []
        target_count = max(1, int(max_anchors_per_phrase))
        if len(context_frames) <= target_count:
            target_frames = list(context_frames)
        else:
            target_indices = np.linspace(0, len(context_frames) - 1, num=target_count, dtype=np.int32)
            target_frames = [context_frames[int(index)] for index in target_indices.tolist()]
        min_sep = max(8, int(total_frames // max(6, int(max_anchors_per_phrase) * 2)))
        best_per_frame: dict[int, dict[str, Any]] = {}
        for row in detections:
            frame_index = int(row["frame_index"])
            current = best_per_frame.get(frame_index)
            if current is None or float(row["score"]) > float(current["score"]):
                best_per_frame[frame_index] = row
        per_frame_rows = list(best_per_frame.values())
        for target_frame in target_frames:
            candidates = per_frame_rows or detections
            ranked = sorted(
                candidates,
                key=lambda item: (
                    abs(int(item["frame_index"]) - int(target_frame)) / frame_norm
                    - 0.55 * float(item["score"])
                ),
            )
            picked = None
            for row in ranked:
                frame_index = int(row["frame_index"])
                if any(abs(frame_index - used) < min_sep for used in used_frames):
                    continue
                picked = row
                break
            if picked is None:
                continue
            chosen.append(picked)
            used_frames.append(int(picked["frame_index"]))
            if len(chosen) >= int(max_anchors_per_phrase):
                break
        if not chosen:
            chosen = [detections[0]]
        anchor_map[phrase] = sorted(chosen, key=lambda item: int(item["frame_index"]))
    return anchor_map


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def run_grounded_sam2_query(
    dataset_dir: str | Path,
    query_plan_path: str | Path,
    output_dir: str | Path,
    grounding_model_id: str = "IDEA-Research/grounding-dino-base",
    sam2_model_id: str = "facebook/sam2-hiera-large",
    detector_frame_stride: int = 12,
    max_detector_frames: int = 12,
    detection_top_k: int = 3,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    prompt_type: str = "point",
    num_point_prompts: int = 16,
    track_window_radius: int = 120,
    frame_subsample_stride: int = 10,
    num_anchor_seeds: int = 3,
) -> Path:
    dataset_dir = Path(dataset_dir)
    query_plan_path = Path(query_plan_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phrase_dir = output_dir / "phrases"
    phrase_dir.mkdir(parents=True, exist_ok=True)

    query_plan = _read_json(query_plan_path)
    detector_phrases = [str(item).strip() for item in query_plan.get("detector_phrases", []) if str(item).strip()]
    must_track_phrases = [str(item).strip() for item in query_plan.get("must_track_phrases", []) if str(item).strip()]
    successor_phrases = [str(item).strip() for item in query_plan.get("query_successor_phrases", []) if str(item).strip()]
    if not detector_phrases:
        raise ValueError(f"No detector_phrases found in {query_plan_path}")

    image_entries = resolve_dataset_image_entries(dataset_dir)
    image_entries = image_entries[:: max(int(frame_subsample_stride), 1)]
    if not image_entries:
        raise ValueError(f"No image entries available after frame subsampling under {dataset_dir}")
    sampled_entries = _sample_search_entries(
        image_entries,
        frame_stride=detector_frame_stride,
        max_frames=max_detector_frames,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(grounding_model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(device)
    grounding_model.eval()
    image_predictor = SAM2ImagePredictor.from_pretrained(sam2_model_id)
    video_predictor = SAM2VideoPredictor.from_pretrained(sam2_model_id)

    detections_by_phrase: dict[str, list[dict[str, Any]]] = {}
    for phrase in detector_phrases:
        detections_by_phrase[phrase] = _run_phrase_detection(
            phrase=phrase,
            sampled_entries=sampled_entries,
            processor=processor,
            grounding_model=grounding_model,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            top_k=detection_top_k,
        )
    selected_anchor_by_phrase = _select_anchor_detections(
        detections_by_phrase=detections_by_phrase,
        must_track_phrases=must_track_phrases,
        total_frames=len(image_entries),
        top_n=min(4, max(1, int(detection_top_k))),
        successor_phrases=successor_phrases,
    )
    multi_anchor_by_phrase = _select_multi_anchor_detections(
        detections_by_phrase=detections_by_phrase,
        detector_phrases=detector_phrases,
        query_plan=query_plan,
        total_frames=len(image_entries),
        max_anchors_per_phrase=max(1, int(num_anchor_seeds)),
        top_n_per_anchor=max(2, int(detection_top_k)),
    )
    max_frame_index = max(int(entry["frame_index"]) for entry in image_entries)

    phrase_payloads: list[dict[str, Any]] = []
    phrase_tracks: list[dict[str, Any]] = []
    for phrase_index, phrase in enumerate(detector_phrases):
        detections = detections_by_phrase.get(phrase, [])
        phrase_output_dir = phrase_dir / f"{phrase_index:02d}_{phrase.replace(' ', '_')}"
        phrase_output_dir.mkdir(parents=True, exist_ok=True)
        if not detections:
            phrase_payloads.append(
                {
                    "phrase": phrase,
                    "object_id": int(phrase_index + 1),
                    "detections": [],
                    "status": "no_detection",
                }
            )
            phrase_tracks.append(
                {
                    "phrase": phrase,
                    "object_id": int(phrase_index + 1),
                    "status": "no_detection",
                    "anchor_frame_index": None,
                    "frames": [
                        {
                            "frame_index": int(entry["frame_index"]),
                            "image_id": str(entry["image_id"]),
                            "time_value": float(entry["time_value"]),
                            "active": False,
                            "bbox_xyxy": None,
                        }
                        for entry in image_entries
                    ],
                }
            )
            continue

        selected_anchors = multi_anchor_by_phrase.get(phrase) or [selected_anchor_by_phrase.get(phrase, detections[0])]
        best = selected_anchor_by_phrase.get(phrase, selected_anchors[0])
        object_id = int(phrase_index + 1)
        effective_track_window_radius = max(
            int(track_window_radius),
            int(np.ceil(float(len(image_entries)) / max(len(selected_anchors), 1))),
        )
        anchors_dir = phrase_output_dir / "anchors"
        anchors_dir.mkdir(parents=True, exist_ok=True)
        combined_segments: dict[int, np.ndarray] = {}
        anchor_rows: list[dict[str, Any]] = []
        first_mask_bbox = None
        first_preview_path = None
        first_anchor_mask_preview_path = None
        for anchor_index, anchor_detection in enumerate(selected_anchors):
            anchor_frame_index = int(anchor_detection["frame_index"])
            anchor_image = Image.open(anchor_detection["image_path"]).convert("RGB")
            preview = _draw_box_preview(anchor_image, anchor_detection["bbox_xyxy"], phrase, anchor_detection["score"])
            preview_path = anchors_dir / f"anchor_{anchor_index:02d}_detection.png"
            preview.save(preview_path)

            image_predictor.set_image(np.array(anchor_image.convert("RGB")))
            box = np.asarray([anchor_detection["bbox_xyxy"]], dtype=np.float32)
            masks, _, _ = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False,
            )
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            mask = masks[0].astype(np.uint8)
            mask_bbox = _mask_bbox(mask)
            anchor_mask_preview = _draw_mask_preview(anchor_image, mask, phrase, anchor_detection["score"])
            anchor_mask_preview_path = anchors_dir / f"anchor_{anchor_index:02d}_mask.png"
            anchor_mask_preview.save(anchor_mask_preview_path)
            if first_mask_bbox is None:
                first_mask_bbox = mask_bbox
                first_preview_path = preview_path
                first_anchor_mask_preview_path = anchor_mask_preview_path

            track_start = max(0, anchor_frame_index - int(effective_track_window_radius))
            track_end = min(max_frame_index, anchor_frame_index + int(effective_track_window_radius))
            local_frame_dir, local_entries = _materialize_local_jpeg_frame_dir(
                image_entries=image_entries,
                output_dir=phrase_output_dir / f"video_window_jpg_{anchor_index:02d}",
                start_frame_index=track_start,
                end_frame_index=track_end,
            )
            local_anchor_candidates = [
                index
                for index, entry in enumerate(local_entries)
                if int(entry["frame_index"]) == int(anchor_frame_index)
            ]
            if not local_anchor_candidates:
                raise ValueError(f"Unable to find anchor frame {anchor_frame_index} inside the local window for phrase '{phrase}'.")
            local_anchor_index = int(local_anchor_candidates[0])
            inference_state = video_predictor.init_state(
                video_path=str(local_frame_dir),
                offload_video_to_cpu=True,
                async_loading_frames=True,
            )
            if prompt_type == "mask":
                video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=local_anchor_index,
                    obj_id=object_id,
                    mask=mask,
                )
            elif prompt_type == "box":
                video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=local_anchor_index,
                    obj_id=object_id,
                    box=np.asarray(anchor_detection["bbox_xyxy"], dtype=np.float32),
                )
            else:
                points = sample_points_from_masks(masks=masks.astype(np.uint8), num_points=int(num_point_prompts))[0]
                labels = np.ones((points.shape[0],), dtype=np.int32)
                video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=local_anchor_index,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                )

            local_video_segments: dict[int, np.ndarray] = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
                for mask_index, out_obj_id in enumerate(out_obj_ids):
                    if int(out_obj_id) != object_id:
                        continue
                    local_video_segments[int(out_frame_idx)] = (
                        out_mask_logits[mask_index] > 0.0
                    ).detach().cpu().numpy().astype(np.uint8)
            for local_frame_index, mask_array in local_video_segments.items():
                if local_frame_index < 0 or local_frame_index >= len(local_entries):
                    continue
                global_frame_index = int(local_entries[local_frame_index]["frame_index"])
                flat_mask = np.asarray(mask_array[0] if mask_array.ndim == 3 else mask_array, dtype=np.uint8)
                if global_frame_index in combined_segments:
                    combined_segments[global_frame_index] = np.maximum(combined_segments[global_frame_index], flat_mask)
                else:
                    combined_segments[global_frame_index] = flat_mask
            anchor_rows.append(
                {
                    "anchor_index": int(anchor_index),
                    "anchor_frame_index": int(anchor_frame_index),
                    "anchor_image_id": str(anchor_detection["image_id"]),
                    "anchor_time_value": float(anchor_detection["time_value"]),
                    "anchor_bbox_xyxy": anchor_detection["bbox_xyxy"],
                    "anchor_score": float(anchor_detection["score"]),
                    "anchor_label": str(anchor_detection["label"]),
                    "anchor_preview_path": str(preview_path),
                    "anchor_mask_preview_path": str(anchor_mask_preview_path),
                    "anchor_mask_bbox_xyxy": mask_bbox,
                "track_window": {
                        "start_frame_index": int(track_start),
                        "end_frame_index": int(track_end),
                    },
                }
            )

        bbox_by_global_frame: dict[int, list[int] | None] = {}
        track_mask_dir = phrase_output_dir / "track_masks"
        track_overlay_dir = phrase_output_dir / "track_overlays"
        track_mask_dir.mkdir(parents=True, exist_ok=True)
        track_overlay_dir.mkdir(parents=True, exist_ok=True)
        frame_rows: list[dict[str, Any]] = []
        for global_frame_index in sorted(combined_segments.keys()):
            matching_entries = [entry for entry in image_entries if int(entry["frame_index"]) == int(global_frame_index)]
            if not matching_entries:
                continue
            local_entry = matching_entries[0]
            flat_mask = combined_segments[global_frame_index]
            bbox = _mask_bbox(flat_mask)
            bbox_by_global_frame[global_frame_index] = bbox
            mask_path = track_mask_dir / f"{global_frame_index:05d}.png"
            Image.fromarray((flat_mask > 0).astype(np.uint8) * 255, mode="L").save(mask_path)
            source_image = Image.open(local_entry["image_path"]).convert("RGB")
            overlay_preview = _draw_mask_preview(source_image, flat_mask, phrase, float(best["score"]))
            overlay_path = track_overlay_dir / f"{global_frame_index:05d}.png"
            overlay_preview.save(overlay_path)
            component_bboxes = _mask_component_bboxes(flat_mask, min_area=64, min_area_ratio=0.06)
            frame_rows.append(
                {
                    "frame_index": global_frame_index,
                    "image_id": str(local_entry["image_id"]),
                    "time_value": float(local_entry["time_value"]),
                    "active": True,
                    "bbox_xyxy": bbox,
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path),
                    "component_count": int(len(component_bboxes)),
                    "component_bboxes": component_bboxes,
                    "mask_area_px": int((flat_mask > 0).sum()),
                }
            )

        frame_rows_by_index = {int(row["frame_index"]): row for row in frame_rows}
        split_frames = _stable_split_frames(frame_rows)

        phrase_payloads.append(
            {
                "phrase": phrase,
                "object_id": object_id,
                "status": "seeded",
                "anchor_frame_index": int(best["frame_index"]),
                "anchor_image_id": best["image_id"],
                "anchor_time_value": float(best["time_value"]),
                "anchor_bbox_xyxy": best["bbox_xyxy"],
                "anchor_score": float(best["score"]),
                "anchor_label": str(best["label"]),
                "anchor_mask_bbox_xyxy": first_mask_bbox,
                "track_window": anchor_rows[0]["track_window"] if anchor_rows else None,
                "anchor_preview_path": None if first_preview_path is None else str(first_preview_path),
                "anchor_mask_preview_path": None if first_anchor_mask_preview_path is None else str(first_anchor_mask_preview_path),
                "anchors": anchor_rows,
                "track_mask_dir": str(track_mask_dir),
                "track_overlay_dir": str(track_overlay_dir),
                "split_frames": split_frames,
                "detections": detections[: min(len(detections), 12)],
            }
        )
        track_frames = []
        for entry in image_entries:
            frame_index = int(entry["frame_index"])
            row = frame_rows_by_index.get(frame_index)
            if row is None:
                track_frames.append(
                    {
                        "frame_index": int(frame_index),
                        "image_id": str(entry["image_id"]),
                        "time_value": float(entry["time_value"]),
                        "active": False,
                        "bbox_xyxy": None,
                        "mask_path": None,
                        "overlay_path": None,
                        "component_count": 0,
                        "component_bboxes": [],
                        "mask_area_px": 0,
                    }
                )
                continue
            track_frames.append(row)
        phrase_tracks.append(
            {
                "phrase": phrase,
                "object_id": object_id,
                "status": "seeded",
                "anchor_frame_index": anchor_frame_index,
                "split_frames": split_frames,
                "frames": track_frames,
            }
        )

    payload = {
        "schema_version": 1,
        "dataset_dir": str(dataset_dir),
        "query_plan_path": str(query_plan_path),
        "grounding_model_id": grounding_model_id,
        "sam2_model_id": sam2_model_id,
        "prompt_type": prompt_type,
        "frame_subsample_stride": int(frame_subsample_stride),
        "num_tracking_frames": int(len(image_entries)),
        "detector_frame_stride": int(detector_frame_stride),
        "max_detector_frames": int(max_detector_frames),
        "track_window_radius": int(track_window_radius),
        "num_anchor_seeds": int(num_anchor_seeds),
        "phrases": phrase_payloads,
        "tracks": phrase_tracks,
    }
    _write_json(output_dir / "grounded_sam2_query_tracks.json", payload)
    return output_dir
