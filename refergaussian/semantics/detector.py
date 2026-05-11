from __future__ import annotations

import json
import colorsys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from .bootstrap import _resolve_source_images


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _normalize_query_text(query: str) -> str:
    text = " ".join(str(query).strip().lower().split())
    if not text:
        raise ValueError("query must be non-empty")
    if not text.endswith("."):
        text = f"{text}."
    return text


def _sample_entries(
    image_entries: list[dict[str, Any]],
    frame_stride: int,
    max_frames: int,
) -> list[dict[str, Any]]:
    sampled = image_entries[:: max(int(frame_stride), 1)]
    if max_frames > 0 and len(sampled) > int(max_frames):
        indices = np.linspace(0, len(sampled) - 1, num=int(max_frames), dtype=np.int32)
        sampled = [sampled[int(index)] for index in indices.tolist()]
    return sampled


def _prompt_points_from_box(box: list[float]) -> list[list[float]]:
    left, top, right, bottom = [float(value) for value in box]
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (top + bottom)
    width = max(right - left, 1.0)
    height = max(bottom - top, 1.0)
    dx = 0.22 * width
    dy = 0.22 * height
    return [
        [center_x, center_y],
        [center_x - dx, center_y],
        [center_x + dx, center_y],
        [center_x, center_y - dy],
        [center_x, center_y + dy],
    ]


def _draw_detections(
    image: Image.Image,
    detections: list[dict[str, Any]],
    query_text: str,
) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((8, 8, min(canvas.width - 8, 8 + 18 * len(query_text)), 38), fill=(18, 18, 18, 210))
    draw.text((16, 14), f"query detector: {query_text}", fill=(240, 240, 240, 255))
    for index, detection in enumerate(detections):
        left, top, right, bottom = [float(v) for v in detection["bbox_xyxy"]]
        hue = float((index * 0.213 + 0.08) % 1.0)
        rgb = tuple(int(round(value * 255.0)) for value in colorsys.hsv_to_rgb(hue, 0.75, 1.0))
        draw.rectangle((left, top, right, bottom), fill=rgb + (52,), outline=rgb + (255,), width=4)
        center_x, center_y = detection["center_xy"]
        draw.line((center_x - 10, center_y, center_x + 10, center_y), fill=rgb + (255,), width=3)
        draw.line((center_x, center_y - 10, center_x, center_y + 10), fill=rgb + (255,), width=3)
        tag = f"{index}: {detection['label']} {detection['score']:.2f}"
        draw.rectangle((left, max(0.0, top - 24.0), min(canvas.width - 1.0, left + 180.0), top), fill=(18, 18, 18, 220))
        draw.text((left + 4.0, max(0.0, top - 20.0)), tag, fill=rgb + (255,))
    return canvas


def detect_query_proposals(
    dataset_dir: str | Path,
    query: str,
    output_dir: str | Path | None = None,
    model_id: str = "IDEA-Research/grounding-dino-base",
    frame_stride: int = 4,
    max_frames: int = 24,
    top_k: int = 3,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    device: str | None = None,
) -> Path:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir) if output_dir is not None else dataset_dir / "query_detector"
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "frames"
    debug_dir.mkdir(parents=True, exist_ok=True)

    image_entries = _resolve_source_images(dataset_dir)
    if not image_entries:
        raise FileNotFoundError(f"No source images found under {dataset_dir}")
    sampled_entries = _sample_entries(image_entries, frame_stride=frame_stride, max_frames=max_frames)
    query_text = _normalize_query_text(query)

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model = model.to(resolved_device)
    model.eval()

    frames_payload: list[dict[str, Any]] = []
    global_candidates: list[dict[str, Any]] = []
    for frame_index, entry in enumerate(sampled_entries):
        image_path = Path(entry["image_path"])
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=query_text, return_tensors="pt")
        moved_inputs = {}
        for key, value in inputs.items():
            moved_inputs[key] = value.to(resolved_device) if hasattr(value, "to") else value

        with torch.no_grad():
            outputs = model(**moved_inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            moved_inputs["input_ids"],
            box_threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            target_sizes=[image.size[::-1]],
        )[0]

        detections = []
        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        labels = results.get("labels", [])
        for detection_index, (box_tensor, score_tensor, label) in enumerate(zip(boxes, scores, labels)):
            if detection_index >= int(top_k):
                break
            box = [float(value) for value in box_tensor.detach().cpu().tolist()]
            left, top, right, bottom = box
            center_xy = [0.5 * (left + right), 0.5 * (top + bottom)]
            prompt_points = _prompt_points_from_box(box)
            detection = {
                "rank": int(detection_index),
                "label": str(label),
                "score": float(score_tensor.detach().cpu().item()),
                "bbox_xyxy": box,
                "center_xy": center_xy,
                "prompt_points_xy": prompt_points,
                "bbox_xyxy_norm": [
                    float(left / max(image.width, 1)),
                    float(top / max(image.height, 1)),
                    float(right / max(image.width, 1)),
                    float(bottom / max(image.height, 1)),
                ],
                "center_xy_norm": [
                    float(center_xy[0] / max(image.width, 1)),
                    float(center_xy[1] / max(image.height, 1)),
                ],
            }
            detections.append(detection)
            global_candidates.append(
                {
                    "frame_index": int(frame_index),
                    "image_id": entry["image_id"],
                    "time_value": float(entry["time_value"]),
                    **detection,
                }
            )

        debug_image = _draw_detections(image, detections, query_text=query)
        debug_path = debug_dir / f"{frame_index:04d}_{entry['image_id']}.png"
        debug_image.save(debug_path)
        frames_payload.append(
            {
                "frame_index": int(frame_index),
                "image_id": entry["image_id"],
                "image_path": str(image_path),
                "time_value": float(entry["time_value"]),
                "image_size": [int(image.width), int(image.height)],
                "debug_path": str(debug_path),
                "detections": detections,
            }
        )

    global_candidates.sort(key=lambda item: item["score"], reverse=True)
    payload = {
        "schema_version": 1,
        "query": query,
        "query_text_detector": query_text,
        "dataset_dir": str(dataset_dir),
        "model_id": model_id,
        "device": resolved_device,
        "frame_stride": int(frame_stride),
        "max_frames": int(max_frames),
        "top_k_per_frame": int(top_k),
        "box_threshold": float(box_threshold),
        "text_threshold": float(text_threshold),
        "num_frames_scored": int(len(frames_payload)),
        "num_candidates": int(len(global_candidates)),
        "frames": frames_payload,
        "top_candidates": global_candidates[: min(len(global_candidates), 64)],
        "prompt_points": [
            {
                "frame_index": item["frame_index"],
                "image_id": item["image_id"],
                "time_value": item["time_value"],
                "score": item["score"],
                "label": item["label"],
                "center_xy": item["center_xy"],
                "center_xy_norm": item["center_xy_norm"],
                "prompt_points_xy": item["prompt_points_xy"],
            }
            for item in global_candidates[: min(len(global_candidates), 32)]
        ],
    }
    _write_json(output_dir / "query_detection_proposals.json", payload)
    return output_dir
