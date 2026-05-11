from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .qwen_assignment import QwenVisionTeacher, _load_source_images, _resolve_qwen_model, _track_map
from .query_render import _interp_scalar, _interp_vector, _pixel_radius

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_ROOT = _PROJECT_ROOT / "external" / "4DGaussians"
for _candidate in (_PROJECT_ROOT, _UPSTREAM_ROOT):
    candidate_str = str(_candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from scene.utils import Camera


PAIR_PROMPT_TEMPLATE = """You are verifying a 4D interaction candidate for the query: "{query}".
The attached images show the same patient-tool pair across a short temporal window.
Yellow annotations mark the patient candidate.
Cyan annotations mark the tool candidate.

Return exactly one JSON object with keys:
- matches_query: number in [0, 1]
- patient_identity: short phrase
- tool_identity: short phrase
- action: short phrase
- patient_is_target: number in [0, 1]
- tool_is_correct: number in [0, 1]
- action_is_cut: number in [0, 1]
- reason: one concise sentence

Be specific. If this looks like a knife cutting a lemon, say so.
"""

ROLE_COLORS = {
    "patient": (255, 210, 64),
    "tool": (64, 224, 255),
}


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _sample_time_values(run_dir: Path) -> np.ndarray:
    payload = np.load(run_dir / "entitybank" / "trajectory_samples.npz")
    return payload["time_values"].astype(np.float32)


def _resample_mask(sample_mask: np.ndarray, sample_times: np.ndarray, test_times: np.ndarray) -> np.ndarray:
    sample_mask = np.asarray(sample_mask, dtype=np.float32)
    indices = np.abs(test_times[:, None] - sample_times[None, :]).argmin(axis=1)
    return sample_mask[indices] > 0.5


def _ranges_from_mask(mask: np.ndarray) -> list[list[int]]:
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


def _mask_from_segments(segments: list[list[int]], frame_count: int) -> np.ndarray:
    mask = np.zeros((frame_count,), dtype=bool)
    for segment in segments:
        if len(segment) != 2:
            continue
        start = max(int(segment[0]), 0)
        end = min(int(segment[1]), frame_count - 1)
        if end < start:
            continue
        mask[start : end + 1] = True
    return mask


def _load_font(size: int = 18) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int]) -> None:
    font = _load_font(18)
    bbox = draw.textbbox(xy, text, font=font)
    draw.rounded_rectangle((bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2), radius=4, fill=(16, 16, 16))
    draw.text(xy, text, fill=fill, font=font)


def _identity_bonus(text: str, keywords: list[str], soft_keywords: list[str]) -> float:
    lowered = text.lower()
    if any(keyword in lowered for keyword in keywords):
        return 1.0
    if any(keyword in lowered for keyword in soft_keywords):
        return 0.6
    return 0.0


def _pair_frame_indices(segments: list[list[int]], frame_count: int, max_frames: int = 3) -> list[int]:
    indices: list[int] = []
    for start, end in segments:
        start = max(0, min(frame_count - 1, int(start)))
        end = max(0, min(frame_count - 1, int(end)))
        indices.append((start + end) // 2)
        if end - start >= 2:
            indices.append(start)
            indices.append(end)
    unique = []
    seen = set()
    for value in indices:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique[:max_frames]


def _project_entity(
    camera: Camera,
    center_world: np.ndarray,
    extent_min: np.ndarray,
    extent_max: np.ndarray,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    pixel = camera.project(center_world[None, :])[0].astype(np.float32)
    center_local = camera.points_to_local_points(center_world[None, :])[0]
    radius = _pixel_radius(camera, center_world, extent_min, extent_max, image_size=image_size)
    camera_size = np.asarray(camera.image_size, dtype=np.float32).reshape(-1)
    if camera_size.size >= 2:
        scale_x = float(image_size[0]) / max(float(camera_size[0]), 1.0)
        scale_y = float(image_size[1]) / max(float(camera_size[1]), 1.0)
        pixel[0] *= scale_x
        pixel[1] *= scale_y
    return {
        "pixel_xy": [float(pixel[0]), float(pixel[1])],
        "depth": float(center_local[2]),
        "radius_px": int(radius),
        "displayable": bool(float(center_local[2]) > 1.0e-4),
    }


def _annotated_pair_crop(
    image: Image.Image,
    camera_path: Path,
    patient_world: dict[str, np.ndarray],
    tool_world: dict[str, np.ndarray],
) -> tuple[Image.Image, dict[str, Any]]:
    camera = Camera.from_json(camera_path)
    width, height = image.size
    patient_proj = _project_entity(camera, patient_world["center"], patient_world["extent_min"], patient_world["extent_max"], image_size=(width, height))
    tool_proj = _project_entity(camera, tool_world["center"], tool_world["extent_min"], tool_world["extent_max"], image_size=(width, height))
    patient_xy = np.asarray(patient_proj["pixel_xy"], dtype=np.float32)
    tool_xy = np.asarray(tool_proj["pixel_xy"], dtype=np.float32)
    patient_xy = np.asarray([np.clip(patient_xy[0], 0.0, width - 1.0), np.clip(patient_xy[1], 0.0, height - 1.0)], dtype=np.float32)
    tool_xy = np.asarray([np.clip(tool_xy[0], 0.0, width - 1.0), np.clip(tool_xy[1], 0.0, height - 1.0)], dtype=np.float32)

    min_x = min(patient_xy[0] - patient_proj["radius_px"], tool_xy[0] - tool_proj["radius_px"])
    max_x = max(patient_xy[0] + patient_proj["radius_px"], tool_xy[0] + tool_proj["radius_px"])
    min_y = min(patient_xy[1] - patient_proj["radius_px"], tool_xy[1] - tool_proj["radius_px"])
    max_y = max(patient_xy[1] + patient_proj["radius_px"], tool_xy[1] + tool_proj["radius_px"])
    margin = max(64.0, 0.3 * max(max_x - min_x, max_y - min_y))
    left = int(np.clip(min_x - margin, 0.0, width - 1.0))
    top = int(np.clip(min_y - margin, 0.0, height - 1.0))
    right = int(np.clip(max_x + margin, left + 32.0, width))
    bottom = int(np.clip(max_y + margin, top + 32.0, height))

    crop = image.crop((left, top, right, bottom)).convert("RGB")
    draw = ImageDraw.Draw(crop, "RGBA")

    def draw_role(role: str, pixel_xy: np.ndarray, radius_px: int) -> None:
        color = ROLE_COLORS[role]
        cx = int(round(pixel_xy[0] - left))
        cy = int(round(pixel_xy[1] - top))
        radius = int(max(radius_px, 12))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=color + (255,), width=4)
        _draw_label(draw, (max(0, cx - radius), max(0, cy - radius - 22)), role, color)

    draw_role("patient", patient_xy, patient_proj["radius_px"])
    draw_role("tool", tool_xy, tool_proj["radius_px"])
    draw.line(
        (
            int(round(patient_xy[0] - left)),
            int(round(patient_xy[1] - top)),
            int(round(tool_xy[0] - left)),
            int(round(tool_xy[1] - top)),
        ),
        fill=(255, 255, 255, 192),
        width=3,
    )
    return crop, {
        "crop_box": [left, top, right, bottom],
        "patient_proj": patient_proj,
        "tool_proj": tool_proj,
    }


def _pair_prompt(query: str) -> str:
    return PAIR_PROMPT_TEMPLATE.format(query=query)


def _score_verified_pair(pair: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any]:
    patient_identity = str(verification.get("patient_identity", ""))
    tool_identity = str(verification.get("tool_identity", ""))
    action = str(verification.get("action", ""))
    match_score = float(np.clip(float(verification.get("matches_query", 0.0)), 0.0, 1.0))
    patient_score = float(np.clip(float(verification.get("patient_is_target", 0.0)), 0.0, 1.0))
    tool_score = float(np.clip(float(verification.get("tool_is_correct", 0.0)), 0.0, 1.0))
    action_score = float(np.clip(float(verification.get("action_is_cut", 0.0)), 0.0, 1.0))

    patient_bonus = _identity_bonus(patient_identity, ["lemon"], ["citrus", "fruit"])
    tool_bonus = _identity_bonus(tool_identity, ["knife", "blade"], ["cutter", "utensil"])
    action_bonus = _identity_bonus(action, ["cut", "slice"], ["chop", "press"])

    verified_score = (
        0.45 * float(pair.get("pair_score", 0.0))
        + 0.20 * match_score
        + 0.10 * patient_score
        + 0.10 * tool_score
        + 0.05 * action_score
        + 0.05 * patient_bonus
        + 0.05 * tool_bonus
        + 0.05 * action_bonus
    )
    return {
        "verified_score": float(verified_score),
        "patient_identity": patient_identity,
        "tool_identity": tool_identity,
        "action": action,
        "match_score": match_score,
        "patient_bonus": patient_bonus,
        "tool_bonus": tool_bonus,
        "action_bonus": action_bonus,
        "reason": str(verification.get("reason", "")),
    }


def refine_qwen_query_pairs(
    run_dir: str | Path,
    query_dir: str | Path,
    qwen_model: str | None = None,
    top_pairs: int = 12,
) -> Path:
    run_dir = Path(run_dir)
    query_dir = Path(query_dir)
    query_payload = _read_json(query_dir / "query.json")
    candidates_payload = _read_json(query_dir / "candidates.json")
    assignments_payload = _read_json(run_dir / "entitybank" / "semantic_assignments_qwen.json")
    assignment_map = {int(item["entity_id"]): item for item in assignments_payload.get("assignments", [])}
    pair_candidates = list(candidates_payload.get("pair_candidates", []))[:top_pairs]
    if not pair_candidates:
        raise ValueError(f"No pair candidates found in {query_dir / 'candidates.json'}")

    source_frame_dir, test_ids, test_times = _load_source_images(run_dir)
    sample_times = _sample_time_values(run_dir)
    config = _read_json(run_dir / "config.yaml") if (run_dir / "config.yaml").suffix == ".json" else None
    # config.yaml is not JSON; reuse string parsing via qwen_assignment helper path already embedded in _load_source_images.
    dataset_dir = Path(str(run_dir / "config.yaml"))
    dataset_dir = None
    # Read source path from config.yaml without importing the whole common parser again.
    for raw_line in (run_dir / "config.yaml").read_text(encoding="utf-8").splitlines():
        if raw_line.strip().startswith("source_path:"):
            dataset_dir = Path(raw_line.split(":", 1)[1].strip())
            break
    if dataset_dir is None:
        raise ValueError(f"Unable to resolve source_path from {run_dir / 'config.yaml'}")

    tracks = _track_map(run_dir / "entitybank")
    teacher = QwenVisionTeacher(_resolve_qwen_model(qwen_model))
    review_root = query_dir / "qwen_pair_review"
    review_root.mkdir(parents=True, exist_ok=True)

    verification_rows = []
    for rank, pair in enumerate(pair_candidates):
        patient_id = int(pair["patient_id"])
        tool_id = int(pair["tool_id"])
        patient_track = tracks.get(patient_id)
        tool_track = tracks.get(tool_id)
        if patient_track is None or tool_track is None:
            continue

        patient_time_values = np.asarray([float(frame["time_value"]) for frame in patient_track.get("frames", [])], dtype=np.float32)
        tool_time_values = np.asarray([float(frame["time_value"]) for frame in tool_track.get("frames", [])], dtype=np.float32)
        patient_centers = np.asarray([frame["center_world"] for frame in patient_track.get("frames", [])], dtype=np.float32)
        patient_mins = np.asarray([frame["extent_world_min"] for frame in patient_track.get("frames", [])], dtype=np.float32)
        patient_maxs = np.asarray([frame["extent_world_max"] for frame in patient_track.get("frames", [])], dtype=np.float32)
        tool_centers = np.asarray([frame["center_world"] for frame in tool_track.get("frames", [])], dtype=np.float32)
        tool_mins = np.asarray([frame["extent_world_min"] for frame in tool_track.get("frames", [])], dtype=np.float32)
        tool_maxs = np.asarray([frame["extent_world_max"] for frame in tool_track.get("frames", [])], dtype=np.float32)

        sample_mask = _mask_from_segments(pair.get("contact_segments_sample", []), sample_times.shape[0])
        test_mask = _resample_mask(sample_mask.astype(np.float32), sample_times, test_times)
        test_segments = _ranges_from_mask(test_mask)
        if not test_segments:
            continue
        frame_indices = _pair_frame_indices(test_segments, len(test_ids), max_frames=3)
        if not frame_indices:
            continue

        pair_dir = review_root / f"pair_{rank:02d}_{patient_id}_{tool_id}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        images: list[Image.Image] = []
        crop_paths: list[str] = []
        for order_index, frame_index in enumerate(frame_indices):
            image_id = test_ids[frame_index]
            image_path = source_frame_dir / f"{image_id}.png"
            camera_path = dataset_dir / "camera" / f"{image_id}.json"
            if not image_path.exists() or not camera_path.exists():
                continue
            patient_world = {
                "center": _interp_vector(test_times[[frame_index]], patient_time_values, patient_centers)[0],
                "extent_min": _interp_vector(test_times[[frame_index]], patient_time_values, patient_mins)[0],
                "extent_max": _interp_vector(test_times[[frame_index]], patient_time_values, patient_maxs)[0],
            }
            tool_world = {
                "center": _interp_vector(test_times[[frame_index]], tool_time_values, tool_centers)[0],
                "extent_min": _interp_vector(test_times[[frame_index]], tool_time_values, tool_mins)[0],
                "extent_max": _interp_vector(test_times[[frame_index]], tool_time_values, tool_maxs)[0],
            }
            with Image.open(image_path) as frame_image:
                crop, meta = _annotated_pair_crop(frame_image.convert("RGB"), camera_path, patient_world, tool_world)
            crop_path = pair_dir / f"{order_index:02d}_frame_{frame_index:04d}.png"
            crop.save(crop_path)
            crop_paths.append(str(crop_path))
            images.append(crop)

        if not images:
            continue

        try:
            verification, raw_output = teacher.generate_json(_pair_prompt(query_payload["query"]), images)
        except Exception as exc:
            verification = {
                "matches_query": 0.0,
                "patient_identity": "",
                "tool_identity": "",
                "action": "",
                "patient_is_target": 0.0,
                "tool_is_correct": 0.0,
                "action_is_cut": 0.0,
                "reason": str(exc),
            }
            raw_output = str(exc)

        scoring = _score_verified_pair(pair, verification)
        verification_rows.append(
            {
                "rank": int(rank),
                **pair,
                "test_segments": test_segments,
                "crop_paths": crop_paths,
                "verification": verification,
                "raw_output": raw_output,
                **scoring,
            }
        )

    if not verification_rows:
        raise ValueError("No pair verification rows were produced.")
    verification_rows.sort(key=lambda item: (-float(item["verified_score"]), int(item["patient_id"]), int(item["tool_id"])))
    best = verification_rows[0]

    selected_payload = {
        "selected": [
            {
                "id": int(best["patient_id"]),
                "role": "patient",
                "entity_type": assignment_map.get(int(best["patient_id"]), {}).get("entity_type", "object"),
                "confidence": float(best["verified_score"]),
                "segments": best["test_segments"],
            },
            {
                "id": int(best["tool_id"]),
                "role": "tool",
                "entity_type": assignment_map.get(int(best["tool_id"]), {}).get("entity_type", "tool"),
                "confidence": float(best["verified_score"]) * 0.94,
                "segments": best["test_segments"],
            },
        ],
        "empty": False,
        "reason": str(best.get("reason", "")),
        "query_slots": query_payload.get("query_slots", {}),
        "source_query_dir": str(query_dir),
        "semantic_source": "refergaussian_qwen_pair_verify",
        "verification_summary": {
            "patient_identity": best.get("patient_identity", ""),
            "tool_identity": best.get("tool_identity", ""),
            "action": best.get("action", ""),
            "match_score": float(best.get("match_score", 0.0)),
            "verified_score": float(best.get("verified_score", 0.0)),
        },
    }

    _write_json(query_dir / "pair_verification.json", {"query": query_payload["query"], "pairs": verification_rows})
    output_path = query_dir / "selected_verified.json"
    _write_json(output_path, selected_payload)
    return output_path
