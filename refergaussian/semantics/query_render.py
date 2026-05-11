from __future__ import annotations

import colorsys
import importlib.util
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import imageio.v2 as iio_v2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .source_images import resolve_dataset_image_entries

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_ROOT = _PROJECT_ROOT / "external" / "4DGaussians"


def _load_camera_class():
    utils_path = _UPSTREAM_ROOT / "scene" / "utils.py"
    spec = importlib.util.spec_from_file_location("refergaussian_scene_utils", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Camera utilities from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Camera


Camera = _load_camera_class()


ROLE_COLORS = {
    "patient": (255, 210, 64),
    "tool": (64, 224, 255),
    "agent": (255, 96, 96),
    "entity": (255, 64, 196),
    "other": (180, 180, 180),
}


@dataclass
class TrackSample:
    entity_id: int
    time_values: np.ndarray
    centers: np.ndarray
    extents_min: np.ndarray
    extents_max: np.ndarray
    visibility: np.ndarray
    support_score: np.ndarray


@dataclass
class EntityCloud:
    entity_id: int
    sample_times: np.ndarray
    trajectories: np.ndarray
    gate: np.ndarray
    spatial_extent: np.ndarray


@dataclass
class QueryTrack:
    phrase: str
    frames: list[dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _normalize_track_state_mode(value: Any) -> str | None:
    text = " ".join(str(value).strip().lower().replace("-", "_").split())
    if not text:
        return None
    normalized = text.replace(" ", "_")
    if normalized in {"none", "null", "unknown", "n/a", "na"}:
        return None
    return normalized


def _selection_track_state_mode(selection_payload: dict[str, Any]) -> str | None:
    for key in ("track_state_mode", "state_mode", "query_state_mode"):
        normalized = _normalize_track_state_mode(selection_payload.get(key))
        if normalized:
            return normalized

    notes = str(selection_payload.get("notes", "")).strip()
    for marker in ("Track state mode=", "State mode="):
        if marker not in notes:
            continue
        tail = notes.split(marker, 1)[1]
        candidate = tail.split(";", 1)[0].strip()
        normalized = _normalize_track_state_mode(candidate)
        if normalized:
            return normalized

    contact_pair = selection_payload.get("contact_pair") or {}
    source = _normalize_track_state_mode(contact_pair.get("source"))
    if not source:
        return None
    if source.startswith("single_subject_track_"):
        suffix = source.removeprefix("single_subject_track_")
        return suffix or "support"
    if "support" in source:
        return "support"
    return None


def _merge_ranges(frame_indices: list[int]) -> list[list[int]]:
    if not frame_indices:
        return []
    sorted_indices = sorted(set(int(v) for v in frame_indices))
    merged: list[list[int]] = []
    start = sorted_indices[0]
    prev = sorted_indices[0]
    for value in sorted_indices[1:]:
        if value == prev + 1:
            prev = value
            continue
        merged.append([start, prev])
        start = value
        prev = value
    merged.append([start, prev])
    return merged


def _find_render_dir(run_dir: Path) -> Path:
    test_dir = run_dir / "test"
    candidates = sorted(test_dir.glob("ours_*/renders"))
    if not candidates:
        raise FileNotFoundError(f"No render directory found under {test_dir}")
    return candidates[-1]


def _find_source_frame_dir(dataset_dir: Path, target_size: tuple[int, int]) -> Path:
    rgb_root = dataset_dir / "rgb"
    if not rgb_root.exists():
        raise FileNotFoundError(f"No rgb directory found under {dataset_dir}")
    candidates = [path for path in rgb_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No rgb scale directories found under {rgb_root}")

    best_dir = None
    best_score = None
    for directory in sorted(candidates):
        sample = next(iter(sorted(directory.glob("*.png"))), None)
        if sample is None:
            continue
        with Image.open(sample) as image:
            size = image.size
        score = abs(size[0] - target_size[0]) + abs(size[1] - target_size[1])
        if best_score is None or score < best_score:
            best_dir = directory
            best_score = score
    if best_dir is None:
        raise FileNotFoundError(f"No source images found under {rgb_root}")
    return best_dir


def _hypernerf_test_ids(dataset_dir: Path) -> tuple[list[str], np.ndarray] | None:
    """Return (test_ids, time_values) for HyperNeRF datasets, or None for DyNeRF datasets."""
    if not (dataset_dir / "dataset.json").exists():
        # DyNeRF dataset — cannot resolve from metadata; caller should use render files
        return None
    dataset_payload = _read_json(dataset_dir / "dataset.json")
    metadata_payload = _read_json(dataset_dir / "metadata.json")
    all_ids = list(dataset_payload["ids"])
    if dataset_payload.get("val_ids"):
        val_ids = set(dataset_payload["val_ids"])
        test_ids = [image_id for image_id in all_ids if image_id in val_ids]
    else:
        i_train = [index for index in range(len(all_ids)) if index % 4 == 0]
        i_test = (np.asarray(i_train, dtype=np.int64) + 2)[:-1]
        test_ids = [all_ids[int(index)] for index in i_test]
    max_time = max(float(metadata_payload[image_id]["warp_id"]) for image_id in all_ids)
    time_values = np.asarray(
        [float(metadata_payload[image_id]["warp_id"]) / max(max_time, 1.0) for image_id in test_ids],
        dtype=np.float32,
    )
    return test_ids, time_values


def _dynerf_test_ids_from_renders(render_dir: Path) -> tuple[list[str], np.ndarray]:
    """Generate test IDs and time values for DyNeRF datasets based on rendered frames."""
    render_files = sorted(render_dir.glob("*.png"))
    n = len(render_files)
    if n == 0:
        raise FileNotFoundError(f"No rendered frames found in {render_dir}")
    test_ids = [f.stem for f in render_files]  # e.g. "00000", "00001", ...
    time_values = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
    return test_ids, time_values


def _load_tracks(run_dir: Path) -> dict[int, TrackSample]:
    payload = _read_json(run_dir / "entitybank" / "semantic_tracks.json")
    track_map: dict[int, TrackSample] = {}
    for track in payload.get("tracks", []):
        frame_payload = track.get("frames", [])
        if not frame_payload:
            continue
        time_values = np.asarray([float(frame["time_value"]) for frame in frame_payload], dtype=np.float32)
        centers = np.asarray([frame["center_world"] for frame in frame_payload], dtype=np.float32)
        extents_min = np.asarray([frame["extent_world_min"] for frame in frame_payload], dtype=np.float32)
        extents_max = np.asarray([frame["extent_world_max"] for frame in frame_payload], dtype=np.float32)
        visibility = np.asarray([float(frame["visibility"]) for frame in frame_payload], dtype=np.float32)
        support_score = np.asarray([float(frame["support_score"]) for frame in frame_payload], dtype=np.float32)
        track_map[int(track["entity_id"])] = TrackSample(
            entity_id=int(track["entity_id"]),
            time_values=time_values,
            centers=centers,
            extents_min=extents_min,
            extents_max=extents_max,
            visibility=visibility,
            support_score=support_score,
        )
    return track_map


def _load_entity_static_texts(run_dir: Path) -> dict[int, str]:
    payload = _read_json(run_dir / "entitybank" / "entities.json")
    return {
        int(entity["id"]): str(entity.get("static_text", "")).strip()
        for entity in payload.get("entities", [])
    }


def _resolve_query_tracks(selection_path: Path) -> dict[str, QueryTrack]:
    try:
        query_root = selection_path.parents[2]
    except IndexError:
        return {}
    tracks_path = query_root / "grounded_sam2" / "grounded_sam2_query_tracks.json"
    if not tracks_path.exists():
        return {}
    payload = _read_json(tracks_path)
    tracks: dict[str, QueryTrack] = {}
    for track in payload.get("tracks", []):
        phrase = " ".join(str(track.get("phrase", "")).strip().lower().split())
        if not phrase:
            continue
        tracks[phrase] = QueryTrack(
            phrase=phrase,
            frames=[frame for frame in track.get("frames", []) if bool(frame.get("active")) and frame.get("mask_path")],
        )
    return tracks


def _query_track_mask_for_time(track: QueryTrack | None, time_value: float, tolerance: float = 0.012) -> np.ndarray | None:
    if track is None or not track.frames:
        return None
    fallback_scale = max(1.0, float(os.environ.get("GS_QUERY_TRACK_FALLBACK_SCALE", "1.0")))
    frame_times = np.asarray([float(frame.get("time_value", 0.0)) for frame in track.frames], dtype=np.float32)
    adaptive_tolerance = float(tolerance)
    if frame_times.size >= 2:
        diffs = np.diff(np.sort(frame_times))
        if diffs.size:
            adaptive_tolerance = max(adaptive_tolerance, float(np.median(diffs)) * 0.65)
    best = min(track.frames, key=lambda frame: abs(float(frame.get("time_value", 0.0)) - float(time_value)))
    time_delta = abs(float(best.get("time_value", 0.0)) - float(time_value))
    if time_delta > adaptive_tolerance * fallback_scale:
        return None
    mask_path = best.get("mask_path")
    if not mask_path:
        return None
    with Image.open(mask_path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def _shift_binary_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    shifted = np.zeros_like(binary, dtype=bool)
    if height <= 0 or width <= 0:
        return shifted
    src_x0 = max(0, -int(dx))
    src_x1 = min(width, width - int(dx)) if int(dx) >= 0 else width
    dst_x0 = max(0, int(dx))
    dst_x1 = min(width, width + int(dx)) if int(dx) <= 0 else width
    src_y0 = max(0, -int(dy))
    src_y1 = min(height, height - int(dy)) if int(dy) >= 0 else height
    dst_y0 = max(0, int(dy))
    dst_y1 = min(height, height + int(dy)) if int(dy) <= 0 else height
    if src_x1 <= src_x0 or src_y1 <= src_y0 or dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return shifted
    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = binary[src_y0:src_y1, src_x0:src_x1]
    return shifted


def _bbox_center(bbox: list[int] | list[float] | tuple[float, ...]) -> tuple[float, float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def _bbox_area(bbox: list[int] | list[float] | tuple[float, ...] | None) -> int:
    if bbox is None:
        return 0
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox]
    width = max(0, x1 - x0 + 1)
    height = max(0, y1 - y0 + 1)
    return int(width * height)


def _box_mask_from_bbox(shape: tuple[int, int], bbox: list[int] | list[float] | tuple[float, ...] | None) -> np.ndarray | None:
    if bbox is None:
        return None
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        return None
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox]
    x0 = int(np.clip(x0, 0, max(width - 1, 0)))
    x1 = int(np.clip(x1, 0, max(width - 1, 0)))
    y0 = int(np.clip(y0, 0, max(height - 1, 0)))
    y1 = int(np.clip(y1, 0, max(height - 1, 0)))
    if x1 < x0 or y1 < y0:
        return None
    mask = np.zeros((height, width), dtype=bool)
    mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


def _expand_cloud_mask_with_projected_box(
    cloud_mask: np.ndarray | None,
    projected_bbox: list[int] | list[float] | tuple[float, ...] | None,
) -> np.ndarray | None:
    if cloud_mask is None:
        return None
    cloud_binary = np.asarray(cloud_mask, dtype=bool)
    box_mask = _box_mask_from_bbox(cloud_binary.shape, projected_bbox)
    if box_mask is None:
        return cloud_binary
    box_area = int(box_mask.sum())
    if box_area < 12000:
        return cloud_binary
    fill_ratio = float(cloud_binary.sum()) / max(float(box_area), 1.0)
    if fill_ratio >= 0.72:
        return cloud_binary
    support = box_mask & _dilate_binary_mask(cloud_binary, kernel_size=21)
    return cloud_binary | support


def _align_query_track_mask(
    query_track_mask: np.ndarray,
    reference_bbox: list[int] | list[float] | tuple[float, ...] | None,
) -> tuple[np.ndarray, list[int] | None]:
    query_binary = np.asarray(query_track_mask, dtype=bool)
    query_bbox = _entity_mask_bbox(query_binary)
    if query_bbox is None or reference_bbox is None:
        return query_binary, query_bbox
    query_center = _bbox_center(query_bbox)
    reference_center = _bbox_center(reference_bbox)
    dx = int(round(reference_center[0] - query_center[0]))
    dy = int(round(reference_center[1] - query_center[1]))
    max_dx = max(6, int(round(query_binary.shape[1] * 0.12)))
    max_dy = max(6, int(round(query_binary.shape[0] * 0.12)))
    if abs(dx) > max_dx or abs(dy) > max_dy:
        return query_binary, query_bbox
    shifted = _shift_binary_mask(query_binary, dx=dx, dy=dy)
    shifted_bbox = _entity_mask_bbox(shifted)
    if shifted_bbox is None:
        return query_binary, query_bbox
    return shifted, shifted_bbox


def _dilate_binary_mask(mask: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {binary.shape}")
    if kernel_size <= 1:
        return binary > 0
    if kernel_size % 2 == 0:
        kernel_size += 1
    image = Image.fromarray(binary * 255, mode="L")
    dilated = image.filter(ImageFilter.MaxFilter(kernel_size))
    return np.asarray(dilated, dtype=np.uint8) > 0


def _fuse_query_and_cloud_masks(
    query_track_mask: np.ndarray | None,
    cloud_mask: np.ndarray | None,
    projected_bbox: list[int] | list[float] | tuple[float, ...] | None = None,
    track_state_mode: str | None = None,
    selected_item_count: int = 1,
) -> tuple[np.ndarray | None, list[int] | None, list[int] | None]:
    if query_track_mask is None and cloud_mask is None:
        return None, None, None
    cloud_binary = _expand_cloud_mask_with_projected_box(cloud_mask, projected_bbox=projected_bbox)
    cloud_bbox = _entity_mask_bbox(cloud_binary) if cloud_binary is not None else None
    if query_track_mask is None:
        return cloud_binary, cloud_bbox, None
    query_binary = np.asarray(query_track_mask, dtype=bool)
    query_bbox = _entity_mask_bbox(query_binary)
    if query_bbox is None:
        return cloud_binary, cloud_bbox, None
    normalized_state = _normalize_track_state_mode(track_state_mode)
    support_like = normalized_state in {None, "support"}
    if cloud_binary is not None and support_like and int(selected_item_count) <= 1:
        # For support-style single-object queries, use the 2D track as the main mask and
        # let the projected worldtube fill only local holes near that evidence.
        support = cloud_binary & _dilate_binary_mask(query_binary, kernel_size=11)
        if bool(support.any()):
            fused = query_binary | support
            fused_bbox = _entity_mask_bbox(fused)
            return fused, fused_bbox, query_bbox
    return query_binary, query_bbox, query_bbox


def _interp_vector(query_times: np.ndarray, sample_times: np.ndarray, values: np.ndarray) -> np.ndarray:
    query_times = np.asarray(query_times, dtype=np.float32)
    sample_times = np.asarray(sample_times, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    result = np.zeros((query_times.shape[0], values.shape[1]), dtype=np.float32)
    for dim in range(values.shape[1]):
        result[:, dim] = np.interp(query_times, sample_times, values[:, dim])
    return result


def _interp_scalar(query_times: np.ndarray, sample_times: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.interp(
        np.asarray(query_times, dtype=np.float32),
        np.asarray(sample_times, dtype=np.float32),
        np.asarray(values, dtype=np.float32),
    ).astype(np.float32)


def _selected_item_gaussian_ids(
    entity_map: dict[int, np.ndarray],
    item: dict[str, Any],
) -> np.ndarray:
    override = np.asarray(item.get("gaussian_ids", []), dtype=np.int64).reshape(-1)
    if override.size > 0:
        return override
    return np.asarray(entity_map.get(int(item["id"]), []), dtype=np.int64).reshape(-1)


def _load_entity_clouds(run_dir: Path, selected_items: list[dict[str, Any]]) -> dict[int, EntityCloud]:
    entitybank_dir = run_dir / "entitybank"
    entities_payload = _read_json(entitybank_dir / "entities.json")
    entity_map = {
        int(entity["id"]): np.asarray(entity.get("gaussian_ids", []), dtype=np.int64)
        for entity in entities_payload.get("entities", [])
    }
    trajectory_payload = np.load(entitybank_dir / "trajectory_samples.npz")
    sample_times = np.asarray(trajectory_payload["time_values"], dtype=np.float32)
    trajectories = np.asarray(trajectory_payload["trajectories"], dtype=np.float32)
    gate = np.asarray(trajectory_payload["gate"], dtype=np.float32).reshape(trajectories.shape[0], trajectories.shape[1])
    spatial_extent = np.asarray(trajectory_payload["spatial_extent"], dtype=np.float32).reshape(-1)

    clouds: dict[int, EntityCloud] = {}
    for item_index, item in enumerate(selected_items):
        gaussian_ids = _selected_item_gaussian_ids(entity_map, item)
        if gaussian_ids.size == 0:
            continue
        gaussian_ids = gaussian_ids[(gaussian_ids >= 0) & (gaussian_ids < trajectories.shape[0])]
        gaussian_ids = np.unique(gaussian_ids)
        if gaussian_ids.size == 0:
            continue
        clouds[int(item_index)] = EntityCloud(
            entity_id=int(item.get("id", -1)),
            sample_times=sample_times,
            trajectories=trajectories[gaussian_ids],
            gate=gate[gaussian_ids],
            spatial_extent=spatial_extent[gaussian_ids],
        )
    return clouds


def _frame_mask(frame_count: int, segments: list[list[int]]) -> np.ndarray:
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


def _load_font(font_size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int]) -> None:
    font = _load_font(20)
    left, top = xy
    bbox = draw.textbbox((left, top), text, font=font)
    draw.rounded_rectangle(
        (bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2),
        radius=4,
        fill=(18, 18, 18),
    )
    draw.text((left, top), text, fill=fill, font=font)


def _entity_color(entity_id: int, role: str) -> tuple[int, int, int]:
    if role in {"patient", "tool", "agent"}:
        return ROLE_COLORS[role]
    hue = float((entity_id * 0.61803398875) % 1.0)
    sat = 0.70 if role == "entity" else 0.55
    val = 1.0
    rgb = colorsys.hsv_to_rgb(hue, sat, val)
    return tuple(int(round(channel * 255.0)) for channel in rgb)


def _entity_label(role: str, entity_id: int) -> str:
    if role == "entity":
        return f"entity #{entity_id}"
    return f"{role} #{entity_id}"


def _image_projection_scale(camera: Camera, image_size: tuple[int, int]) -> tuple[float, float]:
    camera_size = np.asarray(camera.image_size, dtype=np.float32).reshape(-1)
    if camera_size.size >= 2:
        camera_width = max(float(camera_size[0]), 1.0)
        camera_height = max(float(camera_size[1]), 1.0)
    else:
        camera_width = max(float(image_size[0]), 1.0)
        camera_height = max(float(image_size[1]), 1.0)
    image_width = max(float(image_size[0]), 1.0)
    image_height = max(float(image_size[1]), 1.0)
    return image_width / camera_width, image_height / camera_height


def _project_points_to_image(camera: Camera, points_world: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    projected = camera.project(points_world).astype(np.float32)
    scale_x, scale_y = _image_projection_scale(camera, image_size)
    projected[:, 0] *= float(scale_x)
    projected[:, 1] *= float(scale_y)
    return projected


def _pixel_radius(
    camera: Camera,
    center_world: np.ndarray,
    extent_min: np.ndarray,
    extent_max: np.ndarray,
    image_size: tuple[int, int] | None = None,
) -> int:
    center_local = camera.points_to_local_points(center_world[None, :])[0]
    depth = float(center_local[2])
    if depth <= 1.0e-4:
        return 10
    extent_radius = 0.5 * float(np.linalg.norm(extent_max - extent_min))
    if image_size is None:
        camera_size = np.asarray(camera.image_size, dtype=np.int32).reshape(-1)
        image_size = (int(camera_size[0]), int(camera_size[1]))
    scale_x, scale_y = _image_projection_scale(camera, image_size)
    scale = 0.5 * (float(scale_x) + float(scale_y))
    pixel_radius = float(camera.focal_length) * scale * extent_radius / max(depth, 1.0e-4)
    return int(np.clip(pixel_radius, 10.0, 72.0))


def _aabb_corners(extent_min: np.ndarray, extent_max: np.ndarray) -> np.ndarray:
    min_x, min_y, min_z = np.asarray(extent_min, dtype=np.float32).tolist()
    max_x, max_y, max_z = np.asarray(extent_max, dtype=np.float32).tolist()
    return np.asarray(
        [
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ],
        dtype=np.float32,
    )


def _project_box(
    camera: Camera,
    extent_min: np.ndarray,
    extent_max: np.ndarray,
    image_size: tuple[int, int],
) -> dict[str, Any] | None:
    corners_world = _aabb_corners(extent_min, extent_max)
    corners_local = camera.points_to_local_points(corners_world)
    valid = np.asarray(corners_local[:, 2] > 1.0e-4, dtype=bool)
    if not valid.any():
        return None

    corners_projected = _project_points_to_image(camera, corners_world[valid], image_size=image_size)
    width, height = image_size
    raw_left = float(corners_projected[:, 0].min())
    raw_top = float(corners_projected[:, 1].min())
    raw_right = float(corners_projected[:, 0].max())
    raw_bottom = float(corners_projected[:, 1].max())
    intersects = not (
        raw_right < 0.0
        or raw_left >= float(width)
        or raw_bottom < 0.0
        or raw_top >= float(height)
    )
    clamped_left = float(np.clip(raw_left, 0.0, max(width - 1, 0)))
    clamped_top = float(np.clip(raw_top, 0.0, max(height - 1, 0)))
    clamped_right = float(np.clip(raw_right, 0.0, max(width - 1, 0)))
    clamped_bottom = float(np.clip(raw_bottom, 0.0, max(height - 1, 0)))
    if clamped_right <= clamped_left:
        clamped_right = min(float(width - 1), clamped_left + 1.0)
    if clamped_bottom <= clamped_top:
        clamped_bottom = min(float(height - 1), clamped_top + 1.0)
    return {
        "raw_xyxy": [raw_left, raw_top, raw_right, raw_bottom],
        "clamped_xyxy": [clamped_left, clamped_top, clamped_right, clamped_bottom],
        "intersects_frame": bool(intersects),
        "projected_corners": corners_projected.astype(float).tolist(),
    }


def _entity_mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _project_entity_cloud_mask(
    camera: Camera,
    image_size: tuple[int, int],
    cloud: EntityCloud,
    time_value: float,
) -> tuple[np.ndarray | None, list[int] | None]:
    if cloud.trajectories.size == 0:
        return None, None
    sample_index = int(np.abs(cloud.sample_times - float(time_value)).argmin())
    points_world = np.asarray(cloud.trajectories[:, sample_index, :], dtype=np.float32)
    gate_values = np.asarray(cloud.gate[:, sample_index], dtype=np.float32).reshape(-1)
    active = gate_values >= max(0.08, float(gate_values.max()) * 0.15)
    if not active.any():
        return None, None
    points_world = points_world[active]
    point_extent = np.asarray(cloud.spatial_extent[active], dtype=np.float32).reshape(-1)
    points_local = camera.points_to_local_points(points_world)
    valid = points_local[:, 2] > 1.0e-4
    if not valid.any():
        return None, None
    points_world = points_world[valid]
    points_local = points_local[valid]
    point_extent = point_extent[valid]
    projected = _project_points_to_image(camera, points_world, image_size=image_size)
    width, height = image_size
    in_bounds = (
        (projected[:, 0] >= -32.0)
        & (projected[:, 0] < float(width + 32))
        & (projected[:, 1] >= -32.0)
        & (projected[:, 1] < float(height + 32))
    )
    if not in_bounds.any():
        return None, None
    projected = projected[in_bounds]
    points_local = points_local[in_bounds]
    point_extent = point_extent[in_bounds]

    mask_image = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask_image)
    scale_x, scale_y = _image_projection_scale(camera, image_size)
    focal = float(camera.focal_length) * 0.5 * (float(scale_x) + float(scale_y))
    for pixel, local_point, extent in zip(projected, points_local, point_extent):
        depth = max(float(local_point[2]), 1.0e-4)
        radius = int(np.clip(0.75 * focal * max(float(extent), 1.0e-4) / depth, 3.0, 18.0))
        cx = int(round(float(pixel[0])))
        cy = int(round(float(pixel[1])))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
    mask_image = mask_image.filter(ImageFilter.MaxFilter(11))
    mask = np.asarray(mask_image, dtype=np.uint8) > 0
    bbox = _entity_mask_bbox(mask)
    if bbox is None:
        return None, None
    return mask, bbox


def _overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: int) -> Image.Image:
    base = image.convert("RGBA")
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay[mask, 0] = color[0]
    overlay[mask, 1] = color[1]
    overlay[mask, 2] = color[2]
    overlay[mask, 3] = int(np.clip(alpha, 0, 255))
    return Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))


def _contact_threshold(patient_extent: np.ndarray, tool_extent: np.ndarray) -> float:
    patient_radius = 0.5 * float(np.linalg.norm(patient_extent))
    tool_radius = 0.5 * float(np.linalg.norm(tool_extent))
    return max(0.10, 0.65 * (patient_radius + tool_radius))


def _video_meta(path: Path, fps: int) -> dict[str, Any]:
    return {
        "path": str(path),
        "fps": int(fps),
        "exists": path.exists(),
    }


def _open_video_writer(path: Path, fps: int):
    try:
        writer = iio_v2.get_writer(
            str(path),
            fps=fps,
            codec="libx264",
            macro_block_size=None,
        )
        return writer, path
    except Exception:
        fallback_path = path.with_suffix(".gif")
        writer = iio_v2.get_writer(str(fallback_path), mode="I", fps=fps)
        return writer, fallback_path


def render_hypernerf_query_video(
    run_dir: str | Path,
    dataset_dir: str | Path,
    selection_path: str | Path,
    output_dir: str | Path | None = None,
    fps: int = 12,
    stride: int = 1,
    background_mode: str = "render",
) -> Path:
    run_dir = Path(run_dir)
    dataset_dir = Path(dataset_dir)
    selection_path = Path(selection_path)
    output_dir = Path(output_dir) if output_dir is not None else selection_path.parent / "native_render"
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_frame_dir = output_dir / "overlay_frames"
    binary_mask_dir = output_dir / "binary_masks"
    overlay_frame_dir.mkdir(parents=True, exist_ok=True)
    binary_mask_dir.mkdir(parents=True, exist_ok=True)

    if background_mode not in {"render", "source"}:
        raise ValueError(f"Unsupported background_mode: {background_mode}")

    source_frame_dir = None
    render_dir: Path | None = None
    render_files: list[Path]
    hypernerf_ids = _hypernerf_test_ids(dataset_dir)
    try:
        render_dir = _find_render_dir(run_dir)
        render_files = sorted(render_dir.glob("*.png"))
        if not render_files:
            raise FileNotFoundError(f"No frames found in {render_dir}")
        with Image.open(render_files[0]) as probe:
            target_size = probe.size
        if background_mode == "source":
            source_frame_dir = _find_source_frame_dir(dataset_dir, target_size)
        if hypernerf_ids is not None:
            test_ids, test_times = hypernerf_ids
            if len(test_ids) != len(render_files):
                raise ValueError(
                    f"Render frame count ({len(render_files)}) does not match HyperNeRF test ids ({len(test_ids)})"
                )
        else:
            # DyNeRF dataset: derive test IDs from render files
            test_ids, test_times = _dynerf_test_ids_from_renders(render_dir)
    except FileNotFoundError:
        if background_mode != "source":
            raise
        entries = resolve_dataset_image_entries(dataset_dir)
        if not entries:
            raise FileNotFoundError(f"No source images found under {dataset_dir}")
        if hypernerf_ids is not None:
            test_ids, test_times = hypernerf_ids
            entry_map = {str(entry["image_id"]): Path(str(entry["image_path"])) for entry in entries}
            missing_ids = [image_id for image_id in test_ids if image_id not in entry_map]
            if missing_ids:
                preview = ", ".join(missing_ids[:5])
                raise FileNotFoundError(
                    f"Missing source frames for {len(missing_ids)} test ids under {dataset_dir}: {preview}"
                )
            render_files = [entry_map[image_id] for image_id in test_ids]
        else:
            # DyNeRF: use all entries as source frames
            render_files = [Path(str(entry["image_path"])) for entry in entries]
            test_ids = [str(entry["image_id"]) for entry in entries]
            test_times = np.linspace(0.0, 1.0, num=len(entries), dtype=np.float32)
        source_frame_dir = render_files[0].parent
        with Image.open(render_files[0]) as probe:
            target_size = probe.size

    selection_payload = _read_json(selection_path)
    selected_items = selection_payload.get("selected", [])
    selection_empty = selection_payload.get("empty", False)
    if not selected_items and selection_empty:
        # Negative query: Qwen determined entity doesn't satisfy the query.
        # Produce all-inactive (all-black) binary masks so evaluator can score correctly.
        frame_records = []
        for frame_index, (test_id, _t) in enumerate(zip(test_ids, test_times)):
            bg_file = (
                render_files[frame_index]
                if frame_index < len(render_files)
                else (render_files[-1] if render_files else None)
            )
            with Image.open(bg_file) as bg_img:
                W, H = bg_img.size
            black_mask = Image.fromarray(np.zeros((H, W), dtype=np.uint8))
            mask_fname = f"{str(test_id).zfill(6)}.png"
            overlay_fname = f"{str(test_id).zfill(6)}.png"
            black_mask.save(binary_mask_dir / mask_fname)
            # Save overlay as the background frame (no highlight)
            bg_file_copy = render_files[frame_index] if render_files else None
            if bg_file_copy is not None:
                shutil.copy2(bg_file_copy, overlay_frame_dir / overlay_fname)
            frame_records.append({
                "frame_index": frame_index,
                "image_id": str(test_id),
                "query_active": False,
                "entity_active": False,
            })
        validation_data = {
            "selection_mode": selection_payload.get("selection_mode", "qwen_plan_empty"),
            "empty_selection": True,
            "frame_exports": {
                "overlay_frames": str(overlay_frame_dir),
                "binary_masks": str(binary_mask_dir),
            },
            "frames": frame_records,
        }
        _write_json(output_dir / "validation.json", validation_data)
        return output_dir
    if not selected_items:
        raise ValueError(f"No selected entities found in {selection_path}")
    track_state_mode = _selection_track_state_mode(selection_payload)

    tracks = _load_tracks(run_dir)
    entity_text_map = _load_entity_static_texts(run_dir)
    query_tracks = _resolve_query_tracks(selection_path)
    entity_clouds = _load_entity_clouds(run_dir, selected_items)
    role_entries = []
    for item_index, item in enumerate(selected_items):
        entity_id = int(item["id"])
        track = tracks.get(entity_id)
        if track is None:
            continue
        role_entries.append(
            {
                "role": str(item.get("role", "other")),
                "entity_id": entity_id,
                "source_entity_id": int(item.get("source_entity_id", -1)),
                "confidence": float(item.get("confidence", 0.0)),
                "segments": item.get("segments", []),
                "track": track,
                "center_world": _interp_vector(test_times, track.time_values, track.centers),
                "extent_min": _interp_vector(test_times, track.time_values, track.extents_min),
                "extent_max": _interp_vector(test_times, track.time_values, track.extents_max),
                "visibility": _interp_scalar(test_times, track.time_values, track.visibility),
                "support_score": _interp_scalar(test_times, track.time_values, track.support_score),
                "cloud": entity_clouds.get(int(item_index)),
                "track_phrase": " ".join(entity_text_map.get(entity_id, "").lower().split()),
                "query_track": query_tracks.get(" ".join(entity_text_map.get(entity_id, "").lower().split())),
            }
        )
    if not role_entries:
        raise ValueError("Selected entities could not be matched to target semantic tracks")

    frame_count = len(render_files)
    for entry in role_entries:
        entry["frame_mask"] = _frame_mask(frame_count, entry["segments"])
    active_mask = np.zeros((frame_count,), dtype=bool)
    for entry in role_entries:
        active_mask |= entry["frame_mask"]

    patient_entry = next((entry for entry in role_entries if entry["role"] == "patient"), None)
    tool_entry = next((entry for entry in role_entries if entry["role"] == "tool"), None)
    contact_mask = np.zeros((frame_count,), dtype=bool)
    proximity_mask = np.zeros((frame_count,), dtype=bool)
    contact_distance = np.full((frame_count,), np.nan, dtype=np.float32)
    if patient_entry is not None and tool_entry is not None:
        patient_extent = patient_entry["extent_max"] - patient_entry["extent_min"]
        tool_extent = tool_entry["extent_max"] - tool_entry["extent_min"]
        threshold = np.asarray(
            [_contact_threshold(patient_extent[i], tool_extent[i]) for i in range(frame_count)],
            dtype=np.float32,
        )
        distance = np.linalg.norm(patient_entry["center_world"] - tool_entry["center_world"], axis=1)
        overlap_mask = patient_entry["frame_mask"] & tool_entry["frame_mask"]
        proximity_mask = overlap_mask & (distance <= threshold)
        contact_mask = overlap_mask.copy()
        contact_distance = distance.astype(np.float32)

    overlay_writer, overlay_path = _open_video_writer(output_dir / "overlay.mp4", fps=fps)
    mask_writer, mask_path = _open_video_writer(output_dir / "mask.mp4", fps=fps)

    first_active_frame = None
    first_contact_frame = None
    frame_records = []
    saved_frames: list[tuple[str, int, Path]] = []

    try:
        for frame_index, (frame_path, image_id, time_value) in enumerate(zip(render_files, test_ids, test_times)):
            if frame_index % max(stride, 1) != 0:
                continue
            if background_mode == "source":
                source_path = source_frame_dir / f"{image_id}.png"
                if not source_path.exists():
                    raise FileNotFoundError(f"Missing source frame {source_path}")
                frame = Image.open(source_path).convert("RGB")
                if frame.size != target_size:
                    frame = frame.resize(target_size, Image.Resampling.BILINEAR)
            else:
                frame = Image.open(frame_path).convert("RGB")
            overlay = frame.copy()
            mask_image = Image.new("RGB", frame.size, (0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay, "RGBA")
            mask_draw = ImageDraw.Draw(mask_image, "RGBA")

            camera = Camera.from_json(dataset_dir / "camera" / f"{image_id}.json")
            frame_roles = []
            for entry in role_entries:
                role = entry["role"]
                active = bool(entry["frame_mask"][frame_index])
                visible = bool(entry["visibility"][frame_index] >= 0.2)
                center_world = entry["center_world"][frame_index]
                pixel = _project_points_to_image(camera, center_world[None, :], image_size=overlay.size)[0]
                center_local = camera.points_to_local_points(center_world[None, :])[0]
                in_front = bool(center_local[2] > 1.0e-4)
                width, height = overlay.size
                in_bounds = bool(0.0 <= pixel[0] < width and 0.0 <= pixel[1] < height)
                displayable = bool(in_front)
                clamped_pixel = np.asarray(
                    [
                        np.clip(pixel[0], 8.0, width - 8.0),
                        np.clip(pixel[1], 8.0, height - 8.0),
                    ],
                    dtype=np.float32,
                )
                radius = _pixel_radius(
                    camera,
                    center_world,
                    entry["extent_min"][frame_index],
                    entry["extent_max"][frame_index],
                    image_size=overlay.size,
                )
                projected_box = _project_box(
                    camera,
                    entry["extent_min"][frame_index],
                    entry["extent_max"][frame_index],
                    overlay.size,
                )
                cloud_mask = None
                cloud_bbox = None
                query_track_mask = _query_track_mask_for_time(entry.get("query_track"), float(time_value))
                query_track_bbox = _entity_mask_bbox(query_track_mask) if query_track_mask is not None else None
                if entry.get("cloud") is not None:
                    cloud_mask, cloud_bbox = _project_entity_cloud_mask(
                        camera,
                        overlay.size,
                        entry["cloud"],
                        float(time_value),
                    )
                role_record = {
                    "role": role,
                    "entity_id": int(entry["entity_id"]),
                    "source_entity_id": int(entry["source_entity_id"]),
                    "confidence": float(entry["confidence"]),
                    "active": active,
                    "visible": visible,
                    "displayable": displayable,
                    "projected": bool(displayable and in_bounds),
                    "offscreen": bool(displayable and not in_bounds),
                    "pixel_xy": [float(pixel[0]), float(pixel[1])],
                    "display_xy": [float(clamped_pixel[0]), float(clamped_pixel[1])],
                    "depth": float(center_local[2]),
                    "radius_px": int(radius),
                    "bbox_xyxy_raw": None if projected_box is None else projected_box["raw_xyxy"],
                    "bbox_xyxy_clamped": None if projected_box is None else projected_box["clamped_xyxy"],
                    "bbox_intersects_frame": False if projected_box is None else bool(projected_box["intersects_frame"]),
                    "query_track_bbox_xyxy": query_track_bbox,
                    "cloud_mask_bbox_xyxy": cloud_bbox,
                    "support_score": float(entry["support_score"][frame_index]),
                }
                frame_roles.append(role_record)
                if not (active and visible and displayable):
                    continue
                color = _entity_color(int(entry["entity_id"]), role)
                cx = int(round(clamped_pixel[0]))
                cy = int(round(clamped_pixel[1]))
                left = cx - radius
                top = cy - radius
                right = cx + radius
                bottom = cy + radius
                fused_mask, fused_bbox, aligned_query_bbox = _fuse_query_and_cloud_masks(
                    query_track_mask,
                    cloud_mask,
                    projected_bbox=None if projected_box is None else projected_box["clamped_xyxy"],
                    track_state_mode=track_state_mode,
                    selected_item_count=len(selected_items),
                )
                if aligned_query_bbox is not None:
                    role_record["query_track_bbox_xyxy"] = aligned_query_bbox
                if fused_mask is not None and fused_bbox is not None and query_track_mask is not None:
                    overlay = _overlay_mask(overlay, fused_mask, color, alpha=132)
                    mask_image = _overlay_mask(mask_image, fused_mask, color, alpha=255).convert("RGB")
                    overlay_draw = ImageDraw.Draw(overlay, "RGBA")
                    mask_draw = ImageDraw.Draw(mask_image, "RGBA")
                    box_left, box_top, box_right, box_bottom = fused_bbox
                    box_left, box_right = sorted((int(round(box_left)), int(round(box_right))))
                    box_top, box_bottom = sorted((int(round(box_top)), int(round(box_bottom))))
                    overlay_draw.rectangle(
                        (box_left, box_top, box_right, box_bottom),
                        outline=color + (255,),
                        width=4,
                    )
                    label_x = max(0, min(box_left, width - 220))
                    label_y = max(0, box_top - 26)
                    _draw_label(overlay_draw, (label_x, label_y), _entity_label(role, int(entry["entity_id"])), color)
                elif cloud_mask is not None and cloud_bbox is not None:
                    overlay = _overlay_mask(overlay, cloud_mask, color, alpha=104)
                    mask_image = _overlay_mask(mask_image, cloud_mask, color, alpha=255).convert("RGB")
                    overlay_draw = ImageDraw.Draw(overlay, "RGBA")
                    mask_draw = ImageDraw.Draw(mask_image, "RGBA")
                    box_left, box_top, box_right, box_bottom = cloud_bbox
                    box_left, box_right = sorted((int(round(box_left)), int(round(box_right))))
                    box_top, box_bottom = sorted((int(round(box_top)), int(round(box_bottom))))
                    overlay_draw.rectangle(
                        (box_left, box_top, box_right, box_bottom),
                        outline=color + (255,),
                        width=4,
                    )
                    label_x = max(0, min(box_left, width - 220))
                    label_y = max(0, box_top - 26)
                    _draw_label(overlay_draw, (label_x, label_y), _entity_label(role, int(entry["entity_id"])), color)
                else:
                    overlay_draw = ImageDraw.Draw(overlay, "RGBA")
                    mask_draw = ImageDraw.Draw(mask_image, "RGBA")
                    if projected_box is not None and projected_box["intersects_frame"]:
                        box_left, box_top, box_right, box_bottom = [int(round(v)) for v in projected_box["clamped_xyxy"]]
                        box_left, box_right = sorted((box_left, box_right))
                        box_top, box_bottom = sorted((box_top, box_bottom))
                        overlay_draw.rectangle(
                            (box_left, box_top, box_right, box_bottom),
                            fill=color + (56,),
                            outline=color + (255,),
                            width=6,
                        )
                        inner_left = min(box_right, box_left + 3)
                        inner_top = min(box_bottom, box_top + 3)
                        inner_right = max(box_left, box_right - 3)
                        inner_bottom = max(box_top, box_bottom - 3)
                        if inner_right >= inner_left and inner_bottom >= inner_top:
                            overlay_draw.rectangle(
                                (inner_left, inner_top, inner_right, inner_bottom),
                                outline=color + (156,),
                                width=2,
                            )
                        mask_draw.rectangle((box_left, box_top, box_right, box_bottom), fill=color + (255,))
                        overlay_draw.line((cx - radius, cy, cx + radius, cy), fill=color + (255,), width=3)
                        overlay_draw.line((cx, cy - radius, cx, cy + radius), fill=color + (255,), width=3)
                        label_x = max(0, min(box_left, width - 220))
                        label_y = max(0, box_top - 26)
                        _draw_label(overlay_draw, (label_x, label_y), _entity_label(role, int(entry["entity_id"])), color)
                    elif in_bounds:
                        overlay_draw.ellipse((left, top, right, bottom), fill=color + (72,), outline=color + (255,), width=5)
                        mask_draw.ellipse((left, top, right, bottom), fill=color + (255,))
                        overlay_draw.line((cx - radius, cy, cx + radius, cy), fill=color + (255,), width=3)
                        overlay_draw.line((cx, cy - radius, cx, cy + radius), fill=color + (255,), width=3)
                        _draw_label(overlay_draw, (left, max(0, top - 24)), _entity_label(role, int(entry["entity_id"])), color)
                    else:
                        overlay_draw.regular_polygon(
                            (cx, cy, max(radius, 16)),
                            n_sides=4,
                            rotation=45,
                            fill=color + (72,),
                            outline=color + (255,),
                            width=5,
                        )
                        mask_draw.regular_polygon((cx, cy, max(radius, 16)), n_sides=4, rotation=45, fill=color + (255,))
                        _draw_label(
                            overlay_draw,
                            (min(width - 260, cx + 12), max(0, cy - 12)),
                            f"{_entity_label(role, int(entry['entity_id']))} center offscreen",
                            color,
                        )

            patient_role = next((record for record in frame_roles if record["role"] == "patient"), None)
            tool_role = next((record for record in frame_roles if record["role"] == "tool"), None)
            is_contact = bool(contact_mask[frame_index]) if frame_index < contact_mask.shape[0] else False
            is_proximity_contact = bool(proximity_mask[frame_index]) if frame_index < proximity_mask.shape[0] else False
            if patient_role and tool_role and patient_role["displayable"] and tool_role["displayable"]:
                p0 = tuple(int(round(v)) for v in patient_role["display_xy"])
                p1 = tuple(int(round(v)) for v in tool_role["display_xy"])
                line_color = (255, 96, 96, 220) if is_contact else (255, 255, 255, 160)
                overlay_draw.line((p0[0], p0[1], p1[0], p1[1]), fill=line_color, width=4)
                mask_draw.line((p0[0], p0[1], p1[0], p1[1]), fill=(255, 255, 255, 255), width=5)
                if is_contact:
                    mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
                    label = "contact" if is_proximity_contact else "interaction"
                    _draw_label(overlay_draw, (mid[0] + 8, mid[1] + 8), label, (255, 96, 96))

            query_active = bool(active_mask[frame_index])
            if query_active and first_active_frame is None:
                first_active_frame = frame_index
            if is_contact and first_contact_frame is None:
                first_contact_frame = frame_index

            query_text = str(selection_payload.get("query", "query"))
            headline = f"ReferGaussian query render: {query_text}"
            visible_count = int(sum(1 for record in frame_roles if record["active"] and record["displayable"]))
            status = f"frame {frame_index:04d}  time={float(time_value):.3f}  active={'yes' if query_active else 'no'}  entities={visible_count}"
            overlay_draw = ImageDraw.Draw(overlay, "RGBA")
            _draw_label(overlay_draw, (16, 16), headline, (240, 240, 240))
            _draw_label(overlay_draw, (16, 44), status, (220, 220, 220))

            overlay_rgb = overlay.convert("RGB") if overlay.mode != "RGB" else overlay
            overlay_np = np.asarray(overlay_rgb, dtype=np.uint8)
            mask_np = np.asarray(mask_image, dtype=np.uint8)
            overlay_writer.append_data(overlay_np)
            mask_writer.append_data(mask_np)
            overlay_rgb.save(overlay_frame_dir / f"{frame_index:05d}.png")
            Image.fromarray(mask_np, mode="RGB").save(binary_mask_dir / f"{frame_index:05d}.png")

            frame_records.append(
                {
                    "frame_index": int(frame_index),
                    "image_id": image_id,
                    "time_value": float(time_value),
                    "query_active": query_active,
                    "contact_active": is_contact,
                    "proximity_contact_active": is_proximity_contact,
                    "contact_distance_world": None
                    if np.isnan(contact_distance[frame_index])
                    else float(contact_distance[frame_index]),
                    "roles": frame_roles,
                }
            )

            should_save = (
                (first_active_frame is not None and frame_index == first_active_frame)
                or (first_contact_frame is not None and frame_index == first_contact_frame)
            )
            if should_save:
                label = "first_contact" if frame_index == first_contact_frame else "first_active"
                out_path = output_dir / f"{label}_{frame_index:04d}.png"
                overlay_rgb.save(out_path)
                saved_frames.append((label, frame_index, out_path))
    finally:
        overlay_writer.close()
        mask_writer.close()

    active_indices = [record["frame_index"] for record in frame_records if record["query_active"]]
    contact_indices = [record["frame_index"] for record in frame_records if record["contact_active"]]
    payload = {
        "schema_version": 1,
        "query": selection_payload.get("query", "query"),
        "selection_path": str(selection_path),
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "native_render": True,
        "background_mode": background_mode,
        "background_frame_dir": None if source_frame_dir is None else str(source_frame_dir),
        "render_dir": str(render_dir) if render_dir is not None else str(source_frame_dir),
        "frame_count": len(frame_records),
        "active_frame_count": int(len(active_indices)),
        "first_active_frame": None if not active_indices else int(active_indices[0]),
        "last_active_frame": None if not active_indices else int(active_indices[-1]),
        "active_segments": _merge_ranges(active_indices),
        "contact_frame_count": int(len(contact_indices)),
        "first_contact_frame": None if not contact_indices else int(contact_indices[0]),
        "last_contact_frame": None if not contact_indices else int(contact_indices[-1]),
        "contact_segments": _merge_ranges(contact_indices),
        "proximity_contact_frame_count": int(sum(record["proximity_contact_active"] for record in frame_records)),
        "proximity_contact_segments": _merge_ranges(
            [record["frame_index"] for record in frame_records if record["proximity_contact_active"]]
        ),
        "roles": [
            {
                "role": entry["role"],
                "entity_id": int(entry["entity_id"]),
                "source_entity_id": int(entry["source_entity_id"]),
                "confidence": float(entry["confidence"]),
                "segments": entry["segments"],
            }
            for entry in role_entries
        ],
        "videos": {
            "overlay": _video_meta(overlay_path, fps),
            "mask": _video_meta(mask_path, fps),
        },
        "frame_exports": {
            "overlay_frames": str(overlay_frame_dir),
            "binary_masks": str(binary_mask_dir),
        },
        "saved_frames": [
            {"label": label, "frame_index": int(frame_index), "path": str(path)}
            for label, frame_index, path in saved_frames
        ],
        "frames": frame_records,
    }
    _write_json(output_dir / "validation.json", payload)
    return output_dir
