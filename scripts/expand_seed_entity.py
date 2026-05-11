import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy import ndimage
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "external" / "4DGaussians"


def _load_camera_class():
    utils_path = UPSTREAM_ROOT / "scene" / "utils.py"
    spec = importlib.util.spec_from_file_location("refergaussian_scene_utils_mask_guided", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Camera utilities from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Camera


Camera = _load_camera_class()


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _find_latest_iteration_dir(run_dir: Path) -> Path:
    point_cloud_root = run_dir / "point_cloud"
    candidates: list[tuple[int, Path]] = []
    for child in point_cloud_root.iterdir():
        if not child.is_dir() or not child.name.startswith("iteration_"):
            continue
        try:
            iteration = int(child.name.split("_", 1)[1])
        except ValueError:
            continue
        candidates.append((iteration, child))
    if not candidates:
        raise FileNotFoundError(f"No iteration_* directory found under {point_cloud_root}")
    return max(candidates, key=lambda item: item[0])[1]


def _safe_quantile(values: np.ndarray, quantile: float, fallback: float) -> float:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return float(fallback)
    return float(np.quantile(array, float(np.clip(quantile, 0.0, 1.0))))


def _regularized_mahalanobis(values: np.ndarray, reference_ids: np.ndarray, ridge: float = 1.0e-4) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    reference_ids = np.asarray(reference_ids, dtype=np.int64).reshape(-1)
    if reference_ids.size == 0:
        zeros = np.zeros((values.shape[0],), dtype=np.float32)
        return zeros, zeros
    reference = values[reference_ids]
    if reference.shape[0] <= 1:
        center = reference.mean(axis=0, keepdims=True)
        scale = max(float(np.linalg.norm(np.std(reference, axis=0))) + ridge, ridge)
        distances = np.linalg.norm(values - center, axis=1) / scale
        return distances.astype(np.float32), distances[reference_ids].astype(np.float32)
    cov = np.cov(reference.T) + np.eye(reference.shape[1], dtype=np.float32) * float(ridge)
    inv_cov = np.linalg.inv(cov)
    center = reference.mean(axis=0)
    distances = np.sqrt(np.einsum("ni,ij,nj->n", values - center[None, :], inv_cov, values - center[None, :]))
    return distances.astype(np.float32), distances[reference_ids].astype(np.float32)


def _norm_phrase(text: str | None) -> str:
    raw = str("" if text is None else text).strip().lower().replace("_", " ").replace("-", " ")
    if "__" in raw:
        raw = raw.split("__", 1)[0]
    return " ".join(raw.split())


def _resample_indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    source = np.asarray(source_times, dtype=np.float32).reshape(-1)
    target = np.asarray(target_times, dtype=np.float32).reshape(-1)
    return np.abs(target[:, None] - source[None, :]).argmin(axis=1).astype(np.int32)


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


def _load_binary_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    if not mask.any():
        return None
    return mask


def _bbox_to_mask(image_size: tuple[int, int], bbox_xyxy: list[float] | None) -> np.ndarray | None:
    if bbox_xyxy is None:
        return None
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        return None
    left, top, right, bottom = [float(v) for v in bbox_xyxy]
    x0 = int(np.clip(np.floor(left), 0, width - 1))
    y0 = int(np.clip(np.floor(top), 0, height - 1))
    x1 = int(np.clip(np.ceil(right), x0 + 1, width))
    y1 = int(np.clip(np.ceil(bottom), y0 + 1, height))
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _positive_mask_from_frame(frame: dict[str, Any], image_size: tuple[int, int]) -> tuple[np.ndarray | None, str]:
    mask_path = frame.get("mask_path")
    if mask_path:
        positive_mask = _load_binary_mask(Path(str(mask_path)))
        if positive_mask is not None:
            return positive_mask, "mask"
    positive_mask = _bbox_to_mask(image_size=image_size, bbox_xyxy=frame.get("bbox_xyxy"))
    if positive_mask is not None:
        return positive_mask, "bbox"
    return None, "missing"


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if ys.size == 0 or xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _frame_masks(
    frame: dict[str, Any],
    image_size: tuple[int, int],
    dilation_radius: int,
    negative_margin: float,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    positive_mask, mask_source = _positive_mask_from_frame(frame=frame, image_size=image_size)
    if positive_mask is None:
        return None, None, {
            "mask_source": "missing",
            "bbox_xyxy": None,
        }

    bbox = _mask_bbox(positive_mask)
    if bbox is None:
        return None, None, {
            "mask_source": "empty",
            "bbox_xyxy": None,
        }
    height, width = positive_mask.shape
    left, top, right, bottom = bbox
    pad_x = max(int(round((right - left + 1) * float(negative_margin))), dilation_radius + 1, 2)
    pad_y = max(int(round((bottom - top + 1) * float(negative_margin))), dilation_radius + 1, 2)
    exp_left = max(0, left - pad_x)
    exp_top = max(0, top - pad_y)
    exp_right = min(width, right + pad_x + 1)
    exp_bottom = min(height, bottom + pad_y + 1)
    expanded = np.zeros_like(positive_mask, dtype=bool)
    expanded[exp_top:exp_bottom, exp_left:exp_right] = True
    dilated = ndimage.binary_dilation(positive_mask, iterations=max(int(dilation_radius), 0)) if dilation_radius > 0 else positive_mask
    negative_mask = np.logical_and(expanded, np.logical_not(dilated))
    return positive_mask, negative_mask, {
        "mask_source": mask_source,
        "bbox_xyxy": [int(left), int(top), int(right), int(bottom)],
        "expanded_bbox_xyxy": [int(exp_left), int(exp_top), int(exp_right - 1), int(exp_bottom - 1)],
    }


def _resolve_query_phrase(matched_entity: dict[str, Any], summary_row: dict[str, Any] | None) -> str:
    for value in (
        None if summary_row is None else summary_row.get("phrase"),
        matched_entity.get("static_text"),
        matched_entity.get("proposal_alias"),
    ):
        phrase = _norm_phrase(None if value is None else str(value))
        if phrase:
            return phrase
    return ""


def _default_query_tracks_path(source_proposal_dir: Path) -> Path | None:
    candidate = source_proposal_dir.parent / "grounded_sam2" / "grounded_sam2_query_tracks.json"
    return candidate if candidate.exists() else None


def _load_query_track_frames(
    query_tracks_path: Path | None,
    query_phrase: str,
    max_frames: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if query_tracks_path is None or not query_tracks_path.exists():
        return None, {
            "enabled": False,
            "skip_reason": "query_tracks_missing",
        }
    payload = _read_json(query_tracks_path)
    phrase_norm = _norm_phrase(query_phrase)
    track = None
    for row in payload.get("tracks", []):
        if _norm_phrase(row.get("phrase")) == phrase_norm:
            track = row
            break
    if track is None and len(payload.get("tracks", [])) == 1:
        track = payload["tracks"][0]
    if track is None:
        return None, {
            "enabled": False,
            "skip_reason": "query_phrase_not_found",
            "query_phrase": phrase_norm,
            "available_phrases": [_norm_phrase(row.get("phrase")) for row in payload.get("tracks", [])],
        }

    active_frames = [
        frame
        for frame in track.get("frames", [])
        if bool(frame.get("active")) and (
            frame.get("bbox_xyxy") is not None or frame.get("mask_path") is not None
        )
    ]
    active_frames = sorted(active_frames, key=lambda item: int(item.get("frame_index", 0)))
    if not active_frames:
        return None, {
            "enabled": False,
            "skip_reason": "no_active_track_frames",
            "query_phrase": phrase_norm,
        }
    if len(active_frames) > int(max_frames):
        indices = np.linspace(0, len(active_frames) - 1, num=int(max_frames), dtype=np.int32)
        active_frames = [active_frames[int(index)] for index in indices.tolist()]

    mask_frame_count = 0
    bbox_only_count = 0
    for frame in active_frames:
        mask_path = frame.get("mask_path")
        has_mask = bool(mask_path) and Path(str(mask_path)).exists()
        if has_mask:
            mask_frame_count += 1
        else:
            bbox_only_count += 1

    other_frames_by_index: dict[int, list[dict[str, Any]]] = {}
    other_track_phrases: list[str] = []
    for row in payload.get("tracks", []):
        row_phrase_norm = _norm_phrase(row.get("phrase"))
        if row is track or row_phrase_norm == phrase_norm:
            continue
        other_track_phrases.append(str(row.get("phrase", "")))
        for frame in row.get("frames", []):
            if not bool(frame.get("active")):
                continue
            if frame.get("bbox_xyxy") is None and frame.get("mask_path") is None:
                continue
            other_frames_by_index.setdefault(int(frame.get("frame_index", 0)), []).append(frame)
    return {
        "dataset_dir": str(payload.get("dataset_dir", "")),
        "track_phrase": str(track.get("phrase", "")),
        "frames": active_frames,
        "other_frames_by_index": other_frames_by_index,
        "other_track_phrases": other_track_phrases,
    }, {
        "enabled": True,
        "query_phrase": phrase_norm,
        "track_phrase": str(track.get("phrase", "")),
        "frame_count": int(len(active_frames)),
        "mask_frame_count": int(mask_frame_count),
        "bbox_only_frame_count": int(bbox_only_count),
        "other_track_count": int(len(other_track_phrases)),
        "other_track_phrases": other_track_phrases,
        "query_tracks_path": str(query_tracks_path),
    }


def _connection_radius(points: np.ndarray, radius_scale: float, neighbor_rank: int = 8) -> float:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] <= 1:
        return max(float(np.linalg.norm(np.std(points, axis=0))) * max(float(radius_scale), 1.0) + 1.0e-4, 1.0e-4)
    k = int(min(max(int(neighbor_rank) + 1, 2), points.shape[0]))
    tree = cKDTree(points)
    distances = tree.query(points, k=k)[0]
    if distances.ndim == 1:
        neighbor = distances.reshape(-1)
    else:
        neighbor = distances[:, -1].reshape(-1)
    return max(_safe_quantile(neighbor, 0.85, float(np.max(neighbor))) * float(radius_scale), 1.0e-4)


def _connected_components_from_radius(points: np.ndarray, radius: float) -> list[np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == 0:
        return []
    tree = cKDTree(points)
    visited = np.zeros((points.shape[0],), dtype=bool)
    components: list[np.ndarray] = []
    for start_index in range(points.shape[0]):
        if visited[start_index]:
            continue
        stack = [int(start_index)]
        visited[start_index] = True
        component: list[int] = []
        while stack:
            current = int(stack.pop())
            component.append(current)
            for neighbor in tree.query_ball_point(points[current], r=float(radius)):
                neighbor = int(neighbor)
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                stack.append(neighbor)
        components.append(np.asarray(component, dtype=np.int64))
    components.sort(key=lambda row: int(row.size), reverse=True)
    return components


def _cluster_refine_ids(
    selected_ids: np.ndarray,
    ranked_pool: np.ndarray,
    traj_mean_all: np.ndarray,
    score: np.ndarray,
    feature_affinity: np.ndarray,
    spatial_affinity: np.ndarray,
    color_affinity: np.ndarray,
    pos_rate: np.ndarray,
    neg_rate: np.ndarray,
    mask_contrast: np.ndarray,
    vote_margin: np.ndarray,
    pos_vote_rate: np.ndarray,
    neg_vote_rate: np.ndarray,
    foreground_vote_mask: np.ndarray,
    background_vote_mask: np.ndarray,
    seed_ids: np.ndarray,
    target_count: int,
    radius_scale: float,
    min_component_size: int,
    min_mask_contrast: float,
    max_neg_rate: float,
    center_distance_scale: float,
    refill_radius_scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    selected_ids = np.unique(np.asarray(selected_ids, dtype=np.int64).reshape(-1))
    if selected_ids.size == 0:
        return selected_ids, {
            "enabled": False,
            "skip_reason": "selected_ids_empty",
        }
    if selected_ids.size < max(int(min_component_size), 4):
        return selected_ids, {
            "enabled": False,
            "skip_reason": "selected_ids_below_component_threshold",
            "selected_count": int(selected_ids.size),
        }

    selected_points = np.asarray(traj_mean_all[selected_ids], dtype=np.float32)
    connection_radius = _connection_radius(selected_points, radius_scale=float(radius_scale))
    components = _connected_components_from_radius(selected_points, radius=float(connection_radius))
    if not components:
        return selected_ids, {
            "enabled": False,
            "skip_reason": "no_components_found",
            "selected_count": int(selected_ids.size),
        }

    rows: list[dict[str, Any]] = []
    for component_index, component in enumerate(components):
        component_ids = selected_ids[component]
        component_points = traj_mean_all[component_ids]
        center = component_points.mean(axis=0).astype(np.float32)
        center_distance = np.linalg.norm(component_points - center[None, :], axis=1)
        local_radius = max(_safe_quantile(center_distance, 0.90, float(center_distance.max()) if center_distance.size else 0.0), 1.0e-4)
        seed_overlap_count = int(np.isin(component_ids, seed_ids).sum())
        size_score = min(float(np.log1p(component_ids.size) / np.log(128.0)), 1.0)
        seed_score = min(float(seed_overlap_count) / max(float(seed_ids.size) * 0.25, 1.0), 1.0)
        mean_mask_contrast = float(mask_contrast[component_ids].mean())
        mean_pos_rate = float(pos_rate[component_ids].mean())
        mean_neg_rate = float(neg_rate[component_ids].mean())
        mean_vote_margin = float(vote_margin[component_ids].mean())
        mean_pos_vote_rate = float(pos_vote_rate[component_ids].mean())
        mean_neg_vote_rate = float(neg_vote_rate[component_ids].mean())
        foreground_vote_ratio = float(np.asarray(foreground_vote_mask[component_ids], dtype=np.float32).mean())
        background_vote_ratio = float(np.asarray(background_vote_mask[component_ids], dtype=np.float32).mean())
        mean_feature_affinity = float(feature_affinity[component_ids].mean())
        mean_spatial_affinity = float(spatial_affinity[component_ids].mean())
        mean_color_affinity = float(color_affinity[component_ids].mean())
        mean_score = float(score[component_ids].mean())
        appearance_support = (
            0.42 * mean_feature_affinity
            + 0.33 * mean_spatial_affinity
            + 0.25 * mean_color_affinity
        )
        priority = (
            mean_mask_contrast
            + 0.18 * mean_vote_margin
            + 0.18 * mean_score
            + 0.12 * appearance_support
            + 0.10 * size_score
            + 0.08 * foreground_vote_ratio
            + 0.05 * seed_score
            - 0.12 * mean_neg_rate
            - 0.06 * background_vote_ratio
        )
        rows.append(
            {
                "component_index": int(component_index),
                "ids": component_ids,
                "size": int(component_ids.size),
                "center": center,
                "local_radius": float(local_radius),
                "seed_overlap_count": int(seed_overlap_count),
                "mean_mask_contrast": mean_mask_contrast,
                "mean_pos_rate": mean_pos_rate,
                "mean_neg_rate": mean_neg_rate,
                "mean_vote_margin": mean_vote_margin,
                "mean_pos_vote_rate": mean_pos_vote_rate,
                "mean_neg_vote_rate": mean_neg_vote_rate,
                "foreground_vote_ratio": foreground_vote_ratio,
                "background_vote_ratio": background_vote_ratio,
                "mean_feature_affinity": mean_feature_affinity,
                "mean_spatial_affinity": mean_spatial_affinity,
                "mean_color_affinity": mean_color_affinity,
                "appearance_support": float(appearance_support),
                "mean_score": mean_score,
                "priority": float(priority),
            }
        )

    anchor_candidates = [
        row
        for row in rows
        if float(row["mean_mask_contrast"]) >= float(min_mask_contrast)
        and float(row["mean_neg_rate"]) <= float(max_neg_rate)
    ]
    if anchor_candidates:
        anchor_row = max(anchor_candidates, key=lambda row: (int(row["size"]), float(row["priority"])))
    else:
        anchor_row = max(rows, key=lambda row: float(row["priority"]))
    anchor_center = np.asarray(anchor_row["center"], dtype=np.float32)
    anchor_radius = max(float(anchor_row["local_radius"]), float(connection_radius))
    anchor_feature_affinity = float(anchor_row["mean_feature_affinity"])
    anchor_spatial_affinity = float(anchor_row["mean_spatial_affinity"])
    anchor_color_affinity = float(anchor_row["mean_color_affinity"])
    anchor_score = float(anchor_row["mean_score"])

    keep_rows: list[dict[str, Any]] = []
    for row in rows:
        center_distance = float(np.linalg.norm(np.asarray(row["center"], dtype=np.float32) - anchor_center))
        row["center_distance_to_anchor"] = center_distance
        near_anchor = center_distance <= max(anchor_radius, float(row["local_radius"]), float(connection_radius)) * float(center_distance_scale)
        positive_cluster = (
            float(row["mean_mask_contrast"]) >= float(min_mask_contrast)
            and float(row["mean_neg_rate"]) <= float(max_neg_rate)
            and float(row["mean_vote_margin"]) >= max(float(min_mask_contrast) * 0.25, 0.02)
        )
        support_cluster = (
            float(row["mean_pos_rate"]) >= max(float(min_mask_contrast) * 0.8, 0.12)
            and float(row["mean_neg_rate"]) <= float(max_neg_rate)
            and float(row["foreground_vote_ratio"]) >= 0.20
        )
        appearance_cluster = (
            float(row["mean_feature_affinity"]) >= max(anchor_feature_affinity * 0.80, 0.36)
            and float(row["mean_spatial_affinity"]) >= max(anchor_spatial_affinity * 0.82, 0.24)
            and float(row["mean_color_affinity"]) >= max(anchor_color_affinity * 0.84, 0.26)
            and float(row["mean_neg_rate"]) <= float(max_neg_rate) * 1.08
            and float(row["background_vote_ratio"]) <= 0.28
            and (
                float(row["mean_vote_margin"]) >= -0.02
                or float(row["foreground_vote_ratio"]) >= 0.08
                or float(row["mean_score"]) >= anchor_score * 0.82
            )
        )
        seed_cluster = (
            int(row["seed_overlap_count"]) > 0
            and float(row["mean_mask_contrast"]) >= max(float(min_mask_contrast) * 0.40, -0.02)
            and float(row["mean_neg_rate"]) <= float(max_neg_rate) * 1.10
        )
        if row is anchor_row:
            keep_rows.append(row)
            continue
        if int(row["size"]) < int(min_component_size):
            continue
        if near_anchor and (positive_cluster or support_cluster or appearance_cluster or seed_cluster):
            keep_rows.append(row)

    if not keep_rows:
        keep_rows = [anchor_row]

    kept_ids = np.unique(np.concatenate([np.asarray(row["ids"], dtype=np.int64) for row in keep_rows], axis=0)).astype(np.int64)
    kept_points = np.asarray(traj_mean_all[kept_ids], dtype=np.float32)
    refill_radius = _connection_radius(kept_points, radius_scale=float(refill_radius_scale)) if kept_ids.size >= 2 else float(connection_radius)

    refill_target = int(target_count)
    refill_rows: list[dict[str, Any]] = []
    final_ids = kept_ids
    if refill_target > int(kept_ids.size):
        keep_set = set(int(idx) for idx in kept_ids.tolist())
        pool_candidates = np.asarray([int(idx) for idx in np.asarray(ranked_pool, dtype=np.int64).tolist() if int(idx) not in keep_set], dtype=np.int64)
        if pool_candidates.size > 0:
            keep_tree = cKDTree(kept_points)
            pool_distance = keep_tree.query(traj_mean_all[pool_candidates], k=1)[0].reshape(-1)
            kept_feature_floor = max(_safe_quantile(feature_affinity[kept_ids], 0.18, float(feature_affinity[kept_ids].mean())), 0.25)
            kept_spatial_floor = max(_safe_quantile(spatial_affinity[kept_ids], 0.14, float(spatial_affinity[kept_ids].mean())), 0.20)
            kept_color_floor = max(_safe_quantile(color_affinity[kept_ids], 0.18, float(color_affinity[kept_ids].mean())), 0.24)
            kept_score_floor = _safe_quantile(score[kept_ids], 0.12, float(score[kept_ids].min()))
            appearance_support_mask = (
                (feature_affinity[pool_candidates] >= max(kept_feature_floor * 0.92, 0.28))
                & (spatial_affinity[pool_candidates] >= max(kept_spatial_floor * 0.90, 0.18))
                & (color_affinity[pool_candidates] >= max(kept_color_floor * 0.90, 0.20))
                & (score[pool_candidates] >= kept_score_floor * 0.82)
                & (~background_vote_mask[pool_candidates])
            )
            vote_support_mask = (
                foreground_vote_mask[pool_candidates]
                | (vote_margin[pool_candidates] >= max(float(min_mask_contrast) * 0.25, 0.02))
                | (pos_vote_rate[pool_candidates] >= max(float(min_mask_contrast) * 0.55, 0.12))
            )
            candidate_mask = (
                (pool_distance <= float(refill_radius))
                & (mask_contrast[pool_candidates] >= max(float(min_mask_contrast) * 0.35, 0.0))
                & (neg_rate[pool_candidates] <= float(max_neg_rate) * 1.15)
                & (vote_support_mask | appearance_support_mask)
                & (neg_vote_rate[pool_candidates] <= float(max_neg_rate) * 1.05)
            )
            refill_candidates = pool_candidates[candidate_mask]
            if refill_candidates.size > 0:
                refill_ranked = refill_candidates[np.argsort(-score[refill_candidates], kind="mergesort")]
                refill_needed = max(0, int(refill_target - kept_ids.size))
                refill_ids = refill_ranked[:refill_needed].astype(np.int64)
                if refill_ids.size > 0:
                    final_ids = np.unique(np.concatenate([kept_ids, refill_ids], axis=0)).astype(np.int64)
                    refill_rows = [
                        {
                            "candidate_count": int(refill_candidates.size),
                            "added_count": int(final_ids.size - kept_ids.size),
                            "refill_radius": float(refill_radius),
                            "appearance_support_candidate_count": int(np.asarray(appearance_support_mask[candidate_mask], dtype=np.int32).sum()),
                            "vote_support_candidate_count": int(np.asarray(vote_support_mask[candidate_mask], dtype=np.int32).sum()),
                            "mean_added_mask_contrast": float(mask_contrast[refill_ids].mean()),
                            "mean_added_neg_rate": float(neg_rate[refill_ids].mean()),
                        }
                    ]

    final_ids = final_ids[np.argsort(-score[final_ids], kind="mergesort")]
    component_rows = []
    for row in rows[:12]:
        component_rows.append(
            {
                "component_index": int(row["component_index"]),
                "size": int(row["size"]),
                "seed_overlap_count": int(row["seed_overlap_count"]),
                "mean_mask_contrast": float(row["mean_mask_contrast"]),
                "mean_pos_rate": float(row["mean_pos_rate"]),
                "mean_neg_rate": float(row["mean_neg_rate"]),
                "mean_vote_margin": float(row["mean_vote_margin"]),
                "mean_pos_vote_rate": float(row["mean_pos_vote_rate"]),
                "mean_neg_vote_rate": float(row["mean_neg_vote_rate"]),
                "foreground_vote_ratio": float(row["foreground_vote_ratio"]),
                "background_vote_ratio": float(row["background_vote_ratio"]),
                "mean_feature_affinity": float(row["mean_feature_affinity"]),
                "mean_spatial_affinity": float(row["mean_spatial_affinity"]),
                "mean_color_affinity": float(row["mean_color_affinity"]),
                "appearance_support": float(row["appearance_support"]),
                "mean_score": float(row["mean_score"]),
                "priority": float(row["priority"]),
                "center_distance_to_anchor": float(row.get("center_distance_to_anchor", 0.0)),
                "center_xyz": np.asarray(row["center"], dtype=np.float32).round(4).tolist(),
            }
        )
    return final_ids.astype(np.int64), {
        "enabled": True,
        "connection_radius": float(connection_radius),
        "refill_radius": float(refill_radius),
        "selected_count_before_cluster": int(selected_ids.size),
        "component_count": int(len(rows)),
        "kept_component_count": int(len(keep_rows)),
        "kept_count_before_refill": int(kept_ids.size),
        "final_count": int(final_ids.size),
        "target_count": int(refill_target),
        "components": component_rows,
        "refill_rows": refill_rows,
    }


def _mask_guided_select_ids(
    payload: Any,
    trajectories: np.ndarray,
    gate: np.ndarray,
    rgb: np.ndarray,
    opacity: np.ndarray,
    seed_ids: np.ndarray,
    current_ids: np.ndarray,
    sample_start: int,
    sample_end: int,
    run_dir: Path,
    query_tracks_path: Path | None,
    query_phrase: str,
    mask_guided_target_count: int,
    mask_guided_max_frames: int,
    mask_guided_dilation_radius: int,
    mask_guided_negative_margin: float,
    mask_guided_feature_maha_factor: float,
    mask_guided_spatial_maha_factor: float,
    mask_guided_color_scale: float,
    mask_guided_vote_weight: float,
    mask_guided_vote_threshold: float,
    mask_guided_background_margin: float,
    mask_guided_min_pos_votes: int,
    cluster_refine: bool,
    cluster_refine_target_count: int,
    cluster_refine_radius_scale: float,
    cluster_refine_min_component_size: int,
    cluster_refine_min_mask_contrast: float,
    cluster_refine_max_neg_rate: float,
    cluster_refine_center_distance_scale: float,
    cluster_refine_refill_radius_scale: float,
    mask_guided_prune_background_selected: bool,
    mask_guided_prune_max_drop_ratio: float,
    mask_guided_prune_neg_margin: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_count = int(mask_guided_target_count)
    if target_count <= 0:
        return np.asarray(current_ids, dtype=np.int64), {
            "enabled": False,
            "skip_reason": "target_count_non_positive",
        }

    track_payload, track_stats = _load_query_track_frames(
        query_tracks_path=query_tracks_path,
        query_phrase=query_phrase,
        max_frames=int(mask_guided_max_frames),
    )
    if track_payload is None:
        return np.asarray(current_ids, dtype=np.int64), track_stats

    dataset_dir = Path(str(track_payload.get("dataset_dir", "")))
    if not dataset_dir.exists():
        return np.asarray(current_ids, dtype=np.int64), {
            **track_stats,
            "enabled": False,
            "skip_reason": "dataset_dir_missing",
        }

    feature_matrix, aux = _feature_matrix(
        payload=payload,
        trajectories=trajectories,
        gate=gate,
        sample_start=sample_start,
        sample_end=sample_end,
        rgb=rgb,
        opacity=opacity,
    )
    traj_mean_all = np.asarray(aux["traj_mean"], dtype=np.float32)
    feature_maha_all, feature_maha_seed = _regularized_mahalanobis(feature_matrix, seed_ids)
    spatial_maha_all, spatial_maha_seed = _regularized_mahalanobis(traj_mean_all, seed_ids)
    feature_limit = max(_safe_quantile(feature_maha_seed, 0.985, 1.0) * float(mask_guided_feature_maha_factor), 1.0e-4)
    spatial_limit = max(_safe_quantile(spatial_maha_seed, 0.985, 1.0) * float(mask_guided_spatial_maha_factor), 1.0e-4)
    feature_affinity = np.exp(-0.5 * np.square(feature_maha_all / feature_limit)).astype(np.float32)
    spatial_affinity = np.exp(-0.5 * np.square(spatial_maha_all / spatial_limit)).astype(np.float32)
    seed_rgb = rgb[seed_ids].mean(axis=0, keepdims=True).astype(np.float32)
    color_dist = np.linalg.norm(rgb - seed_rgb, axis=1)
    seed_color_dist = color_dist[seed_ids]
    color_limit = max(_safe_quantile(seed_color_dist, 0.985, 0.35) * float(mask_guided_color_scale), 0.15)
    color_affinity = np.exp(-0.5 * np.square(color_dist / color_limit)).astype(np.float32)

    time_values = np.asarray(payload["time_values"], dtype=np.float32).reshape(-1)
    frame_times = np.asarray([float(frame.get("time_value", 0.0)) for frame in track_payload["frames"]], dtype=np.float32)
    sample_indices = _resample_indices(source_times=time_values, target_times=frame_times)
    total_gaussians = int(trajectories.shape[0])
    pos_support = np.zeros((total_gaussians,), dtype=np.float32)
    neg_support = np.zeros((total_gaussians,), dtype=np.float32)
    bbox_support = np.zeros((total_gaussians,), dtype=np.float32)
    visible_support = np.zeros((total_gaussians,), dtype=np.float32)
    mask_hit_count = np.zeros((total_gaussians,), dtype=np.float32)
    visible_vote_count = np.zeros((total_gaussians,), dtype=np.float32)
    pos_vote_count = np.zeros((total_gaussians,), dtype=np.float32)
    neg_vote_count = np.zeros((total_gaussians,), dtype=np.float32)
    bbox_vote_count = np.zeros((total_gaussians,), dtype=np.float32)
    frame_rows: list[dict[str, Any]] = []
    other_negative_frame_count = 0

    for frame, sample_index in zip(track_payload["frames"], sample_indices.tolist()):
        camera_path = dataset_dir / "camera" / f"{frame['image_id']}.json"
        if not camera_path.exists():
            continue
        camera = Camera.from_json(camera_path)
        mask_path = frame.get("mask_path")
        mask_array = None if not mask_path else _load_binary_mask(Path(str(mask_path)))
        if mask_array is not None:
            image_size = (int(mask_array.shape[1]), int(mask_array.shape[0]))
        else:
            camera_size = np.asarray(camera.image_size, dtype=np.int32).reshape(-1)
            if camera_size.size < 2:
                continue
            image_size = (int(camera_size[0]), int(camera_size[1]))
        positive_mask, negative_mask, mask_meta = _frame_masks(
            frame=frame,
            image_size=image_size,
            dilation_radius=int(mask_guided_dilation_radius),
            negative_margin=float(mask_guided_negative_margin),
        )
        if positive_mask is None:
            continue

        other_negative_mask = None
        other_frames = track_payload.get("other_frames_by_index", {}).get(int(frame.get("frame_index", 0)), [])
        for other_frame in other_frames:
            other_positive_mask, _ = _positive_mask_from_frame(frame=other_frame, image_size=image_size)
            if other_positive_mask is None:
                continue
            dilated_other = ndimage.binary_dilation(
                other_positive_mask,
                iterations=max(int(mask_guided_dilation_radius), 1),
            )
            dilated_other = np.logical_and(dilated_other, np.logical_not(positive_mask))
            if not np.any(dilated_other):
                continue
            other_negative_mask = dilated_other if other_negative_mask is None else np.logical_or(other_negative_mask, dilated_other)
        if other_negative_mask is not None:
            negative_mask = other_negative_mask if negative_mask is None else np.logical_or(negative_mask, other_negative_mask)
            other_negative_frame_count += 1

        points_world = np.asarray(trajectories[:, int(sample_index), :], dtype=np.float32)
        points_local = camera.points_to_local_points(points_world)
        projected = _project_points_to_image(camera, points_world, image_size=image_size)
        width, height = int(image_size[0]), int(image_size[1])
        valid = (
            (points_local[:, 2] > 1.0e-4)
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] < float(width))
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] < float(height))
        )
        if not np.any(valid):
            continue

        valid_ids = np.where(valid)[0]
        px = np.clip(np.round(projected[valid_ids, 0]).astype(np.int32), 0, width - 1)
        py = np.clip(np.round(projected[valid_ids, 1]).astype(np.int32), 0, height - 1)
        frame_gate = gate[:, int(sample_index)].reshape(-1).astype(np.float32)
        valid_gate = frame_gate[valid_ids]
        visible_support[valid_ids] += valid_gate

        pos_hits = np.asarray(positive_mask[py, px], dtype=np.float32)
        neg_hits = np.asarray(negative_mask[py, px], dtype=np.float32) if negative_mask is not None else np.zeros_like(pos_hits)
        bbox_hits = np.zeros_like(pos_hits)
        bbox_xyxy = frame.get("bbox_xyxy")
        if bbox_xyxy is not None:
            left, top, right, bottom = [float(v) for v in bbox_xyxy]
            bbox_hits = (
                (projected[valid_ids, 0] >= left)
                & (projected[valid_ids, 0] <= right)
                & (projected[valid_ids, 1] >= top)
                & (projected[valid_ids, 1] <= bottom)
            ).astype(np.float32)

        pos_support[valid_ids] += valid_gate * pos_hits
        neg_support[valid_ids] += valid_gate * neg_hits
        bbox_support[valid_ids] += valid_gate * bbox_hits
        mask_hit_count[valid_ids] += pos_hits
        visible_vote_count[valid_ids] += 1.0
        pos_vote_count[valid_ids] += pos_hits
        neg_vote_count[valid_ids] += neg_hits
        bbox_vote_count[valid_ids] += bbox_hits
        frame_rows.append(
            {
                "frame_index": int(frame["frame_index"]),
                "sample_index": int(sample_index),
                "mask_source": mask_meta["mask_source"],
                "bbox_xyxy": mask_meta["bbox_xyxy"],
                "other_phrase_count": int(len(other_frames)),
                "has_other_phrase_negative": bool(other_negative_mask is not None),
            }
        )

    if not frame_rows:
        return np.asarray(current_ids, dtype=np.int64), {
            **track_stats,
            "enabled": False,
            "skip_reason": "no_projectable_mask_frames",
        }

    current_mask = np.zeros((total_gaussians,), dtype=np.float32)
    current_mask[np.asarray(current_ids, dtype=np.int64)] = 1.0
    seed_mask = np.zeros((total_gaussians,), dtype=np.float32)
    seed_mask[np.asarray(seed_ids, dtype=np.int64)] = 1.0
    denom = np.clip(visible_support, 1.0e-6, None)
    pos_rate = pos_support / denom
    neg_rate = neg_support / denom
    bbox_rate = bbox_support / denom
    vote_denom = np.clip(visible_vote_count, 1.0, None)
    pos_vote_rate = pos_vote_count / vote_denom
    neg_vote_rate = neg_vote_count / vote_denom
    bbox_vote_rate = bbox_vote_count / vote_denom
    visibility_ratio = np.clip(
        visible_support / max(float(np.quantile(visible_support[np.asarray(seed_ids, dtype=np.int64)], 0.75)), 1.0e-4),
        0.0,
        1.0,
    ).astype(np.float32)
    opacity_affinity = np.clip(
        (opacity - float(np.quantile(opacity[np.asarray(seed_ids, dtype=np.int64)], 0.10)))
        / max(float(np.quantile(opacity[np.asarray(seed_ids, dtype=np.int64)], 0.85)) - float(np.quantile(opacity[np.asarray(seed_ids, dtype=np.int64)], 0.10)), 1.0e-4),
        0.0,
        1.0,
    ).astype(np.float32)
    vote_margin = np.clip(
        pos_vote_rate + 0.20 * bbox_vote_rate - 0.90 * neg_vote_rate,
        -1.0,
        1.0,
    ).astype(np.float32)
    vote_threshold = float(mask_guided_vote_threshold)
    background_margin = float(mask_guided_background_margin)
    min_pos_votes = max(int(mask_guided_min_pos_votes), 1)
    vote_affinity = np.clip((vote_margin - vote_threshold + 0.35) / 1.35, 0.0, 1.0).astype(np.float32)
    background_vote_penalty = np.clip(
        (neg_vote_rate - pos_vote_rate - background_margin) / max(1.0 - background_margin, 1.0e-6),
        0.0,
        1.0,
    ).astype(np.float32)
    foreground_vote_mask = (
        ((pos_vote_count >= float(min_pos_votes)) & (vote_margin >= vote_threshold * 0.70))
        | ((pos_vote_rate >= vote_threshold) & (pos_vote_count >= 1.0))
        | ((bbox_vote_rate >= max(vote_threshold * 0.85, 0.18)) & (pos_vote_rate >= neg_vote_rate))
    )
    background_vote_mask = (
        (neg_vote_count >= max(float(min_pos_votes), 2.0))
        & (neg_vote_rate >= pos_vote_rate + background_margin)
        & (neg_vote_rate >= max(vote_threshold * 0.8, 0.12))
    )
    mask_contrast = np.clip(1.35 * pos_rate + 0.30 * bbox_rate - 0.95 * neg_rate, -1.0, 1.5).astype(np.float32)
    score = (
        0.40 * mask_contrast
        + 0.20 * feature_affinity
        + 0.14 * spatial_affinity
        + 0.08 * color_affinity
        + 0.06 * visibility_ratio
        + 0.05 * opacity_affinity
        + 0.04 * current_mask
        + 0.03 * seed_mask
        + float(mask_guided_vote_weight) * vote_affinity
        - 0.08 * background_vote_penalty
    ).astype(np.float32)
    pool_mask = (
        (visible_support > 0.0)
        & (
            (pos_support > 0.0)
            | (bbox_support > 0.0)
            | (feature_affinity >= 0.12)
            | (spatial_affinity >= 0.10)
            | (current_mask > 0.0)
            | (seed_mask > 0.0)
            | foreground_vote_mask
            | (vote_margin >= vote_threshold * 0.80)
        )
    )
    pool_mask = np.logical_and(
        pool_mask,
        np.logical_not(background_vote_mask) | (current_mask > 0.0) | (seed_mask > 0.0) | foreground_vote_mask,
    )
    if int(pool_mask.sum()) < int(target_count):
        pool_mask = (
            ((visible_support > 0.0) & np.logical_not(background_vote_mask))
            | foreground_vote_mask
            | (current_mask > 0.0)
            | (seed_mask > 0.0)
        )
    if int(pool_mask.sum()) < int(target_count):
        pool_mask = np.ones((total_gaussians,), dtype=bool)
    pool_ids = np.where(pool_mask)[0].astype(np.int64)
    ranked_pool = pool_ids[np.argsort(-score[pool_ids], kind="mergesort")]
    target_count = int(min(max(target_count, seed_ids.size), ranked_pool.size))
    seed_ids = np.unique(np.asarray(seed_ids, dtype=np.int64))
    visible_seed_mask = visible_support[seed_ids] > 0.0
    seed_foreground_mask = np.asarray(foreground_vote_mask[seed_ids], dtype=bool)
    seed_background_mask = (
        np.asarray(background_vote_mask[seed_ids], dtype=bool)
        & (~seed_foreground_mask)
        & (neg_rate[seed_ids] >= max(float(mask_guided_negative_margin) * 0.80, 0.12))
    )
    positive_seed_rescue = (
        (mask_contrast[seed_ids] >= 0.06)
        | (vote_margin[seed_ids] >= vote_threshold * 0.75)
        | (pos_rate[seed_ids] >= max(float(np.quantile(pos_rate[seed_ids], 0.45)), 0.05))
    )
    force_seed_mask = (
        (visible_seed_mask & (mask_contrast[seed_ids] >= 0.02))
        | (visible_seed_mask & (vote_margin[seed_ids] >= vote_threshold * 0.50))
        | (visible_seed_mask & (pos_rate[seed_ids] >= max(float(np.quantile(pos_rate[seed_ids], 0.35)), 0.03)))
        | (~visible_seed_mask & (score[seed_ids] >= float(np.quantile(score[seed_ids], 0.35))))
    )
    force_seed_mask = force_seed_mask & ((~seed_background_mask) | positive_seed_rescue)
    forced_ids = seed_ids[force_seed_mask]
    minimum_forced = int(min(seed_ids.size, max(32, int(np.ceil(seed_ids.size * 0.18)))))
    if forced_ids.size < minimum_forced:
        preferred_seed_mask = (~seed_background_mask) | positive_seed_rescue
        preferred_seed_ids = seed_ids[preferred_seed_mask]
        fallback_seed_ids = seed_ids[~preferred_seed_mask]
        preferred_seed_ranked = preferred_seed_ids[np.argsort(-score[preferred_seed_ids], kind="mergesort")] if preferred_seed_ids.size else np.zeros((0,), dtype=np.int64)
        fallback_seed_ranked = fallback_seed_ids[np.argsort(-score[fallback_seed_ids], kind="mergesort")] if fallback_seed_ids.size else np.zeros((0,), dtype=np.int64)
        seed_ranked = np.concatenate([preferred_seed_ranked, fallback_seed_ranked], axis=0)
        forced_ids = np.unique(np.concatenate([forced_ids, seed_ranked[:minimum_forced]], axis=0))
    selected_ids = forced_ids.tolist()
    selected_set = set(int(idx) for idx in forced_ids.tolist())
    for idx in ranked_pool.tolist():
        idx = int(idx)
        if idx in selected_set:
            continue
        selected_ids.append(idx)
        selected_set.add(idx)
        if len(selected_ids) >= target_count:
            break
    selected = np.asarray(selected_ids[:target_count], dtype=np.int64)
    selected = selected[np.argsort(-score[selected], kind="mergesort")]
    cluster_refine_stats = {
        "enabled": bool(cluster_refine),
        "skip_reason": "disabled",
    }
    final_selected = selected.astype(np.int64)
    if bool(cluster_refine):
        final_selected, cluster_refine_stats = _cluster_refine_ids(
            selected_ids=selected,
            ranked_pool=ranked_pool,
            traj_mean_all=traj_mean_all,
            score=score,
            feature_affinity=feature_affinity,
            spatial_affinity=spatial_affinity,
            color_affinity=color_affinity,
            pos_rate=pos_rate,
            neg_rate=neg_rate,
            mask_contrast=mask_contrast,
            vote_margin=vote_margin,
            pos_vote_rate=pos_vote_rate,
            neg_vote_rate=neg_vote_rate,
            foreground_vote_mask=foreground_vote_mask,
            background_vote_mask=background_vote_mask,
            seed_ids=seed_ids,
            target_count=int(cluster_refine_target_count) if int(cluster_refine_target_count) > 0 else int(selected.size),
            radius_scale=float(cluster_refine_radius_scale),
            min_component_size=int(cluster_refine_min_component_size),
            min_mask_contrast=float(cluster_refine_min_mask_contrast),
            max_neg_rate=float(cluster_refine_max_neg_rate),
            center_distance_scale=float(cluster_refine_center_distance_scale),
            refill_radius_scale=float(cluster_refine_refill_radius_scale),
        )
    prune_background_stats: dict[str, Any] = {
        "enabled": bool(mask_guided_prune_background_selected),
        "applied": False,
        "candidate_count": 0,
        "dropped_count": 0,
        "max_drop_ratio": float(mask_guided_prune_max_drop_ratio),
        "neg_margin": float(mask_guided_prune_neg_margin),
    }
    if bool(mask_guided_prune_background_selected) and final_selected.size > 0:
        final_selected = np.unique(np.asarray(final_selected, dtype=np.int64))
        selected_fg_mask = np.asarray(foreground_vote_mask[final_selected], dtype=bool)
        selected_bg_mask = np.asarray(background_vote_mask[final_selected], dtype=bool)
        selected_seed_mask = np.isin(final_selected, seed_ids)
        selected_neg_margin_mask = (
            neg_vote_rate[final_selected]
            >= (pos_vote_rate[final_selected] + float(mask_guided_prune_neg_margin))
        )
        prune_mask = (
            (~selected_seed_mask)
            & (selected_bg_mask | selected_neg_margin_mask)
        )
        prune_candidates = final_selected[prune_mask]
        max_drop = int(
            np.floor(
                float(final_selected.size)
                * float(np.clip(mask_guided_prune_max_drop_ratio, 0.0, 0.95))
            )
        )
        if max_drop > 0 and prune_candidates.size > 0:
            drop_ids = prune_candidates
            if drop_ids.size > max_drop:
                candidate_penalty = (
                    background_vote_penalty[drop_ids]
                    + np.clip(neg_vote_rate[drop_ids] - pos_vote_rate[drop_ids], 0.0, 1.0)
                )
                ranked = np.argsort(-candidate_penalty, kind="mergesort")
                drop_ids = drop_ids[ranked[:max_drop]]
            keep_mask = ~np.isin(final_selected, drop_ids)
            final_selected = final_selected[keep_mask]
            if final_selected.size == 0:
                final_selected = np.unique(seed_ids).astype(np.int64)
            final_selected = final_selected[np.argsort(-score[final_selected], kind="mergesort")]
            prune_background_stats = {
                **prune_background_stats,
                "applied": True,
                "candidate_count": int(prune_candidates.size),
                "dropped_count": int(drop_ids.size),
                "dropped_ratio": float(drop_ids.size / max(prune_candidates.size, 1)),
                "remaining_after_prune": int(final_selected.size),
            }
        else:
            prune_background_stats = {
                **prune_background_stats,
                "candidate_count": int(prune_candidates.size),
                "dropped_count": 0,
                "remaining_after_prune": int(final_selected.size),
            }
    return final_selected.astype(np.int64), {
        **track_stats,
        "enabled": True,
        "target_count": int(mask_guided_target_count),
        "selected_count": int(final_selected.size),
        "selected_count_before_cluster": int(selected.size),
        "pool_count": int(pool_ids.size),
        "frame_rows": frame_rows,
        "forced_seed_count": int(forced_ids.size),
        "pruned_seed_count": int(seed_ids.size - forced_ids.size),
        "other_negative_frame_count": int(other_negative_frame_count),
        "vote_weight": float(mask_guided_vote_weight),
        "vote_threshold": float(mask_guided_vote_threshold),
        "background_vote_margin": float(mask_guided_background_margin),
        "min_pos_votes": int(min_pos_votes),
        "foreground_vote_pool_count": int(np.asarray(foreground_vote_mask, dtype=np.int32).sum()),
        "background_vote_pool_count": int(np.asarray(background_vote_mask, dtype=np.int32).sum()),
        "mean_selected_pos_rate": float(pos_rate[final_selected].mean()) if final_selected.size else 0.0,
        "mean_selected_neg_rate": float(neg_rate[final_selected].mean()) if final_selected.size else 0.0,
        "mean_selected_mask_contrast": float(mask_contrast[final_selected].mean()) if final_selected.size else 0.0,
        "mean_selected_vote_margin": float(vote_margin[final_selected].mean()) if final_selected.size else 0.0,
        "mean_selected_pos_vote_rate": float(pos_vote_rate[final_selected].mean()) if final_selected.size else 0.0,
        "mean_selected_neg_vote_rate": float(neg_vote_rate[final_selected].mean()) if final_selected.size else 0.0,
        "foreground_vote_selected_ratio": float(np.asarray(foreground_vote_mask[final_selected], dtype=np.float32).mean()) if final_selected.size else 0.0,
        "background_vote_selected_ratio": float(np.asarray(background_vote_mask[final_selected], dtype=np.float32).mean()) if final_selected.size else 0.0,
        "mean_selected_feature_affinity": float(feature_affinity[final_selected].mean()) if final_selected.size else 0.0,
        "mean_selected_spatial_affinity": float(spatial_affinity[final_selected].mean()) if final_selected.size else 0.0,
        "score_threshold_at_k": float(score[final_selected[-1]]) if final_selected.size else 0.0,
        "cluster_refine": cluster_refine_stats,
        "prune_background": prune_background_stats,
    }


def _load_seed_selection(
    proposal_dir: Path,
    proposal_alias: str | None,
    proposal_entity_id: int | None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any] | None]:
    if proposal_alias is None and proposal_entity_id is None:
        raise ValueError("proposal_alias or proposal_entity_id is required")

    entities_payload = _read_json(proposal_dir / "entities.json")
    summary_path = proposal_dir / "query_proposal_summary.json"
    summary_payload = _read_json(summary_path) if summary_path.exists() else None
    summary_by_id = {
        int(row.get("id", -1)): row
        for row in (summary_payload or {}).get("phrases", [])
        if int(row.get("id", -1)) >= 0
    }
    summary_by_alias = {
        str(row.get("proposal_alias", "")).strip(): row
        for row in (summary_payload or {}).get("phrases", [])
        if str(row.get("proposal_alias", "")).strip()
    }
    summary_by_phrase = {
        str(row.get("phrase", "")).strip(): row
        for row in (summary_payload or {}).get("phrases", [])
        if str(row.get("phrase", "")).strip()
    }

    matched_entity = None
    alias_norm = None if proposal_alias is None else str(proposal_alias).strip()
    for entity in entities_payload.get("entities", []):
        entity_id = int(entity.get("id", -1))
        alias = str(entity.get("proposal_alias", "")).strip()
        static_text = str(entity.get("static_text", "")).strip()
        if proposal_entity_id is not None and entity_id == int(proposal_entity_id):
            matched_entity = entity
            break
        if alias_norm is not None and alias == alias_norm:
            matched_entity = entity
            break
        if alias_norm is not None and static_text == alias_norm:
            matched_entity = entity
            break
    if matched_entity is None:
        if proposal_entity_id is not None:
            raise KeyError(f"proposal_entity_id={proposal_entity_id} not found in {proposal_dir / 'entities.json'}")
        raise KeyError(f"proposal_alias='{proposal_alias}' not found in {proposal_dir / 'entities.json'}")

    gaussian_ids = np.asarray(matched_entity.get("gaussian_ids", []), dtype=np.int64).reshape(-1)
    if gaussian_ids.size == 0:
        raise ValueError("Matched seed entity has no gaussian_ids")
    summary_row = summary_by_id.get(int(matched_entity["id"]))
    if summary_row is None and str(matched_entity.get("proposal_alias", "")).strip():
        summary_row = summary_by_alias.get(str(matched_entity.get("proposal_alias", "")).strip())
    if summary_row is None and str(matched_entity.get("static_text", "")).strip():
        summary_row = summary_by_phrase.get(str(matched_entity.get("static_text", "")).strip())
    return gaussian_ids, matched_entity, summary_row


def _resolve_sample_window(
    summary_row: dict[str, Any] | None,
    requested_start: int | None,
    requested_end: int | None,
    total_samples: int,
) -> tuple[int, int, dict[str, Any]]:
    if requested_start is not None or requested_end is not None:
        start_index = int(0 if requested_start is None else requested_start)
        end_index = int(total_samples if requested_end is None else requested_end)
        start_index = int(np.clip(start_index, 0, max(total_samples - 1, 0)))
        end_index = int(np.clip(end_index, start_index + 1, total_samples))
        return start_index, end_index, {
            "mode": "explicit",
            "requested_start_index": None if requested_start is None else int(requested_start),
            "requested_end_index": None if requested_end is None else int(requested_end),
        }

    keyframes = []
    if summary_row is not None:
        keyframes = [int(index) for index in summary_row.get("keyframes", [])]
    if keyframes:
        start_index = max(min(keyframes) - 2, 0)
        end_index = min(max(keyframes) + 3, total_samples)
        if end_index <= start_index:
            end_index = min(start_index + 1, total_samples)
        return start_index, end_index, {
            "mode": "keyframes",
            "keyframes": keyframes,
        }

    fallback_end = int(min(max(18, 1), total_samples))
    return 0, max(fallback_end, 1), {
        "mode": "fallback_first_window",
        "fallback_end_index": int(fallback_end),
    }


def _feature_matrix(
    payload: Any,
    trajectories: np.ndarray,
    gate: np.ndarray,
    sample_start: int,
    sample_end: int,
    rgb: np.ndarray,
    opacity: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    window_traj = np.asarray(trajectories[:, sample_start:sample_end, :], dtype=np.float32)
    window_gate = np.asarray(gate[:, sample_start:sample_end], dtype=np.float32)
    sample_times = np.asarray(payload["time_values"][sample_start:sample_end], dtype=np.float32).reshape(1, -1)
    gate_sum = np.clip(window_gate.sum(axis=1, keepdims=True), 1.0e-6, None)
    gate_peak = window_gate.max(axis=1, keepdims=True)
    active_mask = window_gate >= np.maximum(0.18, 0.35 * gate_peak)
    support_center = ((window_gate * sample_times).sum(axis=1, keepdims=True) / gate_sum).astype(np.float32)
    active_ratio = active_mask.mean(axis=1, keepdims=True).astype(np.float32)
    mean_gate = window_gate.mean(axis=1, keepdims=True).astype(np.float32)
    traj_mean = window_traj.mean(axis=1).astype(np.float32)
    displacement = np.asarray(payload.get("displacement", np.zeros_like(traj_mean)), dtype=np.float32)
    velocity = np.asarray(payload.get("velocity", np.zeros_like(traj_mean)), dtype=np.float32)
    acceleration = np.asarray(payload.get("acceleration", np.zeros_like(traj_mean)), dtype=np.float32)
    xyz = np.asarray(payload.get("xyz", trajectories[:, sample_start, :]), dtype=np.float32)
    spatial_scale = np.asarray(payload.get("spatial_scale", np.ones_like(traj_mean)), dtype=np.float32)
    path_length = np.asarray(payload.get("path_length", np.zeros((trajectories.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1, 1)
    motion_score = np.asarray(payload.get("motion_score", np.zeros((trajectories.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1, 1)
    occupancy_mass = np.asarray(payload.get("occupancy_mass", np.ones((trajectories.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1, 1)
    visibility_proxy = np.asarray(payload.get("visibility_proxy", np.ones((trajectories.shape[0], 1), dtype=np.float32)), dtype=np.float32).reshape(-1, 1)
    support_factor = np.asarray(payload.get("support_factor", np.ones((trajectories.shape[0], 1), dtype=np.float32)), dtype=np.float32).reshape(-1, 1)
    effective_support = np.asarray(payload.get("effective_support", np.ones((trajectories.shape[0], 1), dtype=np.float32)), dtype=np.float32).reshape(-1, 1)
    tube_ratio = np.asarray(payload.get("tube_ratio", np.ones((trajectories.shape[0], 1), dtype=np.float32)), dtype=np.float32).reshape(-1, 1)
    anchor = np.asarray(payload.get("anchor", np.zeros((trajectories.shape[0], 1), dtype=np.float32)), dtype=np.float32).reshape(-1, 1)
    scale = np.asarray(payload.get("scale", np.ones((trajectories.shape[0], 1), dtype=np.float32)), dtype=np.float32).reshape(-1, 1)

    feature_parts = [
        traj_mean,
        xyz,
        displacement,
        velocity,
        acceleration,
        spatial_scale,
        path_length,
        motion_score,
        occupancy_mass,
        visibility_proxy,
        support_factor,
        effective_support,
        tube_ratio,
        anchor,
        scale,
        mean_gate,
        active_ratio,
        support_center,
        opacity.reshape(-1, 1).astype(np.float32),
        rgb.astype(np.float32),
    ]
    features = np.concatenate(feature_parts, axis=1).astype(np.float32)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    features_norm = ((features - mean) / np.clip(std, 1.0e-6, None)).astype(np.float32)
    aux = {
        "traj_mean": traj_mean,
        "mean_gate": mean_gate.reshape(-1),
        "active_ratio": active_ratio.reshape(-1),
        "support_center": support_center.reshape(-1),
        "path_length": path_length.reshape(-1),
    }
    return features_norm, aux


def _deep_fill_ids(
    current_ids: np.ndarray,
    payload: Any,
    trajectories: np.ndarray,
    gate: np.ndarray,
    sample_start: int,
    sample_end: int,
    rgb: np.ndarray,
    opacity: np.ndarray,
    color_threshold: float,
    deep_fill_rounds: int,
    deep_fill_knn_factor: float,
    deep_fill_feature_knn_factor: float,
    deep_fill_feature_maha_factor: float,
    deep_fill_spatial_maha_factor: float,
    deep_fill_color_scale: float,
    deep_fill_path_scale: float,
    deep_fill_gate_quantile: float,
    deep_fill_opacity_min: float,
    deep_fill_score_threshold: float,
    deep_fill_max_round_growth: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidate_ids = np.asarray(current_ids, dtype=np.int64).reshape(-1)
    if candidate_ids.size == 0 or int(deep_fill_rounds) <= 0:
        return candidate_ids, {
            "enabled": False,
            "rounds": [],
            "added_count": 0,
        }

    feature_matrix, aux = _feature_matrix(
        payload=payload,
        trajectories=trajectories,
        gate=gate,
        sample_start=sample_start,
        sample_end=sample_end,
        rgb=rgb,
        opacity=opacity,
    )
    traj_mean_all = aux["traj_mean"]
    mean_gate_all = aux["mean_gate"]
    active_ratio_all = aux["active_ratio"]
    support_center_all = aux["support_center"]
    path_length_all = aux["path_length"]

    round_rows: list[dict[str, Any]] = []
    total_added = 0
    for round_index in range(int(deep_fill_rounds)):
        stage_ids = np.unique(candidate_ids).astype(np.int64)
        if stage_ids.size == 0:
            break

        stage_features = feature_matrix[stage_ids]
        stage_traj = traj_mean_all[stage_ids]
        stage_gate = mean_gate_all[stage_ids]
        stage_active_ratio = active_ratio_all[stage_ids]
        stage_support_center = support_center_all[stage_ids]
        stage_path = path_length_all[stage_ids]
        stage_color_all = np.linalg.norm(rgb - rgb[stage_ids].mean(axis=0, keepdims=True), axis=1)
        stage_color_seed = stage_color_all[stage_ids]

        feature_tree = cKDTree(stage_features)
        traj_tree = cKDTree(stage_traj)
        if stage_ids.size >= 2:
            feature_nn = feature_tree.query(stage_features, k=2)[0][:, 1]
            spatial_nn = traj_tree.query(stage_traj, k=2)[0][:, 1]
        else:
            feature_nn = np.asarray([np.linalg.norm(np.std(stage_features, axis=0)) + 1.0e-4], dtype=np.float32)
            spatial_nn = np.asarray([np.linalg.norm(np.std(stage_traj, axis=0)) + 1.0e-4], dtype=np.float32)

        feature_radius = max(_safe_quantile(feature_nn, 0.95, 1.0e-4) * float(deep_fill_feature_knn_factor), 1.0e-4)
        spatial_radius = max(_safe_quantile(spatial_nn, 0.95, 1.0e-4) * float(deep_fill_knn_factor), 1.0e-4)
        feature_nearest = feature_tree.query(feature_matrix, k=1)[0]
        spatial_nearest = traj_tree.query(traj_mean_all, k=1)[0]

        feature_maha_all, feature_maha_stage = _regularized_mahalanobis(feature_matrix, stage_ids)
        spatial_maha_all, spatial_maha_stage = _regularized_mahalanobis(traj_mean_all, stage_ids)
        feature_maha_limit = max(_safe_quantile(feature_maha_stage, 0.98, 1.0) * float(deep_fill_feature_maha_factor), 1.0e-4)
        spatial_maha_limit = max(_safe_quantile(spatial_maha_stage, 0.98, 1.0) * float(deep_fill_spatial_maha_factor), 1.0e-4)
        gate_floor = _safe_quantile(stage_gate, float(deep_fill_gate_quantile), float(stage_gate.min()))
        active_ratio_floor = _safe_quantile(stage_active_ratio, 0.05, float(stage_active_ratio.min()))
        support_center_limit = max(float(np.std(stage_support_center)) * 2.5 + 0.03, 0.03)
        path_limit = max(_safe_quantile(stage_path, 0.99, float(stage_path.max())) * float(deep_fill_path_scale), 1.0e-4)
        color_limit = max(_safe_quantile(stage_color_seed, 0.99, float(color_threshold)) * float(deep_fill_color_scale), float(color_threshold))
        support_center_delta = np.abs(support_center_all - float(np.median(stage_support_center)))

        feature_affinity = np.exp(-0.5 * np.square(feature_nearest / feature_radius)).astype(np.float32)
        spatial_affinity = np.exp(-0.5 * np.square(spatial_nearest / spatial_radius)).astype(np.float32)
        feature_maha_affinity = np.exp(-0.5 * np.square(feature_maha_all / feature_maha_limit)).astype(np.float32)
        spatial_maha_affinity = np.exp(-0.5 * np.square(spatial_maha_all / spatial_maha_limit)).astype(np.float32)
        color_affinity = np.exp(-0.5 * np.square(stage_color_all / max(color_limit, 1.0e-4))).astype(np.float32)
        gate_affinity = np.clip(
            (mean_gate_all - gate_floor) / max(float(np.quantile(stage_gate, 0.9)) - gate_floor, 1.0e-4),
            0.0,
            1.0,
        ).astype(np.float32)
        path_affinity = np.exp(-0.5 * np.square(path_length_all / path_limit)).astype(np.float32)
        support_affinity = np.exp(-0.5 * np.square(support_center_delta / support_center_limit)).astype(np.float32)
        opacity_affinity = np.clip(
            (opacity - float(deep_fill_opacity_min))
            / max(float(np.quantile(opacity[stage_ids], 0.8)) - float(deep_fill_opacity_min), 1.0e-4),
            0.0,
            1.0,
        ).astype(np.float32)

        deep_score = (
            0.27 * feature_affinity
            + 0.19 * spatial_affinity
            + 0.14 * feature_maha_affinity
            + 0.11 * spatial_maha_affinity
            + 0.10 * color_affinity
            + 0.08 * gate_affinity
            + 0.05 * path_affinity
            + 0.04 * support_affinity
            + 0.02 * opacity_affinity
        ).astype(np.float32)
        deep_mask = (
            (feature_nearest <= feature_radius)
            & (spatial_nearest <= spatial_radius)
            & (feature_maha_all <= feature_maha_limit)
            & (spatial_maha_all <= spatial_maha_limit)
            & (stage_color_all <= color_limit)
            & (path_length_all <= path_limit)
            & (mean_gate_all >= gate_floor)
            & (active_ratio_all >= active_ratio_floor * 0.85)
            & (support_center_delta <= support_center_limit)
            & (opacity >= float(deep_fill_opacity_min))
        )
        new_ids = np.where(deep_mask & (deep_score >= float(deep_fill_score_threshold)))[0].astype(np.int64)
        if new_ids.size > 0:
            new_ids = np.setdiff1d(new_ids, stage_ids, assume_unique=False)
        if new_ids.size > 0 and float(deep_fill_max_round_growth) > 0.0:
            max_new = max(1, int(round(float(stage_ids.size) * float(deep_fill_max_round_growth))))
            if new_ids.size > max_new:
                ranked_new = new_ids[np.argsort(-deep_score[new_ids], kind="mergesort")]
                new_ids = ranked_new[:max_new].astype(np.int64)
        candidate_ids = np.unique(np.concatenate([stage_ids, new_ids], axis=0)).astype(np.int64)
        total_added += int(new_ids.size)
        round_rows.append(
            {
                "round_index": int(round_index),
                "input_count": int(stage_ids.size),
                "added_count": int(new_ids.size),
                "output_count": int(candidate_ids.size),
                "feature_radius": float(feature_radius),
                "spatial_radius": float(spatial_radius),
                "feature_maha_limit": float(feature_maha_limit),
                "spatial_maha_limit": float(spatial_maha_limit),
                "gate_floor": float(gate_floor),
                "path_limit": float(path_limit),
                "color_limit": float(color_limit),
                "support_center_limit": float(support_center_limit),
                "mean_added_score": float(deep_score[new_ids].mean()) if new_ids.size else 0.0,
            }
        )
        if new_ids.size == 0:
            break

    return candidate_ids.astype(np.int64), {
        "enabled": True,
        "round_count": int(len(round_rows)),
        "added_count": int(total_added),
        "rounds": round_rows,
        "params": {
            "deep_fill_rounds": int(deep_fill_rounds),
            "deep_fill_knn_factor": float(deep_fill_knn_factor),
            "deep_fill_feature_knn_factor": float(deep_fill_feature_knn_factor),
            "deep_fill_feature_maha_factor": float(deep_fill_feature_maha_factor),
            "deep_fill_spatial_maha_factor": float(deep_fill_spatial_maha_factor),
            "deep_fill_color_scale": float(deep_fill_color_scale),
            "deep_fill_path_scale": float(deep_fill_path_scale),
            "deep_fill_gate_quantile": float(deep_fill_gate_quantile),
            "deep_fill_opacity_min": float(deep_fill_opacity_min),
            "deep_fill_score_threshold": float(deep_fill_score_threshold),
            "deep_fill_max_round_growth": float(deep_fill_max_round_growth),
        },
    }


def _expanded_ids(
    run_dir: Path,
    seed_ids: np.ndarray,
    query_tracks_path: Path | None,
    query_phrase: str,
    sample_start_index: int | None,
    sample_end_index: int | None,
    sample_window_info: dict[str, Any] | None,
    color_threshold: float,
    distance_margin: float,
    path_multiplier: float,
    gate_quantile: float,
    opacity_min: float,
    mode: str,
    ellipsoid_margin: float,
    two_stage: bool,
    interior_backfill: bool,
    interior_knn_factor: float,
    interior_ellipsoid_factor: float,
    interior_color_scale: float,
    interior_path_scale: float,
    interior_gate_quantile: float,
    interior_opacity_min: float,
    deep_fill: bool,
    deep_fill_rounds: int,
    deep_fill_knn_factor: float,
    deep_fill_feature_knn_factor: float,
    deep_fill_feature_maha_factor: float,
    deep_fill_spatial_maha_factor: float,
    deep_fill_color_scale: float,
    deep_fill_path_scale: float,
    deep_fill_gate_quantile: float,
    deep_fill_opacity_min: float,
    deep_fill_score_threshold: float,
    deep_fill_max_round_growth: float,
    mask_guided: bool,
    mask_guided_target_count: int,
    mask_guided_max_frames: int,
    mask_guided_dilation_radius: int,
    mask_guided_negative_margin: float,
    mask_guided_feature_maha_factor: float,
    mask_guided_spatial_maha_factor: float,
    mask_guided_color_scale: float,
    mask_guided_vote_weight: float,
    mask_guided_vote_threshold: float,
    mask_guided_background_margin: float,
    mask_guided_min_pos_votes: int,
    cluster_refine: bool,
    cluster_refine_target_count: int,
    cluster_refine_radius_scale: float,
    cluster_refine_min_component_size: int,
    cluster_refine_min_mask_contrast: float,
    cluster_refine_max_neg_rate: float,
    cluster_refine_center_distance_scale: float,
    cluster_refine_refill_radius_scale: float,
    mask_guided_prune_background_selected: bool,
    mask_guided_prune_max_drop_ratio: float,
    mask_guided_prune_neg_margin: float,
) -> dict[str, Any]:
    payload = np.load(run_dir / "entitybank" / "trajectory_samples.npz")
    trajectories = np.asarray(payload["trajectories"], dtype=np.float32)
    gate = np.asarray(payload["gate"], dtype=np.float32).reshape(trajectories.shape[0], trajectories.shape[1])
    sample_start, sample_end, resolved_window = _resolve_sample_window(
        summary_row=sample_window_info,
        requested_start=sample_start_index,
        requested_end=sample_end_index,
        total_samples=int(trajectories.shape[1]),
    )

    ply = PlyData.read(str(_find_latest_iteration_dir(run_dir) / "point_cloud.ply"))
    vertex = ply["vertex"].data
    opacity = _sigmoid(np.asarray(vertex["opacity"], dtype=np.float32))
    rgb = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)

    seed_traj = trajectories[seed_ids, sample_start:sample_end, :]
    seed_mean = trajectories[seed_ids, sample_start:sample_end, :].mean(axis=1)
    seed_center = seed_traj.mean(axis=(0, 1))
    traj_mean = trajectories[:, sample_start:sample_end, :].mean(axis=1)
    dist = np.linalg.norm(traj_mean - seed_center[None, :], axis=1)
    seed_dist = np.linalg.norm(traj_mean[seed_ids] - seed_center[None, :], axis=1)

    seed_path = np.linalg.norm(np.diff(trajectories[seed_ids, sample_start:sample_end, :], axis=1), axis=2).sum(axis=1)
    all_path = np.linalg.norm(np.diff(trajectories[:, sample_start:sample_end, :], axis=1), axis=2).sum(axis=1)
    seed_gate = gate[seed_ids, sample_start:sample_end].mean(axis=1)
    all_gate = gate[:, sample_start:sample_end].mean(axis=1)

    seed_rgb = rgb[seed_ids].mean(axis=0)
    color_dist = np.linalg.norm(rgb - seed_rgb[None, :], axis=1)

    path_limit = float(np.quantile(seed_path, 0.95) * float(path_multiplier))
    gate_limit = float(np.quantile(seed_gate, float(gate_quantile)))
    base_mask = (
        (all_gate >= gate_limit)
        & (all_path <= path_limit)
        & (color_dist <= float(color_threshold))
        & (opacity >= float(opacity_min))
    )

    if mode == "ellipsoid":
        seed_cov = np.cov(seed_mean.T) + np.eye(3, dtype=np.float32) * 1.0e-4
        inv_cov = np.linalg.inv(seed_cov)
        maha = np.sqrt(np.einsum("ni,ij,nj->n", traj_mean - seed_center[None, :], inv_cov, traj_mean - seed_center[None, :]))
        seed_maha = maha[seed_ids]
        distance_limit = float(np.quantile(seed_maha, 0.95) + float(ellipsoid_margin))
        candidate_ids = np.where(base_mask & (maha <= distance_limit))[0]
    else:
        distance_limit = float(np.quantile(seed_dist, 0.95) + float(distance_margin))
        candidate_ids = np.where(base_mask & (dist <= distance_limit))[0]

    if bool(two_stage) and candidate_ids.size > 0:
        stage1_ids = np.asarray(candidate_ids, dtype=np.int64)
        stage1_mean = trajectories[stage1_ids, sample_start:sample_end, :].mean(axis=(0, 1))
        traj_mean_all = trajectories[:, sample_start:sample_end, :].mean(axis=1)
        stage1_dist = np.linalg.norm(traj_mean_all[stage1_ids] - stage1_mean[None, :], axis=1)
        stage1_rgb = rgb[stage1_ids].mean(axis=0)
        stage1_color = np.linalg.norm(rgb - stage1_rgb[None, :], axis=1)
        stage1_gate = gate[stage1_ids, sample_start:sample_end].mean(axis=1)
        stage1_path = np.linalg.norm(np.diff(trajectories[stage1_ids, sample_start:sample_end, :], axis=1), axis=2).sum(axis=1)
        stage2_mask = (
            (all_gate >= np.quantile(stage1_gate, 0.15))
            & (all_path <= np.quantile(stage1_path, 0.98) * 1.15)
            & (stage1_color <= max(float(color_threshold) * 0.82, 0.25))
            & (opacity >= float(opacity_min))
        )
        if mode == "ellipsoid":
            stage1_mean_pos = trajectories[stage1_ids, sample_start:sample_end, :].mean(axis=1)
            cov2 = np.cov(stage1_mean_pos.T) + np.eye(3, dtype=np.float32) * 1.0e-4
            inv2 = np.linalg.inv(cov2)
            maha2 = np.sqrt(np.einsum("ni,ij,nj->n", traj_mean_all - stage1_mean[None, :], inv2, traj_mean_all - stage1_mean[None, :]))
            stage1_maha = maha2[stage1_ids]
            stage2_mask &= maha2 <= float(np.quantile(stage1_maha, 0.95) + float(ellipsoid_margin) * 0.55)
        else:
            stage2_mask &= np.linalg.norm(traj_mean_all - stage1_mean[None, :], axis=1) <= float(np.quantile(stage1_dist, 0.95) + float(distance_margin) * 0.65)
        candidate_ids = np.where(stage2_mask)[0]

    interior_stats = {
        "enabled": bool(interior_backfill),
        "stage_seed_count": int(candidate_ids.shape[0]),
        "added_count": 0,
    }
    if bool(interior_backfill) and candidate_ids.size > 0:
        stage_ids = np.asarray(candidate_ids, dtype=np.int64)
        traj_mean_all = trajectories[:, sample_start:sample_end, :].mean(axis=1)
        stage_mean_pos = traj_mean_all[stage_ids]
        stage_gate = gate[stage_ids, sample_start:sample_end].mean(axis=1)
        stage_path = np.linalg.norm(np.diff(trajectories[stage_ids, sample_start:sample_end, :], axis=1), axis=2).sum(axis=1)
        stage_color = np.linalg.norm(rgb - rgb[stage_ids].mean(axis=0, keepdims=True), axis=1)

        stage_center = stage_mean_pos.mean(axis=0)
        cov_stage = np.cov(stage_mean_pos.T) + np.eye(3, dtype=np.float32) * 1.0e-4
        inv_stage = np.linalg.inv(cov_stage)
        maha_stage_all = np.sqrt(
            np.einsum(
                "ni,ij,nj->n",
                traj_mean_all - stage_center[None, :],
                inv_stage,
                traj_mean_all - stage_center[None, :],
            )
        )
        stage_maha = maha_stage_all[stage_ids]

        tree = cKDTree(stage_mean_pos)
        if stage_mean_pos.shape[0] >= 2:
            stage_nn = tree.query(stage_mean_pos, k=2)[0][:, 1]
            support_radius = float(np.quantile(stage_nn, 0.95) * float(interior_knn_factor))
        else:
            support_radius = float(np.linalg.norm(np.std(stage_mean_pos, axis=0)) * float(interior_knn_factor) + 1.0e-4)
        nearest_stage_dist = tree.query(traj_mean_all, k=1)[0]

        interior_gate_floor = float(np.quantile(stage_gate, float(interior_gate_quantile)))
        interior_path_limit = float(np.quantile(stage_path, 0.98) * float(interior_path_scale))
        interior_maha_limit = float(np.quantile(stage_maha, 0.98) * float(interior_ellipsoid_factor))
        interior_color_limit = float(max(np.quantile(stage_color[stage_ids], 0.98) * float(interior_color_scale), float(color_threshold)))

        interior_mask = (
            (nearest_stage_dist <= support_radius)
            & (maha_stage_all <= interior_maha_limit)
            & (stage_color <= interior_color_limit)
            & (all_path <= interior_path_limit)
            & (all_gate >= interior_gate_floor)
            & (opacity >= float(interior_opacity_min))
        )
        backfill_ids = np.where(interior_mask)[0].astype(np.int64)
        merged_ids = np.unique(np.concatenate([stage_ids, backfill_ids], axis=0)).astype(np.int64)
        interior_stats = {
            "enabled": True,
            "stage_seed_count": int(stage_ids.shape[0]),
            "added_count": int(merged_ids.shape[0] - stage_ids.shape[0]),
            "support_radius": support_radius,
            "interior_gate_floor": interior_gate_floor,
            "interior_path_limit": interior_path_limit,
            "interior_maha_limit": interior_maha_limit,
            "interior_color_limit": interior_color_limit,
            "interior_opacity_min": float(interior_opacity_min),
        }
        candidate_ids = merged_ids

    deep_fill_stats = {
        "enabled": bool(deep_fill),
        "added_count": 0,
        "rounds": [],
    }
    if bool(deep_fill) and candidate_ids.size > 0:
        deep_ids, deep_fill_stats = _deep_fill_ids(
            current_ids=np.asarray(candidate_ids, dtype=np.int64),
            payload=payload,
            trajectories=trajectories,
            gate=gate,
            sample_start=sample_start,
            sample_end=sample_end,
            rgb=rgb,
            opacity=opacity,
            color_threshold=float(color_threshold),
            deep_fill_rounds=int(deep_fill_rounds),
            deep_fill_knn_factor=float(deep_fill_knn_factor),
            deep_fill_feature_knn_factor=float(deep_fill_feature_knn_factor),
            deep_fill_feature_maha_factor=float(deep_fill_feature_maha_factor),
            deep_fill_spatial_maha_factor=float(deep_fill_spatial_maha_factor),
            deep_fill_color_scale=float(deep_fill_color_scale),
            deep_fill_path_scale=float(deep_fill_path_scale),
            deep_fill_gate_quantile=float(deep_fill_gate_quantile),
            deep_fill_opacity_min=float(deep_fill_opacity_min),
            deep_fill_score_threshold=float(deep_fill_score_threshold),
            deep_fill_max_round_growth=float(deep_fill_max_round_growth),
        )
        candidate_ids = np.asarray(deep_ids, dtype=np.int64)

    mask_guided_stats = {
        "enabled": bool(mask_guided),
        "skip_reason": "disabled",
    }
    if bool(mask_guided):
        refined_ids, mask_guided_stats = _mask_guided_select_ids(
            payload=payload,
            trajectories=trajectories,
            gate=gate,
            rgb=rgb,
            opacity=opacity,
            seed_ids=np.asarray(seed_ids, dtype=np.int64),
            current_ids=np.asarray(candidate_ids, dtype=np.int64),
            sample_start=int(sample_start),
            sample_end=int(sample_end),
            run_dir=run_dir,
            query_tracks_path=query_tracks_path,
            query_phrase=str(query_phrase),
            mask_guided_target_count=int(mask_guided_target_count),
            mask_guided_max_frames=int(mask_guided_max_frames),
            mask_guided_dilation_radius=int(mask_guided_dilation_radius),
            mask_guided_negative_margin=float(mask_guided_negative_margin),
            mask_guided_feature_maha_factor=float(mask_guided_feature_maha_factor),
            mask_guided_spatial_maha_factor=float(mask_guided_spatial_maha_factor),
            mask_guided_color_scale=float(mask_guided_color_scale),
            mask_guided_vote_weight=float(mask_guided_vote_weight),
            mask_guided_vote_threshold=float(mask_guided_vote_threshold),
            mask_guided_background_margin=float(mask_guided_background_margin),
            mask_guided_min_pos_votes=int(mask_guided_min_pos_votes),
            cluster_refine=bool(cluster_refine),
            cluster_refine_target_count=int(cluster_refine_target_count),
            cluster_refine_radius_scale=float(cluster_refine_radius_scale),
            cluster_refine_min_component_size=int(cluster_refine_min_component_size),
            cluster_refine_min_mask_contrast=float(cluster_refine_min_mask_contrast),
            cluster_refine_max_neg_rate=float(cluster_refine_max_neg_rate),
            cluster_refine_center_distance_scale=float(cluster_refine_center_distance_scale),
            cluster_refine_refill_radius_scale=float(cluster_refine_refill_radius_scale),
            mask_guided_prune_background_selected=bool(mask_guided_prune_background_selected),
            mask_guided_prune_max_drop_ratio=float(mask_guided_prune_max_drop_ratio),
            mask_guided_prune_neg_margin=float(mask_guided_prune_neg_margin),
        )
        candidate_ids = np.asarray(refined_ids, dtype=np.int64)

    return {
        "expanded_ids": candidate_ids.astype(np.int64),
        "stats": {
            "seed_count": int(seed_ids.shape[0]),
            "expanded_count": int(candidate_ids.shape[0]),
            "seed_opacity_mean": float(opacity[seed_ids].mean()),
            "expanded_opacity_mean": float(opacity[candidate_ids].mean()) if candidate_ids.size else 0.0,
            "expanded_opacity_p50": float(np.quantile(opacity[candidate_ids], 0.5)) if candidate_ids.size else 0.0,
            "distance_limit": distance_limit,
            "path_limit": path_limit,
            "gate_limit": gate_limit,
            "color_threshold": float(color_threshold),
            "opacity_min": float(opacity_min),
            "sample_start_index": int(sample_start),
            "sample_end_index": int(sample_end),
            "sample_window": resolved_window,
            "mode": str(mode),
            "ellipsoid_margin": float(ellipsoid_margin),
            "two_stage": bool(two_stage),
            "interior_backfill": interior_stats,
            "deep_fill": deep_fill_stats,
            "mask_guided": mask_guided_stats,
        },
    }


def build_expanded_proposal(
    run_dir: Path,
    source_proposal_dir: Path,
    output_dir: Path,
    proposal_alias: str | None,
    proposal_entity_id: int | None,
    sample_start_index: int | None,
    sample_end_index: int | None,
    color_threshold: float,
    distance_margin: float,
    path_multiplier: float,
    gate_quantile: float,
    opacity_min: float,
    mode: str,
    ellipsoid_margin: float,
    two_stage: bool,
    interior_backfill: bool,
    interior_knn_factor: float,
    interior_ellipsoid_factor: float,
    interior_color_scale: float,
    interior_path_scale: float,
    interior_gate_quantile: float,
    interior_opacity_min: float,
    deep_fill: bool,
    deep_fill_rounds: int,
    deep_fill_knn_factor: float,
    deep_fill_feature_knn_factor: float,
    deep_fill_feature_maha_factor: float,
    deep_fill_spatial_maha_factor: float,
    deep_fill_color_scale: float,
    deep_fill_path_scale: float,
    deep_fill_gate_quantile: float,
    deep_fill_opacity_min: float,
    deep_fill_score_threshold: float,
    deep_fill_max_round_growth: float,
    query_tracks_path: Path | None,
    mask_guided: bool,
    mask_guided_target_count: int,
    mask_guided_max_frames: int,
    mask_guided_dilation_radius: int,
    mask_guided_negative_margin: float,
    mask_guided_feature_maha_factor: float,
    mask_guided_spatial_maha_factor: float,
    mask_guided_color_scale: float,
    mask_guided_vote_weight: float,
    mask_guided_vote_threshold: float,
    mask_guided_background_margin: float,
    mask_guided_min_pos_votes: int,
    cluster_refine: bool,
    cluster_refine_target_count: int,
    cluster_refine_radius_scale: float,
    cluster_refine_min_component_size: int,
    cluster_refine_min_mask_contrast: float,
    cluster_refine_max_neg_rate: float,
    cluster_refine_center_distance_scale: float,
    cluster_refine_refill_radius_scale: float,
    mask_guided_prune_background_selected: bool,
    mask_guided_prune_max_drop_ratio: float,
    mask_guided_prune_neg_margin: float,
) -> Path:
    seed_ids, matched_entity, summary_row = _load_seed_selection(
        source_proposal_dir,
        proposal_alias=proposal_alias,
        proposal_entity_id=proposal_entity_id,
    )
    resolved_query_tracks_path = query_tracks_path if query_tracks_path is not None else _default_query_tracks_path(source_proposal_dir)
    query_phrase = _resolve_query_phrase(matched_entity=matched_entity, summary_row=summary_row)
    expanded = _expanded_ids(
        run_dir=run_dir,
        seed_ids=seed_ids,
        query_tracks_path=resolved_query_tracks_path,
        query_phrase=query_phrase,
        sample_start_index=sample_start_index,
        sample_end_index=sample_end_index,
        sample_window_info=summary_row,
        color_threshold=color_threshold,
        distance_margin=distance_margin,
        path_multiplier=path_multiplier,
        gate_quantile=gate_quantile,
        opacity_min=opacity_min,
        mode=mode,
        ellipsoid_margin=ellipsoid_margin,
        two_stage=bool(two_stage),
        interior_backfill=bool(interior_backfill),
        interior_knn_factor=float(interior_knn_factor),
        interior_ellipsoid_factor=float(interior_ellipsoid_factor),
        interior_color_scale=float(interior_color_scale),
        interior_path_scale=float(interior_path_scale),
        interior_gate_quantile=float(interior_gate_quantile),
        interior_opacity_min=float(interior_opacity_min),
        deep_fill=bool(deep_fill),
        deep_fill_rounds=int(deep_fill_rounds),
        deep_fill_knn_factor=float(deep_fill_knn_factor),
        deep_fill_feature_knn_factor=float(deep_fill_feature_knn_factor),
        deep_fill_feature_maha_factor=float(deep_fill_feature_maha_factor),
        deep_fill_spatial_maha_factor=float(deep_fill_spatial_maha_factor),
        deep_fill_color_scale=float(deep_fill_color_scale),
        deep_fill_path_scale=float(deep_fill_path_scale),
        deep_fill_gate_quantile=float(deep_fill_gate_quantile),
        deep_fill_opacity_min=float(deep_fill_opacity_min),
        deep_fill_score_threshold=float(deep_fill_score_threshold),
        deep_fill_max_round_growth=float(deep_fill_max_round_growth),
        mask_guided=bool(mask_guided),
        mask_guided_target_count=int(mask_guided_target_count),
        mask_guided_max_frames=int(mask_guided_max_frames),
        mask_guided_dilation_radius=int(mask_guided_dilation_radius),
        mask_guided_negative_margin=float(mask_guided_negative_margin),
        mask_guided_feature_maha_factor=float(mask_guided_feature_maha_factor),
        mask_guided_spatial_maha_factor=float(mask_guided_spatial_maha_factor),
        mask_guided_color_scale=float(mask_guided_color_scale),
        mask_guided_vote_weight=float(mask_guided_vote_weight),
        mask_guided_vote_threshold=float(mask_guided_vote_threshold),
        mask_guided_background_margin=float(mask_guided_background_margin),
        mask_guided_min_pos_votes=int(mask_guided_min_pos_votes),
        cluster_refine=bool(cluster_refine),
        cluster_refine_target_count=int(cluster_refine_target_count),
        cluster_refine_radius_scale=float(cluster_refine_radius_scale),
        cluster_refine_min_component_size=int(cluster_refine_min_component_size),
        cluster_refine_min_mask_contrast=float(cluster_refine_min_mask_contrast),
        cluster_refine_max_neg_rate=float(cluster_refine_max_neg_rate),
        cluster_refine_center_distance_scale=float(cluster_refine_center_distance_scale),
        cluster_refine_refill_radius_scale=float(cluster_refine_refill_radius_scale),
        mask_guided_prune_background_selected=bool(mask_guided_prune_background_selected),
        mask_guided_prune_max_drop_ratio=float(mask_guided_prune_max_drop_ratio),
        mask_guided_prune_neg_margin=float(mask_guided_prune_neg_margin),
    )

    entities_payload = _read_json(source_proposal_dir / "entities.json")
    summary_path = source_proposal_dir / "query_proposal_summary.json"
    summary_payload = _read_json(summary_path) if summary_path.exists() else {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "source_proposal_dir": str(source_proposal_dir),
        "num_entities": int(len(entities_payload.get("entities", []))),
        "phrases": [],
        "params": {},
    }
    expanded_ids = expanded["expanded_ids"].astype(int).tolist()

    for entity in entities_payload.get("entities", []):
        entity_id = int(entity.get("id", -1))
        if entity_id == int(matched_entity["id"]):
            entity["gaussian_ids"] = expanded_ids
            entity["quality"] = 0.97
            entity["expansion_stats"] = expanded["stats"]

    matched_summary_row = None
    for row in summary_payload.get("phrases", []):
        row_id = int(row.get("id", -1))
        row_alias = str(row.get("proposal_alias", "")).strip()
        if row_id == int(matched_entity["id"]) or (
            row_alias and row_alias == str(matched_entity.get("proposal_alias", "")).strip()
        ):
            row["selected_gaussian_count"] = int(len(expanded_ids))
            row["quality"] = 0.97
            row["expansion_stats"] = expanded["stats"]
            matched_summary_row = row
    if matched_summary_row is None:
        summary_payload.setdefault("phrases", []).append(
            {
                "id": int(matched_entity["id"]),
                "phrase": str(matched_entity.get("static_text", matched_entity.get("proposal_alias", f"entity_{matched_entity['id']}"))),
                "proposal_alias": matched_entity.get("proposal_alias"),
                "phase": matched_entity.get("proposal_phase", "full"),
                "variant_kind": matched_entity.get("proposal_variant", "manual_seed_expand"),
                "entity_type": matched_entity.get("entity_type", "dynamic_object"),
                "selected_gaussian_count": int(len(expanded_ids)),
                "quality": 0.97,
                "expansion_stats": expanded["stats"],
                "segments": matched_entity.get("segments", []),
                "keyframes": matched_entity.get("keyframes", []),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "entities.json", entities_payload)
    _write_json(output_dir / "query_proposal_summary.json", summary_payload)
    _write_json(output_dir / "seed_expansion_stats.json", expanded["stats"])
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-proposal-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--proposal-alias", default=None)
    parser.add_argument("--proposal-entity-id", type=int, default=None)
    parser.add_argument("--sample-start-index", type=int, default=None)
    parser.add_argument("--sample-end-index", type=int, default=None)
    parser.add_argument("--color-threshold", type=float, default=1.2)
    parser.add_argument("--distance-margin", type=float, default=0.18)
    parser.add_argument("--path-multiplier", type=float, default=1.8)
    parser.add_argument("--gate-quantile", type=float, default=0.2)
    parser.add_argument("--opacity-min", type=float, default=0.15)
    parser.add_argument("--mode", choices=["ball", "ellipsoid"], default="ball")
    parser.add_argument("--ellipsoid-margin", type=float, default=0.25)
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument("--interior-backfill", action="store_true")
    parser.add_argument("--interior-knn-factor", type=float, default=2.0)
    parser.add_argument("--interior-ellipsoid-factor", type=float, default=1.15)
    parser.add_argument("--interior-color-scale", type=float, default=1.1)
    parser.add_argument("--interior-path-scale", type=float, default=1.08)
    parser.add_argument("--interior-gate-quantile", type=float, default=0.05)
    parser.add_argument("--interior-opacity-min", type=float, default=0.01)
    parser.add_argument("--deep-fill", action="store_true")
    parser.add_argument("--deep-fill-rounds", type=int, default=2)
    parser.add_argument("--deep-fill-knn-factor", type=float, default=2.2)
    parser.add_argument("--deep-fill-feature-knn-factor", type=float, default=1.5)
    parser.add_argument("--deep-fill-feature-maha-factor", type=float, default=1.2)
    parser.add_argument("--deep-fill-spatial-maha-factor", type=float, default=1.15)
    parser.add_argument("--deep-fill-color-scale", type=float, default=1.08)
    parser.add_argument("--deep-fill-path-scale", type=float, default=1.18)
    parser.add_argument("--deep-fill-gate-quantile", type=float, default=0.02)
    parser.add_argument("--deep-fill-opacity-min", type=float, default=0.005)
    parser.add_argument("--deep-fill-score-threshold", type=float, default=0.56)
    parser.add_argument("--deep-fill-max-round-growth", type=float, default=0.50)
    parser.add_argument("--query-tracks-path", default=None)
    parser.add_argument("--mask-guided", action="store_true")
    parser.add_argument("--mask-guided-target-count", type=int, default=0)
    parser.add_argument("--mask-guided-max-frames", type=int, default=12)
    parser.add_argument("--mask-guided-dilation-radius", type=int, default=3)
    parser.add_argument("--mask-guided-negative-margin", type=float, default=0.22)
    parser.add_argument("--mask-guided-feature-maha-factor", type=float, default=1.10)
    parser.add_argument("--mask-guided-spatial-maha-factor", type=float, default=1.05)
    parser.add_argument("--mask-guided-color-scale", type=float, default=1.05)
    parser.add_argument("--mask-guided-vote-weight", type=float, default=0.18)
    parser.add_argument("--mask-guided-vote-threshold", type=float, default=0.08)
    parser.add_argument("--mask-guided-background-margin", type=float, default=0.10)
    parser.add_argument("--mask-guided-min-pos-votes", type=int, default=2)
    parser.add_argument("--mask-guided-prune-background-selected", action="store_true")
    parser.add_argument("--mask-guided-prune-max-drop-ratio", type=float, default=0.35)
    parser.add_argument("--mask-guided-prune-neg-margin", type=float, default=0.00)
    parser.add_argument("--cluster-refine", action="store_true")
    parser.add_argument("--cluster-refine-target-count", type=int, default=0)
    parser.add_argument("--cluster-refine-radius-scale", type=float, default=1.20)
    parser.add_argument("--cluster-refine-min-component-size", type=int, default=12)
    parser.add_argument("--cluster-refine-min-mask-contrast", type=float, default=0.18)
    parser.add_argument("--cluster-refine-max-neg-rate", type=float, default=0.35)
    parser.add_argument("--cluster-refine-center-distance-scale", type=float, default=1.90)
    parser.add_argument("--cluster-refine-refill-radius-scale", type=float, default=1.35)
    args = parser.parse_args()
    if args.proposal_alias is None and args.proposal_entity_id is None:
        raise ValueError("--proposal-alias or --proposal-entity-id is required")

    output_dir = build_expanded_proposal(
        run_dir=Path(args.run_dir),
        source_proposal_dir=Path(args.source_proposal_dir),
        output_dir=Path(args.output_dir),
        proposal_alias=args.proposal_alias,
        proposal_entity_id=args.proposal_entity_id,
        sample_start_index=None if args.sample_start_index is None else int(args.sample_start_index),
        sample_end_index=None if args.sample_end_index is None else int(args.sample_end_index),
        color_threshold=float(args.color_threshold),
        distance_margin=float(args.distance_margin),
        path_multiplier=float(args.path_multiplier),
        gate_quantile=float(args.gate_quantile),
        opacity_min=float(args.opacity_min),
        mode=str(args.mode),
        ellipsoid_margin=float(args.ellipsoid_margin),
        two_stage=bool(args.two_stage),
        interior_backfill=bool(args.interior_backfill),
        interior_knn_factor=float(args.interior_knn_factor),
        interior_ellipsoid_factor=float(args.interior_ellipsoid_factor),
        interior_color_scale=float(args.interior_color_scale),
        interior_path_scale=float(args.interior_path_scale),
        interior_gate_quantile=float(args.interior_gate_quantile),
        interior_opacity_min=float(args.interior_opacity_min),
        deep_fill=bool(args.deep_fill),
        deep_fill_rounds=int(args.deep_fill_rounds),
        deep_fill_knn_factor=float(args.deep_fill_knn_factor),
        deep_fill_feature_knn_factor=float(args.deep_fill_feature_knn_factor),
        deep_fill_feature_maha_factor=float(args.deep_fill_feature_maha_factor),
        deep_fill_spatial_maha_factor=float(args.deep_fill_spatial_maha_factor),
        deep_fill_color_scale=float(args.deep_fill_color_scale),
        deep_fill_path_scale=float(args.deep_fill_path_scale),
        deep_fill_gate_quantile=float(args.deep_fill_gate_quantile),
        deep_fill_opacity_min=float(args.deep_fill_opacity_min),
        deep_fill_score_threshold=float(args.deep_fill_score_threshold),
        deep_fill_max_round_growth=float(args.deep_fill_max_round_growth),
        query_tracks_path=None if args.query_tracks_path is None else Path(args.query_tracks_path),
        mask_guided=bool(args.mask_guided),
        mask_guided_target_count=int(args.mask_guided_target_count),
        mask_guided_max_frames=int(args.mask_guided_max_frames),
        mask_guided_dilation_radius=int(args.mask_guided_dilation_radius),
        mask_guided_negative_margin=float(args.mask_guided_negative_margin),
        mask_guided_feature_maha_factor=float(args.mask_guided_feature_maha_factor),
        mask_guided_spatial_maha_factor=float(args.mask_guided_spatial_maha_factor),
        mask_guided_color_scale=float(args.mask_guided_color_scale),
        mask_guided_vote_weight=float(args.mask_guided_vote_weight),
        mask_guided_vote_threshold=float(args.mask_guided_vote_threshold),
        mask_guided_background_margin=float(args.mask_guided_background_margin),
        mask_guided_min_pos_votes=int(args.mask_guided_min_pos_votes),
        mask_guided_prune_background_selected=bool(args.mask_guided_prune_background_selected),
        mask_guided_prune_max_drop_ratio=float(args.mask_guided_prune_max_drop_ratio),
        mask_guided_prune_neg_margin=float(args.mask_guided_prune_neg_margin),
        cluster_refine=bool(args.cluster_refine),
        cluster_refine_target_count=int(args.cluster_refine_target_count),
        cluster_refine_radius_scale=float(args.cluster_refine_radius_scale),
        cluster_refine_min_component_size=int(args.cluster_refine_min_component_size),
        cluster_refine_min_mask_contrast=float(args.cluster_refine_min_mask_contrast),
        cluster_refine_max_neg_rate=float(args.cluster_refine_max_neg_rate),
        cluster_refine_center_distance_scale=float(args.cluster_refine_center_distance_scale),
        cluster_refine_refill_radius_scale=float(args.cluster_refine_refill_radius_scale),
    )
    print(output_dir)


if __name__ == "__main__":
    main()
