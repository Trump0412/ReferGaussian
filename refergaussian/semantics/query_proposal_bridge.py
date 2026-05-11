from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_ROOT = _PROJECT_ROOT / "external" / "4DGaussians"
for _candidate in (_PROJECT_ROOT,):
    candidate_str = str(_candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


def _load_camera_class():
    utils_path = _UPSTREAM_ROOT / "scene" / "utils.py"
    spec = importlib.util.spec_from_file_location("refergaussian_scene_utils_bridge", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Camera utilities from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Camera


Camera = _load_camera_class()

from .worldtube_consistency import load_opacity_sigmoid, select_worldtube_consistency_cluster
from .source_images import resolve_dataset_image_entries
from .appearance_backbone import feature_map_from_image, prototypes_from_masks, sample_feature_map, prototype_similarity


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _load_bank(entitybank_dir: Path) -> dict[str, np.ndarray]:
    payload = np.load(entitybank_dir / "trajectory_samples.npz")
    return {key: payload[key] for key in payload.files}


def _resample_indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    source = np.asarray(source_times, dtype=np.float32).reshape(-1)
    target = np.asarray(target_times, dtype=np.float32).reshape(-1)
    return np.abs(target[:, None] - source[None, :]).argmin(axis=1).astype(np.int32)


def _active_track_frames(track: dict[str, Any], max_track_frames: int) -> list[dict[str, Any]]:
    active = [
        frame
        for frame in track.get("frames", [])
        if bool(frame.get("active")) and frame.get("bbox_xyxy") is not None
    ]
    if not active:
        raise ValueError(f"No active bbox frames found for phrase '{track.get('phrase', 'unknown')}'.")
    if len(active) <= int(max_track_frames):
        return active
    indices = np.linspace(0, len(active) - 1, num=int(max_track_frames), dtype=np.int32)
    return [active[int(index)] for index in indices.tolist()]


def _phrase_entity_type(phrase: str) -> str:
    text = str(phrase).lower()
    if any(token in text for token in ("board", "surface", "table", "counter")):
        return "support_surface"
    return "object"


def _phrase_segments(frame_indices: list[int]) -> list[dict[str, Any]]:
    if not frame_indices:
        return []
    frame_indices = sorted(set(int(value) for value in frame_indices))
    segments: list[dict[str, Any]] = []
    start = frame_indices[0]
    prev = frame_indices[0]
    for value in frame_indices[1:]:
        if value == prev + 1:
            prev = value
            continue
        segments.append(
            {
                "segment_id": len(segments),
                "t0": int(start),
                "t1": int(prev + 1),
                "label": "moving",
                "confidence": 1.0,
                "mode": "query_guided_worldtube",
            }
        )
        start = value
        prev = value
    segments.append(
        {
            "segment_id": len(segments),
            "t0": int(start),
            "t1": int(prev + 1),
            "label": "moving",
            "confidence": 1.0,
            "mode": "query_guided_worldtube",
        }
    )
    return segments


def _sample_search_frames(track: dict[str, Any], max_track_frames: int) -> list[dict[str, Any]]:
    frames = _active_track_frames(track, max_track_frames=max_track_frames)
    return sorted(frames, key=lambda item: int(item["frame_index"]))


def _load_mask(mask_path: str | None) -> np.ndarray | None:
    if not mask_path:
        return None
    path = Path(mask_path)
    if not path.exists():
        return None
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def _dilate_mask(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if radius <= 0 or not binary.any():
        return binary
    dilated = binary.copy()
    for _ in range(int(radius)):
        padded = np.pad(dilated, 1, mode="constant", constant_values=False)
        expanded = padded[1:-1, 1:-1].copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                expanded |= padded[1 + dy : 1 + dy + dilated.shape[0], 1 + dx : 1 + dx + dilated.shape[1]]
        dilated = expanded
    return dilated


def _dataset_image_map(dataset_dir: Path) -> dict[str, Path]:
    entries = resolve_dataset_image_entries(dataset_dir)
    return {str(item["image_id"]): Path(str(item["image_path"])) for item in entries}


def _load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _bbox_masks(image_h: int, image_w: int, bbox_xyxy: list[float]) -> tuple[np.ndarray, np.ndarray]:
    left, top, right, bottom = [float(v) for v in bbox_xyxy]
    pos = np.zeros((image_h, image_w), dtype=bool)
    neg = np.zeros((image_h, image_w), dtype=bool)
    x0 = int(np.clip(np.floor(left), 0, image_w - 1))
    y0 = int(np.clip(np.floor(top), 0, image_h - 1))
    x1 = int(np.clip(np.ceil(right), 0, image_w - 1))
    y1 = int(np.clip(np.ceil(bottom), 0, image_h - 1))
    if x1 <= x0 or y1 <= y0:
        return pos, neg

    width = x1 - x0
    height = y1 - y0
    inset_x = max(int(round(0.20 * width)), 1)
    inset_y = max(int(round(0.20 * height)), 1)
    cx0 = min(x0 + inset_x, x1 - 1)
    cy0 = min(y0 + inset_y, y1 - 1)
    cx1 = max(x1 - inset_x, cx0 + 1)
    cy1 = max(y1 - inset_y, cy0 + 1)
    pos[cy0:cy1, cx0:cx1] = True

    out_x = max(int(round(0.25 * width)), 2)
    out_y = max(int(round(0.25 * height)), 2)
    ox0 = max(0, x0 - out_x)
    oy0 = max(0, y0 - out_y)
    ox1 = min(image_w, x1 + out_x)
    oy1 = min(image_h, y1 + out_y)
    neg[oy0:oy1, ox0:ox1] = True
    neg[y0:y1, x0:x1] = False
    return pos, neg


def _region_prototypes(
    image_rgb: np.ndarray,
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    pos_pixels = image_rgb[np.asarray(positive_mask, dtype=bool)]
    neg_pixels = image_rgb[np.asarray(negative_mask, dtype=bool)]
    pos_proto = pos_pixels.mean(axis=0).astype(np.float32) if pos_pixels.size else None
    neg_proto = neg_pixels.mean(axis=0).astype(np.float32) if neg_pixels.size else None
    return pos_proto, neg_proto


def _appearance_contrast(
    colors: np.ndarray,
    pos_proto: np.ndarray | None,
    neg_proto: np.ndarray | None,
    sigma: float = 0.18,
) -> tuple[np.ndarray, np.ndarray]:
    if pos_proto is None:
        return np.ones((colors.shape[0],), dtype=np.float32), np.zeros((colors.shape[0],), dtype=np.float32)
    pos_dist = np.linalg.norm(colors - pos_proto[None, :], axis=1)
    pos_sim = np.exp(-np.square(pos_dist / max(sigma, 1.0e-4))).astype(np.float32)
    if neg_proto is None:
        return pos_sim, np.zeros_like(pos_sim)
    neg_dist = np.linalg.norm(colors - neg_proto[None, :], axis=1)
    neg_sim = np.exp(-np.square(neg_dist / max(sigma, 1.0e-4))).astype(np.float32)
    return pos_sim, neg_sim


def _query_cluster_features(
    bank: dict[str, np.ndarray],
    sampled_indices: np.ndarray,
    opacity_sigmoid: np.ndarray | None,
    appearance_positive: np.ndarray,
    appearance_negative: np.ndarray,
) -> np.ndarray:
    trajectories = np.asarray(bank["trajectories"], dtype=np.float32)
    query_traj = trajectories[:, sampled_indices, :]
    query_mean = query_traj.mean(axis=1).astype(np.float32)
    query_delta = (query_traj[:, -1, :] - query_traj[:, 0, :]).astype(np.float32)
    query_path = np.linalg.norm(np.diff(query_traj, axis=1), axis=2).sum(axis=1, keepdims=True).astype(np.float32)
    xyz = np.asarray(bank.get("xyz", trajectories[:, 0, :]), dtype=np.float32)
    velocity = np.asarray(bank.get("velocity", np.zeros_like(query_delta)), dtype=np.float32)
    spatial_scale = np.asarray(bank.get("spatial_scale", np.ones_like(query_delta)), dtype=np.float32)
    scale_norm = np.linalg.norm(spatial_scale, axis=1, keepdims=True).astype(np.float32)
    support_center = np.asarray(bank["gate"], dtype=np.float32).reshape(trajectories.shape[0], trajectories.shape[1])
    time_values = np.asarray(bank["time_values"], dtype=np.float32).reshape(1, -1)
    gate_sum = np.clip(support_center.sum(axis=1), 1.0e-6, None)
    support_center = ((support_center * time_values).sum(axis=1) / gate_sum)[:, None].astype(np.float32)
    parts = [
        query_mean,
        query_delta,
        query_path,
        xyz,
        velocity,
        scale_norm,
        support_center,
        appearance_positive[:, None].astype(np.float32),
        appearance_negative[:, None].astype(np.float32),
    ]
    if opacity_sigmoid is not None:
        parts.append(np.asarray(opacity_sigmoid, dtype=np.float32)[:, None])
    features = np.concatenate(parts, axis=1).astype(np.float32)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return ((features - mean) / np.clip(std, 1.0e-6, None)).astype(np.float32)


def _cluster_refine_ids(
    pool_ids: np.ndarray,
    query_mean_xyz: np.ndarray,
    feature_matrix: np.ndarray,
    ranking_score: np.ndarray,
    hit_ratio: np.ndarray,
    opacity_sigmoid: np.ndarray | None,
    appearance_positive: np.ndarray,
    appearance_negative: np.ndarray,
    min_gaussians: int,
    max_gaussians: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.cluster.vq import kmeans2

    def _label_with_hdbscan(features: np.ndarray) -> np.ndarray | None:
        try:
            import hdbscan  # type: ignore
        except Exception:
            return None
        if features.shape[0] < 128:
            return None
        min_cluster_size = int(np.clip(features.shape[0] // 10, 96, 512))
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=max(min_cluster_size // 4, 8),
            cluster_selection_epsilon=0.02,
            allow_single_cluster=True,
            core_dist_n_jobs=1,
        )
        labels = clusterer.fit_predict(features)
        if np.all(labels < 0):
            return None
        return labels.astype(np.int32)

    pool_ids = np.asarray(pool_ids, dtype=np.int64)
    if pool_ids.size <= max(int(min_gaussians), 256):
        selected = pool_ids[: min(pool_ids.size, int(max_gaussians))]
        return selected, {
            "refine_cluster_count": 1,
            "refine_selected_cluster_size": int(selected.size),
        }

    k = int(np.clip(pool_ids.size // 768, 2, 6))
    pool_features = np.asarray(feature_matrix[pool_ids], dtype=np.float32)
    labels = _label_with_hdbscan(pool_features)
    centroids = None
    if labels is None:
        centroids, labels = kmeans2(pool_features, k, minit="points", iter=20)
        labels = labels.astype(np.int32)

    unique_labels = sorted(int(idx) for idx in np.unique(labels) if idx >= 0)
    cluster_scores: list[tuple[float, int]] = []
    for cluster_id in unique_labels:
        members = pool_ids[labels == cluster_id]
        if members.size == 0:
            continue
        xyz = query_mean_xyz[members]
        extent = np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))
        compactness = 1.0 / max(extent, 1.0e-3)
        score = (
            0.28 * float(np.mean(ranking_score[members]))
            + 0.16 * float(np.mean(hit_ratio[members]))
            + 0.18 * float(np.mean(appearance_positive[members]))
            - 0.14 * float(np.mean(appearance_negative[members]))
            + 0.12 * compactness
            + 0.12 * (0.0 if opacity_sigmoid is None else float(np.mean(opacity_sigmoid[members])))
            - 0.08 * min(float(members.size) / max(float(max_gaussians), 1.0), 1.0)
        )
        cluster_scores.append((score, int(cluster_id)))
    if not cluster_scores:
        selected = pool_ids[: min(pool_ids.size, int(max_gaussians))]
        return selected, {
            "refine_cluster_count": int(max(len(unique_labels), 1)),
            "refine_selected_cluster_size": int(selected.size),
        }

    cluster_scores.sort(reverse=True)
    best_cluster_id = cluster_scores[0][1]
    best_members = pool_ids[labels == best_cluster_id]
    if centroids is not None:
        centroid = centroids[best_cluster_id]
    else:
        centroid = pool_features[labels == best_cluster_id].mean(axis=0)
    if best_members.size < int(min_gaussians):
        remaining = pool_ids[labels != best_cluster_id]
        if remaining.size > 0:
            remaining_features = pool_features[labels != best_cluster_id]
            distances = np.linalg.norm(remaining_features - centroid[None, :], axis=1)
            order = np.argsort(distances, kind="mergesort")
            need = int(min_gaussians) - int(best_members.size)
            extra = remaining[order[:need]]
            best_members = np.concatenate([best_members, extra], axis=0)
    final_keep = int(np.clip(best_members.size, min_gaussians, max_gaussians))
    ranked = best_members[np.argsort(-ranking_score[best_members], kind="mergesort")]
    selected = ranked[:final_keep].astype(np.int64)
    return selected, {
        "refine_cluster_count": int(max(len(unique_labels), 1)),
        "refine_selected_cluster_size": int(selected.size),
    }


def _mask_components(
    mask: np.ndarray,
    min_area: int = 128,
    min_area_ratio: float = 0.08,
) -> list[dict[str, Any]]:
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return []
    labeled, num_labels = ndimage.label(binary.astype(np.uint8))
    if num_labels <= 0:
        return []
    sizes = ndimage.sum(binary, labeled, index=np.arange(1, num_labels + 1)).astype(np.float32)
    if sizes.size == 0:
        return []
    max_size = float(np.max(sizes))
    min_keep = max(float(min_area), max_size * float(min_area_ratio))
    components: list[dict[str, Any]] = []
    for label_id, size in enumerate(sizes.tolist(), start=1):
        if float(size) < min_keep:
            continue
        component = labeled == label_id
        ys, xs = np.nonzero(component)
        if ys.size == 0:
            continue
        xmin = int(xs.min())
        xmax = int(xs.max())
        ymin = int(ys.min())
        ymax = int(ys.max())
        center = np.array(
            [
                0.5 * float(xmin + xmax),
                0.5 * float(ymin + ymax),
            ],
            dtype=np.float32,
        )
        components.append(
            {
                "mask": component,
                "bbox_xyxy": [xmin, ymin, xmax, ymax],
                "center_xy": center,
                "area_px": int(size),
            }
        )
    components.sort(key=lambda item: (float(item["center_xy"][0]), -float(item["area_px"])))
    return components


def _split_frame_start(track: dict[str, Any]) -> int | None:
    active = [frame for frame in track.get("frames", []) if bool(frame.get("active"))]
    if not active:
        return None
    streak = 0
    for frame in active:
        if int(frame.get("component_count", 0)) >= 2:
            streak += 1
        else:
            streak = 0
        if streak >= 2:
            return int(frame["frame_index"])
    for frame in active:
        if int(frame.get("component_count", 0)) >= 2:
            return int(frame["frame_index"])
    return None


def _qwen_transition_hint_frame(
    phrase: str,
    query_plan: dict[str, Any] | None,
    active_frames: list[dict[str, Any]],
) -> int | None:
    if not query_plan or not active_frames:
        return None
    context_frames = query_plan.get("context_frames", [])
    if not isinstance(context_frames, list) or not context_frames:
        return None
    phrase_norm = " ".join(str(phrase).strip().lower().split())
    hints = query_plan.get("phase_transition_hints", [])
    if not isinstance(hints, list):
        return None
    matched_hint = None
    for hint in hints:
        if " ".join(str(hint.get("phrase", "")).strip().lower().split()) == phrase_norm:
            matched_hint = hint
            break
    if matched_hint is None:
        return None

    def _frame_index(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    def _slot_frame(slot_value: Any) -> int | None:
        if slot_value is None:
            return None
        try:
            slot_index = int(slot_value)
        except Exception:
            return None
        if slot_index < 0 or slot_index >= len(context_frames):
            return None
        try:
            return int(context_frames[slot_index]["frame_index"])
        except Exception:
            return None

    last_pre_frame = _frame_index(matched_hint.get("last_pre_change_frame_index"))
    first_post_frame = _frame_index(matched_hint.get("first_post_change_frame_index"))
    if last_pre_frame is None:
        last_pre_frame = _slot_frame(matched_hint.get("last_pre_change_slot"))
    if first_post_frame is None:
        first_post_frame = _slot_frame(matched_hint.get("first_post_change_slot"))
    if last_pre_frame is not None and first_post_frame is not None:
        hint_frame = int(round(0.5 * (last_pre_frame + first_post_frame)))
    else:
        hint_frame = first_post_frame if first_post_frame is not None else last_pre_frame
    if hint_frame is None:
        return None

    active_indices = np.asarray([int(frame["frame_index"]) for frame in active_frames], dtype=np.int32)
    nearest_index = int(np.abs(active_indices - int(hint_frame)).argmin())
    return int(active_indices[nearest_index])


def _merged_split_frame(
    phrase: str,
    track: dict[str, Any],
    query_plan: dict[str, Any] | None,
) -> int | None:
    active = [frame for frame in track.get("frames", []) if bool(frame.get("active"))]
    if not active:
        return None
    visible_split = _split_frame_start(track)
    hinted_split = _qwen_transition_hint_frame(phrase=phrase, query_plan=query_plan, active_frames=active)
    if visible_split is None:
        return hinted_split
    if hinted_split is None:
        return visible_split
    return int(min(visible_split, hinted_split))


def _subsample_frames(frames: list[dict[str, Any]], max_track_frames: int) -> list[dict[str, Any]]:
    if len(frames) <= int(max_track_frames):
        return list(frames)
    indices = np.linspace(0, len(frames) - 1, num=int(max_track_frames), dtype=np.int32)
    return [frames[int(index)] for index in indices.tolist()]


def _phase_aware_track_variants(
    phrase: str,
    track: dict[str, Any],
    max_track_frames: int,
    query_plan: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    active = [frame for frame in track.get("frames", []) if bool(frame.get("active")) and frame.get("bbox_xyxy") is not None]
    active = sorted(active, key=lambda item: int(item["frame_index"]))
    if not active:
        return []

    split_frame = _merged_split_frame(phrase=phrase, track=track, query_plan=query_plan)
    if split_frame is None:
        return [
            {
                "alias": str(phrase),
                "base_phrase": str(phrase),
                "variant_kind": "main",
                "phase": "full",
                "description": f"Query-guided entity for '{phrase}' across its full visible support.",
                "frames": _subsample_frames(active, max_track_frames=max_track_frames),
            }
        ]

    pre_frames = [frame for frame in active if int(frame["frame_index"]) < int(split_frame)]
    post_frames = [frame for frame in active if int(frame["frame_index"]) >= int(split_frame)]
    variants: list[dict[str, Any]] = []

    if pre_frames:
        variants.append(
            {
                "alias": f"{phrase}__pre_split",
                "base_phrase": str(phrase),
                "variant_kind": "pre_split",
                "phase": "pre_split",
                "description": f"Query-guided entity for '{phrase}' before the tracked object splits into multiple components.",
                "frames": _subsample_frames(pre_frames, max_track_frames=max_track_frames),
            }
        )

    if post_frames:
        variants.append(
            {
                "alias": f"{phrase}__post_split_union",
                "base_phrase": str(phrase),
                "variant_kind": "post_split_union",
                "phase": "post_split",
                "description": f"Query-guided entity for all visible '{phrase}' components after the object split.",
                "frames": _subsample_frames(post_frames, max_track_frames=max_track_frames),
            }
        )

        per_component_frames: dict[int, list[dict[str, Any]]] = {}
        for frame in post_frames:
            mask = _load_mask(frame.get("mask_path"))
            if mask is None:
                continue
            components = _mask_components(mask, min_area=96, min_area_ratio=0.10)
            if len(components) < 2:
                continue
            for component_index, component in enumerate(components[:4]):
                component_frame = dict(frame)
                component_frame["mask_array"] = component["mask"]
                component_frame["bbox_xyxy"] = component["bbox_xyxy"]
                component_frame["component_index"] = int(component_index)
                per_component_frames.setdefault(int(component_index), []).append(component_frame)

        for component_index, frames in sorted(per_component_frames.items()):
            if len(frames) < 2:
                continue
            variants.append(
                {
                    "alias": f"{phrase}__post_split_part_{component_index}",
                    "base_phrase": str(phrase),
                    "variant_kind": "post_split_part",
                    "phase": "post_split",
                    "description": f"Query-guided component proposal {component_index} for '{phrase}' after the tracked object split.",
                    "frames": _subsample_frames(frames, max_track_frames=max_track_frames),
                }
            )

    if not variants:
        variants.append(
            {
                "alias": str(phrase),
                "base_phrase": str(phrase),
                "variant_kind": "main",
                "phase": "full",
                "description": f"Query-guided entity for '{phrase}' across its full visible support.",
                "frames": _subsample_frames(active, max_track_frames=max_track_frames),
            }
        )
    return variants


def _bbox_area(bbox_xyxy: list[float]) -> float:
    left, top, right, bottom = [float(value) for value in bbox_xyxy]
    width = max(right - left, 1.0)
    height = max(bottom - top, 1.0)
    return float(width * height)


def _select_phrase_gaussians(
    phrase: str,
    sampled_frames: list[dict[str, Any]],
    dataset_dir: Path,
    bank: dict[str, np.ndarray],
    proposal_keep_ratio: float,
    min_gaussians: int,
    max_gaussians: int,
    chunk_size: int,
    opacity_sigmoid: np.ndarray | None = None,
    opacity_power: float = 0.0,
    cluster_mode: str = "support_only",
    seed_ratio: float = 0.05,
    expansion_factor: float = 4.0,
) -> dict[str, Any]:
    trajectories = np.asarray(bank["trajectories"], dtype=np.float32)
    gate = np.asarray(bank["gate"], dtype=np.float32).reshape(trajectories.shape[0], trajectories.shape[1])
    time_values = np.asarray(bank["time_values"], dtype=np.float32).reshape(-1)
    spatial_scale = np.asarray(bank["spatial_scale"], dtype=np.float32)
    num_gaussians = trajectories.shape[0]
    entity_type = _phrase_entity_type(phrase)

    if not sampled_frames:
        raise ValueError(f"No active search frames found for phrase '{phrase}'.")
    sampled_frames = sorted(sampled_frames, key=lambda item: int(item["frame_index"]))
    sampled_times = np.asarray([float(frame["time_value"]) for frame in sampled_frames], dtype=np.float32)
    sampled_indices = _resample_indices(source_times=time_values, target_times=sampled_times)
    image_path_by_id = _dataset_image_map(dataset_dir)

    positive_score_accum = np.zeros((num_gaussians,), dtype=np.float32)
    negative_score_accum = np.zeros((num_gaussians,), dtype=np.float32)
    appearance_positive_accum = np.zeros((num_gaussians,), dtype=np.float32)
    appearance_negative_accum = np.zeros((num_gaussians,), dtype=np.float32)
    appearance_weight_accum = np.zeros((num_gaussians,), dtype=np.float32)
    hit_count = np.zeros((num_gaussians,), dtype=np.float32)
    negative_hit_count = np.zeros((num_gaussians,), dtype=np.float32)
    areas = np.asarray([_bbox_area(frame["bbox_xyxy"]) for frame in sampled_frames], dtype=np.float32)

    for sampled_frame, bank_index in zip(sampled_frames, sampled_indices.tolist()):
        camera = Camera.from_json(dataset_dir / "camera" / f"{sampled_frame['image_id']}.json")
        left, top, right, bottom = [float(value) for value in sampled_frame["bbox_xyxy"]]
        mask = np.asarray(sampled_frame.get("mask_array"), dtype=bool) if sampled_frame.get("mask_array") is not None else _load_mask(sampled_frame.get("mask_path"))
        positive_context_mask = _dilate_mask(mask, radius=2) if mask is not None else None
        negative_ring_mask = None
        positive_region_mask = None
        negative_region_mask = None
        if mask is not None:
            outer_mask = _dilate_mask(mask, radius=7)
            negative_ring_mask = np.logical_and(outer_mask, np.logical_not(positive_context_mask))
            positive_region_mask = mask
            negative_region_mask = negative_ring_mask
        image_rgb = None
        pos_proto = None
        neg_proto = None
        image_path = image_path_by_id.get(str(sampled_frame["image_id"]))
        if image_path is not None and image_path.exists():
            image_rgb = _load_rgb_image(image_path)
            if positive_region_mask is None or negative_region_mask is None:
                positive_region_mask, negative_region_mask = _bbox_masks(
                    image_h=image_rgb.shape[0],
                    image_w=image_rgb.shape[1],
                    bbox_xyxy=[left, top, right, bottom],
                )
            pos_proto, neg_proto = _region_prototypes(image_rgb, positive_region_mask, negative_region_mask)
            try:
                feature_map = feature_map_from_image(image_rgb)
                pos_proto_dense, neg_proto_dense = prototypes_from_masks(feature_map, positive_region_mask, negative_region_mask)
            except Exception:
                feature_map = None
                pos_proto_dense = None
                neg_proto_dense = None
        else:
            feature_map = None
            pos_proto_dense = None
            neg_proto_dense = None

        for start in range(0, num_gaussians, chunk_size):
            end = min(start + chunk_size, num_gaussians)
            points = trajectories[start:end, int(bank_index), :]
            local_points = camera.points_to_local_points(points)
            depth_valid = local_points[:, 2] > 1.0e-4
            pixels = camera.project(points)
            camera_width = float(np.asarray(camera.image_size)[0])
            camera_height = float(np.asarray(camera.image_size)[1])
            projection_valid = (
                depth_valid
                & np.all(np.isfinite(pixels), axis=1)
                & (pixels[:, 0] >= 0.0)
                & (pixels[:, 0] <= max(camera_width - 1.0, 0.0))
                & (pixels[:, 1] >= 0.0)
                & (pixels[:, 1] <= max(camera_height - 1.0, 0.0))
            )
            if not np.any(projection_valid):
                continue
            depth_values = np.asarray(local_points[:, 2], dtype=np.float32)
            valid_depth_values = depth_values[projection_valid]
            near_depth = float(np.quantile(valid_depth_values, 0.05))
            far_depth = float(np.quantile(valid_depth_values, 0.90))
            depth_span = max(far_depth - near_depth, 1.0e-4)
            depth_weight = np.exp(-np.clip((depth_values - near_depth) / depth_span, 0.0, 8.0)).astype(np.float32)
            pos_app = np.ones((end - start,), dtype=np.float32)
            neg_app = np.zeros((end - start,), dtype=np.float32)
            if image_rgb is not None:
                image_h, image_w = image_rgb.shape[:2]
                scale_x_img = float(image_w) / max(float(np.asarray(camera.image_size)[0]), 1.0)
                scale_y_img = float(image_h) / max(float(np.asarray(camera.image_size)[1]), 1.0)
                valid_pixels = pixels[projection_valid]
                xs_img_valid = np.clip(np.rint(valid_pixels[:, 0] * scale_x_img).astype(np.int64), 0, image_w - 1)
                ys_img_valid = np.clip(np.rint(valid_pixels[:, 1] * scale_y_img).astype(np.int64), 0, image_h - 1)
                colors = image_rgb[ys_img_valid, xs_img_valid]
                pos_app_valid, neg_app_valid = _appearance_contrast(colors, pos_proto, neg_proto)
                if feature_map is not None:
                    dense_features = sample_feature_map(
                        feature_map,
                        np.stack([xs_img_valid, ys_img_valid], axis=1),
                        image_h=image_h,
                        image_w=image_w,
                    )
                    pos_dense, neg_dense = prototype_similarity(dense_features, pos_proto_dense, neg_proto_dense)
                    pos_app_valid = 0.45 * pos_app_valid + 0.55 * pos_dense
                    neg_app_valid = 0.45 * neg_app_valid + 0.55 * neg_dense
                pos_app[projection_valid] = pos_app_valid
                neg_app[projection_valid] = neg_app_valid
            if mask is not None:
                mask_h, mask_w = mask.shape[:2]
                image_width = float(np.asarray(camera.image_size)[0])
                image_height = float(np.asarray(camera.image_size)[1])
                scale_x = float(mask_w) / max(image_width, 1.0)
                scale_y = float(mask_h) / max(image_height, 1.0)
                xs = np.zeros((end - start,), dtype=np.int64)
                ys = np.zeros((end - start,), dtype=np.int64)
                valid_pixels = pixels[projection_valid]
                xs_valid = np.clip(np.rint(valid_pixels[:, 0] * scale_x).astype(np.int64), 0, mask_w - 1)
                ys_valid = np.clip(np.rint(valid_pixels[:, 1] * scale_y).astype(np.int64), 0, mask_h - 1)
                xs[projection_valid] = xs_valid
                ys[projection_valid] = ys_valid
                inside_mask = np.zeros((end - start,), dtype=bool)
                inside_mask[projection_valid] = mask[ys_valid, xs_valid]
                if positive_context_mask is not None:
                    context_mask = np.zeros((end - start,), dtype=bool)
                    context_mask[projection_valid] = positive_context_mask[ys_valid, xs_valid]
                else:
                    context_mask = inside_mask.copy()
                if negative_ring_mask is not None:
                    ring_mask = np.zeros((end - start,), dtype=bool)
                    ring_mask[projection_valid] = negative_ring_mask[ys_valid, xs_valid]
                else:
                    ring_mask = np.zeros_like(inside_mask)
                front_inside_mask = inside_mask.copy()
                back_inside_mask = np.zeros_like(inside_mask)
                if entity_type == "object" and np.any(inside_mask):
                    inside_depth = depth_values[inside_mask]
                    front_depth = float(np.quantile(inside_depth, 0.45))
                    front_inside_mask = inside_mask & (depth_values <= front_depth + 0.05 * depth_span)
                    back_inside_mask = np.logical_and(inside_mask, np.logical_not(front_inside_mask))
                    if positive_context_mask is not None:
                        context_mask = context_mask & (depth_values <= front_depth + 0.15 * depth_span)
                gate_weight = gate[start:end, int(bank_index)]
                positive_frame_score = gate_weight * depth_weight * (
                    1.10 * front_inside_mask.astype(np.float32)
                    + 0.35 * inside_mask.astype(np.float32)
                    + 0.10 * context_mask.astype(np.float32)
                ) * (0.35 + 0.65 * pos_app)
                negative_frame_score = gate_weight * depth_weight * (
                    1.00 * ring_mask.astype(np.float32) + 0.35 * back_inside_mask.astype(np.float32)
                ) * (0.20 + 0.80 * neg_app)
                positive_score_accum[start:end] += positive_frame_score
                negative_score_accum[start:end] += negative_frame_score
                appearance_positive_accum[start:end] += gate_weight * depth_weight * pos_app
                appearance_negative_accum[start:end] += gate_weight * depth_weight * neg_app
                appearance_weight_accum[start:end] += gate_weight * depth_weight
                hit_count[start:end] += inside_mask.astype(np.float32)
                negative_hit_count[start:end] += (ring_mask.astype(np.float32) + 0.35 * back_inside_mask.astype(np.float32))
                continue
            inside = (
                depth_valid
                & (pixels[:, 0] >= left)
                & (pixels[:, 0] <= right)
                & (pixels[:, 1] >= top)
                & (pixels[:, 1] <= bottom)
            )
            gate_weight = gate[start:end, int(bank_index)]
            frame_score = gate_weight * depth_weight * inside.astype(np.float32) * (0.35 + 0.65 * pos_app)
            positive_score_accum[start:end] += frame_score
            appearance_positive_accum[start:end] += gate_weight * depth_weight * pos_app
            appearance_negative_accum[start:end] += gate_weight * depth_weight * neg_app
            appearance_weight_accum[start:end] += gate_weight * depth_weight
            hit_count[start:end] += inside.astype(np.float32)

    num_sampled = max(int(len(sampled_frames)), 1)
    hit_ratio = hit_count / float(num_sampled)
    negative_hit_ratio = negative_hit_count / float(num_sampled)
    positive_score = positive_score_accum / float(num_sampled)
    negative_score = negative_score_accum / float(num_sampled)
    appearance_positive = appearance_positive_accum / np.clip(appearance_weight_accum, 1.0e-6, None)
    appearance_negative = appearance_negative_accum / np.clip(appearance_weight_accum, 1.0e-6, None)
    contrastive_margin = positive_score - 0.70 * negative_score
    exclusive_ratio = positive_score / np.clip(positive_score + negative_score, 1.0e-6, None)
    support_score = (
        0.48 * positive_score
        + 0.22 * hit_ratio
        + 0.18 * np.clip(contrastive_margin, 0.0, None)
        + 0.12 * exclusive_ratio
        - 0.08 * negative_hit_ratio
    ).astype(np.float32)
    ranking_score = support_score.copy()
    if opacity_sigmoid is not None and float(opacity_power) > 0.0:
        opacity_weight = np.power(np.clip(np.asarray(opacity_sigmoid, dtype=np.float32), 1.0e-6, 1.0), float(opacity_power))
        ranking_score = support_score * opacity_weight
    max_support = float(np.max(support_score)) if support_score.size else 0.0
    min_hits = max(1, int(np.ceil(0.20 * num_sampled)))
    eligible = np.where(
        (support_score > 0.0)
        & (
            (hit_count >= float(min_hits))
            | (support_score >= 0.55 * max(max_support, 1.0e-6))
        )
        & (exclusive_ratio >= 0.45)
    )[0]
    if eligible.size == 0:
        eligible = np.where(support_score > 0.0)[0]
    if eligible.size == 0:
        raise ValueError(f"No Gaussian support found for phrase '{phrase}'.")

    refine_summary = {
        "refine_cluster_count": None,
        "refine_selected_cluster_size": None,
    }
    if cluster_mode == "worldtube_consistency":
        if opacity_sigmoid is None:
            raise ValueError("worldtube_consistency mode requires opacity_sigmoid")
        consistency = select_worldtube_consistency_cluster(
            bank=bank,
            sampled_indices=np.asarray(sampled_indices, dtype=np.int32),
            support_score=support_score,
            hit_count=hit_count,
            opacity_sigmoid=opacity_sigmoid,
            proposal_keep_ratio=proposal_keep_ratio,
            min_gaussians=min_gaussians,
            max_gaussians=max_gaussians,
            seed_ratio=float(seed_ratio),
            expansion_factor=float(expansion_factor),
        )
        selected = consistency.selected_ids
        selected_scores = consistency.selected_scores
        extra_summary = dict(consistency.summary)
    else:
        keep_count = int(
            np.clip(
                round(float(eligible.size) * float(proposal_keep_ratio)),
                int(min_gaussians),
                int(max_gaussians),
            )
        )
        keep_count = min(keep_count, int(eligible.size))
        ranked = eligible[np.argsort(-ranking_score[eligible], kind="mergesort")]
        selected = ranked[:keep_count].astype(np.int64)
        selected_scores = ranking_score[selected].astype(np.float32)
        selected_scores /= max(float(selected_scores.max()), 1.0e-6)
        extra_summary = {
            "seed_count": None,
            "candidate_count": int(eligible.size),
            "selected_count": int(selected.size),
            "mean_feature_similarity": None,
            "mean_proximity_score": None,
            "mean_overlap_score": None,
        }

    pool_size = int(np.clip(max(selected.shape[0] * 3, min_gaussians * 2), min_gaussians, min(max_gaussians * 2, eligible.size)))
    pool_ids = eligible[np.argsort(-ranking_score[eligible], kind="mergesort")[:pool_size]]
    cluster_features = _query_cluster_features(
        bank=bank,
        sampled_indices=np.asarray(sampled_indices, dtype=np.int32),
        opacity_sigmoid=opacity_sigmoid,
        appearance_positive=appearance_positive,
        appearance_negative=appearance_negative,
    )
    query_mean_xyz = np.asarray(bank["trajectories"], dtype=np.float32)[:, sampled_indices, :].mean(axis=1).astype(np.float32)
    selected, refine_summary = _cluster_refine_ids(
        pool_ids=pool_ids,
        query_mean_xyz=query_mean_xyz,
        feature_matrix=cluster_features,
        ranking_score=ranking_score,
        hit_ratio=hit_ratio,
        opacity_sigmoid=opacity_sigmoid,
        appearance_positive=appearance_positive,
        appearance_negative=appearance_negative,
        min_gaussians=min_gaussians,
        max_gaussians=max_gaussians,
    )
    selected_scores = ranking_score[selected].astype(np.float32)
    selected_scores /= max(float(selected_scores.max()), 1.0e-6)

    mean_area = float(np.mean(areas)) if areas.size else 0.0
    source_keyframes = sorted(set(int(index) for index in sampled_indices.tolist()))
    return {
        "phrase": str(phrase),
        "entity_type": _phrase_entity_type(phrase),
        "selected_gaussian_ids": selected,
        "selected_scores": selected_scores,
        "sampled_frames": sampled_frames,
        "sampled_indices": np.asarray(sampled_indices, dtype=np.int32),
        "mean_mask_area": mean_area,
        "mean_hit_ratio": float(np.mean(hit_ratio[selected])) if selected.size else 0.0,
        "mean_support_score": float(np.mean(support_score[selected])) if selected.size else 0.0,
        "mean_ranking_score": float(np.mean(ranking_score[selected])) if selected.size else 0.0,
        "mean_opacity_sigmoid": float(np.mean(opacity_sigmoid[selected])) if (selected.size and opacity_sigmoid is not None) else None,
        "mean_negative_score": float(np.mean(negative_score[selected])) if selected.size else 0.0,
        "mean_exclusive_ratio": float(np.mean(exclusive_ratio[selected])) if selected.size else 0.0,
        "mean_appearance_positive": float(np.mean(appearance_positive[selected])) if selected.size else 0.0,
        "mean_appearance_negative": float(np.mean(appearance_negative[selected])) if selected.size else 0.0,
        "cluster_mode": str(cluster_mode),
        **extra_summary,
        **refine_summary,
        "keyframes": source_keyframes[:8],
    }


def _phrase_world_payload(
    phrase_payload: dict[str, Any],
    bank: dict[str, np.ndarray],
) -> dict[str, Any]:
    time_values = np.asarray(bank["time_values"], dtype=np.float32).reshape(-1)
    trajectories = np.asarray(bank["trajectories"], dtype=np.float32)
    gate = np.asarray(bank["gate"], dtype=np.float32).reshape(trajectories.shape[0], trajectories.shape[1])
    spatial_scale = np.asarray(bank["spatial_scale"], dtype=np.float32)
    gaussian_ids = np.asarray(phrase_payload["selected_gaussian_ids"], dtype=np.int64)
    base_scores = np.asarray(phrase_payload["selected_scores"], dtype=np.float32)

    center_world = np.zeros((time_values.shape[0], 3), dtype=np.float32)
    center_valid = np.zeros((time_values.shape[0],), dtype=bool)
    bbox_world = np.zeros((time_values.shape[0], 6), dtype=np.float32)
    bbox_valid = np.zeros((time_values.shape[0],), dtype=bool)
    visibility = np.zeros((time_values.shape[0],), dtype=bool)

    per_time_weights = gate[gaussian_ids] * base_scores[:, None]
    mask_area = np.zeros((time_values.shape[0],), dtype=np.float32)
    source_indices = np.asarray(phrase_payload["sampled_indices"], dtype=np.int32)
    source_areas = np.asarray(
        [_bbox_area(frame["bbox_xyxy"]) for frame in phrase_payload["sampled_frames"]],
        dtype=np.float32,
    )
    if source_indices.size > 0:
        target_indices = np.arange(time_values.shape[0], dtype=np.int32)
        nearest = np.abs(target_indices[:, None] - source_indices[None, :]).argmin(axis=1).astype(np.int32)
        mask_area = source_areas[nearest]

    for time_index in range(time_values.shape[0]):
        weights = per_time_weights[:, time_index]
        active = weights > max(float(weights.max()) * 0.05, 1.0e-5)
        if not np.any(active):
            continue
        member_ids = gaussian_ids[active]
        member_weights = weights[active]
        member_points = trajectories[member_ids, time_index, :]
        member_scale = spatial_scale[member_ids]
        center_world[time_index] = np.average(member_points, weights=member_weights, axis=0).astype(np.float32)
        mins = np.min(member_points - member_scale, axis=0)
        maxs = np.max(member_points + member_scale, axis=0)
        bbox_world[time_index] = np.concatenate([mins, maxs], axis=0).astype(np.float32)
        center_valid[time_index] = True
        bbox_valid[time_index] = True
        visibility[time_index] = True

    quality = float(np.clip(base_scores.mean(), 0.0, 1.0))
    keyframes = [int(index) for index in phrase_payload["keyframes"]]
    segments = _phrase_segments(keyframes)
    visibility_ratio = float(visibility.mean())
    return {
        "center_world": center_world,
        "center_valid": center_valid,
        "bbox_world": bbox_world,
        "bbox_valid": bbox_valid,
        "visibility": visibility,
        "mask_area": mask_area,
        "quality": quality,
        "keyframes": keyframes,
        "segments": segments,
        "visibility_ratio": visibility_ratio,
    }


def build_query_proposal_dir(
    run_dir: str | Path,
    dataset_dir: str | Path,
    tracks_path: str | Path,
    output_dir: str | Path,
    max_track_frames: int = 16,
    proposal_keep_ratio: float = 0.03,
    min_gaussians: int = 256,
    max_gaussians: int = 4096,
    chunk_size: int = 4096,
    opacity_power: float = 0.0,
    cluster_mode: str = "support_only",
    seed_ratio: float = 0.05,
    expansion_factor: float = 4.0,
) -> Path:
    run_dir = Path(run_dir)
    dataset_dir = Path(dataset_dir)
    tracks_path = Path(tracks_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    track_payload = _read_json(tracks_path)
    query_plan_path = tracks_path.parent.parent / "query_plan.json"
    query_plan = _read_json(query_plan_path) if query_plan_path.exists() else None
    tracks = [track for track in track_payload.get("tracks", []) if str(track.get("status", "")) == "seeded"]
    if not tracks:
        raise ValueError(f"No seeded phrase tracks found in {tracks_path}")

    bank = _load_bank(run_dir / "entitybank")
    opacity_sigmoid = load_opacity_sigmoid(run_dir) if float(opacity_power) > 0.0 else None
    phrase_rows: list[dict[str, Any]] = []
    entities_json_rows: list[dict[str, Any]] = []
    phrase_payloads: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    time_values = np.asarray(bank["time_values"], dtype=np.float32).reshape(-1)
    for track in tracks:
        base_phrase = str(track["phrase"])
        for variant in _phase_aware_track_variants(
            base_phrase,
            track,
            max_track_frames=max_track_frames,
            query_plan=query_plan,
        ):
            try:
                phrase_selection = _select_phrase_gaussians(
                    phrase=variant["alias"],
                    sampled_frames=variant["frames"],
                    dataset_dir=dataset_dir,
                    bank=bank,
                    proposal_keep_ratio=proposal_keep_ratio,
                    min_gaussians=min_gaussians,
                    max_gaussians=max_gaussians,
                    chunk_size=chunk_size,
                    opacity_sigmoid=opacity_sigmoid,
                    opacity_power=float(opacity_power),
                    cluster_mode=str(cluster_mode),
                    seed_ratio=float(seed_ratio),
                    expansion_factor=float(expansion_factor),
                )
            except ValueError as _phrase_err:
                # Phrase has no Gaussian support in the reconstruction — skip gracefully.
                # This can happen for post-split sub-variants that were tracked by GSAM2
                # but don't have a distinct Gaussian cluster in the bank.
                print(f"[query_proposal_bridge] WARNING: Skipping phrase '{variant['alias']}': {_phrase_err}")
                continue
            world_payload = _phrase_world_payload(phrase_selection, bank=bank)
            phrase_payloads.append((track, variant, {"selection": phrase_selection, "world": world_payload}))

    num_entities = len(phrase_payloads)
    centroid_world = np.zeros((num_entities, time_values.shape[0], 3), dtype=np.float32)
    centroid_world_valid = np.zeros((num_entities, time_values.shape[0]), dtype=bool)
    bbox_world = np.zeros((num_entities, time_values.shape[0], 6), dtype=np.float32)
    bbox_world_valid = np.zeros((num_entities, time_values.shape[0]), dtype=bool)
    visibility = np.zeros((num_entities, time_values.shape[0]), dtype=bool)
    mask_area = np.zeros((num_entities, time_values.shape[0]), dtype=np.float32)
    quality = np.zeros((num_entities,), dtype=np.float32)

    for entity_id, (track, variant, payload) in enumerate(phrase_payloads):
        phrase_selection = payload["selection"]
        world_payload = payload["world"]
        base_phrase = str(variant["base_phrase"])
        proposal_alias = str(variant["alias"])

        centroid_world[entity_id] = world_payload["center_world"]
        centroid_world_valid[entity_id] = world_payload["center_valid"]
        bbox_world[entity_id] = world_payload["bbox_world"]
        bbox_world_valid[entity_id] = world_payload["bbox_valid"]
        visibility[entity_id] = world_payload["visibility"]
        mask_area[entity_id] = world_payload["mask_area"]
        quality[entity_id] = float(world_payload["quality"])

        phrase_rows.append(
            {
                "id": int(entity_id),
                "phrase": base_phrase,
                "proposal_alias": proposal_alias,
                "phase": str(variant["phase"]),
                "variant_kind": str(variant["variant_kind"]),
                "entity_type": phrase_selection["entity_type"],
                "selected_gaussian_count": int(len(phrase_selection["selected_gaussian_ids"])),
                "quality": float(world_payload["quality"]),
                "visibility_ratio": float(world_payload["visibility_ratio"]),
                "mean_mask_area": float(np.mean(world_payload["mask_area"][world_payload["mask_area"] > 0.0])) if np.any(world_payload["mask_area"] > 0.0) else 0.0,
                "mean_hit_ratio": float(phrase_selection["mean_hit_ratio"]),
                "mean_support_score": float(phrase_selection["mean_support_score"]),
                "mean_ranking_score": float(phrase_selection["mean_ranking_score"]),
                "mean_opacity_sigmoid": phrase_selection["mean_opacity_sigmoid"],
                "cluster_mode": phrase_selection["cluster_mode"],
                "seed_count": phrase_selection.get("seed_count"),
                "candidate_count": phrase_selection.get("candidate_count"),
                "mean_feature_similarity": phrase_selection.get("mean_feature_similarity"),
                "mean_proximity_score": phrase_selection.get("mean_proximity_score"),
                "mean_overlap_score": phrase_selection.get("mean_overlap_score"),
                "keyframes": world_payload["keyframes"],
                "segments": world_payload["segments"],
            }
        )
        entities_json_rows.append(
            {
                "id": int(entity_id),
                "static_text": base_phrase,
                "proposal_alias": proposal_alias,
                "proposal_phase": str(variant["phase"]),
                "proposal_variant": str(variant["variant_kind"]),
                "global_desc": str(variant["description"]),
                "dyn_desc": [str(variant["description"])],
                "gaussian_ids": phrase_selection["selected_gaussian_ids"].astype(int).tolist(),
                "gaussian_scores": phrase_selection["selected_scores"].astype(float).tolist(),
                "visibility_ratio": float(world_payload["visibility_ratio"]),
                "mean_mask_area": float(np.mean(world_payload["mask_area"][world_payload["mask_area"] > 0.0])) if np.any(world_payload["mask_area"] > 0.0) else 0.0,
                "quality": float(world_payload["quality"]),
                "entity_type": phrase_selection["entity_type"],
                "role_hints": [],
                "keyframes": world_payload["keyframes"],
                "segments": world_payload["segments"],
            }
        )

    torch.save(
        {
            "time_values": torch.from_numpy(time_values.astype(np.float32)),
            "centroid_world": torch.from_numpy(centroid_world),
            "centroid_world_valid": torch.from_numpy(centroid_world_valid),
            "bbox_world": torch.from_numpy(bbox_world),
            "bbox_world_valid": torch.from_numpy(bbox_world_valid),
            "visibility": torch.from_numpy(visibility),
            "mask_area": torch.from_numpy(mask_area),
            "quality": torch.from_numpy(quality),
        },
        output_dir / "entities.pt",
    )

    entities_payload = {
        "schema_version": 1,
        "source_tracks_path": str(tracks_path),
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "num_entities": int(len(entities_json_rows)),
        "frame_count": int(time_values.shape[0]),
        "time_values": time_values.astype(float).tolist(),
        "entities": entities_json_rows,
    }
    _write_json(output_dir / "entities.json", entities_payload)
    _write_json(
        output_dir / "query_proposal_summary.json",
        {
            "schema_version": 1,
            "run_dir": str(run_dir),
            "dataset_dir": str(dataset_dir),
            "tracks_path": str(tracks_path),
            "num_entities": int(len(phrase_rows)),
            "phrases": phrase_rows,
            "params": {
                "max_track_frames": int(max_track_frames),
                "proposal_keep_ratio": float(proposal_keep_ratio),
                "min_gaussians": int(min_gaussians),
                "max_gaussians": int(max_gaussians),
                "chunk_size": int(chunk_size),
                "opacity_power": float(opacity_power),
                "cluster_mode": str(cluster_mode),
                "seed_ratio": float(seed_ratio),
                "expansion_factor": float(expansion_factor),
            },
        },
    )
    return output_dir
