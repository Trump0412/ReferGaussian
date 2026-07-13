from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree

from refergaussian.entitybank.tube_bank import load_gaussian_state
from refergaussian.semantics.query_proposal_bridge import (
    _phase_aware_track_variants,
    _phrase_entity_type,
    _phrase_world_payload,
    _read_json,
    _resample_indices,
    _write_json,
)
from refergaussian.semantics.query_proposal_bridge import Camera
from refergaussian.semantics.semantic_renderer import (
    PreparedSemanticFrame,
    alignment_metrics,
    gaussian_region_alpha_masses,
    prepare_semantic_frame_inputs,
    render_selection_mask,
)
from refergaussian.semantics.surface_mask_field import build_mask_regions


@dataclass
class LiftingSample:
    frame: dict[str, Any]
    bank_index: int
    mask: np.ndarray
    outer_mask: np.ndarray
    prepared: PreparedSemanticFrame


@dataclass
class LiftingSupport:
    alias: str
    base_phrase: str
    source_object_id: int | None
    source_track_id: str
    source_instance_group_id: str | None
    source_instance_index: int | None
    phase: str
    variant_kind: str
    description: str
    entity_type: str
    sampled_frames: list[dict[str, Any]]
    sampled_indices: np.ndarray
    samples: list[LiftingSample]
    distance_regions: list[dict[str, np.ndarray]]
    presence: np.ndarray
    core_presence: np.ndarray
    full_mean: np.ndarray
    support_score: np.ndarray
    positive_score: np.ndarray
    negative_score: np.ndarray
    visible_count: np.ndarray
    distance_positive_score: np.ndarray
    distance_negative_score: np.ndarray
    distance_force_add: np.ndarray
    distance_force_drop: np.ndarray
    hit_count: np.ndarray
    core_hit_count: np.ndarray
    purity: np.ndarray
    core_ratio: np.ndarray
    outer_ratio: np.ndarray
    mean_mask_area: float


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    text = str(raw).strip().lower()
    if not text:
        return bool(default)
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _lifting_mode() -> str:
    for key in ("QUERY_LIFT_MODE", "QUERY_LIFT_CANDIDATE_MODE"):
        value = os.environ.get(key)
        if value:
            normalized = "_".join(str(value).strip().lower().replace("-", "_").split())
            # mask_coverage_refine_v3 is an alias for mask_shape_refine_v2 with
            # coverage-oriented env var defaults (set by the calling shell profile).
            if normalized == "mask_coverage_refine_v3":
                return "mask_shape_refine_v2"
            if normalized in {
                "mask_coverage_refine_v4",
                "public_time_shape_v4_recall",
                "shape_v4_recall",
                "public_time_boundary_gated_v5",
                "boundary_gated_gaussian_v5",
                "r4d_boundary_gated_v5",
                "r4d_multi_instance_boundary_v6",
            }:
                return "mask_bootstrap_refine"
            return normalized
    if _env_flag("QUERY_LIFT_HYBRID_RECALL", False):
        return "hybrid_recall"
    if _env_flag("QUERY_LIFT_V1_COMPAT", False):
        return "v1_compat"
    return "v2"


def _load_bank(entitybank_dir: Path) -> dict[str, np.ndarray]:
    payload = np.load(entitybank_dir / "trajectory_samples.npz")
    return {key: payload[key] for key in payload.files}


def _load_mask(mask_path: str | Path, width: int, height: int) -> np.ndarray:
    from PIL import Image

    with Image.open(mask_path) as image:
        binary = image.convert("L")
        if binary.size != (int(width), int(height)):
            binary = binary.resize((int(width), int(height)), resample=Image.Resampling.NEAREST)
        return np.asarray(binary, dtype=np.uint8) > 0


def _outer_mask(mask: np.ndarray) -> np.ndarray:
    regions = build_mask_regions(mask, core_kernel=5, outer_kernel=17)
    return np.asarray(regions["outer"], dtype=bool)


def _build_distance_regions(mask: np.ndarray, thin_mode: bool = False) -> dict[str, np.ndarray]:
    """Build conservative signed-distance bands for training-free membership refinement.

    Boundary bands are deliberately neutral: Stage1/SAM boundaries are noisy, so only
    deep interior strongly adds Gaussians and far exterior strongly removes them.
    Thresholds scale with mask size so small/thin objects still have useful interior
    evidence instead of an empty force-add region.
    """
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {binary.shape}")
    inside_dist = ndimage.distance_transform_edt(binary).astype(np.float32)
    outside_dist = ndimage.distance_transform_edt(~binary).astype(np.float32)
    area_scale = float(np.sqrt(max(float(binary.sum()), 1.0)))
    if thin_mode:
        inner_soft_px = max(2.0, min(5.0, area_scale * 0.008))
        inner_core_px = max(4.0, min(10.0, area_scale * 0.016))
        inner_force_px = max(7.0, min(16.0, area_scale * 0.026))
        outer_near_px = max(4.0, min(8.0, area_scale * 0.012))
        outer_far_px = max(18.0, min(34.0, area_scale * 0.055))
    else:
        inner_soft_px = max(1.5, min(4.0, area_scale * 0.007))
        inner_core_px = max(3.0, min(9.0, area_scale * 0.014))
        inner_force_px = max(6.0, min(15.0, area_scale * 0.024))
        outer_near_px = max(5.0, min(10.0, area_scale * 0.016))
        outer_far_px = max(20.0, min(42.0, area_scale * 0.070))
    return {
        "inside": binary,
        "inside_soft": binary & (inside_dist >= inner_soft_px),
        "inside_core": binary & (inside_dist >= inner_core_px),
        "inside_force": binary & (inside_dist >= inner_force_px),
        "boundary_neutral": (inside_dist < inner_soft_px) & (outside_dist <= outer_near_px),
        "outer_near": (~binary) & (outside_dist > outer_near_px) & (outside_dist < outer_far_px),
        "outer_far": (~binary) & (outside_dist >= outer_far_px),
        "inside_distance": inside_dist,
        "outside_distance": outside_dist,
        "thresholds": np.asarray([inner_soft_px, inner_core_px, inner_force_px, outer_near_px, outer_far_px], dtype=np.float32),
    }


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        value = int(float(raw))
    except Exception:
        return int(default)
    return max(int(minimum), value)


def _env_float(name: str, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    raw = os.environ.get(name)
    if raw is None:
        value = float(default)
    else:
        try:
            value = float(raw)
        except Exception:
            value = float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    if maximum is not None:
        value = min(float(maximum), value)
    return float(value)


def _env_optional_int(name: str, default: int | None = None, minimum: int = 1) -> int | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(float(raw))
    except Exception:
        return default
    if value <= 0:
        return None
    return max(int(minimum), value)


def _safe_quantile(values: np.ndarray, q: float, default: float) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default)
    return float(np.quantile(values, float(q)))


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size <= 1:
        return np.ones_like(values, dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, values.size, dtype=np.float32)
    return ranks.astype(np.float32)


def _feature_matrix(bank: dict[str, np.ndarray], support: LiftingSupport) -> np.ndarray:
    trajectories = np.asarray(bank["trajectories"], dtype=np.float32)
    sampled = np.asarray(support.sampled_indices, dtype=np.int32)
    if sampled.size:
        traj_mean = trajectories[:, sampled, :].mean(axis=1)
        traj_std = trajectories[:, sampled, :].std(axis=1)
    else:
        traj_mean = trajectories.mean(axis=1)
        traj_std = trajectories.std(axis=1)
    parts = [
        traj_mean,
        traj_std,
        np.asarray(bank.get("spatial_scale", np.ones_like(traj_mean)), dtype=np.float32),
        support.support_score[:, None],
        support.purity[:, None],
        support.core_ratio[:, None],
        support.outer_ratio[:, None],
        np.asarray(bank.get("motion_score", np.zeros((traj_mean.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1, 1),
        np.asarray(bank.get("visibility_proxy", np.ones((traj_mean.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1, 1),
    ]
    features = np.concatenate(parts, axis=1).astype(np.float32)
    return ((features - features.mean(axis=0, keepdims=True)) / np.clip(features.std(axis=0, keepdims=True), 1.0e-6, None)).astype(np.float32)


def _mask_geometry(mask: np.ndarray) -> tuple[float, float, float]:
    """Return aspect ratio, bounding-box fill, and image-area fraction for a mask."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        return 1.0, 1.0, 0.0
    rows, cols = np.nonzero(binary)
    height, width = binary.shape
    box_height = max(int(rows.max()) - int(rows.min()) + 1, 1)
    box_width = max(int(cols.max()) - int(cols.min()) + 1, 1)
    aspect = max(float(box_width) / float(box_height), float(box_height) / float(box_width))
    fill = float(binary.sum()) / float(box_height * box_width)
    area_ratio = float(binary.sum()) / max(float(height * width), 1.0)
    return aspect, fill, area_ratio


def _is_geometrically_thin_mask(mask: np.ndarray) -> bool:
    """Identify thin or hollow support from mask geometry, never from object names."""
    if not _env_flag("QUERY_LIFT_GEOMETRY_THIN_RELAXED_GATE", True):
        return False
    aspect, fill, area_ratio = _mask_geometry(mask)
    aspect_min = _env_float("QUERY_LIFT_GEOMETRY_THIN_ASPECT_MIN", 3.0, minimum=1.0)
    fill_max = _env_float("QUERY_LIFT_GEOMETRY_THIN_FILL_MAX", 0.42, minimum=0.0, maximum=1.0)
    area_max = _env_float("QUERY_LIFT_GEOMETRY_THIN_AREA_MAX", 0.045, minimum=0.0, maximum=1.0)
    return bool(area_ratio <= area_max and (aspect >= aspect_min or fill <= fill_max))


def _support_is_geometrically_thin(support: LiftingSupport) -> bool:
    if not support.samples:
        return False
    required_fraction = _env_float(
        "QUERY_LIFT_GEOMETRY_THIN_VOTE_FRACTION",
        0.45,
        minimum=0.0,
        maximum=1.0,
    )
    votes = sum(_is_geometrically_thin_mask(sample.mask) for sample in support.samples)
    return float(votes) / float(len(support.samples)) >= required_fraction


def _collect_support(
    variant: dict[str, Any],
    dataset_dir: Path,
    state: Any,
    bank: dict[str, np.ndarray],
    max_gaussians_per_frame: int,
    gate_threshold: float,
    device: str,
) -> LiftingSupport:
    trajectories = np.asarray(bank["trajectories"], dtype=np.float32)
    gate = np.asarray(bank["gate"], dtype=np.float32).reshape(trajectories.shape[0], trajectories.shape[1])
    time_values = np.asarray(bank["time_values"], dtype=np.float32).reshape(-1)
    sampled_frames = sorted(list(variant.get("frames", [])), key=lambda row: int(row["frame_index"]))
    sampled_times = np.asarray([float(frame["time_value"]) for frame in sampled_frames], dtype=np.float32)
    sampled_indices = _resample_indices(time_values, sampled_times)
    num_gaussians = trajectories.shape[0]

    positive = np.zeros((num_gaussians,), dtype=np.float32)
    negative = np.zeros((num_gaussians,), dtype=np.float32)
    visible = np.zeros((num_gaussians,), dtype=np.float32)
    distance_positive = np.zeros((num_gaussians,), dtype=np.float32)
    distance_negative = np.zeros((num_gaussians,), dtype=np.float32)
    distance_force_add = np.zeros((num_gaussians,), dtype=np.float32)
    distance_force_drop = np.zeros((num_gaussians,), dtype=np.float32)
    hit_count = np.zeros((num_gaussians,), dtype=np.float32)
    core_hit_count = np.zeros((num_gaussians,), dtype=np.float32)
    full_accum = np.zeros((num_gaussians,), dtype=np.float32)
    core_accum = np.zeros((num_gaussians,), dtype=np.float32)
    outer_accum = np.zeros((num_gaussians,), dtype=np.float32)
    samples: list[LiftingSample] = []
    distance_regions: list[dict[str, np.ndarray]] = []
    mask_areas: list[float] = []

    for frame, bank_index in zip(sampled_frames, sampled_indices.tolist()):
        image_id = str(frame["image_id"])
        camera = Camera.from_json(dataset_dir / "camera" / f"{image_id}.json")
        width = int(max(round(float(np.asarray(camera.image_size)[0])), 1))
        height = int(max(round(float(np.asarray(camera.image_size)[1])), 1))
        mask = _load_mask(frame["mask_path"], width=width, height=height)
        regions = build_mask_regions(mask, core_kernel=5, outer_kernel=17)
        distance_region = _build_distance_regions(mask, thin_mode=_is_geometrically_thin_mask(mask))
        prepared = prepare_semantic_frame_inputs(
            camera=camera,
            frame_index=int(frame["frame_index"]),
            image_id=image_id,
            time_value=float(frame["time_value"]),
            points=trajectories[:, int(bank_index), :],
            spatial_scale=np.asarray(bank["spatial_scale"], dtype=np.float32),
            opacity=np.asarray(state.opacity, dtype=np.float32),
            visibility_gate=gate[:, int(bank_index)],
            image_scale=1.0,
            max_gaussians=int(max_gaussians_per_frame),
            gate_threshold=float(gate_threshold),
            device=device,
        )
        if int(prepared.gaussian_ids.numel()) <= 0:
            continue
        masses = gaussian_region_alpha_masses(prepared, regions)
        ids = prepared.gaussian_ids.detach().cpu().numpy().astype(np.int64)
        visible_mass = masses["visible"].detach().cpu().numpy().astype(np.float32)
        full_ratio = (masses["full"] / masses["visible"].clamp_min(1.0e-6)).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
        core_ratio = (masses["core"] / masses["visible"].clamp_min(1.0e-6)).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
        boundary_ratio = (masses["boundary"] / masses["visible"].clamp_min(1.0e-6)).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
        outer_ratio = (masses["outer"] / masses["visible"].clamp_min(1.0e-6)).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
        positive_prior = np.clip(0.52 * np.sqrt(full_ratio) + 0.38 * np.sqrt(core_ratio) + 0.10 * boundary_ratio - 0.28 * outer_ratio, 0.0, 1.0)
        negative_prior = np.clip(0.86 * outer_ratio + 0.24 * boundary_ratio - 0.12 * full_ratio, 0.0, 1.0)
        distance_masses = gaussian_region_alpha_masses(prepared, {
            "full": distance_region["inside_soft"],
            "core": distance_region["inside_core"],
            "boundary": distance_region["inside_force"],
            "outer": distance_region["outer_far"],
        })
        dist_visible = distance_masses["visible"].clamp_min(1.0e-6)
        dist_inside_soft = (distance_masses["full"] / dist_visible).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
        dist_inside_core = (distance_masses["core"] / dist_visible).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
        dist_inside_force = (distance_masses["boundary"] / dist_visible).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
        dist_outer_far = (distance_masses["outer"] / dist_visible).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
        dist_outer_near = np.clip(1.0 - dist_inside_soft - dist_outer_far, 0.0, 1.0).astype(np.float32)
        distance_positive[ids] += 0.48 * dist_inside_soft + 0.88 * dist_inside_core + 1.35 * dist_inside_force
        distance_negative[ids] += 0.22 * dist_outer_near + 0.90 * dist_outer_far
        distance_force_add[ids] += (dist_inside_core >= 0.28).astype(np.float32) + (dist_inside_force >= 0.22).astype(np.float32)
        distance_force_drop[ids] += (dist_outer_far >= 0.68).astype(np.float32)
        positive[ids] += positive_prior
        negative[ids] += negative_prior
        visible[ids] += (visible_mass > 1.0e-6).astype(np.float32)
        hit_count[ids] += (full_ratio >= 0.08).astype(np.float32)
        core_hit_count[ids] += (core_ratio >= 0.04).astype(np.float32)
        full_accum[ids] += full_ratio
        core_accum[ids] += core_ratio
        outer_accum[ids] += outer_ratio
        samples.append(LiftingSample(frame=frame, bank_index=int(bank_index), mask=mask, outer_mask=_outer_mask(mask), prepared=prepared))
        distance_regions.append(distance_region)
        mask_areas.append(float(mask.sum()))

    denom = np.clip(visible, 1.0, None)
    positive_mean = positive / denom
    negative_mean = negative / denom
    presence = hit_count / max(float(len(samples)), 1.0)
    core_presence = core_hit_count / max(float(len(samples)), 1.0)
    full_mean = full_accum / denom
    distance_positive_mean = distance_positive / denom
    distance_negative_mean = distance_negative / denom
    distance_force_add_mean = distance_force_add / max(float(len(samples)), 1.0)
    distance_force_drop_mean = distance_force_drop / max(float(len(samples)), 1.0)
    core_mean = core_accum / denom
    outer_mean = outer_accum / denom
    purity = np.clip(positive_mean / np.clip(positive_mean + negative_mean + 0.20 * outer_mean, 1.0e-6, None), 0.0, 1.0)
    temporal_support = np.sqrt(np.clip(presence, 0.0, 1.0))
    support_score = np.clip(
        0.48 * positive_mean + 0.24 * temporal_support + 0.18 * core_presence + 0.16 * purity - 0.34 * negative_mean,
        0.0,
        1.0,
    ).astype(np.float32)

    return LiftingSupport(
        alias=str(variant["alias"]),
        base_phrase=str(variant["base_phrase"]),
        source_object_id=(
            None
            if variant.get("source_object_id") is None
            else int(variant.get("source_object_id"))
        ),
        source_track_id=str(variant.get("source_track_id") or ""),
        source_instance_group_id=(
            None
            if variant.get("source_instance_group_id") is None
            else str(variant.get("source_instance_group_id"))
        ),
        source_instance_index=(
            None
            if variant.get("source_instance_index") is None
            else int(variant.get("source_instance_index"))
        ),
        phase=str(variant["phase"]),
        variant_kind=str(variant["variant_kind"]),
        description=str(variant["description"]),
        entity_type=_phrase_entity_type(str(variant["base_phrase"])),
        sampled_frames=sampled_frames,
        sampled_indices=np.asarray(sampled_indices, dtype=np.int32),
        samples=samples,
        distance_regions=distance_regions,
        presence=presence.astype(np.float32),
        core_presence=core_presence.astype(np.float32),
        full_mean=full_mean.astype(np.float32),
        support_score=support_score,
        positive_score=positive_mean.astype(np.float32),
        negative_score=negative_mean.astype(np.float32),
        visible_count=visible.astype(np.float32),
        distance_positive_score=distance_positive_mean.astype(np.float32),
        distance_negative_score=distance_negative_mean.astype(np.float32),
        distance_force_add=distance_force_add_mean.astype(np.float32),
        distance_force_drop=distance_force_drop_mean.astype(np.float32),
        hit_count=hit_count.astype(np.float32),
        core_hit_count=core_hit_count.astype(np.float32),
        purity=purity.astype(np.float32),
        core_ratio=core_mean.astype(np.float32),
        outer_ratio=outer_mean.astype(np.float32),
        mean_mask_area=float(np.mean(mask_areas)) if mask_areas else 0.0,
    )


def _component_labels(points: np.ndarray, ids: np.ndarray, graph_knn: int, radius_scale: float) -> list[np.ndarray]:
    ids = np.unique(np.asarray(ids, dtype=np.int64).reshape(-1))
    if ids.size <= 1:
        return [ids] if ids.size else []
    local_points = np.asarray(points[ids], dtype=np.float32)
    tree = cKDTree(local_points)
    k = int(min(max(2, graph_knn + 1), local_points.shape[0]))
    distances = tree.query(local_points, k=k)[0]
    neighbor = distances[:, -1] if distances.ndim > 1 else distances.reshape(-1)
    radius = max(_safe_quantile(neighbor, 0.75, 0.02) * float(radius_scale), 1.0e-4)
    # Query every radius neighborhood in one SciPy call.  The former per-point
    # Python calls were equivalent, but became a substantial CPU bottleneck for
    # large entities during the ten-round coverage refinement.
    neighborhoods = _radius_neighbor_lists(tree, local_points, radius)
    visited = np.zeros((ids.size,), dtype=bool)
    components: list[np.ndarray] = []
    for start in range(ids.size):
        if visited[start]:
            continue
        queue = [start]
        visited[start] = True
        component = []
        head = 0
        while head < len(queue):
            node = int(queue[head])
            head += 1
            component.append(node)
            for neighbor_id in neighborhoods[node]:
                neighbor_id = int(neighbor_id)
                if visited[neighbor_id]:
                    continue
                visited[neighbor_id] = True
                queue.append(neighbor_id)
        components.append(ids[np.asarray(component, dtype=np.int64)])
    components.sort(key=lambda item: int(item.size), reverse=True)
    return components


def _radius_neighbor_lists(
    tree: cKDTree,
    points: np.ndarray,
    radius: float,
) -> list[np.ndarray]:
    """Return exact radius neighborhoods using a batched, deterministic query.

    Batched queries keep the same candidates as calling ``query_ball_point`` for
    each point, while avoiding thousands of Python-to-SciPy transitions.  The
    compatibility fallback supports older SciPy releases that lack ``workers``
    or ``return_sorted``.
    """
    locations = np.asarray(points, dtype=np.float32)
    try:
        raw = tree.query_ball_point(
            locations,
            r=float(radius),
            workers=-1,
            return_sorted=True,
        )
    except TypeError:
        raw = tree.query_ball_point(locations, r=float(radius))
    return [np.asarray(neighbors, dtype=np.int64) for neighbors in list(raw)]


def _top_component_union(points: np.ndarray, ids: np.ndarray, scores: np.ndarray, graph_knn: int, radius_scale: float, top_k: int, max_gaussians: int) -> np.ndarray:
    components = _component_labels(points, ids, graph_knn=graph_knn, radius_scale=radius_scale)
    if not components:
        return np.empty((0,), dtype=np.int64)
    components.sort(key=lambda comp: float(np.mean(scores[comp])) * np.sqrt(float(comp.size)), reverse=True)
    selected = np.unique(np.concatenate(components[: int(max(top_k, 1))]).astype(np.int64))
    if selected.size > int(max_gaussians):
        selected = selected[np.argsort(-scores[selected], kind="mergesort")[: int(max_gaussians)]]
    return np.unique(selected.astype(np.int64))

def _high_precision_core(support: LiftingSupport, min_gaussians: int, max_gaussians: int) -> np.ndarray:
    candidate = np.where((support.support_score > 0.0) & (support.purity >= 0.52) & (support.negative_score <= 0.42))[0]
    if candidate.size == 0:
        candidate = np.where(support.support_score > 0.0)[0]
    if candidate.size == 0:
        return np.empty((0,), dtype=np.int64)
    score = support.support_score + 0.18 * support.core_ratio + 0.10 * support.purity - 0.20 * support.outer_ratio
    floor = max(_safe_quantile(score[candidate], 0.78, 0.05), 0.04)
    core = candidate[score[candidate] >= floor]
    min_core = min(max(24, int(min_gaussians * 0.35)), int(max_gaussians))
    if core.size < min_core:
        ranked = candidate[np.argsort(-score[candidate], kind="mergesort")]
        core = ranked[: min(min_core, ranked.size)]
    if core.size > int(max_gaussians):
        core = core[np.argsort(-score[core], kind="mergesort")[: int(max_gaussians)]]
    return np.unique(core.astype(np.int64))


def _graph_expand(
    support: LiftingSupport,
    bank: dict[str, np.ndarray],
    core_ids: np.ndarray,
    max_gaussians: int,
    radius_scale: float,
    graph_knn: int,
    expansion_factor: float,
) -> np.ndarray:
    core_ids = np.asarray(core_ids, dtype=np.int64).reshape(-1)
    if core_ids.size == 0:
        return core_ids
    sampled = np.asarray(support.sampled_indices, dtype=np.int32)
    trajectories = np.asarray(bank["trajectories"], dtype=np.float32)
    points = trajectories[:, sampled, :].mean(axis=1) if sampled.size else trajectories.mean(axis=1)
    features = _feature_matrix(bank, support)
    tree = cKDTree(points)
    core_points = points[core_ids]
    k = int(min(max(2, graph_knn + 1), max(core_points.shape[0], 2)))
    distances = cKDTree(core_points).query(core_points, k=k)[0] if core_points.shape[0] > 1 else np.asarray([0.02], dtype=np.float32)
    neighbor = distances[:, -1] if distances.ndim > 1 else distances.reshape(-1)
    radius = max(_safe_quantile(neighbor, 0.85, 0.025) * float(radius_scale), 1.0e-4)
    feature_center = features[core_ids].mean(axis=0)
    feature_center /= max(float(np.linalg.norm(feature_center)), 1.0e-6)
    normalized = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1.0e-6, None)
    similarity = ((normalized @ feature_center) + 1.0) * 0.5
    score = (
        0.36 * support.support_score
        + 0.22 * support.purity
        + 0.18 * support.core_ratio
        + 0.16 * similarity
        + 0.08 * _rank_normalize(support.visible_count)
        - 0.32 * support.negative_score
        - 0.24 * support.outer_ratio
    ).astype(np.float32)
    score_floor = max(_safe_quantile(score[core_ids], 0.10, 0.05) * 0.78, 0.035)
    purity_floor = max(_safe_quantile(support.purity[core_ids], 0.10, 0.45) * 0.82, 0.38)
    negative_ceiling = min(_safe_quantile(support.negative_score[core_ids], 0.90, 0.45) * 1.35, 0.58)
    budget = int(min(max_gaussians, max(core_ids.size, round(core_ids.size * float(expansion_factor)))))

    visited = np.zeros((points.shape[0],), dtype=bool)
    visited[core_ids] = True
    queue = list(core_ids.tolist())
    head = 0
    while head < len(queue) and int(np.sum(visited)) < budget:
        node = int(queue[head])
        head += 1
        neighbors = tree.query_ball_point(points[node], r=radius)
        if len(neighbors) > int(graph_knn) + 1:
            neighbors = sorted(neighbors, key=lambda idx: float(score[int(idx)]), reverse=True)[: int(graph_knn) + 1]
        for neighbor_id in neighbors:
            neighbor_id = int(neighbor_id)
            if visited[neighbor_id]:
                continue
            if score[neighbor_id] < score_floor:
                continue
            if support.purity[neighbor_id] < purity_floor:
                continue
            if support.negative_score[neighbor_id] > negative_ceiling and support.core_ratio[neighbor_id] < 0.04:
                continue
            visited[neighbor_id] = True
            queue.append(neighbor_id)
            if int(np.sum(visited)) >= budget:
                break
    selected = np.where(visited)[0].astype(np.int64)
    if selected.size > budget:
        selected = selected[np.argsort(-score[selected], kind="mergesort")[:budget]]
    return np.unique(selected.astype(np.int64))


def _selection_metrics(
    selection_ids: np.ndarray,
    support: LiftingSupport,
    num_gaussians: int,
    device: str,
    scoring_mode: str | None = None,
    relative_threshold: float = 0.18,
    absolute_threshold: float = 0.015,
    max_frames: int | None = None,
    sigma_scale: float = 1.0,
    max_splat_radius: int = 18,
) -> dict[str, Any]:
    mask = np.zeros((num_gaussians,), dtype=np.float32)
    mask[np.asarray(selection_ids, dtype=np.int64)] = 1.0
    return _selection_mask_metrics(
        mask,
        support=support,
        device=device,
        scoring_mode=scoring_mode,
        relative_threshold=relative_threshold,
        absolute_threshold=absolute_threshold,
        max_frames=max_frames,
        sigma_scale=sigma_scale,
        max_splat_radius=max_splat_radius,
    )


def _empty_metric_row() -> dict[str, Any]:
    return {
        "score": float("-inf"),
        "rendered_iou_stage1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "area_ratio": 0.0,
        "outer_leakage": 1.0,
        "active_frame_coverage": 0.0,
        "area_cv": 1.0,
        "mean_pred_area": 0.0,
    }


def _support_proxy_metrics(selection_ids: np.ndarray, support: LiftingSupport, scoring_mode: str | None = None) -> dict[str, Any]:
    """Fast internal metric for bootstrap membership updates.

    Full rendered metrics are still used for final candidate validation. During
    bootstrap, however, repeated full-resolution splat rendering is too slow for
    large public objects. This proxy uses the signed-distance evidence already
    accumulated from multi-frame masks to steer recall growth and leakage drops.
    """
    ids = np.unique(np.asarray(selection_ids, dtype=np.int64).reshape(-1))
    ids = ids[(ids >= 0) & (ids < int(support.support_score.shape[0]))]
    if ids.size == 0:
        return _empty_metric_row()

    positive_field = (
        0.38 * support.distance_positive_score
        + 0.24 * support.full_mean
        + 0.16 * support.positive_score
        + 0.14 * support.presence
        + 0.08 * support.support_score
    ).astype(np.float32)
    negative_field = (
        0.38 * support.distance_negative_score
        + 0.28 * support.outer_ratio
        + 0.20 * support.negative_score
        + 0.14 * support.distance_force_drop
    ).astype(np.float32)
    eligible = (
        (positive_field > 1.0e-5)
        & (support.distance_force_drop < 0.92)
        & (support.outer_ratio < 0.98)
    )
    target_mass = float(np.sum(positive_field[eligible]))
    selected_pos = float(np.sum(positive_field[ids]))
    selected_neg = float(np.sum(negative_field[ids]))
    if target_mass <= 1.0e-6:
        target_mass = float(np.sum(positive_field)) + 1.0e-6
    recall = float(np.clip(selected_pos / max(target_mass, 1.0e-6), 0.0, 1.0))
    precision = float(np.clip(selected_pos / max(selected_pos + selected_neg, 1.0e-6), 0.0, 1.0))
    area_ratio = float(np.clip(selected_pos / max(target_mass, 1.0e-6), 0.0, 2.0))
    outer_leakage = float(np.clip(selected_neg / max(selected_pos + selected_neg, 1.0e-6), 0.0, 1.0))
    denom = max(float(len(support.samples)), 1.0)
    point_visibility = np.minimum(support.visible_count[ids] / denom, 1.0)
    active_frame_coverage = float(np.clip(np.percentile(point_visibility, 90), 0.0, 1.0))
    # Entity-level activity is not the mean visibility of each selected Gaussian:
    # a cup shell may be represented by different Gaussians across views while
    # still producing a valid entity mask in every support frame.
    active_frame_coverage = max(active_frame_coverage, float(np.clip(0.35 + 0.65 * recall, 0.0, 1.0)))
    rendered_iou = float(
        (precision * recall) / max(precision + recall - precision * recall, 1.0e-6)
    )
    area_cv = 0.35
    mode = str(scoring_mode or _lifting_mode())
    if mode in {"hybrid_recall", "v1_compat"}:
        underfill = max(0.0, 0.35 - area_ratio)
        overfill = max(0.0, area_ratio - 1.50)
        score = float(
            rendered_iou
            + 0.20 * recall
            + 0.10 * precision
            + 0.20 * active_frame_coverage
            - 0.35 * underfill
            - 0.35 * overfill
            - 0.60 * outer_leakage
            - 0.15 * area_cv
        )
    else:
        underfill = max(0.0, 0.70 - area_ratio)
        overfill = max(0.0, area_ratio - 1.35)
        score = float(
            rendered_iou
            + 0.95 * recall
            + 0.12 * precision
            + 0.16 * active_frame_coverage
            - 0.28 * overfill
            - 0.30 * underfill
            - 0.52 * outer_leakage
            - 0.06 * area_cv
        )
    return {
        "score": score,
        "rendered_iou_stage1": rendered_iou,
        "precision": precision,
        "recall": recall,
        "area_ratio": area_ratio,
        "outer_leakage": outer_leakage,
        "active_frame_coverage": active_frame_coverage,
        "area_cv": area_cv,
        "mean_pred_area": float(area_ratio * max(float(support.mean_mask_area), 1.0)),
        "proxy_metric": True,
    }


def _bootstrap_proxy_evidence_selection(
    seed_ids: np.ndarray,
    support: LiftingSupport,
    num_gaussians: int,
    target_area_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    seed_ids = np.unique(np.asarray(seed_ids, dtype=np.int64).reshape(-1))
    seed_ids = seed_ids[(seed_ids >= 0) & (seed_ids < int(num_gaussians))]
    positive_field = (
        0.38 * support.distance_positive_score
        + 0.24 * support.full_mean
        + 0.16 * support.positive_score
        + 0.14 * support.presence
        + 0.08 * support.support_score
    ).astype(np.float32)
    negative_field = (
        0.38 * support.distance_negative_score
        + 0.28 * support.outer_ratio
        + 0.20 * support.negative_score
        + 0.14 * support.distance_force_drop
    ).astype(np.float32)
    ranking = (
        positive_field
        - 0.34 * negative_field
        + 0.12 * support.core_presence
        + 0.08 * support.purity
    ).astype(np.float32)
    eligible = np.where(
        ((support.support_score > 0.0) | (support.distance_positive_score > 0.02) | (support.full_mean > 0.01))
        & (support.distance_force_drop < 0.92)
        & (support.outer_ratio < 0.98)
        & (negative_field < max(float(np.percentile(negative_field, 96)), 0.60))
    )[0].astype(np.int64)
    if eligible.size == 0:
        eligible = seed_ids.copy()
    order = eligible[np.argsort(-ranking[eligible], kind="mergesort")]
    max_keep_default = min(int(num_gaussians), _env_int("QUERY_LIFT_MAX_GAUSSIANS", 6144, minimum=128))
    max_keep = int(min(int(num_gaussians), _env_int("QUERY_LIFT_BOOTSTRAP_PROXY_MAX_GAUSSIANS", max_keep_default, minimum=128)))
    min_keep = int(min(max_keep, max(seed_ids.size, _env_int("QUERY_LIFT_BOOTSTRAP_PROXY_MIN_GAUSSIANS", 256, minimum=16))))
    target_mass = float(target_area_ratio) * float(np.sum(positive_field[eligible]))
    if target_mass <= 1.0e-6:
        target_mass = float(np.sum(positive_field[order])) * 0.70
    cumulative = np.cumsum(np.maximum(positive_field[order], 0.0), dtype=np.float64)
    keep_count = int(np.searchsorted(cumulative, target_mass, side="left") + 1) if cumulative.size else 0
    keep_count = int(min(max_keep, max(min_keep, keep_count)))
    selected = order[:keep_count].astype(np.int64)
    if seed_ids.size:
        selected = np.unique(np.concatenate([selected, seed_ids]).astype(np.int64))
    if selected.size > max_keep:
        selected = selected[np.argsort(-ranking[selected], kind="mergesort")[:max_keep]]
    selected = np.unique(selected.astype(np.int64))
    metrics = _support_proxy_metrics(selected, support=support, scoring_mode="hybrid_recall")
    history = [
        {
            "iter": 1,
            "render_eval": False,
            "gaussian_count": int(selected.size),
            "iou": float(metrics.get("rendered_iou_stage1", 0.0)),
            "rendered_iou_stage1": float(metrics.get("rendered_iou_stage1", 0.0)),
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "area_ratio": float(metrics.get("area_ratio", 0.0)),
            "outer_leakage": float(metrics.get("outer_leakage", 0.0)),
            "score": float(metrics.get("score", 0.0)),
            "inside_reward": float(np.mean(support.distance_positive_score[selected])) if selected.size else 0.0,
            "outside_leakage": float(np.mean(support.distance_negative_score[selected])) if selected.size else 0.0,
        }
    ]
    info = {
        "score": float(metrics.get("score", 0.0)),
        "rendered_iou_stage1": float(metrics.get("rendered_iou_stage1", 0.0)),
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "area_ratio": float(metrics.get("area_ratio", 0.0)),
        "outer_leakage": float(metrics.get("outer_leakage", 0.0)),
        "active_frame_coverage": float(metrics.get("active_frame_coverage", 0.0)),
        "area_cv": float(metrics.get("area_cv", 1.0)),
        "mean_pred_area": float(metrics.get("mean_pred_area", 0.0)),
        "bootstrap_iters": 1,
        "bootstrap_best_iou": float(metrics.get("rendered_iou_stage1", 0.0)),
        "bootstrap_best_precision": float(metrics.get("precision", 0.0)),
        "bootstrap_best_recall": float(metrics.get("recall", 0.0)),
        "bootstrap_best_area_ratio": float(metrics.get("area_ratio", 0.0)),
        "bootstrap_target_area_ratio": float(target_area_ratio),
        "bootstrap_used_feasible_area_candidate": True,
        "bootstrap_history": history,
        "bootstrap_proxy_evidence_only": True,
    }
    return selected, info


def _selection_mask_metrics(
    mask: np.ndarray,
    support: LiftingSupport,
    device: str,
    scoring_mode: str | None = None,
    relative_threshold: float = 0.18,
    absolute_threshold: float = 0.015,
    max_frames: int | None = None,
    sigma_scale: float = 1.0,
    max_splat_radius: int = 18,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=np.float32).reshape(-1)
    try:
        resolved_sigma_scale = float(sigma_scale)
    except (TypeError, ValueError):
        resolved_sigma_scale = 1.0
    if not np.isfinite(resolved_sigma_scale):
        resolved_sigma_scale = 1.0
    resolved_sigma_scale = float(np.clip(resolved_sigma_scale, 0.25, 8.0))
    try:
        resolved_max_splat_radius = int(float(max_splat_radius))
    except (TypeError, ValueError):
        resolved_max_splat_radius = 18
    resolved_max_splat_radius = int(np.clip(resolved_max_splat_radius, 1, 64))
    rows: list[dict[str, float]] = []
    pred_areas: list[float] = []
    active_hits = 0
    metric_samples = support.samples
    max_metric_frames = (
        int(max_frames)
        if max_frames is not None
        else _env_optional_int("QUERY_LIFT_METRIC_MAX_FRAMES", None, minimum=1)
    )
    if max_metric_frames is not None and len(metric_samples) > int(max_metric_frames):
        sample_indices = _resample_indices(len(metric_samples), int(max_metric_frames))
        metric_samples = [metric_samples[int(index)] for index in sample_indices.tolist()]
    for sample in metric_samples:
        prepared = PreparedSemanticFrame(
            frame_index=sample.prepared.frame_index,
            time_value=sample.prepared.time_value,
            image_id=sample.prepared.image_id,
            width=sample.prepared.width,
            height=sample.prepared.height,
            image_scale=sample.prepared.image_scale,
            gaussian_ids=sample.prepared.gaussian_ids.to(device=device),
            centers_xy=sample.prepared.centers_xy.to(device=device),
            sigma_px=sample.prepared.sigma_px.to(device=device) * resolved_sigma_scale,
            alpha_weight=sample.prepared.alpha_weight.to(device=device),
            depth=sample.prepared.depth.to(device=device),
        )
        local_ids = prepared.gaussian_ids.detach().cpu().numpy().astype(np.int64)
        local_weights = torch.as_tensor(mask[local_ids], dtype=torch.float32, device=device)
        pred, _alpha = render_selection_mask(
            prepared,
            local_weights,
            relative_threshold=float(relative_threshold),
            absolute_threshold=float(absolute_threshold),
            max_splat_radius=resolved_max_splat_radius,
        )
        pred_area = float(np.asarray(pred, dtype=bool).sum())
        pred_areas.append(pred_area)
        if pred_area > 0.0:
            active_hits += 1
        rows.append(alignment_metrics(pred_mask=pred, full_mask=sample.mask, outer_mask=sample.outer_mask))
    if not rows:
        return _empty_metric_row()
    mean = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    precision = float(mean.get("rendered_precision_stage1", 0.0))
    recall = float(mean.get("rendered_recall_stage1", 0.0))
    area_ratio = float(mean["area_ratio"])
    outer_leakage = float(mean["outer_leakage"])
    active_frame_coverage = float(active_hits / max(len(metric_samples), 1))
    active_areas = np.asarray([area for area in pred_areas if area > 0.0], dtype=np.float32)
    area_cv = float(np.std(active_areas) / max(float(np.mean(active_areas)), 1.0)) if active_areas.size else 1.0
    mode = str(scoring_mode or _lifting_mode())
    if mode in {"mask_boundary_refine", "boundary_refine"}:
        overfill = max(0.0, area_ratio - 1.15)
        severe_overfill = max(0.0, area_ratio - 1.75)
        underfill = max(0.0, 0.58 - area_ratio)
        score = float(
            2.10 * mean["rendered_iou_stage1"]
            + 0.38 * precision
            + 0.62 * recall
            + 0.28 * active_frame_coverage
            - 0.92 * overfill
            - 1.55 * severe_overfill
            - 0.32 * underfill
            - 1.10 * outer_leakage
            - 0.18 * min(area_cv, 2.0)
        )
    elif mode in {"mask_shape_refine_v2", "shape_refine_v2"}:
        overfill = max(0.0, area_ratio - 1.25)
        severe_overfill = max(0.0, area_ratio - 1.55)
        underfill = max(0.0, 0.58 - area_ratio)
        thin_penalty = 0.0
        if _support_is_geometrically_thin(support):
            thin_penalty = 0.14 * max(0.0, area_ratio - 1.15) + 0.20 * outer_leakage
        score = float(
            1.80 * mean["rendered_iou_stage1"]
            + 0.42 * precision
            + 0.58 * recall
            + 0.24 * active_frame_coverage
            - 0.78 * overfill
            - 1.30 * severe_overfill
            - 0.28 * underfill
            - 0.90 * outer_leakage
            - 0.14 * min(area_cv, 2.0)
            - thin_penalty
        )
    elif mode in {"hybrid_recall", "v1_compat"}:
        underfill = max(0.0, 0.35 - area_ratio)
        overfill = max(0.0, area_ratio - 1.50)
        score = float(
            mean["rendered_iou_stage1"]
            + 0.20 * recall
            + 0.10 * precision
            + 0.20 * active_frame_coverage
            - 0.35 * underfill
            - 0.35 * overfill
            - 0.60 * outer_leakage
            - 0.15 * min(area_cv, 2.0)
        )
    else:
        overfill = max(0.0, area_ratio - 1.35)
        severe_overfill = max(0.0, area_ratio - 2.25)
        underfill = max(0.0, 0.82 - area_ratio)
        thin_penalty = 0.0
        if _support_is_geometrically_thin(support):
            thin_penalty = 0.18 * max(0.0, area_ratio - 1.20) + 0.24 * outer_leakage
        score = float(
            mean["rendered_iou_stage1"]
            + 0.10 * precision
            + 1.05 * recall
            + 0.10 * active_frame_coverage
            - 0.32 * overfill
            - 0.48 * severe_overfill
            - 0.10 * underfill
            - 0.52 * outer_leakage
            - 0.06 * min(area_cv, 2.0)
            - thin_penalty
        )
    return {
        "score": score,
        "rendered_iou_stage1": float(mean["rendered_iou_stage1"]),
        "precision": precision,
        "recall": recall,
        "area_ratio": area_ratio,
        "outer_leakage": outer_leakage,
        "active_frame_coverage": active_frame_coverage,
        "area_cv": area_cv,
        "mean_pred_area": float(np.mean(pred_areas)) if pred_areas else 0.0,
        "alpha_relative_threshold": float(relative_threshold),
        "alpha_absolute_threshold": float(absolute_threshold),
        "alpha_sigma_scale": float(resolved_sigma_scale),
        "alpha_max_splat_radius": int(resolved_max_splat_radius),
    }


def _candidate_variants_v2(support: LiftingSupport, bank: dict[str, np.ndarray], min_gaussians: int, max_gaussians: int, graph_knn: int, radius_scale: float) -> list[tuple[str, np.ndarray]]:
    sampled = np.asarray(support.sampled_indices, dtype=np.int32)
    trajectories = np.asarray(bank["trajectories"], dtype=np.float32)
    points = trajectories[:, sampled, :].mean(axis=1) if sampled.size else trajectories.mean(axis=1)
    score = support.support_score + 0.16 * support.purity + 0.12 * support.core_ratio + 0.08 * support.presence - 0.28 * support.negative_score - 0.20 * support.outer_ratio
    core = _high_precision_core(support, min_gaussians=min_gaussians, max_gaussians=max_gaussians)
    variants: list[tuple[str, np.ndarray]] = [("core_only", core)]
    if core.size:
        largest = _top_component_union(points, core, score, graph_knn=graph_knn, radius_scale=radius_scale, top_k=1, max_gaussians=max_gaussians)
        variants.append(("largest_core_component", largest))
        for top_k in (2, 3):
            variants.append((f"component_union_top{top_k}", _top_component_union(points, core, score, graph_knn=graph_knn, radius_scale=radius_scale * 1.15, top_k=top_k, max_gaussians=max_gaussians)))
    for name, factor, rscale in (
        ("strict_graph_expand", 1.25, 0.90),
        ("mid_graph_expand", 1.85, 1.10),
        ("relaxed_graph_expand", 2.75, 1.35),
    ):
        expanded = _graph_expand(
            support=support,
            bank=bank,
            core_ids=core,
            max_gaussians=max_gaussians,
            radius_scale=radius_scale * rscale,
            graph_knn=graph_knn,
            expansion_factor=factor,
        )
        variants.append((name, expanded))
        if expanded.size:
            variants.append((f"{name}_largest_component", _top_component_union(points, expanded, score, graph_knn=graph_knn, radius_scale=radius_scale * rscale, top_k=1, max_gaussians=max_gaussians)))
    eligible = np.where((support.support_score > 0.0) & (support.purity >= 0.24) & (support.negative_score <= 0.82))[0]
    if eligible.size:
        ranking = score + 0.24 * support.full_mean + 0.14 * support.presence + 0.08 * support.core_presence
        keep_values = (
            min_gaussians,
            int((min_gaussians + max_gaussians) * 0.5),
            max_gaussians,
        )
        recall_pool = eligible[np.argsort(-ranking[eligible], kind="mergesort")[: int(min(max_gaussians, eligible.size))]]
        if recall_pool.size:
            variants.append(("recall_component_union_top4", _top_component_union(points, recall_pool, ranking, graph_knn=graph_knn, radius_scale=radius_scale * 1.55, top_k=4, max_gaussians=max_gaussians)))
        for keep in keep_values:
            keep = int(min(max(keep, 1), eligible.size, max_gaussians))
            ids = eligible[np.argsort(-ranking[eligible], kind="mergesort")[:keep]]
            variants.append((f"ranked_recall_top_{keep}", ids.astype(np.int64)))
    if _support_is_geometrically_thin(support) and eligible.size:
        thin_ranking = 0.58 * support.full_mean + 0.24 * support.presence + 0.16 * support.core_ratio + 0.12 * support.purity - 0.34 * support.outer_ratio - 0.22 * support.negative_score
        thin_pool = eligible[np.argsort(-thin_ranking[eligible], kind="mergesort")[: int(min(max_gaussians, max(min_gaussians, eligible.size)) )]]
        variants.append(("thin_object_boundary_recall", _top_component_union(points, thin_pool, thin_ranking, graph_knn=graph_knn, radius_scale=radius_scale * 1.10, top_k=4, max_gaussians=max_gaussians)))
    unique: dict[str, np.ndarray] = {}
    seen: set[tuple[int, ...]] = set()
    for name, ids in variants:
        ids = np.unique(np.asarray(ids, dtype=np.int64))
        if not ids.size:
            continue
        key = tuple(int(value) for value in ids[: min(ids.size, 2048)]) + (int(ids.size),)
        if key in seen:
            continue
        seen.add(key)
        unique[name] = ids
    return list(unique.items())



def _safe_graph_expand_from_selection(
    ids: np.ndarray,
    support: LiftingSupport,
    bank: dict[str, np.ndarray] | None,
    max_add: int,
    graph_knn: int = 24,
    radius_scale: float = 1.65,
) -> np.ndarray:
    if bank is None or "trajectories" not in bank:
        return np.asarray(ids, dtype=np.int64)
    ids = np.unique(np.asarray(ids, dtype=np.int64))
    if ids.size == 0 or int(max_add) <= 0:
        return ids
    sampled = np.asarray(support.sampled_indices, dtype=np.int32)
    trajectories = np.asarray(bank["trajectories"], dtype=np.float32)
    points = trajectories[:, sampled, :].mean(axis=1) if sampled.size else trajectories.mean(axis=1)
    tree = cKDTree(points)
    local_points = points[ids]
    if local_points.shape[0] <= 1:
        radius = 0.03 * float(radius_scale)
    else:
        k = int(min(max(2, graph_knn + 1), local_points.shape[0]))
        distances = cKDTree(local_points).query(local_points, k=k)[0]
        neighbor = distances[:, -1] if distances.ndim > 1 else distances.reshape(-1)
        radius = max(_safe_quantile(neighbor, 0.80, 0.025) * float(radius_scale), 1.0e-4)
    neighbor_lists = _radius_neighbor_lists(tree, points[ids], radius)
    neighbor_parts = [neighbors for neighbors in neighbor_lists if neighbors.size]
    if not neighbor_parts:
        return ids
    candidates = np.unique(np.concatenate(neighbor_parts).astype(np.int64))
    candidates = np.setdiff1d(candidates, ids, assume_unique=True)
    if candidates.size == 0:
        return ids
    inside_score = 0.34 * support.distance_positive_score + 0.22 * support.full_mean + 0.18 * support.presence + 0.16 * support.positive_score + 0.10 * support.support_score
    penalty = 0.28 * support.distance_negative_score + 0.18 * support.outer_ratio + 0.10 * support.negative_score
    score = inside_score - penalty
    keep = candidates[
        ((support.support_score[candidates] > 0.0) | (support.distance_positive_score[candidates] > 0.08))
        & (support.outer_ratio[candidates] < 0.82)
        & (support.negative_score[candidates] < 0.88)
        & (support.distance_force_drop[candidates] < 0.75)
    ]
    if keep.size == 0:
        return ids
    keep = keep[np.argsort(-score[keep], kind="mergesort")[: int(max_add)]]
    return np.unique(np.concatenate([ids, keep]).astype(np.int64))


def _bootstrap_refine_selection(
    seed_ids: np.ndarray,
    support: LiftingSupport,
    num_gaussians: int,
    device: str,
    seed_index: int = 0,
    bank: dict[str, np.ndarray] | None = None,
    max_gaussians: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    max_iter = _env_int("QUERY_LIFT_BOOTSTRAP_MAX_ITER", 5, minimum=1)
    wall_timeout = _env_float("QUERY_LIFT_BOOTSTRAP_TIMEOUT", 60.0, minimum=10.0)
    t_start = time.time()
    batch_size = _env_int("QUERY_LIFT_BOOTSTRAP_BATCH_FRAMES", 6, minimum=1)
    eval_interval = _env_int("QUERY_LIFT_BOOTSTRAP_RENDER_EVAL_INTERVAL", 2, minimum=1)
    target_iou = _env_float("QUERY_LIFT_BOOTSTRAP_TARGET_IOU", 0.90, minimum=0.0, maximum=1.0)
    force_add = _env_float("QUERY_LIFT_BOOTSTRAP_FORCE_ADD", 0.56, minimum=0.0)
    force_drop = _env_float("QUERY_LIFT_BOOTSTRAP_FORCE_DROP", 0.50, minimum=0.0)
    keep_threshold = _env_float("QUERY_LIFT_BOOTSTRAP_KEEP_THRESHOLD", 0.10)
    thin_mode = _support_is_geometrically_thin(support)
    target_area_ratio = _env_float(
        "QUERY_LIFT_BOOTSTRAP_THIN_TARGET_AREA_RATIO" if thin_mode else "QUERY_LIFT_BOOTSTRAP_TARGET_AREA_RATIO",
        0.30 if thin_mode else 0.35,
        minimum=0.01,
        maximum=2.0,
    )
    feasible_precision = _env_float("QUERY_LIFT_BOOTSTRAP_MIN_PRECISION", 0.22 if thin_mode else 0.18, minimum=0.0, maximum=1.0)
    feasible_leakage = _env_float("QUERY_LIFT_BOOTSTRAP_MAX_OUTER_LEAKAGE", 0.35 if thin_mode else 0.32, minimum=0.0, maximum=1.0)

    membership = np.zeros((num_gaussians,), dtype=np.float32)
    seed_ids = np.unique(np.asarray(seed_ids, dtype=np.int64))
    seed_ids = seed_ids[(seed_ids >= 0) & (seed_ids < num_gaussians)]
    force_seed = np.where((support.distance_force_add >= 0.08) & (support.distance_force_drop < 0.55))[0].astype(np.int64)
    if force_seed.size:
        seed_ids = np.unique(np.concatenate([seed_ids, force_seed]).astype(np.int64))
    # Safety cap: subsample huge seeds to avoid multi-hour bootstrap stalls
    max_seed_size = _env_int("QUERY_LIFT_BOOTSTRAP_MAX_SEED_SIZE", 5000, minimum=128)
    if seed_ids.size > max_seed_size:
        rng = np.random.default_rng(int(seed_index) + 42)
        seed_ids = rng.choice(seed_ids, size=max_seed_size, replace=False).astype(np.int64)
    membership[seed_ids] = 1.0
    score = (
        0.30 * support.distance_positive_score
        + 0.24 * support.full_mean
        + 0.18 * support.presence
        + 0.14 * support.positive_score
        + 0.10 * support.support_score
        - 0.18 * support.distance_negative_score
        - 0.08 * support.outer_ratio
        - 0.04 * support.negative_score
    ).astype(np.float32)
    score[seed_ids] += 0.22
    selection_cap = int(
        min(
            num_gaussians,
            max(1, int(max_gaussians)) if max_gaussians is not None else num_gaussians,
        )
    )
    if seed_ids.size > selection_cap:
        seed_ids = seed_ids[
            np.argsort(-score[seed_ids], kind="mergesort")[:selection_cap]
        ]
        membership[:] = 0.0
        membership[seed_ids] = 1.0
    frame_count = len(support.samples)
    if frame_count == 0:
        return seed_ids, {"bootstrap_iters": 0, "bootstrap_best_iou": 0.0}
    if _env_flag("QUERY_LIFT_BOOTSTRAP_PROXY_EVIDENCE_ONLY", False):
        return _bootstrap_proxy_evidence_selection(
            seed_ids,
            support=support,
            num_gaussians=num_gaussians,
            target_area_ratio=target_area_ratio,
        )
    # Ensure all support sample prepared frames are on the target device
    # so that gaussian_region_alpha_masses runs on GPU, not CPU.
    for _si in range(frame_count):
        _sp = support.samples[_si].prepared
        if str(_sp.gaussian_ids.device) != str(device):
            _sp.gaussian_ids = _sp.gaussian_ids.to(device)
            _sp.centers_xy = _sp.centers_xy.to(device)
            _sp.sigma_px = _sp.sigma_px.to(device)
            _sp.alpha_weight = _sp.alpha_weight.to(device)
            _sp.depth = _sp.depth.to(device)
    all_indices = np.arange(frame_count, dtype=np.int32)
    best_ids = seed_ids.copy()
    fast_internal_metrics = _env_flag("QUERY_LIFT_BOOTSTRAP_FAST_INTERNAL_METRICS", True)

    def internal_metrics(ids_for_metric: np.ndarray) -> dict[str, Any]:
        if fast_internal_metrics:
            return _support_proxy_metrics(ids_for_metric, support=support, scoring_mode="hybrid_recall")
        return _selection_metrics(ids_for_metric, support=support, num_gaussians=num_gaussians, device=device, scoring_mode="hybrid_recall")

    best_metrics = internal_metrics(best_ids)
    last_metrics = dict(best_metrics)
    best_score = float(best_metrics.get("score", float("-inf")))
    best_feasible_ids = best_ids.copy() if (
        float(best_metrics.get("area_ratio", 0.0)) >= target_area_ratio
        and float(best_metrics.get("precision", 0.0)) >= feasible_precision
        and float(best_metrics.get("outer_leakage", 1.0)) <= feasible_leakage
    ) else np.empty((0,), dtype=np.int64)
    best_feasible_metrics = best_metrics if best_feasible_ids.size else None
    best_feasible_score = best_score if best_feasible_ids.size else float("-inf")
    history = []
    stagnant_rounds = 0
    last_eval_iou = float("-inf")

    for iteration in range(max_iter):
        if time.time() - t_start > wall_timeout:
            break
        if frame_count <= batch_size:
            batch = all_indices
        else:
            start = (int(seed_index) * 3 + iteration * batch_size) % frame_count
            batch = np.asarray([(start + offset) % frame_count for offset in range(batch_size)], dtype=np.int32)
        delta = np.zeros((num_gaussians,), dtype=np.float32)
        seen = np.zeros((num_gaussians,), dtype=np.float32)
        force_add_votes = np.zeros((num_gaussians,), dtype=np.float32)
        force_drop_votes = np.zeros((num_gaussians,), dtype=np.float32)
        # Area-target pressure: if the rendered entity is still too small, lower
        # the effective threshold and increase positive interior evidence in the
        # next update. Boundary remains mostly neutral; expansion is graph-local.
        current_area_ratio = float(last_metrics.get("area_ratio", 0.0))
        area_deficit = max(0.0, float(target_area_ratio) - current_area_ratio)
        positive_boost = 1.0 + min(1.25, 2.2 * area_deficit / max(float(target_area_ratio), 1.0e-6))
        effective_keep_threshold = float(keep_threshold) - min(0.12, 0.26 * area_deficit)
        for sample_index in batch.tolist():
            sample = support.samples[int(sample_index)]
            regions = support.distance_regions[int(sample_index)]
            masses = gaussian_region_alpha_masses(sample.prepared, {
                "full": regions["inside_soft"],
                "core": regions["inside_core"],
                "boundary": regions["inside_force"],
                "outer": regions["outer_far"],
            })
            ids = sample.prepared.gaussian_ids.detach().cpu().numpy().astype(np.int64)
            visible = masses["visible"].clamp_min(1.0e-6)
            inside_soft = (masses["full"] / visible).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
            inside_core = (masses["core"] / visible).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
            inside_force = (masses["boundary"] / visible).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
            outer_far = (masses["outer"] / visible).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
            outer_near = np.clip(1.0 - inside_soft - outer_far, 0.0, 1.0).astype(np.float32)
            # Boundary is neutral. Interior grows with distance from boundary;
            # exterior deletion grows with distance and becomes hard past outer_far.
            boundary_weight = 0.04 if thin_mode else 0.02
            delta_local = (
                positive_boost * (0.62 * inside_soft + 1.22 * inside_core + 1.85 * inside_force)
                + 0.48 * support.distance_positive_score[ids]
                + boundary_weight * support.core_ratio[ids]
                - 0.22 * outer_near
                - 1.05 * outer_far
                - 0.12 * support.distance_negative_score[ids]
            )
            delta[ids] += delta_local.astype(np.float32)
            seen[ids] += 1.0
            force_add_votes[ids] += (inside_force >= force_add).astype(np.float32)
            force_drop_votes[ids] += (outer_far >= force_drop).astype(np.float32)
        valid = seen > 0.0
        averaged = np.zeros_like(delta)
        averaged[valid] = delta[valid] / np.clip(seen[valid], 1.0, None)
        score = 0.78 * score + 0.22 * averaged
        add_prior = (support.distance_force_add >= 0.08) & (support.distance_force_drop < 0.55)
        drop_prior = (support.distance_force_drop >= 0.75) & (support.distance_positive_score < 0.08)
        force_add_votes[add_prior] += max(1.0, 0.35 * float(batch.size))
        force_drop_votes[drop_prior] += max(1.0, 0.45 * float(batch.size))
        add_mask = force_add_votes >= max(1.0, 0.40 * float(batch.size))
        drop_mask = force_drop_votes >= max(1.0, 0.45 * float(batch.size))
        membership = np.where(score >= effective_keep_threshold, 1.0, membership * 0.65).astype(np.float32)
        membership[add_mask] = 1.0
        membership[drop_mask] = 0.0
        membership[(support.outer_ratio > 0.88) & (support.positive_score < 0.08) & (support.distance_positive_score < 0.12)] = 0.0
        ids = np.where(membership > 0.5)[0].astype(np.int64)
        if ids.size > selection_cap:
            ids = ids[np.argsort(-score[ids], kind="mergesort")[:selection_cap]]
            membership[:] = 0.0
            membership[ids] = 1.0
        if area_deficit > 0.0 and ids.size:
            expand_add = int(max(16, min(512, round(ids.size * min(0.35, area_deficit / max(float(target_area_ratio), 1.0e-6))))))
            expanded_ids = _safe_graph_expand_from_selection(ids, support=support, bank=bank or _BOOTSTRAP_BANK_CACHE, max_add=expand_add, graph_knn=24, radius_scale=2.10 if thin_mode else 1.75)
            if expanded_ids.size > ids.size:
                ids = expanded_ids.astype(np.int64)
                membership[:] = 0.0
                membership[ids] = 1.0
        if ids.size > selection_cap:
            ids = ids[np.argsort(-score[ids], kind="mergesort")[:selection_cap]]
            membership[:] = 0.0
            membership[ids] = 1.0
        if ids.size == 0:
            ids = np.argsort(-score, kind="mergesort")[: max(16, min(256, num_gaussians))].astype(np.int64)
            membership[ids] = 1.0
        # Light component cleanup only for non-thin objects. Thin shells may be disconnected.
        if not thin_mode and ids.size > 32:
            sampled = np.asarray(support.sampled_indices, dtype=np.int32)
            trajectories = np.asarray((bank or _BOOTSTRAP_BANK_CACHE or {}).get("trajectories"), dtype=np.float32) if (bank or _BOOTSTRAP_BANK_CACHE) else None
            if trajectories is not None and trajectories.size:
                points = trajectories[:, sampled, :].mean(axis=1) if sampled.size else trajectories.mean(axis=1)
                comps = _component_labels(points, ids, graph_knn=24, radius_scale=1.8)
                keep = [comp for comp in comps if comp.size >= max(8, int(0.015 * ids.size))]
                if keep:
                    ids = np.unique(np.concatenate(keep).astype(np.int64))
                    membership[:] = 0.0
                    membership[ids] = 1.0
        render_eval = ((iteration + 1) % max(int(eval_interval), 1) == 0) or (iteration + 1 == max_iter)
        if render_eval:
            metrics = internal_metrics(ids)
            last_metrics = dict(metrics)
        else:
            metrics = dict(last_metrics)
        batch_iou = float(metrics.get("rendered_iou_stage1", 0.0))
        history.append({
            "iter": int(iteration + 1),
            "render_eval": bool(render_eval),
            "gaussian_count": int(ids.size),
            "iou": batch_iou,
            "rendered_iou_stage1": batch_iou,
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "area_ratio": float(metrics.get("area_ratio", 0.0)),
            "outer_leakage": float(metrics.get("outer_leakage", 0.0)),
            "score": float(metrics.get("score", 0.0)),
            "inside_reward": float(np.mean(support.distance_positive_score[ids])) if ids.size else 0.0,
            "outside_leakage": float(np.mean(support.distance_negative_score[ids])) if ids.size else 0.0,
        })
        candidate_score = float(metrics.get("score", float("-inf")))
        if render_eval and candidate_score > best_score:
            best_score = candidate_score
            best_ids = ids.copy()
            best_metrics = metrics
        feasible = (
            float(metrics.get("area_ratio", 0.0)) >= target_area_ratio
            and float(metrics.get("precision", 0.0)) >= feasible_precision
            and float(metrics.get("outer_leakage", 1.0)) <= feasible_leakage
        )
        if render_eval and feasible and candidate_score > best_feasible_score:
            best_feasible_score = candidate_score
            best_feasible_ids = ids.copy()
            best_feasible_metrics = metrics
        if render_eval:
            if batch_iou > last_eval_iou + 1e-4:
                stagnant_rounds = 0
                last_eval_iou = batch_iou
            else:
                stagnant_rounds += 1
            if batch_iou >= target_iou:
                break
            if stagnant_rounds >= 3:
                break
    if best_feasible_ids.size:
        best_ids = best_feasible_ids
        best_metrics = best_feasible_metrics or best_metrics
    if _env_flag("QUERY_LIFT_BOOTSTRAP_FINAL_RENDER_METRICS", True):
        best_metrics = _selection_metrics(best_ids, support=support, num_gaussians=num_gaussians, device=device, scoring_mode="hybrid_recall")
    info = {
        "score": float(best_metrics.get("score", 0.0)),
        "rendered_iou_stage1": float(best_metrics.get("rendered_iou_stage1", 0.0)),
        "precision": float(best_metrics.get("precision", 0.0)),
        "recall": float(best_metrics.get("recall", 0.0)),
        "area_ratio": float(best_metrics.get("area_ratio", 0.0)),
        "outer_leakage": float(best_metrics.get("outer_leakage", 0.0)),
        "active_frame_coverage": float(best_metrics.get("active_frame_coverage", 0.0)),
        "area_cv": float(best_metrics.get("area_cv", 1.0)),
        "mean_pred_area": float(best_metrics.get("mean_pred_area", 0.0)),
        "bootstrap_iters": int(len(history)),
        "bootstrap_best_iou": float(best_metrics.get("rendered_iou_stage1", 0.0)),
        "bootstrap_best_precision": float(best_metrics.get("precision", 0.0)),
        "bootstrap_best_recall": float(best_metrics.get("recall", 0.0)),
        "bootstrap_best_area_ratio": float(best_metrics.get("area_ratio", 0.0)),
        "bootstrap_target_area_ratio": float(target_area_ratio),
        "bootstrap_used_feasible_area_candidate": bool(best_feasible_ids.size),
        "bootstrap_history": history,
    }
    return np.unique(best_ids.astype(np.int64)), info


_BOOTSTRAP_BANK_CACHE: dict[str, np.ndarray] | None = None
_BOUNDARY_REFINE_INFO_CACHE: dict[str, dict[str, Any]] = {}


def _refine_cache_key(alias: str, name: str) -> str:
    return f"{str(alias)}::{str(name)}"


def _query_phase_mode_from_plan(query_plan: dict[str, Any] | None) -> str:
    if not isinstance(query_plan, dict):
        return "unspecified"
    text = " ".join(
        str(query_plan.get(key, ""))
        for key in ("query", "query_text", "raw_query")
        if query_plan.get(key) is not None
    ).lower()
    before_terms = (
        "before", "pre-cut", "pre split", "pre_split", "intact", "empty",
    )
    after_terms = (
        "after", "post-cut", "post split", "post_split", "broken", "full", "melted",
    )
    action_terms = (
        "during", "while", "action", "cutting", "breaking", "pouring",
    )
    if any(term in text for term in before_terms):
        return "before"
    if any(term in text for term in after_terms):
        return "after"
    if any(term in text for term in action_terms):
        return "action"
    return "unspecified"


def _filter_variants_for_query_phase(variants: list[dict[str, Any]], query_plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    mode = _query_phase_mode_from_plan(query_plan)
    if mode == "unspecified" or not variants:
        return variants

    def variant_text(row: dict[str, Any]) -> str:
        return " ".join(str(row.get(key, "")) for key in ("alias", "phase", "variant_kind", "description")).lower()

    if mode == "before":
        keep = [
            row for row in variants
            if any(term in variant_text(row) for term in ("pre", "before", "initial", "empty", "intact"))
            and not any(term in variant_text(row) for term in ("post", "after", "successor", "broken", "full", "melted"))
        ]
        return keep or variants
    if mode == "after":
        keep = [
            row for row in variants
            if any(term in variant_text(row) for term in ("post", "after", "successor", "broken", "full", "melted"))
            and not any(term in variant_text(row) for term in ("pre", "before", "initial", "empty", "intact"))
        ]
        return keep or variants
    if mode == "action":
        keep = [
            row for row in variants
            if any(term in variant_text(row) for term in ("action", "active", "during", "interaction", "contact", "window"))
        ]
        selected = keep or variants
    else:
        selected = variants
    max_phase_variants = _env_optional_int("QUERY_LIFT_MAX_PHASE_VARIANTS", 1, minimum=1)
    if max_phase_variants is not None and len(selected) > max_phase_variants:
        return selected[: int(max_phase_variants)]
    return selected


def _boundary_refine_selection(
    seed_ids: np.ndarray,
    support: LiftingSupport,
    num_gaussians: int,
    device: str,
    seed_index: int = 0,
    min_gaussians: int = 64,
    max_gaussians: int = 1280,
) -> tuple[np.ndarray, dict[str, Any]]:
    max_iter = _env_int("QUERY_LIFT_BOUNDARY_MAX_ITER", 10, minimum=1)
    batch_size = _env_int("QUERY_LIFT_BOUNDARY_BATCH_FRAMES", 12, minimum=1)
    target_iou = _env_float("QUERY_LIFT_BOUNDARY_TARGET_IOU", 0.90, minimum=0.0, maximum=1.0)
    keep_threshold = _env_float("QUERY_LIFT_BOUNDARY_KEEP_THRESHOLD", 0.20)
    hard_drop_vote = _env_float("QUERY_LIFT_BOUNDARY_DROP_VOTE", 0.28, minimum=0.0, maximum=1.0)
    hard_add_vote = _env_float("QUERY_LIFT_BOUNDARY_ADD_VOTE", 0.36, minimum=0.0, maximum=1.0)
    target_area_ratio = _env_float("QUERY_LIFT_BOUNDARY_TARGET_AREA_RATIO", 1.00, minimum=0.05, maximum=2.0)
    max_area_ratio = _env_float("QUERY_LIFT_BOUNDARY_MAX_AREA_RATIO", 1.18, minimum=0.10, maximum=4.0)
    min_area_ratio = _env_float("QUERY_LIFT_BOUNDARY_MIN_AREA_RATIO", 0.45, minimum=0.0, maximum=2.0)
    min_keep = int(max(8, min(int(min_gaussians), _env_int("QUERY_LIFT_BOUNDARY_MIN_GAUSSIANS", 64, minimum=8))))
    max_keep = int(max(min_keep, min(int(max_gaussians), _env_int("QUERY_LIFT_BOUNDARY_MAX_GAUSSIANS", int(max_gaussians), minimum=16))))

    frame_count = len(support.samples)
    seed_ids = np.unique(np.asarray(seed_ids, dtype=np.int64).reshape(-1))
    seed_ids = seed_ids[(seed_ids >= 0) & (seed_ids < num_gaussians)]
    if frame_count == 0:
        return seed_ids, {"boundary_refine_iters": 0, "boundary_refine_best_iou": 0.0}

    base_score = (
        0.62 * support.distance_positive_score
        + 0.24 * support.full_mean
        + 0.18 * support.core_ratio
        + 0.16 * support.presence
        + 0.12 * support.purity
        - 0.92 * support.distance_negative_score
        - 0.38 * support.outer_ratio
        - 0.18 * support.negative_score
    ).astype(np.float32)
    force_inside = np.where((support.distance_force_add >= 0.16) & (support.distance_force_drop < 0.34))[0].astype(np.int64)
    if force_inside.size:
        seed_ids = np.unique(np.concatenate([seed_ids, force_inside]).astype(np.int64))
    score = base_score.copy()
    score[seed_ids] += 0.18

    all_indices = np.arange(frame_count, dtype=np.int32)
    best_ids = seed_ids.copy()
    if best_ids.size > max_keep:
        best_ids = best_ids[np.argsort(-score[best_ids], kind="mergesort")[:max_keep]]
    best_metrics = _selection_metrics(best_ids, support=support, num_gaussians=num_gaussians, device=device, scoring_mode="mask_boundary_refine")
    best_score = float(best_metrics.get("score", float("-inf")))
    best_iou = float(best_metrics.get("rendered_iou_stage1", 0.0))
    history: list[dict[str, Any]] = []
    threshold = float(keep_threshold)
    stop_reason = "max_iter"

    for iteration in range(max_iter):
        if frame_count <= batch_size:
            batch = all_indices
        else:
            start = (int(seed_index) * 5 + iteration * batch_size) % frame_count
            batch = np.asarray([(start + offset) % frame_count for offset in range(batch_size)], dtype=np.int32)

        delta = np.zeros((num_gaussians,), dtype=np.float32)
        seen = np.zeros((num_gaussians,), dtype=np.float32)
        add_votes = np.zeros((num_gaussians,), dtype=np.float32)
        drop_votes = np.zeros((num_gaussians,), dtype=np.float32)
        for sample_index in batch.tolist():
            sample = support.samples[int(sample_index)]
            regions = support.distance_regions[int(sample_index)]
            masses = gaussian_region_alpha_masses(
                sample.prepared,
                {
                    "full": regions["inside_soft"],
                    "core": regions["inside_core"],
                    "boundary": regions["inside_force"],
                    "outer": regions["outer_far"],
                },
            )
            ids = sample.prepared.gaussian_ids.detach().cpu().numpy().astype(np.int64)
            visible = masses["visible"].clamp_min(1.0e-6)
            inside_soft = (masses["full"] / visible).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
            inside_core = (masses["core"] / visible).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
            inside_force = (masses["boundary"] / visible).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
            outer_far = (masses["outer"] / visible).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
            outer_near = np.clip(1.0 - inside_soft - outer_far, 0.0, 1.0).astype(np.float32)
            local_delta = (
                0.72 * inside_soft
                + 1.42 * inside_core
                + 2.35 * inside_force
                + 0.34 * support.distance_positive_score[ids]
                - 0.54 * outer_near
                - 2.30 * outer_far
                - 0.40 * support.distance_negative_score[ids]
            )
            delta[ids] += local_delta.astype(np.float32)
            seen[ids] += 1.0
            add_votes[ids] += ((inside_core >= 0.30) | (inside_force >= 0.20)).astype(np.float32)
            drop_votes[ids] += (outer_far >= 0.46).astype(np.float32)

        valid = seen > 0.0
        averaged = np.zeros_like(delta)
        averaged[valid] = delta[valid] / np.clip(seen[valid], 1.0, None)
        score = (0.56 * score + 0.44 * averaged).astype(np.float32)
        score += 0.08 * support.distance_positive_score.astype(np.float32)
        score -= 0.18 * support.distance_negative_score.astype(np.float32)

        add_mask = (add_votes / max(float(batch.size), 1.0)) >= hard_add_vote
        drop_mask = (drop_votes / max(float(batch.size), 1.0)) >= hard_drop_vote
        drop_mask |= (support.distance_force_drop >= 0.42) & (support.distance_positive_score < 0.12)
        drop_mask |= (support.outer_ratio >= 0.72) & (support.core_ratio < 0.04)

        candidate_mask = (score >= threshold) | add_mask
        candidate_mask[drop_mask] = False
        candidate_ids = np.where(candidate_mask)[0].astype(np.int64)
        if candidate_ids.size < min_keep:
            eligible = np.where(~drop_mask & ((support.support_score > 0.0) | (support.distance_positive_score > 0.0)))[0].astype(np.int64)
            if eligible.size:
                ranked = eligible[np.argsort(-score[eligible], kind="mergesort")]
                candidate_ids = np.unique(np.concatenate([candidate_ids, ranked[: min(min_keep, ranked.size)]]).astype(np.int64))
        if candidate_ids.size > max_keep:
            candidate_ids = candidate_ids[np.argsort(-score[candidate_ids], kind="mergesort")[:max_keep]]

        metrics = _selection_metrics(candidate_ids, support=support, num_gaussians=num_gaussians, device=device, scoring_mode="mask_boundary_refine")
        area_ratio = float(metrics.get("area_ratio", 0.0))
        if area_ratio > max_area_ratio and candidate_ids.size > min_keep:
            keep_count = int(max(min_keep, round(candidate_ids.size * max(0.50, target_area_ratio / max(area_ratio, 1.0e-6)))))
            candidate_ids = candidate_ids[np.argsort(-score[candidate_ids], kind="mergesort")[:keep_count]]
            threshold += 0.035
            metrics = _selection_metrics(candidate_ids, support=support, num_gaussians=num_gaussians, device=device, scoring_mode="mask_boundary_refine")
            area_ratio = float(metrics.get("area_ratio", 0.0))
        elif area_ratio < min_area_ratio:
            threshold -= 0.020

        direct_iou = float(metrics.get("rendered_iou_stage1", 0.0))
        history.append(
            {
                "iter": int(iteration + 1),
                "gaussian_count": int(candidate_ids.size),
                "iou": direct_iou,
                "direct_gaussian_iou_stage1": direct_iou,
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "area_ratio": float(metrics.get("area_ratio", 0.0)),
                "outer_leakage": float(metrics.get("outer_leakage", 0.0)),
                "score": float(metrics.get("score", 0.0)),
                "keep_threshold": float(threshold),
                "mean_inside_reward": float(np.mean(support.distance_positive_score[candidate_ids])) if candidate_ids.size else 0.0,
                "mean_outside_penalty": float(np.mean(support.distance_negative_score[candidate_ids])) if candidate_ids.size else 0.0,
            }
        )
        candidate_score = float(metrics.get("score", float("-inf")))
        if direct_iou > best_iou or (abs(direct_iou - best_iou) <= 1.0e-6 and candidate_score > best_score):
            best_score = candidate_score
            best_iou = direct_iou
            best_ids = candidate_ids.copy()
            best_metrics = metrics
        if direct_iou >= target_iou:
            stop_reason = "target_iou"
            best_score = candidate_score
            best_iou = direct_iou
            best_ids = candidate_ids.copy()
            best_metrics = metrics
            break

    best_ids = np.unique(np.asarray(best_ids, dtype=np.int64))
    info = {
        "boundary_refine_iters": int(len(history)),
        "boundary_refine_best_iou": float(best_metrics.get("rendered_iou_stage1", 0.0)),
        "boundary_refine_best_precision": float(best_metrics.get("precision", 0.0)),
        "boundary_refine_best_recall": float(best_metrics.get("recall", 0.0)),
        "boundary_refine_best_area_ratio": float(best_metrics.get("area_ratio", 0.0)),
        "boundary_refine_best_outer_leakage": float(best_metrics.get("outer_leakage", 0.0)),
        "boundary_refine_target_area_ratio": float(target_area_ratio),
        "boundary_refine_target_iou": float(target_iou),
        "boundary_refine_stop_reason": str(stop_reason),
        "boundary_refine_direct_gaussian_iou_stage1": float(best_metrics.get("rendered_iou_stage1", 0.0)),
        "boundary_refine_history": history,
    }
    return best_ids, info


def _shape_refine_selection(
    seed_ids: np.ndarray,
    support: LiftingSupport,
    num_gaussians: int,
    device: str,
    seed_index: int = 0,
    min_gaussians: int = 128,
    max_gaussians: int = 3072,
) -> tuple[np.ndarray, dict[str, Any]]:
    max_iter = _env_int("QUERY_LIFT_SHAPE_MAX_ITER", 10, minimum=1)
    batch_size = _env_int("QUERY_LIFT_SHAPE_BATCH_FRAMES", 12, minimum=1)
    target_iou = _env_float("QUERY_LIFT_SHAPE_TARGET_IOU", 0.70, minimum=0.0, maximum=1.0)
    target_area_ratio = _env_float("QUERY_LIFT_SHAPE_TARGET_AREA_RATIO", 0.85, minimum=0.05, maximum=2.0)
    min_area_ratio = _env_float("QUERY_LIFT_SHAPE_MIN_AREA_RATIO", 0.45, minimum=0.0, maximum=2.0)
    max_area_ratio = _env_float("QUERY_LIFT_SHAPE_MAX_AREA_RATIO", 1.25, minimum=0.10, maximum=4.0)
    min_precision = _env_float("QUERY_LIFT_SHAPE_MIN_PRECISION", 0.55, minimum=0.0, maximum=1.0)
    max_outer_leakage = _env_float("QUERY_LIFT_SHAPE_MAX_OUTER_LEAKAGE", 0.16, minimum=0.0, maximum=1.0)
    keep_threshold = _env_float("QUERY_LIFT_SHAPE_KEEP_THRESHOLD", 0.18, minimum=0.0)
    hard_drop_vote = _env_float("QUERY_LIFT_SHAPE_DROP_VOTE", 0.28, minimum=0.0, maximum=1.0)
    hard_add_vote = _env_float("QUERY_LIFT_SHAPE_ADD_VOTE", 0.36, minimum=0.0, maximum=1.0)
    min_keep = int(max(8, min(int(min_gaussians), _env_int("QUERY_LIFT_SHAPE_MIN_GAUSSIANS", int(min_gaussians), minimum=8))))
    max_keep = int(max(min_keep, min(int(max_gaussians), _env_int("QUERY_LIFT_SHAPE_MAX_GAUSSIANS", int(max_gaussians), minimum=16))))

    frame_count = len(support.samples)
    seed_ids = np.unique(np.asarray(seed_ids, dtype=np.int64).reshape(-1))
    seed_ids = seed_ids[(seed_ids >= 0) & (seed_ids < num_gaussians)]
    if frame_count == 0:
        return seed_ids, {"shape_refine_iters": 0, "shape_refine_best_iou": 0.0}

    thin_mode = _support_is_geometrically_thin(support)

    # For thin/transparent objects, reduce penalty on outer_ratio and negative_score
    # because Gaussians near the thin surface inherently sit in boundary/outer bands.
    if thin_mode:
        outer_penalty = 0.34  # moderate penalty: 0.8x of baseline 0.42
        neg_penalty = 0.16
    else:
        outer_penalty = 0.42
        neg_penalty = 0.20

    base_score = (
        0.60 * support.distance_positive_score
        + 0.26 * support.full_mean
        + 0.20 * support.core_ratio
        + 0.16 * support.presence
        + 0.12 * support.purity
        - 0.88 * support.distance_negative_score
        - outer_penalty * support.outer_ratio
        - neg_penalty * support.negative_score
    ).astype(np.float32)
    force_inside = np.where((support.distance_force_add >= 0.16) & (support.distance_force_drop < 0.34))[0].astype(np.int64)
    if force_inside.size:
        seed_ids = np.unique(np.concatenate([seed_ids, force_inside]).astype(np.int64))
    score = base_score.copy()
    score[seed_ids] += 0.18

    all_indices = np.arange(frame_count, dtype=np.int32)
    best_ids = seed_ids.copy()
    if best_ids.size > max_keep:
        best_ids = best_ids[np.argsort(-score[best_ids], kind="mergesort")[:max_keep]]
    best_metrics = _selection_metrics(best_ids, support=support, num_gaussians=num_gaussians, device=device, scoring_mode="mask_shape_refine_v2")
    best_score = float(best_metrics.get("score", float("-inf")))
    best_iou = float(best_metrics.get("rendered_iou_stage1", 0.0))
    history: list[dict[str, Any]] = []
    threshold = float(keep_threshold)
    stop_reason = "max_iter"

    for iteration in range(max_iter):
        if frame_count <= batch_size:
            batch = all_indices
        else:
            start = (int(seed_index) * 7 + iteration * batch_size) % frame_count
            batch = np.asarray([(start + offset) % frame_count for offset in range(batch_size)], dtype=np.int32)

        delta = np.zeros((num_gaussians,), dtype=np.float32)
        seen = np.zeros((num_gaussians,), dtype=np.float32)
        add_votes = np.zeros((num_gaussians,), dtype=np.float32)
        drop_votes = np.zeros((num_gaussians,), dtype=np.float32)
        for sample_index in batch.tolist():
            sample = support.samples[int(sample_index)]
            regions = support.distance_regions[int(sample_index)]
            masses = gaussian_region_alpha_masses(
                sample.prepared,
                {
                    "full": regions["inside_soft"],
                    "core": regions["inside_core"],
                    "boundary": regions["inside_force"],
                    "outer": regions["outer_far"],
                },
            )
            ids = sample.prepared.gaussian_ids.detach().cpu().numpy().astype(np.int64)
            visible = masses["visible"].clamp_min(1.0e-6)
            inside_soft = (masses["full"] / visible).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
            inside_core = (masses["core"] / visible).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
            inside_force = (masses["boundary"] / visible).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
            outer_far = (masses["outer"] / visible).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
            outer_near = np.clip(1.0 - inside_soft - outer_far, 0.0, 1.0).astype(np.float32)
            # Shape v2: stronger interior rewards, weaker boundary, aggressive exterior penalty.
            # For thin objects, reduce outer_near penalty and boost interior signals to
            # allow Gaussians near the thin surface (cup wall, bottle edge) to survive.
            if thin_mode:
                # Moderate interior boost: +50% over baseline, penalty ×0.8.
                local_delta = (
                    1.02 * inside_soft
                    + 2.07 * inside_core
                    + 3.30 * inside_force
                    + 0.57 * support.distance_positive_score[ids]
                    - 0.38 * outer_near
                    - 1.92 * outer_far
                    - 0.34 * support.distance_negative_score[ids]
                )
            else:
                local_delta = (
                    0.68 * inside_soft
                    + 1.38 * inside_core
                    + 2.20 * inside_force
                    + 0.38 * support.distance_positive_score[ids]
                    - 0.48 * outer_near
                    - 2.40 * outer_far
                    - 0.42 * support.distance_negative_score[ids]
                )
            delta[ids] += local_delta.astype(np.float32)
            seen[ids] += 1.0
            add_votes[ids] += ((inside_core >= 0.28) | (inside_force >= 0.18)).astype(np.float32)
            drop_votes[ids] += (outer_far >= 0.52).astype(np.float32)

        valid = seen > 0.0
        averaged = np.zeros_like(delta)
        averaged[valid] = delta[valid] / np.clip(seen[valid], 1.0, None)
        score = (0.58 * score + 0.42 * averaged).astype(np.float32)
        score += 0.10 * support.distance_positive_score.astype(np.float32)
        score -= 0.20 * support.distance_negative_score.astype(np.float32)

        add_mask = (add_votes / max(float(batch.size), 1.0)) >= hard_add_vote
        drop_mask = (drop_votes / max(float(batch.size), 1.0)) >= hard_drop_vote
        drop_mask |= (support.distance_force_drop >= 0.46) & (support.distance_positive_score < 0.10)
        # For thin objects, be more lenient on drop
        if not thin_mode:
            drop_mask |= (support.outer_ratio >= 0.68) & (support.core_ratio < 0.05)

        candidate_mask = (score >= threshold) | add_mask
        candidate_mask[drop_mask] = False
        candidate_ids = np.where(candidate_mask)[0].astype(np.int64)
        if candidate_ids.size < min_keep:
            eligible = np.where(~drop_mask & ((support.support_score > 0.0) | (support.distance_positive_score > 0.0)))[0].astype(np.int64)
            if eligible.size:
                ranked = eligible[np.argsort(-score[eligible], kind="mergesort")]
                candidate_ids = np.unique(np.concatenate([candidate_ids, ranked[: min(min_keep, ranked.size)]]).astype(np.int64))
        if candidate_ids.size > max_keep:
            candidate_ids = candidate_ids[np.argsort(-score[candidate_ids], kind="mergesort")[:max_keep]]

        metrics = _selection_metrics(candidate_ids, support=support, num_gaussians=num_gaussians, device=device, scoring_mode="mask_shape_refine_v2")
        area_ratio = float(metrics.get("area_ratio", 0.0))
        if area_ratio > max_area_ratio and candidate_ids.size > min_keep:
            keep_count = int(max(min_keep, round(candidate_ids.size * max(0.50, target_area_ratio / max(area_ratio, 1.0e-6)))))
            candidate_ids = candidate_ids[np.argsort(-score[candidate_ids], kind="mergesort")[:keep_count]]
            threshold += 0.030
            metrics = _selection_metrics(candidate_ids, support=support, num_gaussians=num_gaussians, device=device, scoring_mode="mask_shape_refine_v2")
            area_ratio = float(metrics.get("area_ratio", 0.0))
        elif area_ratio < min_area_ratio and candidate_ids.size < max_keep:
            threshold -= 0.018
            # For thin objects, accelerate threshold decay to admit more boundary Gaussians
            if thin_mode and area_ratio < 0.15:
                threshold -= 0.022

        direct_iou = float(metrics.get("rendered_iou_stage1", 0.0))
        history.append({
            "iter": int(iteration + 1),
            "gaussian_count": int(candidate_ids.size),
            "rendered_iou_stage1": direct_iou,
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "area_ratio": float(metrics.get("area_ratio", 0.0)),
            "outer_leakage": float(metrics.get("outer_leakage", 0.0)),
            "score": float(metrics.get("score", 0.0)),
            "keep_threshold": float(threshold),
            "inside_reward": float(np.mean(support.distance_positive_score[candidate_ids])) if candidate_ids.size else 0.0,
            "outside_penalty": float(np.mean(support.distance_negative_score[candidate_ids])) if candidate_ids.size else 0.0,
        })
        candidate_score = float(metrics.get("score", float("-inf")))
        if direct_iou > best_iou or (abs(direct_iou - best_iou) <= 1.0e-6 and candidate_score > best_score):
            best_score = candidate_score
            best_iou = direct_iou
            best_ids = candidate_ids.copy()
            best_metrics = metrics
        if direct_iou >= target_iou:
            stop_reason = "target_iou"
            best_score = candidate_score
            best_iou = direct_iou
            best_ids = candidate_ids.copy()
            best_metrics = metrics
            break

    best_ids = np.unique(np.asarray(best_ids, dtype=np.int64))
    info = {
        "shape_refine_iters": int(len(history)),
        "shape_refine_best_iou": float(best_metrics.get("rendered_iou_stage1", 0.0)),
        "shape_refine_best_precision": float(best_metrics.get("precision", 0.0)),
        "shape_refine_best_recall": float(best_metrics.get("recall", 0.0)),
        "shape_refine_best_area_ratio": float(best_metrics.get("area_ratio", 0.0)),
        "shape_refine_best_outer_leakage": float(best_metrics.get("outer_leakage", 0.0)),
        "shape_refine_target_area_ratio": float(target_area_ratio),
        "shape_refine_target_iou": float(target_iou),
        "shape_refine_stop_reason": str(stop_reason),
        "shape_refine_history": history,
    }
    return best_ids, info


def _candidate_variants_v1_compat(support: LiftingSupport, bank: dict[str, np.ndarray], min_gaussians: int, max_gaussians: int, graph_knn: int, radius_scale: float) -> list[tuple[str, np.ndarray]]:
    del graph_knn, radius_scale
    score = (
        0.46 * support.support_score
        + 0.22 * support.positive_score
        + 0.18 * support.full_mean
        + 0.14 * support.presence
        + 0.08 * support.core_presence
        + 0.06 * support.purity
        - 0.16 * support.outer_ratio
        - 0.12 * support.negative_score
    ).astype(np.float32)
    eligible = np.where((support.support_score > 0.0) & (support.negative_score <= 0.90))[0]
    variants: list[tuple[str, np.ndarray]] = []
    core = _high_precision_core(support, min_gaussians=min_gaussians, max_gaussians=max_gaussians)
    if core.size:
        variants.append(("v1_core", core))
    if eligible.size:
        for keep in (min_gaussians, 256, 384, 512, 768, 1024, max_gaussians):
            keep = int(min(max(int(keep), 1), eligible.size, max_gaussians))
            ids = eligible[np.argsort(-score[eligible], kind="mergesort")[:keep]]
            variants.append((f"v1_compat_top_{keep}", ids.astype(np.int64)))
    return _dedupe_variants(variants)


def _component_union_until_area_variants(
    support: LiftingSupport,
    points: np.ndarray,
    ids: np.ndarray,
    scores: np.ndarray,
    graph_knn: int,
    radius_scale: float,
    max_gaussians: int,
) -> list[tuple[str, np.ndarray]]:
    components = _component_labels(points, ids, graph_knn=graph_knn, radius_scale=radius_scale)
    if not components:
        return []
    components.sort(key=lambda comp: float(np.mean(scores[comp])) * np.sqrt(float(comp.size)), reverse=True)
    variants: list[tuple[str, np.ndarray]] = []
    for top_k in (2, 4, 8):
        selected = np.unique(np.concatenate(components[: min(top_k, len(components))]).astype(np.int64))
        if selected.size > int(max_gaussians):
            selected = selected[np.argsort(-scores[selected], kind="mergesort")[: int(max_gaussians)]]
        variants.append((f"component_union_top{top_k}", selected.astype(np.int64)))
    for target in (0.35, 0.50, 0.70):
        selected_parts = []
        count = 0
        budget = int(max(1, round(float(max_gaussians) * float(target))))
        for comp in components:
            selected_parts.append(comp)
            count += int(comp.size)
            if count >= budget:
                break
        if selected_parts:
            selected = np.unique(np.concatenate(selected_parts).astype(np.int64))
            if selected.size > int(max_gaussians):
                selected = selected[np.argsort(-scores[selected], kind="mergesort")[: int(max_gaussians)]]
            variants.append((f"component_union_until_area_{int(target * 100):02d}", selected.astype(np.int64)))
    return variants


def _candidate_variants_hybrid_recall(support: LiftingSupport, bank: dict[str, np.ndarray], min_gaussians: int, max_gaussians: int, graph_knn: int, radius_scale: float) -> list[tuple[str, np.ndarray]]:
    sampled = np.asarray(support.sampled_indices, dtype=np.int32)
    trajectories = np.asarray(bank["trajectories"], dtype=np.float32)
    points = trajectories[:, sampled, :].mean(axis=1) if sampled.size else trajectories.mean(axis=1)
    soft_score = (
        0.34 * support.distance_positive_score
        + 0.24 * support.full_mean
        + 0.18 * support.presence
        + 0.14 * support.positive_score
        + 0.10 * support.support_score
        - 0.16 * support.distance_negative_score
        - 0.08 * support.outer_ratio
        - 0.04 * support.negative_score
    ).astype(np.float32)
    eligible = np.where(((support.support_score > 0.0) | (support.distance_positive_score > 0.08)) & (support.outer_ratio <= 0.94) & (support.negative_score <= 0.94) & (support.distance_force_drop < 0.80))[0]
    if eligible.size == 0:
        eligible = np.where(support.support_score > 0.0)[0]
    variants = _candidate_variants_v1_compat(support, bank=bank, min_gaussians=min_gaussians, max_gaussians=max_gaussians, graph_knn=graph_knn, radius_scale=radius_scale)
    if eligible.size:
        ordered = eligible[np.argsort(-soft_score[eligible], kind="mergesort")]
        for keep in (128, 256, 384, 512, 768, 1024, 1536, 2048, 3072):
            if keep > max(max_gaussians, 3072):
                continue
            keep_eff = int(min(max(int(keep), 1), ordered.size, max(int(max_gaussians), int(keep))))
            if keep_eff <= 0:
                continue
            variants.append((f"soft_recall_pool_top_{keep_eff}", ordered[:keep_eff].astype(np.int64)))
            variants.append((f"mask_area_matched_top_{keep_eff}", ordered[:keep_eff].astype(np.int64)))
        component_budget = int(min(max(int(max_gaussians), 3072), ordered.size))
        component_pool = ordered[:component_budget]
        variants.extend(_component_union_until_area_variants(
            support=support,
            points=points,
            ids=component_pool,
            scores=soft_score,
            graph_knn=graph_knn,
            radius_scale=radius_scale * 1.65,
            max_gaussians=int(min(max(int(max_gaussians), 3072), ordered.size)),
        ))
        if _support_is_geometrically_thin(support):
            shell_score = (
                0.38 * support.full_mean
                + 0.28 * support.presence
                + 0.18 * support.positive_score
                + 0.10 * support.support_score
                + 0.08 * support.core_ratio
                - 0.10 * support.outer_ratio
                - 0.04 * support.negative_score
            ).astype(np.float32)
            shell_eligible = np.where((support.support_score > 0.0) & (support.outer_ratio <= 0.97))[0]
            if shell_eligible.size:
                shell_order = shell_eligible[np.argsort(-shell_score[shell_eligible], kind="mergesort")]
                for keep in (512, 1024, 1536, 2048, 3072):
                    keep_eff = int(min(max(int(keep), 1), shell_order.size, max(int(max_gaussians), int(keep))))
                    variants.append((f"thin_shell_top_{keep_eff}", shell_order[:keep_eff].astype(np.int64)))
                variants.extend(_component_union_until_area_variants(
                    support=support,
                    points=points,
                    ids=shell_order[: int(min(max(int(max_gaussians), 3072), shell_order.size))],
                    scores=shell_score,
                    graph_knn=graph_knn,
                    radius_scale=radius_scale * 2.10,
                    max_gaussians=int(min(max(int(max_gaussians), 3072), shell_order.size)),
                ))
    return _dedupe_variants(variants)


def _candidate_variants_shape_v2(support: LiftingSupport, bank: dict[str, np.ndarray], min_gaussians: int, max_gaussians: int, graph_knn: int, radius_scale: float) -> list[tuple[str, np.ndarray]]:
    """Shape v2 candidates: start from hybrid_recall variants and add larger pools."""
    variants = _candidate_variants_hybrid_recall(support, bank=bank, min_gaussians=min_gaussians, max_gaussians=max_gaussians, graph_knn=graph_knn, radius_scale=radius_scale)
    sampled = np.asarray(support.sampled_indices, dtype=np.int32)
    trajectories = np.asarray(bank["trajectories"], dtype=np.float32)
    points = trajectories[:, sampled, :].mean(axis=1) if sampled.size else trajectories.mean(axis=1)

    soft_score = (
        0.34 * support.distance_positive_score
        + 0.24 * support.full_mean
        + 0.18 * support.presence
        + 0.14 * support.positive_score
        + 0.10 * support.support_score
        - 0.16 * support.distance_negative_score
        - 0.08 * support.outer_ratio
        - 0.04 * support.negative_score
    ).astype(np.float32)

    eligible = np.where(((support.support_score > 0.0) | (support.distance_positive_score > 0.08)) & (support.outer_ratio <= 0.94) & (support.negative_score <= 0.94) & (support.distance_force_drop < 0.80))[0]
    if eligible.size == 0:
        eligible = np.where(support.support_score > 0.0)[0]

    if eligible.size:
        ordered = eligible[np.argsort(-soft_score[eligible], kind="mergesort")]
        # Add larger pool sizes for shape v2 (2048, 3072)
        for keep in (2048, 3072):
            keep_eff = int(min(max(int(keep), 1), ordered.size, int(max_gaussians)))
            if keep_eff > 0:
                variants.append((f"soft_recall_pool_top_{keep_eff}", ordered[:keep_eff].astype(np.int64)))

        # Add component union variants for area targets
        component_budget = int(min(max(int(max_gaussians), 3072), ordered.size))
        component_pool = ordered[:component_budget]
        variants.extend(_component_union_until_area_variants(
            support=support,
            points=points,
            ids=component_pool,
            scores=soft_score,
            graph_knn=graph_knn,
            radius_scale=radius_scale * 1.80,
            max_gaussians=int(min(max(int(max_gaussians), 3072), ordered.size)),
        ))

        # Add area-targeted component unions (70, 90, 110)
        for area_target in (70, 90, 110):
            target_count = int(max(1, round(float(max_gaussians) * float(area_target) / 100.0)))
            pool = ordered[:min(target_count, ordered.size)]
            variants.extend(_component_union_until_area_variants(
                support=support,
                points=points,
                ids=pool,
                scores=soft_score,
                graph_knn=graph_knn,
                radius_scale=radius_scale * 1.65,
                max_gaussians=target_count,
            ))

        # For thin objects, don't force largest component
        if _support_is_geometrically_thin(support):
            shell_score = (
                0.38 * support.full_mean
                + 0.28 * support.presence
                + 0.18 * support.positive_score
                + 0.10 * support.support_score
                + 0.08 * support.core_ratio
                - 0.10 * support.outer_ratio
                - 0.04 * support.negative_score
            ).astype(np.float32)
            shell_eligible = np.where((support.support_score > 0.0) & (support.outer_ratio <= 0.97))[0]
            if shell_eligible.size:
                shell_order = shell_eligible[np.argsort(-shell_score[shell_eligible], kind="mergesort")]
                for keep in (2048, 3072):
                    keep_eff = int(min(max(int(keep), 1), shell_order.size, int(max_gaussians)))
                    if keep_eff > 0:
                        variants.append((f"thin_shell_top_{keep_eff}", shell_order[:keep_eff].astype(np.int64)))

    return _dedupe_variants(variants)


def _dedupe_variants(variants: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    unique: dict[str, np.ndarray] = {}
    seen: set[tuple[int, ...]] = set()
    for name, ids in variants:
        ids = np.unique(np.asarray(ids, dtype=np.int64))
        if not ids.size:
            continue
        key = tuple(int(value) for value in ids[: min(ids.size, 4096)]) + (int(ids.size),)
        if key in seen:
            continue
        seen.add(key)
        unique[name] = ids
    return list(unique.items())


def _boundary_refine_selection_utility(row: dict[str, Any], target_area_ratio: float) -> float:
    area_ratio = float(row.get("area_ratio", 0.0))
    outer_leakage = float(row.get("outer_leakage", 1.0))
    precision = float(row.get("precision", 0.0))
    recall = float(row.get("recall", 0.0))
    rendered_iou = float(row.get("rendered_iou_stage1", 0.0))
    active_frame_coverage = float(row.get("active_frame_coverage", 0.0))
    area_cv = float(row.get("area_cv", 1.0))
    gaussian_count = float(row.get("gaussian_count", 0.0))
    area_term = min(area_ratio, float(target_area_ratio)) / max(float(target_area_ratio), 1.0e-6)
    over_target = max(0.0, area_ratio - float(target_area_ratio))
    return float(
        0.72 * recall
        + 0.62 * rendered_iou
        + 0.22 * precision
        + 0.30 * area_term
        + 0.22 * active_frame_coverage
        + 0.02 * np.log1p(max(gaussian_count, 0.0))
        - 0.62 * outer_leakage
        - 0.22 * over_target
        - 0.14 * min(area_cv, 2.5)
    )


def _coverage_v4_selection_utility(row: dict[str, Any], target_area_ratio: float) -> float:
    """Coverage-refine utility that strongly penalizes underfilled entities."""
    area_ratio = float(row.get("area_ratio", 0.0))
    target = max(float(target_area_ratio), 1.0e-6)
    min_area = _env_float(
        "QUERY_LIFT_BOOTSTRAP_MIN_AREA_RATIO",
        min(0.35, max(0.25, 0.45 * target)),
        minimum=0.0,
        maximum=2.0,
    )
    under_target = max(0.0, target - area_ratio) / target
    severe_underfill = max(0.0, min_area - area_ratio) / max(min_area, 1.0e-6)
    recall = float(row.get("recall", 0.0))
    rendered_iou = float(row.get("rendered_iou_stage1", 0.0))
    area_term = min(area_ratio, target) / target
    return float(
        _boundary_refine_selection_utility(row, target_area_ratio=target)
        + 0.22 * recall
        + 0.18 * rendered_iou
        + 0.16 * area_term
        - 0.55 * under_target
        - 0.85 * severe_underfill
    )


def _alpha_calibration_levels() -> list[tuple[float, float]]:
    """Read a small, reproducible alpha-binarization grid for v4 lifting."""
    raw = os.environ.get(
        "QUERY_LIFT_ALPHA_CALIBRATION_LEVELS",
        "0.18:0.015,0.08:0.006,0.03:0.002,0.01:0.001",
    )
    levels: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for item in str(raw).split(","):
        pieces = [piece.strip() for piece in item.split(":")]
        if len(pieces) != 2:
            continue
        try:
            relative = float(pieces[0])
            absolute = float(pieces[1])
        except ValueError:
            continue
        if not (np.isfinite(relative) and np.isfinite(absolute)):
            continue
        level = (
            float(np.clip(relative, 0.0, 1.0)),
            float(max(absolute, 0.0)),
        )
        if level not in seen:
            levels.append(level)
            seen.add(level)
    return levels or [(0.18, 0.015)]


def _alpha_geometry_levels() -> list[tuple[float, int]]:
    """Read scene-agnostic Gaussian footprint candidates for v4 calibration."""
    raw = os.environ.get("QUERY_LIFT_ALPHA_GEOMETRY_LEVELS", "1.0:18")
    levels: list[tuple[float, int]] = []
    seen: set[tuple[float, int]] = set()
    for item in str(raw).split(","):
        pieces = [piece.strip() for piece in item.split(":")]
        if len(pieces) != 2:
            continue
        try:
            sigma_scale = float(pieces[0])
            max_splat_radius = int(float(pieces[1]))
        except ValueError:
            continue
        if not np.isfinite(sigma_scale):
            continue
        level = (
            float(np.clip(sigma_scale, 0.25, 8.0)),
            int(np.clip(max_splat_radius, 1, 64)),
        )
        if level not in seen:
            levels.append(level)
            seen.add(level)
    return levels or [(1.0, 18)]


def _calibrate_alpha_threshold(
    selection_ids: np.ndarray,
    support: LiftingSupport,
    num_gaussians: int,
    device: str,
    scoring_mode: str,
) -> dict[str, Any]:
    """Choose a Gaussian alpha threshold from held multi-frame mask evidence.

    This calibrates how a selected 3D Gaussian entity is *rendered*. It does
    not copy, fuse, or return a Stage-1 2D mask, so the final output remains a
    projection of the same selected Gaussian set.
    """
    max_frames = _env_int("QUERY_LIFT_ALPHA_CALIBRATION_MAX_FRAMES", 8, minimum=1)
    trials: list[dict[str, Any]] = []
    for sigma_scale, max_splat_radius in _alpha_geometry_levels():
        for relative, absolute in _alpha_calibration_levels():
            metrics = _selection_metrics(
                selection_ids,
                support=support,
                num_gaussians=num_gaussians,
                device=device,
                scoring_mode=scoring_mode,
                relative_threshold=relative,
                absolute_threshold=absolute,
                max_frames=max_frames,
                sigma_scale=sigma_scale,
                max_splat_radius=max_splat_radius,
            )
            utility = (
                float(metrics.get("score", float("-inf")))
                + 0.20 * float(metrics.get("rendered_iou_stage1", 0.0))
                + 0.08 * float(metrics.get("recall", 0.0))
                - 0.08 * float(metrics.get("outer_leakage", 1.0))
            )
            metrics["alpha_calibration_utility"] = float(utility)
            metrics["alpha_calibration_method"] = "multiframe_rendered_overlap_geometry"
            metrics["alpha_calibration_frame_count"] = int(min(len(support.samples), max_frames))
            trials.append(
                {
                    "relative_threshold": float(relative),
                    "absolute_threshold": float(absolute),
                    "sigma_scale": float(sigma_scale),
                    "max_splat_radius": int(max_splat_radius),
                    "utility": float(utility),
                    "rendered_iou_stage1": float(metrics.get("rendered_iou_stage1", 0.0)),
                    "precision": float(metrics.get("precision", 0.0)),
                    "recall": float(metrics.get("recall", 0.0)),
                    "area_ratio": float(metrics.get("area_ratio", 0.0)),
                    "outer_leakage": float(metrics.get("outer_leakage", 1.0)),
                    "metrics": metrics,
                }
            )

    best = max(
        trials,
        key=lambda row: (
            float(row["utility"]),
            float(row["rendered_iou_stage1"]),
            float(row["recall"]),
            -float(row["outer_leakage"]),
            -float(row["sigma_scale"]),
            -int(row["max_splat_radius"]),
        ),
    )
    result = dict(best["metrics"])
    result["alpha_calibration_trials"] = [
        {key: value for key, value in row.items() if key != "metrics"}
        for row in trials
    ]
    return result


def _calibrate_v4_candidate_rows(
    scored: list[dict[str, Any]],
    support: LiftingSupport,
    num_gaussians: int,
    device: str,
    target_area_ratio: float,
) -> None:
    """Calibrate a bounded candidate shortlist, then validate it on all frames."""
    if not _env_flag("QUERY_LIFT_ALPHA_THRESHOLD_CALIBRATION", False) or not scored:
        return
    max_candidates = _env_int("QUERY_LIFT_ALPHA_CALIBRATION_MAX_CANDIDATES", 3, minimum=1)
    ranked = sorted(
        scored,
        key=lambda row: _coverage_v4_selection_utility(row, target_area_ratio=target_area_ratio),
        reverse=True,
    )
    selected = list(ranked[:max_candidates])
    largest = max(scored, key=lambda row: int(row.get("gaussian_count", 0)))
    # The largest compatible candidate is a useful recall probe. Reserve a
    # bounded slot for it instead of letting a cluster of tiny, high-precision
    # candidates monopolize threshold calibration.
    if all(row is not largest for row in selected):
        if selected:
            selected[-1] = largest
        else:
            selected.append(largest)

    for row in selected:
        calibrated = _calibrate_alpha_threshold(
            np.asarray(row["ids"], dtype=np.int64),
            support=support,
            num_gaussians=num_gaussians,
            device=device,
            scoring_mode="mask_bootstrap_refine",
        )
        relative = float(calibrated["alpha_relative_threshold"])
        absolute = float(calibrated["alpha_absolute_threshold"])
        sigma_scale = float(calibrated.get("alpha_sigma_scale", 1.0))
        max_splat_radius = int(calibrated.get("alpha_max_splat_radius", 18))
        full_metrics = _selection_metrics(
            np.asarray(row["ids"], dtype=np.int64),
            support=support,
            num_gaussians=num_gaussians,
            device=device,
            scoring_mode="mask_bootstrap_refine",
            relative_threshold=relative,
            absolute_threshold=absolute,
            sigma_scale=sigma_scale,
            max_splat_radius=max_splat_radius,
        )
        full_metrics.update(
            {
                "alpha_calibration_method": calibrated["alpha_calibration_method"],
                "alpha_calibration_frame_count": calibrated["alpha_calibration_frame_count"],
                "alpha_calibration_utility": calibrated["alpha_calibration_utility"],
                "alpha_calibration_trials": calibrated["alpha_calibration_trials"],
                "alpha_sigma_scale": sigma_scale,
                "alpha_max_splat_radius": max_splat_radius,
            }
        )
        row.update(full_metrics)


def _coverage_v4_quality_gate(row: dict[str, Any], target_area_ratio: float, thin_object: bool = False) -> tuple[bool, str]:
    """Reject tiny/wrong lifted entities before they become final proposals."""
    target = max(float(target_area_ratio), 1.0e-6)
    thin_relaxed = bool(thin_object) and _env_flag("QUERY_LIFT_THIN_RELAXED_GATE", True)
    default_min_area = 0.08 if thin_relaxed else min(0.35, max(0.25, 0.45 * target))
    default_min_precision = 0.55 if thin_relaxed else 0.35
    default_max_leakage = 0.18 if thin_relaxed else 0.22
    default_min_recall = 0.08 if thin_relaxed else 0.35
    default_min_iou = 0.07 if thin_relaxed else 0.20
    if thin_relaxed:
        min_area = _env_float("QUERY_LIFT_THIN_MIN_AREA_RATIO", default_min_area, minimum=0.0, maximum=2.0)
        min_precision = _env_float("QUERY_LIFT_THIN_MIN_PRECISION", default_min_precision, minimum=0.0, maximum=1.0)
        max_leakage = _env_float("QUERY_LIFT_THIN_MAX_OUTER_LEAKAGE", default_max_leakage, minimum=0.0, maximum=1.0)
        min_recall = _env_float("QUERY_LIFT_THIN_MIN_RECALL", default_min_recall, minimum=0.0, maximum=1.0)
        min_iou = _env_float("QUERY_LIFT_THIN_MIN_RENDERED_IOU_STAGE1", default_min_iou, minimum=0.0, maximum=1.0)
    else:
        min_area = _env_float("QUERY_LIFT_BOOTSTRAP_MIN_AREA_RATIO", default_min_area, minimum=0.0, maximum=2.0)
        min_precision = _env_float("QUERY_LIFT_BOOTSTRAP_MIN_PRECISION", default_min_precision, minimum=0.0, maximum=1.0)
        max_leakage = _env_float("QUERY_LIFT_BOOTSTRAP_MAX_OUTER_LEAKAGE", default_max_leakage, minimum=0.0, maximum=1.0)
        min_recall = _env_float("QUERY_LIFT_MIN_RECALL", default_min_recall, minimum=0.0, maximum=1.0)
        min_iou = _env_float("QUERY_LIFT_MIN_RENDERED_IOU_STAGE1", default_min_iou, minimum=0.0, maximum=1.0)
    min_active_coverage = _env_float("QUERY_LIFT_MIN_ACTIVE_FRAME_COVERAGE", 0.55, minimum=0.0, maximum=1.0)

    failures: list[str] = []
    if float(row.get("area_ratio", 0.0)) < min_area:
        failures.append(f"area_ratio<{min_area:.3f}")
    if float(row.get("recall", 0.0)) < min_recall:
        failures.append(f"recall<{min_recall:.3f}")
    if float(row.get("rendered_iou_stage1", 0.0)) < min_iou:
        failures.append(f"rendered_iou_stage1<{min_iou:.3f}")
    if float(row.get("precision", 0.0)) < min_precision:
        failures.append(f"precision<{min_precision:.3f}")
    if float(row.get("outer_leakage", 1.0)) > max_leakage:
        failures.append(f"outer_leakage>{max_leakage:.3f}")
    if float(row.get("active_frame_coverage", 0.0)) < min_active_coverage:
        failures.append(f"active_frame_coverage<{min_active_coverage:.3f}")
    return (not failures, "ok" if not failures else ";".join(failures))


def _choose_boundary_refine_candidate(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer non-tiny, low-leakage candidates for boundary-refine lifting.

    Final rendering for this profile is Gaussian-only, so the candidate must
    have enough direct projection coverage by itself. A very high-precision but
    tiny core usually starves vIoU; a very large low-precision shell is not a
    valid entity either.
    """
    if not scored:
        raise ValueError("No boundary-refine candidates to choose from")
    mode = _lifting_mode()
    if mode in {"mask_shape_refine_v2", "shape_refine_v2"}:
        # Shape v2 selection: prefer 0.45-1.25 area ratio, precision >= 0.55, leakage <= 0.16
        shape_floor = _env_float("QUERY_LIFT_SHAPE_MIN_AREA_RATIO", 0.45, minimum=0.0, maximum=2.0)
        shape_ceiling = _env_float("QUERY_LIFT_SHAPE_MAX_AREA_RATIO", 1.25, minimum=0.05, maximum=4.0)
        shape_min_precision = _env_float("QUERY_LIFT_SHAPE_MIN_PRECISION", 0.55, minimum=0.0, maximum=1.0)
        shape_max_leakage = _env_float("QUERY_LIFT_SHAPE_MAX_OUTER_LEAKAGE", 0.16, minimum=0.0, maximum=1.0)
        feasible = [
            row for row in scored
            if shape_floor <= float(row.get("area_ratio", 0.0)) <= shape_ceiling
            and float(row.get("precision", 0.0)) >= shape_min_precision
            and float(row.get("outer_leakage", 1.0)) <= shape_max_leakage
        ]
        if feasible:
            feasible.sort(key=lambda row: float(row["score"]), reverse=True)
            chosen = feasible[0]
            chosen["selection_strategy"] = "shape_v2_strict"
            chosen["selection_utility"] = float(chosen.get("score", 0.0))
            return chosen
        # Relaxed fallback: wider area range, but still prefer larger area candidates
        relaxed = [
            row for row in scored
            if 0.25 <= float(row.get("area_ratio", 0.0)) <= 1.55
            and float(row.get("outer_leakage", 1.0)) <= 0.28
        ]
        if relaxed:
            relaxed.sort(key=lambda row: float(row["score"]), reverse=True)
            chosen = relaxed[0]
            chosen["selection_strategy"] = "shape_v2_relaxed_fallback"
            chosen["selection_utility"] = float(chosen.get("score", 0.0))
            return chosen
        # Thin-object / sparse-Gaussian fallback: prefer refined candidates with
        # higher area_ratio over raw-score-only selection. This is critical for
        # transparent / thin-shell objects (glass cups, bottles) where Gaussian
        # coverage is inherently sparse and area_ratio will never reach 0.45.
        utility_candidates = []
        for row in scored:
            area = float(row.get("area_ratio", 0.0))
            prec = float(row.get("precision", 0.0))
            recall = float(row.get("recall", 0.0))
            leak = float(row.get("outer_leakage", 1.0))
            score = float(row.get("score", 0.0))
            gauss_count = float(row.get("gaussian_count", 0.0))
            has_refine = "shape_refine" in str(row.get("name", "")) or "boundary_refine" in str(row.get("name", ""))
            has_refine = has_refine or "bootstrap_refine" in str(row.get("name", ""))
            # Prefer: larger direct projection coverage, recall, precision, and
            # refined candidates. Do not saturate area once it reaches 0.20:
            # sparse high-precision cores are the main low-vIoU failure mode.
            target_area = _env_float("QUERY_LIFT_SHAPE_TARGET_AREA_RATIO", 0.70, minimum=0.05, maximum=2.0)
            area_term = min(area, target_area) / max(target_area, 1.0e-6)
            severe_underfill = max(0.0, 0.25 - area) / 0.25
            mild_underfill = max(0.0, 0.50 - area) / 0.50
            leakage_penalty = 0.0 if leak <= 0.10 else (0.60 * (leak - 0.10))
            refine_bonus = 0.06 if has_refine else 0.0
            utility = (
                0.30 * score
                + 0.30 * area_term
                + 0.18 * recall
                + 0.16 * prec
                + 0.10 * np.log1p(max(gauss_count, 1.0)) / np.log1p(3072.0)
                + refine_bonus
                - leakage_penalty
                - 0.20 * severe_underfill
                - 0.08 * mild_underfill
            )
            utility_candidates.append((utility, row))
        if utility_candidates:
            utility_candidates.sort(key=lambda item: item[0], reverse=True)
            chosen = utility_candidates[0][1]
            chosen["selection_strategy"] = "shape_v2_utility_fallback"
            if float(chosen.get("area_ratio", 0.0)) < 0.25:
                chosen["selection_strategy"] = "shape_v2_sparse_fallback"
            chosen["selection_utility"] = float(utility_candidates[0][0])
            return chosen
        # Absolute final fallback (should never reach here)
        chosen = scored[0]
        chosen["selection_strategy"] = "shape_v2_score_fallback"
        chosen["selection_utility"] = float(chosen.get("score", 0.0))
        return chosen
    score_best = scored[0]
    score_best["selection_strategy"] = "score_best"
    floor = _env_float("QUERY_LIFT_BOUNDARY_SELECT_AREA_FLOOR", 0.05, minimum=0.0, maximum=2.0)
    target = _env_float("QUERY_LIFT_BOUNDARY_SELECT_TARGET_AREA_RATIO", 0.08, minimum=0.01, maximum=2.0)
    ceiling = _env_float("QUERY_LIFT_BOUNDARY_SELECT_AREA_CEILING", 1.25, minimum=0.05, maximum=4.0)
    max_leakage = _env_float("QUERY_LIFT_BOUNDARY_SELECT_MAX_OUTER_LEAKAGE", 0.08, minimum=0.0, maximum=1.0)
    min_precision = _env_float("QUERY_LIFT_BOUNDARY_SELECT_MIN_PRECISION", 0.80, minimum=0.0, maximum=1.0)
    min_active_coverage = _env_float("QUERY_LIFT_BOUNDARY_SELECT_MIN_ACTIVE_COVERAGE", 0.78, minimum=0.0, maximum=1.0)
    feasible = [
        row
        for row in scored
        if floor <= float(row.get("area_ratio", 0.0)) <= ceiling
        and float(row.get("outer_leakage", 1.0)) <= max_leakage
        and float(row.get("precision", 0.0)) >= min_precision
        and float(row.get("active_frame_coverage", 0.0)) >= min_active_coverage
    ]
    strategy = "coverage_feasible"
    if not feasible:
        relaxed_floor = max(0.08, floor * 0.65)
        feasible = [
            row
            for row in scored
            if relaxed_floor <= float(row.get("area_ratio", 0.0)) <= ceiling
            and float(row.get("outer_leakage", 1.0)) <= min(0.48, max_leakage + 0.12)
            and float(row.get("precision", 0.0)) >= max(0.06, min_precision * 0.65)
            and float(row.get("active_frame_coverage", 0.0)) >= max(0.50, min_active_coverage * 0.75)
        ]
        strategy = "coverage_relaxed"
    if not feasible:
        return score_best
    feasible.sort(key=lambda row: _boundary_refine_selection_utility(row, target_area_ratio=target), reverse=True)
    chosen = feasible[0]
    chosen["selection_strategy"] = strategy
    chosen["selection_utility"] = _boundary_refine_selection_utility(chosen, target_area_ratio=target)
    chosen["selection_area_floor"] = float(floor)
    chosen["selection_area_target"] = float(target)
    return chosen


def _boundary_refine_seed_variants(variants: list[tuple[str, np.ndarray]], candidate_limit: int) -> list[tuple[str, np.ndarray]]:
    limit = int(max(1, candidate_limit))
    selected: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()

    def add_matches(predicate, max_add: int | None = None) -> None:
        added = 0
        for name, ids in variants:
            if len(selected) >= limit:
                return
            if str(name) in seen or not predicate(str(name), np.asarray(ids)):
                continue
            selected.append((name, ids))
            seen.add(str(name))
            added += 1
            if max_add is not None and added >= int(max_add):
                return

    add_matches(lambda _name, _ids: True, max_add=min(4, limit))
    add_matches(lambda name, _ids: name.startswith("thin_shell_top_") and any(tag in name for tag in ("1536", "2048", "3072")), max_add=2)
    add_matches(lambda name, _ids: name in {"component_union_top4", "component_union_top8", "component_union_until_area_35"}, max_add=3)
    add_matches(lambda name, _ids: name.startswith("soft_recall_pool_top_") and any(tag in name for tag in ("1024", "1536", "2048", "3072")), max_add=2)
    add_matches(lambda _name, _ids: True)
    return selected[:limit]


def _candidate_variants(support: LiftingSupport, bank: dict[str, np.ndarray], min_gaussians: int, max_gaussians: int, graph_knn: int, radius_scale: float) -> list[tuple[str, np.ndarray]]:
    global _BOOTSTRAP_BANK_CACHE
    mode = _lifting_mode()
    if mode == "v1_compat":
        return _candidate_variants_v1_compat(support, bank=bank, min_gaussians=min_gaussians, max_gaussians=max_gaussians, graph_knn=graph_knn, radius_scale=radius_scale)
    if mode in {"hybrid_recall", "mask_bootstrap_refine", "bootstrap_refine", "mask_boundary_refine", "boundary_refine", "mask_shape_refine_v2", "shape_refine_v2"}:
        if mode in {"mask_shape_refine_v2", "shape_refine_v2"}:
            variants = _candidate_variants_shape_v2(support, bank=bank, min_gaussians=min_gaussians, max_gaussians=max_gaussians, graph_knn=graph_knn, radius_scale=radius_scale)
        else:
            variants = _candidate_variants_hybrid_recall(support, bank=bank, min_gaussians=min_gaussians, max_gaussians=max_gaussians, graph_knn=graph_knn, radius_scale=radius_scale)
        if mode in {"mask_bootstrap_refine", "bootstrap_refine"}:
            if _env_flag("QUERY_LIFT_SKIP_BOOTSTRAP_REFINE", False):
                max_candidates = _env_optional_int("QUERY_LIFT_MAX_CANDIDATES_PER_VARIANT", None, minimum=1)
                variants = _dedupe_variants(variants)
                if max_candidates is not None:
                    variants = variants[: int(max_candidates)]
                print(
                    f"[mask_supported_lifting] skip bootstrap refine variant='{support.alias}' "
                    f"candidates={len(variants)}",
                    flush=True,
                )
                return variants
            _BOOTSTRAP_BANK_CACHE = bank
            num_gaussians = int(np.asarray(bank["trajectories"]).shape[0])
            refined: list[tuple[str, np.ndarray]] = []
            candidate_limit = _env_int("QUERY_LIFT_BOOTSTRAP_SEEDS", 8, minimum=1)
            seed_variants = _boundary_refine_seed_variants(variants, candidate_limit)
            candidate_timeout = _env_float("QUERY_LIFT_CANDIDATE_TIMEOUT", 0.0, minimum=0.0)
            candidate_t0 = time.time()
            for idx, (name, ids) in enumerate(seed_variants):
                if candidate_timeout > 0 and (time.time() - candidate_t0) > candidate_timeout:
                    print(
                        f"[mask_supported_lifting] bootstrap candidate timeout "
                        f"variant='{support.alias}' elapsed={time.time() - candidate_t0:.1f}s",
                        flush=True,
                    )
                    break
                refined_name = f"{name}_bootstrap_refined"
                print(
                    f"[mask_supported_lifting] bootstrap seed {idx + 1}/{len(seed_variants)} "
                    f"variant='{support.alias}' seed='{name}' ids={np.asarray(ids).size}",
                    flush=True,
                )
                refine_t0 = time.time()
                refined_ids, _info = _bootstrap_refine_selection(
                    ids,
                    support=support,
                    num_gaussians=num_gaussians,
                    device=str("cuda" if torch.cuda.is_available() else "cpu"),
                    bank=bank,
                    max_gaussians=max_gaussians,
                )
                print(
                    f"[mask_supported_lifting] bootstrap done {idx + 1}/{candidate_limit} "
                    f"variant='{support.alias}' refined_ids={np.asarray(refined_ids).size} "
                    f"best_iou={float(_info.get('bootstrap_best_iou', 0.0)):.4f} "
                    f"elapsed={time.time() - refine_t0:.1f}s",
                    flush=True,
                )
                _BOUNDARY_REFINE_INFO_CACHE[_refine_cache_key(support.alias, refined_name)] = _info
                refined.append((refined_name, refined_ids))
            variants = refined + variants
        if mode in {"mask_boundary_refine", "boundary_refine"}:
            num_gaussians = int(np.asarray(bank["trajectories"]).shape[0])
            refined = []
            candidate_limit = _env_int("QUERY_LIFT_BOUNDARY_SEEDS", 6, minimum=1)
            seed_variants = _boundary_refine_seed_variants(variants, candidate_limit)
            for idx, (name, ids) in enumerate(seed_variants):
                refined_name = f"{name}_boundary_refined"
                refined_ids, info = _boundary_refine_selection(
                    ids,
                    support=support,
                    num_gaussians=num_gaussians,
                    device=str("cuda" if torch.cuda.is_available() else "cpu"),
                    seed_index=idx,
                    min_gaussians=min_gaussians,
                    max_gaussians=max_gaussians,
                )
                _BOUNDARY_REFINE_INFO_CACHE[_refine_cache_key(support.alias, refined_name)] = info
                refined.append((refined_name, refined_ids))
            variants = refined + variants
        if mode in {"mask_shape_refine_v2", "shape_refine_v2"}:
            num_gaussians = int(np.asarray(bank["trajectories"]).shape[0])
            refined = []
            candidate_limit = _env_int("QUERY_LIFT_SHAPE_SEEDS", 12, minimum=1)
            seed_variants = _boundary_refine_seed_variants(variants, candidate_limit)
            for idx, (name, ids) in enumerate(seed_variants):
                refined_name = f"{name}_shape_refined_v2"
                refined_ids, info = _shape_refine_selection(
                    ids,
                    support=support,
                    num_gaussians=num_gaussians,
                    device=str("cuda" if torch.cuda.is_available() else "cpu"),
                    seed_index=idx,
                    min_gaussians=min_gaussians,
                    max_gaussians=max_gaussians,
                )
                _BOUNDARY_REFINE_INFO_CACHE[_refine_cache_key(support.alias, refined_name)] = info
                refined.append((refined_name, refined_ids))
            variants = refined + variants
        return _dedupe_variants(variants)
    return _candidate_variants_v2(support, bank=bank, min_gaussians=min_gaussians, max_gaussians=max_gaussians, graph_knn=graph_knn, radius_scale=radius_scale)


def build_mask_supported_lifting_proposal_dir(
    run_dir: str | Path,
    dataset_dir: str | Path,
    tracks_path: str | Path,
    output_dir: str | Path,
    max_track_frames: int = 16,
    min_gaussians: int = 192,
    max_gaussians: int = 1024,
    max_gaussians_per_frame: int = 18000,
    gate_threshold: float = 0.01,
    graph_knn: int = 24,
    graph_radius_scale: float = 1.35,
    device: str | None = None,
) -> Path:
    run_dir = Path(run_dir)
    dataset_dir = Path(dataset_dir)
    tracks_path = Path(tracks_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    state, _config, _iteration = load_gaussian_state(run_dir)
    bank = _load_bank(run_dir / "entitybank")
    time_values = np.asarray(bank["time_values"], dtype=np.float32).reshape(-1)
    track_payload = _read_json(tracks_path)
    query_plan_path = tracks_path.parent.parent / "query_plan.json"
    query_plan = _read_json(query_plan_path) if query_plan_path.exists() else None
    tracks = [track for track in track_payload.get("tracks", []) if str(track.get("status", "")) == "seeded"]
    if not tracks:
        raise ValueError(f"No seeded phrase tracks found in {tracks_path}")

    phrase_rows: list[dict[str, Any]] = []
    entities_json_rows: list[dict[str, Any]] = []
    centroid_world = []
    centroid_valid = []
    bbox_world = []
    bbox_valid = []
    visibility = []
    mask_area = []
    quality = []
    diagnostics_rows: list[dict[str, Any]] = []
    build_t0 = time.time()
    total_timeout = _env_float("QUERY_LIFT_TOTAL_TIMEOUT", 0.0, minimum=0.0)
    variant_timeout = _env_float("QUERY_LIFT_VARIANT_TIMEOUT", 0.0, minimum=0.0)

    for track in tracks:
        if total_timeout > 0 and (time.time() - build_t0) > total_timeout:
            diagnostics_rows.append(
                {
                    "entity_id": None,
                    "status": "lifting_total_timeout",
                    "elapsed_seconds": float(time.time() - build_t0),
                    "timeout_seconds": float(total_timeout),
                }
            )
            print(
                f"[mask_supported_lifting] total timeout after {time.time() - build_t0:.1f}s",
                flush=True,
            )
            break
        base_phrase = str(track["phrase"])
        variants = _phase_aware_track_variants(base_phrase, track, max_track_frames=max_track_frames, query_plan=query_plan)
        original_variant_count = len(variants)
        variants = _filter_variants_for_query_phase(variants, query_plan)
        # Hard limit: prevent pathological post_split_union from stalling
        max_phase = _env_optional_int("QUERY_LIFT_MAX_PHASE_VARIANTS", 1, minimum=1)
        if max_phase is not None and len(variants) > int(max_phase):
            variants = variants[: int(max_phase)]
        print(
            f"[mask_supported_lifting] track='{base_phrase}' variants={len(variants)}/{original_variant_count} "
            f"phase_filter={_query_phase_mode_from_plan(query_plan)} mode={_lifting_mode()}",
            flush=True,
        )
        for variant in variants:
            if total_timeout > 0 and (time.time() - build_t0) > total_timeout:
                diagnostics_rows.append(
                    {
                        "entity_id": None,
                        "variant_alias": str(variant.get("alias", "")),
                        "status": "lifting_total_timeout",
                        "elapsed_seconds": float(time.time() - build_t0),
                        "timeout_seconds": float(total_timeout),
                    }
                )
                break
            variant_t0 = time.time()
            support = _collect_support(
                variant=variant,
                dataset_dir=dataset_dir,
                state=state,
                bank=bank,
                max_gaussians_per_frame=max_gaussians_per_frame,
                gate_threshold=gate_threshold,
                device=device,
            )
            if not support.samples:
                print(f"[mask_supported_lifting] skip variant='{variant.get('alias')}' no support samples", flush=True)
                continue
            if variant_timeout > 0 and (time.time() - variant_t0) > variant_timeout:
                diagnostics_rows.append(
                    {
                        "entity_id": None,
                        "variant_alias": str(variant.get("alias", "")),
                        "status": "variant_timeout_after_support",
                        "elapsed_seconds": float(time.time() - variant_t0),
                        "timeout_seconds": float(variant_timeout),
                    }
                )
                print(
                    f"[mask_supported_lifting] skip variant='{variant.get('alias')}' "
                    f"timeout after support elapsed={time.time() - variant_t0:.1f}s",
                    flush=True,
                )
                continue
            print(
                f"[mask_supported_lifting] support variant='{support.alias}' frames={len(support.samples)} "
                f"mean_mask_area={support.mean_mask_area:.1f}",
                flush=True,
            )
            candidates = _candidate_variants(support, bank=bank, min_gaussians=min_gaussians, max_gaussians=max_gaussians, graph_knn=graph_knn, radius_scale=graph_radius_scale)
            print(f"[mask_supported_lifting] candidates variant='{support.alias}' count={len(candidates)}", flush=True)
            if variant_timeout > 0 and (time.time() - variant_t0) > variant_timeout:
                diagnostics_rows.append(
                    {
                        "entity_id": None,
                        "variant_alias": support.alias,
                        "status": "variant_timeout_after_candidates",
                        "elapsed_seconds": float(time.time() - variant_t0),
                        "timeout_seconds": float(variant_timeout),
                    }
                )
                print(
                    f"[mask_supported_lifting] skip variant='{support.alias}' "
                    f"timeout after candidates elapsed={time.time() - variant_t0:.1f}s",
                    flush=True,
                )
                continue
            scored: list[dict[str, Any]] = []
            num_gaussians = int(np.asarray(bank["trajectories"]).shape[0])
            for candidate_name, ids in candidates:
                refine_info = _BOUNDARY_REFINE_INFO_CACHE.get(_refine_cache_key(support.alias, candidate_name))
                cached_metric_keys = {
                    "score",
                    "rendered_iou_stage1",
                    "precision",
                    "recall",
                    "area_ratio",
                    "outer_leakage",
                    "active_frame_coverage",
                    "area_cv",
                }
                if refine_info and cached_metric_keys.issubset(set(refine_info.keys())):
                    metrics = {key: refine_info[key] for key in cached_metric_keys}
                    metrics["mean_pred_area"] = float(refine_info.get("mean_pred_area", 0.0))
                else:
                    metrics = _selection_metrics(ids, support=support, num_gaussians=num_gaussians, device=device, scoring_mode=_lifting_mode())
                metrics["gaussian_count"] = int(np.asarray(ids, dtype=np.int64).size)
                if refine_info:
                    metrics.update(refine_info)
                scored.append({"name": candidate_name, "ids": ids, **metrics})
            if not scored:
                continue
            scored.sort(key=lambda row: float(row["score"]), reverse=True)
            mode = _lifting_mode()
            if _env_flag("QUERY_LIFT_DISABLE_RENDER_VALIDATION", False):
                best = scored[0]
                best["selection_strategy"] = "raw_score_no_render_validation"
                best["selection_utility"] = float(best.get("score", 0.0))
            elif mode in {"mask_boundary_refine", "boundary_refine", "mask_shape_refine_v2", "shape_refine_v2"}:
                best = _choose_boundary_refine_candidate(scored)
            elif mode in {"mask_bootstrap_refine", "bootstrap_refine"}:
                target = _env_float("QUERY_LIFT_BOOTSTRAP_TARGET_AREA_RATIO", 0.70, minimum=0.05, maximum=2.0)
                _calibrate_v4_candidate_rows(
                    scored,
                    support=support,
                    num_gaussians=num_gaussians,
                    device=device,
                    target_area_ratio=target,
                )
                feasible = []
                thin_object = _support_is_geometrically_thin(support)
                for row in scored:
                    passed, reason = _coverage_v4_quality_gate(row, target_area_ratio=target, thin_object=thin_object)
                    row["quality_gate_pass"] = bool(passed)
                    row["quality_gate_reason"] = reason
                    if passed:
                        feasible.append(row)
                if feasible:
                    feasible.sort(key=lambda row: _coverage_v4_selection_utility(row, target_area_ratio=target), reverse=True)
                    best = feasible[0]
                    best["selection_strategy"] = "coverage_v4_feasible"
                    best["selection_utility"] = _coverage_v4_selection_utility(best, target_area_ratio=target)
                else:
                    scored.sort(key=lambda row: _coverage_v4_selection_utility(row, target_area_ratio=target), reverse=True)
                    rejected = scored[0]
                    rejected["selection_utility"] = _coverage_v4_selection_utility(rejected, target_area_ratio=target)
                    diagnostics_rows.append(
                        {
                            "entity_id": None,
                            "variant_alias": support.alias,
                            "status": "coverage_v4_quality_gate_failed",
                            "best_rejected_candidate": {k: v for k, v in rejected.items() if k != "ids"},
                            "candidate_variants": [{k: v for k, v in row.items() if k != "ids"} for row in scored[:8]],
                        }
                    )
                    if not _env_flag("QUERY_LIFT_ALLOW_SPARSE_FALLBACK", False):
                        print(
                            f"[mask_supported_lifting] reject variant='{support.alias}' "
                            f"coverage_v4 gate failed best='{rejected.get('name')}' "
                            f"reason={rejected.get('quality_gate_reason')} "
                            f"iou={float(rejected.get('rendered_iou_stage1', 0.0)):.4f} "
                            f"recall={float(rejected.get('recall', 0.0)):.4f} "
                            f"area_ratio={float(rejected.get('area_ratio', 0.0)):.4f}",
                            flush=True,
                        )
                        continue
                    best = rejected
                    best["selection_strategy"] = (
                        "coverage_v4_sparse_fallback"
                        if float(best.get("area_ratio", 0.0)) < 0.25
                        else "coverage_v4_relaxed_fallback"
                    )
            elif mode == "hybrid_recall":
                feasible = [
                    row for row in scored
                    if 0.35 <= float(row.get("area_ratio", 0.0)) <= 1.50
                    and float(row.get("outer_leakage", 1.0)) < 0.25
                    and float(row.get("precision", 0.0)) > 0.25
                ]
                if _support_is_geometrically_thin(support):
                    thin_feasible = [
                        row for row in scored
                        if float(row.get("area_ratio", 0.0)) >= 0.30
                        and float(row.get("outer_leakage", 1.0)) < 0.35
                        and float(row.get("precision", 0.0)) > 0.20
                    ]
                    if thin_feasible:
                        feasible = thin_feasible
                if feasible:
                    feasible.sort(key=lambda row: float(row["score"]), reverse=True)
                    best = feasible[0]
                else:
                    best = scored[0]
            else:
                best = scored[0]
            selected_ids = np.asarray(best["ids"], dtype=np.int64)
            print(
                f"[mask_supported_lifting] best variant='{support.alias}' strategy={best.get('selection_strategy')} "
                f"gaussians={selected_ids.size} iou={float(best.get('rendered_iou_stage1', 0.0)):.4f} "
                f"precision={float(best.get('precision', 0.0)):.4f} recall={float(best.get('recall', 0.0)):.4f} "
                f"area_ratio={float(best.get('area_ratio', 0.0)):.4f} elapsed={time.time() - variant_t0:.1f}s",
                flush=True,
            )
            base_scores = support.support_score[selected_ids].astype(np.float32)
            selected_scores = base_scores / max(float(base_scores.max()), 1.0e-6)
            phrase_payload = {
                "phrase": support.alias,
                "entity_type": support.entity_type,
                "selected_gaussian_ids": selected_ids,
                "selected_scores": selected_scores,
                "sampled_frames": support.sampled_frames,
                "sampled_indices": support.sampled_indices,
                "mean_mask_area": support.mean_mask_area,
                "mean_hit_ratio": float(np.mean(support.hit_count[selected_ids] / max(len(support.samples), 1))) if selected_ids.size else 0.0,
                "keyframes": sorted(set(int(index) for index in support.sampled_indices.tolist()))[:8],
            }
            world = _phrase_world_payload(phrase_payload, bank=bank)
            entity_id = len(entities_json_rows)
            centroid_world.append(world["center_world"])
            centroid_valid.append(world["center_valid"])
            bbox_world.append(world["bbox_world"])
            bbox_valid.append(world["bbox_valid"])
            visibility.append(world["visibility"])
            mask_area.append(world["mask_area"])
            quality.append(float(best["score"]))
            support_stats = {
                "mean_support_score": float(np.mean(support.support_score[selected_ids])) if selected_ids.size else 0.0,
                "mean_positive_score": float(np.mean(support.positive_score[selected_ids])) if selected_ids.size else 0.0,
                "mean_negative_score": float(np.mean(support.negative_score[selected_ids])) if selected_ids.size else 0.0,
                "mean_purity": float(np.mean(support.purity[selected_ids])) if selected_ids.size else 0.0,
                "mean_core_ratio": float(np.mean(support.core_ratio[selected_ids])) if selected_ids.size else 0.0,
                "mean_outer_ratio": float(np.mean(support.outer_ratio[selected_ids])) if selected_ids.size else 0.0,
                "visible_frame_mean": float(np.mean(support.visible_count[selected_ids])) if selected_ids.size else 0.0,
            "mean_presence": float(np.mean(support.presence[selected_ids])) if selected_ids.size else 0.0,
            "mean_core_presence": float(np.mean(support.core_presence[selected_ids])) if selected_ids.size else 0.0,
            "shared_gaussian_entity": True,
            }
            rendered_diagnostics = {
                key: best[key]
                for key in (
                    "rendered_iou_stage1",
                    "precision",
                    "recall",
                    "area_ratio",
                    "outer_leakage",
                    "active_frame_coverage",
                    "area_cv",
                    "mean_pred_area",
                    "score",
                    "boundary_refine_iters",
                    "boundary_refine_best_iou",
                    "boundary_refine_best_precision",
                    "boundary_refine_best_recall",
                    "boundary_refine_best_area_ratio",
                    "boundary_refine_best_outer_leakage",
                    "boundary_refine_target_area_ratio",
                    "boundary_refine_target_iou",
                    "boundary_refine_stop_reason",
                    "boundary_refine_direct_gaussian_iou_stage1",
                    "shape_refine_iters",
                    "shape_refine_best_iou",
                    "shape_refine_best_area_ratio",
                    "shape_refine_best_outer_leakage",
                    "shape_refine_history",
                    "bootstrap_iters",
                    "bootstrap_best_iou",
                    "bootstrap_best_precision",
                    "bootstrap_best_recall",
                    "bootstrap_best_area_ratio",
                    "bootstrap_target_area_ratio",
                    "bootstrap_used_feasible_area_candidate",
                    "bootstrap_history",
                    "selection_strategy",
                    "selection_utility",
                    "selection_area_floor",
                    "selection_area_target",
                    "quality_gate_pass",
                    "quality_gate_reason",
                    "alpha_relative_threshold",
                    "alpha_absolute_threshold",
                    "alpha_sigma_scale",
                    "alpha_max_splat_radius",
                    "alpha_calibration_method",
                    "alpha_calibration_frame_count",
                    "alpha_calibration_utility",
                    "alpha_calibration_trials",
                )
                if key in best
            }
            if "boundary_refine_history" in best:
                rendered_diagnostics["boundary_refine_history"] = best["boundary_refine_history"]
            active_segments = world["segments"]
            phrase_row = {
                "id": int(entity_id),
                "phrase": support.base_phrase,
                "proposal_alias": support.alias,
                "stage1_object_id": support.source_object_id,
                "stage1_track_id": support.source_track_id,
                "stage1_instance_group_id": support.source_instance_group_id,
                "stage1_instance_index": support.source_instance_index,
                "phase": support.phase,
                "variant_kind": support.variant_kind,
                "entity_type": support.entity_type,
                "selected_gaussian_count": int(selected_ids.size),
                "shared_gaussian_entity": True,
                "quality": float(best["score"]),
                "visibility_ratio": float(world["visibility_ratio"]),
                "cluster_mode": f"mask_supported_lifting:{_lifting_mode()}",
                "lifting_mode": _lifting_mode(),
                "best_variant": str(best["name"]),
                "candidate_count": int(len(scored)),
                "selection_strategy": str(best.get("selection_strategy", "score_best")),
                "keyframes": world["keyframes"],
                "segments": active_segments,
                **support_stats,
                **rendered_diagnostics,
            }
            phrase_rows.append(phrase_row)
            diagnostics_rows.append({"entity_id": int(entity_id), "candidate_variants": [{k: v for k, v in row.items() if k != "ids"} for row in scored]})
            entities_json_rows.append(
                {
                    "id": int(entity_id),
                    "static_text": support.base_phrase,
                    "proposal_alias": support.alias,
                    "stage1_object_id": support.source_object_id,
                    "stage1_track_id": support.source_track_id,
                    "stage1_instance_group_id": support.source_instance_group_id,
                    "stage1_instance_index": support.source_instance_index,
                    "proposal_phase": support.phase,
                    "proposal_variant": support.variant_kind,
                    "global_desc": support.description,
                    "dyn_desc": [support.description],
                    "gaussian_ids": selected_ids.astype(int).tolist(),
                    "gaussian_scores": selected_scores.astype(float).tolist(),
                    "shared_gaussian_entity": True,
                    "visibility_ratio": float(world["visibility_ratio"]),
                    "mean_mask_area": float(support.mean_mask_area),
                    "quality": float(best["score"]),
                    "entity_type": support.entity_type,
                    "role_hints": ["training_free", "mask_supported_lifting"],
                    "keyframes": world["keyframes"],
                    "segments": active_segments,
                    "active_segments": active_segments,
                    "support_stats": support_stats,
                    "rendered_diagnostics": rendered_diagnostics,
                    "best_variant": str(best["name"]),
                }
            )

    if not entities_json_rows:
        _write_json(
            output_dir / "query_proposal_summary.json",
            {
                "schema_version": 1,
                "shared_gaussian_entity": True,
                "run_dir": str(run_dir),
                "dataset_dir": str(dataset_dir),
                "tracks_path": str(tracks_path),
                "num_entities": 0,
                "phrases": [],
                "diagnostics": diagnostics_rows,
                "params": {
                    "cluster_mode": f"mask_supported_lifting:{_lifting_mode()}",
                    "lifting_mode": _lifting_mode(),
                    "max_track_frames": int(max_track_frames),
                    "min_gaussians": int(min_gaussians),
                    "max_gaussians": int(max_gaussians),
                    "max_gaussians_per_frame": int(max_gaussians_per_frame),
                    "gate_threshold": float(gate_threshold),
                    "graph_knn": int(graph_knn),
                    "graph_radius_scale": float(graph_radius_scale),
                    "training_free": True,
                    "empty_reason": "no_candidate_passed_quality_gate_or_no_support",
                },
            },
        )
        raise ValueError("mask_supported_lifting could not export any proposal entities")

    torch.save(
        {
            "time_values": torch.from_numpy(time_values.astype(np.float32)),
            "centroid_world": torch.from_numpy(np.stack(centroid_world, axis=0).astype(np.float32)),
            "centroid_world_valid": torch.from_numpy(np.stack(centroid_valid, axis=0).astype(bool)),
            "bbox_world": torch.from_numpy(np.stack(bbox_world, axis=0).astype(np.float32)),
            "bbox_world_valid": torch.from_numpy(np.stack(bbox_valid, axis=0).astype(bool)),
            "visibility": torch.from_numpy(np.stack(visibility, axis=0).astype(bool)),
            "mask_area": torch.from_numpy(np.stack(mask_area, axis=0).astype(np.float32)),
            "quality": torch.from_numpy(np.asarray(quality, dtype=np.float32)),
        },
        output_dir / "entities.pt",
    )
    _write_json(
        output_dir / "entities.json",
        {
            "schema_version": 1,
            "source_tracks_path": str(tracks_path),
            "run_dir": str(run_dir),
            "dataset_dir": str(dataset_dir),
            "num_entities": int(len(entities_json_rows)),
            "frame_count": int(time_values.shape[0]),
            "time_values": time_values.astype(float).tolist(),
            "entities": entities_json_rows,
        },
    )
    _write_json(
        output_dir / "query_proposal_summary.json",
        {
            "schema_version": 1,
            "shared_gaussian_entity": True,
            "run_dir": str(run_dir),
            "dataset_dir": str(dataset_dir),
            "tracks_path": str(tracks_path),
            "num_entities": int(len(phrase_rows)),
            "phrases": phrase_rows,
            "diagnostics": diagnostics_rows,
            "params": {
                "cluster_mode": f"mask_supported_lifting:{_lifting_mode()}",
                "lifting_mode": _lifting_mode(),
                "final_profile_expected": (
                    str(os.environ.get("QUERY_EVAL_PROFILE", "")).strip().lower()
                    if str(os.environ.get("QUERY_EVAL_PROFILE", "")).strip().lower()
                    in {
                        "public_time_boundary_gated_v5",
                        "r4d_boundary_gated_v5",
                        "r4d_multi_instance_boundary_v6",
                    }
                    else (
                        "public_time_shape_v4_recall"
                        if _lifting_mode() in {"mask_bootstrap_refine", "bootstrap_refine", "mask_coverage_refine_v4"}
                        else ("boundary_shape_v2" if _lifting_mode() in {"mask_shape_refine_v2", "shape_refine_v2"} else "boundary_refine_v1")
                    )
                ),
                "max_track_frames": int(max_track_frames),
                "min_gaussians": int(min_gaussians),
                "max_gaussians": int(max_gaussians),
                "max_gaussians_per_frame": int(max_gaussians_per_frame),
                "gate_threshold": float(gate_threshold),
                "graph_knn": int(graph_knn),
                "graph_radius_scale": float(graph_radius_scale),
                "training_free": True,
            },
        },
    )
    return output_dir
