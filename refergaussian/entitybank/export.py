from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.cluster.vq import kmeans2
import torch

from .tube_bank import GaussianState, load_gaussian_state, sample_tube_bank, save_tube_bank


def _resample_indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    source = np.asarray(source_times, dtype=np.float32).reshape(-1)
    target = np.asarray(target_times, dtype=np.float32).reshape(-1)
    return np.abs(target[:, None] - source[None, :]).argmin(axis=1).astype(np.int32)


def _proposal_priority(entity: dict[str, Any]) -> float:
    entity_type = str(entity.get("entity_type", "object"))
    type_bonus = {
        "tool": 1.15,
        "object": 1.00,
        "support_surface": 0.82,
        "background_stuff": 0.50,
    }.get(entity_type, 0.90)
    return float(entity.get("quality", 0.0)) * type_bonus * (0.6 + 0.4 * float(entity.get("visibility_ratio", 0.0)))


def _load_proposals(
    proposal_dir: Path,
    target_time_values: np.ndarray,
    max_entities: int,
) -> list[dict[str, Any]]:
    entities_json = proposal_dir / "entities.json"
    entities_pt = proposal_dir / "entities.pt"
    with open(entities_json, "r", encoding="utf-8") as handle:
        entities_payload = json.load(handle)
    proposal_payload = torch.load(entities_pt, map_location="cpu")

    entities = sorted(
        entities_payload.get("entities", []),
        key=_proposal_priority,
        reverse=True,
    )
    if max_entities:
        entities = entities[: max_entities]

    source_times = proposal_payload["time_values"].detach().cpu().numpy().astype(np.float32)
    frame_indices = _resample_indices(source_times, target_time_values)
    centroid_world = proposal_payload["centroid_world"].detach().cpu().numpy().astype(np.float32)
    centroid_world_valid = proposal_payload["centroid_world_valid"].detach().cpu().numpy().astype(bool)
    bbox_world = proposal_payload["bbox_world"].detach().cpu().numpy().astype(np.float32)
    bbox_world_valid = proposal_payload["bbox_world_valid"].detach().cpu().numpy().astype(bool)
    visibility = proposal_payload["visibility"].detach().cpu().numpy().astype(bool)
    mask_area = proposal_payload["mask_area"].detach().cpu().numpy().astype(np.float32)

    loaded: list[dict[str, Any]] = []
    for entity in entities:
        entity_id = int(entity["id"])
        center = centroid_world[entity_id][frame_indices]
        center_valid = centroid_world_valid[entity_id][frame_indices]
        bbox = bbox_world[entity_id][frame_indices]
        bbox_valid = bbox_world_valid[entity_id][frame_indices]
        vis = visibility[entity_id][frame_indices]
        area = mask_area[entity_id][frame_indices]
        bbox_extent = np.clip(bbox[:, 3:] - bbox[:, :3], 1.0e-4, None)
        extent = np.linalg.norm(bbox_extent, axis=1)
        extent = np.where(bbox_valid, extent, np.nan)
        mean_extent = float(np.nanmean(extent)) if np.isfinite(extent).any() else 0.05
        loaded.append(
            {
                "entity": entity,
                "gaussian_ids": np.asarray(entity.get("gaussian_ids", []), dtype=np.int64),
                "gaussian_scores": np.asarray(entity.get("gaussian_scores", []), dtype=np.float32),
                "center_world": center,
                "center_valid": center_valid & vis,
                "bbox_world": bbox,
                "bbox_valid": bbox_valid,
                "visibility": vis,
                "mask_area": area,
                "mean_extent": max(mean_extent, 0.02),
                "priority": _proposal_priority(entity),
            }
        )
    return loaded


def _normalize_features(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.clip(std, 1.0e-6, None)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size <= 1:
        return np.ones_like(flat, dtype=np.float32)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, flat.size, dtype=np.float32)
    return ranks


def _support_stats(bank: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    gate = np.asarray(bank["gate"], dtype=np.float32).reshape(bank["gate"].shape[0], bank["gate"].shape[1])
    time_values = np.asarray(bank["time_values"], dtype=np.float32).reshape(1, -1)
    gate_sum = np.clip(gate.sum(axis=1), 1.0e-6, None)
    gate_peak = gate.max(axis=1)
    active = gate >= np.maximum(0.18, gate_peak[:, None] * 0.35)
    if active.ndim != 2:
        active = active.reshape(gate.shape[0], gate.shape[1])

    support_center = (gate * time_values).sum(axis=1) / gate_sum
    support_span = active.mean(axis=1).astype(np.float32)
    support_start = np.zeros((gate.shape[0],), dtype=np.int32)
    support_end = np.full((gate.shape[0],), gate.shape[1], dtype=np.int32)
    for index in range(gate.shape[0]):
        indices = np.flatnonzero(active[index])
        if indices.size == 0:
            peak = int(np.argmax(gate[index]))
            support_start[index] = peak
            support_end[index] = min(peak + 1, gate.shape[1])
            continue
        support_start[index] = int(indices[0])
        support_end[index] = int(indices[-1]) + 1
    return {
        "gate_flat": gate,
        "support_center": support_center.astype(np.float32),
        "support_span": support_span.astype(np.float32),
        "support_start": support_start,
        "support_end": support_end,
        "support_peak": gate_peak.astype(np.float32),
    }


def _build_cluster_inputs(state: GaussianState, bank: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    support = _support_stats(bank)
    trajectories = np.asarray(bank["trajectories"], dtype=np.float32)
    trajectory_mean = trajectories.mean(axis=1)
    displacement = np.asarray(bank["displacement"], dtype=np.float32)
    velocity = np.asarray(state.velocity, dtype=np.float32)
    acceleration = np.asarray(state.acceleration, dtype=np.float32)
    rgb = np.asarray(state.rgb, dtype=np.float32)
    occupancy = np.asarray(bank["occupancy_mass"], dtype=np.float32).reshape(-1)
    visibility = np.asarray(bank["visibility_proxy"], dtype=np.float32).reshape(-1)
    support_factor = np.asarray(bank["support_factor"], dtype=np.float32).reshape(-1)
    effective_support = np.asarray(bank["effective_support"], dtype=np.float32).reshape(-1)
    motion = np.asarray(bank["motion_score"], dtype=np.float32).reshape(-1)
    path_length = np.asarray(bank["path_length"], dtype=np.float32).reshape(-1)
    spatial_extent = np.linalg.norm(np.asarray(state.spatial_scale, dtype=np.float32), axis=1)
    opacity = _sigmoid(np.asarray(state.opacity, dtype=np.float32).reshape(-1))

    scene_center = np.median(trajectory_mean, axis=0, keepdims=True)
    centrality = 1.0 - _rank_normalize(np.linalg.norm(trajectory_mean - scene_center, axis=1))

    occupancy_rank = _rank_normalize(occupancy)
    visibility_rank = _rank_normalize(visibility)
    opacity_rank = _rank_normalize(opacity)
    support_rank = _rank_normalize(effective_support)
    scale_rank = _rank_normalize(spatial_extent)
    path_rank = _rank_normalize(path_length)
    centrality_rank = _rank_normalize(centrality)

    salience = (
        0.28 * occupancy_rank
        + 0.18 * support_rank
        + 0.14 * visibility_rank
        + 0.12 * opacity_rank
        + 0.10 * scale_rank
        + 0.10 * path_rank
        + 0.08 * centrality_rank
    ).astype(np.float32)

    cluster_features = np.concatenate(
        [
            trajectory_mean,
            displacement,
            velocity,
            rgb,
            support["support_center"][:, None],
            support["support_span"][:, None],
            occupancy[:, None],
            visibility[:, None],
            path_length[:, None],
            centrality[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    merge_features = np.concatenate(
        [
            trajectory_mean,
            displacement,
            rgb,
            support["support_center"][:, None],
            support["support_span"][:, None],
            motion[:, None],
            occupancy[:, None],
            visibility[:, None],
            centrality[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    return {
        "trajectory_mean": trajectory_mean,
        "displacement": displacement,
        "velocity": velocity,
        "acceleration": acceleration,
        "rgb": rgb,
        "occupancy": occupancy,
        "visibility": visibility,
        "support_factor": support_factor,
        "effective_support": effective_support,
        "motion": motion,
        "path_length": path_length,
        "spatial_extent": spatial_extent,
        "opacity": opacity,
        "centrality": centrality,
        "salience": salience,
        "cluster_features": _normalize_features(cluster_features),
        "merge_features": _normalize_features(merge_features),
        **support,
    }


def _select_core_indices(
    salience: np.ndarray,
    sample_ratio: float,
    min_cluster_size: int,
) -> np.ndarray:
    num_points = salience.shape[0]
    if num_points == 0:
        return np.empty((0,), dtype=np.int32)
    core_ratio = float(np.clip(max(sample_ratio * 6.0, 0.12), 0.12, 0.35))
    core_count = int(np.clip(num_points * core_ratio, min_cluster_size * 64, min(num_points, 12000)))
    order = np.argsort(-salience, kind="mergesort")
    return np.sort(order[:core_count].astype(np.int32))


def _assign_to_centers(features: np.ndarray, centers: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    labels = np.empty((features.shape[0],), dtype=np.int32)
    for start in range(0, features.shape[0], chunk_size):
        end = min(start + chunk_size, features.shape[0])
        chunk = features[start:end, None, :] - centers[None, :, :]
        distances = np.linalg.norm(chunk, axis=2)
        labels[start:end] = np.argmin(distances, axis=1).astype(np.int32)
    return labels


def _window_iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    inter = max(0, min(end_a, end_b) - max(start_a, start_b))
    if inter <= 0:
        return 0.0
    union = max(end_a, end_b) - min(start_a, start_b)
    return float(inter / max(union, 1))


def _window_gap(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    if end_a < start_b:
        return float(start_b - end_a)
    if end_b < start_a:
        return float(start_a - end_b)
    return 0.0


def _remap_labels(labels: np.ndarray, priorities: dict[int, float] | None = None) -> np.ndarray:
    remapped = -np.ones_like(labels, dtype=np.int32)
    cluster_ids = [int(cluster_id) for cluster_id in np.unique(labels) if int(cluster_id) >= 0]
    if priorities:
        cluster_ids.sort(key=lambda cluster_id: (-priorities.get(cluster_id, 0.0), cluster_id))
    else:
        cluster_ids.sort()
    for new_id, cluster_id in enumerate(cluster_ids):
        remapped[labels == cluster_id] = int(new_id)
    return remapped


def _cluster_table(
    labels: np.ndarray,
    inputs: dict[str, np.ndarray],
    min_gaussians_per_entity: int,
) -> dict[int, dict[str, Any]]:
    table: dict[int, dict[str, Any]] = {}
    for cluster_id in [int(cluster_id) for cluster_id in np.unique(labels) if int(cluster_id) >= 0]:
        members = labels == cluster_id
        size = int(members.sum())
        if size == 0:
            continue
        salience_values = inputs["salience"][members]
        table[cluster_id] = {
            "cluster_id": cluster_id,
            "size": size,
            "small": size < int(min_gaussians_per_entity),
            "priority": float(
                0.45 * salience_values.mean()
                + 0.25 * np.log1p(size) / np.log(256.0)
                + 0.15 * inputs["occupancy"][members].mean()
                + 0.15 * inputs["support_span"][members].mean()
            ),
            "salience_mean": float(salience_values.mean()),
            "salience_sum": float(salience_values.sum()),
            "merge_feature": inputs["merge_features"][members].mean(axis=0),
            "trajectory_mean": inputs["trajectory_mean"][members].mean(axis=0),
            "velocity_mean": inputs["velocity"][members].mean(axis=0),
            "rgb_mean": inputs["rgb"][members].mean(axis=0),
            "support_center": float(inputs["support_center"][members].mean()),
            "support_span": float(inputs["support_span"][members].mean()),
            "support_start": int(np.round(inputs["support_start"][members].mean())),
            "support_end": int(np.round(inputs["support_end"][members].mean())),
            "occupancy_mean": float(inputs["occupancy"][members].mean()),
            "visibility_mean": float(inputs["visibility"][members].mean()),
        }
    return table


def _merge_cost(source: dict[str, Any], target: dict[str, Any]) -> float:
    feature_gap = float(np.linalg.norm(source["merge_feature"] - target["merge_feature"]))
    support_iou = _window_iou(
        source["support_start"],
        source["support_end"],
        target["support_start"],
        target["support_end"],
    )
    support_gap = _window_gap(
        source["support_start"],
        source["support_end"],
        target["support_start"],
        target["support_end"],
    )
    velocity_gap = float(np.linalg.norm(source["velocity_mean"] - target["velocity_mean"]))
    rgb_gap = float(np.linalg.norm(source["rgb_mean"] - target["rgb_mean"]))
    cost = feature_gap + 0.20 * velocity_gap + 0.12 * rgb_gap + 0.03 * support_gap - 0.30 * support_iou
    if source["support_span"] <= 0.20 and target["support_span"] >= 0.45:
        cost *= 0.82
    if source["occupancy_mean"] <= 0.08 and target["occupancy_mean"] >= source["occupancy_mean"]:
        cost *= 0.85
    return float(cost)


def _merge_cluster_labels(
    labels: np.ndarray,
    inputs: dict[str, np.ndarray],
    max_entities: int,
    min_gaussians_per_entity: int,
) -> np.ndarray:
    merged = labels.astype(np.int32).copy()
    safety = 0
    while safety < 512:
        safety += 1
        table = _cluster_table(merged, inputs, min_gaussians_per_entity=min_gaussians_per_entity)
        cluster_ids = sorted(table.keys())
        if not cluster_ids:
            return merged
        if len(cluster_ids) <= int(max_entities) and all(not table[cluster_id]["small"] for cluster_id in cluster_ids):
            return _remap_labels(merged, priorities={cluster_id: table[cluster_id]["priority"] for cluster_id in cluster_ids})

        source_id = min(
            cluster_ids,
            key=lambda cluster_id: (
                0 if table[cluster_id]["small"] else 1,
                table[cluster_id]["priority"],
                table[cluster_id]["size"],
            ),
        )
        source = table[source_id]
        targets = [cluster_id for cluster_id in cluster_ids if cluster_id != source_id]
        if not targets:
            break
        target_id = min(targets, key=lambda cluster_id: _merge_cost(source, table[cluster_id]))
        merged[merged == source_id] = int(target_id)
        merged = _remap_labels(merged)

    return _remap_labels(merged)


def _natural_merge_clusters(
    labels: np.ndarray,
    inputs: dict[str, np.ndarray],
    min_gaussians_per_entity: int,
) -> np.ndarray:
    merged = labels.astype(np.int32).copy()
    safety = 0
    while safety < 256:
        safety += 1
        table = _cluster_table(merged, inputs, min_gaussians_per_entity=min_gaussians_per_entity)
        cluster_ids = sorted(table.keys())
        if len(cluster_ids) <= 1:
            break

        best: tuple[float, int, int] | None = None
        for index, source_id in enumerate(cluster_ids):
            for target_id in cluster_ids[index + 1 :]:
                source = table[source_id]
                target = table[target_id]
                cost = _merge_cost(source, target)
                both_long = source["support_span"] >= 0.80 and target["support_span"] >= 0.80
                both_short = source["support_span"] <= 0.28 and target["support_span"] <= 0.28
                if not (cost <= 0.95 or (both_short and cost <= 1.40) or (both_long and cost <= 1.05)):
                    continue
                if best is None or cost < best[0]:
                    best = (float(cost), int(source_id), int(target_id))

        if best is None:
            break

        _, source_id, target_id = best
        if table[source_id]["priority"] > table[target_id]["priority"]:
            source_id, target_id = target_id, source_id
        merged[merged == source_id] = int(target_id)
        merged = _remap_labels(merged)

    return _remap_labels(merged)


def _is_fragment_cluster(cluster: dict[str, Any]) -> bool:
    return bool(
        (cluster["support_span"] <= 0.12 and cluster["occupancy_mean"] <= 0.08)
        or (cluster["support_span"] <= 0.08 and cluster["visibility_mean"] >= 0.82)
        or (cluster["support_span"] <= 0.06 and cluster["salience_mean"] <= 0.42)
    )


def _absorb_fragment_clusters(
    labels: np.ndarray,
    inputs: dict[str, np.ndarray],
    min_gaussians_per_entity: int,
) -> np.ndarray:
    merged = labels.astype(np.int32).copy()
    safety = 0
    while safety < 256:
        safety += 1
        table = _cluster_table(merged, inputs, min_gaussians_per_entity=min_gaussians_per_entity)
        cluster_ids = sorted(table.keys())
        fragment_ids = [cluster_id for cluster_id in cluster_ids if _is_fragment_cluster(table[cluster_id])]
        if not fragment_ids:
            break
        stable_ids = [cluster_id for cluster_id in cluster_ids if cluster_id not in fragment_ids]
        if not stable_ids:
            break

        source_id = min(
            fragment_ids,
            key=lambda cluster_id: (
                table[cluster_id]["support_span"],
                table[cluster_id]["occupancy_mean"],
                table[cluster_id]["priority"],
                table[cluster_id]["size"],
            ),
        )
        source = table[source_id]

        def fragment_target_score(target_id: int) -> float:
            target = table[target_id]
            base = _merge_cost(source, target)
            if target["support_span"] >= 0.75:
                base *= 0.72
            if target["occupancy_mean"] >= 0.20:
                base *= 0.82
            if target["visibility_mean"] <= 0.85:
                base *= 0.92
            return float(base)

        target_id = min(stable_ids, key=fragment_target_score)
        merged[merged == source_id] = int(target_id)
        merged = _remap_labels(merged)

    return _remap_labels(merged)


def _support_aware_cluster(
    state: GaussianState,
    bank: dict[str, np.ndarray],
    sample_ratio: float,
    min_cluster_size: int,
    min_gaussians_per_entity: int,
    max_entities: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    inputs = _build_cluster_inputs(state, bank)
    core_indices = _select_core_indices(inputs["salience"], sample_ratio=sample_ratio, min_cluster_size=min_cluster_size)
    if core_indices.size == 0:
        return np.empty((0,), dtype=np.int32), {"method": "support_aware_worldtube_kmeans"}

    seed_count = int(
        np.clip(
            round(np.sqrt(float(core_indices.size)) / 2.8),
            8,
            max(8, int(max_entities * 1.5)),
        )
    )
    seed_count = min(seed_count, core_indices.size)
    if seed_count <= 1:
        labels = np.zeros((inputs["cluster_features"].shape[0],), dtype=np.int32)
    else:
        np.random.seed(0)
        core_features = inputs["cluster_features"][core_indices]
        centers, _ = kmeans2(core_features, k=seed_count, minit="points", iter=32)
        labels = _assign_to_centers(inputs["cluster_features"], centers)

    labels = _merge_cluster_labels(
        labels,
        inputs=inputs,
        max_entities=max_entities,
        min_gaussians_per_entity=min_gaussians_per_entity,
    )
    labels = _split_oversized_clusters(
        labels,
        inputs=inputs,
        min_gaussians_per_entity=min_gaussians_per_entity,
        max_entities=max_entities,
    )
    labels = _natural_merge_clusters(
        labels,
        inputs=inputs,
        min_gaussians_per_entity=min_gaussians_per_entity,
    )
    labels = _absorb_fragment_clusters(
        labels,
        inputs=inputs,
        min_gaussians_per_entity=min_gaussians_per_entity,
    )
    raw_cluster_ids = [int(cluster_id) for cluster_id in np.unique(labels) if int(cluster_id) >= 0]
    priorities = _cluster_table(labels, inputs, min_gaussians_per_entity=min_gaussians_per_entity)
    labels = _remap_labels(labels, priorities={cluster_id: priorities[cluster_id]["priority"] for cluster_id in priorities})
    return labels, {
        "method": "support_aware_worldtube_kmeans",
        "core_count": int(core_indices.size),
        "seed_count": int(seed_count),
        "raw_cluster_count": int(len(raw_cluster_ids)),
    }


def _split_oversized_clusters(
    labels: np.ndarray,
    inputs: dict[str, np.ndarray],
    min_gaussians_per_entity: int,
    max_entities: int,
) -> np.ndarray:
    split_labels = labels.astype(np.int32).copy()
    safety = 0
    while safety < 32:
        safety += 1
        table = _cluster_table(split_labels, inputs, min_gaussians_per_entity=min_gaussians_per_entity)
        cluster_ids = sorted(table.keys())
        if not cluster_ids:
            return split_labels
        if len(cluster_ids) >= int(max_entities):
            return split_labels
        sizes = np.asarray([table[cluster_id]["size"] for cluster_id in cluster_ids], dtype=np.float32)
        median_size = float(np.median(sizes))
        did_split = False
        next_cluster_id = max(cluster_ids) + 1

        for cluster_id in sorted(cluster_ids, key=lambda item: table[item]["size"], reverse=True):
            cluster = table[cluster_id]
            oversized = cluster["size"] >= max(6 * int(min_gaussians_per_entity), int(max(2.6 * median_size, 2600)))
            stable = cluster["support_span"] >= 0.55 and cluster["occupancy_mean"] >= 0.18
            if not (oversized and stable):
                continue
            member_indices = np.where(split_labels == cluster_id)[0]
            if member_indices.size < max(6 * int(min_gaussians_per_entity), 512):
                continue
            sub_features = np.concatenate(
                [
                    inputs["trajectory_mean"][member_indices],
                    inputs["velocity"][member_indices],
                    inputs["rgb"][member_indices],
                    inputs["support_center"][member_indices, None],
                ],
                axis=1,
            ).astype(np.float32)
            sub_features = _normalize_features(sub_features)
            k = 3 if cluster["size"] >= 9000 and len(cluster_ids) + 2 <= int(max_entities) else 2
            if len(cluster_ids) + (k - 1) > int(max_entities):
                continue
            np.random.seed(0)
            centers, sub_labels = kmeans2(sub_features, k=k, minit="points", iter=24)
            sub_sizes = [int((sub_labels == sub_id).sum()) for sub_id in range(k)]
            if min(sub_sizes) < max(4 * int(min_gaussians_per_entity), 256):
                continue
            center_dists = []
            for index in range(k):
                for other_index in range(index + 1, k):
                    center_dists.append(float(np.linalg.norm(centers[index] - centers[other_index])))
            if not center_dists or max(center_dists) < 0.75:
                continue

            for sub_id in range(k):
                if sub_id == 0:
                    continue
                split_labels[member_indices[sub_labels == sub_id]] = int(next_cluster_id)
                next_cluster_id += 1
            split_labels = _remap_labels(split_labels)
            did_split = True
            break

        if not did_split:
            break

    return _remap_labels(split_labels)


def _segments_from_trajectory(trajectory: np.ndarray, gate: np.ndarray) -> list[dict[str, Any]]:
    speed = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    if speed.size == 0:
        return []
    threshold = float(np.quantile(speed, 0.65))
    labels = np.where(speed > threshold, "moving", "stationary")

    segments: list[dict[str, Any]] = []
    start = 0
    current = labels[0]
    for idx in range(1, labels.shape[0]):
        if labels[idx] == current:
            continue
        segments.append(
            {
                "segment_id": len(segments),
                "t0": int(start),
                "t1": int(idx),
                "label": current,
                "confidence": float(np.clip(gate[start : idx + 1].mean(), 0.0, 1.0)),
                "mode": "world_trajectory",
            }
        )
        start = idx
        current = labels[idx]
    segments.append(
        {
            "segment_id": len(segments),
            "t0": int(start),
            "t1": int(labels.shape[0]),
            "label": current,
            "confidence": float(np.clip(gate[start:].mean(), 0.0, 1.0)),
            "mode": "world_trajectory",
        }
    )
    return segments


def _keyframes_from_gate(gate: np.ndarray, top_k: int = 8) -> list[int]:
    indices = np.argsort(-gate.squeeze(-1))[:top_k]
    return sorted(int(index) for index in indices.tolist())


def _support_window_from_gate(gate: np.ndarray, threshold: float = 0.35) -> dict[str, Any]:
    flat_gate = np.asarray(gate, dtype=np.float32).reshape(-1)
    if flat_gate.size == 0:
        return {"frame_start": 0, "frame_end": 0, "frame_peak": 0, "span_ratio": 0.0, "peak_value": 0.0}
    active = flat_gate >= threshold
    if not active.any():
        peak = int(np.argmax(flat_gate))
        active[max(0, peak - 1) : min(flat_gate.size, peak + 2)] = True
    active_indices = np.where(active)[0]
    frame_start = int(active_indices[0])
    frame_end = int(active_indices[-1]) + 1
    frame_peak = int(np.argmax(flat_gate))
    return {
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_peak": frame_peak,
        "span_ratio": float((frame_end - frame_start) / max(flat_gate.size, 1)),
        "peak_value": float(flat_gate[frame_peak]),
    }


def _segment_frame_ratio(segments: list[dict[str, Any]], label: str, total_frames: int) -> float:
    covered = 0
    for segment in segments:
        if segment.get("label") != label:
            continue
        t0 = int(segment.get("t0", 0))
        t1 = int(segment.get("t1", t0))
        covered += max(t1 - t0, 0)
    return float(covered / max(total_frames, 1))


def _temporal_mode(
    moving_ratio: float,
    stationary_ratio: float,
    support_span_ratio: float,
    visibility_ratio: float,
) -> str:
    if moving_ratio < 0.10 and stationary_ratio > 0.65:
        return "static_region"
    if moving_ratio > 0.15 and support_span_ratio < 0.50:
        return "transient_event"
    if moving_ratio > 0.15:
        return "dynamic_object"
    if visibility_ratio < 0.50 and support_span_ratio < 0.50:
        return "temporally_localized_region"
    return "static_region"


def _role_hints(
    temporal_mode: str,
    support_span_ratio: float,
    occupancy_mean: float,
    tube_ratio_mean: float,
) -> list[str]:
    hints: list[str] = []
    if temporal_mode == "dynamic_object":
        hints.extend(["dynamic", "trackable"])
    elif temporal_mode == "transient_event":
        hints.extend(["dynamic", "temporally_localized"])
    elif temporal_mode == "temporally_localized_region":
        hints.append("temporally_localized")
    else:
        hints.extend(["static", "persistent"])
    if support_span_ratio >= 0.75:
        hints.append("long_horizon")
    elif support_span_ratio <= 0.35:
        hints.append("short_horizon")
    if occupancy_mean >= 0.20:
        hints.append("high_occupancy")
    if tube_ratio_mean >= 0.20:
        hints.append("motion_aligned")
    return hints


def _proposal_labels(
    bank: dict[str, np.ndarray],
    proposals: list[dict[str, Any]],
    min_gaussians_per_entity: int,
    strict_proposals: bool = False,
) -> np.ndarray:
    num_gaussians = bank["trajectories"].shape[0]
    num_proposals = len(proposals)
    if num_proposals == 0:
        return -np.ones((num_gaussians,), dtype=np.int32)

    if strict_proposals:
        labels = -np.ones((num_gaussians,), dtype=np.int32)
        best_scores = np.full((num_gaussians,), -np.inf, dtype=np.float32)
        for proposal_id, proposal in enumerate(proposals):
            gaussian_ids = np.asarray(proposal.get("gaussian_ids", []), dtype=np.int64).reshape(-1)
            if gaussian_ids.size == 0:
                continue
            gaussian_ids = gaussian_ids[(gaussian_ids >= 0) & (gaussian_ids < num_gaussians)]
            gaussian_scores = np.asarray(proposal.get("gaussian_scores", []), dtype=np.float32).reshape(-1)
            if gaussian_scores.size != gaussian_ids.size:
                gaussian_scores = np.ones((gaussian_ids.size,), dtype=np.float32)
            proposal_score = gaussian_scores + 0.05 * float(proposal.get("priority", 0.0))
            better = proposal_score > best_scores[gaussian_ids]
            if not np.any(better):
                continue
            chosen_ids = gaussian_ids[better]
            best_scores[chosen_ids] = proposal_score[better]
            labels[chosen_ids] = int(proposal_id)

        counts = {cluster_id: int((labels == cluster_id).sum()) for cluster_id in range(num_proposals)}
        kept = [cluster_id for cluster_id, count in counts.items() if count >= int(min_gaussians_per_entity)]
        if not kept:
            kept = [cluster_id for cluster_id, count in counts.items() if count > 0]
        if not kept:
            return -np.ones((num_gaussians,), dtype=np.int32)

        remapped = np.full_like(labels, -1)
        for new_id, cluster_id in enumerate(kept):
            remapped[labels == cluster_id] = int(new_id)
        return _remap_labels(remapped)

    trajectories = np.asarray(bank["trajectories"], dtype=np.float32)
    gate = np.asarray(bank["gate"], dtype=np.float32).reshape(num_gaussians, -1)
    gate_peak = gate.max(axis=1, keepdims=True)
    gate_active = gate >= np.maximum(0.15, gate_peak * 0.35)

    proposal_centers = np.stack([proposal["center_world"] for proposal in proposals], axis=0).astype(np.float32)
    proposal_valid = np.stack([proposal["center_valid"] for proposal in proposals], axis=0).astype(bool)
    proposal_extent = np.asarray([proposal["mean_extent"] for proposal in proposals], dtype=np.float32)
    proposal_priority = np.asarray([proposal["priority"] for proposal in proposals], dtype=np.float32)

    labels = np.full((num_gaussians,), -1, dtype=np.int32)
    chunk_size = 1024
    for start in range(0, num_gaussians, chunk_size):
        end = min(start + chunk_size, num_gaussians)
        traj = trajectories[start:end, None, :, :]  # [B,1,T,3]
        dist = np.linalg.norm(traj - proposal_centers[None, :, :, :], axis=3)
        dist = dist / np.clip(proposal_extent[None, :, None], 1.0e-3, None)

        weights = gate[start:end, None, :] * proposal_valid[None, :, :]
        overlap = (gate_active[start:end, None, :] & proposal_valid[None, :, :]).sum(axis=2).astype(np.float32)
        active_count = np.clip(gate_active[start:end].sum(axis=1, keepdims=True).astype(np.float32), 1.0, None)
        overlap_ratio = overlap / active_count
        weight_sum = weights.sum(axis=2)

        score = (dist * weights).sum(axis=2) / np.clip(weight_sum, 1.0e-6, None)
        score = score - 0.30 * overlap_ratio - 0.08 * proposal_priority[None, :]
        score = np.where(overlap_ratio >= 0.10, score, 1.0e6)
        labels[start:end] = np.argmin(score, axis=1).astype(np.int32)

    counts = {cluster_id: int((labels == cluster_id).sum()) for cluster_id in range(num_proposals)}
    kept = [cluster_id for cluster_id, count in counts.items() if count >= int(min_gaussians_per_entity)]
    if not kept:
        kept = [int(max(counts, key=counts.get))]

    remapped = np.full_like(labels, -1)
    for new_id, cluster_id in enumerate(kept):
        remapped[labels == cluster_id] = int(new_id)

    if np.any(remapped < 0):
        kept_centers = np.stack([proposals[cluster_id]["center_world"] for cluster_id in kept], axis=0).astype(np.float32)
        kept_valid = np.stack([proposals[cluster_id]["center_valid"] for cluster_id in kept], axis=0).astype(bool)
        kept_extent = np.asarray([proposals[cluster_id]["mean_extent"] for cluster_id in kept], dtype=np.float32)
        orphan_indices = np.where(remapped < 0)[0]
        for start in range(0, orphan_indices.size, chunk_size):
            end = min(start + chunk_size, orphan_indices.size)
            chunk_ids = orphan_indices[start:end]
            traj = trajectories[chunk_ids, None, :, :]
            dist = np.linalg.norm(traj - kept_centers[None, :, :, :], axis=3)
            dist = dist / np.clip(kept_extent[None, :, None], 1.0e-3, None)
            weights = gate[chunk_ids, None, :] * kept_valid[None, :, :]
            score = (dist * weights).sum(axis=2) / np.clip(weights.sum(axis=2), 1.0e-6, None)
            remapped[chunk_ids] = np.argmin(score, axis=1).astype(np.int32)

    return _remap_labels(remapped)


def export_entitybank(
    run_dir: str | Path,
    num_frames: int = 64,
    sample_ratio: float = 0.02,
    min_cluster_size: int = 10,
    min_gaussians_per_entity: int = 32,
    max_entities: int = 30,
    proposal_dir: str | Path | None = None,
    proposal_strict: bool = False,
    output_dir: str | Path | None = None,
) -> Path:
    run_dir = Path(run_dir)
    state, config, iteration = load_gaussian_state(run_dir)
    gate_sharpness = float(config.get("temporal_gate_sharpness", 1.0))
    drift_mix = float(config.get("temporal_drift_mix", 1.0))
    bank = sample_tube_bank(
        state,
        num_frames=num_frames,
        gate_sharpness=gate_sharpness,
        drift_mix=drift_mix,
        config=config,
    )
    out_dir = save_tube_bank(
        run_dir,
        state,
        bank,
        iteration,
        output_dir=None if output_dir is None else Path(output_dir),
    )

    proposal_entities: list[dict[str, Any]] | None = None
    labels: np.ndarray | None = None
    if proposal_dir is not None:
        proposal_dir = Path(proposal_dir)
        proposal_entities = _load_proposals(
            proposal_dir=proposal_dir,
            target_time_values=np.asarray(bank["time_values"], dtype=np.float32),
            max_entities=max_entities,
        )
        if not proposal_strict:
            labels = _proposal_labels(
                bank=bank,
                proposals=proposal_entities,
                min_gaussians_per_entity=min_gaussians_per_entity,
                strict_proposals=False,
            )
        cluster_meta = {
            "method": "trase_proposal_worldtube_strict" if proposal_strict else "trase_proposal_worldtube_reassignment",
            "proposal_dir": str(proposal_dir),
            "proposal_count": int(len(proposal_entities)),
            "raw_cluster_count": int(len(proposal_entities)) if proposal_strict else int(len([cluster_id for cluster_id in np.unique(labels) if cluster_id >= 0])),
            "proposal_strict": bool(proposal_strict),
        }
    else:
        labels, cluster_meta = _support_aware_cluster(
            state,
            bank,
            sample_ratio=sample_ratio,
            min_cluster_size=min_cluster_size,
            min_gaussians_per_entity=min_gaussians_per_entity,
            max_entities=max_entities,
        )
    if proposal_entities is not None and proposal_strict:
        raw_cluster_ids = []
        for proposal_index, proposal in enumerate(proposal_entities):
            gaussian_ids = np.asarray(proposal.get("gaussian_ids", []), dtype=np.int64).reshape(-1)
            gaussian_ids = gaussian_ids[(gaussian_ids >= 0) & (gaussian_ids < state.xyz.shape[0])]
            if gaussian_ids.size >= int(min_gaussians_per_entity):
                raw_cluster_ids.append(int(proposal_index))
        if not raw_cluster_ids and proposal_entities:
            raw_cluster_ids = [int(np.argmax([float(item.get("priority", 0.0)) for item in proposal_entities]))]
    else:
        raw_cluster_ids = sorted(cluster_id for cluster_id in np.unique(labels) if cluster_id >= 0)

    clusters: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    for cluster_id in raw_cluster_ids:
        proposal_entity = None
        if proposal_entities is not None and cluster_id < len(proposal_entities):
            proposal_entity = proposal_entities[cluster_id]["entity"]

        if proposal_entity is not None and proposal_strict:
            gaussian_ids = np.asarray(proposal_entity.get("gaussian_ids", []), dtype=np.int64).reshape(-1)
            gaussian_ids = gaussian_ids[(gaussian_ids >= 0) & (gaussian_ids < state.xyz.shape[0])]
            gaussian_ids = np.unique(gaussian_ids)
        else:
            gaussian_ids = np.where(labels == cluster_id)[0]
        if gaussian_ids.size == 0:
            continue

        cluster_traj = bank["trajectories"][gaussian_ids].mean(axis=0)
        cluster_gate = bank["gate"][gaussian_ids].mean(axis=0)
        cluster_motion = float(bank["motion_score"][gaussian_ids].mean())
        cluster_path = float(bank["path_length"][gaussian_ids].mean())
        cluster_occupancy = float(bank["occupancy_mass"][gaussian_ids].mean())
        cluster_visibility = float(bank["visibility_proxy"][gaussian_ids].mean())
        cluster_tube_ratio = float(bank["tube_ratio"][gaussian_ids].mean())
        cluster_support_factor = float(bank["support_factor"][gaussian_ids].mean())
        cluster_effective_support = float(bank["effective_support"][gaussian_ids].mean())
        cluster_rgb = state.rgb[gaussian_ids].mean(axis=0)
        segments = _segments_from_trajectory(cluster_traj, cluster_gate)
        keyframes = _keyframes_from_gate(cluster_gate)
        support_window = _support_window_from_gate(cluster_gate)
        visibility_ratio = float(np.clip((cluster_gate > 0.15).mean(), 0.0, 1.0))
        moving_ratio = _segment_frame_ratio(segments, "moving", cluster_traj.shape[0])
        stationary_ratio = _segment_frame_ratio(segments, "stationary", cluster_traj.shape[0])
        temporal_mode = _temporal_mode(
            moving_ratio=moving_ratio,
            stationary_ratio=stationary_ratio,
            support_span_ratio=float(support_window["span_ratio"]),
            visibility_ratio=visibility_ratio,
        )
        role_hints = _role_hints(
            temporal_mode=temporal_mode,
            support_span_ratio=float(support_window["span_ratio"]),
            occupancy_mean=cluster_occupancy,
            tube_ratio_mean=cluster_tube_ratio,
        )

        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "num_gaussians": int(gaussian_ids.size),
                "motion_score": cluster_motion,
                "path_length": cluster_path,
                "anchor_mean": float(state.anchor[gaussian_ids].mean()),
                "scale_mean": float(state.scale[gaussian_ids].mean()),
                "occupancy_mean": cluster_occupancy,
                "visibility_mean": cluster_visibility,
                "tube_ratio_mean": cluster_tube_ratio,
                "support_factor_mean": cluster_support_factor,
                "effective_support_mean": cluster_effective_support,
                "support_frame_start": int(support_window["frame_start"]),
                "support_frame_end": int(support_window["frame_end"]),
                "support_frame_peak": int(support_window["frame_peak"]),
                "support_span_ratio": float(support_window["span_ratio"]),
                "dynamic_frame_ratio": moving_ratio,
                "stationary_frame_ratio": stationary_ratio,
                "temporal_mode": temporal_mode,
                "mean_rgb": cluster_rgb.astype(float).tolist(),
                "proposal_entity_id": int(proposal_entity["id"]) if proposal_entity is not None else None,
            }
        )

        entities.append(
            {
                "id": len(entities),
                "source_cluster_id": int(proposal_entity["id"]) if proposal_entity is not None else int(cluster_id),
                "proposal_alias": proposal_entity.get("proposal_alias") if proposal_entity is not None else None,
                "proposal_phase": proposal_entity.get("proposal_phase") if proposal_entity is not None else None,
                "proposal_variant": proposal_entity.get("proposal_variant") if proposal_entity is not None else None,
                "proposal_support_frames": proposal_entity.get("support_frames", []) if proposal_entity is not None else [],
                "gaussian_ids": gaussian_ids.astype(int).tolist(),
                "keyframes": proposal_entity.get("keyframes", keyframes) if proposal_entity is not None else keyframes,
                "segments": proposal_entity.get("segments", segments) if proposal_entity is not None else segments,
                "static_text": proposal_entity.get("static_text", "") if proposal_entity is not None else "",
                "global_desc": proposal_entity.get("global_desc", f"ReferGaussian entity cluster {cluster_id} with {gaussian_ids.size} Gaussians.") if proposal_entity is not None else f"ReferGaussian entity cluster {cluster_id} with {gaussian_ids.size} Gaussians.",
                "dyn_desc": proposal_entity.get("dyn_desc", []) if proposal_entity is not None else [],
                "visibility_ratio": visibility_ratio,
                "mean_mask_area": float(proposal_entity.get("mean_mask_area", 0.0)) if proposal_entity is not None else 0.0,
                "quality": float(
                    0.5 * float(np.clip(cluster_motion / max(cluster_path, 1.0e-6), 0.0, 1.0))
                    + 0.5 * float(proposal_entity.get("quality", 0.0))
                ) if proposal_entity is not None else float(np.clip(cluster_motion / max(cluster_path, 1.0e-6), 0.0, 1.0)),
                "occupancy_mean": cluster_occupancy,
                "visibility_mean": cluster_visibility,
                "tube_ratio_mean": cluster_tube_ratio,
                "support_factor_mean": cluster_support_factor,
                "effective_support_mean": cluster_effective_support,
                "support_frame_start": int(support_window["frame_start"]),
                "support_frame_end": int(support_window["frame_end"]),
                "support_frame_peak": int(support_window["frame_peak"]),
                "support_span_ratio": float(support_window["span_ratio"]),
                "support_peak_value": float(support_window["peak_value"]),
                "dynamic_frame_ratio": moving_ratio,
                "stationary_frame_ratio": stationary_ratio,
                "entity_type": proposal_entity.get("entity_type", temporal_mode) if proposal_entity is not None else temporal_mode,
                "temporal_mode": temporal_mode,
                "role_hints": proposal_entity.get("role_hints", role_hints) if proposal_entity is not None else role_hints,
                "mask_refine_source": "worldtube_reassigned_trase_proposal" if proposal_entity is not None else "refergaussian_tube_bank",
                "bbox_image_pt_key": "bbox_image",
            }
        )

    cluster_stats = {
        "schema_version": 2,
        "num_gaussians": int(state.xyz.shape[0]),
        "num_clusters_raw": int(cluster_meta.get("raw_cluster_count", len(raw_cluster_ids))),
        "num_clusters_kept": int(len(clusters)),
        "params": {
            "method": cluster_meta.get("method", "support_aware_worldtube_kmeans"),
            "sample_ratio": sample_ratio,
            "min_cluster_size": min_cluster_size,
            "min_gaussians_per_entity": min_gaussians_per_entity,
            "max_entities": max_entities,
            "core_count": int(cluster_meta.get("core_count", 0)),
            "seed_count": int(cluster_meta.get("seed_count", 0)),
            "proposal_dir": cluster_meta.get("proposal_dir"),
            "proposal_count": cluster_meta.get("proposal_count"),
            "merge_cosine_threshold": None,
            "merge_iou_threshold": None,
        },
        "clusters": clusters,
        "iteration": iteration,
    }

    entities_payload = {
        "schema_version": 2,
        "iteration": iteration,
        "split": run_dir.parts[-2] if len(run_dir.parts) >= 2 else "unknown",
        "frame_count": int(bank["time_values"].shape[0]),
        "frame_names": [f"frame_{idx:04d}" for idx in range(bank["time_values"].shape[0])],
        "time_values": bank["time_values"].astype(float).tolist(),
        "num_entities": int(len(entities)),
        "decomposition": {
            "mode": "proposal_worldtube_reassignment" if proposal_entities is not None else "support_aware_worldtube",
            "primitive": "refergaussian_tube_bank",
        },
        "entities": entities,
    }

    with open(out_dir / "cluster_stats.json", "w", encoding="utf-8") as handle:
        json.dump(cluster_stats, handle, indent=2)
    with open(out_dir / "entities.json", "w", encoding="utf-8") as handle:
        json.dump(entities_payload, handle, indent=2)
    return out_dir
