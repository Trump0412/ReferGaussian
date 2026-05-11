from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData


@dataclass
class ConsistencySelection:
    selected_ids: np.ndarray
    selected_scores: np.ndarray
    summary: dict[str, float | int | None]


def load_opacity_sigmoid(run_dir: Path) -> np.ndarray:
    point_cloud_root = run_dir / "point_cloud"
    candidates: list[tuple[int, Path]] = []
    for child in point_cloud_root.iterdir():
        if not child.is_dir() or not child.name.startswith("iteration_"):
            continue
        try:
            iteration = int(child.name.split("_", 1)[1])
        except ValueError:
            continue
        candidates.append((iteration, child / "point_cloud.ply"))
    if not candidates:
        raise FileNotFoundError(f"No iteration_* directories found under {point_cloud_root}")
    _, ply_path = max(candidates, key=lambda item: item[0])
    ply = PlyData.read(str(ply_path))
    opacity = np.asarray(ply["vertex"].data["opacity"], dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-opacity))).astype(np.float32)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return (values - mean) / np.clip(std, 1.0e-6, None)


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norm, 1.0e-6, None)


def _rank_scale(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size <= 1:
        return np.ones_like(flat, dtype=np.float32)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, flat.size, dtype=np.float32)
    return ranks


def _query_gate_stats(gate: np.ndarray, sampled_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query_gate = np.asarray(gate[:, sampled_indices], dtype=np.float32)
    gate_peak = query_gate.max(axis=1, keepdims=True)
    active = query_gate >= np.maximum(0.15, 0.35 * gate_peak)
    active_ratio = active.mean(axis=1).astype(np.float32)
    mean_gate = query_gate.mean(axis=1).astype(np.float32)
    return active_ratio, mean_gate


def _query_features(bank: dict[str, np.ndarray], sampled_indices: np.ndarray, opacity_sigmoid: np.ndarray) -> dict[str, np.ndarray]:
    trajectories = np.asarray(bank["trajectories"], dtype=np.float32)
    velocity = np.asarray(bank.get("velocity", np.zeros((trajectories.shape[0], 3), dtype=np.float32)), dtype=np.float32)
    displacement = np.asarray(bank["displacement"], dtype=np.float32)
    xyz = np.asarray(bank.get("xyz", trajectories[:, 0, :]), dtype=np.float32)
    spatial_scale = np.asarray(bank.get("spatial_scale", np.ones((trajectories.shape[0], 3), dtype=np.float32)), dtype=np.float32)
    gate = np.asarray(bank["gate"], dtype=np.float32).reshape(trajectories.shape[0], trajectories.shape[1])
    query_traj = trajectories[:, sampled_indices, :]
    query_mean = query_traj.mean(axis=1).astype(np.float32)
    query_delta = (query_traj[:, -1, :] - query_traj[:, 0, :]).astype(np.float32)
    query_path = np.linalg.norm(np.diff(query_traj, axis=1), axis=2).sum(axis=1, keepdims=True).astype(np.float32)
    scale_norm = np.linalg.norm(spatial_scale, axis=1, keepdims=True).astype(np.float32)
    active_ratio, mean_gate = _query_gate_stats(gate, sampled_indices=sampled_indices)

    feature_stack = np.concatenate(
        [
            query_mean,
            query_delta,
            velocity,
            displacement,
            xyz,
            scale_norm,
            query_path,
            active_ratio[:, None],
            mean_gate[:, None],
            opacity_sigmoid[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    feature_norm = _l2_normalize(_zscore(feature_stack))

    return {
        "query_traj": query_traj,
        "query_mean": query_mean,
        "query_delta": query_delta,
        "query_path": query_path.reshape(-1),
        "active_ratio": active_ratio,
        "mean_gate": mean_gate,
        "scale_norm": scale_norm.reshape(-1),
        "features": feature_norm,
    }


def _seed_ids(
    support_score: np.ndarray,
    hit_count: np.ndarray,
    opacity_sigmoid: np.ndarray,
    active_ratio: np.ndarray,
    min_hits: int,
    seed_ratio: float,
) -> np.ndarray:
    support_score = np.asarray(support_score, dtype=np.float32).reshape(-1)
    hit_count = np.asarray(hit_count, dtype=np.float32).reshape(-1)
    seed_score = support_score * np.clip(opacity_sigmoid, 1.0e-6, 1.0) * (0.5 + 0.5 * active_ratio)
    eligible = np.where((support_score > 0.0) & (hit_count >= float(min_hits)))[0]
    if eligible.size == 0:
        eligible = np.where(support_score > 0.0)[0]
    if eligible.size == 0:
        return np.empty((0,), dtype=np.int64)
    seed_count = int(np.clip(round(float(eligible.size) * float(seed_ratio)), 256, min(1024, eligible.size)))
    ranked = eligible[np.argsort(-seed_score[eligible], kind="mergesort")]
    return ranked[:seed_count].astype(np.int64)


def _weighted_center(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    values = np.asarray(values, dtype=np.float32)
    weights = weights / np.clip(weights.sum(), 1.0e-6, None)
    return np.tensordot(weights, values, axes=(0, 0)).astype(np.float32)


def _trajectory_proximity(
    query_traj: np.ndarray,
    scale_norm: np.ndarray,
    seed_ids: np.ndarray,
    seed_weights: np.ndarray,
) -> np.ndarray:
    seed_template = _weighted_center(query_traj[seed_ids], seed_weights)
    seed_distances = np.linalg.norm(query_traj[seed_ids] - seed_template[None, :, :], axis=2)
    seed_radius = np.quantile(seed_distances, 0.80, axis=0).astype(np.float32)
    scale_bias = float(np.median(scale_norm[seed_ids])) if seed_ids.size else 0.05
    seed_radius = np.clip(seed_radius + scale_bias, 1.0e-3, None)
    distances = np.linalg.norm(query_traj - seed_template[None, :, :], axis=2)
    normalized = distances / seed_radius[None, :]
    return np.exp(-np.clip(normalized.mean(axis=1), 0.0, 12.0)).astype(np.float32)


def select_worldtube_consistency_cluster(
    bank: dict[str, np.ndarray],
    sampled_indices: np.ndarray,
    support_score: np.ndarray,
    hit_count: np.ndarray,
    opacity_sigmoid: np.ndarray,
    proposal_keep_ratio: float,
    min_gaussians: int,
    max_gaussians: int,
    seed_ratio: float = 0.05,
    expansion_factor: float = 4.0,
) -> ConsistencySelection:
    support_score = np.asarray(support_score, dtype=np.float32).reshape(-1)
    hit_count = np.asarray(hit_count, dtype=np.float32).reshape(-1)
    opacity_sigmoid = np.asarray(opacity_sigmoid, dtype=np.float32).reshape(-1)
    sampled_indices = np.asarray(sampled_indices, dtype=np.int32).reshape(-1)
    if sampled_indices.size == 0:
        raise ValueError("sampled_indices must not be empty")

    features = _query_features(bank=bank, sampled_indices=sampled_indices, opacity_sigmoid=opacity_sigmoid)
    min_hits = max(1, int(np.ceil(0.30 * sampled_indices.size)))
    seed_ids = _seed_ids(
        support_score=support_score,
        hit_count=hit_count,
        opacity_sigmoid=opacity_sigmoid,
        active_ratio=features["active_ratio"],
        min_hits=min_hits,
        seed_ratio=seed_ratio,
    )
    if seed_ids.size == 0:
        raise ValueError("No valid worldtube seeds were found for consistency clustering")

    support_norm = support_score / max(float(np.max(support_score)), 1.0e-6)
    seed_weights = support_norm[seed_ids] * np.clip(opacity_sigmoid[seed_ids], 1.0e-6, 1.0)
    seed_feature_center = _weighted_center(features["features"][seed_ids], seed_weights)
    feature_similarity = ((features["features"] @ seed_feature_center) + 1.0) * 0.5
    proximity_score = _trajectory_proximity(
        query_traj=features["query_traj"],
        scale_norm=features["scale_norm"],
        seed_ids=seed_ids,
        seed_weights=seed_weights,
    )
    opacity_weight = np.clip(opacity_sigmoid, 1.0e-6, 1.0).astype(np.float32)
    overlap_score = np.clip(0.6 * features["active_ratio"] + 0.4 * features["mean_gate"], 0.0, 1.0).astype(np.float32)

    ranking_score = (
        0.28 * support_norm
        + 0.22 * feature_similarity.astype(np.float32)
        + 0.20 * proximity_score
        + 0.15 * overlap_score
        + 0.15 * opacity_weight
    ).astype(np.float32)

    seed_feature_floor = float(np.quantile(feature_similarity[seed_ids], 0.15)) if seed_ids.size > 0 else 0.0
    seed_proximity_floor = float(np.quantile(proximity_score[seed_ids], 0.10)) if seed_ids.size > 0 else 0.0
    seed_opacity_floor = float(np.quantile(opacity_sigmoid[seed_ids], 0.10)) if seed_ids.size > 0 else 0.0
    candidate_mask = (
        (overlap_score >= 0.05)
        & (opacity_sigmoid >= max(seed_opacity_floor * 0.35, 0.05))
        & (
            (support_score > 0.0)
            | (feature_similarity >= max(seed_feature_floor - 0.05, 0.35))
            | (proximity_score >= max(seed_proximity_floor - 0.08, 0.45))
        )
    )
    candidate_ids = np.where(candidate_mask)[0]
    if candidate_ids.size == 0:
        candidate_ids = np.argsort(-ranking_score, kind="mergesort")[: max(min_gaussians, seed_ids.size * 2)]

    keep_count = int(
        np.clip(
            max(
                round(float(candidate_ids.size) * float(max(proposal_keep_ratio, 0.04))),
                round(float(seed_ids.size) * float(expansion_factor)),
            ),
            int(min_gaussians),
            int(max_gaussians),
        )
    )
    keep_count = min(keep_count, int(candidate_ids.size))
    ranked_candidates = candidate_ids[np.argsort(-ranking_score[candidate_ids], kind="mergesort")]
    selected = ranked_candidates[:keep_count].astype(np.int64)
    selected_scores = ranking_score[selected].astype(np.float32)
    selected_scores /= max(float(selected_scores.max()), 1.0e-6)

    summary = {
        "seed_count": int(seed_ids.size),
        "candidate_count": int(candidate_ids.size),
        "selected_count": int(selected.size),
        "mean_support_score": float(np.mean(support_score[selected])) if selected.size else 0.0,
        "mean_ranking_score": float(np.mean(ranking_score[selected])) if selected.size else 0.0,
        "mean_opacity_sigmoid": float(np.mean(opacity_sigmoid[selected])) if selected.size else 0.0,
        "mean_feature_similarity": float(np.mean(feature_similarity[selected])) if selected.size else 0.0,
        "mean_proximity_score": float(np.mean(proximity_score[selected])) if selected.size else 0.0,
        "mean_overlap_score": float(np.mean(overlap_score[selected])) if selected.size else 0.0,
    }
    return ConsistencySelection(
        selected_ids=selected,
        selected_scores=selected_scores,
        summary=summary,
    )
