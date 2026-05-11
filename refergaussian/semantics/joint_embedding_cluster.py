from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .source_images import resolve_dataset_image_entries
from .query_proposal_bridge import (
    Camera,
    _bbox_area,
    _dilate_mask,
    _load_bank,
    _load_mask,
    _phase_aware_track_variants,
    _phrase_entity_type,
    _phrase_world_payload,
    _read_json,
    _resample_indices,
    _write_json,
)
from .worldtube_consistency import load_opacity_sigmoid


@dataclass
class VariantSupport:
    alias: str
    base_phrase: str
    phase: str
    variant_kind: str
    description: str
    entity_type: str
    sampled_frames: list[dict[str, Any]]
    sampled_indices: np.ndarray
    positive_score: np.ndarray
    negative_score: np.ndarray
    hit_count: np.ndarray
    support_score: np.ndarray
    mean_mask_area: float


def _support_stats(gate: np.ndarray, time_values: np.ndarray) -> dict[str, np.ndarray]:
    gate = np.asarray(gate, dtype=np.float32)
    time_values = np.asarray(time_values, dtype=np.float32).reshape(1, -1)
    gate_sum = np.clip(gate.sum(axis=1), 1.0e-6, None)
    gate_peak = gate.max(axis=1)
    active = gate >= np.maximum(0.18, 0.35 * gate_peak[:, None])
    support_center = (gate * time_values).sum(axis=1) / gate_sum
    active_ratio = active.mean(axis=1).astype(np.float32)
    mean_gate = gate.mean(axis=1).astype(np.float32)
    return {
        "support_center": support_center.astype(np.float32),
        "active_ratio": active_ratio,
        "mean_gate": mean_gate,
    }


def _build_feature_matrix(bank: dict[str, np.ndarray], opacity_sigmoid: np.ndarray, appearance_features: np.ndarray | None = None) -> np.ndarray:
    trajectories = np.asarray(bank["trajectories"], dtype=np.float32)
    gate = np.asarray(bank["gate"], dtype=np.float32).reshape(trajectories.shape[0], trajectories.shape[1])
    support = _support_stats(gate=gate, time_values=np.asarray(bank["time_values"], dtype=np.float32))
    xyz = np.asarray(bank.get("xyz", trajectories[:, 0, :]), dtype=np.float32)
    displacement = np.asarray(bank["displacement"], dtype=np.float32)
    velocity = np.asarray(bank.get("velocity", np.zeros_like(displacement)), dtype=np.float32)
    acceleration = np.asarray(bank.get("acceleration", np.zeros_like(displacement)), dtype=np.float32)
    spatial_scale = np.asarray(bank.get("spatial_scale", np.ones_like(displacement)), dtype=np.float32)
    anchor = np.asarray(bank.get("anchor", np.zeros((trajectories.shape[0], 1), dtype=np.float32)), dtype=np.float32)
    scale = np.asarray(bank.get("scale", np.ones((trajectories.shape[0], 1), dtype=np.float32)), dtype=np.float32)
    motion_score = np.asarray(bank.get("motion_score", np.zeros((trajectories.shape[0],), dtype=np.float32)), dtype=np.float32)[:, None]
    path_length = np.asarray(bank.get("path_length", np.zeros((trajectories.shape[0],), dtype=np.float32)), dtype=np.float32)[:, None]
    occupancy = np.asarray(bank.get("occupancy_mass", np.ones((trajectories.shape[0],), dtype=np.float32)), dtype=np.float32)[:, None]
    visibility_proxy = np.asarray(bank.get("visibility_proxy", np.ones((trajectories.shape[0], 1), dtype=np.float32)), dtype=np.float32)
    effective_support = np.asarray(bank.get("effective_support", np.ones((trajectories.shape[0], 1), dtype=np.float32)), dtype=np.float32)

    feature_parts = [
        xyz,
        displacement,
        velocity,
        acceleration,
        spatial_scale,
        anchor,
        scale,
        motion_score,
        path_length,
        occupancy,
        visibility_proxy,
        effective_support,
        support["support_center"][:, None],
        support["active_ratio"][:, None],
        support["mean_gate"][:, None],
        opacity_sigmoid[:, None],
    ]
    if appearance_features is not None:
        feature_parts.append(np.asarray(appearance_features, dtype=np.float32))

    features = np.concatenate(
        feature_parts,
        axis=1,
    ).astype(np.float32)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return ((features - mean) / np.clip(std, 1.0e-6, None)).astype(np.float32)


def _appearance_features(
    dataset_dir: Path,
    bank: dict[str, np.ndarray],
    sampled_frames: list[dict[str, Any]],
    chunk_size: int,
) -> np.ndarray:
    trajectories = np.asarray(bank["trajectories"], dtype=np.float32)
    gate = np.asarray(bank["gate"], dtype=np.float32).reshape(trajectories.shape[0], trajectories.shape[1])
    time_values = np.asarray(bank["time_values"], dtype=np.float32).reshape(-1)
    num_gaussians = trajectories.shape[0]
    if not sampled_frames:
        return np.zeros((num_gaussians, 6), dtype=np.float32)

    image_entries = resolve_dataset_image_entries(dataset_dir)
    image_path_by_id = {str(item["image_id"]): str(item["image_path"]) for item in image_entries}
    unique_frames: dict[str, dict[str, Any]] = {}
    for frame in sampled_frames:
        unique_frames[str(frame["image_id"])] = frame

    rgb_sum = np.zeros((num_gaussians, 3), dtype=np.float32)
    rgb_sq_sum = np.zeros((num_gaussians, 3), dtype=np.float32)
    weight_sum = np.zeros((num_gaussians,), dtype=np.float32)

    for image_id, frame in unique_frames.items():
        image_path = image_path_by_id.get(str(image_id))
        if image_path is None:
            continue
        bank_index = int(_resample_indices(source_times=time_values, target_times=np.asarray([float(frame["time_value"])], dtype=np.float32))[0])
        image = np.asarray(torchvision_read_image(image_path), dtype=np.float32) / 255.0
        image_h, image_w = image.shape[:2]
        camera = Camera.from_json(dataset_dir / "camera" / f"{image_id}.json")

        for start in range(0, num_gaussians, chunk_size):
            end = min(start + chunk_size, num_gaussians)
            points = trajectories[start:end, bank_index, :]
            local_points = camera.points_to_local_points(points)
            depth_valid = local_points[:, 2] > 1.0e-4
            if not np.any(depth_valid):
                continue
            pixels = camera.project(points)
            scale_x = float(image_w) / max(float(np.asarray(camera.image_size)[0]), 1.0)
            scale_y = float(image_h) / max(float(np.asarray(camera.image_size)[1]), 1.0)
            xs = np.clip(np.round(pixels[:, 0] * scale_x).astype(np.int64), 0, image_w - 1)
            ys = np.clip(np.round(pixels[:, 1] * scale_y).astype(np.int64), 0, image_h - 1)
            colors = image[ys, xs]
            weights = gate[start:end, bank_index] * depth_valid.astype(np.float32)
            rgb_sum[start:end] += colors * weights[:, None]
            rgb_sq_sum[start:end] += np.square(colors) * weights[:, None]
            weight_sum[start:end] += weights

    valid = weight_sum > 1.0e-6
    mean_rgb = np.zeros((num_gaussians, 3), dtype=np.float32)
    std_rgb = np.zeros((num_gaussians, 3), dtype=np.float32)
    if np.any(valid):
        mean_rgb[valid] = rgb_sum[valid] / weight_sum[valid, None]
        variance = np.clip(rgb_sq_sum[valid] / weight_sum[valid, None] - np.square(mean_rgb[valid]), 0.0, None)
        std_rgb[valid] = np.sqrt(variance)
    return np.concatenate([mean_rgb, std_rgb], axis=1).astype(np.float32)


def torchvision_read_image(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _collect_variant_support(
    variant: dict[str, Any],
    dataset_dir: Path,
    bank: dict[str, np.ndarray],
    chunk_size: int,
) -> VariantSupport:
    trajectories = np.asarray(bank["trajectories"], dtype=np.float32)
    gate = np.asarray(bank["gate"], dtype=np.float32).reshape(trajectories.shape[0], trajectories.shape[1])
    time_values = np.asarray(bank["time_values"], dtype=np.float32).reshape(-1)
    num_gaussians = trajectories.shape[0]
    sampled_frames = sorted(list(variant["frames"]), key=lambda item: int(item["frame_index"]))
    sampled_times = np.asarray([float(frame["time_value"]) for frame in sampled_frames], dtype=np.float32)
    sampled_indices = _resample_indices(source_times=time_values, target_times=sampled_times)

    positive_score_accum = np.zeros((num_gaussians,), dtype=np.float32)
    negative_score_accum = np.zeros((num_gaussians,), dtype=np.float32)
    hit_count = np.zeros((num_gaussians,), dtype=np.float32)
    areas = np.asarray([_bbox_area(frame["bbox_xyxy"]) for frame in sampled_frames], dtype=np.float32)
    entity_type = _phrase_entity_type(variant["alias"])

    for sampled_frame, bank_index in zip(sampled_frames, sampled_indices.tolist()):
        camera = Camera.from_json(dataset_dir / "camera" / f"{sampled_frame['image_id']}.json")
        left, top, right, bottom = [float(value) for value in sampled_frame["bbox_xyxy"]]
        mask = np.asarray(sampled_frame.get("mask_array"), dtype=bool) if sampled_frame.get("mask_array") is not None else _load_mask(sampled_frame.get("mask_path"))
        positive_context_mask = _dilate_mask(mask, radius=2) if mask is not None else None
        negative_ring_mask = None
        if mask is not None:
            outer_mask = _dilate_mask(mask, radius=7)
            negative_ring_mask = np.logical_and(outer_mask, np.logical_not(positive_context_mask))

        for start in range(0, num_gaussians, chunk_size):
            end = min(start + chunk_size, num_gaussians)
            points = trajectories[start:end, int(bank_index), :]
            local_points = camera.points_to_local_points(points)
            depth_valid = local_points[:, 2] > 1.0e-4
            if not np.any(depth_valid):
                continue
            pixels = camera.project(points)
            depth_values = np.asarray(local_points[:, 2], dtype=np.float32)
            valid_depth_values = depth_values[depth_valid]
            near_depth = float(np.quantile(valid_depth_values, 0.05))
            far_depth = float(np.quantile(valid_depth_values, 0.90))
            depth_span = max(far_depth - near_depth, 1.0e-4)
            depth_weight = np.exp(-np.clip((depth_values - near_depth) / depth_span, 0.0, 8.0)).astype(np.float32)

            if mask is not None:
                mask_h, mask_w = mask.shape[:2]
                image_width = float(np.asarray(camera.image_size)[0])
                image_height = float(np.asarray(camera.image_size)[1])
                scale_x = float(mask_w) / max(image_width, 1.0)
                scale_y = float(mask_h) / max(image_height, 1.0)
                xs = np.clip(np.round(pixels[:, 0] * scale_x).astype(np.int64), 0, mask_w - 1)
                ys = np.clip(np.round(pixels[:, 1] * scale_y).astype(np.int64), 0, mask_h - 1)

                inside_mask = depth_valid & mask[ys, xs]
                context_mask = depth_valid & positive_context_mask[ys, xs] if positive_context_mask is not None else inside_mask
                ring_mask = depth_valid & negative_ring_mask[ys, xs] if negative_ring_mask is not None else np.zeros_like(inside_mask)
                front_inside_mask = inside_mask.copy()
                back_inside_mask = np.zeros_like(inside_mask)
                if entity_type == "object" and np.any(inside_mask):
                    inside_depth = depth_values[inside_mask]
                    front_depth = float(np.quantile(inside_depth, 0.45))
                    front_inside_mask = inside_mask & (depth_values <= front_depth + 0.05 * depth_span)
                    back_inside_mask = np.logical_and(inside_mask, np.logical_not(front_inside_mask))
                    context_mask = context_mask & (depth_values <= front_depth + 0.15 * depth_span)

                gate_weight = gate[start:end, int(bank_index)]
                positive_frame_score = gate_weight * depth_weight * (
                    1.10 * front_inside_mask.astype(np.float32)
                    + 0.35 * inside_mask.astype(np.float32)
                    + 0.10 * context_mask.astype(np.float32)
                )
                negative_frame_score = gate_weight * depth_weight * (
                    1.00 * ring_mask.astype(np.float32) + 0.35 * back_inside_mask.astype(np.float32)
                )
                positive_score_accum[start:end] += positive_frame_score
                negative_score_accum[start:end] += negative_frame_score
                hit_count[start:end] += inside_mask.astype(np.float32)
                continue

            inside_bbox = (
                depth_valid
                & (pixels[:, 0] >= left)
                & (pixels[:, 0] <= right)
                & (pixels[:, 1] >= top)
                & (pixels[:, 1] <= bottom)
            )
            gate_weight = gate[start:end, int(bank_index)]
            positive_frame_score = gate_weight * depth_weight * inside_bbox.astype(np.float32)
            positive_score_accum[start:end] += positive_frame_score
            hit_count[start:end] += inside_bbox.astype(np.float32)
    num_sampled = max(int(len(sampled_frames)), 1)
    hit_ratio = hit_count / float(num_sampled)
    positive_score = positive_score_accum / float(num_sampled)
    negative_score = negative_score_accum / float(num_sampled)
    contrastive_margin = positive_score - 0.70 * negative_score
    exclusive_ratio = positive_score / np.clip(positive_score + negative_score, 1.0e-6, None)
    support_score = (
        0.48 * positive_score
        + 0.22 * hit_ratio
        + 0.18 * np.clip(contrastive_margin, 0.0, None)
        + 0.12 * exclusive_ratio
    ).astype(np.float32)

    return VariantSupport(
        alias=str(variant["alias"]),
        base_phrase=str(variant["base_phrase"]),
        phase=str(variant["phase"]),
        variant_kind=str(variant["variant_kind"]),
        description=str(variant["description"]),
        entity_type=str(entity_type),
        sampled_frames=sampled_frames,
        sampled_indices=np.asarray(sampled_indices, dtype=np.int32),
        positive_score=positive_score.astype(np.float32),
        negative_score=negative_score.astype(np.float32),
        hit_count=hit_count.astype(np.float32),
        support_score=support_score.astype(np.float32),
        mean_mask_area=float(np.mean(areas)) if areas.size else 0.0,
    )


class JointEmbeddingModel(torch.nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, num_variants: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, embed_dim),
        )
        self.prototypes = torch.nn.Parameter(torch.randn(num_variants, embed_dim))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = F.normalize(self.net(features), dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        logits = embedding @ prototypes.T
        return embedding, logits


def _soft_targets(positive_scores: np.ndarray, negative_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(positive_scores, dtype=np.float32)
    neg = np.asarray(negative_scores, dtype=np.float32)
    other = np.clip(pos.sum(axis=1, keepdims=True) - pos, 0.0, None)
    target = pos / np.clip(pos + neg + other, 1.0e-6, None)
    weight = pos + neg + 0.5 * other
    return target.astype(np.float32), weight.astype(np.float32)


def _train_joint_embedding(
    features: np.ndarray,
    support_matrix: np.ndarray,
    negative_matrix: np.ndarray,
    num_steps: int = 400,
    lr: float = 1.0e-2,
    embed_dim: int = 16,
    device: str = "cuda",
) -> tuple[JointEmbeddingModel, np.ndarray]:
    target, weight = _soft_targets(support_matrix, negative_matrix)
    supervised = np.where(weight.max(axis=1) > 1.0e-4)[0]
    if supervised.size == 0:
        raise ValueError("No supervised Gaussian rows for joint embedding training")

    x_all = torch.from_numpy(features.astype(np.float32)).to(device)
    x_sup = x_all[torch.from_numpy(supervised).to(device)]
    y_sup = torch.from_numpy(target[supervised]).to(device)
    w_sup = torch.from_numpy(weight[supervised]).to(device)

    model = JointEmbeddingModel(in_dim=features.shape[1], embed_dim=embed_dim, num_variants=support_matrix.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    seed_rows = []
    seed_labels = []
    for variant_index in range(support_matrix.shape[1]):
        scores = support_matrix[:, variant_index]
        candidate_ids = np.where(scores > 0.0)[0]
        if candidate_ids.size == 0:
            continue
        ranked = candidate_ids[np.argsort(-scores[candidate_ids], kind="mergesort")]
        chosen = ranked[: min(512, ranked.size)]
        seed_rows.append(chosen)
        seed_labels.append(np.full((chosen.size,), variant_index, dtype=np.int64))
    if seed_rows:
        seed_rows_np = np.concatenate(seed_rows, axis=0)
        seed_labels_np = np.concatenate(seed_labels, axis=0)
        x_seed = x_all[torch.from_numpy(seed_rows_np).to(device)]
        y_seed = torch.from_numpy(seed_labels_np).to(device)
    else:
        x_seed = None
        y_seed = None

    for _step in range(int(num_steps)):
        optimizer.zero_grad(set_to_none=True)
        _, logits_sup = model(x_sup)
        bce_loss = F.binary_cross_entropy_with_logits(logits_sup, y_sup, weight=torch.clamp(w_sup, 0.0, 5.0))

        loss = bce_loss
        if x_seed is not None and y_seed is not None:
            _, logits_seed = model(x_seed)
            seed_loss = F.cross_entropy(logits_seed * 6.0, y_seed)
            loss = loss + 0.45 * seed_loss

        proto = F.normalize(model.prototypes, dim=-1)
        proto_sim = proto @ proto.T
        eye = torch.eye(proto.shape[0], device=proto.device)
        separation_loss = ((proto_sim - eye) ** 2).mean()
        loss = loss + 0.05 * separation_loss
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, logits_all = model(x_all)
        probs = torch.sigmoid(logits_all).detach().cpu().numpy().astype(np.float32)
    return model, probs


def _select_variant_gaussians(
    variant_index: int,
    support_scores: np.ndarray,
    negative_scores: np.ndarray,
    probabilities: np.ndarray,
    opacity_sigmoid: np.ndarray,
    proposal_keep_ratio: float,
    min_gaussians: int,
    max_gaussians: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    support = np.asarray(support_scores[:, variant_index], dtype=np.float32)
    neg = np.asarray(negative_scores[:, variant_index], dtype=np.float32)
    prob = np.asarray(probabilities[:, variant_index], dtype=np.float32)
    other = np.max(np.delete(probabilities, variant_index, axis=1), axis=1) if probabilities.shape[1] > 1 else np.zeros_like(prob)
    support_norm = support / max(float(np.max(support)), 1.0e-6)
    contrast = np.clip(prob - other, 0.0, 1.0)
    exclusive = support / np.clip(support + neg, 1.0e-6, None)
    opacity_weight = np.sqrt(np.clip(opacity_sigmoid, 1.0e-6, 1.0)).astype(np.float32)
    ranking = (
        0.35 * prob
        + 0.25 * contrast
        + 0.20 * support_norm
        + 0.10 * exclusive
        + 0.10 * opacity_weight
    ).astype(np.float32)
    eligible = np.where(
        (support > 0.0)
        | (prob >= float(np.quantile(prob, 0.97)))
        | (contrast >= 0.10)
    )[0]
    if eligible.size == 0:
        eligible = np.argsort(-ranking, kind="mergesort")[: max(min_gaussians, 512)]
    keep_count = int(
        np.clip(
            round(float(eligible.size) * float(proposal_keep_ratio)),
            int(min_gaussians),
            int(max_gaussians),
        )
    )
    keep_count = min(keep_count, int(eligible.size))
    ranked = eligible[np.argsort(-ranking[eligible], kind="mergesort")]
    selected = ranked[:keep_count].astype(np.int64)
    selected_scores = ranking[selected].astype(np.float32)
    selected_scores /= max(float(selected_scores.max()), 1.0e-6)
    summary = {
        "selected_count": int(selected.size),
        "mean_support_score": float(np.mean(support[selected])) if selected.size else 0.0,
        "mean_ranking_score": float(np.mean(ranking[selected])) if selected.size else 0.0,
        "mean_opacity_sigmoid": float(np.mean(opacity_sigmoid[selected])) if selected.size else 0.0,
        "mean_probability": float(np.mean(prob[selected])) if selected.size else 0.0,
        "mean_contrast": float(np.mean(contrast[selected])) if selected.size else 0.0,
        "mean_exclusive_ratio": float(np.mean(exclusive[selected])) if selected.size else 0.0,
        "candidate_count": int(eligible.size),
    }
    return selected, selected_scores, summary


def build_joint_query_proposal_dir(
    run_dir: str | Path,
    dataset_dir: str | Path,
    tracks_path: str | Path,
    output_dir: str | Path,
    max_track_frames: int = 16,
    proposal_keep_ratio: float = 0.10,
    min_gaussians: int = 2048,
    max_gaussians: int = 4096,
    chunk_size: int = 4096,
    embed_dim: int = 16,
    num_steps: int = 400,
    lr: float = 1.0e-2,
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
    opacity_sigmoid = load_opacity_sigmoid(run_dir)

    variants: list[dict[str, Any]] = []
    supports: list[VariantSupport] = []
    for track in tracks:
        base_phrase = str(track["phrase"])
        for variant in _phase_aware_track_variants(base_phrase, track, max_track_frames=max_track_frames, query_plan=query_plan):
            variants.append(variant)
            supports.append(
                _collect_variant_support(
                    variant=variant,
                    dataset_dir=dataset_dir,
                    bank=bank,
                    chunk_size=chunk_size,
                )
            )
    if not supports:
        raise ValueError("No valid phrase variants collected for joint proposal directory")

    all_sampled_frames = []
    seen_pairs: set[tuple[str, int]] = set()
    for support in supports:
        for frame in support.sampled_frames:
            key = (str(frame["image_id"]), int(frame["frame_index"]))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            all_sampled_frames.append(frame)
    appearance_features = _appearance_features(
        dataset_dir=dataset_dir,
        bank=bank,
        sampled_frames=all_sampled_frames,
        chunk_size=chunk_size,
    )
    feature_matrix = _build_feature_matrix(
        bank=bank,
        opacity_sigmoid=opacity_sigmoid,
        appearance_features=appearance_features,
    )

    support_matrix = np.stack([item.support_score for item in supports], axis=1).astype(np.float32)
    negative_matrix = np.stack([item.negative_score for item in supports], axis=1).astype(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model, probabilities = _train_joint_embedding(
        features=feature_matrix,
        support_matrix=support_matrix,
        negative_matrix=negative_matrix,
        num_steps=num_steps,
        lr=lr,
        embed_dim=embed_dim,
        device=device,
    )

    entities_json_rows: list[dict[str, Any]] = []
    phrase_rows: list[dict[str, Any]] = []
    phrase_payloads: list[dict[str, Any]] = []
    time_values = np.asarray(bank["time_values"], dtype=np.float32).reshape(-1)

    for variant_index, (variant, support) in enumerate(zip(variants, supports)):
        selected_ids, selected_scores, summary = _select_variant_gaussians(
            variant_index=variant_index,
            support_scores=support_matrix,
            negative_scores=negative_matrix,
            probabilities=probabilities,
            opacity_sigmoid=opacity_sigmoid,
            proposal_keep_ratio=proposal_keep_ratio,
            min_gaussians=min_gaussians,
            max_gaussians=max_gaussians,
        )
        phrase_payload = {
            "phrase": str(variant["alias"]),
            "entity_type": support.entity_type,
            "selected_gaussian_ids": selected_ids,
            "selected_scores": selected_scores,
            "sampled_frames": support.sampled_frames,
            "sampled_indices": support.sampled_indices,
            "mean_mask_area": support.mean_mask_area,
            "mean_hit_ratio": float(np.mean(support.hit_count[selected_ids] / max(len(support.sampled_frames), 1))) if selected_ids.size else 0.0,
            "mean_support_score": summary["mean_support_score"],
            "mean_ranking_score": summary["mean_ranking_score"],
            "mean_opacity_sigmoid": summary["mean_opacity_sigmoid"],
            "mean_probability": summary["mean_probability"],
            "mean_contrast": summary["mean_contrast"],
            "mean_exclusive_ratio": summary["mean_exclusive_ratio"],
            "cluster_mode": "joint_worldtube_embedding",
            "candidate_count": summary["candidate_count"],
            "keyframes": sorted(set(int(index) for index in support.sampled_indices.tolist()))[:8],
        }
        world_payload = _phrase_world_payload(phrase_payload, bank=bank)
        phrase_payloads.append({"variant": variant, "support": support, "selection": phrase_payload, "world": world_payload})

    num_entities = len(phrase_payloads)
    centroid_world = np.zeros((num_entities, time_values.shape[0], 3), dtype=np.float32)
    centroid_world_valid = np.zeros((num_entities, time_values.shape[0]), dtype=bool)
    bbox_world = np.zeros((num_entities, time_values.shape[0], 6), dtype=np.float32)
    bbox_world_valid = np.zeros((num_entities, time_values.shape[0]), dtype=bool)
    visibility = np.zeros((num_entities, time_values.shape[0]), dtype=bool)
    mask_area = np.zeros((num_entities, time_values.shape[0]), dtype=np.float32)
    quality = np.zeros((num_entities,), dtype=np.float32)

    for entity_id, payload in enumerate(phrase_payloads):
        variant = payload["variant"]
        selection = payload["selection"]
        world = payload["world"]
        support = payload["support"]
        centroid_world[entity_id] = world["center_world"]
        centroid_world_valid[entity_id] = world["center_valid"]
        bbox_world[entity_id] = world["bbox_world"]
        bbox_world_valid[entity_id] = world["bbox_valid"]
        visibility[entity_id] = world["visibility"]
        mask_area[entity_id] = world["mask_area"]
        quality[entity_id] = float(world["quality"])

        phrase_rows.append(
            {
                "id": int(entity_id),
                "phrase": support.base_phrase,
                "proposal_alias": support.alias,
                "phase": support.phase,
                "variant_kind": support.variant_kind,
                "entity_type": support.entity_type,
                "selected_gaussian_count": int(len(selection["selected_gaussian_ids"])),
                "quality": float(world["quality"]),
                "visibility_ratio": float(world["visibility_ratio"]),
                "mean_mask_area": float(np.mean(world["mask_area"][world["mask_area"] > 0.0])) if np.any(world["mask_area"] > 0.0) else 0.0,
                "mean_hit_ratio": float(selection["mean_hit_ratio"]),
                "mean_support_score": float(selection["mean_support_score"]),
                "mean_ranking_score": float(selection["mean_ranking_score"]),
                "mean_opacity_sigmoid": selection["mean_opacity_sigmoid"],
                "mean_probability": selection["mean_probability"],
                "mean_contrast": selection["mean_contrast"],
                "mean_exclusive_ratio": selection["mean_exclusive_ratio"],
                "candidate_count": selection["candidate_count"],
                "cluster_mode": selection["cluster_mode"],
                "keyframes": world["keyframes"],
                "segments": world["segments"],
            }
        )
        entities_json_rows.append(
            {
                "id": int(entity_id),
                "static_text": support.base_phrase,
                "proposal_alias": support.alias,
                "proposal_phase": support.phase,
                "proposal_variant": support.variant_kind,
                "global_desc": support.description,
                "dyn_desc": [support.description],
                "gaussian_ids": selection["selected_gaussian_ids"].astype(int).tolist(),
                "gaussian_scores": selection["selected_scores"].astype(float).tolist(),
                "visibility_ratio": float(world["visibility_ratio"]),
                "mean_mask_area": float(np.mean(world["mask_area"][world["mask_area"] > 0.0])) if np.any(world["mask_area"] > 0.0) else 0.0,
                "quality": float(world["quality"]),
                "entity_type": support.entity_type,
                "role_hints": [],
                "keyframes": world["keyframes"],
                "segments": world["segments"],
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
                "embed_dim": int(embed_dim),
                "num_steps": int(num_steps),
                "lr": float(lr),
                "cluster_mode": "joint_worldtube_embedding",
            },
        },
    )
    return output_dir
