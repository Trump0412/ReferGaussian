from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .surface_mask_field import SurfaceMaskField


@dataclass
class PreparedSemanticFrame:
    frame_index: int
    time_value: float
    image_id: str
    width: int
    height: int
    image_scale: float
    gaussian_ids: torch.Tensor
    centers_xy: torch.Tensor
    sigma_px: torch.Tensor
    alpha_weight: torch.Tensor
    depth: torch.Tensor


def _camera_image_size(camera: Any) -> tuple[int, int]:
    raw = np.asarray(getattr(camera, "image_size"), dtype=np.float32).reshape(-1)
    if raw.size < 2:
        raise ValueError("camera.image_size must contain width and height")
    width = int(max(round(float(raw[0])), 1))
    height = int(max(round(float(raw[1])), 1))
    return width, height


def _opacity_sigmoid(values: np.ndarray | torch.Tensor) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    return 1.0 / (1.0 + np.exp(-array))


def _estimate_sigma_pixels(
    camera: Any,
    points: np.ndarray,
    spatial_scale: np.ndarray,
    centers_xy: np.ndarray,
    image_scale: float,
) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    deltas = [
        np.stack([spatial_scale[:, 0], np.zeros((points.shape[0],), dtype=np.float32), np.zeros((points.shape[0],), dtype=np.float32)], axis=1),
        np.stack([np.zeros((points.shape[0],), dtype=np.float32), spatial_scale[:, 1], np.zeros((points.shape[0],), dtype=np.float32)], axis=1),
        np.stack([np.zeros((points.shape[0],), dtype=np.float32), np.zeros((points.shape[0],), dtype=np.float32), spatial_scale[:, 2]], axis=1),
    ]
    dists: list[np.ndarray] = []
    for delta in deltas:
        projected = np.asarray(camera.project(points + delta), dtype=np.float32)
        projected = projected * float(image_scale)
        dists.append(np.linalg.norm(projected - centers_xy, axis=1).astype(np.float32))
    sigma = np.mean(np.stack(dists, axis=1), axis=1).astype(np.float32)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    return np.clip(sigma * 0.85, 0.75, 16.0).astype(np.float32)


def prepare_semantic_frame_inputs(
    camera: Any,
    frame_index: int,
    image_id: str,
    time_value: float,
    points: np.ndarray,
    spatial_scale: np.ndarray,
    opacity: np.ndarray,
    visibility_gate: np.ndarray,
    image_scale: float = 1.0,
    max_gaussians: int = 16000,
    gate_threshold: float = 0.01,
    device: str | torch.device = "cpu",
) -> PreparedSemanticFrame:
    width_full, height_full = _camera_image_size(camera)
    width = max(1, int(round(width_full * float(image_scale))))
    height = max(1, int(round(height_full * float(image_scale))))

    local_points = np.asarray(camera.points_to_local_points(points), dtype=np.float32)
    depth = local_points[:, 2].astype(np.float32)
    depth_valid = depth > 1.0e-4
    gate = np.asarray(visibility_gate, dtype=np.float32).reshape(-1)
    opacity_sigmoid = _opacity_sigmoid(opacity)
    alpha_weight = (gate * opacity_sigmoid).astype(np.float32)
    projected = np.asarray(camera.project(points), dtype=np.float32)
    centers_xy = projected * float(image_scale)
    in_image = (
        (centers_xy[:, 0] >= -8.0)
        & (centers_xy[:, 0] <= float(width) + 8.0)
        & (centers_xy[:, 1] >= -8.0)
        & (centers_xy[:, 1] <= float(height) + 8.0)
    )
    valid = depth_valid & in_image & (gate >= float(gate_threshold)) & np.isfinite(alpha_weight)
    valid_ids = np.where(valid)[0]
    if valid_ids.size == 0:
        return PreparedSemanticFrame(
            frame_index=int(frame_index),
            time_value=float(time_value),
            image_id=str(image_id),
            width=int(width),
            height=int(height),
            image_scale=float(image_scale),
            gaussian_ids=torch.empty((0,), dtype=torch.long, device=device),
            centers_xy=torch.empty((0, 2), dtype=torch.float32, device=device),
            sigma_px=torch.empty((0,), dtype=torch.float32, device=device),
            alpha_weight=torch.empty((0,), dtype=torch.float32, device=device),
            depth=torch.empty((0,), dtype=torch.float32, device=device),
        )

    if valid_ids.size > int(max_gaussians):
        support = alpha_weight[valid_ids] / np.clip(depth[valid_ids], 1.0e-3, None)
        order = np.argsort(-support, kind="mergesort")
        valid_ids = valid_ids[order[: int(max_gaussians)]]

    sigma_px = _estimate_sigma_pixels(
        camera=camera,
        points=np.asarray(points[valid_ids], dtype=np.float32),
        spatial_scale=np.asarray(spatial_scale[valid_ids], dtype=np.float32),
        centers_xy=np.asarray(centers_xy[valid_ids], dtype=np.float32),
        image_scale=float(image_scale),
    )
    return PreparedSemanticFrame(
        frame_index=int(frame_index),
        time_value=float(time_value),
        image_id=str(image_id),
        width=int(width),
        height=int(height),
        image_scale=float(image_scale),
        gaussian_ids=torch.as_tensor(valid_ids, dtype=torch.long, device=device),
        centers_xy=torch.as_tensor(centers_xy[valid_ids], dtype=torch.float32, device=device),
        sigma_px=torch.as_tensor(sigma_px, dtype=torch.float32, device=device),
        alpha_weight=torch.as_tensor(alpha_weight[valid_ids], dtype=torch.float32, device=device),
        depth=torch.as_tensor(depth[valid_ids], dtype=torch.float32, device=device),
    )


def _splat_channel_values(
    prepared: PreparedSemanticFrame,
    channel_values: torch.Tensor,
    normalize: bool = True,
    chunk_size: int = 2048,
    max_splat_radius: int = 18,
) -> tuple[torch.Tensor, torch.Tensor]:
    if channel_values.ndim == 1:
        channel_values = channel_values[:, None]
    if channel_values.shape[0] != prepared.gaussian_ids.shape[0]:
        raise ValueError("channel_values row count must match prepared.gaussian_ids")

    device = channel_values.device
    width = int(prepared.width)
    height = int(prepared.height)
    num_channels = int(channel_values.shape[1])
    alpha_accum = torch.zeros((height * width,), dtype=torch.float32, device=device)
    channel_accum = torch.zeros((num_channels, height * width), dtype=torch.float32, device=device)
    centers = prepared.centers_xy.to(device=device, dtype=torch.float32)
    sigma = prepared.sigma_px.to(device=device, dtype=torch.float32).clamp_min(0.75)
    alpha_weight = prepared.alpha_weight.to(device=device, dtype=torch.float32).clamp_min(0.0)

    if centers.shape[0] == 0:
        prob_map = torch.zeros((num_channels, height, width), dtype=torch.float32, device=device)
        alpha_map = torch.zeros((1, height, width), dtype=torch.float32, device=device)
        return prob_map, alpha_map

    radius_cap = max(int(max_splat_radius), 1)
    for start in range(0, centers.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), centers.shape[0])
        center_chunk = centers[start:end]
        sigma_chunk = sigma[start:end]
        alpha_chunk = alpha_weight[start:end]
        value_chunk = channel_values[start:end]

        radius_chunk = torch.ceil(sigma_chunk * 3.0).to(dtype=torch.long).clamp(min=1, max=radius_cap)
        patch_radius = int(radius_chunk.max().item())
        grid_y, grid_x = torch.meshgrid(
            torch.arange(-patch_radius, patch_radius + 1, device=device, dtype=torch.float32),
            torch.arange(-patch_radius, patch_radius + 1, device=device, dtype=torch.float32),
            indexing="ij",
        )
        offsets_x = grid_x.reshape(1, -1)
        offsets_y = grid_y.reshape(1, -1)
        patch_dist2 = offsets_x.square() + offsets_y.square()

        cx = center_chunk[:, 0:1]
        cy = center_chunk[:, 1:2]
        px = torch.round(cx) + offsets_x
        py = torch.round(cy) + offsets_y
        patch_valid = (
            (px >= 0.0)
            & (px < float(width))
            & (py >= 0.0)
            & (py < float(height))
            & (patch_dist2 <= (radius_chunk[:, None].to(dtype=torch.float32) + 0.5).square())
        )
        weights = alpha_chunk[:, None] * torch.exp(-0.5 * patch_dist2 / sigma_chunk[:, None].square().clamp_min(1.0e-4))
        weights = weights * patch_valid.to(dtype=torch.float32)

        flat_indices = (py.to(dtype=torch.long) * int(width) + px.to(dtype=torch.long)).reshape(-1)
        flat_valid = patch_valid.reshape(-1)
        flat_weights = weights.reshape(-1)
        flat_indices = flat_indices[flat_valid]
        flat_weights = flat_weights[flat_valid]
        if flat_weights.numel() == 0:
            continue

        alpha_accum.index_add_(0, flat_indices, flat_weights)
        flat_values = value_chunk[:, None, :].expand(-1, patch_valid.shape[1], -1).reshape(-1, num_channels)
        expanded_weights = flat_weights[:, None] * flat_values[flat_valid]
        for channel_index in range(num_channels):
            channel_accum[channel_index].index_add_(0, flat_indices, expanded_weights[:, channel_index])

    alpha_map = alpha_accum.reshape(1, height, width)
    if normalize:
        prob_map = channel_accum / alpha_accum.clamp_min(1.0e-6)[None, :]
        prob_map = prob_map.reshape(num_channels, height, width) * alpha_map.clamp(max=1.0)
    else:
        prob_map = channel_accum.reshape(num_channels, height, width)
    return prob_map, alpha_map


def render_semantic_probs(
    viewpoint_camera: Any,
    gaussians: PreparedSemanticFrame,
    deformation_model: Any,
    surface_field: SurfaceMaskField,
    timestamp: float,
    active_channels: list[int] | None = None,
    bg_color: Any = None,
):
    del viewpoint_camera, deformation_model, timestamp, bg_color
    probs = surface_field.probs()[gaussians.gaussian_ids]
    if active_channels is None:
        channel_ids = list(range(int(probs.shape[1])))
    else:
        channel_ids = [int(item) for item in active_channels]
    values = probs[:, channel_ids]
    prob_map, alpha_map = _splat_channel_values(gaussians, values, normalize=True)
    return prob_map, alpha_map, {
        "active_channels": channel_ids,
        "gaussian_ids": gaussians.gaussian_ids.detach().cpu(),
    }


def render_selection_alpha_map(
    prepared: PreparedSemanticFrame,
    selected_weights: torch.Tensor,
    chunk_size: int = 2048,
    max_splat_radius: int = 18,
) -> torch.Tensor:
    if selected_weights.ndim == 1:
        selected_weights = selected_weights[:, None]
    alpha_mass, _alpha_map = _splat_channel_values(
        prepared,
        selected_weights,
        normalize=False,
        chunk_size=chunk_size,
        max_splat_radius=max_splat_radius,
    )
    return alpha_mass[0]


def binarize_alpha_map(
    alpha_map: torch.Tensor,
    relative_threshold: float = 0.18,
    absolute_threshold: float = 0.015,
) -> torch.Tensor:
    flat = alpha_map.reshape(-1)
    peak = torch.max(flat) if flat.numel() else torch.tensor(0.0, device=alpha_map.device)
    threshold = torch.maximum(
        peak * float(relative_threshold),
        torch.as_tensor(float(absolute_threshold), dtype=alpha_map.dtype, device=alpha_map.device),
    )
    return alpha_map >= threshold


def render_selection_mask(
    prepared: PreparedSemanticFrame,
    selected_weights: torch.Tensor,
    relative_threshold: float = 0.18,
    absolute_threshold: float = 0.015,
    max_splat_radius: int = 18,
) -> tuple[np.ndarray, np.ndarray]:
    alpha_map = render_selection_alpha_map(
        prepared,
        selected_weights,
        max_splat_radius=max_splat_radius,
    )
    binary = binarize_alpha_map(
        alpha_map=alpha_map,
        relative_threshold=relative_threshold,
        absolute_threshold=absolute_threshold,
    )
    return (
        binary.detach().cpu().numpy().astype(bool),
        alpha_map.detach().cpu().numpy().astype(np.float32),
    )


def gaussian_region_alpha_masses(
    prepared: PreparedSemanticFrame,
    region_masks: dict[str, np.ndarray],
    chunk_size: int = 2048,
) -> dict[str, torch.Tensor]:
    device = prepared.gaussian_ids.device
    num_gaussians = int(prepared.gaussian_ids.shape[0])
    masses = {
        "visible": torch.zeros((num_gaussians,), dtype=torch.float32, device=device),
        "full": torch.zeros((num_gaussians,), dtype=torch.float32, device=device),
        "core": torch.zeros((num_gaussians,), dtype=torch.float32, device=device),
        "boundary": torch.zeros((num_gaussians,), dtype=torch.float32, device=device),
        "outer": torch.zeros((num_gaussians,), dtype=torch.float32, device=device),
    }
    if num_gaussians == 0:
        return masses

    width = int(prepared.width)
    height = int(prepared.height)
    centers = prepared.centers_xy.to(device=device, dtype=torch.float32)
    sigma = prepared.sigma_px.to(device=device, dtype=torch.float32).clamp_min(0.75)
    alpha_weight = prepared.alpha_weight.to(device=device, dtype=torch.float32).clamp_min(0.0)
    region_tensors = {
        name: torch.as_tensor(mask.astype(np.float32), dtype=torch.float32, device=device)
        for name, mask in region_masks.items()
    }

    for start in range(0, num_gaussians, int(chunk_size)):
        end = min(start + int(chunk_size), num_gaussians)
        center_chunk = centers[start:end]
        sigma_chunk = sigma[start:end]
        alpha_chunk = alpha_weight[start:end]

        radius_chunk = torch.ceil(sigma_chunk * 3.0).to(dtype=torch.long).clamp(min=1, max=18)
        max_radius = int(radius_chunk.max().item())
        grid_y, grid_x = torch.meshgrid(
            torch.arange(-max_radius, max_radius + 1, device=device, dtype=torch.float32),
            torch.arange(-max_radius, max_radius + 1, device=device, dtype=torch.float32),
            indexing="ij",
        )
        offsets_x = grid_x.reshape(1, -1)
        offsets_y = grid_y.reshape(1, -1)
        patch_dist2 = offsets_x.square() + offsets_y.square()

        cx = center_chunk[:, 0:1]
        cy = center_chunk[:, 1:2]
        px = torch.round(cx) + offsets_x
        py = torch.round(cy) + offsets_y
        patch_valid = (
            (px >= 0.0)
            & (px < float(width))
            & (py >= 0.0)
            & (py < float(height))
            & (patch_dist2 <= (radius_chunk[:, None].to(dtype=torch.float32) + 0.5).square())
        )
        weights = alpha_chunk[:, None] * torch.exp(-0.5 * patch_dist2 / sigma_chunk[:, None].square().clamp_min(1.0e-4))
        weights = weights * patch_valid.to(dtype=torch.float32)
        masses["visible"][start:end] = weights.sum(dim=1)

        px_long = px.to(dtype=torch.long).clamp(min=0, max=max(int(width - 1), 0))
        py_long = py.to(dtype=torch.long).clamp(min=0, max=max(int(height - 1), 0))
        for name in ("full", "core", "boundary", "outer"):
            region_values = region_tensors[name][py_long, px_long]
            masses[name][start:end] = (weights * region_values).sum(dim=1)
    return masses


def alignment_metrics(
    pred_mask: np.ndarray,
    full_mask: np.ndarray,
    outer_mask: np.ndarray | None = None,
) -> dict[str, float]:
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(full_mask, dtype=bool)
    outer = np.zeros_like(gt, dtype=bool) if outer_mask is None else np.asarray(outer_mask, dtype=bool)
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    pred_area = float(pred.sum())
    gt_area = float(gt.sum())
    precision = inter / pred_area if pred_area > 0.0 else 0.0
    recall = inter / gt_area if gt_area > 0.0 else 0.0
    iou = inter / union if union > 0.0 else 0.0
    area_ratio = pred_area / gt_area if gt_area > 0.0 else 0.0
    outer_leakage = float(np.logical_and(pred, outer).sum()) / max(pred_area, 1.0)
    return {
        "rendered_iou_stage1": float(iou),
        "rendered_precision_stage1": float(precision),
        "rendered_recall_stage1": float(recall),
        "area_ratio": float(area_ratio),
        "outer_leakage": float(outer_leakage),
        "pred_area": float(pred_area),
        "gt_area": float(gt_area),
    }
