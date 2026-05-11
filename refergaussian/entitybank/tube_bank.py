from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from plyfile import PlyData


@dataclass
class GaussianState:
    xyz: np.ndarray
    rgb: np.ndarray
    spatial_scale: np.ndarray
    anchor: np.ndarray
    scale: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    opacity: np.ndarray


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() in {"true", "false"}:
            payload[key] = value.lower() == "true"
            continue
        try:
            if "." in value:
                payload[key] = float(value)
            else:
                payload[key] = int(value)
            continue
        except ValueError:
            payload[key] = value
    return payload


def find_latest_iteration_dir(run_dir: Path) -> Path:
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


def load_gaussian_state(run_dir: Path) -> tuple[GaussianState, dict[str, Any], int]:
    iteration_dir = find_latest_iteration_dir(run_dir)
    ply_path = iteration_dir / "point_cloud.ply"
    temporal_path = iteration_dir / "temporal_params.pth"

    ply = PlyData.read(str(ply_path))
    vertices = ply["vertex"]
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    rgb = np.stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]], axis=1).astype(np.float32)
    opacity = np.asarray(vertices["opacity"]).astype(np.float32)
    spatial_scale_names = [name for name in vertices.data.dtype.names if name.startswith("scale_")]
    spatial_scale_names = sorted(spatial_scale_names, key=lambda item: int(item.split("_")[-1]))
    spatial_scale = np.stack([vertices[name] for name in spatial_scale_names], axis=1).astype(np.float32)
    spatial_scale = np.exp(spatial_scale)

    temporal_payload = torch.load(temporal_path, map_location="cpu")
    anchor = torch.sigmoid(temporal_payload["time_anchor"].float()).numpy()
    scale = (torch.nn.functional.softplus(temporal_payload["time_scale"].float()) + 1.0e-6).numpy()
    velocity = temporal_payload["time_velocity"].float().numpy()
    acceleration_raw = temporal_payload.get("time_acceleration")
    if acceleration_raw is None:
        acceleration = np.zeros_like(velocity, dtype=np.float32)
    else:
        acceleration = acceleration_raw.float().numpy()

    config = _read_simple_yaml(run_dir / "config.yaml")
    iteration = int(iteration_dir.name.split("_", 1)[1])
    state = GaussianState(
        xyz=xyz,
        rgb=rgb,
        spatial_scale=spatial_scale.astype(np.float32),
        anchor=anchor.astype(np.float32),
        scale=scale.astype(np.float32),
        velocity=velocity.astype(np.float32),
        acceleration=acceleration.astype(np.float32),
        opacity=opacity.astype(np.float32),
    )
    return state, config, iteration


def sample_tube_bank(
    state: GaussianState,
    num_frames: int = 64,
    gate_sharpness: float = 1.0,
    drift_mix: float = 1.0,
    config: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    config = config or {}
    time_values = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    delta = time_values[None, :, None] - state.anchor[:, None, :]
    safe_scale = np.clip(state.scale[:, None, :], 1.0e-4, None)

    tube_span = max(float(config.get("temporal_worldtube_span", 1.0)), 1.0e-4)
    support_gain = float(config.get("temporal_worldtube_support_gain", 1.0))
    visibility_mix = float(config.get("temporal_worldtube_visibility_mix", 0.35))
    support_min_factor = min(
        float(config.get("temporal_worldtube_support_min_factor", 0.75)),
        float(config.get("temporal_worldtube_support_max_factor", 2.5)),
    )
    support_max_factor = max(
        float(config.get("temporal_worldtube_support_min_factor", 0.75)),
        float(config.get("temporal_worldtube_support_max_factor", 2.5)),
    )
    target_ratio = max(float(config.get("temporal_worldtube_ratio_target", 1.0)), 1.0e-4)
    adaptive_support = bool(config.get("temporal_worldtube_adaptive_support", False))
    spatial_extent = np.clip(state.spatial_scale.max(axis=1, keepdims=True), 1.0e-4, None)
    speed = np.linalg.norm(state.velocity, axis=1, keepdims=True)
    acceleration = np.linalg.norm(state.acceleration, axis=1, keepdims=True)
    base_support = state.scale * tube_span
    base_drift = speed * base_support
    if bool(config.get("temporal_acceleration_enabled", False)):
        base_drift = base_drift + 0.5 * acceleration * np.square(base_support)
    base_ratio = base_drift / spatial_extent
    opacity = 1.0 / (1.0 + np.exp(-state.opacity.reshape(-1, 1)))
    visibility_proxy = np.clip((1.0 - visibility_mix) + visibility_mix * opacity, 1.0e-3, None)
    support_factor = np.ones_like(base_ratio, dtype=np.float32)
    if adaptive_support:
        ratio_error = (target_ratio - base_ratio) / target_ratio
        support_factor = np.clip(
            1.0 + support_gain * ratio_error * visibility_proxy,
            support_min_factor,
            support_max_factor,
        ).astype(np.float32)
    effective_scale = state.scale * support_factor
    normalized_time = delta / np.clip(effective_scale[:, None, :], 1.0e-4, None)
    gate = np.exp(-0.5 * gate_sharpness * np.square(normalized_time)).astype(np.float32)

    linear = state.velocity[:, None, :] * delta
    quadratic = 0.5 * state.acceleration[:, None, :] * np.square(delta)
    trajectories = state.xyz[:, None, :] + drift_mix * (linear + quadratic)

    displacement = trajectories[:, -1, :] - trajectories[:, 0, :]
    path_length = np.linalg.norm(np.diff(trajectories, axis=1), axis=-1).sum(axis=1)
    motion_score = path_length * gate.mean(axis=1).squeeze(-1)
    effective_support = effective_scale * tube_span
    effective_drift = speed * effective_support
    if bool(config.get("temporal_acceleration_enabled", False)):
        effective_drift = effective_drift + 0.5 * acceleration * np.square(effective_support)
    tube_ratio = effective_drift / spatial_extent
    occupancy_mass = effective_support.squeeze(-1) * (1.0 + tube_ratio.squeeze(-1)) * visibility_proxy.squeeze(-1)

    return {
        "time_values": time_values,
        "trajectories": trajectories.astype(np.float32),
        "gate": gate.astype(np.float32),
        "displacement": displacement.astype(np.float32),
        "path_length": path_length.astype(np.float32),
        "motion_score": motion_score.astype(np.float32),
        "spatial_extent": spatial_extent.astype(np.float32),
        "support_factor": support_factor.astype(np.float32),
        "effective_scale": effective_scale.astype(np.float32),
        "effective_support": effective_support.astype(np.float32),
        "tube_ratio": tube_ratio.astype(np.float32),
        "occupancy_mass": occupancy_mass.astype(np.float32),
        "visibility_proxy": visibility_proxy.astype(np.float32),
    }


def summarize_tube_bank(state: GaussianState, bank: dict[str, np.ndarray]) -> dict[str, Any]:
    motion_score = bank["motion_score"]
    displacement_norm = np.linalg.norm(bank["displacement"], axis=1)
    speed = np.linalg.norm(state.velocity, axis=1)
    acceleration = np.linalg.norm(state.acceleration, axis=1)
    return {
        "schema_version": 1,
        "num_gaussians": int(state.xyz.shape[0]),
        "num_frames": int(bank["time_values"].shape[0]),
        "motion_score_mean": float(motion_score.mean()),
        "motion_score_max": float(motion_score.max()),
        "displacement_mean": float(displacement_norm.mean()),
        "displacement_max": float(displacement_norm.max()),
        "speed_mean": float(speed.mean()),
        "speed_max": float(speed.max()),
        "acceleration_mean": float(acceleration.mean()),
        "acceleration_max": float(acceleration.max()),
        "support_factor_mean": float(bank["support_factor"].mean()),
        "support_factor_max": float(bank["support_factor"].max()),
        "effective_support_mean": float(bank["effective_support"].mean()),
        "effective_support_max": float(bank["effective_support"].max()),
        "tube_ratio_mean": float(bank["tube_ratio"].mean()),
        "tube_ratio_max": float(bank["tube_ratio"].max()),
        "occupancy_mean": float(bank["occupancy_mass"].mean()),
        "occupancy_max": float(bank["occupancy_mass"].max()),
        "visibility_mean": float(bank["visibility_proxy"].mean()),
        "visibility_max": float(bank["visibility_proxy"].max()),
    }


def save_tube_bank(
    run_dir: Path,
    state: GaussianState,
    bank: dict[str, np.ndarray],
    iteration: int,
    output_dir: Path | None = None,
) -> Path:
    out_dir = Path(output_dir) if output_dir is not None else (run_dir / "entitybank")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "trajectory_samples.npz",
        time_values=bank["time_values"],
        trajectories=bank["trajectories"],
        gate=bank["gate"],
        displacement=bank["displacement"],
        path_length=bank["path_length"],
        motion_score=bank["motion_score"],
        spatial_extent=bank["spatial_extent"],
        support_factor=bank["support_factor"],
        effective_scale=bank["effective_scale"],
        effective_support=bank["effective_support"],
        tube_ratio=bank["tube_ratio"],
        occupancy_mass=bank["occupancy_mass"],
        visibility_proxy=bank["visibility_proxy"],
        xyz=state.xyz,
        spatial_scale=state.spatial_scale,
        velocity=state.velocity,
        acceleration=state.acceleration,
        anchor=state.anchor,
        scale=state.scale,
    )

    tube_payload = summarize_tube_bank(state, bank)
    tube_payload["iteration"] = iteration
    with open(out_dir / "tube_bank.json", "w", encoding="utf-8") as handle:
        json.dump(tube_payload, handle, indent=2)
    return out_dir
