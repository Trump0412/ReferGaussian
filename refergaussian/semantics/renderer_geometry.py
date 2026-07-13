"""Renderer-consistent Gaussian geometry caches for query-time lifting.

The training-free semantic stages must use the same deformed Gaussian state as
the reconstruction renderer.  This module provides a compact on-disk cache
contract and the GPU exporter that evaluates that state at source-camera times.
It deliberately fails when a requested state is unavailable rather than
silently falling back to the old analytic trajectory approximation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


GEOMETRY_SCHEMA_VERSION = 1
GEOMETRY_MANIFEST_NAME = "geometry_manifest.json"


@dataclass(frozen=True)
class RendererGeometryFrame:
    """Renderer-state arrays for one source-camera frame."""

    image_id: str
    time_value: float
    centers: np.ndarray
    covariance_packed: np.ndarray | None
    opacity_logit: np.ndarray
    exact_image_id: bool


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    """Read the scalar run metadata written by ``scripts/train.sh``."""
    payload: dict[str, Any] = {}
    if not path.exists():
        return payload
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
            payload[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            payload[key] = float(value)
            continue
        except ValueError:
            pass
        payload[key] = value
    return payload


def _pack_covariance(covariance: np.ndarray) -> np.ndarray:
    array = np.asarray(covariance, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (3, 3):
        raise ValueError(f"Expected covariance [N,3,3], got {array.shape}")
    return np.stack(
        [
            array[:, 0, 0],
            array[:, 0, 1],
            array[:, 0, 2],
            array[:, 1, 1],
            array[:, 1, 2],
            array[:, 2, 2],
        ],
        axis=1,
    ).astype(np.float32)


def unpack_packed_covariance(covariance: np.ndarray) -> np.ndarray:
    """Convert ``[xx, xy, xz, yy, yz, zz]`` covariance rows to ``[3,3]``."""
    array = np.asarray(covariance, dtype=np.float32)
    if array.ndim == 3 and array.shape[1:] == (3, 3):
        return array
    if array.ndim != 2 or array.shape[1] != 6:
        raise ValueError(f"Expected packed covariance [N,6], got {array.shape}")
    unpacked = np.zeros((array.shape[0], 3, 3), dtype=np.float32)
    unpacked[:, 0, 0] = array[:, 0]
    unpacked[:, 0, 1] = unpacked[:, 1, 0] = array[:, 1]
    unpacked[:, 0, 2] = unpacked[:, 2, 0] = array[:, 2]
    unpacked[:, 1, 1] = array[:, 3]
    unpacked[:, 1, 2] = unpacked[:, 2, 1] = array[:, 4]
    unpacked[:, 2, 2] = array[:, 5]
    return unpacked


class RendererGeometryCache:
    """Memory-mapped renderer states indexed by source image id and Gaussian id."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        manifest_path = self.root / GEOMETRY_MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing renderer geometry manifest: {manifest_path}")
        self.manifest = _read_json(manifest_path)
        if int(self.manifest.get("schema_version", -1)) != GEOMETRY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported renderer geometry schema in {manifest_path}: "
                f"{self.manifest.get('schema_version')}"
            )

        self.image_ids = [str(value) for value in self.manifest.get("image_ids", [])]
        self.time_values = np.asarray(self.manifest.get("time_values", []), dtype=np.float32)
        if not self.image_ids or len(self.image_ids) != int(self.time_values.size):
            raise ValueError(f"Invalid image_ids/time_values in {manifest_path}")
        self.gaussian_ids = np.load(self.root / "gaussian_ids.npy", mmap_mode="r")
        self.centers = np.load(self.root / "centers.npy", mmap_mode="r")
        self.opacity_logit = np.load(self.root / "opacity_logit.npy", mmap_mode="r")
        covariance_path = self.root / "covariance_packed.npy"
        self.covariance_packed = np.load(covariance_path, mmap_mode="r") if covariance_path.is_file() else None

        expected_frames = len(self.image_ids)
        expected_gaussians = int(self.gaussian_ids.size)
        if self.centers.shape != (expected_frames, expected_gaussians, 3):
            raise ValueError(
                f"Invalid centers shape {self.centers.shape}; expected "
                f"({expected_frames}, {expected_gaussians}, 3)"
            )
        if self.opacity_logit.shape != (expected_frames, expected_gaussians):
            raise ValueError(
                f"Invalid opacity shape {self.opacity_logit.shape}; expected "
                f"({expected_frames}, {expected_gaussians})"
            )
        if self.covariance_packed is not None and self.covariance_packed.shape != (
            expected_frames,
            expected_gaussians,
            6,
        ):
            raise ValueError(
                f"Invalid covariance shape {self.covariance_packed.shape}; expected "
                f"({expected_frames}, {expected_gaussians}, 6)"
            )
        if self.gaussian_ids.ndim != 1 or not np.all(self.gaussian_ids[:-1] < self.gaussian_ids[1:]):
            raise ValueError("renderer geometry Gaussian ids must be sorted and unique")
        self._image_to_index = {image_id: index for index, image_id in enumerate(self.image_ids)}

    @property
    def gaussian_count(self) -> int:
        return int(self.gaussian_ids.size)

    def columns_for_gaussian_ids(self, gaussian_ids: np.ndarray, *, require_all: bool = True) -> np.ndarray:
        requested = np.asarray(gaussian_ids, dtype=np.int64).reshape(-1)
        positions = np.searchsorted(self.gaussian_ids, requested)
        valid = positions < self.gaussian_ids.size
        valid[valid] &= self.gaussian_ids[positions[valid]] == requested[valid]
        if require_all and not bool(valid.all()):
            missing = requested[~valid]
            raise KeyError(
                "Renderer geometry cache does not contain requested Gaussian ids: "
                + ", ".join(str(int(value)) for value in missing[:8])
            )
        return positions[valid] if not require_all else positions

    def resolve(
        self,
        image_id: str | None,
        time_value: float,
        gaussian_ids: np.ndarray | None = None,
        *,
        require_exact_image_id: bool = True,
    ) -> RendererGeometryFrame:
        normalized_image_id = "" if image_id is None else str(image_id)
        if normalized_image_id and normalized_image_id in self._image_to_index:
            frame_index = self._image_to_index[normalized_image_id]
            exact_image_id = True
        else:
            if require_exact_image_id:
                raise KeyError(
                    f"Renderer geometry cache has no source state for image_id='{normalized_image_id}' "
                    f"under {self.root}"
                )
            frame_index = int(np.abs(self.time_values - float(time_value)).argmin())
            exact_image_id = False

        if gaussian_ids is None:
            columns: np.ndarray | slice = slice(None)
        else:
            columns = self.columns_for_gaussian_ids(gaussian_ids, require_all=True)
        return RendererGeometryFrame(
            image_id=self.image_ids[frame_index],
            time_value=float(self.time_values[frame_index]),
            centers=np.asarray(self.centers[frame_index, columns], dtype=np.float32),
            covariance_packed=(
                None
                if self.covariance_packed is None
                else np.asarray(self.covariance_packed[frame_index, columns], dtype=np.float32)
            ),
            opacity_logit=np.asarray(self.opacity_logit[frame_index, columns], dtype=np.float32),
            exact_image_id=exact_image_id,
        )


def load_renderer_geometry_cache(path: str | Path | None) -> RendererGeometryCache | None:
    if path is None:
        return None
    root = Path(path)
    if not root.exists():
        return None
    return RendererGeometryCache(root)


def _source_entries(dataset_dir: Path) -> list[dict[str, Any]]:
    from .source_images import resolve_dataset_image_entries

    entries = resolve_dataset_image_entries(dataset_dir)
    normalized = [
        {
            "image_id": str(entry["image_id"]),
            "frame_index": int(entry["frame_index"]),
            "time_value": float(entry["time_value"]),
        }
        for entry in entries
    ]
    return sorted(normalized, key=lambda row: row["frame_index"])


def frame_requests_from_tracks(tracks_path: str | Path) -> list[dict[str, Any]]:
    payload = _read_json(Path(tracks_path))
    by_image_id: dict[str, dict[str, Any]] = {}
    for track in payload.get("tracks", []):
        if str(track.get("status", "")) != "seeded":
            continue
        for frame in track.get("frames", []):
            if not bool(frame.get("active")) or not frame.get("image_id"):
                continue
            image_id = str(frame["image_id"])
            by_image_id.setdefault(
                image_id,
                {
                    "image_id": image_id,
                    "frame_index": int(frame.get("frame_index", 0)),
                    "time_value": float(frame.get("time_value", 0.0)),
                },
            )
    if not by_image_id:
        raise ValueError(f"No active seeded frames found in {tracks_path}")
    return sorted(by_image_id.values(), key=lambda row: row["frame_index"])


def gaussian_ids_from_entitybank(entitybank_dir: str | Path) -> np.ndarray:
    payload = _read_json(Path(entitybank_dir) / "entities.json")
    values: list[int] = []
    for entity in payload.get("entities", []):
        values.extend(int(value) for value in entity.get("gaussian_ids", []))
    ids = np.unique(np.asarray(values, dtype=np.int64))
    if ids.size == 0:
        raise ValueError(f"No Gaussian ids found in {Path(entitybank_dir) / 'entities.json'}")
    return ids


def _prepare_external_imports(project_root: Path) -> None:
    external_root = project_root / "external" / "4DGaussians"
    if not (external_root / "train.py").is_file():
        raise FileNotFoundError(
            f"Missing external/4DGaussians under {project_root}. "
            "Run scripts/bootstrap_external.sh before renderer-geometry export."
        )
    for candidate in (str(project_root), str(external_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


def _overlay_run_config(args: Any, config_path: Path) -> Any:
    for key, value in _read_simple_yaml(config_path).items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def _load_renderer_runtime(run_dir: Path) -> tuple[Any, Any, Any, Any, int]:
    """Load the exact fine-stage model used by ``external/4DGaussians/render.py``."""
    project_root = Path(__file__).resolve().parents[2]
    _prepare_external_imports(project_root)

    from arguments import ModelHiddenParams, ModelParams, PipelineParams, get_combined_args
    from gaussian_renderer import GaussianModel
    from scene import Scene
    from utils.config_utils import apply_config_file

    from refergaussian.temporal import attach_temporal_warp, build_temporal_warp, load_temporal_warp

    argv_before = sys.argv[:]
    try:
        # Reuse the upstream parser and cfg_args loader, then restore the explicit
        # ReferGaussian values recorded by scripts/train.sh.  The latter is needed
        # because the base 4DGaussians config intentionally has no knowledge of
        # ReferGaussian-only temporal flags.
        sys.argv = [argv_before[0] if argv_before else "renderer_geometry", "-m", str(run_dir)]
        parser = argparse.ArgumentParser(description="ReferGaussian renderer geometry")
        model = ModelParams(parser, sentinel=True)
        PipelineParams(parser)
        hidden = ModelHiddenParams(parser)
        args = get_combined_args(parser)
        if getattr(args, "configs", None):
            args = apply_config_file(args, args.configs)
        args = _overlay_run_config(args, run_dir / "config.yaml")
    finally:
        sys.argv = argv_before

    gaussians = GaussianModel(args.sh_degree, hidden.extract(args))
    scene = Scene(model.extract(args), gaussians, load_iteration=-1, shuffle=False)
    temporal_warp = build_temporal_warp(hidden.extract(args))
    attach_temporal_warp(gaussians, temporal_warp)
    warp_loaded = load_temporal_warp(temporal_warp, args.model_path, iteration=scene.loaded_iter)
    if bool(getattr(args, "warp_enabled", False)) and not warp_loaded:
        raise RuntimeError(
            f"ReferGaussian temporal warp is enabled but no compatible checkpoint was loaded from {run_dir}"
        )
    return args, gaussians, temporal_warp, scene, int(scene.loaded_iter)


def _select_context(gaussians: Any, indices: Any) -> dict[str, Any]:
    context = dict(gaussians.get_temporal_context())
    total = int(gaussians.get_xyz.shape[0])
    selected: dict[str, Any] = {}
    for key, value in context.items():
        if hasattr(value, "ndim") and value.ndim >= 1 and int(value.shape[0]) == total:
            selected[key] = value.index_select(0, indices)
        else:
            selected[key] = value
    return selected


def _renderer_state_at_time(
    gaussians: Any,
    temporal_warp: Any,
    time_value: float,
    gaussian_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate one fine-stage state, matching the renderer before rasterization."""
    import torch

    from utils.general_utils import build_scaling_rotation

    ids = torch.as_tensor(np.asarray(gaussian_ids, dtype=np.int64), dtype=torch.long, device=gaussians.get_xyz.device)
    with torch.no_grad():
        base_xyz = gaussians.get_xyz.index_select(0, ids)
        raw_scaling = gaussians._scaling.index_select(0, ids)
        raw_rotation = gaussians._rotation.index_select(0, ids)
        raw_opacity = gaussians._opacity.index_select(0, ids)
        shs = gaussians.get_features.index_select(0, ids)
        raw_time = torch.full(
            (base_xyz.shape[0], 1),
            float(time_value),
            dtype=base_xyz.dtype,
            device=base_xyz.device,
        )
        means = base_xyz
        time_for_deformation = raw_time
        temporal_tube_slice = None
        temporal_slice = None

        if bool(getattr(gaussians, "temporal_extent_enabled", False)):
            if bool(getattr(gaussians, "temporal_worldtube_enabled", False)):
                raise RuntimeError(
                    "renderer_geometry does not yet export multi-sample temporal_worldtube states; "
                    "disable that experimental renderer mode rather than using an inconsistent semantic projection"
                )
            if bool(getattr(gaussians, "temporal_tube_enabled", False)):
                temporal_tube_slice = gaussians.get_temporal_tube_slice(raw_time, indices=ids)
                means = means + float(getattr(gaussians, "temporal_drift_mix", 1.0)) * temporal_tube_slice["mean_drift"]
            else:
                temporal_slice = gaussians.get_temporal_slice(raw_time, indices=ids)
                means = means + float(getattr(gaussians, "temporal_drift_mix", 1.0)) * temporal_slice["drift"]

        if temporal_warp is not None:
            context = _select_context(gaussians, ids)
            if temporal_tube_slice is not None:
                context["query_delta"] = temporal_tube_slice["delta"]
                context["query_normalized_time"] = temporal_tube_slice["normalized_time"]
                context["query_gate"] = temporal_tube_slice["gate"]
            elif temporal_slice is not None:
                context["query_delta"] = temporal_slice["delta"]
                context["query_normalized_time"] = temporal_slice["normalized_time"]
                context["query_gate"] = temporal_slice["gate"]
            time_for_deformation = temporal_warp(raw_time, context=context)

        means_final, scaling_raw_final, rotation_raw_final, opacity_raw_final, _ = gaussians._deformation(
            means,
            raw_scaling,
            raw_rotation,
            raw_opacity,
            shs,
            time_for_deformation,
        )
        scaling_final = gaussians.scaling_activation(scaling_raw_final)
        rotation_final = gaussians.rotation_activation(rotation_raw_final)
        covariance_factor = build_scaling_rotation(scaling_final, rotation_final)
        covariance = covariance_factor @ covariance_factor.transpose(1, 2)
        if temporal_tube_slice is not None:
            covariance = covariance + float(getattr(gaussians, "temporal_tube_covariance_mix", 1.0)) * temporal_tube_slice["covariance"]

    return (
        means_final.detach().cpu().numpy().astype(np.float32),
        _pack_covariance(covariance.detach().cpu().numpy()),
        opacity_raw_final.detach().cpu().numpy().reshape(-1).astype(np.float32),
    )


def _valid_existing_cache(
    output_dir: Path,
    frame_requests: list[dict[str, Any]],
    gaussian_ids: np.ndarray,
) -> bool:
    try:
        cache = RendererGeometryCache(output_dir)
    except (FileNotFoundError, ValueError, OSError):
        return False
    return (
        cache.image_ids == [str(row["image_id"]) for row in frame_requests]
        and np.array_equal(np.asarray(cache.gaussian_ids, dtype=np.int64), gaussian_ids)
    )


def export_renderer_geometry(
    run_dir: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    frame_requests: Iterable[dict[str, Any]],
    gaussian_ids: np.ndarray | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Export exact Gaussian states for requested source frames.

    The cache is intentionally query-local: support lifting may need all
    Gaussians for a small Stage-1 frame set, while final rendering needs only
    selected Gaussians for a larger evaluation frame set.
    """
    del dataset_dir  # The source-camera metadata is already carried by frame_requests.
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    normalized_requests = [
        {
            "image_id": str(row["image_id"]),
            "frame_index": int(row.get("frame_index", index)),
            "time_value": float(row["time_value"]),
        }
        for index, row in enumerate(frame_requests)
    ]
    if not normalized_requests:
        raise ValueError("renderer geometry export requires at least one frame request")
    normalized_requests.sort(key=lambda row: row["frame_index"])
    duplicate_ids = [
        row["image_id"]
        for index, row in enumerate(normalized_requests)
        if index and row["image_id"] == normalized_requests[index - 1]["image_id"]
    ]
    if duplicate_ids:
        raise ValueError(f"renderer geometry frame requests contain duplicate image ids: {duplicate_ids[:3]}")

    args, gaussians, temporal_warp, _scene, iteration = _load_renderer_runtime(run_dir)
    total_gaussians = int(gaussians.get_xyz.shape[0])
    if gaussian_ids is None:
        ids = np.arange(total_gaussians, dtype=np.int64)
    else:
        ids = np.unique(np.asarray(gaussian_ids, dtype=np.int64).reshape(-1))
        if ids.size == 0 or int(ids[0]) < 0 or int(ids[-1]) >= total_gaussians:
            raise ValueError(
                f"Requested Gaussian ids must be within [0, {total_gaussians}), got "
                f"{ids[:3].tolist()}...{ids[-3:].tolist()}"
            )

    if output_dir.exists():
        if _valid_existing_cache(output_dir, normalized_requests, ids):
            return output_dir
        if not overwrite:
            raise FileExistsError(
                f"Renderer geometry output already exists with a different contract: {output_dir}. "
                "Use a new output path or explicitly request overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    frame_count = len(normalized_requests)
    gaussian_count = int(ids.size)
    centers = np.lib.format.open_memmap(
        output_dir / "centers.npy",
        mode="w+",
        dtype=np.float32,
        shape=(frame_count, gaussian_count, 3),
    )
    covariance = np.lib.format.open_memmap(
        output_dir / "covariance_packed.npy",
        mode="w+",
        dtype=np.float32,
        shape=(frame_count, gaussian_count, 6),
    )
    opacity = np.lib.format.open_memmap(
        output_dir / "opacity_logit.npy",
        mode="w+",
        dtype=np.float32,
        shape=(frame_count, gaussian_count),
    )
    start = time.monotonic()
    for frame_index, request in enumerate(normalized_requests):
        state_centers, state_covariance, state_opacity = _renderer_state_at_time(
            gaussians,
            temporal_warp,
            float(request["time_value"]),
            ids,
        )
        centers[frame_index] = state_centers
        covariance[frame_index] = state_covariance
        opacity[frame_index] = state_opacity
    centers.flush()
    covariance.flush()
    opacity.flush()
    np.save(output_dir / "gaussian_ids.npy", ids)
    manifest = {
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "geometry_mode": "renderer_consistent_fine_state",
        "run_dir": str(run_dir.resolve()),
        "iteration": int(iteration),
        "source_path": str(getattr(args, "source_path", "")),
        "image_ids": [row["image_id"] for row in normalized_requests],
        "frame_indices": [int(row["frame_index"]) for row in normalized_requests],
        "time_values": [float(row["time_value"]) for row in normalized_requests],
        "gaussian_count": gaussian_count,
        "state_fields": ["centers", "covariance_packed", "opacity_logit"],
        "temporal_warp_enabled": bool(getattr(args, "warp_enabled", False)),
        "temporal_warp_type": str(getattr(args, "temporal_warp_type", "identity")),
        "temporal_extent_enabled": bool(getattr(args, "temporal_extent_enabled", False)),
        "temporal_tube_enabled": bool(getattr(args, "temporal_tube_enabled", False)),
        "elapsed_seconds": float(time.monotonic() - start),
    }
    _write_json(output_dir / GEOMETRY_MANIFEST_NAME, manifest)
    return output_dir

