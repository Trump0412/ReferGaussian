"""Renderer-consistent Gaussian geometry caches for query-time lifting.

The training-free semantic stages must use the same deformed Gaussian state as
the frozen upstream 4DGS renderer.  This module provides a compact on-disk cache
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


GEOMETRY_SCHEMA_VERSION = 2
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


@dataclass(frozen=True)
class RendererProjectionCamera:
    """The source-camera projection actually consumed by the 4DGS renderer.

    The upstream renderer does not project through HyperNeRF's JSON camera
    model directly: it constructs a centered-FoV Graphdeco camera at the
    training image resolution.  Semantic lifting must use that same matrix
    contract, otherwise a seemingly small principal-point difference shifts a
    projected entity by dozens of pixels.

    Matrices use the upstream row-vector convention, i.e. homogeneous world
    points are multiplied as ``[x, y, z, 1] @ matrix``.
    """

    image_size: np.ndarray
    world_view_transform: np.ndarray
    full_proj_transform: np.ndarray

    def __post_init__(self) -> None:
        image_size = np.asarray(self.image_size, dtype=np.int32).reshape(-1)
        world_view = np.asarray(self.world_view_transform, dtype=np.float32)
        full_proj = np.asarray(self.full_proj_transform, dtype=np.float32)
        if image_size.shape != (2,) or np.any(image_size <= 0):
            raise ValueError(f"Renderer projection image_size must be [width, height], got {image_size}")
        if world_view.shape != (4, 4) or full_proj.shape != (4, 4):
            raise ValueError(
                "Renderer projection matrices must both be [4,4], got "
                f"{world_view.shape} and {full_proj.shape}"
            )
        object.__setattr__(self, "image_size", image_size)
        object.__setattr__(self, "world_view_transform", world_view)
        object.__setattr__(self, "full_proj_transform", full_proj)

    @staticmethod
    def _homogeneous(points: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        array = np.asarray(points, dtype=np.float32)
        if array.shape[-1:] != (3,):
            raise ValueError(f"Expected points ending in xyz, got {array.shape}")
        shape = array.shape
        flat = array.reshape(-1, 3)
        homogeneous = np.concatenate(
            [flat, np.ones((flat.shape[0], 1), dtype=np.float32)], axis=1
        )
        return homogeneous, shape

    def points_to_local_points(self, points: np.ndarray) -> np.ndarray:
        homogeneous, shape = self._homogeneous(points)
        local = homogeneous @ self.world_view_transform
        return np.asarray(local[:, :3], dtype=np.float32).reshape(shape)

    def project(self, points: np.ndarray) -> np.ndarray:
        homogeneous, shape = self._homogeneous(points)
        clip = homogeneous @ self.full_proj_transform
        denominator = clip[:, 3]
        safe_denominator = np.where(
            np.abs(denominator) > 1.0e-8,
            denominator,
            np.where(denominator >= 0.0, 1.0e-8, -1.0e-8),
        )
        ndc = clip[:, :2] / safe_denominator[:, None]
        width, height = (int(self.image_size[0]), int(self.image_size[1]))
        pixels = np.stack(
            [
                (ndc[:, 0] + 1.0) * (0.5 * float(width)),
                (ndc[:, 1] + 1.0) * (0.5 * float(height)),
            ],
            axis=1,
        )
        return np.asarray(pixels, dtype=np.float32).reshape((*shape[:-1], 2))


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


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
        self.projection_world_view = np.load(self.root / "projection_world_view.npy", mmap_mode="r")
        self.projection_full = np.load(self.root / "projection_full.npy", mmap_mode="r")
        self.projection_image_sizes = np.load(self.root / "projection_image_sizes.npy", mmap_mode="r")
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
        if self.projection_world_view.shape != (expected_frames, 4, 4):
            raise ValueError(
                f"Invalid projection world-view shape {self.projection_world_view.shape}; expected "
                f"({expected_frames}, 4, 4)"
            )
        if self.projection_full.shape != (expected_frames, 4, 4):
            raise ValueError(
                f"Invalid projection full-matrix shape {self.projection_full.shape}; expected "
                f"({expected_frames}, 4, 4)"
            )
        if self.projection_image_sizes.shape != (expected_frames, 2):
            raise ValueError(
                f"Invalid projection image-size shape {self.projection_image_sizes.shape}; expected "
                f"({expected_frames}, 2)"
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

    def _resolve_frame_index(
        self,
        image_id: str | None,
        time_value: float,
        *,
        require_exact_image_id: bool,
    ) -> tuple[int, bool]:
        normalized_image_id = "" if image_id is None else str(image_id)
        if normalized_image_id and normalized_image_id in self._image_to_index:
            return self._image_to_index[normalized_image_id], True
        if require_exact_image_id:
            raise KeyError(
                f"Renderer geometry cache has no source state for image_id='{normalized_image_id}' "
                f"under {self.root}"
            )
        return int(np.abs(self.time_values - float(time_value)).argmin()), False

    def resolve(
        self,
        image_id: str | None,
        time_value: float,
        gaussian_ids: np.ndarray | None = None,
        *,
        require_exact_image_id: bool = True,
    ) -> RendererGeometryFrame:
        frame_index, exact_image_id = self._resolve_frame_index(
            image_id,
            time_value,
            require_exact_image_id=require_exact_image_id,
        )

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

    def resolve_projection_camera(
        self,
        image_id: str | None,
        time_value: float,
        *,
        require_exact_image_id: bool = True,
    ) -> RendererProjectionCamera:
        """Return the exact source-camera projection used by the frozen 4DGS renderer."""
        frame_index, _exact_image_id = self._resolve_frame_index(
            image_id,
            time_value,
            require_exact_image_id=require_exact_image_id,
        )
        return RendererProjectionCamera(
            image_size=np.asarray(self.projection_image_sizes[frame_index], dtype=np.int32),
            world_view_transform=np.asarray(self.projection_world_view[frame_index], dtype=np.float32),
            full_proj_transform=np.asarray(self.projection_full[frame_index], dtype=np.float32),
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
            "image_path": str(entry["image_path"]),
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


def _load_renderer_runtime(run_dir: Path) -> tuple[Any, Any, Any, Any, int]:
    """Load a frozen model with the pinned upstream 4DGaussians runtime."""
    project_root = Path(__file__).resolve().parents[2]
    _prepare_external_imports(project_root)

    from arguments import ModelHiddenParams, ModelParams, PipelineParams, get_combined_args
    from gaussian_renderer import GaussianModel
    from scene import Scene

    argv_before = sys.argv[:]
    try:
        # get_combined_args reads the standard cfg_args stored in the run root.
        sys.argv = [argv_before[0] if argv_before else "renderer_geometry", "-m", str(run_dir)]
        parser = argparse.ArgumentParser(description="ReferGaussian renderer geometry")
        model = ModelParams(parser, sentinel=True)
        PipelineParams(parser)
        hidden = ModelHiddenParams(parser)
        args = get_combined_args(parser)
    finally:
        sys.argv = argv_before

    gaussians = GaussianModel(args.sh_degree, hidden.extract(args))
    scene = Scene(model.extract(args), gaussians, load_iteration=-1, shuffle=False)
    return args, gaussians, None, scene, int(scene.loaded_iter)


def _renderer_state_at_time(
    gaussians: Any,
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

        means_final, scaling_raw_final, rotation_raw_final, opacity_raw_final, _ = gaussians._deformation(
            means,
            raw_scaling,
            raw_rotation,
            raw_opacity,
            shs,
            raw_time,
        )
        scaling_final = gaussians.scaling_activation(scaling_raw_final)
        rotation_final = gaussians.rotation_activation(rotation_raw_final)
        covariance_factor = build_scaling_rotation(scaling_final, rotation_final)
        covariance = covariance_factor @ covariance_factor.transpose(1, 2)

    return (
        means_final.detach().cpu().numpy().astype(np.float32),
        _pack_covariance(covariance.detach().cpu().numpy()),
        opacity_raw_final.detach().cpu().numpy().reshape(-1).astype(np.float32),
    )


def _renderer_projection_metadata(
    dataset_dir: Path,
    source_entry: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild the upstream centered-FoV camera for one source image.

    HyperNeRF camera JSON includes an off-centre principal point and distortion,
    while upstream 4DGaussians deliberately trains with a Graphdeco FoV camera.
    We persist the latter exactly.  The source image dimensions determine the
    renderer raster resolution; the FoV is still derived from the native JSON
    image size, matching ``Load_hyper_data(..., ratio=0.5)``.
    """
    from PIL import Image

    from scene.utils import Camera as SourceCamera
    from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2

    image_id = str(source_entry["image_id"])
    camera_path = dataset_dir / "camera" / f"{image_id}.json"
    if not camera_path.is_file():
        raise FileNotFoundError(
            "Renderer-consistent semantic projection requires source camera JSON for "
            f"image_id='{image_id}', missing {camera_path}"
        )
    image_path = Path(str(source_entry["image_path"]))
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Renderer-consistent semantic projection cannot find source image {image_path}"
        )
    source_camera = SourceCamera.from_json(camera_path)
    with Image.open(image_path) as image:
        image_width, image_height = (int(image.size[0]), int(image.size[1]))
    native_size = np.asarray(source_camera.image_size, dtype=np.float32).reshape(-1)
    if native_size.size < 2:
        raise ValueError(f"Invalid source camera image size in {camera_path}: {native_size}")
    native_width = int(max(round(float(native_size[0])), 1))
    native_height = int(max(round(float(native_size[1])), 1))
    focal_values = np.asarray(source_camera.focal_length, dtype=np.float32).reshape(-1)
    if focal_values.size == 0:
        raise ValueError(f"Invalid source camera focal length in {camera_path}")
    focal = float(focal_values[0])
    rotation = np.asarray(source_camera.orientation, dtype=np.float32).T
    translation = -np.asarray(source_camera.position, dtype=np.float32) @ rotation
    world_view = np.asarray(getWorld2View2(rotation, translation), dtype=np.float32).T
    projection = getProjectionMatrix(
        znear=0.01,
        zfar=100.0,
        fovX=focal2fov(focal, native_width),
        fovY=focal2fov(focal, native_height),
    )
    projection = projection.detach().cpu().numpy().astype(np.float32).T
    full_projection = (world_view @ projection).astype(np.float32)
    return (
        world_view,
        full_projection,
        np.asarray([image_width, image_height], dtype=np.int32),
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
    run_dir = Path(run_dir)
    dataset_dir = Path(dataset_dir)
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

    source_entries = {
        str(entry["image_id"]): entry
        for entry in _source_entries(dataset_dir)
    }
    missing_source_entries = [
        str(request["image_id"])
        for request in normalized_requests
        if str(request["image_id"]) not in source_entries
    ]
    if missing_source_entries:
        raise KeyError(
            "Renderer geometry requests are not present in the source-image map: "
            + ", ".join(missing_source_entries[:8])
        )

    args, gaussians, _unused_adapter, _scene, iteration = _load_renderer_runtime(run_dir)
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
    projection_world_view = np.lib.format.open_memmap(
        output_dir / "projection_world_view.npy",
        mode="w+",
        dtype=np.float32,
        shape=(frame_count, 4, 4),
    )
    projection_full = np.lib.format.open_memmap(
        output_dir / "projection_full.npy",
        mode="w+",
        dtype=np.float32,
        shape=(frame_count, 4, 4),
    )
    projection_image_sizes = np.lib.format.open_memmap(
        output_dir / "projection_image_sizes.npy",
        mode="w+",
        dtype=np.int32,
        shape=(frame_count, 2),
    )
    start = time.monotonic()
    for frame_index, request in enumerate(normalized_requests):
        state_centers, state_covariance, state_opacity = _renderer_state_at_time(
            gaussians,
            float(request["time_value"]),
            ids,
        )
        centers[frame_index] = state_centers
        covariance[frame_index] = state_covariance
        opacity[frame_index] = state_opacity
        world_view, full_projection, image_size = _renderer_projection_metadata(
            dataset_dir,
            source_entries[str(request["image_id"])],
        )
        projection_world_view[frame_index] = world_view
        projection_full[frame_index] = full_projection
        projection_image_sizes[frame_index] = image_size
    centers.flush()
    covariance.flush()
    opacity.flush()
    projection_world_view.flush()
    projection_full.flush()
    projection_image_sizes.flush()
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
        "projection_fields": ["projection_world_view", "projection_full", "projection_image_sizes"],
        "projection_mode": "4dgs_renderer_camera",
        "input_backbone": "upstream_4dgaussians",
        "elapsed_seconds": float(time.monotonic() - start),
    }
    _write_json(output_dir / GEOMETRY_MANIFEST_NAME, manifest)
    return output_dir
