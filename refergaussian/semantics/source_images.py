from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_hypernerf_all_images(dataset_dir: Path) -> list[dict[str, Any]]:
    dataset_payload = _read_json(dataset_dir / "dataset.json")
    metadata_payload = _read_json(dataset_dir / "metadata.json")
    all_ids = list(dataset_payload.get("ids", []))
    if not all_ids:
        raise FileNotFoundError(f"No ids found in {dataset_dir / 'dataset.json'}")
    rgb_root = dataset_dir / "rgb"
    scale_dir = None
    for candidate in ("2x", "1x", "4x"):
        if (rgb_root / candidate).is_dir():
            scale_dir = rgb_root / candidate
            break
    if scale_dir is None:
        raise FileNotFoundError(f"No RGB scale dir found under {rgb_root}")
    max_time = max(float(metadata_payload[item]["warp_id"]) for item in all_ids)
    if max_time <= 0.0:
        max_time = 1.0
    entries: list[dict[str, Any]] = []
    for frame_index, image_id in enumerate(all_ids):
        image_path = scale_dir / f"{image_id}.png"
        if not image_path.exists():
            continue
        entries.append(
            {
                "frame_index": int(frame_index),
                "image_id": str(image_id),
                "image_path": str(image_path),
                "time_value": float(metadata_payload[image_id]["warp_id"]) / max_time,
                "dataset_type": "hypernerf",
            }
        )
    return entries


def _preferred_dynerf_camera_dir(dataset_dir: Path) -> Path:
    candidates = []
    for camera_dir in sorted(dataset_dir.glob("cam*")):
        image_dir = camera_dir / "images"
        image_count = len(list(image_dir.glob("*.png"))) if image_dir.is_dir() else 0
        if image_count <= 0:
            continue
        preferred = 1 if camera_dir.name == "cam00" else 0
        candidates.append((preferred, image_count, camera_dir.name, camera_dir))
    if not candidates:
        raise FileNotFoundError(f"No DyNeRF camera image directories found under {dataset_dir}")
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return candidates[0][3]


def _resolve_dynerf_all_images(dataset_dir: Path) -> list[dict[str, Any]]:
    camera_dir = _preferred_dynerf_camera_dir(dataset_dir)
    image_dir = camera_dir / "images"
    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG frames found under {image_dir}")
    denom = max(len(image_paths) - 1, 1)
    entries: list[dict[str, Any]] = []
    for frame_index, image_path in enumerate(image_paths):
        entries.append(
            {
                "frame_index": int(frame_index),
                "image_id": str(image_path.stem),
                "image_path": str(image_path),
                "time_value": float(frame_index) / float(denom),
                "dataset_type": "dynerf",
                "camera_name": str(camera_dir.name),
            }
        )
    return entries


def resolve_dataset_image_entries(dataset_dir: str | Path) -> list[dict[str, Any]]:
    dataset_dir = Path(dataset_dir)
    if (dataset_dir / "dataset.json").exists() and (dataset_dir / "metadata.json").exists():
        return _resolve_hypernerf_all_images(dataset_dir)
    if any((dataset_dir / name / "images").is_dir() for name in ("cam00", "cam01", "cam02")) or list(dataset_dir.glob("cam*/images")):
        return _resolve_dynerf_all_images(dataset_dir)
    raise FileNotFoundError(
        f"Unsupported dataset layout under {dataset_dir}. "
        "Expected HyperNeRF dataset.json/metadata.json or DyNeRF camXX/images directories."
    )


def resolve_dataset_time_values(dataset_dir: str | Path) -> tuple[list[str], np.ndarray]:
    entries = resolve_dataset_image_entries(dataset_dir)
    image_ids = [str(entry["image_id"]) for entry in entries]
    time_values = np.asarray([float(entry["time_value"]) for entry in entries], dtype=np.float32)
    return image_ids, time_values
