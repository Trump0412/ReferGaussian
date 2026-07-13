from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy import ndimage as _ndimage
except Exception:  # pragma: no cover - optional scipy fallback
    _ndimage = None


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _norm_text(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("-", " ").replace("_", " ").split())


def _status_tags_from_phrase(phrase: str) -> list[str]:
    text = _norm_text(phrase)
    tags: list[str] = []
    for token in (
        "empty",
        "full",
        "opened",
        "closed",
        "broken",
        "complete",
        "intact",
        "left",
        "right",
    ):
        if token in text:
            tags.append(token)
    return sorted(set(tags))


def build_mask_regions(mask: np.ndarray, core_kernel: int = 5, outer_kernel: int = 15) -> dict[str, np.ndarray]:
    full = np.asarray(mask, dtype=bool)
    if full.ndim != 2:
        raise ValueError(f"mask must be HxW, got shape {full.shape}")
    if _ndimage is None:
        # Conservative fallback when scipy is unavailable.
        return {
            "full": full,
            "core": full.copy(),
            "boundary": np.zeros_like(full, dtype=bool),
            "outer": np.zeros_like(full, dtype=bool),
        }

    core_kernel = max(int(core_kernel), 1)
    outer_kernel = max(int(outer_kernel), core_kernel + 2)
    core_structure = np.ones((core_kernel, core_kernel), dtype=bool)
    outer_structure = np.ones((outer_kernel, outer_kernel), dtype=bool)

    core = _ndimage.binary_erosion(full, structure=core_structure)
    if not np.any(core):
        core = _ndimage.binary_erosion(full, structure=np.ones((3, 3), dtype=bool))
    if not np.any(core):
        core = full.copy()

    dilate_small = _ndimage.binary_dilation(full, structure=core_structure)
    erode_small = _ndimage.binary_erosion(full, structure=core_structure)
    boundary = np.logical_and(dilate_small, np.logical_not(erode_small))

    dilate_large = _ndimage.binary_dilation(full, structure=outer_structure)
    outer = np.logical_and(dilate_large, np.logical_not(dilate_small))
    return {
        "full": full,
        "core": np.asarray(core, dtype=bool),
        "boundary": np.asarray(boundary, dtype=bool),
        "outer": np.asarray(outer, dtype=bool),
    }


@dataclass
class Stage1MaskFrame:
    object_id: int
    base_phrase: str
    frame_index: int
    image_id: str
    time_value: float
    mask_path: str
    bbox_xyxy: list[float] | None
    aliases: list[str]
    status_tags: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "object_id": int(self.object_id),
            "base_phrase": str(self.base_phrase),
            "frame_index": int(self.frame_index),
            "image_id": str(self.image_id),
            "time_value": float(self.time_value),
            "mask_path": str(self.mask_path),
            "bbox_xyxy": None if self.bbox_xyxy is None else [float(v) for v in self.bbox_xyxy],
            "aliases": [str(v) for v in self.aliases],
            "status_tags": [str(v) for v in self.status_tags],
        }


@dataclass
class Stage1Object:
    object_id: int
    base_phrase: str
    aliases: list[str]
    status_tags: list[str]
    track_id: str
    frames: list[Stage1MaskFrame]

    def to_json(self) -> dict[str, Any]:
        return {
            "object_id": int(self.object_id),
            "base_phrase": str(self.base_phrase),
            "aliases": [str(v) for v in self.aliases],
            "status_tags": [str(v) for v in self.status_tags],
            "track_id": str(self.track_id),
            "frames": [frame.to_json() for frame in self.frames],
        }


class SurfaceMaskField(torch.nn.Module):
    def __init__(
        self,
        num_gaussians: int,
        num_instances: int,
        feature_dim: int = 32,
        mode: str = "direct_logits",
    ):
        super().__init__()
        self.num_gaussians = int(num_gaussians)
        self.num_instances = int(num_instances)
        self.feature_dim = int(feature_dim)
        self.mode = str(mode)
        self.instance_logits = torch.nn.Parameter(
            torch.zeros(self.num_gaussians, self.num_instances + 1, dtype=torch.float32)
        )
        self.surface_features = torch.nn.Parameter(
            torch.randn(self.num_gaussians, self.feature_dim, dtype=torch.float32) * 0.01
        )
        self.feature_classifier = torch.nn.Linear(self.feature_dim, self.num_instances + 1, bias=True)
        torch.nn.init.zeros_(self.feature_classifier.weight)
        torch.nn.init.zeros_(self.feature_classifier.bias)
        self.feature_logit_scale = torch.nn.Parameter(torch.tensor(0.35, dtype=torch.float32))

    def logits(self) -> torch.Tensor:
        normalized_features = F.normalize(self.surface_features, dim=-1)
        feature_logits = self.feature_classifier(normalized_features)
        scale = self.feature_logit_scale.clamp(0.0, 2.0)
        return self.instance_logits + scale * feature_logits

    def probs(self) -> torch.Tensor:
        return torch.softmax(self.logits(), dim=-1)

    def object_probs(self) -> torch.Tensor:
        return self.probs()[:, 1:]

    def background_probs(self) -> torch.Tensor:
        return self.probs()[:, 0]

    def normalized_surface_features(self) -> torch.Tensor:
        return F.normalize(self.surface_features, dim=-1)


def _objects_from_track_payload(track_payload: dict[str, Any]) -> list[Stage1Object]:
    phrase_meta_by_key = {
        (str(item.get("phrase", "")).strip(), int(item.get("object_id", index))): item
        for index, item in enumerate(track_payload.get("phrases", []))
    }
    objects: list[Stage1Object] = []
    for track_index, track in enumerate(track_payload.get("tracks", [])):
        if str(track.get("status", "")).strip().lower() != "seeded":
            continue
        phrase = str(track.get("phrase", "")).strip()
        if not phrase:
            continue
        object_id = int(track.get("object_id", track_index + 1))
        phrase_meta = phrase_meta_by_key.get((phrase, object_id), {})
        aliases = [phrase]
        detections = phrase_meta.get("detections", [])
        if detections:
            aliases.extend(str(item.get("label", "")).strip() for item in detections if str(item.get("label", "")).strip())
        aliases = sorted({value for value in aliases if value})
        status_tags = _status_tags_from_phrase(phrase)
        frames: list[Stage1MaskFrame] = []
        for frame in track.get("frames", []):
            if not bool(frame.get("active")):
                continue
            mask_path = str(frame.get("mask_path") or "").strip()
            if not mask_path:
                continue
            frames.append(
                Stage1MaskFrame(
                    object_id=object_id,
                    base_phrase=phrase,
                    frame_index=int(frame["frame_index"]),
                    image_id=str(frame["image_id"]),
                    time_value=float(frame["time_value"]),
                    mask_path=mask_path,
                    bbox_xyxy=None if frame.get("bbox_xyxy") is None else [float(v) for v in frame["bbox_xyxy"]],
                    aliases=aliases,
                    status_tags=status_tags,
                )
            )
        if not frames:
            continue
        frames.sort(key=lambda item: item.frame_index)
        objects.append(
            Stage1Object(
                object_id=object_id,
                base_phrase=phrase,
                aliases=aliases,
                status_tags=status_tags,
                track_id=f"gsam_track_{track_index:04d}",
                frames=frames,
            )
        )
    objects.sort(key=lambda item: (item.object_id, item.base_phrase))
    return objects


def _objects_from_object_map(root: Path, payload: dict[str, Any]) -> list[Stage1Object]:
    objects: list[Stage1Object] = []
    for item in payload.get("objects", []):
        frames: list[Stage1MaskFrame] = []
        for frame in item.get("frames", []):
            mask_path = str(frame.get("mask_path") or "").strip()
            if not mask_path:
                continue
            if not Path(mask_path).is_absolute():
                mask_path = str((root / mask_path).resolve())
            frames.append(
                Stage1MaskFrame(
                    object_id=int(item["object_id"]),
                    base_phrase=str(item["base_phrase"]),
                    frame_index=int(frame["frame_index"]),
                    image_id=str(frame["image_id"]),
                    time_value=float(frame["time_value"]),
                    mask_path=mask_path,
                    bbox_xyxy=None if frame.get("bbox_xyxy") is None else [float(v) for v in frame["bbox_xyxy"]],
                    aliases=[str(v) for v in item.get("aliases", [])],
                    status_tags=[str(v) for v in item.get("status_tags", [])],
                )
            )
        if not frames:
            continue
        frames.sort(key=lambda entry: entry.frame_index)
        objects.append(
            Stage1Object(
                object_id=int(item["object_id"]),
                base_phrase=str(item["base_phrase"]),
                aliases=[str(v) for v in item.get("aliases", [])],
                status_tags=[str(v) for v in item.get("status_tags", [])],
                track_id=str(item.get("track_id", f"object_{int(item['object_id']):04d}")),
                frames=frames,
            )
        )
    objects.sort(key=lambda item: (item.object_id, item.base_phrase))
    return objects


def load_stage1_objects(stage1_root: str | Path) -> tuple[list[Stage1Object], dict[str, Any]]:
    stage1_root = Path(stage1_root)
    if stage1_root.is_file():
        if stage1_root.name == "grounded_sam2_query_tracks.json":
            payload = _read_json(stage1_root)
            objects = _objects_from_track_payload(payload)
            return objects, stage1_object_payload(objects, source_payload=payload, source_root=stage1_root.parent)
        if stage1_root.name == "object_id_map.json":
            payload = _read_json(stage1_root)
            objects = _objects_from_object_map(stage1_root.parent, payload)
            return objects, payload
        raise FileNotFoundError(f"Unsupported Stage1 file: {stage1_root}")

    object_map_path = stage1_root / "object_id_map.json"
    tracks_path = stage1_root / "grounded_sam2_query_tracks.json"
    if object_map_path.exists():
        payload = _read_json(object_map_path)
        objects = _objects_from_object_map(stage1_root, payload)
        return objects, payload
    if tracks_path.exists():
        payload = _read_json(tracks_path)
        objects = _objects_from_track_payload(payload)
        return objects, stage1_object_payload(objects, source_payload=payload, source_root=stage1_root)
    grounded_tracks = stage1_root / "grounded_sam2" / "grounded_sam2_query_tracks.json"
    if grounded_tracks.exists():
        payload = _read_json(grounded_tracks)
        objects = _objects_from_track_payload(payload)
        return objects, stage1_object_payload(objects, source_payload=payload, source_root=grounded_tracks.parent)
    raise FileNotFoundError(
        f"Unable to resolve Stage1 objects under {stage1_root}. "
        "Expected object_id_map.json or grounded_sam2_query_tracks.json."
    )


def stage1_object_payload(
    objects: list[Stage1Object],
    source_payload: dict[str, Any] | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_type": "grounded_sam2_tracks" if source_payload is not None else "surface_stage1_objects",
        "source_root": None if source_root is None else str(source_root),
        "num_objects": int(len(objects)),
        "objects": [item.to_json() for item in objects],
    }


def write_stage1_objects(output_root: str | Path, objects: list[Stage1Object], payload: dict[str, Any] | None = None) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    object_map = payload if payload is not None else stage1_object_payload(objects, source_payload=None, source_root=output_root)
    _write_json(output_root / "object_id_map.json", object_map)
    return output_root / "object_id_map.json"


def save_surface_field_checkpoint(
    output_path: str | Path,
    field: SurfaceMaskField,
    object_payload: dict[str, Any],
    extra_state: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "num_gaussians": int(field.num_gaussians),
        "num_instances": int(field.num_instances),
        "feature_dim": int(field.feature_dim),
        "mode": str(field.mode),
        "state_dict": field.state_dict(),
        "object_id_map": object_payload,
        "extra_state": extra_state or {},
    }
    torch.save(payload, output_path)
    return output_path


def load_surface_field_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[SurfaceMaskField, dict[str, Any], dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=map_location)
    field = SurfaceMaskField(
        num_gaussians=int(payload["num_gaussians"]),
        num_instances=int(payload["num_instances"]),
        feature_dim=int(payload.get("feature_dim", 32)),
        mode=str(payload.get("mode", "direct_logits")),
    )
    missing, unexpected = field.load_state_dict(payload["state_dict"], strict=False)
    allowed_missing = {
        "feature_classifier.weight",
        "feature_classifier.bias",
        "feature_logit_scale",
    }
    if unexpected or any(key not in allowed_missing for key in missing):
        raise RuntimeError(
            f"Unexpected SurfaceMaskField checkpoint mismatch for {checkpoint_path}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return field, dict(payload.get("object_id_map", {})), dict(payload.get("extra_state", {}))
