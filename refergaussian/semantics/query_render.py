from __future__ import annotations

import colorsys
import importlib.util
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import imageio.v2 as iio_v2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .source_images import resolve_dataset_image_entries

try:
    from scipy import ndimage as _ndimage
except Exception:  # pragma: no cover - optional fallback when scipy is unavailable
    _ndimage = None

try:
    import torch

    from .semantic_renderer import prepare_semantic_frame_inputs, render_selection_mask
except Exception:  # pragma: no cover - keep point projection available in light envs
    torch = None
    prepare_semantic_frame_inputs = None
    render_selection_mask = None

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_ROOT = _PROJECT_ROOT / "external" / "4DGaussians"


def _load_camera_class():
    utils_path = _UPSTREAM_ROOT / "scene" / "utils.py"
    spec = importlib.util.spec_from_file_location("refergaussian_scene_utils", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Camera utilities from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Camera


_CAMERA_CLASS: Any | None = None


def _camera_class():
    """Load upstream camera support only when 3D projection is requested."""
    global _CAMERA_CLASS
    if _CAMERA_CLASS is None:
        _CAMERA_CLASS = _load_camera_class()
    return _CAMERA_CLASS


ROLE_COLORS = {
    "patient": (255, 210, 64),
    "tool": (64, 224, 255),
    "agent": (255, 96, 96),
    "entity": (255, 64, 196),
    "other": (180, 180, 180),
}


# v4 remains available for reproducing earlier exploratory runs. These profiles
# are the auditable release path: a selected Gaussian entity is rendered first,
# then clipped by a synchronized Stage-1 boundary neighborhood. The Stage-1
# mask is never a standalone final prediction.
_FORMAL_BOUNDARY_GATED_PROFILES = frozenset(
    {
        "public_time_boundary_gated_v5",
        "boundary_gated_gaussian_v5",
        "r4d_boundary_gated_v5",
        "r4d_multi_instance_boundary_v6",
    }
)
_COVERAGE_RENDER_PROFILES = frozenset(
    {
        "public_time_shape_v3",
        "mask_coverage_refine_v3",
        "public_time_shape_v4_recall",
        "mask_coverage_refine_v4",
        "shape_v4_recall",
        "r4d_shape_v4_recall",
        "r4d_time_shape_v4_recall",
        *_FORMAL_BOUNDARY_GATED_PROFILES,
    }
)


@dataclass
class TrackSample:
    entity_id: int
    time_values: np.ndarray
    centers: np.ndarray
    extents_min: np.ndarray
    extents_max: np.ndarray
    visibility: np.ndarray
    support_score: np.ndarray


@dataclass
class EntityCloud:
    entity_id: int
    sample_times: np.ndarray
    trajectories: np.ndarray
    gate: np.ndarray
    spatial_extent: np.ndarray
    spatial_scale: np.ndarray | None = None
    opacity_logit: np.ndarray | None = None
    opacity_source: str = "unavailable"
    alpha_relative_threshold: float | None = None
    alpha_absolute_threshold: float | None = None
    alpha_sigma_scale: float | None = None
    alpha_max_splat_radius: int | None = None


@dataclass
class QueryTrack:
    phrase: str
    frames: list[dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _normalize_track_state_mode(value: Any) -> str | None:
    text = " ".join(str(value).strip().lower().replace("-", "_").split())
    if not text:
        return None
    normalized = text.replace(" ", "_")
    if normalized in {"none", "null", "unknown", "n/a", "na"}:
        return None
    return normalized


def _selection_track_state_mode(selection_payload: dict[str, Any]) -> str | None:
    for key in ("track_state_mode", "state_mode", "query_state_mode"):
        normalized = _normalize_track_state_mode(selection_payload.get(key))
        if normalized:
            return normalized

    notes = str(selection_payload.get("notes", "")).strip()
    for marker in ("Track state mode=", "State mode="):
        if marker not in notes:
            continue
        tail = notes.split(marker, 1)[1]
        candidate = tail.split(";", 1)[0].strip()
        normalized = _normalize_track_state_mode(candidate)
        if normalized:
            return normalized

    contact_pair = selection_payload.get("contact_pair") or {}
    source = _normalize_track_state_mode(contact_pair.get("source"))
    if not source:
        return None
    if source.startswith("single_subject_track_"):
        suffix = source.removeprefix("single_subject_track_")
        return suffix or "support"
    if "support" in source:
        return "support"
    return None


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


def _odd_kernel(value: int) -> int:
    value = int(max(1, value))
    return value if value % 2 == 1 else value + 1


def _resolve_eval_profile(explicit_profile: str | None = None) -> str:
    for candidate in (
        explicit_profile,
        os.environ.get("REFERGAUSSIAN_QUERY_EVAL_PROFILE"),
        os.environ.get("QUERY_EVAL_PROFILE"),
    ):
        if candidate is None:
            continue
        normalized = " ".join(str(candidate).strip().lower().replace("-", "_").split())
        if normalized:
            return normalized
    return "default"


def _is_coverage_render_profile(eval_profile: str | None) -> bool:
    return _resolve_eval_profile(eval_profile) in _COVERAGE_RENDER_PROFILES


def _is_formal_boundary_gated_profile(eval_profile: str | None) -> bool:
    return _resolve_eval_profile(eval_profile) in _FORMAL_BOUNDARY_GATED_PROFILES


def _query_intent_mode(query_text: str, track_state_mode: str | None = None) -> str:
    text = " ".join(str(query_text).strip().lower().split())
    if not text:
        return "generic"
    if any(token in text for token in ("broken", "pieces", "piece", "fragment", "crumb", "split", "detached")):
        return "multi_component"
    if any(token in text for token in ("complete", "whole", "intact", "unbroken", "full")):
        return "single_component"
    normalized_state = _normalize_track_state_mode(track_state_mode)
    if normalized_state in {"support", "static_full_video"}:
        return "single_component"
    return "generic"


def _fusion_options_for_profile(eval_profile: str) -> dict[str, Any]:
    normalized = _resolve_eval_profile(eval_profile)
    options = {
        "profile": normalized,
        "support_kernel": _odd_kernel(_env_int("GS_QUERY_FUSE_SUPPORT_KERNEL", 11)),
        "state_kernel": _odd_kernel(_env_int("GS_QUERY_FUSE_STATE_KERNEL", 15)),
        "expand_kernel": _odd_kernel(_env_int("GS_QUERY_FUSE_EXPAND_KERNEL", 21)),
        "close_kernel": _odd_kernel(_env_int("GS_QUERY_FUSE_CLOSE_KERNEL", 7)),
        "open_kernel": _odd_kernel(_env_int("GS_QUERY_FUSE_OPEN_KERNEL", 3)),
        "min_component_area": _env_int("GS_QUERY_FUSE_MIN_COMPONENT_AREA", 72, minimum=1),
        "w_prev_iou": _env_float("GS_QUERY_FUSE_PREV_IOU_WEIGHT", 0.24, minimum=0.0),
        "w_query": _env_float("GS_QUERY_FUSE_QUERY_WEIGHT", 0.40, minimum=0.0),
        "w_cloud": _env_float("GS_QUERY_FUSE_CLOUD_WEIGHT", 0.30, minimum=0.0),
        "clip_to_query_track": _env_flag("GS_QUERY_FUSE_CLIP_TO_QUERY_TRACK", False),
        "clip_kernel": _odd_kernel(_env_int("GS_QUERY_FUSE_CLIP_KERNEL", 9)),
        "recovery_kernel": _odd_kernel(_env_int("GS_QUERY_FUSE_RECOVERY_KERNEL", 1)),
        "recovery_min_clip_area_ratio": _env_float("GS_QUERY_FUSE_RECOVERY_MIN_CLIP_AREA_RATIO", 0.0, minimum=0.0),
        "recovery_min_query_recall": _env_float("GS_QUERY_FUSE_RECOVERY_MIN_QUERY_RECALL", 0.0, minimum=0.0, maximum=1.0),
        "prefer_clipped_cloud": _env_flag("GS_QUERY_FUSE_PREFER_CLIPPED_CLOUD", False),
        "allow_query_supported_by_cloud": _env_flag("GS_QUERY_FUSE_ALLOW_QUERY_SUPPORTED_BY_CLOUD", False),
        "query_support_kernel": _odd_kernel(_env_int("GS_QUERY_FUSE_QUERY_SUPPORT_KERNEL", 17, minimum=1)),
        "align_query_track_to_cloud": _env_flag("GS_QUERY_ALIGN_TRACK_TO_CLOUD", True),
        "allow_cloud_only_with_query": _env_flag("GS_QUERY_ALLOW_CLOUD_ONLY_WITH_QUERY", True),
        "strict_gaussian_projection": False,
        # Final binary masks should remain 3D-entity grounded unless a legacy
        # 2D-track fallback is explicitly re-enabled for debugging.
        "allow_direct_query_track": _env_flag("GS_QUERY_ALLOW_DIRECT_2D_MASKS", False),
        "final_erode_kernel": _odd_kernel(_env_int("GS_QUERY_FINAL_ERODE_KERNEL", 1, minimum=1)),
    }
    if normalized == "viou_boost_v1":
        # Keep defaults from environment if user already pinned them.
        options["support_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_SUPPORT_KERNEL", 13))
        options["state_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_STATE_KERNEL", 17))
        options["expand_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_EXPAND_KERNEL", 25))
        options["close_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_CLOSE_KERNEL", 9))
        options["open_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_OPEN_KERNEL", 5))
        options["min_component_area"] = _env_int("GS_QUERY_FUSE_MIN_COMPONENT_AREA", 96, minimum=1)
        options["w_prev_iou"] = _env_float("GS_QUERY_FUSE_PREV_IOU_WEIGHT", 0.28, minimum=0.0)
        options["w_query"] = _env_float("GS_QUERY_FUSE_QUERY_WEIGHT", 0.42, minimum=0.0)
        options["w_cloud"] = _env_float("GS_QUERY_FUSE_CLOUD_WEIGHT", 0.30, minimum=0.0)
    if normalized in {"boundary_refine_v1", "public_boundary_v1", "mask_boundary_refine"}:
        options["support_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_SUPPORT_KERNEL", 7))
        options["state_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_STATE_KERNEL", 9))
        options["expand_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_EXPAND_KERNEL", 11))
        options["close_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_CLOSE_KERNEL", 5))
        options["open_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_OPEN_KERNEL", 3))
        options["min_component_area"] = _env_int("GS_QUERY_FUSE_MIN_COMPONENT_AREA", 48, minimum=1)
        options["w_prev_iou"] = _env_float("GS_QUERY_FUSE_PREV_IOU_WEIGHT", 0.18, minimum=0.0)
        options["w_query"] = _env_float("GS_QUERY_FUSE_QUERY_WEIGHT", 0.58, minimum=0.0)
        options["w_cloud"] = _env_float("GS_QUERY_FUSE_CLOUD_WEIGHT", 0.18, minimum=0.0)
        options["clip_to_query_track"] = _env_flag("GS_QUERY_FUSE_CLIP_TO_QUERY_TRACK", False)
        options["clip_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_CLIP_KERNEL", 9))
        options["recovery_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_RECOVERY_KERNEL", 35))
        options["recovery_min_clip_area_ratio"] = _env_float("GS_QUERY_FUSE_RECOVERY_MIN_CLIP_AREA_RATIO", 0.45, minimum=0.0)
        options["recovery_min_query_recall"] = _env_float("GS_QUERY_FUSE_RECOVERY_MIN_QUERY_RECALL", 0.22, minimum=0.0, maximum=1.0)
        options["prefer_clipped_cloud"] = _env_flag("GS_QUERY_FUSE_PREFER_CLIPPED_CLOUD", False)
        options["allow_query_supported_by_cloud"] = _env_flag("GS_QUERY_FUSE_ALLOW_QUERY_SUPPORTED_BY_CLOUD", False)
        options["query_support_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_QUERY_SUPPORT_KERNEL", 17, minimum=1))
        options["align_query_track_to_cloud"] = _env_flag("GS_QUERY_ALIGN_TRACK_TO_CLOUD", False)
        options["allow_cloud_only_with_query"] = _env_flag("GS_QUERY_ALLOW_CLOUD_ONLY_WITH_QUERY", True)
        options["strict_gaussian_projection"] = _env_flag("GS_QUERY_STRICT_GAUSSIAN_PROJECTION", True)
    if normalized in {"boundary_shape_v2", "mask_shape_refine_v2"}:
        options["support_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_SUPPORT_KERNEL", 7))
        options["state_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_STATE_KERNEL", 9))
        options["expand_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_EXPAND_KERNEL", 11))
        options["close_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_CLOSE_KERNEL", 5))
        options["open_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_OPEN_KERNEL", 3))
        options["min_component_area"] = _env_int("GS_QUERY_FUSE_MIN_COMPONENT_AREA", 48, minimum=1)
        options["w_prev_iou"] = _env_float("GS_QUERY_FUSE_PREV_IOU_WEIGHT", 0.18, minimum=0.0)
        options["w_query"] = _env_float("GS_QUERY_FUSE_QUERY_WEIGHT", 0.58, minimum=0.0)
        options["w_cloud"] = _env_float("GS_QUERY_FUSE_CLOUD_WEIGHT", 0.18, minimum=0.0)
        options["clip_to_query_track"] = _env_flag("GS_QUERY_FUSE_CLIP_TO_QUERY_TRACK", False)
        options["clip_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_CLIP_KERNEL", 9))
        options["recovery_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_RECOVERY_KERNEL", 35))
        options["recovery_min_clip_area_ratio"] = _env_float("GS_QUERY_FUSE_RECOVERY_MIN_CLIP_AREA_RATIO", 0.45, minimum=0.0)
        options["recovery_min_query_recall"] = _env_float("GS_QUERY_FUSE_RECOVERY_MIN_QUERY_RECALL", 0.22, minimum=0.0, maximum=1.0)
        options["prefer_clipped_cloud"] = _env_flag("GS_QUERY_FUSE_PREFER_CLIPPED_CLOUD", False)
        options["allow_query_supported_by_cloud"] = _env_flag("GS_QUERY_FUSE_ALLOW_QUERY_SUPPORTED_BY_CLOUD", False)
        options["query_support_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_QUERY_SUPPORT_KERNEL", 17, minimum=1))
        options["align_query_track_to_cloud"] = _env_flag("GS_QUERY_ALIGN_TRACK_TO_CLOUD", False)
        options["allow_cloud_only_with_query"] = _env_flag("GS_QUERY_ALLOW_CLOUD_ONLY_WITH_QUERY", True)
        options["strict_gaussian_projection"] = _env_flag("GS_QUERY_STRICT_GAUSSIAN_PROJECTION", True)
    if _is_coverage_render_profile(normalized):
        options["support_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_SUPPORT_KERNEL", 7))
        options["state_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_STATE_KERNEL", 9))
        options["expand_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_EXPAND_KERNEL", 11))
        options["close_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_CLOSE_KERNEL", 5))
        options["open_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_OPEN_KERNEL", 3))
        options["min_component_area"] = _env_int("GS_QUERY_FUSE_MIN_COMPONENT_AREA", 48, minimum=1)
        options["w_prev_iou"] = _env_float("GS_QUERY_FUSE_PREV_IOU_WEIGHT", 0.18, minimum=0.0)
        options["w_query"] = _env_float("GS_QUERY_FUSE_QUERY_WEIGHT", 0.58, minimum=0.0)
        options["w_cloud"] = _env_float("GS_QUERY_FUSE_CLOUD_WEIGHT", 0.18, minimum=0.0)
        options["clip_to_query_track"] = _env_flag("GS_QUERY_FUSE_CLIP_TO_QUERY_TRACK", False)
        options["clip_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_CLIP_KERNEL", 9))
        options["recovery_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_RECOVERY_KERNEL", 35))
        options["recovery_min_clip_area_ratio"] = _env_float("GS_QUERY_FUSE_RECOVERY_MIN_CLIP_AREA_RATIO", 0.45, minimum=0.0)
        options["recovery_min_query_recall"] = _env_float("GS_QUERY_FUSE_RECOVERY_MIN_QUERY_RECALL", 0.22, minimum=0.0, maximum=1.0)
        options["prefer_clipped_cloud"] = _env_flag("GS_QUERY_FUSE_PREFER_CLIPPED_CLOUD", False)
        options["allow_query_supported_by_cloud"] = _env_flag("GS_QUERY_FUSE_ALLOW_QUERY_SUPPORTED_BY_CLOUD", False)
        options["query_support_kernel"] = _odd_kernel(_env_int("GS_QUERY_FUSE_QUERY_SUPPORT_KERNEL", 17, minimum=1))
        options["align_query_track_to_cloud"] = _env_flag("GS_QUERY_ALIGN_TRACK_TO_CLOUD", False)
        options["allow_cloud_only_with_query"] = _env_flag("GS_QUERY_ALLOW_CLOUD_ONLY_WITH_QUERY", True)
        options["strict_gaussian_projection"] = _env_flag("GS_QUERY_STRICT_GAUSSIAN_PROJECTION", True)
    return options


def _apply_render_profile_env_defaults(eval_profile: str) -> None:
    normalized = _resolve_eval_profile(eval_profile)
    if normalized not in {"boundary_refine_v1", "public_boundary_v1", "mask_boundary_refine",
                          "boundary_shape_v2", "mask_shape_refine_v2",
                          "public_time_shape_v3", "mask_coverage_refine_v3",
                          *_COVERAGE_RENDER_PROFILES}:
        return
    if _is_coverage_render_profile(normalized):
        is_formal_boundary_profile = _is_formal_boundary_gated_profile(normalized)
        defaults = {
            "REFERGAUSSIAN_QUERY_EVAL_PROFILE": normalized if is_formal_boundary_profile else "public_time_shape_v4_recall",
            "GS_QUERY_TRACK_FALLBACK_SCALE": "1.0",
            "GS_QUERY_TRACK_STRICT_FALLBACK_SCALE": "1.0",
            "GS_QUERY_ALLOW_STALE_STAGE1_BOUNDARY": "0" if is_formal_boundary_profile else "1",
            "GS_QUERY_CLOUD_BOX_EXPANSION": "0",
            "GS_QUERY_CLOUD_RENDER_MODE": "gaussian_alpha",
            "GS_QUERY_ALPHA_GATE_THRESHOLD": "0.01",
            "GS_QUERY_ALPHA_REL_THRESHOLD": "0.18",
            "GS_QUERY_ALPHA_ABS_THRESHOLD": "0.015",
            "GS_QUERY_ALPHA_SIGMA_SCALE": "1.0",
            "GS_QUERY_ALPHA_MAX_SPLAT_RADIUS": "18",
            "GS_QUERY_ALPHA_POSTFILTER_KERNEL": "1",
            "GS_QUERY_ALPHA_REQUIRE_SUCCESS": "1",
            "GS_QUERY_ALPHA_REQUIRE_OPACITY": "1",
            "GS_QUERY_CLOUD_POINT_RADIUS_SCALE": "3.00",
            "GS_QUERY_CLOUD_POINT_RADIUS_MIN": "1",
            "GS_QUERY_CLOUD_POINT_RADIUS_MAX": "84",
            "GS_QUERY_CLOUD_POSTFILTER_KERNEL": "15",
            "GS_QUERY_CLOUD_FILL_CONVEX_HULL": "0",
            "GS_QUERY_CLOUD_COMPONENT_HULL": "1",
            "GS_QUERY_CLOUD_COMPONENT_HULL_MAX_AREA_MULTIPLIER": "11.0",
            "GS_QUERY_CLOUD_COMPONENT_HULL_MIN_POINTS": "6",
            "GS_QUERY_CLOUD_COMPONENT_CLOSE_KERNEL": "17",
            "GS_QUERY_CLOUD_COMPONENT_FILL_HOLES": "1",
            "GS_QUERY_CLOUD_NO_STAGE1_RADIUS_MULT": "1.0",
            "GS_QUERY_FUSE_CLIP_KERNEL": "13",
            "GS_QUERY_FUSE_RECOVERY_KERNEL": "25",
            "GS_QUERY_FUSE_RECOVERY_MIN_CLIP_AREA_RATIO": "0.25",
            "GS_QUERY_FUSE_RECOVERY_MIN_QUERY_RECALL": "0.12",
            "GS_QUERY_FUSE_CLIP_TO_QUERY_TRACK": "1",
            "GS_QUERY_FUSE_PREFER_CLIPPED_CLOUD": "1",
            "GS_QUERY_FUSE_ALLOW_QUERY_SUPPORTED_BY_CLOUD": "0",
            "GS_QUERY_FUSE_QUERY_SUPPORT_KERNEL": "17",
            "GS_QUERY_ALIGN_TRACK_TO_CLOUD": "0",
            "GS_QUERY_ALLOW_CLOUD_ONLY_WITH_QUERY": "0",
            "GS_QUERY_STRICT_GAUSSIAN_PROJECTION": "1",
            "GS_QUERY_REQUIRE_STAGE1_TRACKS": "1",
            "GS_QUERY_REQUIRE_SYNCHRONIZED_STAGE1_BOUNDARY": "1" if is_formal_boundary_profile else "0",
            "GS_QUERY_EXPORT_ENTITY_LIFECYCLE": "1",
        }
    elif normalized in {"public_time_shape_v3", "mask_coverage_refine_v3"}:
        defaults = {
            "REFERGAUSSIAN_QUERY_EVAL_PROFILE": "public_time_shape_v3",
            "GS_QUERY_TRACK_FALLBACK_SCALE": "1.0",
            "GS_QUERY_TRACK_STRICT_FALLBACK_SCALE": "1.0",
            "GS_QUERY_CLOUD_BOX_EXPANSION": "0",
            "GS_QUERY_CLOUD_RENDER_MODE": "component_shape_hull",
            "GS_QUERY_CLOUD_POINT_RADIUS_SCALE": "0.75",
            "GS_QUERY_CLOUD_POINT_RADIUS_MIN": "1",
            "GS_QUERY_CLOUD_POINT_RADIUS_MAX": "24",
            "GS_QUERY_CLOUD_POSTFILTER_KERNEL": "5",
            "GS_QUERY_CLOUD_FILL_CONVEX_HULL": "0",
            "GS_QUERY_CLOUD_COMPONENT_HULL": "1",
            "GS_QUERY_CLOUD_COMPONENT_HULL_MAX_AREA_MULTIPLIER": "4.0",
            "GS_QUERY_CLOUD_COMPONENT_HULL_MIN_POINTS": "8",
            "GS_QUERY_CLOUD_COMPONENT_CLOSE_KERNEL": "7",
            "GS_QUERY_CLOUD_COMPONENT_FILL_HOLES": "1",
            "GS_QUERY_CLOUD_NO_STAGE1_RADIUS_MULT": "1.0",
            "GS_QUERY_FUSE_RECOVERY_KERNEL": "35",
            "GS_QUERY_FUSE_RECOVERY_MIN_CLIP_AREA_RATIO": "0.45",
            "GS_QUERY_FUSE_RECOVERY_MIN_QUERY_RECALL": "0.22",
            "GS_QUERY_FUSE_CLIP_TO_QUERY_TRACK": "0",
            "GS_QUERY_FUSE_ALLOW_QUERY_SUPPORTED_BY_CLOUD": "0",
            "GS_QUERY_FUSE_QUERY_SUPPORT_KERNEL": "17",
            "GS_QUERY_ALIGN_TRACK_TO_CLOUD": "0",
            "GS_QUERY_STRICT_GAUSSIAN_PROJECTION": "1",
            "GS_QUERY_EXPORT_ENTITY_LIFECYCLE": "1",
        }
    elif normalized in {"boundary_shape_v2", "mask_shape_refine_v2"}:
        defaults = {
            "REFERGAUSSIAN_QUERY_EVAL_PROFILE": "boundary_shape_v2",
            "GS_QUERY_TRACK_FALLBACK_SCALE": "1.0",
            "GS_QUERY_TRACK_STRICT_FALLBACK_SCALE": "1.0",
            "GS_QUERY_CLOUD_BOX_EXPANSION": "0",
            "GS_QUERY_CLOUD_RENDER_MODE": "point_hull",
            "GS_QUERY_CLOUD_POINT_RADIUS_SCALE": "0.50",
            "GS_QUERY_CLOUD_POINT_RADIUS_MIN": "1",
            "GS_QUERY_CLOUD_POINT_RADIUS_MAX": "8",
            "GS_QUERY_CLOUD_POSTFILTER_KERNEL": "5",
            "GS_QUERY_CLOUD_FILL_CONVEX_HULL": "1",
            "GS_QUERY_CLOUD_HULL_MAX_AREA_MULTIPLIER": "6",
            "GS_QUERY_CLOUD_COMPONENT_HULL": "1",
            "GS_QUERY_CLOUD_COMPONENT_HULL_MAX_AREA_MULTIPLIER": "3.0",
            "GS_QUERY_CLOUD_COMPONENT_HULL_MIN_POINTS": "8",
            "GS_QUERY_CLOUD_COMPONENT_CLOSE_KERNEL": "5",
            "GS_QUERY_CLOUD_COMPONENT_FILL_HOLES": "1",
            "GS_QUERY_CLOUD_NO_STAGE1_RADIUS_MULT": "1.0",
            "GS_QUERY_FUSE_RECOVERY_KERNEL": "35",
            "GS_QUERY_FUSE_RECOVERY_MIN_CLIP_AREA_RATIO": "0.45",
            "GS_QUERY_FUSE_RECOVERY_MIN_QUERY_RECALL": "0.22",
            "GS_QUERY_FUSE_CLIP_TO_QUERY_TRACK": "0",
            "GS_QUERY_FUSE_ALLOW_QUERY_SUPPORTED_BY_CLOUD": "0",
            "GS_QUERY_FUSE_QUERY_SUPPORT_KERNEL": "17",
            "GS_QUERY_ALIGN_TRACK_TO_CLOUD": "0",
            "GS_QUERY_STRICT_GAUSSIAN_PROJECTION": "1",
            "GS_QUERY_EXPORT_ENTITY_LIFECYCLE": "1",
        }
    else:
        defaults = {
            "REFERGAUSSIAN_QUERY_EVAL_PROFILE": "boundary_refine_v1",
            "GS_QUERY_TRACK_FALLBACK_SCALE": "1.0",
            "GS_QUERY_TRACK_STRICT_FALLBACK_SCALE": "1.0",
            "GS_QUERY_CLOUD_BOX_EXPANSION": "0",
            "GS_QUERY_CLOUD_RENDER_MODE": "point_hull",
            "GS_QUERY_CLOUD_POINT_RADIUS_SCALE": "0.50",
            "GS_QUERY_CLOUD_POINT_RADIUS_MIN": "1",
            "GS_QUERY_CLOUD_POINT_RADIUS_MAX": "8",
            "GS_QUERY_CLOUD_POSTFILTER_KERNEL": "5",
            "GS_QUERY_CLOUD_FILL_CONVEX_HULL": "1",
            "GS_QUERY_CLOUD_HULL_MAX_AREA_MULTIPLIER": "6",
            "GS_QUERY_CLOUD_NO_STAGE1_RADIUS_MULT": "1.0",
            "GS_QUERY_FUSE_RECOVERY_KERNEL": "35",
            "GS_QUERY_FUSE_RECOVERY_MIN_CLIP_AREA_RATIO": "0.45",
            "GS_QUERY_FUSE_RECOVERY_MIN_QUERY_RECALL": "0.22",
            "GS_QUERY_FUSE_CLIP_TO_QUERY_TRACK": "0",
            "GS_QUERY_FUSE_ALLOW_QUERY_SUPPORTED_BY_CLOUD": "0",
            "GS_QUERY_FUSE_QUERY_SUPPORT_KERNEL": "17",
            "GS_QUERY_ALIGN_TRACK_TO_CLOUD": "0",
            "GS_QUERY_STRICT_GAUSSIAN_PROJECTION": "1",
            "GS_QUERY_EXPORT_ENTITY_LIFECYCLE": "1",
        }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    if _is_formal_boundary_gated_profile(normalized):
        # Formal profiles are intentionally not overrideable through inherited
        # shell state. The strict runner also clears inherited tuning, but this
        # guard keeps direct Python entry points on the same contract.
        os.environ.update(
            {
                "REFERGAUSSIAN_QUERY_EVAL_PROFILE": normalized,
                "GS_QUERY_ALLOW_STALE_STAGE1_BOUNDARY": "0",
                "GS_QUERY_REQUIRE_STAGE1_TRACKS": "1",
                "GS_QUERY_REQUIRE_SYNCHRONIZED_STAGE1_BOUNDARY": "1",
                "GS_QUERY_FUSE_CLIP_TO_QUERY_TRACK": "1",
                "GS_QUERY_FUSE_PREFER_CLIPPED_CLOUD": "1",
                "GS_QUERY_ALLOW_CLOUD_ONLY_WITH_QUERY": "0",
                "GS_QUERY_ALLOW_DIRECT_2D_MASKS": "0",
                "GS_QUERY_STRICT_GAUSSIAN_PROJECTION": "1",
            }
        )


def _merge_ranges(frame_indices: list[int]) -> list[list[int]]:
    if not frame_indices:
        return []
    sorted_indices = sorted(set(int(v) for v in frame_indices))
    merged: list[list[int]] = []
    start = sorted_indices[0]
    prev = sorted_indices[0]
    for value in sorted_indices[1:]:
        if value == prev + 1:
            prev = value
            continue
        merged.append([start, prev])
        start = value
        prev = value
    merged.append([start, prev])
    return merged


def _find_render_dir(run_dir: Path) -> Path:
    test_dir = run_dir / "test"
    if test_dir.is_symlink() and not test_dir.exists():
        test_dir.unlink()
    candidates = sorted(test_dir.glob("ours_*/renders"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"No ReferGaussian test render found under {test_dir}. "
        "Run scripts/eval.sh for this exact model directory before query inference; "
        "source RGB frames are never substituted for model renders."
    )


def _find_source_frame_dir(dataset_dir: Path, target_size: tuple[int, int]) -> Path:
    rgb_root = dataset_dir / "rgb"
    if not rgb_root.exists():
        raise FileNotFoundError(f"No rgb directory found under {dataset_dir}")
    candidates = [path for path in rgb_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No rgb scale directories found under {rgb_root}")

    best_dir = None
    best_score = None
    for directory in sorted(candidates):
        sample = next(iter(sorted(directory.glob("*.png"))), None)
        if sample is None:
            continue
        with Image.open(sample) as image:
            size = image.size
        score = abs(size[0] - target_size[0]) + abs(size[1] - target_size[1])
        if best_score is None or score < best_score:
            best_dir = directory
            best_score = score
    if best_dir is None:
        raise FileNotFoundError(f"No source images found under {rgb_root}")
    return best_dir


def _hypernerf_test_ids(dataset_dir: Path) -> tuple[list[str], np.ndarray] | None:
    """Return (test_ids, time_values) for HyperNeRF datasets, or None for DyNeRF datasets."""
    if not (dataset_dir / "dataset.json").exists():
        # DyNeRF dataset — cannot resolve from metadata; caller should use render files
        return None
    dataset_payload = _read_json(dataset_dir / "dataset.json")
    metadata_payload = _read_json(dataset_dir / "metadata.json")
    all_ids = list(dataset_payload["ids"])
    if dataset_payload.get("val_ids"):
        val_ids = set(dataset_payload["val_ids"])
        test_ids = [image_id for image_id in all_ids if image_id in val_ids]
    else:
        i_train = [index for index in range(len(all_ids)) if index % 4 == 0]
        i_test = (np.asarray(i_train, dtype=np.int64) + 2)[:-1]
        test_ids = [all_ids[int(index)] for index in i_test]
    max_time = max(float(metadata_payload[image_id]["warp_id"]) for image_id in all_ids)
    time_values = np.asarray(
        [float(metadata_payload[image_id]["warp_id"]) / max(max_time, 1.0) for image_id in test_ids],
        dtype=np.float32,
    )
    return test_ids, time_values


def _dynerf_test_ids_from_renders(render_dir: Path) -> tuple[list[str], np.ndarray]:
    """Generate test IDs and time values for DyNeRF datasets based on rendered frames."""
    render_files = sorted(render_dir.glob("*.png"))
    n = len(render_files)
    if n == 0:
        raise FileNotFoundError(f"No rendered frames found in {render_dir}")
    test_ids = [f.stem for f in render_files]  # e.g. "00000", "00001", ...
    time_values = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
    return test_ids, time_values


def _load_tracks(run_dir: Path) -> dict[int, TrackSample]:
    payload = _read_json(run_dir / "entitybank" / "semantic_tracks.json")
    track_map: dict[int, TrackSample] = {}
    for track in payload.get("tracks", []):
        frame_payload = track.get("frames", [])
        if not frame_payload:
            continue
        time_values = np.asarray([float(frame["time_value"]) for frame in frame_payload], dtype=np.float32)
        centers = np.asarray([frame["center_world"] for frame in frame_payload], dtype=np.float32)
        extents_min = np.asarray([frame["extent_world_min"] for frame in frame_payload], dtype=np.float32)
        extents_max = np.asarray([frame["extent_world_max"] for frame in frame_payload], dtype=np.float32)
        visibility = np.asarray([float(frame["visibility"]) for frame in frame_payload], dtype=np.float32)
        support_score = np.asarray([float(frame["support_score"]) for frame in frame_payload], dtype=np.float32)
        track_map[int(track["entity_id"])] = TrackSample(
            entity_id=int(track["entity_id"]),
            time_values=time_values,
            centers=centers,
            extents_min=extents_min,
            extents_max=extents_max,
            visibility=visibility,
            support_score=support_score,
        )
    return track_map


def _load_entity_static_texts(run_dir: Path) -> dict[int, str]:
    payload = _read_json(run_dir / "entitybank" / "entities.json")
    return {
        int(entity["id"]): str(entity.get("static_text", "")).strip()
        for entity in payload.get("entities", [])
    }


def _normalize_phrase_key(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def _load_entity_phrase_hints(run_dir: Path) -> dict[int, list[str]]:
    payload = _read_json(run_dir / "entitybank" / "entities.json")
    hints: dict[int, list[str]] = {}
    for entity in payload.get("entities", []):
        entity_id = int(entity["id"])
        values = [
            entity.get("static_text", ""),
            entity.get("proposal_alias", ""),
            entity.get("proposal_variant", ""),
            entity.get("global_desc", ""),
        ]
        stage1_object_id = entity.get("stage1_object_id")
        if stage1_object_id is not None:
            values.insert(0, f"stage1 object id {stage1_object_id}")
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = _normalize_phrase_key(value)
            if key and key not in seen:
                seen.add(key)
                normalized.append(key)
        hints[entity_id] = normalized
    return hints


def _resolve_query_track_for_hints(query_tracks: dict[str, QueryTrack], hints: list[str]) -> QueryTrack | None:
    if not query_tracks:
        return None
    normalized_hints = [_normalize_phrase_key(hint) for hint in hints if _normalize_phrase_key(hint)]
    for hint in normalized_hints:
        if hint in query_tracks:
            return query_tracks[hint]
    containment_matches: list[tuple[int, QueryTrack]] = []
    for hint in normalized_hints:
        for phrase, track in query_tracks.items():
            phrase_key = _normalize_phrase_key(phrase)
            if phrase_key and (hint in phrase_key or phrase_key in hint):
                containment_matches.append((abs(len(phrase_key) - len(hint)), track))
    if containment_matches:
        containment_matches.sort(key=lambda item: item[0])
        return containment_matches[0][1]
    return None


def _resolve_query_tracks(selection_path: Path) -> dict[str, QueryTrack]:
    try:
        query_root = selection_path.parents[2]
    except IndexError:
        return {}
    tracks_path = query_root / "grounded_sam2" / "grounded_sam2_query_tracks.json"
    if not tracks_path.exists():
        return {}
    payload = _read_json(tracks_path)
    tracks: dict[str, QueryTrack] = {}
    for track in payload.get("tracks", []):
        phrase = _normalize_phrase_key(track.get("phrase", ""))
        if not phrase:
            continue
        query_track = QueryTrack(
            phrase=phrase,
            frames=[frame for frame in track.get("frames", []) if bool(frame.get("active")) and frame.get("mask_path")],
        )
        # A phrase can legitimately map to several same-category instances.
        # Preserve their stable Stage-1 ids for entity-specific boundary gates
        # while retaining the legacy phrase key for ordinary single-instance runs.
        try:
            object_id = int(track.get("object_id"))
        except (TypeError, ValueError):
            object_id = None
        if object_id is not None:
            tracks[f"stage1 object id {object_id}"] = query_track
        tracks.setdefault(phrase, query_track)
    return tracks


def _query_track_match_for_time(
    track: QueryTrack | None,
    time_value: float,
    tolerance: float = 0.012,
    strict: bool = False,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    meta: dict[str, Any] = {
        "stage1_track_available": bool(track is not None and bool(getattr(track, "frames", []))),
        "stage1_mask_matched": False,
        "stage1_match_frame_index": None,
        "stage1_match_time_value": None,
        "stage1_match_time_delta": None,
        "stage1_match_tolerance": float(tolerance),
        "stage1_match_mode": "missing_track",
    }
    if track is None or not track.frames:
        return None, meta

    legacy_scale = max(1.0, float(os.environ.get("GS_QUERY_TRACK_FALLBACK_SCALE", "1.25")))
    strict_scale = max(1.0e-3, float(os.environ.get("GS_QUERY_TRACK_STRICT_FALLBACK_SCALE", "1.0")))
    fallback_scale = strict_scale if strict else legacy_scale
    frame_times = np.asarray([float(frame.get("time_value", 0.0)) for frame in track.frames], dtype=np.float32)
    adaptive_tolerance = float(tolerance)
    if frame_times.size >= 2:
        diffs = np.diff(np.sort(frame_times))
        if diffs.size:
            adaptive_tolerance = max(adaptive_tolerance, float(np.median(diffs)) * 0.65)
    allowed_delta = adaptive_tolerance * fallback_scale
    best = min(track.frames, key=lambda frame: abs(float(frame.get("time_value", 0.0)) - float(time_value)))
    best_time = float(best.get("time_value", 0.0))
    time_delta = abs(best_time - float(time_value))
    meta.update(
        {
            "stage1_match_frame_index": None if best.get("frame_index") is None else int(best.get("frame_index")),
            "stage1_match_time_value": best_time,
            "stage1_match_time_delta": float(time_delta),
            "stage1_match_tolerance": float(allowed_delta),
            "stage1_match_mode": "strict_nearest" if strict else "legacy_nearest",
        }
    )
    allow_stale_nearest = _env_flag("GS_QUERY_ALLOW_STALE_STAGE1_BOUNDARY", False)
    if time_delta > allowed_delta and not allow_stale_nearest:
        meta["stage1_match_mode"] = "strict_rejected_stale" if strict else "legacy_rejected_stale"
        return None, meta
    if time_delta > allowed_delta and allow_stale_nearest:
        meta["stage1_match_mode"] = "strict_stale_nearest" if strict else "legacy_stale_nearest"
    mask_path = best.get("mask_path")
    if not mask_path:
        meta["stage1_match_mode"] = "missing_mask_path"
        return None, meta
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    meta["stage1_mask_matched"] = True
    if time_delta <= max(1.0e-6, float(tolerance)):
        meta["stage1_match_mode"] = "strict_exact_or_near" if strict else "legacy_exact_or_near"
    return mask, meta


def _stage1_boundary_coverage_summary(
    frame_records: list[dict[str, Any]],
    *,
    required: bool,
) -> dict[str, Any]:
    """Summarize the per-active-role Stage-1 boundary matches for auditing."""
    active_role_count = 0
    matched_role_count = 0
    stale_matches: list[dict[str, Any]] = []
    missing_matches: list[dict[str, Any]] = []
    for frame_record in frame_records:
        if not bool(frame_record.get("query_active", False)):
            continue
        for role_record in frame_record.get("roles", []):
            if not bool(role_record.get("active", False)):
                continue
            active_role_count += 1
            match_mode = str(role_record.get("stage1_match_mode", "missing_track"))
            evidence = {
                "frame_index": int(frame_record.get("frame_index", -1)),
                "entity_id": int(role_record.get("entity_id", -1)),
                "role": str(role_record.get("role", "other")),
                "match_mode": match_mode,
                "time_delta": role_record.get("stage1_match_time_delta"),
                "tolerance": role_record.get("stage1_match_tolerance"),
            }
            if bool(role_record.get("stage1_mask_matched", False)):
                matched_role_count += 1
            else:
                missing_matches.append(evidence)
            if "stale" in match_mode:
                stale_matches.append(evidence)

    return {
        "required": bool(required),
        "active_role_frame_count": int(active_role_count),
        "matched_role_frame_count": int(matched_role_count),
        "coverage": float(matched_role_count / active_role_count) if active_role_count else 1.0,
        "stale_match_count": int(len(stale_matches)),
        "missing_match_count": int(len(missing_matches)),
        "stale_examples": stale_matches[:20],
        "missing_examples": missing_matches[:20],
    }


def _query_track_mask_for_time(track: QueryTrack | None, time_value: float, tolerance: float = 0.012) -> np.ndarray | None:
    mask, _meta = _query_track_match_for_time(track, time_value, tolerance=tolerance, strict=False)
    return mask


def _resize_mask_to_shape(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    binary = np.asarray(mask, dtype=bool)
    target_h, target_w = int(shape[0]), int(shape[1])
    if binary.shape == (target_h, target_w):
        return binary
    image = Image.fromarray(binary.astype(np.uint8) * 255, mode="L")
    resized = image.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 0


def _shift_binary_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    shifted = np.zeros_like(binary, dtype=bool)
    if height <= 0 or width <= 0:
        return shifted
    src_x0 = max(0, -int(dx))
    src_x1 = min(width, width - int(dx)) if int(dx) >= 0 else width
    dst_x0 = max(0, int(dx))
    dst_x1 = min(width, width + int(dx)) if int(dx) <= 0 else width
    src_y0 = max(0, -int(dy))
    src_y1 = min(height, height - int(dy)) if int(dy) >= 0 else height
    dst_y0 = max(0, int(dy))
    dst_y1 = min(height, height + int(dy)) if int(dy) <= 0 else height
    if src_x1 <= src_x0 or src_y1 <= src_y0 or dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return shifted
    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = binary[src_y0:src_y1, src_x0:src_x1]
    return shifted


def _bbox_center(bbox: list[int] | list[float] | tuple[float, ...]) -> tuple[float, float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def _bbox_area(bbox: list[int] | list[float] | tuple[float, ...] | None) -> int:
    if bbox is None:
        return 0
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox]
    width = max(0, x1 - x0 + 1)
    height = max(0, y1 - y0 + 1)
    return int(width * height)


def _box_mask_from_bbox(shape: tuple[int, int], bbox: list[int] | list[float] | tuple[float, ...] | None) -> np.ndarray | None:
    if bbox is None:
        return None
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        return None
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox]
    x0 = int(np.clip(x0, 0, max(width - 1, 0)))
    x1 = int(np.clip(x1, 0, max(width - 1, 0)))
    y0 = int(np.clip(y0, 0, max(height - 1, 0)))
    y1 = int(np.clip(y1, 0, max(height - 1, 0)))
    if x1 < x0 or y1 < y0:
        return None
    mask = np.zeros((height, width), dtype=bool)
    mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


def _expand_cloud_mask_with_projected_box(
    cloud_mask: np.ndarray | None,
    projected_bbox: list[int] | list[float] | tuple[float, ...] | None,
) -> np.ndarray | None:
    if cloud_mask is None:
        return None
    if not _env_flag("GS_QUERY_CLOUD_BOX_EXPANSION", True):
        return np.asarray(cloud_mask, dtype=bool)
    cloud_binary = np.asarray(cloud_mask, dtype=bool)
    box_mask = _box_mask_from_bbox(cloud_binary.shape, projected_bbox)
    if box_mask is None:
        return cloud_binary
    box_area = int(box_mask.sum())
    min_box_area = _env_int("GS_QUERY_CLOUD_BOX_MIN_AREA", 12000, minimum=1)
    if box_area < min_box_area:
        return cloud_binary
    fill_ratio = float(cloud_binary.sum()) / max(float(box_area), 1.0)
    max_fill_ratio = _env_float("GS_QUERY_CLOUD_BOX_MAX_FILL_RATIO", 0.72, minimum=0.0)
    if fill_ratio >= max_fill_ratio:
        return cloud_binary
    support_kernel = _odd_kernel(_env_int("GS_QUERY_CLOUD_BOX_SUPPORT_KERNEL", 21, minimum=1))
    support = box_mask & _dilate_binary_mask(cloud_binary, kernel_size=support_kernel)
    return cloud_binary | support


def _align_query_track_mask(
    query_track_mask: np.ndarray,
    reference_bbox: list[int] | list[float] | tuple[float, ...] | None,
) -> tuple[np.ndarray, list[int] | None]:
    query_binary = np.asarray(query_track_mask, dtype=bool)
    query_bbox = _entity_mask_bbox(query_binary)
    if query_bbox is None or reference_bbox is None:
        return query_binary, query_bbox
    query_center = _bbox_center(query_bbox)
    reference_center = _bbox_center(reference_bbox)
    dx = int(round(reference_center[0] - query_center[0]))
    dy = int(round(reference_center[1] - query_center[1]))
    max_dx = max(6, int(round(query_binary.shape[1] * 0.12)))
    max_dy = max(6, int(round(query_binary.shape[0] * 0.12)))
    if abs(dx) > max_dx or abs(dy) > max_dy:
        return query_binary, query_bbox
    shifted = _shift_binary_mask(query_binary, dx=dx, dy=dy)
    shifted_bbox = _entity_mask_bbox(shifted)
    if shifted_bbox is None:
        return query_binary, query_bbox
    return shifted, shifted_bbox


def _dilate_binary_mask(mask: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {binary.shape}")
    if kernel_size <= 1:
        return binary > 0
    if kernel_size % 2 == 0:
        kernel_size += 1
    torch_result = _torch_binary_rank_filter(binary, kernel_size=kernel_size, mode="max")
    if torch_result is not None:
        return torch_result
    image = Image.fromarray(binary * 255, mode="L")
    dilated = image.filter(ImageFilter.MaxFilter(kernel_size))
    return np.asarray(dilated, dtype=np.uint8) > 0


def _erode_binary_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {binary.shape}")
    if kernel_size <= 1:
        return binary > 0
    if kernel_size % 2 == 0:
        kernel_size += 1
    torch_result = _torch_binary_rank_filter(binary, kernel_size=kernel_size, mode="min")
    if torch_result is not None:
        return torch_result
    image = Image.fromarray(binary * 255, mode="L")
    eroded = image.filter(ImageFilter.MinFilter(kernel_size))
    return np.asarray(eroded, dtype=np.uint8) > 0


def _torch_binary_rank_filter(mask: np.ndarray, kernel_size: int, mode: str) -> np.ndarray | None:
    """Optional torch implementation of PIL MaxFilter/MinFilter for binary masks."""
    if torch is None or not _env_flag("QUERY_RENDER_TORCH_MORPHOLOGY", False):
        return None
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {binary.shape}")
    if kernel_size <= 1:
        return binary > 0
    if kernel_size % 2 == 0:
        kernel_size += 1
    if mode not in {"max", "min"}:
        raise ValueError(f"Unsupported binary rank filter mode: {mode}")
    device_name = os.environ.get("QUERY_RENDER_TORCH_MORPHOLOGY_DEVICE", "").strip()
    if not device_name:
        device_name = "cuda" if bool(getattr(torch, "cuda", None)) and torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and (not bool(getattr(torch, "cuda", None)) or not torch.cuda.is_available()):
        return None
    try:
        tensor = torch.from_numpy((binary > 0).astype(np.float32))[None, None].to(device_name)
        if mode == "max":
            filtered = torch.nn.functional.max_pool2d(tensor, kernel_size, stride=1, padding=kernel_size // 2)
        else:
            filtered = -torch.nn.functional.max_pool2d(-tensor, kernel_size, stride=1, padding=kernel_size // 2)
        return (filtered[0, 0].detach().cpu().numpy() > 0.5)
    except Exception as exc:
        if _env_flag("QUERY_RENDER_TORCH_MORPHOLOGY_WARN", False):
            print(f"[warn] torch morphology failed; falling back to PIL: {exc}", file=sys.stderr)
        return None


def _close_binary_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return np.asarray(mask, dtype=bool)
    return _erode_binary_mask(_dilate_binary_mask(mask, kernel_size=kernel_size), kernel_size=kernel_size)


def _open_binary_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return np.asarray(mask, dtype=bool)
    return _dilate_binary_mask(_erode_binary_mask(mask, kernel_size=kernel_size), kernel_size=kernel_size)


def _mask_iou(mask_a: np.ndarray | None, mask_b: np.ndarray | None) -> float:
    if mask_a is None and mask_b is None:
        return 1.0
    if mask_a is None or mask_b is None:
        return 0.0
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    union = int((a | b).sum())
    if union == 0:
        return 1.0
    inter = int((a & b).sum())
    return float(inter) / float(max(union, 1))


def _mask_precision_recall(pred_mask: np.ndarray | None, gt_mask: np.ndarray | None) -> tuple[float, float]:
    if pred_mask is None and gt_mask is None:
        return 1.0, 1.0
    if pred_mask is None or gt_mask is None:
        return 0.0, 0.0
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    inter = int((pred & gt).sum())
    pred_count = int(pred.sum())
    gt_count = int(gt.sum())
    if pred_count == 0 and gt_count == 0:
        return 1.0, 1.0
    precision = float(inter) / float(max(pred_count, 1))
    recall = float(inter) / float(max(gt_count, 1))
    return precision, recall


def _component_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return np.zeros_like(binary, dtype=np.int32), 0
    if _ndimage is None:
        labels = binary.astype(np.int32)
        return labels, int(binary.any())
    structure = np.ones((3, 3), dtype=np.int8)
    labels, count = _ndimage.label(binary, structure=structure)
    return labels.astype(np.int32), int(count)


def _component_count(mask: np.ndarray) -> int:
    _, count = _component_labels(mask)
    return int(count)


def _largest_component(mask: np.ndarray, min_area: int = 1) -> np.ndarray:
    labels, count = _component_labels(mask)
    if count <= 1:
        return np.asarray(mask, dtype=bool)
    best_label = 0
    best_area = 0
    for label_id in range(1, count + 1):
        area = int((labels == label_id).sum())
        if area > best_area:
            best_area = area
            best_label = label_id
    if best_label <= 0 or best_area < int(min_area):
        return np.zeros_like(labels, dtype=bool)
    return labels == best_label


def _drop_small_components(mask: np.ndarray, min_area: int = 64) -> np.ndarray:
    labels, count = _component_labels(mask)
    if count <= 1:
        binary = np.asarray(mask, dtype=bool)
        if int(binary.sum()) < int(min_area):
            return np.zeros_like(binary, dtype=bool)
        return binary
    output = np.zeros_like(labels, dtype=bool)
    for label_id in range(1, count + 1):
        component = labels == label_id
        if int(component.sum()) >= int(min_area):
            output |= component
    return output


def _mask_match_score(
    candidate: np.ndarray,
    aligned_query: np.ndarray | None,
    cloud_binary: np.ndarray | None,
    previous_mask: np.ndarray | None,
    intent_mode: str,
    options: dict[str, Any],
) -> float:
    candidate = np.asarray(candidate, dtype=bool)
    area = int(candidate.sum())
    if area <= 0:
        return -1.0e6

    score = 0.0
    if aligned_query is not None:
        query_binary = np.asarray(aligned_query, dtype=bool)
        query_area = int(query_binary.sum())
        if query_area > 0:
            query_recall = float((candidate & query_binary).sum()) / float(query_area)
            score += float(options["w_query"]) * query_recall
    if cloud_binary is not None:
        cloud_mask = np.asarray(cloud_binary, dtype=bool)
        cloud_area = int(cloud_mask.sum())
        if cloud_area > 0:
            cloud_recall = float((candidate & cloud_mask).sum()) / float(cloud_area)
            score += float(options["w_cloud"]) * cloud_recall
            score += 0.20 * _mask_iou(candidate, cloud_mask)
            ratio = float(area) / float(max(cloud_area, 1))
            score -= 0.04 * abs(np.log(max(ratio, 1.0e-3)))
    if previous_mask is not None:
        score += float(options["w_prev_iou"]) * _mask_iou(candidate, previous_mask)

    comp_count = _component_count(candidate)
    if intent_mode == "single_component":
        if comp_count > 1:
            score -= min(0.30, 0.06 * float(comp_count - 1))
    elif intent_mode == "multi_component":
        if comp_count <= 1:
            score -= 0.08
        score += min(0.10, 0.02 * float(max(comp_count - 1, 0)))
    return float(score)


def _fuse_query_and_cloud_masks(
    query_track_mask: np.ndarray | None,
    cloud_mask: np.ndarray | None,
    projected_bbox: list[int] | list[float] | tuple[float, ...] | None = None,
    track_state_mode: str | None = None,
    selected_item_count: int = 1,
    previous_mask: np.ndarray | None = None,
    query_intent_mode: str = "generic",
    fusion_options: dict[str, Any] | None = None,
) -> tuple[np.ndarray | None, list[int] | None, list[int] | None, str]:
    options = fusion_options or _fusion_options_for_profile("default")
    allow_direct_query_track = bool(options.get("allow_direct_query_track", False))
    if query_track_mask is None and cloud_mask is None:
        return None, None, None, "none"
    cloud_binary = _expand_cloud_mask_with_projected_box(cloud_mask, projected_bbox=projected_bbox)
    cloud_bbox = _entity_mask_bbox(cloud_binary) if cloud_binary is not None else None
    if query_track_mask is None:
        if bool(options.get("strict_gaussian_projection", False)):
            return cloud_binary, cloud_bbox, None, "gaussian_projection_no_stage1_boundary"
        return cloud_binary, cloud_bbox, None, "cloud_only"
    query_binary = np.asarray(query_track_mask, dtype=bool)
    if cloud_binary is not None:
        resized = _resize_mask_to_shape(query_binary, cloud_binary.shape)
        if resized is not None:
            query_binary = resized
    query_bbox = _entity_mask_bbox(query_binary)
    if query_bbox is None:
        if bool(options.get("strict_gaussian_projection", False)):
            return cloud_binary, cloud_bbox, None, "gaussian_projection_no_stage1_boundary"
        return cloud_binary, cloud_bbox, None, "cloud_only"
    if bool(options.get("strict_gaussian_projection", False)) and not bool(options.get("clip_to_query_track", False)):
        return cloud_binary, cloud_bbox, query_bbox, "gaussian_projection"
    aligned_query = query_binary
    aligned_query_bbox = query_bbox
    if cloud_bbox is not None and bool(options.get("align_query_track_to_cloud", True)):
        aligned_query, aligned_query_bbox = _align_query_track_mask(query_binary, cloud_bbox)
    if cloud_binary is None:
        if allow_direct_query_track:
            clean_query = _drop_small_components(aligned_query, min_area=int(options["min_component_area"]))
            if clean_query.any():
                return clean_query, _entity_mask_bbox(clean_query), aligned_query_bbox, "query_track"
            return aligned_query, aligned_query_bbox, aligned_query_bbox, "query_track"
        return None, None, aligned_query_bbox, "cloud_unavailable"

    normalized_state = _normalize_track_state_mode(track_state_mode)
    support_like = normalized_state in {None, "support"}

    support_kernel = int(options["support_kernel"] if support_like else options["state_kernel"])
    original_cloud_binary = cloud_binary
    cloud_was_clipped = False
    cloud_was_recovered = False
    recovery_candidate: np.ndarray | None = None
    if bool(options.get("clip_to_query_track", False)):
        clip_region = _dilate_binary_mask(aligned_query, kernel_size=int(options.get("clip_kernel", support_kernel)))
        clipped = cloud_binary & clip_region
        if clipped.any():
            cloud_binary = clipped
            cloud_was_clipped = True
            recovery_kernel = int(options.get("recovery_kernel", 1))
            query_area = float(np.asarray(aligned_query, dtype=bool).sum())
            clipped_area = float(np.asarray(clipped, dtype=bool).sum())
            clipped_query_recall = (
                float((np.asarray(clipped, dtype=bool) & np.asarray(aligned_query, dtype=bool)).sum()) / max(query_area, 1.0)
                if query_area > 0.0
                else 0.0
            )
            needs_recovery = (
                recovery_kernel > int(options.get("clip_kernel", support_kernel))
                and query_area > 0.0
                and (
                    clipped_area / max(query_area, 1.0) < float(options.get("recovery_min_clip_area_ratio", 0.0))
                    or clipped_query_recall < float(options.get("recovery_min_query_recall", 0.0))
                )
            )
            if needs_recovery:
                recovered = original_cloud_binary & _dilate_binary_mask(aligned_query, kernel_size=recovery_kernel)
                if recovered.any() and int(recovered.sum()) > int(clipped.sum()):
                    recovery_candidate = recovered
                    cloud_was_recovered = True
        else:
            recovery_kernel = int(options.get("recovery_kernel", 1))
            recovered = original_cloud_binary & _dilate_binary_mask(aligned_query, kernel_size=recovery_kernel)
            if recovered.any():
                cloud_binary = recovered
                cloud_was_recovered = True
                recovery_candidate = recovered
            else:
                cloud_binary = clipped
        cloud_bbox = _entity_mask_bbox(cloud_binary)
    support = cloud_binary & _dilate_binary_mask(aligned_query, kernel_size=support_kernel)
    expanded_support = cloud_binary & _dilate_binary_mask(aligned_query, kernel_size=int(options["expand_kernel"]))
    query_supported = None
    if bool(options.get("allow_query_supported_by_cloud", False)):
        query_supported = aligned_query & _dilate_binary_mask(
            cloud_binary,
            kernel_size=int(options.get("query_support_kernel", support_kernel)),
        )

    candidates: list[tuple[str, np.ndarray]] = []
    if support.any():
        candidates.append(("cloud_support", support))
    if expanded_support.any():
        candidates.append(("cloud_expanded_support", expanded_support))
    if query_supported is not None and query_supported.any():
        candidates.append(("query_cloud_supported", query_supported))
    if cloud_was_recovered:
        recovered_mask = recovery_candidate if recovery_candidate is not None else cloud_binary
        if recovered_mask is not None and recovered_mask.any():
            candidates.append(("cloud_boundary_recovery", recovered_mask))
    if bool(options.get("prefer_clipped_cloud", False)) and cloud_was_clipped and cloud_binary.any():
        candidates.append(("cloud_clipped", cloud_binary))
    if bool(options.get("allow_cloud_only_with_query", True)):
        candidates.append(("cloud_only", cloud_binary))
    if allow_direct_query_track:
        candidates.insert(0, ("query_track", aligned_query))
        if support.any():
            candidates.append(("query_plus_support", aligned_query | support))
        if expanded_support.any():
            candidates.append(("query_plus_expanded_support", aligned_query | expanded_support))

    seen_signatures: set[tuple[int, int]] = set()
    prepared_candidates: list[tuple[str, np.ndarray]] = []
    for source_name, raw_mask in candidates:
        mask = np.asarray(raw_mask, dtype=bool)
        if not mask.any():
            continue
        mask = _close_binary_mask(mask, kernel_size=int(options["close_kernel"]))
        mask = _open_binary_mask(mask, kernel_size=int(options["open_kernel"]))
        if bool(options.get("strict_gaussian_projection", False)):
            # Morphological cleanup may fill a small hole or bridge a narrow
            # gap. The formal contract keeps only pixels still supported by
            # the selected Gaussian projection after that cleanup.
            mask &= np.asarray(cloud_binary, dtype=bool)
        mask = _drop_small_components(mask, min_area=int(options["min_component_area"]))
        if not mask.any():
            continue
        if query_intent_mode == "single_component":
            mask = _largest_component(mask, min_area=int(options["min_component_area"]))
            if not mask.any():
                continue
        signature = (int(mask.sum()), _component_count(mask))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        prepared_candidates.append((source_name, mask))

    if not prepared_candidates:
        if allow_direct_query_track:
            return aligned_query, aligned_query_bbox, aligned_query_bbox, "query_track_fallback"
        return None, None, aligned_query_bbox, "gaussian_projection_empty"

    best_source = prepared_candidates[0][0]
    best_mask = prepared_candidates[0][1]
    best_score = -1.0e9
    for source_name, mask in prepared_candidates:
        score = _mask_match_score(
            candidate=mask,
            aligned_query=aligned_query,
            cloud_binary=cloud_binary,
            previous_mask=previous_mask,
            intent_mode=query_intent_mode,
            options=options,
        )
        if score > best_score:
            best_score = score
            best_source = source_name
            best_mask = mask

    best_bbox = _entity_mask_bbox(best_mask)
    if best_bbox is None:
        if allow_direct_query_track:
            return aligned_query, aligned_query_bbox, aligned_query_bbox, "query_track_fallback"
        return None, None, aligned_query_bbox, "gaussian_projection_empty"
    strict_source_map = {
        "cloud_support": "gaussian_projection_clipped",
        "cloud_expanded_support": "gaussian_projection_expanded_stage1_clip",
        "query_cloud_supported": "gaussian_projection_stage1_supported",
        "cloud_boundary_recovery": "gaussian_projection_boundary_recovery",
        "cloud_clipped": "gaussian_projection_clipped",
        "cloud_only": "gaussian_projection_unclipped",
    }
    if bool(options.get("strict_gaussian_projection", False)):
        best_source = strict_source_map.get(str(best_source), f"gaussian_projection_{best_source}")
    return best_mask, best_bbox, aligned_query_bbox, best_source


def _interp_vector(query_times: np.ndarray, sample_times: np.ndarray, values: np.ndarray) -> np.ndarray:
    query_times = np.asarray(query_times, dtype=np.float32)
    sample_times = np.asarray(sample_times, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    result = np.zeros((query_times.shape[0], values.shape[1]), dtype=np.float32)
    for dim in range(values.shape[1]):
        result[:, dim] = np.interp(query_times, sample_times, values[:, dim])
    return result


def _interp_scalar(query_times: np.ndarray, sample_times: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.interp(
        np.asarray(query_times, dtype=np.float32),
        np.asarray(sample_times, dtype=np.float32),
        np.asarray(values, dtype=np.float32),
    ).astype(np.float32)


def _selected_item_gaussian_ids(
    entity_map: dict[int, np.ndarray],
    item: dict[str, Any],
) -> np.ndarray:
    override = np.asarray(item.get("gaussian_ids", []), dtype=np.int64).reshape(-1)
    if override.size > 0:
        return override
    return np.asarray(entity_map.get(int(item["id"]), []), dtype=np.int64).reshape(-1)


def _opacity_logits_from_probabilities(values: np.ndarray) -> np.ndarray | None:
    probabilities = np.asarray(values, dtype=np.float32).reshape(-1)
    if probabilities.size == 0 or not np.all(np.isfinite(probabilities)):
        return None
    probabilities = np.clip(probabilities, 1.0e-6, 1.0 - 1.0e-6)
    return (np.log(probabilities) - np.log1p(-probabilities)).astype(np.float32)


def _load_entity_opacity_logits(
    run_dir: Path,
    trajectory_payload: Any,
    gaussian_count: int,
) -> tuple[np.ndarray | None, str]:
    """Load the same Gaussian opacity used by lifting, without a mask fallback."""
    if "opacity" in trajectory_payload.files:
        raw = np.asarray(trajectory_payload["opacity"], dtype=np.float32).reshape(-1)
        if raw.size == int(gaussian_count) and np.all(np.isfinite(raw)):
            return raw, "trajectory_bank"

    try:
        from .worldtube_consistency import load_opacity_sigmoid

        raw = _opacity_logits_from_probabilities(load_opacity_sigmoid(run_dir))
        if raw is not None and raw.size == int(gaussian_count):
            return raw, "point_cloud"
    except Exception:
        pass
    return None, "unavailable"


def _load_entity_clouds(run_dir: Path, selected_items: list[dict[str, Any]]) -> dict[int, EntityCloud]:
    entitybank_dir = run_dir / "entitybank"
    entities_payload = _read_json(entitybank_dir / "entities.json")
    entity_records = {
        int(entity["id"]): entity
        for entity in entities_payload.get("entities", [])
        if entity.get("id") is not None
    }
    entity_map = {
        int(entity["id"]): np.asarray(entity.get("gaussian_ids", []), dtype=np.int64)
        for entity in entities_payload.get("entities", [])
    }
    trajectory_payload = np.load(entitybank_dir / "trajectory_samples.npz")
    sample_times = np.asarray(trajectory_payload["time_values"], dtype=np.float32)
    trajectories = np.asarray(trajectory_payload["trajectories"], dtype=np.float32)
    gate = np.asarray(trajectory_payload["gate"], dtype=np.float32).reshape(trajectories.shape[0], trajectories.shape[1])
    spatial_extent = np.asarray(trajectory_payload["spatial_extent"], dtype=np.float32).reshape(-1)
    spatial_scale = None
    if "spatial_scale" in trajectory_payload.files:
        spatial_scale = np.asarray(trajectory_payload["spatial_scale"], dtype=np.float32)
        if spatial_scale.ndim != 2 or spatial_scale.shape[0] != trajectories.shape[0] or spatial_scale.shape[1] < 3:
            spatial_scale = None
    opacity_logit, opacity_source = _load_entity_opacity_logits(
        run_dir,
        trajectory_payload,
        gaussian_count=int(trajectories.shape[0]),
    )

    clouds: dict[int, EntityCloud] = {}
    for item_index, item in enumerate(selected_items):
        entity_record = entity_records.get(int(item.get("id", -1)), {})
        diagnostics = entity_record.get("rendered_diagnostics", {})
        relative_threshold = diagnostics.get("alpha_relative_threshold")
        absolute_threshold = diagnostics.get("alpha_absolute_threshold")
        sigma_scale = diagnostics.get("alpha_sigma_scale")
        max_splat_radius = diagnostics.get("alpha_max_splat_radius")
        try:
            relative_threshold = float(relative_threshold)
        except (TypeError, ValueError):
            relative_threshold = None
        try:
            absolute_threshold = float(absolute_threshold)
        except (TypeError, ValueError):
            absolute_threshold = None
        try:
            sigma_scale = float(sigma_scale)
        except (TypeError, ValueError):
            sigma_scale = None
        try:
            max_splat_radius = int(float(max_splat_radius))
        except (TypeError, ValueError):
            max_splat_radius = None
        if relative_threshold is not None and not np.isfinite(relative_threshold):
            relative_threshold = None
        if absolute_threshold is not None and not np.isfinite(absolute_threshold):
            absolute_threshold = None
        if sigma_scale is not None and (not np.isfinite(sigma_scale) or sigma_scale <= 0.0):
            sigma_scale = None
        if max_splat_radius is not None and max_splat_radius < 1:
            max_splat_radius = None
        gaussian_ids = _selected_item_gaussian_ids(entity_map, item)
        if gaussian_ids.size == 0:
            continue
        gaussian_ids = gaussian_ids[(gaussian_ids >= 0) & (gaussian_ids < trajectories.shape[0])]
        gaussian_ids = np.unique(gaussian_ids)
        if gaussian_ids.size == 0:
            continue
        clouds[int(item_index)] = EntityCloud(
            entity_id=int(item.get("id", -1)),
            sample_times=sample_times,
            trajectories=trajectories[gaussian_ids],
            gate=gate[gaussian_ids],
            spatial_extent=spatial_extent[gaussian_ids],
            spatial_scale=None if spatial_scale is None else spatial_scale[gaussian_ids, :3],
            opacity_logit=None if opacity_logit is None else opacity_logit[gaussian_ids],
            opacity_source=opacity_source,
            alpha_relative_threshold=relative_threshold,
            alpha_absolute_threshold=absolute_threshold,
            alpha_sigma_scale=sigma_scale,
            alpha_max_splat_radius=max_splat_radius,
        )
    return clouds


def _frame_mask(frame_count: int, segments: list[list[int]]) -> np.ndarray:
    mask = np.zeros((frame_count,), dtype=bool)
    for segment in segments:
        if len(segment) != 2:
            continue
        start = max(int(segment[0]), 0)
        end = min(int(segment[1]), frame_count - 1)
        if end < start:
            continue
        mask[start : end + 1] = True
    return mask


def _frame_mask_at_render_times(
    segments: list[list[int]],
    reference_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Evaluate selection segments on another camera/time sampling grid.

    Entity selections are defined on the reconstruction test sequence.  A
    benchmark can request masks from a different set of cameras at the same
    temporal positions, so re-indexing a segment by the output-list position
    would change the query's temporal meaning.  Hold the original selection at
    the nearest reconstruction timestamp instead.
    """
    reference = np.asarray(reference_times, dtype=np.float32).reshape(-1)
    target = np.asarray(target_times, dtype=np.float32).reshape(-1)
    if target.size == 0:
        return np.zeros((0,), dtype=bool)
    if reference.size == 0:
        raise ValueError("Cannot map query segments without reference render timestamps.")

    reference_mask = _frame_mask(int(reference.size), segments)
    positions = np.searchsorted(reference, target, side="left")
    positions = np.clip(positions, 0, int(reference.size) - 1)
    previous = np.clip(positions - 1, 0, int(reference.size) - 1)
    choose_previous = np.abs(target - reference[previous]) <= np.abs(reference[positions] - target)
    nearest = np.where(choose_previous, previous, positions)
    return reference_mask[nearest]


def _load_font(font_size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int]) -> None:
    font = _load_font(20)
    left, top = xy
    bbox = draw.textbbox((left, top), text, font=font)
    draw.rounded_rectangle(
        (bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2),
        radius=4,
        fill=(18, 18, 18),
    )
    draw.text((left, top), text, fill=fill, font=font)


def _entity_color(entity_id: int, role: str) -> tuple[int, int, int]:
    if role in {"patient", "tool", "agent"}:
        return ROLE_COLORS[role]
    hue = float((entity_id * 0.61803398875) % 1.0)
    sat = 0.70 if role == "entity" else 0.55
    val = 1.0
    rgb = colorsys.hsv_to_rgb(hue, sat, val)
    return tuple(int(round(channel * 255.0)) for channel in rgb)


def _entity_label(role: str, entity_id: int) -> str:
    if role == "entity":
        return f"entity #{entity_id}"
    return f"{role} #{entity_id}"


def _image_projection_scale(camera: Camera, image_size: tuple[int, int]) -> tuple[float, float]:
    camera_size = np.asarray(camera.image_size, dtype=np.float32).reshape(-1)
    if camera_size.size >= 2:
        camera_width = max(float(camera_size[0]), 1.0)
        camera_height = max(float(camera_size[1]), 1.0)
    else:
        camera_width = max(float(image_size[0]), 1.0)
        camera_height = max(float(image_size[1]), 1.0)
    image_width = max(float(image_size[0]), 1.0)
    image_height = max(float(image_size[1]), 1.0)
    return image_width / camera_width, image_height / camera_height


def _project_points_to_image(camera: Camera, points_world: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    projected = camera.project(points_world).astype(np.float32)
    scale_x, scale_y = _image_projection_scale(camera, image_size)
    projected[:, 0] *= float(scale_x)
    projected[:, 1] *= float(scale_y)
    return projected


def _convex_hull_pixels(points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float32)

    rounded = np.unique(np.round(points).astype(np.int32), axis=0)
    if rounded.shape[0] < 3:
        return rounded.astype(np.float32)

    points_list = sorted((int(x), int(y)) for x, y in rounded.tolist())

    def _cross(origin: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> int:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[int, int]] = []
    for point in points_list:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[int, int]] = []
    for point in reversed(points_list):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return np.asarray(hull, dtype=np.float32).reshape(-1, 2)
    return np.asarray(hull, dtype=np.float32)


def _pixel_radius(
    camera: Camera,
    center_world: np.ndarray,
    extent_min: np.ndarray,
    extent_max: np.ndarray,
    image_size: tuple[int, int] | None = None,
) -> int:
    center_local = camera.points_to_local_points(center_world[None, :])[0]
    depth = float(center_local[2])
    if depth <= 1.0e-4:
        return 10
    extent_radius = 0.5 * float(np.linalg.norm(extent_max - extent_min))
    if image_size is None:
        camera_size = np.asarray(camera.image_size, dtype=np.int32).reshape(-1)
        image_size = (int(camera_size[0]), int(camera_size[1]))
    scale_x, scale_y = _image_projection_scale(camera, image_size)
    scale = 0.5 * (float(scale_x) + float(scale_y))
    pixel_radius = float(camera.focal_length) * scale * extent_radius / max(depth, 1.0e-4)
    return int(np.clip(pixel_radius, 10.0, 72.0))


def _aabb_corners(extent_min: np.ndarray, extent_max: np.ndarray) -> np.ndarray:
    min_x, min_y, min_z = np.asarray(extent_min, dtype=np.float32).tolist()
    max_x, max_y, max_z = np.asarray(extent_max, dtype=np.float32).tolist()
    return np.asarray(
        [
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ],
        dtype=np.float32,
    )


def _project_box(
    camera: Camera,
    extent_min: np.ndarray,
    extent_max: np.ndarray,
    image_size: tuple[int, int],
) -> dict[str, Any] | None:
    corners_world = _aabb_corners(extent_min, extent_max)
    corners_local = camera.points_to_local_points(corners_world)
    valid = np.asarray(corners_local[:, 2] > 1.0e-4, dtype=bool)
    if not valid.any():
        return None

    corners_projected = _project_points_to_image(camera, corners_world[valid], image_size=image_size)
    width, height = image_size
    raw_left = float(corners_projected[:, 0].min())
    raw_top = float(corners_projected[:, 1].min())
    raw_right = float(corners_projected[:, 0].max())
    raw_bottom = float(corners_projected[:, 1].max())
    intersects = not (
        raw_right < 0.0
        or raw_left >= float(width)
        or raw_bottom < 0.0
        or raw_top >= float(height)
    )
    clamped_left = float(np.clip(raw_left, 0.0, max(width - 1, 0)))
    clamped_top = float(np.clip(raw_top, 0.0, max(height - 1, 0)))
    clamped_right = float(np.clip(raw_right, 0.0, max(width - 1, 0)))
    clamped_bottom = float(np.clip(raw_bottom, 0.0, max(height - 1, 0)))
    if clamped_right <= clamped_left:
        clamped_right = min(float(width - 1), clamped_left + 1.0)
    if clamped_bottom <= clamped_top:
        clamped_bottom = min(float(height - 1), clamped_top + 1.0)
    return {
        "raw_xyxy": [raw_left, raw_top, raw_right, raw_bottom],
        "clamped_xyxy": [clamped_left, clamped_top, clamped_right, clamped_bottom],
        "intersects_frame": bool(intersects),
        "projected_corners": corners_projected.astype(float).tolist(),
    }


def _entity_mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _project_entity_cloud_mask(
    camera: Camera,
    image_size: tuple[int, int],
    cloud: EntityCloud,
    time_value: float,
    eval_profile: str | None = None,
    stage1_boundary_available: bool = True,
) -> tuple[np.ndarray | None, list[int] | None]:
    if cloud.trajectories.size == 0:
        return None, None
    sample_index = int(np.abs(cloud.sample_times - float(time_value)).argmin())
    render_mode = " ".join(str(os.environ.get("GS_QUERY_CLOUD_RENDER_MODE", "point_hull")).strip().lower().replace("-", "_").split())
    alpha_modes = {"alpha_splat", "semantic_alpha", "gaussian_alpha"}
    strict_alpha = _env_flag("GS_QUERY_ALPHA_REQUIRE_SUCCESS", False)
    if render_mode in alpha_modes and (torch is None or prepare_semantic_frame_inputs is None or render_selection_mask is None):
        if strict_alpha:
            raise RuntimeError("Gaussian alpha projection is required but semantic rendering dependencies are unavailable")
    if render_mode in alpha_modes and torch is not None and prepare_semantic_frame_inputs is not None and render_selection_mask is not None:
        points_world_all = np.asarray(cloud.trajectories[:, sample_index, :], dtype=np.float32)
        gate_values_all = np.asarray(cloud.gate[:, sample_index], dtype=np.float32).reshape(-1)
        if cloud.spatial_scale is not None:
            spatial_scale_all = np.asarray(cloud.spatial_scale, dtype=np.float32)
        else:
            extent = np.asarray(cloud.spatial_extent, dtype=np.float32).reshape(-1)
            spatial_scale_all = np.repeat(np.maximum(extent[:, None], 1.0e-4), 3, axis=1).astype(np.float32)
        opacity_all = (
            np.zeros((points_world_all.shape[0],), dtype=np.float32)
            if cloud.opacity_logit is None
            else np.asarray(cloud.opacity_logit, dtype=np.float32).reshape(-1)
        )
        if cloud.opacity_logit is None and _env_flag("GS_QUERY_ALPHA_REQUIRE_OPACITY", False):
            raise RuntimeError(f"Gaussian alpha projection requires opacity for entity {cloud.entity_id}")
        if opacity_all.size != points_world_all.shape[0]:
            if strict_alpha:
                raise RuntimeError(
                    f"Gaussian alpha projection opacity count mismatch for entity {cloud.entity_id}: "
                    f"{opacity_all.size} vs {points_world_all.shape[0]}"
                )
            opacity_all = np.zeros((points_world_all.shape[0],), dtype=np.float32)
        scale_x, scale_y = _image_projection_scale(camera, image_size)
        image_scale = float(0.5 * (scale_x + scale_y))
        alpha_device = os.environ.get("GS_QUERY_ALPHA_DEVICE", "").strip()
        if not alpha_device:
            alpha_device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            prepared = prepare_semantic_frame_inputs(
                camera=camera,
                frame_index=0,
                image_id="query",
                time_value=float(time_value),
                points=points_world_all,
                spatial_scale=spatial_scale_all,
                opacity=opacity_all,
                visibility_gate=gate_values_all,
                image_scale=image_scale,
                max_gaussians=_env_int("GS_QUERY_ALPHA_MAX_GAUSSIANS", 20000, minimum=1),
                gate_threshold=_env_float("GS_QUERY_ALPHA_GATE_THRESHOLD", 0.08, minimum=0.0),
                device=alpha_device,
            )
            if int(prepared.gaussian_ids.numel()) > 0:
                sigma_scale = (
                    float(cloud.alpha_sigma_scale)
                    if cloud.alpha_sigma_scale is not None
                    else _env_float("GS_QUERY_ALPHA_SIGMA_SCALE", 1.0, minimum=0.25, maximum=8.0)
                )
                max_splat_radius = (
                    int(cloud.alpha_max_splat_radius)
                    if cloud.alpha_max_splat_radius is not None
                    else _env_int("GS_QUERY_ALPHA_MAX_SPLAT_RADIUS", 18, minimum=1)
                )
                sigma_scale = float(np.clip(sigma_scale, 0.25, 8.0))
                max_splat_radius = int(np.clip(max_splat_radius, 1, 64))
                if sigma_scale != 1.0:
                    prepared.sigma_px = prepared.sigma_px * sigma_scale
                selected_weights = torch.ones(
                    (int(prepared.gaussian_ids.numel()),),
                    dtype=torch.float32,
                    device=prepared.gaussian_ids.device,
                )
                mask, _alpha = render_selection_mask(
                    prepared,
                    selected_weights,
                    relative_threshold=(
                        float(cloud.alpha_relative_threshold)
                        if cloud.alpha_relative_threshold is not None
                        else _env_float("GS_QUERY_ALPHA_REL_THRESHOLD", 0.16, minimum=0.0)
                    ),
                    absolute_threshold=(
                        float(cloud.alpha_absolute_threshold)
                        if cloud.alpha_absolute_threshold is not None
                        else _env_float("GS_QUERY_ALPHA_ABS_THRESHOLD", 0.010, minimum=0.0)
                    ),
                    max_splat_radius=max_splat_radius,
                )
                if mask.shape != (int(image_size[1]), int(image_size[0])):
                    mask = _resize_mask_to_shape(mask, (int(image_size[1]), int(image_size[0])))
                if mask is not None and mask.any():
                    postfilter_kernel = _odd_kernel(_env_int("GS_QUERY_ALPHA_POSTFILTER_KERNEL", 1, minimum=1))
                    if postfilter_kernel > 1:
                        mask = _dilate_binary_mask(mask, kernel_size=postfilter_kernel)
                    bbox = _entity_mask_bbox(mask)
                    if bbox is not None:
                        return mask, bbox
        except Exception as exc:
            if strict_alpha:
                raise RuntimeError(f"Gaussian alpha projection failed for entity {cloud.entity_id}: {exc}") from exc
            print(f"[warn] alpha_splat projection failed for entity {cloud.entity_id}: {exc}", file=sys.stderr)
        if strict_alpha:
            return None, None

    points_world = np.asarray(cloud.trajectories[:, sample_index, :], dtype=np.float32)
    gate_values = np.asarray(cloud.gate[:, sample_index], dtype=np.float32).reshape(-1)
    active = gate_values >= max(0.08, float(gate_values.max()) * 0.15)
    if not active.any():
        return None, None
    points_world = points_world[active]
    point_extent = np.asarray(cloud.spatial_extent[active], dtype=np.float32).reshape(-1)
    points_local = camera.points_to_local_points(points_world)
    valid = points_local[:, 2] > 1.0e-4
    if not valid.any():
        return None, None
    points_world = points_world[valid]
    points_local = points_local[valid]
    point_extent = point_extent[valid]
    projected = _project_points_to_image(camera, points_world, image_size=image_size)
    width, height = image_size
    in_bounds = (
        (projected[:, 0] >= -32.0)
        & (projected[:, 0] < float(width + 32))
        & (projected[:, 1] >= -32.0)
        & (projected[:, 1] < float(height + 32))
    )
    if not in_bounds.any():
        return None, None
    projected = projected[in_bounds]
    points_local = points_local[in_bounds]
    point_extent = point_extent[in_bounds]

    mask_image = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask_image)
    scale_x, scale_y = _image_projection_scale(camera, image_size)
    focal = float(camera.focal_length) * 0.5 * (float(scale_x) + float(scale_y))
    resolved_profile = _resolve_eval_profile(eval_profile)
    boundary_profile = resolved_profile in {"boundary_refine_v1", "public_boundary_v1", "mask_boundary_refine"}
    coverage_profile = _is_coverage_render_profile(resolved_profile)
    strict_projection = (boundary_profile or coverage_profile) and _env_flag("GS_QUERY_STRICT_GAUSSIAN_PROJECTION", True)
    radius_scale = _env_float("GS_QUERY_CLOUD_POINT_RADIUS_SCALE", 0.42 if boundary_profile else (0.95 if coverage_profile else 0.75), minimum=0.05)
    if (boundary_profile or coverage_profile) and not strict_projection and not bool(stage1_boundary_available):
        radius_scale *= _env_float("GS_QUERY_CLOUD_NO_STAGE1_RADIUS_MULT", 0.35, minimum=0.05, maximum=2.0)
    radius_min = _env_int("GS_QUERY_CLOUD_POINT_RADIUS_MIN", 1 if (boundary_profile or coverage_profile) else 3, minimum=1)
    radius_max = _env_int("GS_QUERY_CLOUD_POINT_RADIUS_MAX", 9 if boundary_profile else (28 if coverage_profile else 18), minimum=1)
    postfilter_kernel = _odd_kernel(_env_int("GS_QUERY_CLOUD_POSTFILTER_KERNEL", 3 if boundary_profile else (7 if coverage_profile else 11), minimum=1))
    for pixel, local_point, extent in zip(projected, points_local, point_extent):
        depth = max(float(local_point[2]), 1.0e-4)
        radius = int(
            np.clip(
                float(radius_scale) * focal * max(float(extent), 1.0e-4) / depth,
                float(radius_min),
                float(max(radius_min, radius_max)),
            )
        )
        cx = int(round(float(pixel[0])))
        cy = int(round(float(pixel[1])))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)

    use_component_hull = render_mode in {"component_shape_hull"}
    use_global_convex_hull = _env_flag("GS_QUERY_CLOUD_FILL_CONVEX_HULL", False) and (strict_projection or bool(stage1_boundary_available))

    if use_component_hull:
        base_mask = np.asarray(mask_image, dtype=np.uint8) > 0
        component_hull_enabled = _env_flag("GS_QUERY_CLOUD_COMPONENT_HULL", True)
        comp_hull_min_pts = _env_int("GS_QUERY_CLOUD_COMPONENT_HULL_MIN_POINTS", 8, minimum=3)
        comp_hull_max_area_mult = _env_float("GS_QUERY_CLOUD_COMPONENT_HULL_MAX_AREA_MULTIPLIER", 3.0, minimum=1.0)
        comp_close_kernel = _odd_kernel(_env_int("GS_QUERY_CLOUD_COMPONENT_CLOSE_KERNEL", 5, minimum=1))
        comp_fill_holes = _env_flag("GS_QUERY_CLOUD_COMPONENT_FILL_HOLES", True)

        if component_hull_enabled and projected.shape[0] >= comp_hull_min_pts:
            # Group projected points into connected components in image space
            pixel_round = np.unique(np.round(projected).astype(np.int32), axis=0)
            # Build connected components via distance-based clustering
            if pixel_round.shape[0] >= comp_hull_min_pts:
                from scipy.spatial import cKDTree as _cKDTree_local
                tree = _cKDTree_local(pixel_round.astype(np.float64))
                # Use adaptive radius: roughly the median nearest-neighbor distance
                if pixel_round.shape[0] > 1:
                    distances = tree.query(pixel_round.astype(np.float64), k=min(3, pixel_round.shape[0]))[0]
                    nn_dist = float(np.median(distances[:, -1])) if distances.ndim > 1 else float(np.median(distances))
                else:
                    nn_dist = 5.0
                radius = max(4.0, min(48.0, nn_dist * 2.5))
                visited = np.zeros((pixel_round.shape[0],), dtype=bool)
                components: list[np.ndarray] = []
                for start in range(pixel_round.shape[0]):
                    if visited[start]:
                        continue
                    queue = [start]
                    visited[start] = True
                    comp = []
                    head = 0
                    while head < len(queue):
                        node = queue[head]
                        head += 1
                        comp.append(node)
                        for nb in tree.query_ball_point(pixel_round[node].astype(np.float64), r=radius):
                            nb = int(nb)
                            if not visited[nb]:
                                visited[nb] = True
                                queue.append(nb)
                    components.append(pixel_round[np.asarray(comp, dtype=np.int64)])

                hull_result = np.zeros_like(base_mask, dtype=bool)
                for comp_points in components:
                    if comp_points.shape[0] < comp_hull_min_pts:
                        # Too few points, keep them as-is
                        for px, py in comp_points:
                            px_c = int(np.clip(px, 0, image_size[0] - 1))
                            py_c = int(np.clip(py, 0, image_size[1] - 1))
                            hull_result[py_c, px_c] = True
                        continue

                    # Compute per-component pixel hull
                    comp_hull_pts = _convex_hull_pixels(comp_points.astype(np.float32))
                    if comp_hull_pts.shape[0] < 3:
                        for px, py in comp_points:
                            px_c = int(np.clip(px, 0, image_size[0] - 1))
                            py_c = int(np.clip(py, 0, image_size[1] - 1))
                            hull_result[py_c, px_c] = True
                        continue

                    # Compute component point mask area for the hull region
                    comp_hull_image = Image.new("L", image_size, 0)
                    comp_hull_draw = ImageDraw.Draw(comp_hull_image)
                    comp_hull_draw.polygon([(float(x), float(y)) for x, y in comp_hull_pts], fill=255)
                    comp_hull_mask = np.asarray(comp_hull_image, dtype=np.uint8) > 0

                    # Count how many original points fall within this component's hull
                    comp_mask_area = 0
                    for px, py in comp_points:
                        px_c = int(np.clip(px, 0, image_size[0] - 1))
                        py_c = int(np.clip(py, 0, image_size[1] - 1))
                        if comp_hull_mask[py_c, px_c]:
                            comp_mask_area += 1

                    hull_area = int(comp_hull_mask.sum())
                    if hull_area > 0 and hull_area <= int(comp_hull_max_area_mult * max(comp_mask_area, 1)):
                        hull_result |= comp_hull_mask
                    else:
                        # Reject hull, keep only point mask for this component
                        for px, py in comp_points:
                            px_c = int(np.clip(px, 0, image_size[0] - 1))
                            py_c = int(np.clip(py, 0, image_size[1] - 1))
                            hull_result[py_c, px_c] = True

                mask_image = Image.fromarray((base_mask | hull_result).astype(np.uint8) * 255, mode="L")

                if comp_close_kernel > 1:
                    closed = _close_binary_mask(np.asarray(mask_image, dtype=np.uint8) > 0, kernel_size=comp_close_kernel)
                    mask_image = Image.fromarray(closed.astype(np.uint8) * 255, mode="L")

                if comp_fill_holes:
                    mask_np = np.asarray(mask_image, dtype=np.uint8) > 0
                    if _ndimage is not None:
                        filled = _ndimage.binary_fill_holes(mask_np)
                        mask_image = Image.fromarray(filled.astype(np.uint8) * 255, mode="L")

    elif use_global_convex_hull:
        min_points = _env_int("GS_QUERY_CLOUD_HULL_MIN_POINTS", 12, minimum=3)
        max_area_multiplier = _env_float("GS_QUERY_CLOUD_HULL_MAX_AREA_MULTIPLIER", 4.0, minimum=1.0)
        if projected.shape[0] >= min_points:
            base_mask = np.asarray(mask_image, dtype=np.uint8) > 0
            base_area = int(base_mask.sum())
            hull_points = _convex_hull_pixels(projected)
            if hull_points.shape[0] >= 3:
                hull_image = Image.new("L", image_size, 0)
                hull_draw = ImageDraw.Draw(hull_image)
                hull_draw.polygon([(float(x), float(y)) for x, y in hull_points], fill=255)
                hull_mask = np.asarray(hull_image, dtype=np.uint8) > 0
                hull_area = int(hull_mask.sum())
                if hull_area > 0 and hull_area <= int(max_area_multiplier * max(base_area, 1)):
                    mask_image = Image.fromarray((base_mask | hull_mask).astype(np.uint8) * 255, mode="L")

    mask = np.asarray(mask_image, dtype=np.uint8) > 0
    if postfilter_kernel > 1:
        mask = _dilate_binary_mask(mask, kernel_size=postfilter_kernel)
    bbox = _entity_mask_bbox(mask)
    if bbox is None:
        return None, None
    return mask, bbox


def _overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: int) -> Image.Image:
    base = image.convert("RGBA")
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay[mask, 0] = color[0]
    overlay[mask, 1] = color[1]
    overlay[mask, 2] = color[2]
    overlay[mask, 3] = int(np.clip(alpha, 0, 255))
    return Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))


def _mask_area_fraction(mask: np.ndarray | None) -> float:
    if mask is None:
        return 0.0
    binary = np.asarray(mask, dtype=bool)
    return float(binary.sum()) / max(float(binary.size), 1.0)


def _mask_boundary(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    return _dilate_binary_mask(binary, kernel_size=3) ^ _erode_binary_mask(binary, kernel_size=3)


def _overlay_boundary(image: Image.Image, mask: np.ndarray | None, color: tuple[int, int, int], alpha: int = 230) -> Image.Image:
    boundary = _mask_boundary(mask)
    if boundary is None:
        return image.convert("RGBA")
    base = image.convert("RGBA")
    overlay = np.zeros((boundary.shape[0], boundary.shape[1], 4), dtype=np.uint8)
    overlay[boundary, 0] = color[0]
    overlay[boundary, 1] = color[1]
    overlay[boundary, 2] = color[2]
    overlay[boundary, 3] = int(np.clip(alpha, 0, 255))
    return Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))


def _draw_panel_caption(image: Image.Image, caption: str) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    font = _load_font(18)
    bbox = draw.textbbox((10, 10), caption, font=font)
    draw.rectangle((bbox[0] - 5, bbox[1] - 3, bbox[2] + 5, bbox[3] + 3), fill=(16, 16, 16, 210))
    draw.text((10, 10), caption, fill=(245, 245, 245, 255), font=font)


def _compose_lifecycle_frame(
    source: Image.Image,
    stage1_mask: np.ndarray | None,
    cloud_mask: np.ndarray | None,
    final_mask: np.ndarray | None,
    frame_index: int,
    query_active: bool,
    fusion_source: str,
) -> Image.Image:
    src = source.convert("RGB")
    stage = _overlay_boundary(src, stage1_mask, (60, 255, 120), alpha=245).convert("RGB")
    cloud = _overlay_mask(src, np.asarray(cloud_mask, dtype=bool), (255, 176, 48), alpha=104).convert("RGB") if cloud_mask is not None else src.copy()
    cloud = _overlay_boundary(cloud, stage1_mask, (60, 255, 120), alpha=220).convert("RGB")
    final = _overlay_mask(src, np.asarray(final_mask, dtype=bool), (255, 64, 196), alpha=132).convert("RGB") if final_mask is not None else src.copy()
    final = _overlay_boundary(final, stage1_mask, (60, 255, 120), alpha=220).convert("RGB")
    panels = [
        (src.copy(), "source"),
        (stage, "stage1 boundary"),
        (cloud, "selected gaussian projection"),
        (final, f"refined final mask  {fusion_source}"),
    ]
    scale = _env_float("GS_QUERY_LIFECYCLE_PANEL_SCALE", 0.50, minimum=0.20, maximum=1.0)
    panel_w = max(1, int(round(src.size[0] * scale)))
    panel_h = max(1, int(round(src.size[1] * scale)))
    canvas = Image.new("RGB", (panel_w * 2, panel_h * 2), (0, 0, 0))
    for idx, (panel, caption) in enumerate(panels):
        resized = panel.resize((panel_w, panel_h), Image.Resampling.BILINEAR)
        suffix = f" frame={frame_index:04d} active={'yes' if query_active else 'no'}" if idx == 0 else ""
        _draw_panel_caption(resized, caption + suffix)
        canvas.paste(resized, ((idx % 2) * panel_w, (idx // 2) * panel_h))
    return canvas


def _contact_threshold(patient_extent: np.ndarray, tool_extent: np.ndarray) -> float:
    patient_radius = 0.5 * float(np.linalg.norm(patient_extent))
    tool_radius = 0.5 * float(np.linalg.norm(tool_extent))
    return max(0.10, 0.65 * (patient_radius + tool_radius))


def _video_meta(path: Path, fps: int) -> dict[str, Any]:
    return {
        "path": str(path),
        "fps": int(fps),
        "exists": path.exists(),
    }


def _open_video_writer(path: Path, fps: int):
    try:
        writer = iio_v2.get_writer(
            str(path),
            fps=fps,
            codec="libx264",
            macro_block_size=None,
        )
        return writer, path
    except Exception:
        fallback_path = path.with_suffix(".gif")
        writer = iio_v2.get_writer(str(fallback_path), mode="I", fps=fps)
        return writer, fallback_path


class _NoopVideoWriter:
    def append_data(self, _frame: np.ndarray) -> None:
        return

    def close(self) -> None:
        return


def render_hypernerf_query_video(
    run_dir: str | Path,
    dataset_dir: str | Path,
    selection_path: str | Path,
    output_dir: str | Path | None = None,
    fps: int = 12,
    stride: int = 1,
    background_mode: str = "render",
    eval_profile: str | None = None,
    image_ids: list[str] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    dataset_dir = Path(dataset_dir)
    selection_path = Path(selection_path)
    output_dir = Path(output_dir) if output_dir is not None else selection_path.parent / "native_render"
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_frame_dir = output_dir / "overlay_frames"
    binary_mask_dir = output_dir / "binary_masks"
    overlay_frame_dir.mkdir(parents=True, exist_ok=True)
    binary_mask_dir.mkdir(parents=True, exist_ok=True)
    resolved_eval_profile = _resolve_eval_profile(eval_profile)
    _apply_render_profile_env_defaults(resolved_eval_profile)
    export_lifecycle = _env_flag(
        "GS_QUERY_EXPORT_ENTITY_LIFECYCLE",
        resolved_eval_profile in {"boundary_refine_v1", "public_boundary_v1", "mask_boundary_refine"},
    )
    lifecycle_dir = output_dir / "entity_lifecycle"
    lifecycle_frame_dir = lifecycle_dir / "frames"
    fast_validation_only = _env_flag("QUERY_FAST_VALIDATION_ONLY", False)
    if fast_validation_only:
        export_lifecycle = False
    if export_lifecycle:
        lifecycle_frame_dir.mkdir(parents=True, exist_ok=True)
    skip_video_export = fast_validation_only or os.environ.get("QUERY_SKIP_VIDEO_EXPORT", "0").strip().lower() in {"1", "true", "yes", "on"}
    save_overlay_frames = not _env_flag("QUERY_SKIP_OVERLAY_FRAME_EXPORT", fast_validation_only)
    save_key_frames = _env_flag("QUERY_SAVE_KEY_FRAMES", not fast_validation_only)
    save_inactive_binary_masks = not _env_flag("QUERY_RENDER_ACTIVE_MASKS_ONLY", fast_validation_only)
    binary_png_compress_level = int(np.clip(_env_int("QUERY_RENDER_MASK_PNG_COMPRESS_LEVEL", 6, minimum=0), 0, 9))

    if background_mode not in {"render", "source"}:
        raise ValueError(f"Unsupported background_mode: {background_mode}")

    source_frame_dir = None
    source_frame_paths: dict[str, Path] = {}
    render_dir: Path | None = None
    render_files: list[Path]
    hypernerf_ids = _hypernerf_test_ids(dataset_dir)
    requested_image_ids = [str(value).strip() for value in image_ids or [] if str(value).strip()]
    requested_image_ids = list(dict.fromkeys(requested_image_ids))

    # The selection's temporal segments live on the reconstruction test grid.
    # Keep that grid even when the final entity is projected into an explicit
    # evaluation-camera set below.
    reference_render_dir = _find_render_dir(run_dir)
    reference_render_files = sorted(reference_render_dir.glob("*.png"))
    if not reference_render_files:
        raise FileNotFoundError(f"No frames found in {reference_render_dir}")
    if hypernerf_ids is not None:
        reference_ids, reference_times = hypernerf_ids
        if len(reference_ids) != len(reference_render_files):
            raise ValueError(
                "Render frame count "
                f"({len(reference_render_files)}) does not match HyperNeRF test ids ({len(reference_ids)})"
            )
    else:
        reference_ids, reference_times = _dynerf_test_ids_from_renders(reference_render_dir)

    evaluation_camera_export = bool(requested_image_ids)
    if evaluation_camera_export:
        if background_mode != "source":
            raise ValueError("Explicit image_ids require background_mode='source' so camera/image pairs stay synchronized.")
        entries = resolve_dataset_image_entries(dataset_dir)
        entry_by_id = {str(entry["image_id"]): entry for entry in entries}
        # Retain the complete reconstruction test grid for temporal metrics,
        # then add exact benchmark cameras for spatial metrics.  A target-only
        # export could make a sparse active-frame subset look temporally perfect.
        camera_image_ids = list(dict.fromkeys([*map(str, reference_ids), *requested_image_ids]))
        missing_ids = [image_id for image_id in camera_image_ids if image_id not in entry_by_id]
        if missing_ids:
            raise FileNotFoundError(
                "Requested evaluation image ids are absent from the dataset: " + ", ".join(missing_ids[:8])
            )
        requested_entries = sorted(
            (entry_by_id[image_id] for image_id in camera_image_ids),
            key=lambda entry: int(entry["frame_index"]),
        )
        render_files = [Path(str(entry["image_path"])) for entry in requested_entries]
        source_frame_paths = {
            str(entry["image_id"]): Path(str(entry["image_path"]))
            for entry in requested_entries
        }
        parents = {path.parent for path in source_frame_paths.values()}
        source_frame_dir = next(iter(parents)) if len(parents) == 1 else None
        test_ids = [str(entry["image_id"]) for entry in requested_entries]
        test_times = np.asarray([float(entry["time_value"]) for entry in requested_entries], dtype=np.float32)
        with Image.open(render_files[0]) as probe:
            target_size = probe.size
    else:
        render_dir = reference_render_dir
        render_files = reference_render_files
        test_ids = list(reference_ids)
        test_times = np.asarray(reference_times, dtype=np.float32)
        with Image.open(render_files[0]) as probe:
            target_size = probe.size
        if background_mode == "source":
            # This affects only the visualization backdrop. Entity masks and
            # geometry still come from the ReferGaussian render and 3D cloud.
            source_frame_dir = _find_source_frame_dir(dataset_dir, target_size)

    selection_payload = _read_json(selection_path)
    fusion_options = _fusion_options_for_profile(resolved_eval_profile)
    selected_items = selection_payload.get("selected", [])
    selection_empty = selection_payload.get("empty", False)
    if not selected_items:
        # Negative query: Qwen determined entity doesn't satisfy the query.
        # Produce all-inactive (all-black) binary masks so evaluator can score correctly.
        frame_records = []
        for frame_index, (test_id, time_value) in enumerate(zip(test_ids, test_times)):
            if save_inactive_binary_masks:
                bg_file = (
                    render_files[frame_index]
                    if frame_index < len(render_files)
                    else (render_files[-1] if render_files else None)
                )
                with Image.open(bg_file) as bg_img:
                    W, H = bg_img.size
                black_mask = Image.fromarray(np.zeros((H, W), dtype=np.uint8))
                mask_fname = f"{str(test_id).zfill(6)}.png"
                overlay_fname = f"{str(test_id).zfill(6)}.png"
                black_mask.save(binary_mask_dir / mask_fname, compress_level=binary_png_compress_level)
                # Save overlay as the background frame (no highlight)
                bg_file_copy = render_files[frame_index] if render_files else None
                if save_overlay_frames and bg_file_copy is not None:
                    shutil.copy2(bg_file_copy, overlay_frame_dir / overlay_fname)
            frame_records.append({
                "frame_index": frame_index,
                "image_id": str(test_id),
                "time_value": float(time_value),
                "query_active": False,
                "entity_active": False,
            })
        validation_data = {
            "query": selection_payload.get("query", "query"),
            "selection_mode": selection_payload.get("selection_mode", "qwen_plan_empty"),
            "eval_profile": resolved_eval_profile,
            "fusion_options": fusion_options,
            "query_intent_mode": _query_intent_mode(str(selection_payload.get("query", "")), track_state_mode=None),
            "empty_selection": True,
            "camera_export": {
                "mode": "explicit_source_camera" if evaluation_camera_export else "reconstruction_test_camera",
                "image_ids": list(test_ids),
                "requested_image_ids": list(requested_image_ids),
            },
            "frame_exports": {
                "overlay_frames": str(overlay_frame_dir),
                "binary_masks": str(binary_mask_dir),
                "entity_lifecycle": str(lifecycle_frame_dir) if export_lifecycle else None,
            },
            "frames": frame_records,
        }
        _write_json(output_dir / "validation.json", validation_data)
        return output_dir
    track_state_mode = _selection_track_state_mode(selection_payload)
    query_text = str(selection_payload.get("query", "query"))
    query_intent_mode = _query_intent_mode(query_text, track_state_mode=track_state_mode)

    tracks = _load_tracks(run_dir)
    entity_text_map = _load_entity_static_texts(run_dir)
    entity_phrase_hints = _load_entity_phrase_hints(run_dir)
    query_tracks = _resolve_query_tracks(selection_path)
    require_query_tracks = _env_flag(
        "GS_QUERY_REQUIRE_STAGE1_TRACKS",
        _is_coverage_render_profile(resolved_eval_profile),
    )
    require_synchronized_boundaries = _env_flag(
        "GS_QUERY_REQUIRE_SYNCHRONIZED_STAGE1_BOUNDARY",
        _is_formal_boundary_gated_profile(resolved_eval_profile),
    )
    if require_query_tracks and not query_tracks:
        raise FileNotFoundError(
            "Stage1 query tracks are required for this render profile, but none were found. "
            "Keep the original query root layout or copy the complete query root including grounded_sam2/."
        )
    entity_clouds = _load_entity_clouds(run_dir, selected_items)
    role_entries = []
    for item_index, item in enumerate(selected_items):
        entity_id = int(item["id"])
        track = tracks.get(entity_id)
        if track is None:
            continue
        phrase_hints = entity_phrase_hints.get(entity_id, [])
        if not phrase_hints:
            phrase_hints = [_normalize_phrase_key(entity_text_map.get(entity_id, ""))]
        query_track = _resolve_query_track_for_hints(query_tracks, phrase_hints)
        role_entries.append(
            {
                "role": str(item.get("role", "other")),
                "entity_id": entity_id,
                "source_entity_id": int(item.get("source_entity_id", -1)),
                "confidence": float(item.get("confidence", 0.0)),
                "segments": item.get("segments", []),
                "track": track,
                "center_world": _interp_vector(test_times, track.time_values, track.centers),
                "extent_min": _interp_vector(test_times, track.time_values, track.extents_min),
                "extent_max": _interp_vector(test_times, track.time_values, track.extents_max),
                "visibility": _interp_scalar(test_times, track.time_values, track.visibility),
                "support_score": _interp_scalar(test_times, track.time_values, track.support_score),
                "cloud": entity_clouds.get(int(item_index)),
                "track_phrase": _normalize_phrase_key(entity_text_map.get(entity_id, "")),
                "track_phrase_hints": phrase_hints,
                "query_track": query_track,
                }
            )
    if require_query_tracks:
        missing_track_roles = [
            f"{entry['role']}:{entry['entity_id']}"
            for entry in role_entries
            if entry.get("query_track") is None
        ]
        if missing_track_roles:
            raise ValueError(
                "Stage1 query tracks could not be matched for selected entities: "
                + ", ".join(missing_track_roles)
            )
    if not role_entries:
        raise ValueError("Selected entities could not be matched to target semantic tracks")

    frame_count = len(render_files)
    for entry in role_entries:
        entry["frame_mask"] = _frame_mask_at_render_times(
            entry["segments"],
            np.asarray(reference_times, dtype=np.float32),
            np.asarray(test_times, dtype=np.float32),
        )
    active_mask = np.zeros((frame_count,), dtype=bool)
    for entry in role_entries:
        active_mask |= entry["frame_mask"]

    patient_entry = next((entry for entry in role_entries if entry["role"] == "patient"), None)
    tool_entry = next((entry for entry in role_entries if entry["role"] == "tool"), None)
    contact_mask = np.zeros((frame_count,), dtype=bool)
    proximity_mask = np.zeros((frame_count,), dtype=bool)
    contact_distance = np.full((frame_count,), np.nan, dtype=np.float32)
    if patient_entry is not None and tool_entry is not None:
        patient_extent = patient_entry["extent_max"] - patient_entry["extent_min"]
        tool_extent = tool_entry["extent_max"] - tool_entry["extent_min"]
        threshold = np.asarray(
            [_contact_threshold(patient_extent[i], tool_extent[i]) for i in range(frame_count)],
            dtype=np.float32,
        )
        distance = np.linalg.norm(patient_entry["center_world"] - tool_entry["center_world"], axis=1)
        overlap_mask = patient_entry["frame_mask"] & tool_entry["frame_mask"]
        proximity_mask = overlap_mask & (distance <= threshold)
        contact_mask = overlap_mask.copy()
        contact_distance = distance.astype(np.float32)

    if skip_video_export:
        overlay_path = output_dir / "overlay.mp4"
        mask_path = output_dir / "mask.mp4"
        overlay_writer = _NoopVideoWriter()
        mask_writer = _NoopVideoWriter()
        lifecycle_path_video = lifecycle_dir / "lifecycle.mp4"
        lifecycle_writer = _NoopVideoWriter()
    else:
        overlay_writer, overlay_path = _open_video_writer(output_dir / "overlay.mp4", fps=fps)
        mask_writer, mask_path = _open_video_writer(output_dir / "mask.mp4", fps=fps)
        if export_lifecycle:
            lifecycle_writer, lifecycle_path_video = _open_video_writer(lifecycle_dir / "lifecycle.mp4", fps=fps)
        else:
            lifecycle_path_video = lifecycle_dir / "lifecycle.mp4"
            lifecycle_writer = _NoopVideoWriter()

    previous_role_masks: dict[int, np.ndarray] = {}
    first_active_frame = None
    first_contact_frame = None
    frame_records = []
    saved_frames: list[tuple[str, int, Path]] = []

    try:
        for frame_index, (frame_path, image_id, time_value) in enumerate(zip(render_files, test_ids, test_times)):
            if frame_index % max(stride, 1) != 0:
                continue
            query_active = bool(active_mask[frame_index])
            if fast_validation_only and not query_active:
                frame_records.append(
                    {
                        "frame_index": int(frame_index),
                        "image_id": image_id,
                        "time_value": float(time_value),
                        "query_active": False,
                        "contact_active": False,
                        "proximity_contact_active": False,
                        "contact_distance_world": None,
                        "stage1_area_fraction": 0.0,
                        "cloud_projection_area_fraction": 0.0,
                        "final_mask_area_fraction": 0.0,
                        "fusion_sources": [],
                        "final_grounding_sources": [],
                        "used_direct_stage1_mask": False,
                        "final_to_cloud_iou": 1.0,
                        "cloud_to_stage1_iou": 1.0,
                        "cloud_to_stage1_precision": 1.0,
                        "cloud_to_stage1_recall": 1.0,
                        "final_to_stage1_iou": 1.0,
                        "final_to_stage1_precision": 1.0,
                        "final_to_stage1_recall": 1.0,
                        "lifecycle_frame": None,
                        "roles": [],
                    }
                )
                continue
            if fast_validation_only:
                frame = Image.new("RGB", target_size, color=(0, 0, 0))
            elif background_mode == "source":
                source_path = source_frame_paths.get(str(image_id))
                if source_path is None and source_frame_dir is not None:
                    source_path = source_frame_dir / f"{image_id}.png"
                if source_path is None:
                    raise FileNotFoundError(f"No source image path resolved for {image_id}")
                if not source_path.exists():
                    raise FileNotFoundError(f"Missing source frame {source_path}")
                frame = Image.open(source_path).convert("RGB")
                if frame.size != target_size:
                    frame = frame.resize(target_size, Image.Resampling.BILINEAR)
            else:
                frame = Image.open(frame_path).convert("RGB")
            overlay = frame.copy()
            binary_union_mask = np.zeros((frame.size[1], frame.size[0]), dtype=bool)
            lifecycle_stage1_union = np.zeros_like(binary_union_mask, dtype=bool)
            lifecycle_cloud_union = np.zeros_like(binary_union_mask, dtype=bool)
            frame_fusion_sources: list[str] = []
            overlay_draw = ImageDraw.Draw(overlay, "RGBA")

            camera = _camera_class().from_json(dataset_dir / "camera" / f"{image_id}.json")
            frame_roles = []
            for entry in role_entries:
                role = entry["role"]
                active = bool(entry["frame_mask"][frame_index])
                visible = bool(entry["visibility"][frame_index] >= 0.2)
                center_world = entry["center_world"][frame_index]
                pixel = _project_points_to_image(camera, center_world[None, :], image_size=overlay.size)[0]
                center_local = camera.points_to_local_points(center_world[None, :])[0]
                in_front = bool(center_local[2] > 1.0e-4)
                width, height = overlay.size
                in_bounds = bool(0.0 <= pixel[0] < width and 0.0 <= pixel[1] < height)
                displayable = bool(in_front)
                clamped_pixel = np.asarray(
                    [
                        np.clip(pixel[0], 8.0, width - 8.0),
                        np.clip(pixel[1], 8.0, height - 8.0),
                    ],
                    dtype=np.float32,
                )
                radius = _pixel_radius(
                    camera,
                    center_world,
                    entry["extent_min"][frame_index],
                    entry["extent_max"][frame_index],
                    image_size=overlay.size,
                )
                projected_box = _project_box(
                    camera,
                    entry["extent_min"][frame_index],
                    entry["extent_max"][frame_index],
                    overlay.size,
                )
                cloud_mask = None
                cloud_bbox = None
                strict_stage1_match = bool(fusion_options.get("strict_gaussian_projection", False))
                query_track_mask, query_track_meta = _query_track_match_for_time(
                    entry.get("query_track"),
                    float(time_value),
                    strict=strict_stage1_match,
                )
                query_track_mask = _resize_mask_to_shape(query_track_mask, (overlay.size[1], overlay.size[0]))
                query_track_bbox = _entity_mask_bbox(query_track_mask) if query_track_mask is not None else None
                if query_track_mask is not None:
                    lifecycle_stage1_union |= np.asarray(query_track_mask, dtype=bool)
                if entry.get("cloud") is not None:
                    cloud_mask, cloud_bbox = _project_entity_cloud_mask(
                        camera,
                        overlay.size,
                        entry["cloud"],
                        float(time_value),
                        eval_profile=resolved_eval_profile,
                        stage1_boundary_available=query_track_mask is not None,
                    )
                    if cloud_mask is not None:
                        lifecycle_cloud_union |= np.asarray(cloud_mask, dtype=bool)
                role_record = {
                    "role": role,
                    "entity_id": int(entry["entity_id"]),
                    "source_entity_id": int(entry["source_entity_id"]),
                    "confidence": float(entry["confidence"]),
                    "active": active,
                    "visible": visible,
                    "displayable": displayable,
                    "projected": bool(displayable and in_bounds),
                    "offscreen": bool(displayable and not in_bounds),
                    "pixel_xy": [float(pixel[0]), float(pixel[1])],
                    "display_xy": [float(clamped_pixel[0]), float(clamped_pixel[1])],
                    "depth": float(center_local[2]),
                    "radius_px": int(radius),
                    "bbox_xyxy_raw": None if projected_box is None else projected_box["raw_xyxy"],
                    "bbox_xyxy_clamped": None if projected_box is None else projected_box["clamped_xyxy"],
                    "bbox_intersects_frame": False if projected_box is None else bool(projected_box["intersects_frame"]),
                    "query_track_bbox_xyxy": query_track_bbox,
                    "cloud_mask_bbox_xyxy": cloud_bbox,
                    "query_track_area_fraction": _mask_area_fraction(query_track_mask),
                    "cloud_mask_area_fraction": _mask_area_fraction(cloud_mask),
                    "cloud_projection_mode": os.environ.get("GS_QUERY_CLOUD_RENDER_MODE", "point_hull"),
                    "cloud_opacity_source": None if entry.get("cloud") is None else entry["cloud"].opacity_source,
                    "cloud_alpha_relative_threshold": (
                        None
                        if entry.get("cloud") is None
                        else entry["cloud"].alpha_relative_threshold
                    ),
                    "cloud_alpha_absolute_threshold": (
                        None
                        if entry.get("cloud") is None
                        else entry["cloud"].alpha_absolute_threshold
                    ),
                    "cloud_alpha_sigma_scale": (
                        None
                        if entry.get("cloud") is None
                        else entry["cloud"].alpha_sigma_scale
                    ),
                    "cloud_alpha_max_splat_radius": (
                        None
                        if entry.get("cloud") is None
                        else entry["cloud"].alpha_max_splat_radius
                    ),
                    "support_score": float(entry["support_score"][frame_index]),
                    **query_track_meta,
                }
                frame_roles.append(role_record)
                if not (active and visible and displayable):
                    continue
                color = _entity_color(int(entry["entity_id"]), role)
                cx = int(round(clamped_pixel[0]))
                cy = int(round(clamped_pixel[1]))
                left = cx - radius
                top = cy - radius
                right = cx + radius
                bottom = cy + radius
                previous_mask = previous_role_masks.get(int(entry["entity_id"]))
                fused_mask, fused_bbox, aligned_query_bbox, fusion_source = _fuse_query_and_cloud_masks(
                    query_track_mask,
                    cloud_mask,
                    projected_bbox=None if projected_box is None else projected_box["clamped_xyxy"],
                    track_state_mode=track_state_mode,
                    selected_item_count=len(selected_items),
                    previous_mask=previous_mask,
                    query_intent_mode=query_intent_mode,
                    fusion_options=fusion_options,
                )
                role_record["fusion_source"] = fusion_source
                role_record["final_grounding_source"] = fusion_source
                role_record["used_direct_stage1_mask"] = str(fusion_source).startswith("query_track")
                role_record["final_to_cloud_iou"] = _mask_iou(fused_mask, cloud_mask)
                frame_fusion_sources.append(str(fusion_source))
                if aligned_query_bbox is not None:
                    role_record["query_track_bbox_xyxy"] = aligned_query_bbox
                if fused_mask is not None and fused_bbox is not None:
                    fused_binary = np.asarray(fused_mask, dtype=bool)
                    previous_role_masks[int(entry["entity_id"])] = fused_binary
                    role_record["fusion_prev_iou"] = _mask_iou(previous_mask, fused_mask)
                    role_record["fused_mask_area_fraction"] = _mask_area_fraction(fused_binary)
                    overlay = _overlay_mask(overlay, fused_binary, color, alpha=132)
                    if query_active:
                        binary_union_mask |= fused_binary
                    overlay_draw = ImageDraw.Draw(overlay, "RGBA")
                    box_left, box_top, box_right, box_bottom = fused_bbox
                    box_left, box_right = sorted((int(round(box_left)), int(round(box_right))))
                    box_top, box_bottom = sorted((int(round(box_top)), int(round(box_bottom))))
                    overlay_draw.rectangle(
                        (box_left, box_top, box_right, box_bottom),
                        outline=color + (255,),
                        width=4,
                    )
                    label_x = max(0, min(box_left, width - 220))
                    label_y = max(0, box_top - 26)
                    _draw_label(overlay_draw, (label_x, label_y), _entity_label(role, int(entry["entity_id"])), color)
                elif cloud_mask is not None and cloud_bbox is not None and not bool(fusion_options.get("strict_gaussian_projection", False)) and not (
                    bool(fusion_options.get("clip_to_query_track", False)) and query_track_mask is not None
                ):
                    cloud_binary = np.asarray(cloud_mask, dtype=bool)
                    previous_role_masks[int(entry["entity_id"])] = cloud_binary
                    role_record["fusion_prev_iou"] = _mask_iou(previous_mask, cloud_mask)
                    role_record["fused_mask_area_fraction"] = _mask_area_fraction(cloud_binary)
                    overlay = _overlay_mask(overlay, cloud_binary, color, alpha=104)
                    if query_active:
                        binary_union_mask |= cloud_binary
                    overlay_draw = ImageDraw.Draw(overlay, "RGBA")
                    box_left, box_top, box_right, box_bottom = cloud_bbox
                    box_left, box_right = sorted((int(round(box_left)), int(round(box_right))))
                    box_top, box_bottom = sorted((int(round(box_top)), int(round(box_bottom))))
                    overlay_draw.rectangle(
                        (box_left, box_top, box_right, box_bottom),
                        outline=color + (255,),
                        width=4,
                    )
                    label_x = max(0, min(box_left, width - 220))
                    label_y = max(0, box_top - 26)
                    _draw_label(overlay_draw, (label_x, label_y), _entity_label(role, int(entry["entity_id"])), color)
                else:
                    role_record["fusion_prev_iou"] = None
                    overlay_draw = ImageDraw.Draw(overlay, "RGBA")
                    if projected_box is not None and projected_box["intersects_frame"]:
                        box_left, box_top, box_right, box_bottom = [int(round(v)) for v in projected_box["clamped_xyxy"]]
                        box_left, box_right = sorted((box_left, box_right))
                        box_top, box_bottom = sorted((box_top, box_bottom))
                        overlay_draw.rectangle(
                            (box_left, box_top, box_right, box_bottom),
                            fill=color + (56,),
                            outline=color + (255,),
                            width=6,
                        )
                        inner_left = min(box_right, box_left + 3)
                        inner_top = min(box_bottom, box_top + 3)
                        inner_right = max(box_left, box_right - 3)
                        inner_bottom = max(box_top, box_bottom - 3)
                        if inner_right >= inner_left and inner_bottom >= inner_top:
                            overlay_draw.rectangle(
                                (inner_left, inner_top, inner_right, inner_bottom),
                                outline=color + (156,),
                                width=2,
                            )
                        overlay_draw.line((cx - radius, cy, cx + radius, cy), fill=color + (255,), width=3)
                        overlay_draw.line((cx, cy - radius, cx, cy + radius), fill=color + (255,), width=3)
                        label_x = max(0, min(box_left, width - 220))
                        label_y = max(0, box_top - 26)
                        _draw_label(overlay_draw, (label_x, label_y), _entity_label(role, int(entry["entity_id"])), color)
                    elif in_bounds:
                        overlay_draw.ellipse((left, top, right, bottom), fill=color + (72,), outline=color + (255,), width=5)
                        overlay_draw.line((cx - radius, cy, cx + radius, cy), fill=color + (255,), width=3)
                        overlay_draw.line((cx, cy - radius, cx, cy + radius), fill=color + (255,), width=3)
                        _draw_label(overlay_draw, (left, max(0, top - 24)), _entity_label(role, int(entry["entity_id"])), color)
                    else:
                        overlay_draw.regular_polygon(
                            (cx, cy, max(radius, 16)),
                            n_sides=4,
                            rotation=45,
                            fill=color + (72,),
                            outline=color + (255,),
                            width=5,
                        )
                        _draw_label(
                            overlay_draw,
                            (min(width - 260, cx + 12), max(0, cy - 12)),
                            f"{_entity_label(role, int(entry['entity_id']))} center offscreen",
                            color,
                        )

            final_erode_kernel = int(fusion_options.get("final_erode_kernel", 1) or 1)
            if final_erode_kernel > 1 and binary_union_mask.any():
                binary_union_mask = _erode_binary_mask(binary_union_mask, kernel_size=_odd_kernel(final_erode_kernel))

            patient_role = next((record for record in frame_roles if record["role"] == "patient"), None)
            tool_role = next((record for record in frame_roles if record["role"] == "tool"), None)
            is_contact = bool(contact_mask[frame_index]) if frame_index < contact_mask.shape[0] else False
            is_proximity_contact = bool(proximity_mask[frame_index]) if frame_index < proximity_mask.shape[0] else False
            if patient_role and tool_role and patient_role["displayable"] and tool_role["displayable"]:
                p0 = tuple(int(round(v)) for v in patient_role["display_xy"])
                p1 = tuple(int(round(v)) for v in tool_role["display_xy"])
                line_color = (255, 96, 96, 220) if is_contact else (255, 255, 255, 160)
                overlay_draw.line((p0[0], p0[1], p1[0], p1[1]), fill=line_color, width=4)
                if is_contact:
                    mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
                    label = "contact" if is_proximity_contact else "interaction"
                    _draw_label(overlay_draw, (mid[0] + 8, mid[1] + 8), label, (255, 96, 96))

            if query_active and first_active_frame is None:
                first_active_frame = frame_index
            if is_contact and first_contact_frame is None:
                first_contact_frame = frame_index

            headline = f"ReferGaussian query render: {query_text}"
            visible_count = int(sum(1 for record in frame_roles if record["active"] and record["displayable"]))
            status = f"frame {frame_index:04d}  time={float(time_value):.3f}  active={'yes' if query_active else 'no'}  entities={visible_count}"
            overlay_draw = ImageDraw.Draw(overlay, "RGBA")
            _draw_label(overlay_draw, (16, 16), headline, (240, 240, 240))
            _draw_label(overlay_draw, (16, 44), status, (220, 220, 220))

            overlay_rgb = overlay.convert("RGB") if overlay.mode != "RGB" else overlay
            overlay_np = np.asarray(overlay_rgb, dtype=np.uint8)
            mask_uint8 = binary_union_mask.astype(np.uint8) * 255
            mask_np = np.repeat(mask_uint8[:, :, None], 3, axis=2)
            overlay_writer.append_data(overlay_np)
            mask_writer.append_data(mask_np)
            if save_overlay_frames:
                overlay_rgb.save(overlay_frame_dir / f"{frame_index:05d}.png")
            Image.fromarray(mask_uint8, mode="L").save(
                binary_mask_dir / f"{frame_index:05d}.png",
                compress_level=binary_png_compress_level,
            )
            lifecycle_path = None
            if export_lifecycle:
                source_names = sorted(set(frame_fusion_sources)) if frame_fusion_sources else ["none"]
                frame_grounding_sources = sorted(
                    set(str(record.get("final_grounding_source", record.get("fusion_source", "none"))) for record in frame_roles)
                ) if frame_roles else ["none"]
                lifecycle_image = _compose_lifecycle_frame(
                    source=frame,
                    stage1_mask=lifecycle_stage1_union if lifecycle_stage1_union.any() else None,
                    cloud_mask=lifecycle_cloud_union if lifecycle_cloud_union.any() else None,
                    final_mask=binary_union_mask if binary_union_mask.any() else None,
                    frame_index=frame_index,
                    query_active=query_active,
                    fusion_source="+".join(frame_grounding_sources[:3]),
                )
                lifecycle_path = lifecycle_frame_dir / f"{frame_index:05d}.png"
                lifecycle_image.save(lifecycle_path)
                lifecycle_writer.append_data(np.asarray(lifecycle_image.convert("RGB"), dtype=np.uint8))

            cloud_to_stage1_precision, cloud_to_stage1_recall = _mask_precision_recall(
                lifecycle_cloud_union,
                lifecycle_stage1_union,
            )
            final_to_stage1_precision, final_to_stage1_recall = _mask_precision_recall(
                binary_union_mask,
                lifecycle_stage1_union,
            )
            frame_records.append(
                {
                    "frame_index": int(frame_index),
                    "image_id": image_id,
                    "time_value": float(time_value),
                    "query_active": query_active,
                    "contact_active": is_contact,
                    "proximity_contact_active": is_proximity_contact,
                    "contact_distance_world": None
                    if np.isnan(contact_distance[frame_index])
                    else float(contact_distance[frame_index]),
                    "stage1_area_fraction": _mask_area_fraction(lifecycle_stage1_union),
                    "cloud_projection_area_fraction": _mask_area_fraction(lifecycle_cloud_union),
                    "final_mask_area_fraction": _mask_area_fraction(binary_union_mask),
                    "fusion_sources": sorted(set(frame_fusion_sources)),
                    "final_grounding_sources": sorted(
                        set(str(record.get("final_grounding_source", record.get("fusion_source", "none"))) for record in frame_roles)
                    ) if frame_roles else ["none"],
                    "used_direct_stage1_mask": bool(any(bool(record.get("used_direct_stage1_mask", False)) for record in frame_roles)),
                    "final_to_cloud_iou": _mask_iou(binary_union_mask, lifecycle_cloud_union),
                    "cloud_to_stage1_iou": _mask_iou(lifecycle_cloud_union, lifecycle_stage1_union),
                    "cloud_to_stage1_precision": cloud_to_stage1_precision,
                    "cloud_to_stage1_recall": cloud_to_stage1_recall,
                    "final_to_stage1_iou": _mask_iou(binary_union_mask, lifecycle_stage1_union),
                    "final_to_stage1_precision": final_to_stage1_precision,
                    "final_to_stage1_recall": final_to_stage1_recall,
                    "lifecycle_frame": None if lifecycle_path is None else str(lifecycle_path),
                    "roles": frame_roles,
                }
            )

            should_save = (
                (first_active_frame is not None and frame_index == first_active_frame)
                or (first_contact_frame is not None and frame_index == first_contact_frame)
            )
            if should_save and save_key_frames:
                label = "first_contact" if frame_index == first_contact_frame else "first_active"
                out_path = output_dir / f"{label}_{frame_index:04d}.png"
                overlay_rgb.save(out_path)
                saved_frames.append((label, frame_index, out_path))
    finally:
        overlay_writer.close()
        mask_writer.close()
        lifecycle_writer.close()

    active_indices = [record["frame_index"] for record in frame_records if record["query_active"]]
    contact_indices = [record["frame_index"] for record in frame_records if record["contact_active"]]
    boundary_coverage = _stage1_boundary_coverage_summary(
        frame_records,
        required=require_synchronized_boundaries,
    )
    payload = {
        "schema_version": 1,
        "query": selection_payload.get("query", "query"),
        "eval_profile": resolved_eval_profile,
        "fusion_options": fusion_options,
        "query_intent_mode": query_intent_mode,
        "track_state_mode": track_state_mode,
        "selection_path": str(selection_path),
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "native_render": True,
        "camera_export": {
            "mode": "explicit_source_camera" if evaluation_camera_export else "reconstruction_test_camera",
            "image_ids": list(test_ids),
            "requested_image_ids": list(requested_image_ids),
            "selection_temporal_reference_frame_count": int(len(reference_times)),
        },
        "background_mode": background_mode,
        "background_frame_dir": None if source_frame_dir is None else str(source_frame_dir),
        "render_dir": str(render_dir) if render_dir is not None else str(source_frame_dir),
        "frame_count": len(frame_records),
        "active_frame_count": int(len(active_indices)),
        "first_active_frame": None if not active_indices else int(active_indices[0]),
        "last_active_frame": None if not active_indices else int(active_indices[-1]),
        "active_segments": _merge_ranges(active_indices),
        "contact_frame_count": int(len(contact_indices)),
        "first_contact_frame": None if not contact_indices else int(contact_indices[0]),
        "last_contact_frame": None if not contact_indices else int(contact_indices[-1]),
        "contact_segments": _merge_ranges(contact_indices),
        "proximity_contact_frame_count": int(sum(record["proximity_contact_active"] for record in frame_records)),
        "proximity_contact_segments": _merge_ranges(
            [record["frame_index"] for record in frame_records if record["proximity_contact_active"]]
        ),
        "roles": [
            {
                "role": entry["role"],
                "entity_id": int(entry["entity_id"]),
                "source_entity_id": int(entry["source_entity_id"]),
                "confidence": float(entry["confidence"]),
                "segments": entry["segments"],
            }
            for entry in role_entries
        ],
        "video_export_disabled": bool(skip_video_export),
        "binary_mask_semantics": (
            "union_of_synchronized_boundary_gated_gaussian_entity_masks"
            if _is_formal_boundary_gated_profile(resolved_eval_profile)
            else "union_of_fused_or_cloud_entity_masks_only"
        ),
        "stage1_boundary_coverage": boundary_coverage,
        "entity_lifecycle_exported": bool(export_lifecycle),
        "area_fraction_summary": {
            "stage1_mean": float(np.mean([record["stage1_area_fraction"] for record in frame_records])) if frame_records else 0.0,
            "cloud_projection_mean": float(np.mean([record["cloud_projection_area_fraction"] for record in frame_records])) if frame_records else 0.0,
            "final_mask_mean": float(np.mean([record["final_mask_area_fraction"] for record in frame_records])) if frame_records else 0.0,
            "active_final_mask_mean": float(np.mean([record["final_mask_area_fraction"] for record in frame_records if record["query_active"]])) if active_indices else 0.0,
            "active_cloud_to_stage1_iou_mean": float(np.mean([record["cloud_to_stage1_iou"] for record in frame_records if record["query_active"]])) if active_indices else 0.0,
            "active_cloud_to_stage1_precision_mean": float(np.mean([record["cloud_to_stage1_precision"] for record in frame_records if record["query_active"]])) if active_indices else 0.0,
            "active_cloud_to_stage1_recall_mean": float(np.mean([record["cloud_to_stage1_recall"] for record in frame_records if record["query_active"]])) if active_indices else 0.0,
            "active_final_to_stage1_iou_mean": float(np.mean([record["final_to_stage1_iou"] for record in frame_records if record["query_active"]])) if active_indices else 0.0,
            "active_final_to_stage1_precision_mean": float(np.mean([record["final_to_stage1_precision"] for record in frame_records if record["query_active"]])) if active_indices else 0.0,
            "active_final_to_stage1_recall_mean": float(np.mean([record["final_to_stage1_recall"] for record in frame_records if record["query_active"]])) if active_indices else 0.0,
        },
        "videos": {
            "overlay": _video_meta(overlay_path, fps),
            "mask": _video_meta(mask_path, fps),
            "entity_lifecycle": _video_meta(lifecycle_path_video, fps) if export_lifecycle else None,
        },
        "frame_exports": {
            "overlay_frames": str(overlay_frame_dir),
            "binary_masks": str(binary_mask_dir),
            "entity_lifecycle": str(lifecycle_frame_dir) if export_lifecycle else None,
        },
        "saved_frames": [
            {"label": label, "frame_index": int(frame_index), "path": str(path)}
            for label, frame_index, path in saved_frames
        ],
        "frames": frame_records,
    }
    _write_json(output_dir / "validation.json", payload)
    if require_synchronized_boundaries and (
        int(boundary_coverage["missing_match_count"]) > 0
        or int(boundary_coverage["stale_match_count"]) > 0
    ):
        raise RuntimeError(
            "Formal boundary-gated rendering requires synchronized Stage-1 masks for every "
            "active selected entity. See validation.json stage1_boundary_coverage for diagnostics."
        )
    return output_dir
