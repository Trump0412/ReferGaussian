from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .slots import export_semantic_slots
from .tracks import export_semantic_tracks


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _phase_ranges(frames: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    start_frame = None
    scores: list[float] = []
    for frame in frames:
        frame_idx = int(frame.get("frame_index", 0))
        matches = bool(frame.get("is_active")) and str(frame.get("motion_label")) == label
        if matches and start_frame is None:
            start_frame = frame_idx
            scores = [float(frame.get("support_score", 0.0))]
            continue
        if matches:
            scores.append(float(frame.get("support_score", 0.0)))
            continue
        if start_frame is not None:
            ranges.append(
                {
                    "frame_start": int(start_frame),
                    "frame_end": int(frame_idx),
                    "mean_support_score": float(sum(scores) / max(len(scores), 1)),
                }
            )
            start_frame = None
            scores = []
    if start_frame is not None:
        end_frame = int(frames[-1].get("frame_index", 0)) + 1 if frames else start_frame + 1
        ranges.append(
            {
                "frame_start": int(start_frame),
                "frame_end": int(end_frame),
                "mean_support_score": float(sum(scores) / max(len(scores), 1)),
            }
        )
    return ranges


def _top_frames(
    frames: list[dict[str, Any]],
    motion_label: str | None,
    top_k: int,
) -> list[int]:
    candidates = []
    for frame in frames:
        if not bool(frame.get("is_active")):
            continue
        if motion_label is not None and str(frame.get("motion_label")) != motion_label:
            continue
        candidates.append(frame)
    candidates.sort(
        key=lambda item: (-float(item.get("support_score", 0.0)), int(item.get("frame_index", 0)))
    )
    return [int(frame.get("frame_index", 0)) for frame in candidates[:top_k]]


def _interaction_prompts(slot: dict[str, Any], dynamic_ratio: float, static_ratio: float) -> list[str]:
    temporal_mode = str(slot.get("temporal_mode", slot.get("entity_type", "unknown")))
    support_span_ratio = float(slot.get("support_span_ratio", 1.0))
    prompts = ["interaction-centric worldtube region", "time-localized semantic event"]
    if temporal_mode == "transient_event":
        prompts = ["contact or manipulation event", "interaction window with localized support"] + prompts
    elif dynamic_ratio > 0.20 and static_ratio > 0.20:
        prompts = ["dynamic object entering contact", "action phase with static support"] + prompts
    elif support_span_ratio <= 0.35:
        prompts = ["short temporal interaction", "brief action segment"] + prompts
    return prompts


def _semantic_head(slot: dict[str, Any], dynamic_ratio: float, static_ratio: float) -> str:
    temporal_mode = str(slot.get("temporal_mode", slot.get("entity_type", "unknown")))
    support_span_ratio = float(slot.get("support_span_ratio", 1.0))
    if temporal_mode == "transient_event":
        return "interaction"
    if temporal_mode == "dynamic_object":
        if dynamic_ratio >= 0.20 and static_ratio >= 0.20 and support_span_ratio <= 0.45:
            return "interaction"
        return "dynamic"
    if temporal_mode == "temporally_localized_region" and support_span_ratio <= 0.45:
        return "interaction"
    if dynamic_ratio >= 0.20 and static_ratio >= 0.20 and support_span_ratio <= 0.60:
        return "interaction"
    if dynamic_ratio >= 0.12:
        return "dynamic"
    return "static"


def export_semantic_priors(
    run_dir: str | Path,
    min_quality: float = 0.0,
    top_k_frames: int = 6,
) -> Path:
    run_dir = Path(run_dir)
    entitybank_dir = run_dir / "entitybank"

    slots_path = entitybank_dir / "semantic_slots.json"
    tracks_path = entitybank_dir / "semantic_tracks.json"
    if not slots_path.exists():
        export_semantic_slots(run_dir, min_quality=min_quality)
    if not tracks_path.exists():
        export_semantic_tracks(run_dir, min_quality=min_quality)

    slots_payload = _read_json(slots_path)
    tracks_payload = _read_json(tracks_path)
    track_map = {int(track["slot_id"]): track for track in tracks_payload.get("tracks", [])}

    priors = []
    for slot in slots_payload.get("slots", []):
        quality = float(slot.get("quality", 0.0))
        if quality < min_quality:
            continue
        slot_id = int(slot.get("slot_id", -1))
        track = track_map.get(slot_id)
        if track is None:
            continue
        frames = track.get("frames", [])
        dynamic_ratio = float(slot.get("dynamic_frame_ratio", 0.0))
        static_ratio = float(slot.get("stationary_frame_ratio", 0.0))
        head = _semantic_head(slot, dynamic_ratio=dynamic_ratio, static_ratio=static_ratio)
        interaction_prompts = _interaction_prompts(slot, dynamic_ratio=dynamic_ratio, static_ratio=static_ratio)

        support_frames = _top_frames(frames, motion_label=None, top_k=top_k_frames)
        dynamic_frames = _top_frames(frames, motion_label="moving", top_k=top_k_frames)
        static_frames = _top_frames(frames, motion_label="stationary", top_k=top_k_frames)
        dynamic_ranges = _phase_ranges(frames, "moving")
        static_ranges = _phase_ranges(frames, "stationary")

        prior = {
            "prior_id": len(priors),
            "slot_id": slot_id,
            "entity_id": int(slot.get("entity_id", -1)),
            "source_cluster_id": int(slot.get("source_cluster_id", -1)),
            "semantic_head": head,
            "entity_type": slot.get("entity_type", "unknown"),
            "temporal_mode": slot.get("temporal_mode", slot.get("entity_type", "unknown")),
            "quality": quality,
            "role_hints": slot.get("role_hints", []),
            "support_window": {
                "frame_start": int(slot.get("support_frame_start", 0)),
                "frame_end": int(slot.get("support_frame_end", 0)),
                "frame_peak": int(slot.get("support_frame_peak", 0)),
                "span_ratio": float(slot.get("support_span_ratio", 0.0)),
                "peak_value": float(slot.get("support_peak_value", 0.0)),
            },
            "geometry_evidence": {
                "occupancy_mean": float(slot.get("occupancy_mean", 0.0)),
                "visibility_mean": float(slot.get("visibility_mean", 0.0)),
                "tube_ratio_mean": float(slot.get("tube_ratio_mean", 0.0)),
                "support_factor_mean": float(slot.get("support_factor_mean", 0.0)),
                "effective_support_mean": float(slot.get("effective_support_mean", 0.0)),
                "dynamic_frame_ratio": dynamic_ratio,
                "stationary_frame_ratio": static_ratio,
            },
            "static_semantics": {
                "prompt_candidates": slot.get("static_prompt_candidates", []),
                "keyframes": static_frames,
                "frame_ranges": static_ranges,
            },
            "dynamic_semantics": {
                "prompt_candidates": slot.get("dynamic_prompt_candidates", []),
                "keyframes": dynamic_frames,
                "frame_ranges": dynamic_ranges,
            },
            "interaction_semantics": {
                "prompt_candidates": interaction_prompts,
                "keyframes": support_frames[: max(2, min(top_k_frames, 4))],
                "frame_ranges": dynamic_ranges if head == "interaction" else [],
            },
            "query_pack": {
                "global": slot.get("prompt_candidates", []),
                "temporal": slot.get("temporal_prompt_candidates", []),
                "static": slot.get("static_prompt_candidates", []),
                "dynamic": slot.get("dynamic_prompt_candidates", []),
                "interaction": interaction_prompts,
            },
        }
        priors.append(prior)

    payload = {
        "schema_version": 1,
        "iteration": slots_payload.get("iteration"),
        "frame_count": slots_payload.get("frame_count"),
        "num_priors": len(priors),
        "num_static_heads": sum(1 for prior in priors if prior["semantic_head"] == "static"),
        "num_dynamic_heads": sum(1 for prior in priors if prior["semantic_head"] == "dynamic"),
        "num_interaction_heads": sum(1 for prior in priors if prior["semantic_head"] == "interaction"),
        "priors": priors,
    }

    output_path = entitybank_dir / "semantic_priors.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path
