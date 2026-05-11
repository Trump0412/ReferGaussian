from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .slots import export_semantic_slots


def _segment_for_frame(segments: list[dict[str, Any]], frame_idx: int) -> tuple[str, float]:
    for segment in segments:
        t0 = int(segment.get("t0", 0))
        t1 = int(segment.get("t1", t0 + 1))
        if t0 <= frame_idx < t1:
            return str(segment.get("label", "unknown")), float(segment.get("confidence", 0.0))
    return "unknown", 0.0


def _float_list(values: np.ndarray) -> list[float]:
    return [float(v) for v in values.tolist()]


def _inside_support_window(slot: dict[str, Any], frame_idx: int) -> bool:
    frame_start = int(slot.get("support_frame_start", 0))
    frame_end = int(slot.get("support_frame_end", frame_start + 1))
    return frame_start <= frame_idx < max(frame_end, frame_start + 1)


def export_semantic_tracks(
    run_dir: str | Path,
    min_quality: float = 0.0,
    visibility_threshold: float = 0.15,
) -> Path:
    run_dir = Path(run_dir)
    entitybank_dir = run_dir / "entitybank"

    slots_path = entitybank_dir / "semantic_slots.json"
    if not slots_path.exists():
        export_semantic_slots(run_dir, min_quality=min_quality)

    with open(entitybank_dir / "entities.json", "r", encoding="utf-8") as handle:
        entities_payload = json.load(handle)
    with open(slots_path, "r", encoding="utf-8") as handle:
        slots_payload = json.load(handle)

    trajectory_payload = np.load(entitybank_dir / "trajectory_samples.npz")
    trajectories = trajectory_payload["trajectories"]
    gate = trajectory_payload["gate"]
    time_values = trajectory_payload["time_values"]

    entity_map = {int(entity["id"]): entity for entity in entities_payload.get("entities", [])}
    tracks = []
    frame_queries = [
        {
            "frame_index": int(frame_idx),
            "time_value": float(time_values[frame_idx]),
            "active_slots": [],
        }
        for frame_idx in range(int(time_values.shape[0]))
    ]

    for slot in slots_payload.get("slots", []):
        quality = float(slot.get("quality", 0.0))
        if quality < min_quality:
            continue

        entity_id = int(slot["entity_id"])
        entity = entity_map.get(entity_id)
        if entity is None:
            continue

        gaussian_ids = np.asarray(entity.get("gaussian_ids", []), dtype=np.int64)
        if gaussian_ids.size == 0:
            continue

        cluster_trajectories = trajectories[gaussian_ids]
        cluster_gate = gate[gaussian_ids].mean(axis=0).squeeze(-1)
        center_world = cluster_trajectories.mean(axis=0)
        extent_min = cluster_trajectories.min(axis=0)
        extent_max = cluster_trajectories.max(axis=0)

        per_frame_speed = np.zeros((center_world.shape[0],), dtype=np.float32)
        if center_world.shape[0] > 1:
            per_frame_speed[1:] = np.linalg.norm(np.diff(center_world, axis=0), axis=1)

        frames = []
        active_frames = []
        for frame_idx in range(int(center_world.shape[0])):
            motion_label, motion_confidence = _segment_for_frame(slot.get("segments", []), frame_idx)
            visibility = float(cluster_gate[frame_idx])
            inside_support = _inside_support_window(slot, frame_idx)
            occupancy_mean = float(slot.get("occupancy_mean", 0.0))
            support_boost = 1.0 if inside_support else 0.5
            support_score = visibility * max(quality, 1.0e-6) * support_boost * max(occupancy_mean, 0.05)
            is_active = visibility >= visibility_threshold or (inside_support and visibility >= visibility_threshold * 0.5)
            if is_active:
                active_frames.append(int(frame_idx))

            frame_record = {
                "frame_index": int(frame_idx),
                "time_value": float(time_values[frame_idx]),
                "is_active": bool(is_active),
                "inside_support_window": bool(inside_support),
                "is_keyframe": int(frame_idx) in set(slot.get("keyframes", [])),
                "visibility": visibility,
                "support_score": float(support_score),
                "motion_label": motion_label,
                "motion_confidence": float(motion_confidence),
                "center_world": _float_list(center_world[frame_idx]),
                "extent_world_min": _float_list(extent_min[frame_idx]),
                "extent_world_max": _float_list(extent_max[frame_idx]),
                "speed": float(per_frame_speed[frame_idx]),
            }
            frames.append(frame_record)

            if is_active:
                frame_queries[frame_idx]["active_slots"].append(
                    {
                        "slot_id": int(slot["slot_id"]),
                        "entity_id": entity_id,
                        "entity_type": slot.get("entity_type", "unknown"),
                        "temporal_mode": slot.get("temporal_mode", slot.get("entity_type", "unknown")),
                        "quality": quality,
                        "visibility": visibility,
                        "support_score": float(support_score),
                        "motion_label": motion_label,
                        "primary_prompt": (slot.get("prompt_candidates") or ["scene entity"])[0],
                        "prompt_candidates": slot.get("prompt_candidates", []),
                        "temporal_prompt_candidates": slot.get("temporal_prompt_candidates", []),
                        "dynamic_prompt_candidates": slot.get("dynamic_prompt_candidates", []),
                        "static_prompt_candidates": slot.get("static_prompt_candidates", []),
                        "role_hints": slot.get("role_hints", []),
                        "center_world": frame_record["center_world"],
                        "extent_world_min": frame_record["extent_world_min"],
                        "extent_world_max": frame_record["extent_world_max"],
                        "is_keyframe": frame_record["is_keyframe"],
                        "inside_support_window": frame_record["inside_support_window"],
                        "support_frame_start": int(slot.get("support_frame_start", 0)),
                        "support_frame_end": int(slot.get("support_frame_end", 0)),
                    }
                )

        track = {
            "track_id": len(tracks),
            "slot_id": int(slot["slot_id"]),
            "entity_id": entity_id,
            "source_cluster_id": int(slot["source_cluster_id"]),
            "entity_type": slot.get("entity_type", "unknown"),
            "quality": quality,
            "visibility_ratio": float(slot.get("visibility_ratio", 0.0)),
            "motion_score": float(slot.get("motion_score") or 0.0),
            "path_length": float(slot.get("path_length") or 0.0),
            "temporal_mode": slot.get("temporal_mode", slot.get("entity_type", "unknown")),
            "occupancy_mean": float(slot.get("occupancy_mean", 0.0)),
            "visibility_mean": float(slot.get("visibility_mean", 0.0)),
            "support_span_ratio": float(slot.get("support_span_ratio", 0.0)),
            "support_frame_start": int(slot.get("support_frame_start", 0)),
            "support_frame_end": int(slot.get("support_frame_end", 0)),
            "dynamic_frame_ratio": float(slot.get("dynamic_frame_ratio", 0.0)),
            "keyframes": [int(v) for v in slot.get("keyframes", [])],
            "active_frames": active_frames,
            "num_active_frames": int(len(active_frames)),
            "prompt_candidates": slot.get("prompt_candidates", []),
            "temporal_prompt_candidates": slot.get("temporal_prompt_candidates", []),
            "role_hints": slot.get("role_hints", []),
            "segments": slot.get("segments", []),
            "track_mode": "entitybank_world_trajectory",
            "frames": frames,
        }
        tracks.append(track)

    for frame in frame_queries:
        frame["active_slots"].sort(
            key=lambda item: (-float(item["support_score"]), int(item["slot_id"]))
        )
        frame["num_active_slots"] = int(len(frame["active_slots"]))
        frame["num_dynamic_slots"] = int(
            sum(1 for item in frame["active_slots"] if item.get("motion_label") == "moving")
        )
        frame["num_static_slots"] = int(
            sum(1 for item in frame["active_slots"] if item.get("motion_label") == "stationary")
        )

    tracks_payload = {
        "schema_version": 1,
        "iteration": slots_payload.get("iteration"),
        "frame_count": slots_payload.get("frame_count"),
        "num_tracks": len(tracks),
        "visibility_threshold": float(visibility_threshold),
        "tracks": tracks,
    }
    frame_queries_payload = {
        "schema_version": 1,
        "iteration": slots_payload.get("iteration"),
        "frame_count": slots_payload.get("frame_count"),
        "num_tracks": len(tracks),
        "frames": frame_queries,
    }

    tracks_path = entitybank_dir / "semantic_tracks.json"
    frame_queries_path = entitybank_dir / "semantic_frame_queries.json"
    with open(tracks_path, "w", encoding="utf-8") as handle:
        json.dump(tracks_payload, handle, indent=2)
    with open(frame_queries_path, "w", encoding="utf-8") as handle:
        json.dump(frame_queries_payload, handle, indent=2)
    return tracks_path
