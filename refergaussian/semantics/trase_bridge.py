from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _nearest_resample(query_times: np.ndarray, sample_times: np.ndarray, values: np.ndarray) -> np.ndarray:
    indices = np.abs(query_times[:, None] - sample_times[None, :]).argmin(axis=1)
    return values[indices]


def _target_tracks(run_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    entities_payload = _read_json(run_dir / "entitybank" / "entities.json")
    trajectory_payload = np.load(run_dir / "entitybank" / "trajectory_samples.npz")
    trajectories = trajectory_payload["trajectories"]
    gate = trajectory_payload["gate"].squeeze(-1)
    time_values = trajectory_payload["time_values"].astype(np.float32)

    tracks = []
    for entity in entities_payload.get("entities", []):
        gaussian_ids = np.asarray(entity.get("gaussian_ids", []), dtype=np.int64)
        if gaussian_ids.size == 0:
            continue
        center_world = trajectories[gaussian_ids].mean(axis=0)
        visibility = gate[gaussian_ids].mean(axis=0)
        support_mask = visibility >= 0.35
        path_length = float(np.linalg.norm(np.diff(center_world, axis=0), axis=1).sum()) if center_world.shape[0] > 1 else 0.0
        tracks.append(
            {
                "entity_id": int(entity["id"]),
                "entity_type": entity.get("entity_type", "dynamic_object"),
                "quality": float(entity.get("quality", 0.0)),
                "gaussian_count": int(gaussian_ids.size),
                "center_world": center_world.astype(np.float32),
                "visibility": visibility.astype(np.float32),
                "support_mask": support_mask.astype(bool),
                "path_length": path_length,
            }
        )
    return tracks, time_values


def _source_tracks(source_entitybank_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    entities_payload = _read_json(source_entitybank_dir / "entities.json")
    entities_pt = torch.load(source_entitybank_dir / "entities.pt", map_location="cpu")
    sem_static = np.load(source_entitybank_dir / "sem_static.npz", allow_pickle=True)
    sem_dynamic = np.load(source_entitybank_dir / "sem_dynamic.npz", allow_pickle=True)

    static_text = {
        int(entity_id): str(text)
        for entity_id, text in zip(sem_static["entity_ids"].tolist(), sem_static["static_text"].tolist())
    }
    global_desc = {
        int(entity_id): str(text)
        for entity_id, text in zip(sem_dynamic["entity_ids"].tolist(), sem_dynamic["global_desc"].tolist())
    }
    dynamic_segments: dict[int, list[dict[str, Any]]] = {}
    for entity_id, seg_range, seg_desc, seg_structured in zip(
        sem_dynamic["segment_entity_ids"].tolist(),
        sem_dynamic["segment_ranges"].tolist(),
        sem_dynamic["seg_desc"].tolist(),
        sem_dynamic["seg_structured_json"].tolist(),
    ):
        entity_id = int(entity_id)
        dynamic_segments.setdefault(entity_id, []).append(
            {
                "frame_range": [int(seg_range[0]), int(seg_range[1])],
                "caption": str(seg_desc),
                "structured": json.loads(str(seg_structured)),
            }
        )

    centroid_world = entities_pt["centroid_world"].numpy().astype(np.float32)
    centroid_world_valid = entities_pt["centroid_world_valid"].numpy().astype(bool)
    visibility = entities_pt["visibility"].numpy().astype(bool)
    quality = entities_pt["quality"].numpy().astype(np.float32)
    time_values = entities_pt["time_values"].numpy().astype(np.float32)

    entity_map = {int(entity["id"]): entity for entity in entities_payload.get("entities", [])}
    tracks = []
    for entity_id in range(centroid_world.shape[0]):
        entity = entity_map.get(entity_id)
        if entity is None:
            continue
        valid = centroid_world_valid[entity_id] & visibility[entity_id]
        center = centroid_world[entity_id]
        if valid.any():
            diffs = np.diff(center[valid], axis=0)
            path_length = float(np.linalg.norm(diffs, axis=1).sum()) if diffs.size else 0.0
        else:
            path_length = 0.0
        tracks.append(
            {
                "entity_id": entity_id,
                "entity_type": entity.get("entity_type", "object"),
                "quality": float(quality[entity_id]),
                "role_hints": entity.get("role_hints", []),
                "visibility_ratio": float(entity.get("visibility_ratio", 0.0)),
                "center_world": center,
                "valid": valid,
                "path_length": path_length,
                "static_text": static_text.get(entity_id, entity.get("static_text", "")),
                "global_desc": global_desc.get(entity_id, entity.get("global_desc", "")),
                "dyn_desc": dynamic_segments.get(entity_id, []),
            }
        )
    return tracks, time_values


def _match_score(source: dict[str, Any], target: dict[str, Any], source_times: np.ndarray, target_times: np.ndarray, scene_scale: float) -> float:
    target_center = _nearest_resample(source_times, target_times, target["center_world"])
    target_visibility = _nearest_resample(source_times, target_times, target["visibility"])
    target_support = _nearest_resample(source_times, target_times, target["support_mask"].astype(np.float32)) > 0.5
    valid = np.asarray(source["valid"], dtype=bool) & (target_visibility > 0.2)
    if int(valid.sum()) < 4:
        return -1.0

    distance = np.linalg.norm(source["center_world"][valid] - target_center[valid], axis=1)
    median_distance = float(np.median(distance))
    dist_score = np.exp(-median_distance / max(scene_scale, 1.0e-3))

    source_path = float(source["path_length"])
    target_path = float(target["path_length"])
    path_ratio = min(source_path, target_path) / max(source_path, target_path, 1.0e-6)
    visibility_score = float(valid.mean())
    overlap_union = np.asarray(source["valid"], dtype=bool) | target_support
    support_overlap = float(((np.asarray(source["valid"], dtype=bool) & target_support).sum()) / max(overlap_union.sum(), 1))
    quality_score = 0.5 * (float(source["quality"]) + float(target["quality"]))

    return 0.45 * dist_score + 0.20 * path_ratio + 0.15 * support_overlap + 0.10 * visibility_score + 0.10 * quality_score


def transfer_trase_semantics(
    target_run_dir: str | Path,
    source_model_dir: str | Path,
    query_name: str | None = None,
    min_match_score: float = 0.35,
) -> Path:
    target_run_dir = Path(target_run_dir)
    source_model_dir = Path(source_model_dir)
    source_entitybank_dir = source_model_dir / "entitybank"

    target_tracks, target_times = _target_tracks(target_run_dir)
    source_tracks, source_times = _source_tracks(source_entitybank_dir)
    scene_scale = float(np.std(np.concatenate([track["center_world"][track["valid"]] for track in source_tracks if track["valid"].any()], axis=0)))
    if scene_scale <= 0:
        scene_scale = 1.0

    match_candidates = []
    for source in source_tracks:
        for target in target_tracks:
            score = _match_score(source, target, source_times, target_times, scene_scale)
            if score < min_match_score:
                continue
            match_candidates.append(
                (
                    score,
                    int(source["entity_id"]),
                    int(target["entity_id"]),
                )
            )
    match_candidates.sort(reverse=True)

    used_source = set()
    used_target = set()
    assignments = []
    source_to_target: dict[int, int] = {}
    for score, source_id, target_id in match_candidates:
        if source_id in used_source or target_id in used_target:
            continue
        used_source.add(source_id)
        used_target.add(target_id)
        source_to_target[source_id] = target_id
        source_track = next(track for track in source_tracks if track["entity_id"] == source_id)
        target_track = next(track for track in target_tracks if track["entity_id"] == target_id)
        assignments.append(
            {
                "source_entity_id": source_id,
                "target_entity_id": target_id,
                "score": float(score),
                "source_entity_type": source_track["entity_type"],
                "target_entity_type": target_track["entity_type"],
                "role_hints": source_track["role_hints"],
                "static_text": source_track["static_text"],
                "global_desc": source_track["global_desc"],
                "dyn_desc": source_track["dyn_desc"],
            }
        )

    output_dir = target_run_dir / "entitybank" / "trase_bridge"
    output_dir.mkdir(parents=True, exist_ok=True)
    transfer_payload = {
        "schema_version": 1,
        "source_model_dir": str(source_model_dir),
        "target_run_dir": str(target_run_dir),
        "min_match_score": float(min_match_score),
        "num_assignments": len(assignments),
        "assignments": assignments,
    }
    _write_json(output_dir / "semantic_transfer.json", transfer_payload)

    entities_payload = _read_json(target_run_dir / "entitybank" / "entities.json")
    enriched_entities = []
    for entity in entities_payload.get("entities", []):
        entity_copy = dict(entity)
        source_id = None
        transfer_score = None
        transfer = next((item for item in assignments if item["target_entity_id"] == int(entity["id"])), None)
        if transfer is not None:
            source_id = int(transfer["source_entity_id"])
            transfer_score = float(transfer["score"])
            entity_copy["static_text"] = transfer["static_text"]
            entity_copy["global_desc"] = transfer["global_desc"]
            entity_copy["dyn_desc"] = transfer["dyn_desc"]
            entity_copy["entity_type"] = transfer["source_entity_type"]
            entity_copy["role_hints"] = transfer["role_hints"]
            entity_copy["semantic_source"] = "trasepp_qwen_transfer"
        entity_copy["trase_source_entity_id"] = source_id
        entity_copy["trase_transfer_score"] = transfer_score
        enriched_entities.append(entity_copy)
    _write_json(
        output_dir / "entities_semantic_enriched.json",
        {
            **entities_payload,
            "entities": enriched_entities,
            "semantic_source": "trasepp_qwen_transfer",
        },
    )

    if query_name is not None:
        query_dir = source_entitybank_dir / "queries" / query_name
        selected_payload = _read_json(query_dir / "selected.json")
        transferred = []
        for item in selected_payload.get("selected", []):
            source_id = int(item["id"])
            target_id = source_to_target.get(source_id)
            if target_id is None:
                continue
            transferred.append(
                {
                    "id": target_id,
                    "source_entity_id": source_id,
                    "role": item.get("role", "patient"),
                    "entity_type": item.get("entity_type", "object"),
                    "confidence": float(item.get("confidence", 0.0)),
                    "segments": item.get("segments", []),
                }
            )

        query_output_dir = output_dir / "queries" / query_name
        query_output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            query_output_dir / "selected_transferred.json",
            {
                "selected": transferred,
                "empty": len(transferred) == 0,
                "reason": "" if transferred else "no matched target entities",
                "query_slots": selected_payload.get("query_slots", {}),
                "source_query_dir": str(query_dir),
            },
        )
        for name in ("query.json", "candidates.json", "selected.json", "video_binary.mp4", "video_instance.mp4", "video_overlay.mp4"):
            source_path = query_dir / name
            if source_path.exists():
                shutil.copy2(source_path, query_output_dir / name)

        validation = validate_query_video(query_output_dir / "video_binary.mp4", selected_payload.get("selected", []))
        _write_json(query_output_dir / "validation.json", validation)

    return output_dir


def validate_query_video(binary_video_path: str | Path, selected_items: list[dict[str, Any]]) -> dict[str, Any]:
    binary_video_path = Path(binary_video_path)
    payload = {
        "binary_video_path": str(binary_video_path),
        "video_exists": binary_video_path.exists(),
        "selected_roles": [item.get("role") for item in selected_items],
        "selected_entity_ids": [int(item.get("id")) for item in selected_items],
    }
    if not binary_video_path.exists():
        return payload

    frames = np.asarray(iio.imread(binary_video_path))
    if frames.ndim == 3:
        frames = frames[..., None]
    active_pixels = (frames[..., 0] > 16).sum(axis=(1, 2))
    active_frames = np.where(active_pixels > 0)[0]
    payload.update(
        {
            "frame_count": int(frames.shape[0]),
            "height": int(frames.shape[1]),
            "width": int(frames.shape[2]),
            "active_frame_count": int(active_frames.size),
            "first_active_frame": int(active_frames[0]) if active_frames.size else None,
            "last_active_frame": int(active_frames[-1]) if active_frames.size else None,
            "peak_active_pixels": int(active_pixels.max()) if active_pixels.size else 0,
            "mean_active_pixels": float(active_pixels.mean()) if active_pixels.size else 0.0,
        }
    )
    return payload
