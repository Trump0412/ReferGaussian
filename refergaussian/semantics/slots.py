from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _base_prompts(entity: dict[str, Any]) -> list[str]:
    prompts = ["foreground region", "scene entity"]
    entity_type = entity.get("entity_type", "")
    support_span_ratio = float(entity.get("support_span_ratio", 1.0))
    if entity_type == "dynamic_object":
        prompts = ["moving object", "dynamic foreground", "trackable actor"] + prompts
    elif entity_type == "transient_event":
        prompts = ["short-lived interaction", "temporally localized event", "dynamic contact region"] + prompts
    elif entity_type == "temporally_localized_region":
        prompts = ["temporally localized region", "short-horizon scene entity"] + prompts
    elif entity_type == "static_region":
        prompts = ["static region", "background structure", "persistent scene part"] + prompts
    if support_span_ratio <= 0.35:
        prompts = ["temporally localized support"] + prompts
    elif support_span_ratio >= 0.75:
        prompts = ["persistent support through time"] + prompts
    return prompts


def _segment_queries(entity: dict[str, Any]) -> list[dict[str, Any]]:
    queries = []
    for segment in entity.get("segments", []):
        label = segment.get("label", "unknown")
        if label == "moving":
            text = "segment showing active motion over time"
        elif label == "stationary":
            text = "segment showing persistent static support"
        else:
            text = "entity segment"
        queries.append(
            {
                "segment_id": segment.get("segment_id"),
                "t0": segment.get("t0"),
                "t1": segment.get("t1"),
                "label": label,
                "text": text,
                "confidence": segment.get("confidence", 0.0),
            }
        )
    return queries


def _temporal_prompts(entity: dict[str, Any]) -> list[str]:
    mode = entity.get("temporal_mode", entity.get("entity_type", "unknown"))
    support_span_ratio = float(entity.get("support_span_ratio", 1.0))
    prompts = []
    if mode == "transient_event":
        prompts.extend(["short-lived event region", "temporally localized interaction"])
    elif mode == "dynamic_object":
        prompts.extend(["moving object with temporal support", "dynamic worldtube entity"])
    elif mode == "static_region":
        prompts.extend(["persistent static structure", "long-horizon scene support"])
    else:
        prompts.extend(["temporally localized region", "time-selective scene support"])
    if support_span_ratio <= 0.35:
        prompts.append("active over a short time window")
    elif support_span_ratio >= 0.75:
        prompts.append("visible through most of the sequence")
    return prompts


def _dynamic_prompts(entity: dict[str, Any]) -> list[str]:
    mode = entity.get("temporal_mode", entity.get("entity_type", "unknown"))
    if mode == "transient_event":
        return ["dynamic event region", "moving interaction cue"]
    return ["moving part", "dynamic semantic target"]


def _static_prompts(entity: dict[str, Any]) -> list[str]:
    mode = entity.get("temporal_mode", entity.get("entity_type", "unknown"))
    if mode == "static_region":
        return ["static structure", "persistent region"]
    return ["stationary phase of the entity", "temporarily static support"]


def export_semantic_slots(run_dir: str | Path, min_quality: float = 0.0) -> Path:
    run_dir = Path(run_dir)
    entitybank_dir = run_dir / "entitybank"
    entities_path = entitybank_dir / "entities.json"
    clusters_path = entitybank_dir / "cluster_stats.json"

    with open(entities_path, "r", encoding="utf-8") as handle:
        entities_payload = json.load(handle)
    with open(clusters_path, "r", encoding="utf-8") as handle:
        cluster_payload = json.load(handle)

    cluster_map = {
        cluster["cluster_id"]: cluster
        for cluster in cluster_payload.get("clusters", [])
    }

    slots = []
    slot_queries = []
    for entity in entities_payload.get("entities", []):
        quality = float(entity.get("quality", 0.0))
        if quality < min_quality:
            continue

        cluster_id = entity.get("source_cluster_id")
        cluster = cluster_map.get(cluster_id, {})
        prompts = _base_prompts(entity)
        temporal_prompts = _temporal_prompts(entity)
        role_hints = entity.get("role_hints") or []

        slot = {
            "slot_id": len(slots),
            "entity_id": entity.get("id"),
            "source_cluster_id": cluster_id,
            "entity_type": entity.get("entity_type", "unknown"),
            "temporal_mode": entity.get("temporal_mode", entity.get("entity_type", "unknown")),
            "quality": quality,
            "visibility_ratio": float(entity.get("visibility_ratio", 0.0)),
            "keyframes": entity.get("keyframes", []),
            "segments": entity.get("segments", []),
            "prompt_candidates": prompts,
            "temporal_prompt_candidates": temporal_prompts,
            "dynamic_prompt_candidates": _dynamic_prompts(entity),
            "static_prompt_candidates": _static_prompts(entity),
            "role_hints": role_hints,
            "motion_score": cluster.get("motion_score"),
            "path_length": cluster.get("path_length"),
            "anchor_mean": cluster.get("anchor_mean"),
            "scale_mean": cluster.get("scale_mean"),
            "occupancy_mean": float(entity.get("occupancy_mean", cluster.get("occupancy_mean", 0.0))),
            "visibility_mean": float(entity.get("visibility_mean", cluster.get("visibility_mean", 0.0))),
            "tube_ratio_mean": float(entity.get("tube_ratio_mean", cluster.get("tube_ratio_mean", 0.0))),
            "support_factor_mean": float(entity.get("support_factor_mean", cluster.get("support_factor_mean", 0.0))),
            "effective_support_mean": float(entity.get("effective_support_mean", cluster.get("effective_support_mean", 0.0))),
            "support_frame_start": int(entity.get("support_frame_start", cluster.get("support_frame_start", 0) or 0)),
            "support_frame_end": int(entity.get("support_frame_end", cluster.get("support_frame_end", 0) or 0)),
            "support_frame_peak": int(entity.get("support_frame_peak", cluster.get("support_frame_peak", 0) or 0)),
            "support_span_ratio": float(entity.get("support_span_ratio", cluster.get("support_span_ratio", 0.0) or 0.0)),
            "support_peak_value": float(entity.get("support_peak_value", 0.0)),
            "dynamic_frame_ratio": float(entity.get("dynamic_frame_ratio", cluster.get("dynamic_frame_ratio", 0.0) or 0.0)),
            "stationary_frame_ratio": float(entity.get("stationary_frame_ratio", cluster.get("stationary_frame_ratio", 0.0) or 0.0)),
            "mask_refine_source": entity.get("mask_refine_source"),
            "bbox_image_pt_key": entity.get("bbox_image_pt_key"),
        }
        slots.append(slot)

        slot_queries.append(
            {
                "slot_id": slot["slot_id"],
                "entity_id": slot["entity_id"],
                "global_queries": prompts,
                "temporal_queries": temporal_prompts,
                "dynamic_queries": slot["dynamic_prompt_candidates"],
                "static_queries": slot["static_prompt_candidates"],
                "segment_queries": _segment_queries(entity),
                "tracking_keyframes": slot["keyframes"],
            }
        )

    payload = {
        "schema_version": 1,
        "iteration": entities_payload.get("iteration"),
        "frame_count": entities_payload.get("frame_count"),
        "num_slots": len(slots),
        "query_mode": "entitybank_bootstrap",
        "slots": slots,
    }
    queries_payload = {
        "schema_version": 1,
        "iteration": entities_payload.get("iteration"),
        "num_slots": len(slots),
        "slots": slot_queries,
    }

    slots_path = entitybank_dir / "semantic_slots.json"
    queries_path = entitybank_dir / "semantic_slot_queries.json"
    with open(slots_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(queries_path, "w", encoding="utf-8") as handle:
        json.dump(queries_payload, handle, indent=2)
    return slots_path
