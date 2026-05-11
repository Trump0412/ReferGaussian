from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .priors import export_semantic_priors
from .tracks import export_semantic_tracks

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_ROOT = _PROJECT_ROOT / "external" / "4DGaussians"
for _candidate in (_PROJECT_ROOT, _UPSTREAM_ROOT):
    candidate_str = str(_candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from refergaussian.entitybank.tube_bank import load_gaussian_state
from utils.sh_utils import SH2RGB


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _safe_ratio(value: float, denominator: float) -> float:
    return float(value / max(denominator, 1.0e-6))


def _normalize_metric(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo <= 1.0e-6:
        return np.full_like(values, 0.5, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def _rank_map(records: list[dict[str, Any]], key: str) -> dict[int, float]:
    values = np.asarray([float(record[key]) for record in records], dtype=np.float32)
    normalized = _normalize_metric(values)
    return {int(record["entity_id"]): float(normalized[index]) for index, record in enumerate(records)}


def _round_list(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=np.float32).tolist()]


def _phase_segments(track_frames: list[dict[str, Any]], label: str) -> list[list[int]]:
    ranges: list[list[int]] = []
    start = None
    prev = None
    for frame in track_frames:
        frame_idx = int(frame.get("frame_index", 0))
        matches = bool(frame.get("is_active")) and str(frame.get("motion_label")) == label
        if matches and start is None:
            start = frame_idx
            prev = frame_idx
            continue
        if matches:
            prev = frame_idx
            continue
        if start is not None:
            ranges.append([int(start), int(prev)])
            start = None
            prev = None
    if start is not None and prev is not None:
        ranges.append([int(start), int(prev)])
    return ranges


def _color_profile(mean_rgb: np.ndarray) -> dict[str, Any]:
    rgb = np.clip(np.asarray(mean_rgb, dtype=np.float32), 0.0, 1.0)
    brightness = float(rgb.mean())
    spread = float(rgb.max() - rgb.min())
    dominant_index = int(np.argmax(rgb))
    dominant_map = {0: "red", 1: "green", 2: "blue"}
    tags: list[str] = []
    if brightness >= 0.82:
        tags.append("bright")
    elif brightness <= 0.30:
        tags.append("dark")
    else:
        tags.append("midtone")
    if spread <= 0.08:
        tags.append("neutral")
        if brightness >= 0.75:
            tags.append("white_like")
        elif brightness <= 0.45:
            tags.append("gray_like")
    else:
        tags.append(f"{dominant_map[dominant_index]}_dominant")
        if rgb[0] > 0.70 and rgb[1] > 0.70 and rgb[2] < 0.55:
            tags.extend(["yellow_like", "fruit_like", "citrus_like"])
        if rgb[0] > 0.55 and rgb[1] < 0.45 and rgb[2] < 0.45:
            tags.append("warm_object")
        if rgb[0] > 0.65 and rgb[1] > 0.45 and rgb[2] < 0.45:
            tags.append("orange_like")
        if rgb[2] > 0.55 and rgb[0] < 0.45:
            tags.append("cool_object")
    return {
        "mean_rgb": _round_list(rgb),
        "brightness": brightness,
        "color_spread": spread,
        "dominant_channel": dominant_map[dominant_index],
        "tags": sorted(set(tags)),
    }


def _shape_profile(mean_scale: np.ndarray) -> dict[str, Any]:
    scale = np.asarray(mean_scale, dtype=np.float32)
    ordered = np.sort(scale)
    elongation = _safe_ratio(float(ordered[-1]), float(ordered[0]))
    compactness = 1.0 / max(elongation, 1.0)
    tags = []
    if elongation >= 1.35:
        tags.append("elongated")
    elif elongation <= 1.10:
        tags.append("compact")
    else:
        tags.append("mildly_elongated")
    if float(scale.mean()) >= 0.11:
        tags.append("large_scale")
    elif float(scale.mean()) <= 0.05:
        tags.append("small_scale")
    else:
        tags.append("mid_scale")
    return {
        "mean_scale": _round_list(scale),
        "scale_mean": float(scale.mean()),
        "scale_min": float(ordered[0]),
        "scale_max": float(ordered[-1]),
        "elongation": elongation,
        "compactness": compactness,
        "tags": tags,
    }


def _semantic_terms(prior: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for value in prior.get("role_hints", []):
        terms.add(str(value).lower())
    for group_name in ("static_semantics", "dynamic_semantics", "interaction_semantics"):
        for prompt in prior.get(group_name, {}).get("prompt_candidates", []):
            for token in str(prompt).lower().replace("-", " ").split():
                if token.isalpha():
                    terms.add(token)
    for prompt in prior.get("query_pack", {}).get("global", []):
        for token in str(prompt).lower().replace("-", " ").split():
            if token.isalpha():
                terms.add(token)
    semantic_head = str(prior.get("semantic_head", "unknown")).lower()
    temporal_mode = str(prior.get("temporal_mode", "unknown")).lower()
    entity_type = str(prior.get("entity_type", "unknown")).lower()
    terms.update({semantic_head, temporal_mode, entity_type})
    if semantic_head == "interaction":
        terms.update({"interaction", "contact", "manipulation"})
    elif semantic_head == "dynamic":
        terms.update({"dynamic", "moving", "motion"})
    else:
        terms.update({"static", "persistent", "support"})
    return {term for term in terms if term}


def _role_scores(
    semantic_head: str,
    dynamic_rank: float,
    occupancy_rank: float,
    scale_rank: float,
    elongation_rank: float,
    visibility_rank: float,
    compactness: float,
    color_tags: set[str],
    support_span_ratio: float,
) -> dict[str, float]:
    interaction_bonus = 1.0 if semantic_head == "interaction" else 0.0
    dynamic_bonus = 1.0 if semantic_head == "dynamic" else 0.0
    static_bonus = 1.0 if semantic_head == "static" else 0.0
    short_support = float(np.clip((0.55 - support_span_ratio) / 0.55, 0.0, 1.0))
    long_support = float(np.clip((support_span_ratio - 0.55) / 0.45, 0.0, 1.0))

    tool_score = (
        0.28 * dynamic_rank
        + 0.18 * elongation_rank
        + 0.12 * (1.0 - scale_rank)
        + 0.12 * (1.0 - occupancy_rank)
        + 0.16 * interaction_bonus
        + 0.14 * short_support
    )
    patient_score = (
        0.24 * interaction_bonus
        + 0.18 * occupancy_rank
        + 0.15 * compactness
        + 0.16 * (1.0 - static_bonus * 0.25)
        + 0.12 * (1.0 - elongation_rank * 0.5)
        + 0.15 * visibility_rank
    )
    if "yellow_like" in color_tags or "citrus_like" in color_tags or "fruit_like" in color_tags:
        patient_score += 0.15
    if "neutral" in color_tags or "white_like" in color_tags:
        tool_score += 0.05

    support_score = (
        0.30 * static_bonus
        + 0.20 * long_support
        + 0.22 * scale_rank
        + 0.16 * visibility_rank
        + 0.12 * (1.0 - dynamic_rank)
    )
    agent_score = (
        0.20 * dynamic_rank
        + 0.20 * occupancy_rank
        + 0.20 * scale_rank
        + 0.20 * visibility_rank
        + 0.10 * long_support
        + 0.10 * dynamic_bonus
    )
    background_score = (
        0.35 * support_score
        + 0.20 * long_support
        + 0.20 * (1.0 - dynamic_rank)
        + 0.25 * visibility_rank
    )
    role_scores = {
        "patient": float(np.clip(patient_score, 0.0, 1.0)),
        "tool": float(np.clip(tool_score, 0.0, 1.0)),
        "agent": float(np.clip(agent_score, 0.0, 1.0)),
        "support": float(np.clip(support_score, 0.0, 1.0)),
        "background": float(np.clip(background_score, 0.0, 1.0)),
    }
    return role_scores


def _native_texts(
    entity_id: int,
    semantic_head: str,
    temporal_mode: str,
    color_profile: dict[str, Any],
    shape_profile: dict[str, Any],
    role_scores: dict[str, float],
    prior: dict[str, Any],
) -> dict[str, Any]:
    dominant_role = max(role_scores.items(), key=lambda item: item[1])[0]
    color_words = " ".join(color_profile["tags"][:2]) if color_profile["tags"] else "neutral"
    shape_words = " ".join(shape_profile["tags"][:2]) if shape_profile["tags"] else "compact"
    support_window = prior.get("support_window", {})
    frame_start = int(support_window.get("frame_start", 0))
    frame_end = int(support_window.get("frame_end", frame_start))
    global_desc = (
        f"ReferGaussian entity {entity_id}: {color_words} {shape_words} "
        f"{semantic_head} worldtube with {temporal_mode} behavior over frames {frame_start}-{frame_end}."
    )
    static_text = (
        f"{color_words} {shape_words} entity with dominant {dominant_role} role."
    )
    dynamic_desc = [
        f"moving phase over frames {item['frame_start']}-{item['frame_end']}"
        for item in prior.get("dynamic_semantics", {}).get("frame_ranges", [])[:3]
    ]
    interaction_desc = [
        f"interaction phase over frames {item['frame_start']}-{item['frame_end']}"
        for item in prior.get("interaction_semantics", {}).get("frame_ranges", [])[:3]
    ]
    return {
        "static_text": static_text,
        "global_desc": global_desc,
        "dynamic_desc": dynamic_desc,
        "interaction_desc": interaction_desc,
    }


def _interaction_segments(mask: np.ndarray) -> list[list[int]]:
    indices = np.where(mask)[0]
    if indices.size == 0:
        return []
    segments: list[list[int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        segments.append([start, prev])
        start = value
        prev = value
    segments.append([start, prev])
    return segments


def _localized_event_mask(overlap: np.ndarray, distance: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    overlap = np.asarray(overlap, dtype=bool)
    if not overlap.any():
        return np.zeros_like(overlap, dtype=bool), {
            "distance_min": 0.0,
            "distance_median": 0.0,
            "event_fraction": 0.0,
            "contrast": 0.0,
        }
    overlap_indices = np.where(overlap)[0]
    overlap_distance = distance[overlap]
    overlap_threshold = threshold[overlap]
    closest_value = float(overlap_distance.min())
    median_value = float(np.median(overlap_distance))
    contrast = max(0.0, (median_value - closest_value) / max(median_value, 1.0e-6))
    event_fraction = float(np.clip(0.05 + 0.30 * contrast, 0.05, 0.18))
    quantile_value = float(np.quantile(overlap_distance, event_fraction))
    threshold_value = float(np.median(overlap_threshold) * (0.72 + 0.20 * contrast))
    local_threshold = max(closest_value + 1.0e-4, min(quantile_value, threshold_value))
    event_mask = overlap & (distance <= local_threshold)
    if event_mask.any():
        return event_mask, {
            "distance_min": closest_value,
            "distance_median": median_value,
            "event_fraction": event_fraction,
            "contrast": contrast,
        }
    closest_idx = int(overlap_indices[int(np.argmin(overlap_distance))])
    event_radius = 1 if contrast < 0.02 else 2
    event_mask[max(0, closest_idx - event_radius) : min(overlap.shape[0], closest_idx + event_radius + 1)] = True
    return event_mask, {
        "distance_min": closest_value,
        "distance_median": median_value,
        "event_fraction": event_fraction,
        "contrast": contrast,
    }


def export_native_semantic_assignments(
    run_dir: str | Path,
    min_quality: float = 0.0,
    max_partners: int = 8,
) -> Path:
    run_dir = Path(run_dir)
    entitybank_dir = run_dir / "entitybank"
    priors_path = entitybank_dir / "semantic_priors.json"
    tracks_path = entitybank_dir / "semantic_tracks.json"
    entities_path = entitybank_dir / "entities.json"
    if not priors_path.exists():
        export_semantic_priors(run_dir, min_quality=min_quality)
    if not tracks_path.exists():
        export_semantic_tracks(run_dir, min_quality=min_quality)

    priors_payload = _read_json(priors_path)
    tracks_payload = _read_json(tracks_path)
    entities_payload = _read_json(entities_path)
    state, _config, _iteration = load_gaussian_state(run_dir)

    entity_map = {int(entity["id"]): entity for entity in entities_payload.get("entities", [])}
    prior_map = {int(prior["entity_id"]): prior for prior in priors_payload.get("priors", [])}
    track_map = {int(track["entity_id"]): track for track in tracks_payload.get("tracks", [])}

    metric_rows: list[dict[str, Any]] = []
    for entity_id, entity in entity_map.items():
        if float(entity.get("quality", 0.0)) < min_quality:
            continue
        gaussian_ids = np.asarray(entity.get("gaussian_ids", []), dtype=np.int64)
        if gaussian_ids.size == 0:
            continue
        rgb = np.clip(SH2RGB(state.rgb[gaussian_ids]), 0.0, 1.0)
        mean_rgb = rgb.mean(axis=0)
        mean_scale = state.spatial_scale[gaussian_ids].mean(axis=0)
        velocity_norm = np.linalg.norm(state.velocity[gaussian_ids], axis=1)
        acceleration_norm = np.linalg.norm(state.acceleration[gaussian_ids], axis=1)
        metric_rows.append(
            {
                "entity_id": entity_id,
                "quality": float(entity.get("quality", 0.0)),
                "occupancy_mean": float(entity.get("occupancy_mean", 0.0)),
                "visibility_mean": float(entity.get("visibility_mean", 0.0)),
                "tube_ratio_mean": float(entity.get("tube_ratio_mean", 0.0)),
                "scale_mean": float(mean_scale.mean()),
                "elongation": float(np.sort(mean_scale)[-1] / max(np.sort(mean_scale)[0], 1.0e-6)),
                "velocity_mean": float(velocity_norm.mean()),
                "acceleration_mean": float(acceleration_norm.mean()),
                "mean_rgb": mean_rgb,
                "mean_scale": mean_scale,
            }
        )

    dynamic_rank = _rank_map(metric_rows, "velocity_mean")
    occupancy_rank = _rank_map(metric_rows, "occupancy_mean")
    scale_rank = _rank_map(metric_rows, "scale_mean")
    elongation_rank = _rank_map(metric_rows, "elongation")
    visibility_rank = _rank_map(metric_rows, "visibility_mean")

    assignments: list[dict[str, Any]] = []
    for row in metric_rows:
        entity_id = int(row["entity_id"])
        entity = entity_map[entity_id]
        prior = prior_map.get(entity_id)
        track = track_map.get(entity_id)
        if prior is None or track is None:
            continue
        color_profile = _color_profile(np.asarray(row["mean_rgb"], dtype=np.float32))
        shape_profile = _shape_profile(np.asarray(row["mean_scale"], dtype=np.float32))
        semantic_head = str(prior.get("semantic_head", "dynamic"))
        compactness = float(shape_profile["compactness"])
        role_scores = _role_scores(
            semantic_head=semantic_head,
            dynamic_rank=float(dynamic_rank.get(entity_id, 0.5)),
            occupancy_rank=float(occupancy_rank.get(entity_id, 0.5)),
            scale_rank=float(scale_rank.get(entity_id, 0.5)),
            elongation_rank=float(elongation_rank.get(entity_id, 0.5)),
            visibility_rank=float(visibility_rank.get(entity_id, 0.5)),
            compactness=compactness,
            color_tags=set(color_profile["tags"]),
            support_span_ratio=float(entity.get("support_span_ratio", 1.0)),
        )
        concept_tags = set(_semantic_terms(prior))
        concept_tags.update(color_profile["tags"])
        concept_tags.update(shape_profile["tags"])
        strong_roles = [role_name for role_name, score in role_scores.items() if score >= 0.55]
        for role_name, score in role_scores.items():
            if score >= 0.58:
                concept_tags.add(f"{role_name}_like")
        if role_scores["patient"] >= 0.62:
            concept_tags.update({"object", "patient", "target"})
        if role_scores["tool"] >= 0.60:
            concept_tags.update({"tool", "implement"})
            if shape_profile["elongation"] >= 1.08 or "neutral" in color_profile["tags"]:
                concept_tags.update({"knife_like", "blade_like"})
        if role_scores["support"] >= 0.60:
            concept_tags.update({"support_surface", "board_like"})
        if role_scores["agent"] >= 0.60:
            concept_tags.update({"agent", "hand_like"})
        native_texts = _native_texts(
            entity_id=entity_id,
            semantic_head=semantic_head,
            temporal_mode=str(prior.get("temporal_mode", entity.get("temporal_mode", "unknown"))),
            color_profile=color_profile,
            shape_profile=shape_profile,
            role_scores=role_scores,
            prior=prior,
        )
        assignment = {
            "assignment_id": len(assignments),
            "entity_id": entity_id,
            "slot_id": int(prior.get("slot_id", -1)),
            "prior_id": int(prior.get("prior_id", -1)),
            "source_cluster_id": int(prior.get("source_cluster_id", entity.get("source_cluster_id", -1))),
            "semantic_head": semantic_head,
            "entity_type": str(prior.get("entity_type", entity.get("entity_type", "unknown"))),
            "temporal_mode": str(prior.get("temporal_mode", entity.get("temporal_mode", "unknown"))),
            "quality": float(entity.get("quality", 0.0)),
            "support_window": prior.get("support_window", {}),
            "role_hints": sorted(set(entity.get("role_hints", []) + strong_roles)),
            "role_scores": role_scores,
            "geometry_profile": {
                "occupancy_mean": float(entity.get("occupancy_mean", 0.0)),
                "visibility_mean": float(entity.get("visibility_mean", 0.0)),
                "tube_ratio_mean": float(entity.get("tube_ratio_mean", 0.0)),
                "velocity_rank": float(dynamic_rank.get(entity_id, 0.5)),
                "occupancy_rank": float(occupancy_rank.get(entity_id, 0.5)),
                "scale_rank": float(scale_rank.get(entity_id, 0.5)),
                "elongation_rank": float(elongation_rank.get(entity_id, 0.5)),
                "visibility_rank": float(visibility_rank.get(entity_id, 0.5)),
                "shape": shape_profile,
            },
            "color_profile": color_profile,
            "concept_tags": sorted(concept_tags),
            "semantic_terms": sorted(_semantic_terms(prior)),
            "phase_segments": {
                "moving": _phase_segments(track.get("frames", []), "moving"),
                "stationary": _phase_segments(track.get("frames", []), "stationary"),
            },
            "prompt_groups": {
                "global": prior.get("query_pack", {}).get("global", []),
                "static": prior.get("static_semantics", {}).get("prompt_candidates", []),
                "dynamic": prior.get("dynamic_semantics", {}).get("prompt_candidates", []),
                "interaction": prior.get("interaction_semantics", {}).get("prompt_candidates", []),
            },
            "native_text": native_texts,
            "interaction_partners": [],
        }
        assignments.append(assignment)

    assignment_map = {int(item["entity_id"]): item for item in assignments}
    pair_interactions: list[dict[str, Any]] = []
    for index_a, item_a in enumerate(assignments):
        track_a = track_map[item_a["entity_id"]]
        frames_a = track_a.get("frames", [])
        centers_a = np.asarray([frame["center_world"] for frame in frames_a], dtype=np.float32)
        mins_a = np.asarray([frame["extent_world_min"] for frame in frames_a], dtype=np.float32)
        maxs_a = np.asarray([frame["extent_world_max"] for frame in frames_a], dtype=np.float32)
        active_a = np.asarray([bool(frame.get("is_active")) for frame in frames_a], dtype=bool)
        support_a = np.asarray([float(frame.get("support_score", 0.0)) for frame in frames_a], dtype=np.float32)
        radius_a = 0.5 * np.linalg.norm(maxs_a - mins_a, axis=1)
        for item_b in assignments[index_a + 1 :]:
            track_b = track_map[item_b["entity_id"]]
            frames_b = track_b.get("frames", [])
            if len(frames_a) != len(frames_b):
                continue
            centers_b = np.asarray([frame["center_world"] for frame in frames_b], dtype=np.float32)
            mins_b = np.asarray([frame["extent_world_min"] for frame in frames_b], dtype=np.float32)
            maxs_b = np.asarray([frame["extent_world_max"] for frame in frames_b], dtype=np.float32)
            active_b = np.asarray([bool(frame.get("is_active")) for frame in frames_b], dtype=bool)
            support_b = np.asarray([float(frame.get("support_score", 0.0)) for frame in frames_b], dtype=np.float32)
            radius_b = 0.5 * np.linalg.norm(maxs_b - mins_b, axis=1)

            overlap = active_a & active_b
            if int(overlap.sum()) < 6:
                continue
            distance = np.linalg.norm(centers_a - centers_b, axis=1)
            threshold = 0.60 * (radius_a + radius_b) + 0.06
            proximity_mask = overlap & (distance <= threshold)
            event_mask, event_stats = _localized_event_mask(overlap, distance, threshold)
            support_overlap = np.minimum(support_a, support_b)
            interaction_score = float(
                (np.exp(-distance[event_mask] / np.clip(threshold[event_mask], 1.0e-4, None)) * support_overlap[event_mask]).mean()
            )
            event_span_ratio = float(event_mask.mean())
            pair_payload = {
                "entity_a": int(item_a["entity_id"]),
                "entity_b": int(item_b["entity_id"]),
                "interaction_score": interaction_score,
                "overlap_ratio": float(overlap.mean()),
                "proximity_ratio": float(proximity_mask.mean()),
                "contact_ratio": event_span_ratio,
                "contact_segments": _interaction_segments(event_mask),
                "event_contrast": float(event_stats["contrast"]),
                "event_fraction": float(event_stats["event_fraction"]),
                "distance_min": float(event_stats["distance_min"]),
                "distance_median": float(event_stats["distance_median"]),
                "sample_frame_count": int(overlap.sum()),
            }
            if interaction_score <= 0.02 and not pair_payload["contact_segments"]:
                continue
            pair_interactions.append(pair_payload)

    pair_interactions.sort(key=lambda item: (-float(item["interaction_score"]), -float(item["contact_ratio"])))
    for item in pair_interactions:
        for source_key, target_key in (("entity_a", "entity_b"), ("entity_b", "entity_a")):
            assignment = assignment_map.get(int(item[source_key]))
            if assignment is None:
                continue
            assignment["interaction_partners"].append(
                {
                    "entity_id": int(item[target_key]),
                    "interaction_score": float(item["interaction_score"]),
                    "overlap_ratio": float(item["overlap_ratio"]),
                    "contact_ratio": float(item["contact_ratio"]),
                    "contact_segments": item["contact_segments"],
                }
            )
    for assignment in assignments:
        assignment["interaction_partners"] = assignment["interaction_partners"][:max_partners]

    payload = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "frame_count": tracks_payload.get("frame_count"),
        "num_assignments": len(assignments),
        "num_tool_like": int(sum(1 for item in assignments if item["role_scores"]["tool"] >= 0.60)),
        "num_patient_like": int(sum(1 for item in assignments if item["role_scores"]["patient"] >= 0.60)),
        "num_support_like": int(sum(1 for item in assignments if item["role_scores"]["support"] >= 0.60)),
        "num_agent_like": int(sum(1 for item in assignments if item["role_scores"]["agent"] >= 0.60)),
        "num_interaction_pairs": int(len(pair_interactions)),
        "assignments": assignments,
        "pair_interactions": pair_interactions[: max(32, max_partners * 4)],
    }
    output_path = entitybank_dir / "native_semantic_assignments.json"
    _write_json(output_path, payload)

    enriched_entities = []
    for entity in entities_payload.get("entities", []):
        assignment = assignment_map.get(int(entity["id"]))
        entity_copy = dict(entity)
        if assignment is not None:
            entity_copy["native_semantic_head"] = assignment["semantic_head"]
            entity_copy["native_role_scores"] = assignment["role_scores"]
            entity_copy["native_concept_tags"] = assignment["concept_tags"]
            entity_copy["static_text"] = assignment["native_text"]["static_text"]
            entity_copy["global_desc"] = assignment["native_text"]["global_desc"]
            entity_copy["dyn_desc"] = assignment["native_text"]["dynamic_desc"]
            entity_copy["native_semantic_source"] = "refergaussian_native"
        enriched_entities.append(entity_copy)
    _write_json(
        entitybank_dir / "entities_semantic_native.json",
        {
            **entities_payload,
            "entities": enriched_entities,
            "semantic_source": "refergaussian_native",
        },
    )
    return output_path
