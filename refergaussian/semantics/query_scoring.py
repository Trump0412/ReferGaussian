from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .bootstrap import _resolve_source_images
from .native_assignment import export_native_semantic_assignments
from .qwen_assignment import export_qwen_semantic_assignments


STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "with",
    "on",
    "in",
    "of",
    "to",
    "into",
    "over",
    "through",
    "from",
}
ACTION_WORDS = {
    "cut",
    "cuts",
    "cutting",
    "slice",
    "slices",
    "sliced",
    "chop",
    "chops",
    "chopping",
    "split",
    "splits",
    "peel",
    "peels",
    "stir",
    "stirs",
    "pour",
    "pours",
    "pick",
    "holds",
    "hold",
    "holding",
    "move",
    "moving",
    "touch",
    "touches",
    "push",
    "pushing",
}
TOOL_WORDS = {"knife", "blade", "cutter", "scissors", "fork", "spoon", "spatula", "saw"}
PERSON_WORDS = {"person", "human", "hand", "arm", "man", "woman", "chef"}
SUPPORT_WORDS = {"board", "table", "plate", "surface", "counter", "tray"}
BACKGROUND_WORDS = {"background", "wall", "scene", "floor"}
OBJECT_WORDS = {"lemon", "apple", "orange", "fruit", "object", "target", "food"}
ACTION_NEEDS_TOOL = {"cut", "cuts", "cutting", "slice", "slices", "sliced", "chop", "chops", "chopping"}


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = value.strip()
    return payload


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _parse_query(query: str) -> dict[str, Any]:
    tokens = _tokenize(query)
    action_words = [token for token in tokens if token in ACTION_WORDS]
    nouns = [token for token in tokens if token not in STOPWORDS and token not in action_words]
    tool_nouns = [token for token in nouns if token in TOOL_WORDS]
    person_nouns = [token for token in nouns if token in PERSON_WORDS]
    support_nouns = [token for token in nouns if token in SUPPORT_WORDS]
    background_nouns = [token for token in nouns if token in BACKGROUND_WORDS]
    object_nouns = [
        token
        for token in nouns
        if token not in tool_nouns and token not in person_nouns and token not in support_nouns and token not in background_nouns
    ]
    mentions_tool = bool(tool_nouns or any(token in ACTION_NEEDS_TOOL for token in action_words))
    dynamic_query = bool(action_words)
    return {
        "query": query,
        "tokens": tokens,
        "action_words": action_words,
        "nouns": nouns,
        "object_nouns": object_nouns,
        "tool_nouns": tool_nouns,
        "body_part_nouns": [],
        "person_nouns": person_nouns,
        "support_nouns": support_nouns,
        "background_nouns": background_nouns,
        "mentions_support": bool(support_nouns),
        "mentions_background": bool(background_nouns),
        "mentions_body_part": False,
        "mentions_person": bool(person_nouns),
        "mentions_tool": mentions_tool,
        "wants_multiple": bool("and" in tokens or len(object_nouns) > 1 or len(tool_nouns) > 1),
        "dynamic_query": dynamic_query,
    }


def _test_time_values(run_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    config = _read_simple_yaml(run_dir / "config.yaml")
    source_path = Path(config.get("source_path", ""))
    source_images = _resolve_source_images(source_path)
    return source_images, np.asarray([float(item["time_value"]) for item in source_images], dtype=np.float32)


def _sample_time_values(run_dir: Path) -> np.ndarray:
    payload = np.load(run_dir / "entitybank" / "trajectory_samples.npz")
    return payload["time_values"].astype(np.float32)


def _resample_mask(sample_mask: np.ndarray, sample_times: np.ndarray, test_times: np.ndarray) -> np.ndarray:
    sample_mask = np.asarray(sample_mask, dtype=np.float32)
    indices = np.abs(test_times[:, None] - sample_times[None, :]).argmin(axis=1)
    return sample_mask[indices] > 0.5


def _ranges_from_mask(mask: np.ndarray) -> list[list[int]]:
    indices = np.where(np.asarray(mask, dtype=bool))[0]
    if indices.size == 0:
        return []
    ranges: list[list[int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        ranges.append([start, prev])
        start = value
        prev = value
    ranges.append([start, prev])
    return ranges


def _mask_from_segments(segments: list[list[int]], frame_count: int) -> np.ndarray:
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


def _score_text_overlap(tokens: list[str], candidates: list[str]) -> float:
    if not tokens:
        return 0.0
    joined = " ".join(str(item).lower() for item in candidates)
    score = 0.0
    for token in tokens:
        if token in joined:
            score += 1.0
    return float(score / max(len(tokens), 1))


def _entity_candidate(
    assignment: dict[str, Any],
    query_slots: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    concept_tags = [str(tag).lower() for tag in assignment.get("concept_tags", [])]
    prompt_values: list[str] = []
    for value in assignment.get("prompt_groups", {}).values():
        prompt_values.extend(str(item) for item in value)
    prompt_values.extend(str(item) for item in assignment.get("semantic_terms", []))
    prompt_values.extend(
        [
            assignment.get("native_text", {}).get("static_text", ""),
            assignment.get("native_text", {}).get("global_desc", ""),
        ]
    )
    lexical_terms = list(dict.fromkeys(concept_tags + prompt_values))

    role_scores = assignment.get("role_scores", {})
    lexical_score = _score_text_overlap(query_slots.get("nouns", []) + query_slots.get("action_words", []), lexical_terms)
    object_bonus = _score_text_overlap(query_slots.get("object_nouns", []), lexical_terms)
    tool_bonus = _score_text_overlap(query_slots.get("tool_nouns", []), lexical_terms)
    support_bonus = _score_text_overlap(query_slots.get("support_nouns", []), lexical_terms)
    head_bonus = 0.0
    semantic_head = str(assignment.get("semantic_head", "dynamic"))
    if query_slots.get("dynamic_query"):
        head_bonus += 0.18 if semantic_head in {"dynamic", "interaction"} else 0.0
    if role == "tool" and semantic_head == "interaction":
        head_bonus += 0.12
    if role == "patient" and semantic_head in {"interaction", "dynamic"}:
        head_bonus += 0.08

    role_score = float(role_scores.get(role, 0.0))
    score = 0.42 * role_score + 0.20 * lexical_score + 0.10 * object_bonus + 0.10 * tool_bonus + 0.08 * support_bonus + head_bonus + 0.10 * float(assignment.get("quality", 0.0))
    return {
        "id": int(assignment["entity_id"]),
        "score": float(score),
        "role": role,
        "semantic_head": semantic_head,
        "entity_type": assignment.get("entity_type"),
        "quality": float(assignment.get("quality", 0.0)),
        "role_scores": role_scores,
        "concept_tags": assignment.get("concept_tags", []),
        "static_text": assignment.get("native_text", {}).get("static_text", ""),
        "global_desc": assignment.get("native_text", {}).get("global_desc", ""),
        "candidate_segments_sample": assignment.get("phase_segments", {}).get("moving" if query_slots.get("dynamic_query") else "stationary", []),
        "prompt_groups": assignment.get("prompt_groups", {}),
    }


def _pair_score(
    patient: dict[str, Any],
    tool: dict[str, Any],
    pair: dict[str, Any],
    query_slots: dict[str, Any],
) -> float:
    lexical_patient = _score_text_overlap(query_slots.get("object_nouns", []), patient.get("concept_tags", []) + patient.get("prompt_groups", {}).get("global", []))
    lexical_tool = _score_text_overlap((query_slots.get("tool_nouns", []) or ["tool"]), tool.get("concept_tags", []) + tool.get("prompt_groups", {}).get("global", []))
    interaction_score = float(pair.get("interaction_score", 0.0))
    contact_ratio = float(pair.get("contact_ratio", 0.0))
    event_contrast = float(pair.get("event_contrast", 0.0))
    frame_count = max(int(pair.get("sample_frame_count", 0)), 1)
    event_mask = _mask_from_segments(pair.get("contact_segments_sample", []), frame_count)
    patient_moving = _mask_from_segments(patient.get("phase_segments", {}).get("moving", []), frame_count)
    tool_moving = _mask_from_segments(tool.get("phase_segments", {}).get("moving", []), frame_count)
    dynamic_overlap = 0.0
    if event_mask.any():
        dynamic_overlap = 0.5 * (
            float((event_mask & patient_moving).sum() / event_mask.sum())
            + float((event_mask & tool_moving).sum() / event_mask.sum())
        )
    localization_bonus = max(0.0, 1.0 - min(contact_ratio / 0.35, 1.0))
    patient_score = float(patient.get("role_scores", {}).get("patient", 0.0))
    tool_score = float(tool.get("role_scores", {}).get("tool", 0.0))
    semantic_bonus = 0.0
    if patient.get("semantic_head") in {"interaction", "dynamic"}:
        semantic_bonus += 0.08
    if tool.get("semantic_head") in {"interaction", "dynamic"}:
        semantic_bonus += 0.08
    dynamic_query_bias = 0.0
    if query_slots.get("dynamic_query"):
        dynamic_query_bias = 0.30 * dynamic_overlap - 0.12
    return (
        0.32 * interaction_score
        + 0.10 * contact_ratio
        + 0.18 * patient_score
        + 0.18 * tool_score
        + 0.08 * lexical_patient
        + 0.06 * lexical_tool
        + 0.08 * localization_bonus
        + 0.06 * min(event_contrast * 20.0, 1.0)
        + 0.18 * dynamic_overlap
        + dynamic_query_bias
        + semantic_bonus
    )


def _select_top_entity(candidates: list[dict[str, Any]], excluded_ids: set[int] | None = None) -> dict[str, Any] | None:
    excluded_ids = excluded_ids or set()
    filtered = [item for item in candidates if int(item["id"]) not in excluded_ids]
    if not filtered:
        return None
    filtered.sort(key=lambda item: (-float(item["score"]), int(item["id"])))
    return filtered[0]


def score_native_query(
    run_dir: str | Path,
    query: str,
    query_name: str | None = None,
    top_k: int = 20,
    semantic_source: str = "native",
    qwen_model: str | None = None,
) -> Path:
    run_dir = Path(run_dir)
    entitybank_dir = run_dir / "entitybank"
    if semantic_source == "qwen":
        assignments_path = entitybank_dir / "semantic_assignments_qwen.json"
        if not assignments_path.exists():
            export_qwen_semantic_assignments(run_dir, qwen_model=qwen_model, query=query)
    else:
        assignments_path = entitybank_dir / "native_semantic_assignments.json"
        if not assignments_path.exists():
            export_native_semantic_assignments(run_dir)
    assignments_payload = _read_json(assignments_path)
    assignments = assignments_payload.get("assignments", [])
    assignment_map = {int(item["entity_id"]): item for item in assignments}
    pair_map = {}
    for pair in assignments_payload.get("pair_interactions", []):
        key = tuple(sorted((int(pair["entity_a"]), int(pair["entity_b"]))))
        pair_map[key] = pair

    query_slots = _parse_query(query)
    query_dir_name = query_name or re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_") or "query"
    query_root = "qwen_queries" if semantic_source == "qwen" else "native_queries"
    query_dir = entitybank_dir / query_root / query_dir_name
    query_dir.mkdir(parents=True, exist_ok=True)

    patient_candidates = [_entity_candidate(item, query_slots, role="patient") for item in assignments]
    tool_candidates = [_entity_candidate(item, query_slots, role="tool") for item in assignments]
    support_candidates = [_entity_candidate(item, query_slots, role="support") for item in assignments]
    patient_candidates.sort(key=lambda item: (-float(item["score"]), int(item["id"])))
    tool_candidates.sort(key=lambda item: (-float(item["score"]), int(item["id"])))
    support_candidates.sort(key=lambda item: (-float(item["score"]), int(item["id"])))

    pair_candidates = []
    if query_slots.get("dynamic_query"):
        for key, pair in pair_map.items():
            a = assignment_map.get(int(pair["entity_a"]))
            b = assignment_map.get(int(pair["entity_b"]))
            if a is None or b is None:
                continue
            orientations = [("patient_tool", a, b), ("patient_tool", b, a)]
            for _name, patient, tool in orientations:
                score = _pair_score(patient, tool, pair, query_slots)
                frame_count = max(int(pair.get("sample_frame_count", 0)), 1)
                event_mask = _mask_from_segments(pair.get("contact_segments", []), frame_count)
                patient_moving = _mask_from_segments(patient.get("phase_segments", {}).get("moving", []), frame_count)
                tool_moving = _mask_from_segments(tool.get("phase_segments", {}).get("moving", []), frame_count)
                dynamic_overlap = 0.0
                if event_mask.any():
                    dynamic_overlap = 0.5 * (
                        float((event_mask & patient_moving).sum() / event_mask.sum())
                        + float((event_mask & tool_moving).sum() / event_mask.sum())
                    )
                if query_slots.get("dynamic_query") and dynamic_overlap < 0.10:
                    continue
                pair_candidates.append(
                    {
                        "patient_id": int(patient["entity_id"]),
                        "tool_id": int(tool["entity_id"]),
                        "pair_score": float(score),
                        "interaction_score": float(pair.get("interaction_score", 0.0)),
                        "contact_ratio": float(pair.get("contact_ratio", 0.0)),
                        "event_contrast": float(pair.get("event_contrast", 0.0)),
                        "dynamic_overlap": float(dynamic_overlap),
                        "contact_segments_sample": pair.get("contact_segments", []),
                    }
                )
        pair_candidates.sort(key=lambda item: (-float(item["pair_score"]), int(item["patient_id"]), int(item["tool_id"])))

    source_images, test_times = _test_time_values(run_dir)
    sample_times = _sample_time_values(run_dir)

    selected_items: list[dict[str, Any]] = []
    if query_slots.get("dynamic_query") and pair_candidates:
        best_pair = pair_candidates[0]
        sample_mask = np.zeros((sample_times.shape[0],), dtype=bool)
        for start, end in best_pair.get("contact_segments_sample", []):
            sample_mask[max(int(start), 0) : min(int(end) + 1, sample_times.shape[0])] = True
        if not sample_mask.any():
            patient_assignment = assignment_map[int(best_pair["patient_id"])]
            fallback_segments = patient_assignment.get("phase_segments", {}).get("moving", [])
            for start, end in fallback_segments:
                sample_mask[max(int(start), 0) : min(int(end) + 1, sample_times.shape[0])] = True
        test_mask = _resample_mask(sample_mask.astype(np.float32), sample_times, test_times)
        test_segments = _ranges_from_mask(test_mask)
        patient_assignment = assignment_map[int(best_pair["patient_id"])]
        tool_assignment = assignment_map[int(best_pair["tool_id"])]
        selected_items.append(
            {
                "id": int(best_pair["patient_id"]),
                "role": "patient",
                "entity_type": patient_assignment.get("entity_type", "object"),
                "confidence": float(best_pair["pair_score"]),
                "segments": test_segments,
            }
        )
        if query_slots.get("mentions_tool") or query_slots.get("dynamic_query"):
            selected_items.append(
                {
                    "id": int(best_pair["tool_id"]),
                    "role": "tool",
                    "entity_type": tool_assignment.get("entity_type", "tool"),
                    "confidence": float(best_pair["pair_score"]) * 0.92,
                    "segments": test_segments,
                }
            )
    else:
        top_patient = _select_top_entity(patient_candidates)
        if top_patient is not None:
            sample_mask = np.zeros((sample_times.shape[0],), dtype=bool)
            for start, end in top_patient.get("candidate_segments_sample", []):
                sample_mask[max(int(start), 0) : min(int(end) + 1, sample_times.shape[0])] = True
            if not sample_mask.any():
                sample_mask[:] = True
            test_segments = _ranges_from_mask(_resample_mask(sample_mask.astype(np.float32), sample_times, test_times))
            selected_items.append(
                {
                    "id": int(top_patient["id"]),
                    "role": "patient",
                    "entity_type": top_patient.get("entity_type", "object"),
                    "confidence": float(top_patient["score"]),
                    "segments": test_segments,
                }
            )
        if query_slots.get("mentions_support"):
            top_support = _select_top_entity(support_candidates, excluded_ids={item["id"] for item in selected_items})
            if top_support is not None:
                selected_items.append(
                    {
                        "id": int(top_support["id"]),
                        "role": "support",
                        "entity_type": top_support.get("entity_type", "support_surface"),
                        "confidence": float(top_support["score"]),
                        "segments": [[0, max(len(source_images) - 1, 0)]],
                    }
                )

    candidates_payload = {
        "query": {
            "query": query,
            "dynamic_query": bool(query_slots.get("dynamic_query")),
            "query_slots": query_slots,
            "embedding_model": "refergaussian-qwen-heuristic" if semantic_source == "qwen" else "refergaussian-native-heuristic",
            "top_k": int(top_k),
            "exclude_entity_types": ["background_stuff"],
        },
        "pair_candidates": pair_candidates[:top_k],
        "patient_candidates": patient_candidates[:top_k],
        "tool_candidates": tool_candidates[:top_k],
        "support_candidates": support_candidates[:top_k],
    }
    selected_payload = {
        "selected": selected_items,
        "empty": len(selected_items) == 0,
        "reason": "" if selected_items else "no native semantic match",
        "query_slots": query_slots,
        "source_query_dir": None,
        "semantic_source": "refergaussian_qwen" if semantic_source == "qwen" else "refergaussian_native",
    }
    query_payload = {
        "query": query,
        "query_slots": query_slots,
        "native_semantic_source": "refergaussian_qwen" if semantic_source == "qwen" else "refergaussian_native",
        "num_assignments": int(assignments_payload.get("num_assignments", 0)),
        "semantic_source": semantic_source,
    }

    _write_json(query_dir / "query.json", query_payload)
    _write_json(query_dir / "candidates.json", candidates_payload)
    _write_json(query_dir / "selected.json", selected_payload)
    return query_dir
