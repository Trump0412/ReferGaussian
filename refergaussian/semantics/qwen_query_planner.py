from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .source_images import resolve_dataset_image_entries

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QWEN_MODEL = Path(os.environ.get("HYPERGAUSSIAN_QWEN_MODEL", str(_PROJECT_ROOT / "models" / "Qwen3-VL-8B-Instruct")))


QUERY_PLAN_TEMPLATE = """You are planning query-conditioned 4D entity discovery for ReferGaussian.
You will be given a natural-language query and a small set of uniformly sampled video frames.
Your job is to understand which objects appear in the full video, which objects are the true query subjects,
and whether the action creates a successor object state that should be tracked.

Query:
{query}

Observed sampled frames:
{frame_summary}

Return exactly one JSON object with keys:
- query: original query
- video_inventory_phrases: array of short static noun phrases describing the main visible objects in the whole video
- query_subject_phrases: array of short static noun phrases for the primary query objects only
- query_successor_phrases: array of short static noun phrases that appear only after a query-driven state change, such as "lemon halves"
- phase_transition_hints: array of objects, each with keys {{phrase, last_pre_change_slot, first_post_change_slot, reason}}
- detector_phrases: array of short static noun phrases to detect and track
- optional_phrases: array of extra visible nouns that are not query subjects
- interaction_phrase: short phrase describing the interaction
- start_condition: short phrase describing when the query event should begin
- stop_condition: short phrase describing when the query event should stop
- temporal_hints: array of short phase descriptions
- must_track_phrases: array of phrases that must be tracked through time
- preferred_detector: one of grounding_dino, grounded_sam2
- notes: short string

Rules:
- The sampled frames are for whole-video inventory and subject discovery, not for forcing the exact action frame.
- Use the sampled frames to infer which objects exist in the full video, then plan only the query-relevant subjects and successor states.
- Keep the output compact: video_inventory_phrases <= 8, query_subject_phrases <= 3, query_successor_phrases <= 2, detector_phrases <= 4, optional_phrases <= 6, temporal_hints <= 4.
- Do not use role words like patient/tool/agent in the phrase lists.
- Keep phrases concrete and static, such as "knife", "lemon", "hand", "cutting board", "lemon halves".
- `video_inventory_phrases` should summarize the main objects visible across the whole video.
- `query_subject_phrases` should contain only the minimum nouns needed to answer the query.
- `detector_phrases` should normally equal `query_subject_phrases + query_successor_phrases`, and not include unrelated context objects.
- Do not include non-subject context objects like a board or a hand in `detector_phrases` unless they are truly required by the query itself.
- Only use `query_successor_phrases` when the action creates a new stable object state, such as "lemon halves".
- Do not invent count-based successor phrases such as "lemon halves" when downstream mask tracking can discover the split directly.
- `phase_transition_hints` should use the 0-based sampled-frame slot indices from the observed frame list.
- Use `phase_transition_hints` only when the query implies a subject state transition or a before/after distinction.
- Prefer the earliest semantic change point suggested by the sampled frames, not the latest frame where the object is already fully separated.
- For "cut the lemon", a good answer would inventory ["knife", "lemon", "hand", "cutting board"], and use query subjects ["knife", "lemon"].
- For action queries, `start_condition` should begin at direct task-relevant contact, not at coarse pre-contact setup.
- For action queries, `stop_condition` should end when the query-driven state change stabilizes, not when unrelated context remains visible.
- If the query implies temporal change, include before/during/after hints.
- preferred_detector must be "grounded_sam2".
- Output valid JSON only.
"""


TEMPORAL_WINDOW_TEMPLATE = """You are refining the coarse temporal range for a query-conditioned 4D event/state.
You will be given the original query, the already planned subject nouns, and an ordered set of sampled video frames.

Query:
{query}

Planned subject nouns:
{subject_phrases}

Observed ordered sampled frames:
{frame_summary}

Return exactly one JSON object with keys:
- query: original query
- start_slot: 0-based slot index of the earliest sampled frame that should count as active for this query, or null
- end_slot: 0-based slot index of the latest sampled frame that should count as active for this query, or null
- frame_labels: array of objects with keys {{slot, label, reason}}, where label is one of before, inside, after
- notes: short string

Rules:
- Judge semantic activity, not just visibility.
- Keep frame_labels concise and notes short.
- For full-object queries like "the cookie", mark the full lifetime where that queried object/state should count.
- For state queries like "the complete cookie", stop before the object enters the broken state.
- For post-change queries like "the cookie broken into smaller pieces", start when the object first clearly belongs to the changed state.
- If uncertain, prefer a slightly earlier semantic transition rather than waiting for the object to become maximally separated.
- Output valid JSON only.
"""


BOUNDARY_REFINE_TEMPLATE = """You are refining the semantic {boundary_kind} boundary for a query-conditioned 4D event/state.
You will be given the original query, the already planned subject nouns, the relevant boundary condition,
and an ordered set of sampled video frames from a narrow temporal interval.

Query:
{query}

Planned subject nouns:
{subject_phrases}

Boundary kind:
{boundary_kind}

Boundary condition:
{boundary_condition}

Additional semantic guidance:
{state_guidance}

Observed ordered sampled frames in the candidate interval:
{frame_summary}

Return exactly one JSON object with keys:
- query: original query
- boundary_kind: start or end
- boundary_slot: 0-based slot index of the semantic boundary frame inside this interval, or null
- frame_labels: array of objects with keys {{slot, label, reason}}
- notes: short string

Rules:
- If boundary_kind is start, each frame label must be one of before or inside.
- If boundary_kind is end, each frame label must be one of inside or after.
- Keep frame_labels concise and notes short.
- Judge semantic onset/offset, not just maximal visual separation.
- Prefer a slightly earlier semantic onset for start and a slightly later semantic offset for end.
- boundary_slot should be the earliest inside frame for start, or the latest inside frame for end.
- Output valid JSON only.
"""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _clean_llm_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_first_json(text: str) -> dict[str, Any]:
    cleaned = _clean_llm_text(text)
    if not cleaned:
        raise ValueError("Unable to parse JSON object from empty model output.")

    start_index = cleaned.find("{")
    if start_index < 0:
        raise ValueError(f"Unable to find top-level JSON object in model output: {text!r}")

    in_string = False
    escape = False
    depth = 0
    end_index: int | None = None
    for index in range(start_index, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                end_index = index + 1
                break

    if end_index is None:
        snippet = cleaned[start_index : min(len(cleaned), start_index + 400)]
        raise ValueError(f"Top-level JSON object is incomplete or truncated: {snippet!r}")

    candidate = cleaned[start_index:end_index]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse top-level JSON object from model output: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Top-level JSON payload must be an object, got {type(payload).__name__}.")
    return payload


def _resolve_qwen_model(explicit_path: str | None = None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    if DEFAULT_QWEN_MODEL.exists():
        return DEFAULT_QWEN_MODEL
    raise FileNotFoundError(
        f"Unable to resolve a local Qwen model. Expected {DEFAULT_QWEN_MODEL} or pass an explicit path."
    )


def _import_transformers():
    try:
        import transformers  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Python dependency 'transformers'. Install it in the query-planning environment."
        ) from exc
    return transformers


def _qwen_model_load_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": "auto",
        "device_map": "auto",
    }
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            free_gib = max(int(free_bytes // (1024**3)), 1)
            gpu_budget = min(max(free_gib - 2, 8), 16)
            max_memory = {index: f"{gpu_budget}GiB" for index in range(torch.cuda.device_count())}
            max_memory["cpu"] = "160GiB"
            kwargs["max_memory"] = max_memory
    except Exception:
        pass
    return kwargs


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _subsample_entries(entries: list[dict[str, Any]], frame_subsample_stride: int) -> list[dict[str, Any]]:
    stride = max(int(frame_subsample_stride), 1)
    sampled = entries[::stride]
    if entries and sampled and sampled[-1]["frame_index"] != entries[-1]["frame_index"]:
        sampled.append(entries[-1])
    return sampled


def _sample_context_entries(entries: list[dict[str, Any]], num_sampled_frames: int) -> list[dict[str, Any]]:
    if not entries:
        raise ValueError("No image entries available for Qwen query planning.")
    count = max(int(num_sampled_frames), 1)
    if len(entries) <= count:
        return list(entries)
    indices = np.linspace(0, len(entries) - 1, num=count, dtype=np.int32)
    return [entries[int(index)] for index in indices.tolist()]


def _load_context_images(
    dataset_dir: Path,
    frame_subsample_stride: int,
    num_sampled_frames: int,
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    sampled_entries = _load_subsampled_entries(dataset_dir, frame_subsample_stride=frame_subsample_stride)
    context_entries = _sample_context_entries(sampled_entries, num_sampled_frames=num_sampled_frames)
    images = _load_images_for_entries(context_entries)
    return context_entries, images


def _load_subsampled_entries(dataset_dir: Path, frame_subsample_stride: int) -> list[dict[str, Any]]:
    all_entries = resolve_dataset_image_entries(dataset_dir)
    return _subsample_entries(all_entries, frame_subsample_stride=frame_subsample_stride)


def _load_images_for_entries(entries: list[dict[str, Any]]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for entry in entries:
        with Image.open(entry["image_path"]) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            longest = max(width, height)
            if longest > 896:
                scale = 896.0 / float(longest)
                rgb = rgb.resize(
                    (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                    Image.Resampling.BICUBIC,
                )
            images.append(rgb)
    return images


def _frame_summary(entries: list[dict[str, Any]]) -> str:
    summary = [
        {
            "slot": int(index),
            "frame_index": int(entry["frame_index"]),
            "image_id": str(entry["image_id"]),
            "time_value": round(float(entry["time_value"]), 6),
        }
        for index, entry in enumerate(entries)
    ]
    return json.dumps(summary, ensure_ascii=False)


def _query_state_guidance(query: str, boundary_kind: str) -> str:
    query_norm = " ".join(str(query).strip().lower().split())
    intact_keywords = ("complete", "whole", "intact", "unbroken")
    split_keywords = ("broken", "pieces", "halves", "split", "cracked")
    action_keywords = ("cut", "slice", "break", "open", "peel", "pour", "stir", "mix")
    if boundary_kind == "end" and any(keyword in query_norm for keyword in intact_keywords):
        return (
            "Treat the boundary as the last frame where the queried object is still intact. "
            "As soon as a visible crack, break onset, or non-intact state appears, the next frames are after. "
            "Do not wait until the pieces are fully far apart."
        )
    if boundary_kind == "start" and any(keyword in query_norm for keyword in split_keywords):
        return (
            "Treat the boundary as the earliest frame where the object first becomes broken, cracked, or split. "
            "Do not wait for maximal separation between the resulting pieces."
        )
    if boundary_kind == "start" and any(keyword in query_norm for keyword in action_keywords):
        return "Start at the first direct task-relevant contact or action onset."
    if boundary_kind == "end" and any(keyword in query_norm for keyword in action_keywords):
        return "End when the action-driven state change has become established, not when the context disappears."
    return "Use the semantic meaning of the query and the boundary condition to decide the onset/offset."


def _query_semantic_profile(query: str) -> dict[str, Any]:
    query_norm = " ".join(str(query).strip().lower().split())
    tokens = set(re.findall(r"[a-z]+", query_norm))
    intact_keywords = {"complete", "whole", "intact", "unbroken", "before"}
    changed_keywords = {"broken", "pieces", "halves", "split", "cracked", "after"}
    action_keywords = {"cut", "slice", "break", "open", "peel", "pour", "stir", "mix", "fill", "darkening", "roaming"}
    set_keywords = {"all", "everything", "objects", "participants"}
    return {
        "query_norm": query_norm,
        "asks_before_state": bool(tokens & intact_keywords),
        "asks_after_state": bool(tokens & changed_keywords),
        "asks_action_window": bool(tokens & action_keywords),
        "asks_set": bool(tokens & set_keywords),
    }


def _normalize_phrase_list(values: Any) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        phrase = " ".join(str(value).strip().lower().split())
        if not phrase:
            continue
        if phrase in seen:
            continue
        seen.add(phrase)
        normalized.append(phrase)
    return normalized


def _merge_unique_phrases(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for phrase in group:
            if phrase in seen:
                continue
            seen.add(phrase)
            merged.append(phrase)
    return merged


def _normalize_phase_transition_hints(values: Any, valid_phrases: set[str]) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    for value in values or []:
        if not isinstance(value, dict):
            continue
        phrase = " ".join(str(value.get("phrase", "")).strip().lower().split())
        if not phrase or (valid_phrases and phrase not in valid_phrases):
            continue
        last_pre = value.get("last_pre_change_slot")
        first_post = value.get("first_post_change_slot")
        try:
            last_pre = None if last_pre is None else int(last_pre)
        except Exception:
            last_pre = None
        try:
            first_post = None if first_post is None else int(first_post)
        except Exception:
            first_post = None
        try:
            last_pre_frame = value.get("last_pre_change_frame_index")
            last_pre_frame = None if last_pre_frame is None else int(last_pre_frame)
        except Exception:
            last_pre_frame = None
        try:
            first_post_frame = value.get("first_post_change_frame_index")
            first_post_frame = None if first_post_frame is None else int(first_post_frame)
        except Exception:
            first_post_frame = None
        reason = " ".join(str(value.get("reason", "")).strip().split())
        key = (phrase, last_pre, first_post)
        if key in seen:
            continue
        seen.add(key)
        hints.append(
            {
                "phrase": phrase,
                "last_pre_change_slot": last_pre,
                "first_post_change_slot": first_post,
                "last_pre_change_frame_index": last_pre_frame,
                "first_post_change_frame_index": first_post_frame,
                "reason": reason,
            }
        )
    return hints


def _normalize_temporal_window(raw_payload: dict[str, Any], frame_count: int) -> dict[str, Any]:
    try:
        start_slot = raw_payload.get("start_slot")
        start_slot = None if start_slot is None else int(start_slot)
    except Exception:
        start_slot = None
    try:
        end_slot = raw_payload.get("end_slot")
        end_slot = None if end_slot is None else int(end_slot)
    except Exception:
        end_slot = None
    if start_slot is not None:
        start_slot = max(0, min(int(frame_count - 1), start_slot))
    if end_slot is not None:
        end_slot = max(0, min(int(frame_count - 1), end_slot))
    if start_slot is not None and end_slot is not None and end_slot < start_slot:
        start_slot, end_slot = end_slot, start_slot

    frame_labels = []
    for value in raw_payload.get("frame_labels", []) or []:
        if not isinstance(value, dict):
            continue
        try:
            slot = int(value.get("slot"))
        except Exception:
            continue
        if slot < 0 or slot >= frame_count:
            continue
        label = " ".join(str(value.get("label", "")).strip().lower().split())
        if label not in {"before", "inside", "after"}:
            continue
        reason = " ".join(str(value.get("reason", "")).strip().split())
        frame_labels.append({"slot": slot, "label": label, "reason": reason})
    notes = " ".join(str(raw_payload.get("notes", "")).strip().split())
    return {
        "start_slot": start_slot,
        "end_slot": end_slot,
        "frame_labels": frame_labels,
        "notes": notes,
    }


def _normalize_boundary_refinement(
    raw_payload: dict[str, Any],
    frame_count: int,
    boundary_kind: str,
) -> dict[str, Any]:
    valid_labels = {"before", "inside"} if boundary_kind == "start" else {"inside", "after"}
    try:
        boundary_slot = raw_payload.get("boundary_slot")
        boundary_slot = None if boundary_slot is None else int(boundary_slot)
    except Exception:
        boundary_slot = None
    if boundary_slot is not None:
        boundary_slot = max(0, min(int(frame_count - 1), boundary_slot))
    frame_labels = []
    for value in raw_payload.get("frame_labels", []) or []:
        if not isinstance(value, dict):
            continue
        try:
            slot = int(value.get("slot"))
        except Exception:
            continue
        if slot < 0 or slot >= frame_count:
            continue
        label = " ".join(str(value.get("label", "")).strip().lower().split())
        if label not in valid_labels:
            continue
        reason = " ".join(str(value.get("reason", "")).strip().split())
        frame_labels.append({"slot": slot, "label": label, "reason": reason})
    notes = " ".join(str(raw_payload.get("notes", "")).strip().split())
    return {
        "boundary_slot": boundary_slot,
        "frame_labels": frame_labels,
        "notes": notes,
    }


def _normalize_plan(raw_payload: dict[str, Any], query: str, strict: bool = True) -> dict[str, Any]:
    video_inventory_phrases = _normalize_phrase_list(raw_payload.get("video_inventory_phrases", []))[:8]
    query_subject_phrases = _normalize_phrase_list(raw_payload.get("query_subject_phrases", []))[:3]
    query_successor_phrases = _normalize_phrase_list(raw_payload.get("query_successor_phrases", []))[:2]
    raw_optional_phrases = _normalize_phrase_list(raw_payload.get("optional_phrases", []))[:6]
    must_track_phrases = _normalize_phrase_list(raw_payload.get("must_track_phrases", []))[:3]
    temporal_hints = _normalize_phrase_list(raw_payload.get("temporal_hints", []))[:4]
    interaction_phrase = " ".join(str(raw_payload.get("interaction_phrase", query)).strip().split())
    start_condition = " ".join(str(raw_payload.get("start_condition", "")).strip().split())
    stop_condition = " ".join(str(raw_payload.get("stop_condition", "")).strip().split())
    preferred_detector = str(raw_payload.get("preferred_detector", "grounded_sam2")).strip().lower()
    notes = " ".join(str(raw_payload.get("notes", "")).strip().split())

    if strict and not video_inventory_phrases:
        raise ValueError("Strict Qwen planner returned no video_inventory_phrases.")
    if strict and not query_subject_phrases:
        raise ValueError("Strict Qwen planner returned no query_subject_phrases.")
    if not strict and not video_inventory_phrases:
        video_inventory_phrases = _normalize_phrase_list(raw_payload.get("detector_phrases", []))
    if not strict and not query_subject_phrases:
        query_subject_phrases = _normalize_phrase_list(raw_payload.get("detector_phrases", []))[:2]
    if not strict and not video_inventory_phrases:
        video_inventory_phrases = query_subject_phrases[:]

    detector_phrases = _merge_unique_phrases(query_subject_phrases, query_successor_phrases)[:4]
    if strict and not detector_phrases:
        raise ValueError("Strict Qwen planner produced no detector_phrases after subject filtering.")
    if not strict and not detector_phrases:
        detector_phrases = _normalize_phrase_list(raw_payload.get("detector_phrases", []))
    if not strict and not detector_phrases:
        detector_phrases = query_subject_phrases[:]

    optional_phrases = [phrase for phrase in video_inventory_phrases if phrase not in detector_phrases]
    optional_phrases = _merge_unique_phrases(optional_phrases, [phrase for phrase in raw_optional_phrases if phrase not in detector_phrases])[:6]
    must_track_phrases = [phrase for phrase in query_subject_phrases if phrase in must_track_phrases] or query_subject_phrases[:]
    phase_transition_hints = _normalize_phase_transition_hints(
        raw_payload.get("phase_transition_hints", []),
        valid_phrases=set(_merge_unique_phrases(query_subject_phrases, query_successor_phrases, detector_phrases)),
    )

    if preferred_detector != "grounded_sam2":
        preferred_detector = "grounded_sam2"
    if not must_track_phrases:
        must_track_phrases = query_subject_phrases[: min(2, len(query_subject_phrases))]
    if not start_condition:
        start_condition = "when the main query subjects first make task-relevant contact"
    if not stop_condition:
        stop_condition = "when the query-driven object state change has completed"

    return {
        "query": query,
        "video_inventory_phrases": video_inventory_phrases,
        "query_subject_phrases": query_subject_phrases,
        "query_successor_phrases": query_successor_phrases,
        "detector_phrases": detector_phrases,
        "optional_phrases": optional_phrases,
        "interaction_phrase": interaction_phrase,
        "start_condition": start_condition,
        "stop_condition": stop_condition,
        "temporal_hints": temporal_hints,
        "phase_transition_hints": phase_transition_hints,
        "must_track_phrases": must_track_phrases,
        "preferred_detector": preferred_detector,
        "notes": notes,
    }


def _derive_transition_hints_from_window(
    *,
    plan: dict[str, Any],
    window_plan: dict[str, Any] | None,
    sampled_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not window_plan or not sampled_entries:
        return list(plan.get("phase_transition_hints", []))
    subject_phrases = list(plan.get("query_subject_phrases", []))
    if len(subject_phrases) != 1:
        return list(plan.get("phase_transition_hints", []))
    phrase = str(subject_phrases[0])
    start_index = window_plan.get("start_sample_index")
    end_index = window_plan.get("end_sample_index")
    if start_index is None and end_index is None:
        return list(plan.get("phase_transition_hints", []))
    query_norm = " ".join(str(plan.get("query", "")).strip().lower().split())
    split_keywords = ("broken", "pieces", "halves", "split", "cracked", "cut")
    intact_keywords = ("complete", "whole", "intact", "unbroken")
    hint_mode: str | None = None
    if any(keyword in query_norm for keyword in split_keywords):
        hint_mode = "start"
    elif any(keyword in query_norm for keyword in intact_keywords):
        hint_mode = "end"
    elif plan.get("query_successor_phrases"):
        hint_mode = "start"
    if hint_mode is None:
        return list(plan.get("phase_transition_hints", []))

    hints: list[dict[str, Any]] = list(plan.get("phase_transition_hints", []))
    if hint_mode == "start" and start_index is not None and int(start_index) > 0:
        prev_entry = sampled_entries[int(start_index) - 1]
        curr_entry = sampled_entries[int(start_index)]
        hints.append(
            {
                "phrase": phrase,
                "last_pre_change_slot": None,
                "first_post_change_slot": None,
                "last_pre_change_frame_index": int(prev_entry["frame_index"]),
                "first_post_change_frame_index": int(curr_entry["frame_index"]),
                "reason": "Derived from Qwen temporal window start.",
            }
        )
    if hint_mode == "end" and end_index is not None and int(end_index) < int(len(sampled_entries) - 1):
        curr_entry = sampled_entries[int(end_index)]
        next_entry = sampled_entries[int(end_index) + 1]
        hints.append(
            {
                "phrase": phrase,
                "last_pre_change_slot": None,
                "first_post_change_slot": None,
                "last_pre_change_frame_index": int(curr_entry["frame_index"]),
                "first_post_change_frame_index": int(next_entry["frame_index"]),
                "reason": "Derived from Qwen temporal window end.",
            }
        )
    return _normalize_phase_transition_hints(hints, valid_phrases={phrase})


def _entry_index_lookup(entries: list[dict[str, Any]]) -> dict[int, int]:
    return {int(entry["frame_index"]): int(index) for index, entry in enumerate(entries)}


def _interval_indices(start_index: int, end_index: int, sample_count: int) -> list[int]:
    if end_index < start_index:
        start_index, end_index = end_index, start_index
    count = min(max(int(sample_count), 2), int(end_index - start_index + 1))
    raw = np.linspace(start_index, end_index, num=count, dtype=np.int32).tolist()
    ordered: list[int] = []
    seen: set[int] = set()
    for value in raw:
        index = int(value)
        if index in seen:
            continue
        seen.add(index)
        ordered.append(index)
    if start_index not in seen:
        ordered.insert(0, int(start_index))
        seen.add(int(start_index))
    if end_index not in seen:
        ordered.append(int(end_index))
    return sorted(set(int(value) for value in ordered))


def _coarse_index_for_slot(
    slot_value: int | None,
    sampled_entries: list[dict[str, Any]],
    coarse_entries: list[dict[str, Any]],
    lookup: dict[int, int],
) -> int | None:
    if slot_value is None:
        return None
    if int(slot_value) < 0 or int(slot_value) >= len(coarse_entries):
        return None
    frame_index = int(coarse_entries[int(slot_value)]["frame_index"])
    return lookup.get(frame_index)


def _boundary_search_interval(
    *,
    boundary_kind: str,
    coarse_start_index: int | None,
    coarse_end_index: int | None,
    frame_count: int,
) -> tuple[int, int]:
    if frame_count <= 0:
        return 0, 0
    margin = max(3, min(12, frame_count // 8 if frame_count >= 8 else 3))
    anchor = coarse_start_index if boundary_kind == "start" else coarse_end_index
    if anchor is None:
        return 0, int(frame_count - 1)
    low = max(0, int(anchor) - margin)
    high = min(int(frame_count - 1), int(anchor) + margin)
    if high < low:
        low, high = high, low
    return low, high


def _finalize_temporal_window(
    *,
    query: str,
    frame_count: int,
    coarse_start_index: int | None,
    coarse_end_index: int | None,
    refined_start_index: int | None,
    refined_end_index: int | None,
) -> tuple[int | None, int | None]:
    if frame_count <= 0:
        return refined_start_index, refined_end_index
    profile = _query_semantic_profile(query)
    last_index = int(frame_count - 1)
    start_index = refined_start_index
    end_index = refined_end_index

    if start_index is None:
        if profile["asks_after_state"]:
            start_index = coarse_start_index
        else:
            start_index = 0
    if end_index is None:
        if profile["asks_before_state"]:
            end_index = coarse_end_index
        else:
            end_index = last_index

    if start_index is not None:
        start_index = max(0, min(last_index, int(start_index)))
    if end_index is not None:
        end_index = max(0, min(last_index, int(end_index)))

    if start_index is not None and end_index is not None and end_index < start_index:
        if profile["asks_before_state"] and coarse_end_index is not None:
            end_index = max(0, min(last_index, int(coarse_end_index)))
        elif profile["asks_after_state"] and coarse_start_index is not None:
            start_index = max(0, min(last_index, int(coarse_start_index)))
        if end_index < start_index:
            start_index, end_index = min(start_index, end_index), max(start_index, end_index)
    return start_index, end_index


def _refine_boundary_interval(
    *,
    teacher: QwenQueryPlanner,
    query: str,
    subject_phrases: list[str],
    boundary_kind: str,
    boundary_condition: str,
    sampled_entries: list[dict[str, Any]],
    low_index: int,
    high_index: int,
    num_frames: int = 9,
    max_rounds: int = 1,
) -> dict[str, Any]:
    if not sampled_entries:
        return {
            "boundary_index": None,
            "rounds": [],
            "final_interval": {"low_index": None, "high_index": None},
        }
    current_low = max(0, min(int(low_index), len(sampled_entries) - 1))
    current_high = max(0, min(int(high_index), len(sampled_entries) - 1))
    if current_high < current_low:
        current_low, current_high = current_high, current_low
    history: list[dict[str, Any]] = []
    best_index: int | None = None
    for round_index in range(max(int(max_rounds), 0)):
        candidate_indices = _interval_indices(current_low, current_high, sample_count=num_frames)
        candidate_entries = [sampled_entries[index] for index in candidate_indices]
        candidate_images = _load_images_for_entries(candidate_entries)
        prompt = BOUNDARY_REFINE_TEMPLATE.format(
            query=query,
            subject_phrases=json.dumps(subject_phrases, ensure_ascii=False),
            boundary_kind=boundary_kind,
            boundary_condition=boundary_condition,
            state_guidance=_query_state_guidance(query=query, boundary_kind=boundary_kind),
            frame_summary=_frame_summary(candidate_entries),
        )
        raw_payload, raw_output = teacher.generate_json(prompt=prompt, images=candidate_images)
        normalized = _normalize_boundary_refinement(raw_payload, frame_count=len(candidate_entries), boundary_kind=boundary_kind)
        boundary_slot = normalized["boundary_slot"]
        label_rows = normalized["frame_labels"]
        if boundary_kind == "start":
            inside_slots = sorted(int(row["slot"]) for row in label_rows if row["label"] == "inside")
            before_slots = sorted(int(row["slot"]) for row in label_rows if row["label"] == "before")
            if boundary_slot is None:
                boundary_slot = inside_slots[0] if inside_slots else None
            if boundary_slot is None:
                break
            boundary_index = int(candidate_indices[int(boundary_slot)])
            best_index = boundary_index
            next_low = current_low
            before_indices = [int(candidate_indices[slot]) for slot in before_slots if int(candidate_indices[slot]) < boundary_index]
            if before_indices:
                next_low = max(before_indices)
            next_high = boundary_index
        else:
            inside_slots = sorted(int(row["slot"]) for row in label_rows if row["label"] == "inside")
            after_slots = sorted(int(row["slot"]) for row in label_rows if row["label"] == "after")
            if boundary_slot is None:
                boundary_slot = inside_slots[-1] if inside_slots else None
            if boundary_slot is None:
                break
            boundary_index = int(candidate_indices[int(boundary_slot)])
            best_index = boundary_index
            next_low = boundary_index
            next_high = current_high
            after_indices = [int(candidate_indices[slot]) for slot in after_slots if int(candidate_indices[slot]) > boundary_index]
            if after_indices:
                next_high = min(after_indices)
        history.append(
            {
                "round_index": int(round_index),
                "candidate_indices": [int(value) for value in candidate_indices],
                "candidate_frame_indices": [int(sampled_entries[index]["frame_index"]) for index in candidate_indices],
                "boundary_slot": None if boundary_slot is None else int(boundary_slot),
                "boundary_index": int(boundary_index),
                "boundary_frame_index": int(sampled_entries[boundary_index]["frame_index"]),
                "frame_labels": label_rows,
                "notes": normalized["notes"],
                "raw_output": raw_output,
            }
        )
        if next_low == current_low and next_high == current_high:
            break
        if next_high <= next_low:
            current_low, current_high = next_low, next_high
            break
        current_low, current_high = next_low, next_high
        if current_high - current_low <= 1:
            break
    return {
        "boundary_index": best_index,
        "rounds": history,
        "final_interval": {"low_index": int(current_low), "high_index": int(current_high)},
    }


class QwenQueryPlanner:
    def __init__(self, model_name_or_path: str | Path):
        transformers = _import_transformers()
        processor_cls = getattr(transformers, "AutoProcessor", None)
        if processor_cls is None:
            raise RuntimeError("transformers.AutoProcessor is unavailable.")

        model_cls = None
        for candidate in (
            "Qwen3VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen2VLForConditionalGeneration",
        ):
            model_cls = getattr(transformers, candidate, None)
            if model_cls is not None:
                break
        if model_cls is None:
            raise RuntimeError("Unable to find a compatible Qwen vision-language model class.")

        self.processor = processor_cls.from_pretrained(str(model_name_or_path), trust_remote_code=True)
        self.model = model_cls.from_pretrained(
            str(model_name_or_path),
            **_qwen_model_load_kwargs(),
        )

    def generate_json(self, prompt: str, images: list[Image.Image] | None = None) -> tuple[dict[str, Any], str]:
        images = images or []
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in images]
                + [{"type": "text", "text": prompt}],
            }
        ]
        if hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt

        processor_kwargs: dict[str, Any] = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt",
        }
        if images:
            processor_kwargs["images"] = images
        model_inputs = self.processor(**processor_kwargs)
        model_inputs = {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }
        generated = self.model.generate(
            **model_inputs,
            max_new_tokens=1024,
            do_sample=False,
        )
        trimmed = generated[:, model_inputs["input_ids"].shape[1] :]
        output = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return _extract_first_json(output), output


def plan_query_entities(
    query: str,
    dataset_dir: str | Path,
    output_path: str | Path | None = None,
    qwen_model: str | None = None,
    frame_subsample_stride: int = 10,
    num_sampled_frames: int = 9,
    num_boundary_frames: int = 15,
    strict: bool = True,
) -> dict[str, Any]:
    query = " ".join(str(query).strip().split())
    if not query:
        raise ValueError("query must be non-empty")
    dataset_dir = Path(dataset_dir)
    semantic_profile = _query_semantic_profile(query)

    teacher = QwenQueryPlanner(_resolve_qwen_model(qwen_model))
    sampled_entries = _load_subsampled_entries(dataset_dir=dataset_dir, frame_subsample_stride=frame_subsample_stride)
    sampled_lookup = _entry_index_lookup(sampled_entries)

    context_entries = _sample_context_entries(sampled_entries, num_sampled_frames=num_sampled_frames)
    context_images = _load_images_for_entries(context_entries)
    prompt = QUERY_PLAN_TEMPLATE.format(
        query=query,
        frame_summary=_frame_summary(context_entries),
    )
    raw_payload, raw_output = teacher.generate_json(prompt=prompt, images=context_images)
    plan = _normalize_plan(raw_payload, query=query, strict=bool(strict))
    plan["planner_mode"] = "qwen_vision_strict" if strict else "qwen_vision"
    plan["qwen_enabled"] = True
    plan["query_semantic_profile"] = semantic_profile
    plan["raw_output"] = raw_output
    plan["dataset_dir"] = str(dataset_dir)
    plan["frame_subsample_stride"] = int(frame_subsample_stride)
    plan["num_context_frames"] = int(len(context_entries))
    plan["context_frames"] = [
        {
            "frame_index": int(entry["frame_index"]),
            "image_id": str(entry["image_id"]),
            "time_value": float(entry["time_value"]),
            "image_path": str(entry["image_path"]),
        }
        for entry in context_entries
    ]

    boundary_entries = _sample_context_entries(
        sampled_entries,
        num_sampled_frames=max(int(num_boundary_frames), int(num_sampled_frames)),
    )
    boundary_images = _load_images_for_entries(boundary_entries)
    boundary_prompt = TEMPORAL_WINDOW_TEMPLATE.format(
        query=query,
        subject_phrases=json.dumps(plan["query_subject_phrases"], ensure_ascii=False),
        frame_summary=_frame_summary(boundary_entries),
    )
    boundary_raw_payload, boundary_raw_output = teacher.generate_json(prompt=boundary_prompt, images=boundary_images)
    boundary_plan = _normalize_temporal_window(boundary_raw_payload, frame_count=len(boundary_entries))
    coarse_start_index = _coarse_index_for_slot(
        boundary_plan["start_slot"],
        sampled_entries=sampled_entries,
        coarse_entries=boundary_entries,
        lookup=sampled_lookup,
    )
    coarse_end_index = _coarse_index_for_slot(
        boundary_plan["end_slot"],
        sampled_entries=sampled_entries,
        coarse_entries=boundary_entries,
        lookup=sampled_lookup,
    )
    plan["boundary_mode"] = "qwen_temporal_window"
    plan["boundary_num_context_frames"] = int(len(boundary_entries))
    plan["boundary_context_frames"] = [
        {
            "slot": int(index),
            "frame_index": int(entry["frame_index"]),
            "image_id": str(entry["image_id"]),
            "time_value": float(entry["time_value"]),
            "image_path": str(entry["image_path"]),
        }
        for index, entry in enumerate(boundary_entries)
    ]
    plan["coarse_temporal_window"] = {
        "start_slot": boundary_plan["start_slot"],
        "end_slot": boundary_plan["end_slot"],
        "start_sample_index": None if coarse_start_index is None else int(coarse_start_index),
        "end_sample_index": None if coarse_end_index is None else int(coarse_end_index),
        "start_frame_index": None if boundary_plan["start_slot"] is None else int(boundary_entries[int(boundary_plan["start_slot"])]["frame_index"]),
        "end_frame_index": None if boundary_plan["end_slot"] is None else int(boundary_entries[int(boundary_plan["end_slot"])]["frame_index"]),
        "frame_labels": boundary_plan["frame_labels"],
        "notes": boundary_plan["notes"],
        "raw_output": boundary_raw_output,
    }
    refined_start = None
    refined_end = None
    need_start_refine = True
    need_end_refine = True
    if semantic_profile["asks_before_state"]:
        need_start_refine = False
    if semantic_profile["asks_after_state"]:
        need_end_refine = False
    if not semantic_profile["asks_before_state"] and not semantic_profile["asks_after_state"] and not semantic_profile["asks_action_window"] and not plan.get("query_successor_phrases"):
        need_start_refine = False
        need_end_refine = False
    if sampled_entries and need_start_refine:
        start_low, start_high = _boundary_search_interval(
            boundary_kind="start",
            coarse_start_index=coarse_start_index,
            coarse_end_index=coarse_end_index,
            frame_count=len(sampled_entries),
        )
        refined_start = _refine_boundary_interval(
            teacher=teacher,
            query=query,
            subject_phrases=plan["query_subject_phrases"],
            boundary_kind="start",
            boundary_condition=plan["start_condition"],
            sampled_entries=sampled_entries,
            low_index=start_low,
            high_index=start_high,
        )
    if sampled_entries and need_end_refine:
        end_low, end_high = _boundary_search_interval(
            boundary_kind="end",
            coarse_start_index=coarse_start_index,
            coarse_end_index=coarse_end_index,
            frame_count=len(sampled_entries),
        )
        refined_end = _refine_boundary_interval(
            teacher=teacher,
            query=query,
            subject_phrases=plan["query_subject_phrases"],
            boundary_kind="end",
            boundary_condition=plan["stop_condition"],
            sampled_entries=sampled_entries,
            low_index=end_low,
            high_index=end_high,
        )
    refined_start_index = refined_start["boundary_index"] if refined_start and refined_start.get("boundary_index") is not None else coarse_start_index
    refined_end_index = refined_end["boundary_index"] if refined_end and refined_end.get("boundary_index") is not None else coarse_end_index
    refined_start_index, refined_end_index = _finalize_temporal_window(
        query=query,
        frame_count=len(sampled_entries),
        coarse_start_index=coarse_start_index,
        coarse_end_index=coarse_end_index,
        refined_start_index=refined_start_index,
        refined_end_index=refined_end_index,
    )
    plan["temporal_refinement"] = {
        "start": refined_start,
        "end": refined_end,
    }
    plan["refined_temporal_window"] = {
        "start_sample_index": None if refined_start_index is None else int(refined_start_index),
        "end_sample_index": None if refined_end_index is None else int(refined_end_index),
        "start_frame_index": None if refined_start_index is None else int(sampled_entries[int(refined_start_index)]["frame_index"]),
        "end_frame_index": None if refined_end_index is None else int(sampled_entries[int(refined_end_index)]["frame_index"]),
        "start_time_value": None if refined_start_index is None else float(sampled_entries[int(refined_start_index)]["time_value"]),
        "end_time_value": None if refined_end_index is None else float(sampled_entries[int(refined_end_index)]["time_value"]),
    }
    plan["phase_transition_hints"] = _derive_transition_hints_from_window(
        plan=plan,
        window_plan=plan["refined_temporal_window"],
        sampled_entries=sampled_entries,
    )

    if output_path is not None:
        _write_json(Path(output_path), plan)
    return plan
