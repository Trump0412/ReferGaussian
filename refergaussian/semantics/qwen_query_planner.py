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


def _default_qwen_model_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_paths = [os.environ.get("REFERGAUSSIAN_QWEN_MODEL")]
    for raw_path in env_paths:
        if not raw_path:
            continue
        candidates.append(Path(raw_path))

    candidates.extend(
        [
            _PROJECT_ROOT / "models" / "Qwen3-VL-8B-Instruct",
            _PROJECT_ROOT.parent / "models" / "Qwen3-VL-8B-Instruct",
            Path.home() / "models" / "Qwen3-VL-8B-Instruct",
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


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
- primary_subject_phrases: array of the exact entity or entities requested by the query, excluding action context
- query_subject_phrases: array of short static noun phrases for the primary query objects only
- required_identity_attributes: array of visible identity attributes that must be true of the requested entity for a non-empty answer
- temporal_state_attributes: array of query attributes that describe a time-varying state rather than persistent identity
- query_successor_phrases: array of short static noun phrases that appear only after a query-driven state change, such as "object fragments"
- phase_transition_hints: array of objects, each with keys {{phrase, last_pre_change_slot, first_post_change_slot, reason}}
- detector_phrases: array of short static noun phrases to detect and track
- optional_phrases: array of extra visible nouns that are not query subjects
- interaction_phrase: short phrase describing the interaction
- start_condition: short phrase describing when the query event should begin
- stop_condition: short phrase describing when the query event should stop
- temporal_hints: array of short phase descriptions
- must_track_phrases: array of phrases that must be tracked through time
- action_window_hint: short phrase for the action/contact interval, or empty string if not applicable
- support_window_hint: short phrase for the full object/state support interval that should be scored
- absent_query: boolean, true only when the queried object/state is absent from the scene
- preferred_detector: one of grounding_dino, grounded_sam2
- notes: short string

Rules:
- The sampled frames are for whole-video inventory and subject discovery, not for forcing the exact action frame.
- Use the sampled frames to infer which objects exist in the full video, then plan only the query-relevant subjects and successor states.
- Keep the output compact: video_inventory_phrases <= 8, query_subject_phrases <= 3, query_successor_phrases <= 2, detector_phrases <= 4, optional_phrases <= 6, temporal_hints <= 4.
- Use English-only phrase outputs for all phrase lists and conditions. Do not output Chinese phrases.
- Do not use role words like patient/tool/agent in the phrase lists.
- Keep phrases concrete and static, such as "tool", "object", "surface", "object fragments".
- `video_inventory_phrases` should summarize the main objects visible across the whole video.
- `primary_subject_phrases` must name the grammatical referent requested by the query. For a singular
  relational or action query such as "the X while it acts on Y", include only X; Y is context unless the
  query explicitly asks for both entities or for a set.
- Preserve an explicit requested cardinality in `primary_subject_phrases` and `query_subject_phrases`.
  Keep the original quantifier and count rather than silently reducing a plural or counted referent to a
  single generic noun. The count belongs to the semantic query, even when the detector later uses a
  count-neutral noun phrase.
- `query_subject_phrases` should contain only the minimum nouns needed to answer the query.
- `required_identity_attributes` must preserve every non-temporal visual identity constraint in the query that distinguishes
  whether the referent exists, including literal color, material, texture, or permanent-appearance terms. Do not put
  before/after relations, transient state changes, or motion terms there; those describe an entity's lifecycle.
- `temporal_state_attributes` must contain only state conditions that identify a phase of an existing entity. Never put
  a category head, color, material, or other persistent identity attribute in this field.
- If a requested identity attribute is absent, treat this as a ZERO / DISTRACTOR QUERY. Do not silently replace it with
  the closest object of the same broad category.
- `detector_phrases` should normally equal `query_subject_phrases + query_successor_phrases`, and not include unrelated context objects.
- Do not include non-subject context objects in `detector_phrases` unless they are truly required by the query itself.
- Only use `query_successor_phrases` when the action creates a new stable object state, such as "object fragments".
- Do not invent count-based successor phrases when downstream mask tracking can discover the split directly.
- `phase_transition_hints` should use the 0-based sampled-frame slot indices from the observed frame list.
- Use `phase_transition_hints` only when the query implies a subject state transition or a before/after distinction.
- Prefer the earliest semantic change point suggested by the sampled frames, not the latest frame where the object is already fully separated.
- For an action query, keep only the requested entity or explicitly requested entity set as query subjects.
  Interaction partners may appear in the inventory or optional list, but must not be promoted to subjects merely
  because they participate in the action.
- For action queries, `start_condition` should begin at direct task-relevant contact, not at coarse pre-contact setup.
- For action queries, `stop_condition` should end when the query-driven state change stabilizes, not when unrelated context remains visible.
- For action queries, separate the short action/contact interval from the longer object/state support interval.
  The action interval identifies the entity; the support interval is where the final query mask should be active.
- If the query implies temporal change, include before/during/after hints.
- For exclusion queries such as "everything except the named object", `query_subject_phrases` and `detector_phrases`
  should focus on the INCLUDED objects, not the excluded object.
- For set queries such as "all objects that remain stationary", `detector_phrases` may include multiple visible
  objects so downstream stages can filter the correct subset.
- For count-based set queries, preserve the requested entity phrase in `query_subject_phrases` and let the
  detector use an additional count-neutral noun phrase when needed.
- preferred_detector must be "grounded_sam2".
- **ZERO / DISTRACTOR QUERY**: If the queried object does NOT exist in this scene at all
  after checking both its category and every persistent identity constraint, you MUST set query_subject_phrases to [],
  detector_phrases to [], absent_query to true, and add a notes field starting with
  "ZERO_QUERY: <reason>".
  Do NOT hallucinate a closest-match object. Empty detection is the correct answer.
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
- Keep all labels and reasons in concise English.
- Keep frame_labels concise and notes short.
- For full-object queries, mark the full lifetime where that queried object/state should count.
- For intact-state queries, stop before the object enters the changed state.
- For post-change queries, start when the object first clearly belongs to the changed state.
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
- Keep all labels and reasons in concise English.
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
    candidates = _default_qwen_model_candidates()
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Unable to resolve a local Qwen model. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
        + ". Pass an explicit path or set REFERGAUSSIAN_QWEN_MODEL."
    )


def _import_transformers():
    try:
        import transformers  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Python dependency 'transformers'. Install it in the query-planning environment."
        ) from exc
    return transformers


def _qwen_env_int(name: str, default: int, minimum: int = 0, maximum: int | None = None) -> int:
    """Read a bounded planner runtime setting without changing model semantics."""
    try:
        value = int(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        value = int(default)
    value = max(int(minimum), value)
    if maximum is not None:
        value = min(value, int(maximum))
    return int(value)


def _qwen_gpu_memory_budget_gib(free_gib: int) -> int:
    """Reserve headroom while allowing a single planner to use a large GPU fully."""
    reserve_gib = _qwen_env_int("REFERGAUSSIAN_QWEN_GPU_RESERVE_GIB", 4, minimum=0)
    configured_max_gib = _qwen_env_int("REFERGAUSSIAN_QWEN_GPU_MAX_GIB", 0, minimum=0)
    budget_gib = max(int(free_gib) - reserve_gib, 8)
    if configured_max_gib > 0:
        budget_gib = min(budget_gib, configured_max_gib)
    return int(budget_gib)


def _qwen_max_new_tokens() -> int:
    """Keep the established output budget by default, with an explicit runtime override."""
    return _qwen_env_int(
        "REFERGAUSSIAN_QWEN_MAX_NEW_TOKENS",
        1024,
        minimum=128,
        maximum=4096,
    )


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
            gpu_budget = _qwen_gpu_memory_budget_gib(free_gib)
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
    asks_set = bool(tokens & set_keywords) or any(
        marker in query_norm
        for marker in (
            "all objects",
            "everything except",
            "except the",
            "excluding the",
        )
    )
    action_context_markers = ("while", "during", "in the process of")
    return {
        "query_norm": query_norm,
        "asks_before_state": bool(tokens & intact_keywords),
        "asks_after_state": bool(tokens & changed_keywords),
        "asks_action_window": bool(tokens & action_keywords) or any(
            marker in query_norm for marker in action_context_markers
        ),
        "asks_set": asks_set,
    }


def _is_exclusion_query_text(query_norm: str) -> bool:
    text = " ".join(str(query_norm).strip().lower().split())
    return any(
        marker in text
        for marker in (
            "everything except",
            "all objects except",
            "except the",
            "excluding the",
            "other than the",
        )
    )


def _is_static_set_query_text(query_norm: str) -> bool:
    text = " ".join(str(query_norm).strip().lower().split())
    return any(
        marker in text
        for marker in (
            "always stationary",
            "always static",
            "remain stationary throughout",
            "physically stationary throughout",
            "never move",
            "stationary throughout the video",
        )
    )


def _extract_exclusion_phrases(query_norm: str) -> list[str]:
    text = " ".join(str(query_norm).strip().lower().split())
    english_patterns = [
        r"(?:everything|all objects?)\s+except\s+(.+)",
        r"except\s+(.+)",
        r"(?:excluding|other than|besides)\s+(.+)",
    ]
    extracted: list[str] = []
    for pattern in english_patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        tail = match.group(1)
        parts = re.split(r",| and | or |/", tail)
        for part in parts:
            phrase = _canonicalize_phrase(part)
            if phrase:
                extracted.append(phrase)
    return _normalize_phrase_list(extracted)


def _state_detector_phrase_additions(query: str, base_phrases: list[str]) -> list[str]:
    """Generate state-aware detector phrases without scene-specific vocabularies.

    Grounding models often respond better to a phrase that includes the state
    requested by the query.  Build those phrases compositionally from the
    query and the planner's own object phrases, rather than keeping a list of
    benchmark objects or scenes.
    """
    query_norm = " ".join(
        str(query).strip().lower().replace("_", " ").replace("-", " ").split()
    )
    query_tokens = re.findall(r"[a-z]+", query_norm)
    base_set = {_canonicalize_phrase(phrase) for phrase in base_phrases if str(phrase).strip()}
    additions: list[str] = []

    def add(*phrases: str) -> None:
        for phrase in phrases:
            cleaned = " ".join(str(phrase).strip().lower().split())
            if cleaned:
                additions.append(cleaned)

    state_aliases = {
        "opened": "opened",
        "open": "open",
        "closed": "closed",
        "close": "closed",
        "shut": "shut",
        "empty": "empty",
        "full": "full",
        "filled": "filled",
        "complete": "complete",
        "whole": "whole",
        "intact": "intact",
        "unbroken": "unbroken",
        "broken": "broken",
        "split": "split",
        "cracked": "cracked",
        "cut": "cut",
        "sliced": "sliced",
        "melted": "melted",
        "melting": "melting",
        "dark": "dark",
        "darker": "dark",
        "light": "light",
        "lighter": "light",
        "brown": "brown",
        "red": "red",
        "blue": "blue",
        "green": "green",
        "white": "white",
        "black": "black",
    }
    state_words = [state_aliases[token] for token in query_tokens if token in state_aliases]
    state_words = list(dict.fromkeys(state_words))
    has_pieces = any(token in {"piece", "pieces"} for token in query_tokens)
    has_halves = any(token in {"half", "halves"} for token in query_tokens)

    for base_phrase in sorted(base_set):
        for state in state_words:
            add(f"{state} {base_phrase}")
        if has_pieces:
            for state in state_words:
                add(f"{state} {base_phrase} pieces")
            add(f"{base_phrase} pieces")
        if has_halves:
            for state in state_words:
                add(f"{state} {base_phrase} halves")
            add(f"{base_phrase} halves")

    # Preserve state phrases whose head noun is supplied by the query itself,
    # e.g. "closed plastic container" or "light colored liquid".
    direct_state_pattern = re.compile(
        r"\b(?:opened|open|closed|shut|empty|full|filled|complete|whole|intact|"
        r"unbroken|broken|split|cracked|cut|sliced|melted|melting|dark|darker|"
        r"light|lighter|brown|red|blue|green|white|black)\s+"
        r"(?:[a-z]+\s+){0,2}[a-z]+\b"
    )
    base_heads = {phrase.split()[-1] for phrase in base_set if phrase.split()}
    for match in direct_state_pattern.finditer(query_norm):
        candidate = " ".join(match.group(0).split())
        if candidate.split()[-1] in base_heads:
            add(candidate)

    midpoint_match = re.search(r"\b([a-z]+)\s+above\s+(?:the\s+)?midpoint\b", query_norm)
    if midpoint_match:
        add(f"{midpoint_match.group(1)} above midpoint")

    # State carriers such as liquid are query-derived rather than benchmark
    # specific. They help when the visible carrier is distinct from its
    # container or support object.
    carrier_terms = [token for token in query_tokens if token in {"liquid", "contents", "material"}]
    for carrier in dict.fromkeys(carrier_terms):
        for state in state_words:
            add(f"{state} {carrier}")

    merged: list[str] = []
    seen: set[str] = set()
    for phrase in additions:
        if phrase in seen:
            continue
        seen.add(phrase)
        merged.append(phrase)
    return merged


def _normalize_phrase_list(values: Any) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        phrase = _canonicalize_phrase(value)
        if not phrase:
            continue
        if phrase in seen:
            continue
        seen.add(phrase)
        normalized.append(phrase)
    return normalized


def _canonicalize_phrase(value: Any) -> str:
    """Normalize English typography without object-specific aliases."""
    phrase = " ".join(
        str(value).strip().lower().replace("_", " ").replace("-", " ").split()
    )
    if not phrase:
        return ""
    phrase = phrase.strip(" .,!?:;\"'()[]{}")
    for prefix in ("the ", "a ", "an "):
        if phrase.startswith(prefix):
            phrase = phrase[len(prefix):].strip()
    phrase = phrase.strip(" .,!?:;\"'()[]{}")
    return phrase


def _phrase_token_set(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _canonicalize_phrase(value)))


def _attribute_is_covered_by_phrase(attribute: str, phrase: str) -> bool:
    attribute_tokens = _phrase_token_set(attribute)
    phrase_tokens = _phrase_token_set(phrase)
    return bool(attribute_tokens) and attribute_tokens.issubset(phrase_tokens)


def _filter_identity_attributes(
    *,
    identity_attributes: list[str],
    temporal_state_attributes: list[str],
    query_subject_phrases: list[str],
    temporal_hints: list[str],
    query_profile: dict[str, Any],
) -> tuple[list[str], list[dict[str, str]]]:
    """Keep persistent attributes while removing syntactically grounded state labels.

    This is intentionally vocabulary-free. The vision planner declares the two
    scopes, while the normalizer removes category heads and, for explicitly
    temporal queries, resolves a contradictory declaration from an ``initial
    ... state`` or ``final ... phase`` hint. That makes the gate conservative
    without encoding scene, object, color, or action names.
    """
    subject_tokens: set[str] = set()
    for phrase in query_subject_phrases:
        subject_tokens.update(_phrase_token_set(phrase))
    temporal_anchors = ("initial", "final", "state", "phase", "before", "after", "during", "process")
    temporal_hint_phrases = [
        phrase
        for phrase in temporal_hints
        if any(anchor in _phrase_token_set(phrase) for anchor in temporal_anchors)
    ]
    is_temporal_query = bool(
        query_profile.get("asks_before_state")
        or query_profile.get("asks_after_state")
        or query_profile.get("asks_action_window")
    )
    kept: list[str] = []
    filtered: list[dict[str, str]] = []
    for attribute in identity_attributes:
        attribute_tokens = _phrase_token_set(attribute)
        if attribute_tokens and attribute_tokens.issubset(subject_tokens):
            filtered.append({"attribute": attribute, "reason": "subject_head"})
            continue
        if any(_attribute_is_covered_by_phrase(attribute, phrase) for phrase in temporal_state_attributes):
            filtered.append({"attribute": attribute, "reason": "planner_temporal_state"})
            continue
        if is_temporal_query and any(
            _attribute_is_covered_by_phrase(attribute, phrase) for phrase in temporal_hint_phrases
        ):
            filtered.append({"attribute": attribute, "reason": "temporal_hint"})
            continue
        kept.append(attribute)
    return kept, filtered


def _singularize_counted_token(token: str) -> str:
    """Provide a conservative English singular form only for count phrases."""
    if not token.isalpha() or len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "sses", "xes", "zes", "ses")):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _count_neutral_detector_phrases(phrases: list[str], *, limit: int | None = None) -> list[str]:
    """Add generic detector variants for plural/count-based entity requests.

    The query subject itself is retained verbatim for semantic selection.  The
    additional variants simply make open-vocabulary detection robust when a
    model does not understand quantifiers such as ``both`` or ``two``.
    """
    expanded: list[str] = []
    for phrase in phrases:
        normalized = _canonicalize_phrase(phrase)
        if not normalized:
            continue
        expanded.append(normalized)
        tokens = normalized.split()
        start = 0
        if tokens[:1] and tokens[0] in {"both", "two", "three"}:
            start = 1
        elif tokens[:3] == ["a", "pair", "of"]:
            start = 3
        elif tokens[:2] == ["pair", "of"]:
            start = 2
        if start >= len(tokens):
            continue
        if start:
            count_neutral = tokens[start:]
            expanded.append(" ".join(count_neutral))
            singular = count_neutral[:]
            singular[-1] = _singularize_counted_token(singular[-1])
            expanded.append(" ".join(singular))
    output = _normalize_phrase_list(expanded)
    if limit is not None:
        output = output[: int(limit)]
    return output


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


def _phrase_token_span(text: str, phrase: str) -> int | None:
    """Return the first whole-token occurrence of an entity phrase in text."""
    text_tokens = re.findall(r"[a-z]+", str(text).lower())
    phrase_tokens = re.findall(r"[a-z]+", _canonicalize_phrase(phrase))
    if not phrase_tokens or len(phrase_tokens) > len(text_tokens):
        return None
    width = len(phrase_tokens)
    for index in range(len(text_tokens) - width + 1):
        if text_tokens[index : index + width] == phrase_tokens:
            return index
    return None


def _leading_query_subject_phrases(query: str, candidates: list[str]) -> list[str]:
    """Recover a singular English query referent when a planner includes context.

    This is deliberately syntactic rather than vocabulary-driven. It only
    constrains a query's leading noun phrase before a temporal/relative clause,
    so it applies equally to unseen objects and actions.
    """
    query_text = " ".join(str(query).strip().lower().replace("_", " ").split())
    clause = re.search(
        r"\b(?:while|when|as|before|after|during|until|once|that|which|who)\b",
        query_text,
    )
    prefix = query_text[: clause.start()] if clause else query_text
    matches: list[tuple[int, str]] = []
    for phrase in candidates:
        position = _phrase_token_span(prefix, phrase)
        if position is not None:
            matches.append((position, phrase))
    if not matches:
        return []

    matches.sort(key=lambda item: (item[0], -len(item[1].split()), item[1]))
    has_explicit_set = bool(re.search(r"\b(?:both|all|each)\b", prefix))
    has_coordinated_subjects = " and " in prefix and len(matches) > 1
    if has_explicit_set or has_coordinated_subjects:
        return [phrase for _, phrase in matches]
    return [matches[0][1]]


def _phrases_overlap_as_one_referent(first: str, second: str) -> bool:
    """Avoid treating a shortened or extended subject phrase as relation context."""
    first_tokens = {
        _singularize_counted_token(token)
        for token in re.findall(r"[a-z0-9]+", _canonicalize_phrase(first))
    }
    second_tokens = {
        _singularize_counted_token(token)
        for token in re.findall(r"[a-z0-9]+", _canonicalize_phrase(second))
    }
    if not first_tokens or not second_tokens:
        return False
    return first_tokens.issubset(second_tokens) or second_tokens.issubset(first_tokens)


def _relation_context_phrases(
    query: str,
    *,
    query_subject_phrases: list[str],
    raw_primary_subject_phrases: list[str],
) -> list[str]:
    """Retain explicitly named interaction context without promoting it to a subject.

    A vision planner may initially list the agent, tool, and acted-on object as
    ``primary_subject_phrases``.  The grammatical normalizer correctly narrows a
    singular query to its leading referent, but the discarded entities are still
    useful for deciding between visually distinct instances of that referent.
    This keeps only terms stated in the query itself; it never adds a scene or
    benchmark-specific vocabulary.
    """
    if len(query_subject_phrases) != 1 or len(raw_primary_subject_phrases) < 2:
        return []
    subject = query_subject_phrases[0]
    context: list[str] = []
    for phrase in raw_primary_subject_phrases:
        if _phrases_overlap_as_one_referent(phrase, subject):
            continue
        if _phrase_token_span(query, phrase) is None:
            continue
        context.append(phrase)
    return _normalize_phrase_list(context)


def _leading_counted_subject_spec(query: str) -> tuple[str, int] | None:
    """Recover a leading English counted subject that a planner may simplify.

    The vision planner is allowed to use compact noun phrases, but an explicit
    cardinality in the original question is a semantic constraint rather than
    optional wording.  This grammar-only recovery deliberately accepts just a
    leading noun phrase and rejects coordinated sets, which already have an
    explicit multi-subject representation in the plan.
    """
    query_text = " ".join(str(query).strip().lower().replace("_", " ").split())
    clause = re.search(
        r"\b(?:while|when|as|before|after|during|until|once|that|which|who)\b",
        query_text,
    )
    prefix = query_text[: clause.start()] if clause else query_text
    prefix = re.split(r"\b(?:is|are|was|were|has|have|had|does|do|did|will|can|should)\b", prefix, maxsplit=1)[0]
    prefix = _canonicalize_phrase(prefix)
    if prefix.startswith("the "):
        prefix = prefix[4:].strip()
    if " and " in prefix or " or " in prefix:
        return None

    match = re.fullmatch(r"(?P<count>both|two|three)\s+(?:(?:of\s+)?(?:the\s+)?)?(?P<subject>[a-z0-9][a-z0-9 ]*)", prefix)
    if match:
        subject = _canonicalize_phrase(match.group("subject"))
        if subject:
            count_word = str(match.group("count"))
            return (f"{count_word} {subject}", {"both": 2, "two": 2, "three": 3}[count_word])

    pair_match = re.fullmatch(r"(?:(?:a\s+)?pair\s+of)\s+(?:(?:the\s+)?)?(?P<subject>[a-z0-9][a-z0-9 ]*)", prefix)
    if pair_match:
        subject = _canonicalize_phrase(pair_match.group("subject"))
        if subject:
            return (f"pair of {subject}", 2)
    return None


def _counted_subject_matches_plan(counted_phrase: str, planned_phrases: list[str]) -> bool:
    """Check noun-head agreement without relying on an object vocabulary."""
    counted_tokens = re.findall(r"[a-z0-9]+", _canonicalize_phrase(counted_phrase))
    if not counted_tokens:
        return False
    counted_head = _singularize_counted_token(counted_tokens[-1])
    for phrase in planned_phrases:
        tokens = re.findall(r"[a-z0-9]+", _canonicalize_phrase(phrase))
        if tokens and _singularize_counted_token(tokens[-1]) == counted_head:
            return True
    return False


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
    # Set and exclusion questions may legitimately refer to every visible
    # entity. Keep their full compact planner list until query semantics decide
    # whether the ordinary singular/multi-target cap applies.
    query_subject_phrases = _normalize_phrase_list(raw_payload.get("query_subject_phrases", []))[:8]
    primary_subject_phrases = _normalize_phrase_list(raw_payload.get("primary_subject_phrases", []))[:8]
    raw_primary_subject_phrases = primary_subject_phrases[:]
    query_successor_phrases = _normalize_phrase_list(raw_payload.get("query_successor_phrases", []))[:2]
    raw_identity_attributes = raw_payload.get("required_identity_attributes", [])
    if isinstance(raw_identity_attributes, str):
        raw_identity_attributes = [raw_identity_attributes]
    required_identity_attributes = _normalize_phrase_list(raw_identity_attributes)[:4]
    raw_temporal_state_attributes = raw_payload.get("temporal_state_attributes", [])
    if isinstance(raw_temporal_state_attributes, str):
        raw_temporal_state_attributes = [raw_temporal_state_attributes]
    temporal_state_attributes = _normalize_phrase_list(raw_temporal_state_attributes)[:4]
    raw_optional_phrases = _normalize_phrase_list(raw_payload.get("optional_phrases", []))[:6]
    must_track_phrases = _normalize_phrase_list(raw_payload.get("must_track_phrases", []))[:3]
    temporal_hints = _normalize_phrase_list(raw_payload.get("temporal_hints", []))[:4]
    interaction_phrase = " ".join(str(raw_payload.get("interaction_phrase", query)).strip().split())
    start_condition = " ".join(str(raw_payload.get("start_condition", "")).strip().split())
    stop_condition = " ".join(str(raw_payload.get("stop_condition", "")).strip().split())
    preferred_detector = str(raw_payload.get("preferred_detector", "grounded_sam2")).strip().lower()
    notes = " ".join(str(raw_payload.get("notes", "")).strip().split())
    absent_query = bool(raw_payload.get("absent_query", False) or raw_payload.get("empty_query", False) or raw_payload.get("zero_query", False))
    if notes.upper().startswith("ZERO_QUERY"):
        absent_query = True
    action_window_hint = " ".join(str(raw_payload.get("action_window_hint", "")).strip().split())
    support_window_hint = " ".join(str(raw_payload.get("support_window_hint", "")).strip().split())

    if strict and not absent_query and not video_inventory_phrases:
        raise ValueError("Strict Qwen planner returned no video_inventory_phrases.")
    if strict and not absent_query and not query_subject_phrases:
        raise ValueError("Strict Qwen planner returned no query_subject_phrases.")
    if not strict and not video_inventory_phrases:
        video_inventory_phrases = _normalize_phrase_list(raw_payload.get("detector_phrases", []))
    if not strict and not query_subject_phrases:
        query_subject_phrases = _normalize_phrase_list(raw_payload.get("detector_phrases", []))[:2]
    if not strict and not video_inventory_phrases:
        video_inventory_phrases = query_subject_phrases[:]
    query_profile = _query_semantic_profile(query)
    is_exclusion_query = _is_exclusion_query_text(query_profile["query_norm"])
    is_static_set_query = _is_static_set_query_text(query_profile["query_norm"])
    if not is_exclusion_query and not query_profile["asks_set"]:
        # The planner can put interaction context into primary_subject_phrases.
        # Preserve an explicit multi-subject request, but otherwise let the
        # grammatical leading noun phrase decide the requested entity whether
        # it originated from primary_subject_phrases or query_subject_phrases.
        planned_subjects = primary_subject_phrases or query_subject_phrases
        leading_subjects = _leading_query_subject_phrases(query, planned_subjects)
        if leading_subjects:
            query_subject_phrases = leading_subjects
        elif primary_subject_phrases:
            query_subject_phrases = primary_subject_phrases[:3]
    elif primary_subject_phrases and not is_exclusion_query:
        query_subject_phrases = primary_subject_phrases
    counted_subject_spec = None if is_exclusion_query else _leading_counted_subject_spec(query)
    if counted_subject_spec is not None:
        counted_subject_phrase, _ = counted_subject_spec
        if _counted_subject_matches_plan(counted_subject_phrase, query_subject_phrases):
            query_subject_phrases = [counted_subject_phrase]
    primary_subject_phrases = query_subject_phrases[:]
    required_identity_attributes, identity_attribute_filter = _filter_identity_attributes(
        identity_attributes=required_identity_attributes,
        temporal_state_attributes=temporal_state_attributes,
        query_subject_phrases=query_subject_phrases,
        temporal_hints=temporal_hints,
        query_profile=query_profile,
    )
    relation_context_phrases = _relation_context_phrases(
        query,
        query_subject_phrases=query_subject_phrases,
        raw_primary_subject_phrases=raw_primary_subject_phrases,
    )

    state_detector_phrases = _state_detector_phrase_additions(
        query,
        _merge_unique_phrases(query_subject_phrases, query_successor_phrases),
    )
    detector_base_phrases = _count_neutral_detector_phrases(
        _merge_unique_phrases(
            query_subject_phrases,
            relation_context_phrases,
            query_successor_phrases,
        ),
        limit=6,
    )
    detector_phrases = _merge_unique_phrases(detector_base_phrases, state_detector_phrases)[:6]
    if strict and not absent_query and not detector_phrases:
        raise ValueError("Strict Qwen planner produced no detector_phrases after subject filtering.")
    if not strict and not detector_phrases:
        detector_phrases = _normalize_phrase_list(raw_payload.get("detector_phrases", []))
    if not strict and not detector_phrases:
        detector_phrases = query_subject_phrases[:]

    optional_phrases = [phrase for phrase in video_inventory_phrases if phrase not in detector_phrases]
    optional_phrases = _merge_unique_phrases(optional_phrases, [phrase for phrase in raw_optional_phrases if phrase not in detector_phrases])[:6]
    must_track_phrases = [phrase for phrase in query_subject_phrases if phrase in must_track_phrases] or query_subject_phrases[:]
    must_track_phrases = _count_neutral_detector_phrases(must_track_phrases, limit=4)
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

    exclusion_phrases = set(_extract_exclusion_phrases(query_profile["query_norm"]))

    if is_exclusion_query or is_static_set_query or query_profile["asks_set"]:
        broad_candidates = _merge_unique_phrases(
            query_subject_phrases,
            detector_phrases,
            optional_phrases,
            video_inventory_phrases,
        )
        if exclusion_phrases:
            broad_candidates = [
                phrase for phrase in broad_candidates if _canonicalize_phrase(phrase) not in exclusion_phrases
            ]
            query_subject_phrases = [
                phrase for phrase in query_subject_phrases if _canonicalize_phrase(phrase) not in exclusion_phrases
            ]
        detector_limit = 8 if (is_exclusion_query or is_static_set_query) else 6
        detector_phrases = broad_candidates[:detector_limit] or detector_phrases
        must_track_phrases = _merge_unique_phrases(query_subject_phrases, detector_phrases)[:detector_limit]

    return {
        "query": query,
        "video_inventory_phrases": video_inventory_phrases,
        "primary_subject_phrases": primary_subject_phrases,
        "query_subject_phrases": query_subject_phrases,
        "required_identity_attributes": required_identity_attributes,
        "temporal_state_attributes": temporal_state_attributes,
        "identity_attribute_filter": identity_attribute_filter,
        "relation_context_phrases": relation_context_phrases,
        "query_successor_phrases": query_successor_phrases,
        "detector_phrases": detector_phrases,
        "optional_phrases": optional_phrases,
        "interaction_phrase": interaction_phrase,
        "start_condition": start_condition,
        "stop_condition": stop_condition,
        "temporal_hints": temporal_hints,
        "phase_transition_hints": phase_transition_hints,
        "must_track_phrases": must_track_phrases,
        "action_window_hint": action_window_hint,
        "support_window_hint": support_window_hint,
        "requested_instance_count": 1 if counted_subject_spec is None else int(counted_subject_spec[1]),
        "absent_query": bool(absent_query),
        "empty_query": bool(absent_query),
        "empty_reason": notes if absent_query else "",
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
            max_new_tokens=_qwen_max_new_tokens(),
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

    resolved_path = _resolve_qwen_model(qwen_model)
    try:
        from refergaussian.semantics.vlm_backends import get_vlm_planner
        teacher = get_vlm_planner(str(resolved_path))
    except ImportError:
        teacher = QwenQueryPlanner(resolved_path)
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
    if bool(plan.get("empty_query")):
        plan["boundary_mode"] = "skipped_empty_query"
        plan["boundary_num_context_frames"] = 0
        plan["boundary_context_frames"] = []
        plan["coarse_temporal_window"] = {
            "start_slot": None,
            "end_slot": None,
            "start_sample_index": None,
            "end_sample_index": None,
            "start_frame_index": None,
            "end_frame_index": None,
            "frame_labels": [],
            "notes": plan.get("empty_reason") or plan.get("notes", ""),
            "raw_output": "",
        }
        plan["temporal_refinement"] = {"start": None, "end": None}
        plan["refined_temporal_window"] = {
            "start_sample_index": None,
            "end_sample_index": None,
            "start_frame_index": None,
            "end_frame_index": None,
            "start_time_value": None,
            "end_time_value": None,
        }
        plan["phase_transition_hints"] = []
        if output_path is not None:
            _write_json(Path(output_path), plan)
        return plan

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
