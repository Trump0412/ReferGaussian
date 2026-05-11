from __future__ import annotations

import importlib.util
import json
import re
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .native_assignment import export_native_semantic_assignments
from .query_render import _find_render_dir, _find_source_frame_dir, _hypernerf_test_ids, _pixel_radius
from .source_images import resolve_dataset_image_entries

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_ROOT = _PROJECT_ROOT / "external" / "4DGaussians"
for _candidate in (_PROJECT_ROOT,):
    candidate_str = str(_candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


def _load_camera_class():
    utils_path = _UPSTREAM_ROOT / "scene" / "utils.py"
    spec = importlib.util.spec_from_file_location("refergaussian_scene_utils_qwen", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Camera utilities from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Camera


Camera = _load_camera_class()


DEFAULT_QWEN_MODEL = Path(os.environ.get("HYPERGAUSSIAN_QWEN_MODEL", str(_PROJECT_ROOT / "models" / "Qwen3-VL-8B-Instruct")))

ENTITY_PROMPT_TEMPLATE = """You are assigning continuous semantics to one ReferGaussian 4D worldtube entity.
You are given a few representative crops from the same entity across time and a geometry prior derived from the 4D reconstruction.
Your job is to identify the entity and describe its continuous semantics over time.

Entity prior:
{prior_json}

Return exactly one JSON object with keys:
- category: short noun phrase
- canonical_description: one concise sentence
- static_text: one concise sentence describing stable identity cues
- global_desc: one concise sentence describing the entity across time
- dynamic_desc: array of short phrases
- interaction_desc: array of short phrases
- role_likelihood: object with numeric keys patient, tool, agent, support, background, each in [0, 1]
- concept_tags: array of short lowercase tags
- temporal_segments: array of objects with keys frame_start, frame_end, phase, caption
- query_hints: object with array keys patient_terms, tool_terms, action_terms

Rules:
- Be specific when the entity looks like a lemon, knife, hand, cutting board, or another familiar object.
- Use the candidate temporal segments and keep captions short.
- Do not output markdown fences.
- Do not mention uncertainty unless the evidence is truly weak.
"""


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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
    decoder = json.JSONDecoder()
    cleaned = _clean_llm_text(text)
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[index:])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Unable to parse JSON object from model output: {text!r}")


def _import_transformers():
    try:
        import transformers  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Python dependency 'transformers'. Install it inside the ReferGaussian env, "
            "for example with: pip install transformers accelerate sentencepiece"
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


class QwenVisionTeacher:
    def __init__(self, model_name_or_path: str | Path):
        transformers = _import_transformers()
        self.transformers = transformers
        self.model_name_or_path = str(model_name_or_path)
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

        self.processor = processor_cls.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        self.model = model_cls.from_pretrained(
            self.model_name_or_path,
            **_qwen_model_load_kwargs(),
        )

    def generate_json(self, prompt: str, images: list[Image.Image]) -> tuple[dict[str, Any], str]:
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
            max_new_tokens=224,
            do_sample=False,
        )
        trimmed = generated[:, model_inputs["input_ids"].shape[1] :]
        output = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return _extract_first_json(output), output


def _resolve_qwen_model(explicit_path: str | None = None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    if DEFAULT_QWEN_MODEL.exists():
        return DEFAULT_QWEN_MODEL
    raise FileNotFoundError(
        f"Unable to resolve a local Qwen model. Expected {DEFAULT_QWEN_MODEL} or pass an explicit path."
    )


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


def _merge_unique_strings(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for value in group:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            merged.append(text)
    return merged


def _candidate_segments(assignment: dict[str, Any]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for phase_name in ("stationary", "moving"):
        for segment in assignment.get("phase_segments", {}).get(phase_name, []):
            if len(segment) != 2:
                continue
            segments.append(
                {
                    "frame_start": int(segment[0]),
                    "frame_end": int(segment[1]),
                    "phase": "static" if phase_name == "stationary" else "dynamic",
                }
            )
    support_window = assignment.get("support_window", {})
    if support_window:
        segments.append(
            {
                "frame_start": int(support_window.get("frame_start", 0)),
                "frame_end": int(support_window.get("frame_end", 0)),
                "phase": "support",
            }
        )
    segments.sort(key=lambda item: (int(item["frame_start"]), int(item["frame_end"])))
    return segments[:8]


def _track_map(entitybank_dir: Path) -> dict[int, dict[str, Any]]:
    payload = _read_json(entitybank_dir / "semantic_tracks.json")
    return {int(track["entity_id"]): track for track in payload.get("tracks", [])}


def _prior_map(entitybank_dir: Path) -> dict[int, dict[str, Any]]:
    payload = _read_json(entitybank_dir / "semantic_priors.json")
    priors = payload.get("priors", payload.get("semantic_priors", []))
    return {int(prior["entity_id"]): prior for prior in priors}


def _load_source_images(run_dir: Path) -> tuple[Path, list[str], np.ndarray]:
    config = _read_simple_yaml(run_dir / "config.yaml")
    source_path = Path(config.get("source_path", ""))
    if (source_path / "dataset.json").exists() and (source_path / "metadata.json").exists():
        test_ids, test_times = _hypernerf_test_ids(source_path)
        try:
            render_dir = _find_render_dir(run_dir)
            with Image.open(next(iter(sorted(render_dir.glob("*.png"))))) as probe:
                target_size = probe.size
            source_frame_dir = _find_source_frame_dir(source_path, target_size)
        except FileNotFoundError:
            entries = resolve_dataset_image_entries(source_path)
            if not entries:
                raise FileNotFoundError(f"No source images found under {source_path}")
            source_frame_dir = Path(entries[0]["image_path"]).parent
        return source_frame_dir, test_ids, test_times

    render_dir = _find_render_dir(run_dir)
    with Image.open(next(iter(sorted(render_dir.glob("*.png"))))) as probe:
        target_size = probe.size

    gt_dir = render_dir.parent / "gt"
    gt_files = sorted(gt_dir.glob("*.png"))
    if gt_files:
        test_ids = [path.stem for path in gt_files]
        if len(test_ids) <= 1:
            test_times = np.zeros((len(test_ids),), dtype=np.float32)
        else:
            test_times = np.linspace(0.0, 1.0, num=len(test_ids), dtype=np.float32)
        return gt_dir, test_ids, test_times

    entries = resolve_dataset_image_entries(source_path)
    test_ids = [str(entry["image_id"]) for entry in entries]
    test_times = np.asarray([float(entry["time_value"]) for entry in entries], dtype=np.float32)
    source_frame_dir = Path(entries[0]["image_path"]).parent if entries else source_path
    return source_frame_dir, test_ids, test_times


def _nearest_test_frame_index(sample_time: float, test_times: np.ndarray) -> int:
    if test_times.size == 0:
        return 0
    return int(np.abs(np.asarray(test_times, dtype=np.float32) - float(sample_time)).argmin())


def _frame_priority(track: dict[str, Any], frame_index: int) -> float:
    frame = track["frames"][frame_index]
    return (
        1.8 * float(frame.get("support_score", 0.0))
        + 1.5 * float(frame.get("visibility", 0.0))
        + 0.35 * float(frame.get("speed", 0.0))
        + (0.20 if bool(frame.get("is_keyframe")) else 0.0)
    )


def _candidate_frame_indices(assignment: dict[str, Any], track: dict[str, Any], max_frames: int = 4) -> list[int]:
    frame_count = len(track.get("frames", []))
    if frame_count == 0:
        return []
    candidate_indices: list[int] = []
    support_window = assignment.get("support_window", {})
    if support_window:
        candidate_indices.append(int(support_window.get("frame_peak", support_window.get("frame_start", 0))))
    for phase_name in ("moving", "stationary"):
        for segment in assignment.get("phase_segments", {}).get(phase_name, [])[:3]:
            if len(segment) != 2:
                continue
            start = max(0, min(frame_count - 1, int(segment[0])))
            end = max(0, min(frame_count - 1, int(segment[1])))
            candidate_indices.append((start + end) // 2)
            if end - start >= 4:
                candidate_indices.append(start)
                candidate_indices.append(end)
    scored = sorted(
        {index for index in candidate_indices if 0 <= index < frame_count},
        key=lambda index: (-_frame_priority(track, index), index),
    )
    if len(scored) < max_frames:
        all_indices = sorted(range(frame_count), key=lambda index: (-_frame_priority(track, index), index))
        for index in all_indices:
            if index not in scored:
                scored.append(index)
            if len(scored) >= max_frames:
                break
    return scored[:max_frames]


def _crop_entity_image(
    image: Image.Image,
    camera_path: Path,
    frame_record: dict[str, Any],
) -> tuple[Image.Image | None, dict[str, Any]]:
    camera = Camera.from_json(camera_path)
    center_world = np.asarray(frame_record["center_world"], dtype=np.float32)
    extent_min = np.asarray(frame_record["extent_world_min"], dtype=np.float32)
    extent_max = np.asarray(frame_record["extent_world_max"], dtype=np.float32)
    center_local = camera.points_to_local_points(center_world[None, :])[0]
    if float(center_local[2]) <= 1.0e-4:
        return None, {"projected": False, "reason": "behind_camera"}
    width, height = image.size
    camera_size = np.asarray(camera.image_size, dtype=np.float32).reshape(-1)
    scale_x = float(width) / max(float(camera_size[0]), 1.0)
    scale_y = float(height) / max(float(camera_size[1]), 1.0)
    pixel = camera.project(center_world[None, :])[0].astype(np.float32)
    pixel[0] *= scale_x
    pixel[1] *= scale_y
    pixel_x = float(pixel[0])
    pixel_y = float(pixel[1])
    if pixel_x < -0.25 * width or pixel_x > 1.25 * width or pixel_y < -0.25 * height or pixel_y > 1.25 * height:
        return None, {"projected": False, "reason": "far_offscreen"}

    radius = _pixel_radius(camera, center_world, extent_min, extent_max)
    radius = int(max(8, round(float(radius) * 0.5 * (scale_x + scale_y))))
    crop_half = int(np.clip(radius * 2.4, 64.0, 224.0))
    cx = int(round(np.clip(pixel_x, 0.0, width - 1.0)))
    cy = int(round(np.clip(pixel_y, 0.0, height - 1.0)))
    left = max(0, cx - crop_half)
    top = max(0, cy - crop_half)
    right = min(width, cx + crop_half)
    bottom = min(height, cy + crop_half)
    if right - left < 32 or bottom - top < 32:
        return None, {"projected": False, "reason": "tiny_crop"}

    crop = image.crop((left, top, right, bottom)).convert("RGB")
    return crop, {
        "projected": True,
        "pixel_xy": [pixel_x, pixel_y],
        "depth": float(center_local[2]),
        "radius_px": int(radius),
        "crop_box": [int(left), int(top), int(right), int(bottom)],
    }


def _full_frame_context(image: Image.Image, frame_record: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    width, height = image.size
    return image.convert("RGB"), {
        "projected": False,
        "reason": "full_frame_context",
        "pixel_xy": None,
        "depth": None,
        "radius_px": int(max(min(width, height) // 2, 1)),
        "crop_box": [0, 0, int(width), int(height)],
        "visibility": float(frame_record.get("visibility", 0.0)),
        "support_score": float(frame_record.get("support_score", 0.0)),
    }


def _entity_prompt(assignment: dict[str, Any], prior: dict[str, Any], crop_records: list[dict[str, Any]]) -> str:
    prior_payload = {
        "entity_id": int(assignment["entity_id"]),
        "entity_type": assignment.get("entity_type", "unknown"),
        "semantic_head": assignment.get("semantic_head", "unknown"),
        "temporal_mode": assignment.get("temporal_mode", "unknown"),
        "native_role_scores": assignment.get("role_scores", {}),
        "concept_tags": assignment.get("concept_tags", [])[:16],
        "native_text": assignment.get("native_text", {}),
        "prompt_groups": assignment.get("prompt_groups", {}),
        "candidate_segments": _candidate_segments(assignment),
        "support_window": assignment.get("support_window", {}),
        "geometry_evidence": prior.get("geometry_evidence", {}),
        "crop_frames": [
            {
                "frame_index": int(item["frame_index"]),
                "time_value": float(item["time_value"]),
                "kind": item["kind"],
            }
            for item in crop_records
        ],
    }
    return ENTITY_PROMPT_TEMPLATE.format(
        prior_json=json.dumps(prior_payload, ensure_ascii=False, indent=2),
    )


def _normalize_role_payload(payload: dict[str, Any], bootstrap_role_scores: dict[str, float]) -> dict[str, float]:
    raw = payload.get("role_likelihood", {})
    normalized = {key: float(np.clip(float(raw.get(key, bootstrap_role_scores.get(key, 0.0))), 0.0, 1.0)) for key in ("patient", "tool", "agent", "support", "background")}
    if not any(normalized.values()):
        return {key: float(value) for key, value in bootstrap_role_scores.items()}
    return normalized


def _normalize_text_payload(payload: dict[str, Any], bootstrap_text: dict[str, Any]) -> dict[str, Any]:
    static_text = str(payload.get("static_text", "")).strip() or str(bootstrap_text.get("static_text", "")).strip()
    global_desc = str(payload.get("global_desc", "")).strip() or str(bootstrap_text.get("global_desc", "")).strip()
    dynamic_desc = [str(item).strip() for item in payload.get("dynamic_desc", []) if str(item).strip()]
    interaction_desc = [str(item).strip() for item in payload.get("interaction_desc", []) if str(item).strip()]
    if not dynamic_desc:
        dynamic_desc = [str(item).strip() for item in bootstrap_text.get("dynamic_desc", []) if str(item).strip()]
    if not interaction_desc:
        interaction_desc = [str(item).strip() for item in bootstrap_text.get("interaction_desc", []) if str(item).strip()]
    return {
        "static_text": static_text,
        "global_desc": global_desc,
        "dynamic_desc": dynamic_desc[:6],
        "interaction_desc": interaction_desc[:6],
    }


def _normalized_temporal_segments(payload: dict[str, Any], assignment: dict[str, Any]) -> list[dict[str, Any]]:
    frame_cap = int(assignment.get("support_window", {}).get("frame_end", 64))
    segments = []
    for item in payload.get("temporal_segments", []):
        try:
            start = int(item.get("frame_start", 0))
            end = int(item.get("frame_end", start))
        except Exception:
            continue
        start = max(0, start)
        end = min(max(start, end), max(frame_cap - 1, start))
        phase = str(item.get("phase", "dynamic")).strip().lower() or "dynamic"
        caption = str(item.get("caption", "")).strip()
        segments.append(
            {
                "frame_start": start,
                "frame_end": end,
                "phase": phase,
                "caption": caption,
            }
        )
    return segments[:8]


def _qwen_terms(payload: dict[str, Any]) -> tuple[list[str], dict[str, list[str]]]:
    query_hints = payload.get("query_hints", {}) if isinstance(payload.get("query_hints", {}), dict) else {}
    patient_terms = [str(item).strip().lower() for item in query_hints.get("patient_terms", []) if str(item).strip()]
    tool_terms = [str(item).strip().lower() for item in query_hints.get("tool_terms", []) if str(item).strip()]
    action_terms = [str(item).strip().lower() for item in query_hints.get("action_terms", []) if str(item).strip()]
    concept_tags = [str(item).strip().lower() for item in payload.get("concept_tags", []) if str(item).strip()]
    category = str(payload.get("category", "")).strip().lower()
    description_terms = re.findall(r"[a-z]+", " ".join([category, str(payload.get("canonical_description", "")), str(payload.get("global_desc", ""))]).lower())
    merged_terms = _merge_unique_strings(concept_tags, patient_terms, tool_terms, action_terms, description_terms)
    return merged_terms, {
        "patient_terms": patient_terms[:10],
        "tool_terms": tool_terms[:10],
        "action_terms": action_terms[:10],
    }


def _assignment_priority(assignment: dict[str, Any], forced_ids: set[int]) -> tuple[float, int]:
    entity_id = int(assignment["entity_id"])
    role_scores = assignment.get("role_scores", {})
    head = str(assignment.get("semantic_head", "dynamic"))
    head_bonus = 0.20 if head == "interaction" else 0.10 if head == "dynamic" else 0.0
    lexical_bonus = 0.25 if entity_id in forced_ids else 0.0
    score = (
        1.10 * float(assignment.get("quality", 0.0))
        + 0.20 * max(float(role_scores.get("patient", 0.0)), float(role_scores.get("tool", 0.0)), float(role_scores.get("agent", 0.0)))
        + 0.12 * float(role_scores.get("support", 0.0))
        + head_bonus
        + lexical_bonus
    )
    return score, -entity_id


def _forced_entity_ids_for_query(run_dir: Path, query: str, shortlist_k: int) -> set[int]:
    from .query_scoring import score_native_query

    query_dir = score_native_query(
        run_dir=run_dir,
        query=query,
        query_name="_qwen_bootstrap_shortlist",
        top_k=max(shortlist_k, 8),
        semantic_source="native",
    )
    ids: set[int] = set()
    candidates_path = query_dir / "candidates.json"
    if candidates_path.exists():
        payload = _read_json(candidates_path)
        for group_name in ("patient_candidates", "tool_candidates", "support_candidates", "pair_candidates"):
            for item in payload.get(group_name, [])[:shortlist_k]:
                if "id" in item:
                    ids.add(int(item["id"]))
                if "patient_id" in item:
                    ids.add(int(item["patient_id"]))
                if "tool_id" in item:
                    ids.add(int(item["tool_id"]))
    selected_path = query_dir / "selected.json"
    if selected_path.exists():
        payload = _read_json(selected_path)
        for item in payload.get("selected", []):
            ids.add(int(item["id"]))
    return ids


def export_qwen_semantic_assignments(
    run_dir: str | Path,
    qwen_model: str | None = None,
    max_entities: int | None = None,
    query: str | None = None,
    shortlist_k: int = 24,
) -> Path:
    run_dir = Path(run_dir)
    entitybank_dir = run_dir / "entitybank"
    assignments_path = entitybank_dir / "native_semantic_assignments.json"
    if not assignments_path.exists():
        export_native_semantic_assignments(run_dir)

    native_payload = _read_json(assignments_path)
    assignments = native_payload.get("assignments", [])
    if not assignments:
        raise ValueError(f"No native assignments found under {assignments_path}")

    forced_ids: set[int] = set()
    if query:
        forced_ids = _forced_entity_ids_for_query(run_dir, query=query, shortlist_k=shortlist_k)

    ranked_assignments = sorted(assignments, key=lambda item: _assignment_priority(item, forced_ids), reverse=True)
    selected_ids = {int(item["entity_id"]) for item in ranked_assignments[:max_entities]} if max_entities else {int(item["entity_id"]) for item in ranked_assignments}
    selected_ids.update(forced_ids)

    prior_map = _prior_map(entitybank_dir)
    track_map = _track_map(entitybank_dir)
    source_frame_dir, test_ids, test_times = _load_source_images(run_dir)
    config = _read_simple_yaml(run_dir / "config.yaml")
    dataset_dir = Path(config.get("source_path", ""))

    model_path = _resolve_qwen_model(qwen_model)
    teacher = QwenVisionTeacher(model_path)
    crop_root = entitybank_dir / "qwen_teacher" / "crops"
    qwen_records: list[dict[str, Any]] = []
    assignment_map = {int(item["entity_id"]): item for item in assignments}
    qwen_count = 0

    for bootstrap_assignment in assignments:
        entity_id = int(bootstrap_assignment["entity_id"])
        assignment = json.loads(json.dumps(bootstrap_assignment))
        assignment["bootstrap_text"] = assignment.get("native_text", {})
        assignment["bootstrap_role_scores"] = assignment.get("role_scores", {})
        assignment["semantic_source"] = "refergaussian_native_bootstrap"
        assignment["qwen_enabled"] = False
        assignment["qwen_model"] = str(model_path)
        assignment["qwen_prompt"] = None
        assignment["qwen_raw_output"] = None
        assignment["qwen_crop_records"] = []
        assignment["qwen_temporal_segments"] = []
        if entity_id not in selected_ids:
            qwen_records.append(assignment)
            continue

        track = track_map.get(entity_id)
        prior = prior_map.get(entity_id, {})
        if track is None:
            qwen_records.append(assignment)
            continue

        frame_indices = _candidate_frame_indices(assignment, track)
        images: list[Image.Image] = []
        crop_records: list[dict[str, Any]] = []
        entity_crop_dir = crop_root / f"entity_{entity_id:04d}"
        entity_crop_dir.mkdir(parents=True, exist_ok=True)

        attempted_indices: set[int] = set()

        def append_frame_crop(frame_index: int, order_index: int, allow_full_frame: bool = False) -> bool:
            track_frames = track.get("frames", [])
            if frame_index < 0 or frame_index >= len(track_frames):
                return False
            attempted_indices.add(int(frame_index))
            frame_record = track_frames[frame_index]
            sample_time = float(frame_record.get("time_value", 0.0))
            test_frame_index = _nearest_test_frame_index(sample_time, test_times)
            if test_frame_index < 0 or test_frame_index >= len(test_ids):
                return False
            image_id = test_ids[test_frame_index]
            image_path = source_frame_dir / f"{image_id}.png"
            camera_path = dataset_dir / "camera" / f"{image_id}.json"
            if not image_path.exists() or not camera_path.exists():
                return False
            with Image.open(image_path) as full_image:
                full_rgb = full_image.convert("RGB")
                crop_image, crop_meta = _crop_entity_image(full_rgb, camera_path, frame_record)
                if crop_image is None:
                    if not allow_full_frame:
                        return False
                    crop_image, crop_meta = _full_frame_context(full_rgb, frame_record)
                kind = "dynamic"
                moving_segments = assignment.get("phase_segments", {}).get("moving", [])
                stationary_segments = assignment.get("phase_segments", {}).get("stationary", [])
                if any(int(seg[0]) <= frame_index <= int(seg[1]) for seg in moving_segments if len(seg) == 2):
                    kind = "dynamic"
                elif any(int(seg[0]) <= frame_index <= int(seg[1]) for seg in stationary_segments if len(seg) == 2):
                    kind = "static"
                crop_path = entity_crop_dir / f"{order_index:02d}_frame_{frame_index:04d}_{kind}.png"
                crop_image.save(crop_path)
                images.append(crop_image)
                crop_records.append(
                    {
                        "frame_index": int(frame_index),
                        "test_frame_index": int(test_frame_index),
                        "time_value": float(sample_time),
                        "image_id": image_id,
                        "kind": kind,
                        "path": str(crop_path),
                        **crop_meta,
                    }
                )
            return True

        order_index = 0
        for frame_index in frame_indices:
            if append_frame_crop(frame_index, order_index):
                order_index += 1

        if len(images) < 2:
            all_indices = sorted(
                range(len(track.get("frames", []))),
                key=lambda index: (-_frame_priority(track, index), index),
            )
            for frame_index in all_indices:
                if frame_index in attempted_indices:
                    continue
                if append_frame_crop(frame_index, order_index):
                    order_index += 1
                if len(images) >= 4:
                    break

        if not images:
            fallback_indices = sorted(
                range(len(track.get("frames", []))),
                key=lambda index: (-float(track["frames"][index].get("visibility", 0.0)), -_frame_priority(track, index), index),
            )
            for frame_index in fallback_indices[:3]:
                if append_frame_crop(frame_index, order_index, allow_full_frame=True):
                    order_index += 1
                if len(images) >= 2:
                    break

        if not images:
            qwen_records.append(assignment)
            continue

        prompt = _entity_prompt(assignment, prior, crop_records)
        try:
            payload, raw_output = teacher.generate_json(prompt, images)
        except Exception as exc:
            assignment["qwen_error"] = str(exc)
            qwen_records.append(assignment)
            continue

        qwen_role_scores = _normalize_role_payload(payload, assignment["bootstrap_role_scores"])
        qwen_text = _normalize_text_payload(payload, assignment["bootstrap_text"])
        qwen_terms, query_hints = _qwen_terms(payload)
        blended_roles = {
            key: float(
                np.clip(
                    0.45 * float(assignment["bootstrap_role_scores"].get(key, 0.0))
                    + 0.55 * float(qwen_role_scores.get(key, 0.0)),
                    0.0,
                    1.0,
                )
            )
            for key in ("patient", "tool", "agent", "support", "background")
        }
        prompt_groups = assignment.get("prompt_groups", {})
        prompt_groups = {
            "global": _merge_unique_strings(prompt_groups.get("global", []), [payload.get("canonical_description", ""), payload.get("category", "")]),
            "static": _merge_unique_strings(prompt_groups.get("static", []), [qwen_text["static_text"]]),
            "dynamic": _merge_unique_strings(prompt_groups.get("dynamic", []), qwen_text.get("dynamic_desc", [])),
            "interaction": _merge_unique_strings(prompt_groups.get("interaction", []), qwen_text.get("interaction_desc", [])),
            "patient_terms": query_hints.get("patient_terms", []),
            "tool_terms": query_hints.get("tool_terms", []),
            "action_terms": query_hints.get("action_terms", []),
        }
        concept_tags = _merge_unique_strings(
            assignment.get("concept_tags", []),
            qwen_terms,
            [payload.get("category", "")],
        )
        semantic_terms = _merge_unique_strings(
            assignment.get("semantic_terms", []),
            qwen_terms,
            prompt_groups["global"],
            prompt_groups["dynamic"],
            prompt_groups["interaction"],
        )

        assignment["semantic_source"] = "refergaussian_qwen"
        assignment["qwen_enabled"] = True
        assignment["qwen_prompt"] = prompt
        assignment["qwen_raw_output"] = raw_output
        assignment["qwen_crop_records"] = crop_records
        assignment["qwen_text"] = {
            "category": str(payload.get("category", "")).strip(),
            "canonical_description": str(payload.get("canonical_description", "")).strip(),
            **qwen_text,
            "query_hints": query_hints,
        }
        assignment["qwen_role_scores"] = qwen_role_scores
        assignment["role_scores"] = blended_roles
        assignment["native_text"] = qwen_text
        assignment["prompt_groups"] = prompt_groups
        assignment["concept_tags"] = concept_tags[:48]
        assignment["semantic_terms"] = semantic_terms[:64]
        assignment["qwen_temporal_segments"] = _normalized_temporal_segments(payload, assignment)
        qwen_count += 1
        qwen_records.append(assignment)

    qwen_assignment_map = {int(item["entity_id"]): item for item in qwen_records}
    entities_payload = _read_json(entitybank_dir / "entities.json")
    enriched_entities = []
    for entity in entities_payload.get("entities", []):
        entity_copy = dict(entity)
        assignment = qwen_assignment_map.get(int(entity["id"]))
        if assignment is not None:
            entity_copy["semantic_source"] = assignment.get("semantic_source", "refergaussian_native_bootstrap")
            entity_copy["qwen_enabled"] = bool(assignment.get("qwen_enabled", False))
            entity_copy["qwen_role_scores"] = assignment.get("qwen_role_scores", {})
            entity_copy["qwen_concept_tags"] = assignment.get("concept_tags", [])
            entity_copy["static_text"] = assignment.get("native_text", {}).get("static_text", "")
            entity_copy["global_desc"] = assignment.get("native_text", {}).get("global_desc", "")
            entity_copy["dyn_desc"] = assignment.get("native_text", {}).get("dynamic_desc", [])
        enriched_entities.append(entity_copy)

    output_payload = {
        **native_payload,
        "semantic_source": "refergaussian_qwen",
        "qwen_model": str(model_path),
        "num_assignments": int(len(qwen_records)),
        "num_qwen_assignments": int(qwen_count),
        "num_bootstrap_assignments": int(len(qwen_records) - qwen_count),
        "query_conditioning": None if not query else {"query": query, "forced_entity_ids": sorted(int(value) for value in forced_ids)},
        "assignments": qwen_records,
    }
    output_path = entitybank_dir / "semantic_assignments_qwen.json"
    _write_json(output_path, output_payload)
    _write_json(
        entitybank_dir / "entities_semantic_qwen.json",
        {
            **entities_payload,
            "entities": enriched_entities,
            "semantic_source": "refergaussian_qwen",
            "qwen_model": str(model_path),
            "num_qwen_assignments": int(qwen_count),
        },
    )
    return output_path
