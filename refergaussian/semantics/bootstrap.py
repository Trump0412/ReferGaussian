from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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


def _resolve_hypernerf_images(source_path: Path) -> list[dict[str, Any]]:
    dataset_json = _read_json(source_path / "dataset.json")
    metadata_json = _read_json(source_path / "metadata.json")
    if dataset_json is None or metadata_json is None:
        return []

    all_ids = dataset_json.get("ids", [])
    val_ids = set(dataset_json.get("val_ids", []))
    if val_ids:
        test_ids = [item for item in all_ids if item in val_ids]
    else:
        test_ids = []
        for index, item in enumerate(all_ids):
            if index % 4 == 2 and index < len(all_ids) - 1:
                test_ids.append(item)

    max_time = max(float(metadata_json[item]["warp_id"]) for item in all_ids) if all_ids else 1.0
    if max_time <= 0:
        max_time = 1.0

    image_entries = []
    for image_id in test_ids:
        image_path = None
        for scale_dir in ("2x", "1x", "4x"):
            candidate = source_path / "rgb" / scale_dir / f"{image_id}.png"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue
        image_entries.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "time_value": float(metadata_json[image_id]["warp_id"]) / max_time,
                "split": "test",
                "dataset_type": "hypernerf",
            }
        )
    return image_entries


def _resolve_dnerf_images(source_path: Path) -> list[dict[str, Any]]:
    transforms = _read_json(source_path / "transforms_test.json")
    if transforms is None:
        return []

    image_entries = []
    for frame in transforms.get("frames", []):
        file_path = str(frame.get("file_path", ""))
        if not file_path:
            continue
        candidate = source_path / file_path
        if candidate.suffix == "":
            candidate = candidate.with_suffix(".png")
        if not candidate.exists():
            continue
        image_entries.append(
            {
                "image_id": candidate.stem,
                "image_path": str(candidate),
                "time_value": float(frame.get("time", 0.0)),
                "split": "test",
                "dataset_type": "dnerf",
            }
        )
    return image_entries


def _resolve_source_images(source_path: Path) -> list[dict[str, Any]]:
    if (source_path / "dataset.json").exists() and (source_path / "metadata.json").exists():
        return _resolve_hypernerf_images(source_path)
    if (source_path / "transforms_test.json").exists():
        return _resolve_dnerf_images(source_path)
    return []


def _frame_query_score(frame: dict[str, Any], source_time_value: float) -> float:
    frame_time = float(frame.get("time_value", source_time_value))
    active_slots = frame.get("active_slots", [])
    mean_span = 0.0
    if active_slots:
        mean_span = sum(float(slot.get("support_frame_end", 0) - slot.get("support_frame_start", 0)) for slot in active_slots) / max(len(active_slots), 1)
    normalized_scale = max(mean_span / 64.0, 0.05)
    temporal_score = float(pow(2.718281828, -abs(frame_time - source_time_value) / normalized_scale))
    support_mass = sum(float(slot.get("support_score", 0.0)) for slot in active_slots)
    dynamic_mass = sum(
        float(slot.get("support_score", 0.0))
        for slot in active_slots
        if slot.get("temporal_mode") in {"dynamic_object", "transient_event"}
    )
    return temporal_score * (1.0 + 0.02 * len(active_slots)) + 0.05 * support_mass + 0.03 * dynamic_mass


def _select_slots(active_slots: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    dynamic_slots = [
        slot for slot in active_slots if slot.get("motion_label") == "moving"
    ]
    static_slots = [
        slot for slot in active_slots if slot.get("motion_label") == "stationary"
    ]
    selected: list[dict[str, Any]] = []
    dynamic_quota = min(len(dynamic_slots), max(1, top_k // 2))
    static_quota = min(len(static_slots), max(0, top_k - dynamic_quota))
    selected.extend(dynamic_slots[:dynamic_quota])
    selected.extend(static_slots[:static_quota])
    used = {int(slot.get("slot_id", -1)) for slot in selected}
    for slot in active_slots:
        slot_id = int(slot.get("slot_id", -1))
        if slot_id in used:
            continue
        selected.append(slot)
        used.add(slot_id)
        if len(selected) >= top_k:
            break
    return selected[:top_k]


def export_segmentation_bootstrap(run_dir: str | Path, top_k: int = 8) -> Path:
    run_dir = Path(run_dir)
    entitybank_dir = run_dir / "entitybank"
    config = _read_simple_yaml(run_dir / "config.yaml")
    source_path = Path(config.get("source_path", ""))

    frame_queries_payload = _read_json(entitybank_dir / "semantic_frame_queries.json")
    if frame_queries_payload is None:
        raise FileNotFoundError(f"semantic_frame_queries.json not found under {entitybank_dir}")
    prior_payload = _read_json(entitybank_dir / "semantic_priors.json") or {}
    prior_map = {
        int(prior.get("slot_id", -1)): prior
        for prior in prior_payload.get("priors", [])
    }
    time_values = [float(frame["time_value"]) for frame in frame_queries_payload.get("frames", [])]
    source_images = _resolve_source_images(source_path)

    frames = []
    for image in source_images:
        frame_records = frame_queries_payload.get("frames", [])
        if not time_values or not frame_records:
            nearest_index = 0
        else:
            nearest_index = max(
                range(len(time_values)),
                key=lambda idx: _frame_query_score(frame_records[idx], float(image["time_value"])),
            )
        nearest_frame = frame_records[nearest_index]
        active_slots = _select_slots(nearest_frame.get("active_slots", []), top_k=top_k)
        enriched_slots = []
        for slot in active_slots:
            slot_copy = dict(slot)
            prior = prior_map.get(int(slot.get("slot_id", -1)), {})
            motion_label = str(slot_copy.get("motion_label", "unknown"))
            if motion_label == "moving":
                preferred_group = "dynamic"
                preferred_prompts = prior.get("dynamic_semantics", {}).get(
                    "prompt_candidates",
                    slot_copy.get("dynamic_prompt_candidates", []),
                )
            elif motion_label == "stationary":
                preferred_group = "static"
                preferred_prompts = prior.get("static_semantics", {}).get(
                    "prompt_candidates",
                    slot_copy.get("static_prompt_candidates", []),
                )
            else:
                preferred_group = "temporal"
                preferred_prompts = prior.get("query_pack", {}).get(
                    "temporal",
                    slot_copy.get("temporal_prompt_candidates", []),
                )
            slot_copy["semantic_prior_id"] = prior.get("prior_id")
            slot_copy["semantic_head"] = prior.get("semantic_head")
            slot_copy["preferred_prompt_group"] = preferred_group
            slot_copy["preferred_prompt_candidates"] = preferred_prompts
            slot_copy["interaction_prompt_candidates"] = prior.get("interaction_semantics", {}).get(
                "prompt_candidates",
                [],
            )
            slot_copy["support_keyframes"] = prior.get("interaction_semantics", {}).get("keyframes", [])
            enriched_slots.append(slot_copy)
        active_slots = enriched_slots
        dynamic_slots = [
            slot for slot in active_slots if slot.get("motion_label") == "moving"
        ]
        static_slots = [
            slot for slot in active_slots if slot.get("motion_label") == "stationary"
        ]
        frames.append(
            {
                "image_id": image["image_id"],
                "image_path": image["image_path"],
                "split": image["split"],
                "dataset_type": image["dataset_type"],
                "source_time_value": float(image["time_value"]),
                "query_frame_index": int(nearest_index),
                "query_time_value": float(nearest_frame.get("time_value", image["time_value"])),
                "query_strategy": "worldtube_support",
                "num_candidate_slots": int(len(nearest_frame.get("active_slots", []))),
                "num_dynamic_slots": int(len(dynamic_slots)),
                "num_static_slots": int(len(static_slots)),
                "dynamic_slots": dynamic_slots,
                "static_slots": static_slots,
                "slots": active_slots,
            }
        )

    payload = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "source_path": str(source_path),
        "num_images": len(frames),
        "top_k": int(top_k),
        "num_semantic_priors": int(prior_payload.get("num_priors", 0) or 0),
        "frames": frames,
    }
    output_path = entitybank_dir / "semantic_segmentation_bootstrap.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path
