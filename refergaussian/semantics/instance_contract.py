"""Structural contracts for deterministic Stage-1 instance-set selection."""

from __future__ import annotations

import re
from typing import Any


def _normalize_phrase(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def _distinct_object_ids(values: Any) -> set[int]:
    if not isinstance(values, list):
        return set()
    output: set[int] = set()
    for value in values:
        try:
            output.add(int(value))
        except (TypeError, ValueError):
            continue
    return output


def declared_multi_instance_group(
    query_plan_payload: dict[str, Any],
    tracks_payload: dict[str, Any],
    entitybank_payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return a fully lifted declared Stage-1 multi-instance group, if any.

    The contract deliberately contains no object vocabulary.  It is satisfied
    only when a single planned subject exactly matches a Stage-1 group under
    the explicit ``multi_hypothesis`` policy.  When an entitybank is supplied,
    every group member must also have a separately lifted Gaussian entity.
    """
    subjects = [
        _normalize_phrase(value)
        for value in query_plan_payload.get("query_subject_phrases", [])
        if _normalize_phrase(value)
    ]
    successors = [
        _normalize_phrase(value)
        for value in query_plan_payload.get("query_successor_phrases", [])
        if _normalize_phrase(value)
    ]
    if len(subjects) != 1 or successors:
        return None

    lifted_object_ids: set[int] | None = None
    if entitybank_payload is not None:
        entities = entitybank_payload.get("entities", [])
        if not isinstance(entities, list):
            return None
        lifted_object_ids = _distinct_object_ids(
            [entity.get("stage1_object_id") for entity in entities if isinstance(entity, dict)]
        )

    groups = tracks_payload.get("instance_candidate_groups", [])
    if not isinstance(groups, list):
        return None
    for group in groups:
        if not isinstance(group, dict):
            continue
        if str(group.get("selection_policy", "")).strip().lower() != "multi_hypothesis":
            continue
        if _normalize_phrase(group.get("semantic_phrase", "")) != subjects[0]:
            continue
        object_ids = _distinct_object_ids(group.get("object_ids", []))
        if len(object_ids) < 2:
            continue
        if lifted_object_ids is not None and not object_ids.issubset(lifted_object_ids):
            continue
        return {
            "semantic_phrase": subjects[0],
            "object_ids": sorted(object_ids),
            "selection_policy": "multi_hypothesis",
        }
    return None
