#!/usr/bin/env python3
"""Helpers for release-safe English query text handling."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
R4D_ENGLISH_QUERY_MAP_PATH = REPO_ROOT / "configs" / "benchmarks" / "r4d_query_text_en.json"
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def contains_cjk(text: object) -> bool:
    return bool(_CJK_RE.search(str(text or "")))


@lru_cache(maxsize=1)
def load_r4d_english_query_map() -> dict[str, str]:
    if not R4D_ENGLISH_QUERY_MAP_PATH.is_file():
        return {}
    with R4D_ENGLISH_QUERY_MAP_PATH.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value).strip() for key, value in payload.items() if str(value).strip()}


def _record_query_id(record: dict[str, Any]) -> str:
    for key in ("query_id", "id", "qid"):
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def choose_english_query_text(record: dict[str, Any]) -> tuple[str, str]:
    """Return release-canonical English text and a short source label.

    Priority:
    1. Explicit English fields from the benchmark file.
    2. Official R4D ``query_id -> text_en`` map.
    3. Existing generic text fields if they are already English.

    Chinese or mixed-language question fields are intentionally not returned.
    This keeps open-source manifests deterministic even when a legacy
    benchmark JSON still stores ``question`` in Chinese.
    """
    for key in ("query_en", "question_en", "text_en", "caption_en"):
        value = str(record.get(key) or "").strip()
        if value and not contains_cjk(value):
            return value, key

    qid = _record_query_id(record)
    if qid:
        mapped = load_r4d_english_query_map().get(qid, "").strip()
        if mapped:
            return mapped, "r4d_query_text_en"

    for key in ("query", "query_text", "question", "text", "caption"):
        value = str(record.get(key) or "").strip()
        if value and not contains_cjk(value):
            return value, key

    return "", "missing_english_query"


def benchmark_record_with_english_query(record: dict[str, Any], query_text: str, source: str) -> dict[str, Any]:
    """Copy a benchmark record while removing non-release Chinese query fields."""
    sanitized = dict(record)
    sanitized.pop("_resolved_scene", None)
    for key in ("text_zh", "query_zh", "question_zh", "caption_zh"):
        sanitized.pop(key, None)
    for key in ("query", "query_text", "question", "text", "caption"):
        value = str(sanitized.get(key) or "").strip()
        if value and contains_cjk(value):
            sanitized.pop(key, None)
    sanitized["query"] = query_text
    sanitized["text_en"] = query_text
    sanitized["query_text_source"] = source
    return sanitized
