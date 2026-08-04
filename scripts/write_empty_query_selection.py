#!/usr/bin/env python3
"""Write a selected_query_qwen.json payload for an explicitly empty query plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload


def _is_empty_plan(plan: dict[str, Any]) -> bool:
    if bool(plan.get("empty_query") or plan.get("absent_query")):
        return True
    notes = str(plan.get("notes", "")).strip().upper()
    if notes.startswith("ZERO_QUERY"):
        return True
    subjects = [item for item in plan.get("query_subject_phrases", []) if str(item).strip()]
    detectors = [item for item in plan.get("detector_phrases", []) if str(item).strip()]
    return bool(plan.get("empty_reason")) and not subjects and not detectors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--query-plan-path", required=True)
    parser.add_argument("--output-path")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    plan_path = Path(args.query_plan_path)
    plan = _read_json(plan_path)
    if not _is_empty_plan(plan):
        return 1
    if args.check_only:
        return 0
    if not args.output_path:
        raise SystemExit("--output-path is required unless --check-only is used")

    notes = str(plan.get("empty_reason") or plan.get("notes") or "Empty query plan.").strip()
    payload = {
        "query": str(args.query),
        "selected": [],
        "empty": True,
        "empty_prediction": True,
        "selection_status": "semantic_empty",
        "selection_status_reason": "qwen_visual_query_plan_declared_absent_referent",
        "semantic_empty": True,
        "unresolved": False,
        "empty_reason": notes,
        "notes": notes,
        "selection_mode": "query_plan_empty",
        "subject_phrases": [],
        "successor_phrases": [],
        "subject_phrase_matches": {},
        "successor_phrase_matches": {},
        "contact_pair": None,
        "raw_output": str(plan.get("raw_output", "")),
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
