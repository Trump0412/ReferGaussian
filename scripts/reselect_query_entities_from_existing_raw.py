import argparse
import json
from pathlib import Path

from refergaussian.semantics.qwen_query_planner import _extract_first_json
from select_qwen_query_entities import (
    _build_candidates,
    _compose_phrase_grounded_selection,
    _read_json,
    _resolve_query_tracks_payload,
    _test_time_values,
    _write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments-path", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--query-plan-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--existing-selection-path", default=None)
    args = parser.parse_args()

    assignments_path = Path(args.assignments_path)
    query_plan_path = Path(args.query_plan_path)
    output_path = Path(args.output_path)
    existing_path = Path(args.existing_selection_path) if args.existing_selection_path else output_path

    old_payload = _read_json(existing_path) if existing_path.exists() else {}
    raw_output = str(old_payload.get("raw_output", "") or "")
    raw_phrase_payload = _extract_first_json(raw_output) if raw_output else {}

    query_plan_payload = _read_json(query_plan_path)
    tracks_payload = _resolve_query_tracks_payload(query_plan_path)
    run_dir = assignments_path.parents[1]
    candidates, pair_candidates = _build_candidates(_read_json(assignments_path), run_dir=run_dir)
    payload = _compose_phrase_grounded_selection(
        query=str(args.query).strip(),
        query_plan_payload=query_plan_payload,
        candidates=candidates,
        pair_candidates=pair_candidates,
        test_times=_test_time_values(run_dir),
        tracks_payload=tracks_payload,
        raw_phrase_payload=raw_phrase_payload,
        raw_output=raw_output,
    )
    _write_json(output_path, payload)
    print(output_path)


if __name__ == "__main__":
    main()
