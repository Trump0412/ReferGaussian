#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _summary_lines(rows: list[dict]) -> list[str]:
    scene_counts = Counter(str(row.get("scene", "")) for row in rows)
    type_counts = Counter(str(row.get("query_type", "")) for row in rows)
    by_scene = defaultdict(Counter)
    for row in rows:
        by_scene[str(row.get("scene", ""))][str(row.get("query_type", ""))] += 1

    lines = [
        "# Filtered R4D-Bench Queries",
        "",
        f"- Query count: `{len(rows)}`",
        f"- Type counts: `{dict(type_counts)}`",
        "",
        "| Scene | Count | A | B | C |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for scene in sorted(scene_counts):
        lines.append(
            f"| {scene} | {scene_counts[scene]} | {by_scene[scene].get('A', 0)} | {by_scene[scene].get('B', 0)} | {by_scene[scene].get('C', 0)} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter official R4D-Bench query metadata by scene.")
    parser.add_argument("--input-json", required=True, help="Official R4D-Bench_queries.json path.")
    parser.add_argument("--output-json", required=True, help="Filtered output JSON path.")
    parser.add_argument("--output-md", default=None, help="Optional markdown summary output path.")
    parser.add_argument(
        "--exclude-scenes",
        nargs="*",
        default=[],
        help="Scene names to drop, e.g. americano cut_roasted_beef",
    )
    parser.add_argument(
        "--include-scenes",
        nargs="*",
        default=[],
        help="Optional scene allowlist applied before exclusions.",
    )
    args = parser.parse_args()

    rows = _read_json(Path(args.input_json))
    if not isinstance(rows, list):
        raise SystemExit(f"Expected list payload: {args.input_json}")

    excluded = {str(scene).strip() for scene in args.exclude_scenes if str(scene).strip()}
    included = {str(scene).strip() for scene in args.include_scenes if str(scene).strip()}
    filtered = [
        row
        for row in rows
        if (not included or str(row.get("scene", "")).strip() in included)
        and str(row.get("scene", "")).strip() not in excluded
    ]

    _write_json(Path(args.output_json), filtered)
    if args.output_md:
        Path(args.output_md).write_text("\n".join(_summary_lines(filtered)) + "\n", encoding="utf-8")

    print(
        f"[info] input={len(rows)} filtered={len(filtered)} "
        f"included_scenes={sorted(included)} excluded_scenes={sorted(excluded)}"
    )


if __name__ == "__main__":
    main()
