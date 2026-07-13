#!/usr/bin/env python3
"""Aggregate complete per-scene public-protocol evaluations without reweighting scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ("Acc", "vIoU", "temporal_tIoU", "temporal_precision", "temporal_recall")


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def aggregate_payloads(payloads: list[dict[str, Any]], expected_queries: int | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    missing_ids: list[str] = []
    expected_total = 0
    for payload in payloads:
        coverage = payload.get("coverage", {}) if isinstance(payload.get("coverage"), dict) else {}
        expected_total += int(coverage.get("expected_queries", len(payload.get("queries", []))))
        missing_ids.extend(str(item) for item in coverage.get("missing_query_ids", []))
        for row in payload.get("queries", []):
            query_id = str(row.get("query_slug", ""))
            if not query_id:
                raise ValueError("Public evaluation row has no query_slug")
            if query_id in seen_ids:
                raise ValueError(f"Duplicate public query across evaluation inputs: {query_id}")
            seen_ids.add(query_id)
            rows.append(row)

    if expected_queries is not None and expected_total != int(expected_queries):
        raise ValueError(
            f"Expected {expected_queries} public queries but input reports declare {expected_total}"
        )
    if expected_total <= 0:
        raise ValueError("Cannot aggregate an empty public query set")
    valid = [row for row in rows if row.get("Acc") is not None]
    nonempty = [row for row in valid if int(row.get("temporal_gt_active_count", 0)) > 0]
    empty = [row for row in valid if int(row.get("temporal_gt_active_count", 0)) == 0]
    coverage_missing = sorted(set(missing_ids) | {str(row["query_slug"]) for row in rows if row.get("Acc") is None})
    return {
        "summary": {
            "expected_queries": expected_total,
            "valid_queries": len(valid),
            "complete": not coverage_missing and len(valid) == expected_total,
            **{key: _mean(valid, key) for key in METRICS},
            "nonempty_queries": len(nonempty),
            "nonempty_only": {key: _mean(nonempty, key) for key in METRICS},
            "zero_target_queries": len(empty),
            "zero_target_correct": sum(bool(row.get("empty_query_correct")) for row in empty),
            "zero_target_false_positive": sum(not bool(row.get("empty_query_correct")) for row in empty),
        },
        "missing_query_ids": coverage_missing,
        "queries": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Per-scene evaluator JSON files.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--expected-queries", type=int, default=None)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    input_paths = [Path(value) for value in args.inputs]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in input_paths]
    output = aggregate_payloads(payloads, expected_queries=args.expected_queries)
    output["inputs"] = [str(path) for path in input_paths]
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = output["summary"]
    if args.output_md:
        def pct(value: float | None) -> str:
            return "n/a" if value is None else f"{value * 100.0:.2f}%"

        lines = [
            "# Aggregated Public Query Evaluation",
            "",
            f"- Complete coverage: `{summary['complete']}` ({summary['valid_queries']} / {summary['expected_queries']})",
            f"- Acc / vIoU / tIoU: `{pct(summary['Acc'])}` / `{pct(summary['vIoU'])}` / `{pct(summary['temporal_tIoU'])}`",
            f"- Non-empty Acc / vIoU / tIoU: `{pct(summary['nonempty_only']['Acc'])}` / `{pct(summary['nonempty_only']['vIoU'])}` / `{pct(summary['nonempty_only']['temporal_tIoU'])}`",
            f"- Zero-target correctness: `{summary['zero_target_correct']} / {summary['zero_target_queries']}`",
        ]
        if output["missing_query_ids"]:
            lines.append(f"- Missing query ids: `{', '.join(output['missing_query_ids'])}`")
        Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.require_complete and not summary["complete"]:
        raise SystemExit(
            "Incomplete aggregated public evaluation: "
            f"{summary['valid_queries']} / {summary['expected_queries']} valid."
        )


if __name__ == "__main__":
    main()
