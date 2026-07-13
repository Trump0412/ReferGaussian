#!/usr/bin/env python3
"""
Build a JSONL manifest for public scene queries.

Generates one JSON line per query across 4 public scenes (13 queries total).
Each line contains query metadata, run/data paths, and an assigned GPU index
(alternating: GPU 0 for even indices, GPU 1 for odd indices).

Usage:
    python build_public_query_manifest.py \
        --output manifest.jsonl \
        --output-root /path/to/query/outputs \
        --profile boundary_shape_v2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from query_text_utils import contains_cjk

# ---------------------------------------------------------------------------
# Scene definitions
# ---------------------------------------------------------------------------

# Map scene name -> (run path under REFERGAUSSIAN_RUN_ROOT, dataset path under
# REFERGAUSSIAN_DATA_ROOT). These are public protocol locations, not algorithm
# branches; users can override roots for their local layout.
SCENE_PATHS: dict[str, tuple[str, str]] = {
    "americano": (
        "refergaussian/hypernerf/americano",
        "hypernerf/misc/americano",
    ),
    "espresso": (
        "refergaussian/hypernerf/espresso",
        "hypernerf/misc/espresso",
    ),
    "split-cookie": (
        "refergaussian/hypernerf/split-cookie",
        "hypernerf/misc/split-cookie",
    ),
    "chickchicken": (
        "refergaussian/hypernerf/chickchicken",
        "hypernerf/interp/chickchicken",
    ),
}

SCENE_QUERIES_ALL: dict[str, list[str]] = {
    "americano": [
        "the espresso cup",
        "the milk pitcher",
        "the red milk pitcher",
    ],
    "espresso": [
        "the empty glass cup",
        "the white cup",
        "the metal spoon",
        "the knife",
    ],
    "split-cookie": [
        "the left hand",
        "the right hand",
        "the knife",
    ],
    "chickchicken": [
        "the hand holding the knife",
        "the rubber chicken",
        "the spoon",
    ],
}

SCENE_QUERIES_TIME_SENSITIVE: dict[str, list[str]] = {
    "americano": [
        "the glass cup that is glasses contain light-colored liquid",
        "the glass cup that is liquid become darker in glasses",
    ],
    "espresso": [
        "the empty glass cup",
        "the full glass cup",
        "the glass cup with liquid above the midpoint of the cup",
    ],
    "split-cookie": [
        "the cookie broken into smaller pieces",
        "the complete cookie",
    ],
    "chickchicken": [
        "the closed chicken container",
        "the opened chicken container",
    ],
}

QUERY_SETS: dict[str, dict[str, list[str]]] = {
    "all": SCENE_QUERIES_ALL,
    "time_sensitive": SCENE_QUERIES_TIME_SENSITIVE,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Convert a query string into a filesystem-safe slug."""
    return (
        text.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def _validate_path(label: str, path_str: str) -> None:
    """Warn if a directory path does not exist (non-fatal)."""
    try:
        exists = Path(path_str).is_dir()
    except OSError as exc:
        print(
            f"[warn] {label} directory could not be checked: {path_str} ({exc})",
            file=sys.stderr,
        )
        return
    if not exists:
        print(
            f"[warn] {label} directory does not exist: {path_str}",
            file=sys.stderr,
        )


def _resolve_under_root(path_str: str, root_env: str, default_root: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    root = Path(os.environ.get(root_env, default_root))
    return str(root / path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a JSONL manifest for public scene queries."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the JSONL manifest.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Base directory under which per-query output directories are placed.",
    )
    parser.add_argument(
        "--profile",
        default="boundary_shape_v2",
        help="Query eval profile name (default: boundary_shape_v2).",
    )
    parser.add_argument(
        "--query-set",
        choices=sorted(QUERY_SETS.keys()),
        default="all",
        help="Which public query set to emit (default: all).",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="GPU ids to assign round-robin in the manifest (default: 0 1 2).",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_root = str(args.output_root)
    scene_queries = QUERY_SETS[str(args.query_set)]
    run_root = os.environ.get("REFERGAUSSIAN_RUN_ROOT", "runs")
    data_root = os.environ.get("REFERGAUSSIAN_DATA_ROOT", "data")

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    index = 0

    for scene_name in sorted(scene_queries.keys()):
        run_rel, dataset_rel = SCENE_PATHS[scene_name]
        run_dir = _resolve_under_root(run_rel, "REFERGAUSSIAN_RUN_ROOT", run_root)
        dataset_dir = _resolve_under_root(dataset_rel, "REFERGAUSSIAN_DATA_ROOT", data_root)
        queries = scene_queries[scene_name]

        _validate_path(f"[{scene_name}] run_dir", run_dir)
        _validate_path(f"[{scene_name}] dataset_dir", dataset_dir)

        for query_text in queries:
            if contains_cjk(query_text):
                print(
                    f"ERROR: public release query must be English: scene={scene_name} query={query_text!r}",
                    file=sys.stderr,
                )
                return 2
            query_slug = _slugify(query_text)
            query_id = f"{scene_name}__{query_slug}"
            gpu = int(args.gpus[index % len(args.gpus)])

            entry = {
                "query_id": query_id,
                "query": query_text,
                "run_dir": run_dir,
                "dataset_dir": dataset_dir,
                "output_root": output_root,
                "gpu": gpu,
                "scene": scene_name,
                "query_language": "en",
            }
            entries.append(entry)
            index += 1

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Summary
    gpu_counts = {int(gpu): sum(1 for e in entries if int(e["gpu"]) == int(gpu)) for gpu in args.gpus}
    print(f"Wrote {len(entries)} query entries to {output_path}")
    print("  " + "  |  ".join(f"GPU {gpu}: {count}" for gpu, count in gpu_counts.items()))
    print(f"  Profile: {args.profile}")
    print(f"  Query set: {args.query_set}")
    print(f"  Output root: {output_root}")
    print(f"  Run root: {run_root}")
    print(f"  Data root: {data_root}")
    for scene_name in sorted(scene_queries.keys()):
        scene_count = sum(1 for e in entries if e["scene"] == scene_name)
        print(f"    {scene_name}: {scene_count} queries")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
