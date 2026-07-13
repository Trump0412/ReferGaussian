#!/usr/bin/env python3
"""
Build a JSONL query manifest from an R4D-Bench-QA benchmark file.

Reads the benchmark JSON (R4D-Bench-QA format), filters queries belonging to
the requested scenes (flexible matching against scene identifiers), and writes:

1. A JSONL manifest with query_id, query, run_dir, dataset_dir, output_root, gpu,
   scene, and the original benchmark query record.
2. ``query_root_map.json``   -- maps query_id -> output root subdirectory.
3. ``dataset_dir_map.json``  -- maps query_id -> dataset_dir.

The GPU assignment alternates (GPU 0 for even indices, GPU 1 for odd).

Usage:
    python build_r4d_query_manifest.py \
        --benchmark data/benchmarks/r4d_bench_qa/benchmark_all_queries.json \
        --scenes coffee_martini sear_steak cut_roasted_beef \
                 cut_lemon espresso keyboard split_cookie torchchocolate \
        --output manifest.jsonl \
        --output-root /path/to/query/outputs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from query_text_utils import benchmark_record_with_english_query, choose_english_query_text, contains_cjk


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------

# Scene keys as used on the CLI -> (run path under REFERGAUSSIAN_RUN_ROOT,
# dataset path under REFERGAUSSIAN_DATA_ROOT). These are release layout hints,
# not algorithm branches.
SCENE_CONFIG: dict[str, tuple[str, str]] = {
    "cut_lemon": (
        "refergaussian/hypernerf/cut-lemon1",
        "hypernerf/interp/cut-lemon1",
    ),
    "split_cookie": (
        "refergaussian/hypernerf/split-cookie",
        "hypernerf/misc/split-cookie",
    ),
    "espresso": (
        "refergaussian/hypernerf/espresso",
        "hypernerf/misc/espresso",
    ),
    "keyboard": (
        "refergaussian/hypernerf/keyboard",
        "hypernerf/misc/keyboard",
    ),
    "torchchocolate": (
        "refergaussian/hypernerf/torchocolate",
        "hypernerf/interp/torchocolate",
    ),
    "coffee_martini": (
        "refergaussian/dynerf/coffee_martini",
        "dynerf/coffee_martini",
    ),
    "sear_steak": (
        "refergaussian/dynerf/sear_steak",
        "dynerf/sear_steak",
    ),
    "cut_roasted_beef": (
        "refergaussian/dynerf/cut_roasted_beef",
        "dynerf/cut_roasted_beef",
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict | list:
    """Read and parse a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: object) -> None:
    """Write JSON, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)


def _slugify(text: str) -> str:
    """Convert a query string into a filesystem-safe slug."""
    return (
        text.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def _query_id_from_record(raw_query: dict, matched_scene: str, fallback_index: int) -> str:
    """Return the official benchmark query id when available.

    The official evaluator indexes ``query_root_map.json`` by this field.  Using
    a generated slug here makes the evaluator report zero valid queries even
    when every query output exists.
    """
    for key in ("query_id", "id", "qid"):
        value = raw_query.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    query_text = (
        raw_query.get("query_en")
        or raw_query.get("question_en")
        or raw_query.get("text_en")
        or raw_query.get("caption_en")
        or raw_query.get("query")
        or raw_query.get("query_text")
        or raw_query.get("question")
        or raw_query.get("text")
        or raw_query.get("caption")
        or ""
    )
    slug = _slugify(str(query_text))[:80] or f"query_{fallback_index:04d}"
    return f"{matched_scene}__{slug}__{fallback_index:04d}"


def _scene_matches(benchmark_scene_id: str, cli_scene: str) -> bool:
    """Flexible scene matching.

    Checks whether *cli_scene* (lowercased, underscores/whitespace collapsed)
    appears in the benchmark scene identifier (same normalisation).
    """
    norm_bench = " ".join(benchmark_scene_id.strip().lower().replace("-", " ").replace("_", " ").split())
    norm_cli = " ".join(cli_scene.strip().lower().replace("-", " ").replace("_", " ").split())
    return norm_cli in norm_bench


def _record_matches_scene(raw_query: dict, cli_scene: str) -> bool:
    scene_variants = {
        cli_scene.strip().lower(),
        cli_scene.strip().lower().replace("_", "-"),
        cli_scene.strip().lower().replace("-", "_"),
        cli_scene.strip().lower().replace("_", " ").replace("-", " "),
        cli_scene.strip().lower().replace("_", "").replace("-", ""),
    }
    qid = str(raw_query.get("query_id") or raw_query.get("id") or raw_query.get("qid") or "").strip().lower()
    qid_compact = qid.replace("_", "").replace("-", "")
    for variant in scene_variants:
        if variant and (qid.startswith(variant) or qid_compact.startswith(variant.replace("_", "").replace("-", "").replace(" ", ""))):
            return True
    raw_str = json.dumps(raw_query, default=str, ensure_ascii=False).lower()
    raw_compact = raw_str.replace("_", "").replace("-", "").replace(" ", "")
    return any(
        bool(variant) and (variant in raw_str or variant.replace("_", "").replace("-", "").replace(" ", "") in raw_compact)
        for variant in scene_variants
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


def _resolve_under_root(path_str: str, root: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str(Path(root) / path)


def _run_relative_path(run_path: str, namespace: str) -> str:
    """Replace only the release output namespace in a documented run layout."""
    path = Path(run_path)
    if path.is_absolute() or not path.parts or path.parts[0] != "refergaussian":
        return str(path)
    return str(Path(namespace, *path.parts[1:]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a JSONL query manifest from an R4D-Bench-QA benchmark file."
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to the R4D-Bench-QA benchmark JSON file.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        required=True,
        help="List of scene names to include (e.g. cut_lemon split_cookie).",
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
        "--gpus",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="GPU ids to assign round-robin in the manifest (default: 0 1 2).",
    )
    parser.add_argument(
        "--run-root",
        default=os.environ.get("REFERGAUSSIAN_RUN_ROOT", "runs"),
        help="Root containing ReferGaussian outputs (default: REFERGAUSSIAN_RUN_ROOT or runs).",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("REFERGAUSSIAN_DATA_ROOT", "data"),
        help="Root containing prepared datasets (default: REFERGAUSSIAN_DATA_ROOT or data).",
    )
    parser.add_argument(
        "--run-namespace",
        default=os.environ.get("REFERGAUSSIAN_RUN_NAMESPACE", "refergaussian"),
        help="Run namespace beneath --run-root (default: REFERGAUSSIAN_RUN_NAMESPACE or refergaussian).",
    )
    parser.add_argument(
        "--allow-non-english-query-text",
        action="store_true",
        help="Allow manifest rows whose final query text contains CJK characters. "
        "Release reproduction should not use this flag.",
    )
    args = parser.parse_args()

    # --- Validate requested scenes ---
    unknown = [s for s in args.scenes if s not in SCENE_CONFIG]
    if unknown:
        print(
            f"ERROR: unknown scene(s): {sorted(unknown)}. "
            f"Available: {sorted(SCENE_CONFIG.keys())}",
            file=sys.stderr,
        )
        return 1

    # --- Load benchmark ---
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.is_file():
        print(f"ERROR: benchmark file not found: {benchmark_path}", file=sys.stderr)
        return 1

    benchmark_data = _read_json(benchmark_path)

    # The benchmark JSON may be structured in various ways.  We accept:
    #   - a list of query objects
    #   - a dict with a "queries" or "data" key containing a list
    queries_raw: list[dict] = []
    if isinstance(benchmark_data, list):
        queries_raw = benchmark_data
    elif isinstance(benchmark_data, dict):
        for key in ("queries", "data", "predictions", "items", "entries"):
            candidate = benchmark_data.get(key)
            if isinstance(candidate, list):
                queries_raw = candidate
                break
        # Fallback: treat top-level values that are lists of dicts
        if not queries_raw:
            for value in benchmark_data.values():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    queries_raw = value
                    break

    if not queries_raw:
        print(
            "ERROR: could not locate a list of query objects in the benchmark file. "
            "Expected a top-level list, or a dict containing a 'queries' key.",
            file=sys.stderr,
        )
        return 1

    # --- Filter queries by scene ---
    filtered: list[dict] = []
    for raw_query in queries_raw:
        if not isinstance(raw_query, dict):
            continue

        # Try to find a scene identifier in the query object.
        # Common keys: "scene", "scene_name", "scene_id", "video_id"
        scene_id = (
            raw_query.get("scene")
            or raw_query.get("scene_name")
            or raw_query.get("scene_id")
            or raw_query.get("video_id")
            or raw_query.get("id", "")
        )

        # Determine which requested scene this query belongs to
        matched_scene: str | None = None
        for cli_scene in args.scenes:
            if _scene_matches(str(scene_id), cli_scene):
                matched_scene = cli_scene
                break

        if matched_scene is None:
            # Also try matching against the full JSON string representation
            for cli_scene in args.scenes:
                if _record_matches_scene(raw_query, cli_scene):
                    matched_scene = cli_scene
                    break

        if matched_scene is None:
            continue

        filtered.append({**raw_query, "_resolved_scene": matched_scene})

    if not filtered:
        print(
            f"ERROR: no queries matched the requested scenes {args.scenes}. "
            f"Check that the benchmark file contains these scenes.",
            file=sys.stderr,
        )
        return 1

    # --- Build manifest entries ---
    output_root = str(args.output_root)
    run_root = str(args.run_root)
    data_root = str(args.data_root)
    run_namespace = str(args.run_namespace)
    entries: list[dict] = []
    query_root_map: dict[str, str] = {}
    dataset_dir_map: dict[str, str] = {}

    for idx, raw_query in enumerate(filtered):
        matched_scene = raw_query["_resolved_scene"]
        run_rel, dataset_rel = SCENE_CONFIG[matched_scene]
        run_dir = _resolve_under_root(_run_relative_path(run_rel, run_namespace), run_root)
        dataset_dir = _resolve_under_root(dataset_rel, data_root)

        # Determine the release query text.  Open-source reproduction uses
        # English question text only; legacy mixed-language benchmark files are
        # translated through the official query_id -> text_en map.
        query_text, query_text_source = choose_english_query_text(raw_query)
        if not query_text:
            print(
                f"[warn] skipping query at index {idx} in scene {matched_scene}: "
                f"no English query text found",
                file=sys.stderr,
            )
            continue
        if contains_cjk(query_text) and not args.allow_non_english_query_text:
            print(
                f"ERROR: non-English query text selected for "
                f"{_query_id_from_record(raw_query, matched_scene, idx)}: {query_text!r}",
                file=sys.stderr,
            )
            return 2

        query_id = _query_id_from_record(raw_query, matched_scene, idx)
        gpu = int(args.gpus[len(entries) % len(args.gpus)])

        # Per-query output subdirectory
        query_output_dir = os.path.join(output_root, query_id)

        entry = {
            "query_id": query_id,
            "query": str(query_text),
            "run_dir": run_dir,
            "dataset_dir": dataset_dir,
            "output_root": output_root,
            "gpu": gpu,
            "scene": matched_scene,
            "query_language": "en",
            "query_text_source": query_text_source,
            "benchmark_record": benchmark_record_with_english_query(raw_query, str(query_text), query_text_source),
        }
        entries.append(entry)
        query_root_map[query_id] = query_output_dir
        dataset_dir_map[query_id] = dataset_dir

    if not entries:
        print("ERROR: no valid query entries produced after filtering.", file=sys.stderr)
        return 1

    # --- Validate paths (non-fatal warnings) ---
    for scene_name in args.scenes:
        run_rel, dataset_rel = SCENE_CONFIG[scene_name]
        run_dir = _resolve_under_root(_run_relative_path(run_rel, run_namespace), run_root)
        dataset_dir = _resolve_under_root(dataset_rel, data_root)
        _validate_path(f"[{scene_name}] run_dir", run_dir)
        _validate_path(f"[{scene_name}] dataset_dir", dataset_dir)

    # --- Write outputs ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # JSONL manifest
    with open(output_path, "w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    # query_root_map.json (side-by-side with the manifest)
    qrm_path = output_path.parent / "query_root_map.json"
    _write_json(qrm_path, query_root_map)

    # dataset_dir_map.json
    ddm_path = output_path.parent / "dataset_dir_map.json"
    _write_json(ddm_path, dataset_dir_map)

    # --- Summary ---
    gpu_counts = {int(gpu): sum(1 for e in entries if int(e["gpu"]) == int(gpu)) for gpu in args.gpus}
    print(f"Benchmark:  {benchmark_path}")
    print(f"Total raw:  {len(queries_raw)} queries")
    print(f"Filtered:   {len(entries)} queries across {len(args.scenes)} scenes")
    print("  " + "  |  ".join(f"GPU {gpu}: {count}" for gpu, count in gpu_counts.items()))
    print(f"Manifest:   {output_path}")
    print(f"Root map:   {qrm_path}")
    print(f"Dataset map:{ddm_path}")
    print(f"Output root:{output_root}")
    print(f"Run root:   {run_root}")
    print(f"Run namespace: {run_namespace}")
    print(f"Data root:  {data_root}")
    for scene_name in args.scenes:
        scene_count = sum(1 for e in entries if e["scene"] == scene_name)
        print(f"    {scene_name}: {scene_count} queries")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
