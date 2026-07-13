#!/usr/bin/env python3
"""Build JSONL manifests for public ReferGaussian queries.

The time-sensitive benchmark manifest is derived from the protocol generated
from ``video_annotations.json``. This keeps query text and query identifiers
aligned with the public evaluator instead of maintaining a second hand-written
copy of the temporal annotations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from query_text_utils import contains_cjk


# These are dataset-layout hints, not algorithm branches.
SCENE_PATHS: dict[str, tuple[str, str]] = {
    "americano": ("refergaussian/hypernerf/americano", "hypernerf/misc/americano"),
    "espresso": ("refergaussian/hypernerf/espresso", "hypernerf/misc/espresso"),
    "split-cookie": ("refergaussian/hypernerf/split-cookie", "hypernerf/misc/split-cookie"),
    "chickchicken": ("refergaussian/hypernerf/chickchicken", "hypernerf/interp/chickchicken"),
}

# Exploratory object queries are intentionally separate from the official
# time-sensitive protocol. They do not have public benchmark masks by default.
SCENE_QUERIES_ALL: dict[str, list[str]] = {
    "americano": ["the espresso cup", "the milk pitcher", "the red milk pitcher"],
    "espresso": ["the empty glass cup", "the white cup", "the metal spoon", "the knife"],
    "split-cookie": ["the left hand", "the right hand", "the knife"],
    "chickchicken": ["the hand holding the knife", "the rubber chicken", "the spoon"],
}
QUERY_SETS = ("all", "time_sensitive")


def _slugify(text: str) -> str:
    return text.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def _validate_path(label: str, path_str: str) -> None:
    try:
        exists = Path(path_str).is_dir()
    except OSError as exc:
        print(f"[warn] {label} directory could not be checked: {path_str} ({exc})", file=sys.stderr)
        return
    if not exists:
        print(f"[warn] {label} directory does not exist: {path_str}", file=sys.stderr)


def _resolve_under_root(path_str: str, root: str) -> str:
    path = Path(path_str)
    return str(path if path.is_absolute() else Path(root) / path)


def _run_relative_path(run_path: str, namespace: str) -> str:
    """Replace only the release output namespace in a documented run layout."""
    path = Path(run_path)
    if path.is_absolute() or not path.parts or path.parts[0] != "refergaussian":
        return str(path)
    return str(Path(namespace, *path.parts[1:]))


def _time_sensitive_protocol_rows(protocol_path: Path) -> list[tuple[str, str, str]]:
    """Return ``(scene, query_id, query)`` rows from a generated protocol."""

    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    rows = payload.get("queries") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"Protocol has no query list: {protocol_path}")

    selected: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("category") != "temporal_state_reference":
            continue
        scene = str(row.get("scene", "")).rsplit("/", 1)[-1]
        query_id = str(row.get("query_slug", "")).strip()
        query = str(row.get("query", "")).strip()
        if scene not in SCENE_PATHS or not query_id or not query:
            continue
        if query_id in seen:
            raise ValueError(f"Duplicate public protocol query id: {query_id}")
        seen.add(query_id)
        selected.append((scene, query_id, query))

    if not selected:
        raise ValueError(f"Protocol contains no supported time-sensitive queries: {protocol_path}")
    return selected


def _build_entry(
    *,
    scene: str,
    query_id: str,
    query: str,
    output_root: str,
    gpu: int,
    run_root: str,
    data_root: str,
    run_namespace: str,
) -> dict[str, object]:
    if contains_cjk(query):
        raise ValueError(f"Public release query must be English: scene={scene} query={query!r}")
    run_rel, dataset_rel = SCENE_PATHS[scene]
    return {
        "query_id": query_id,
        "query": query,
        "run_dir": _resolve_under_root(_run_relative_path(run_rel, run_namespace), run_root),
        "dataset_dir": _resolve_under_root(dataset_rel, data_root),
        "output_root": output_root,
        "gpu": int(gpu),
        "scene": scene,
        "query_language": "en",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Path to write the JSONL manifest.")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Base directory under which per-query output directories are placed.",
    )
    parser.add_argument(
        "--profile",
        default="boundary_shape_v2",
        help="Query evaluation profile recorded in the manifest summary.",
    )
    parser.add_argument(
        "--query-set",
        choices=QUERY_SETS,
        default="time_sensitive",
        help="Query set to emit (default: time_sensitive).",
    )
    parser.add_argument(
        "--protocol-json",
        help="Output of build_4dlangsplat_query_protocol.py; required for time_sensitive.",
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
        "--gpus",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="GPU ids to assign round-robin in the manifest (default: 0 1 2).",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_root = str(args.output_root)
    run_root = str(args.run_root)
    data_root = str(args.data_root)
    run_namespace = str(args.run_namespace)
    entries: list[dict[str, object]] = []

    try:
        if args.query_set == "time_sensitive":
            if not args.protocol_json:
                parser.error("--protocol-json is required when --query-set time_sensitive")
            protocol_path = Path(args.protocol_json)
            if not protocol_path.is_file():
                parser.error(f"protocol not found: {protocol_path}")
            source_rows = _time_sensitive_protocol_rows(protocol_path)
        else:
            source_rows = [
                (scene, f"{scene}__{_slugify(query)}", query)
                for scene in sorted(SCENE_QUERIES_ALL)
                for query in SCENE_QUERIES_ALL[scene]
            ]

        for index, (scene, query_id, query) in enumerate(source_rows):
            entries.append(
                _build_entry(
                    scene=scene,
                    query_id=query_id,
                    query=query,
                    output_root=output_root,
                    gpu=args.gpus[index % len(args.gpus)],
                    run_root=run_root,
                    data_root=data_root,
                    run_namespace=run_namespace,
                )
            )
    except ValueError as exc:
        parser.error(str(exc))

    for scene in sorted({str(entry["scene"]) for entry in entries}):
        run_rel, dataset_rel = SCENE_PATHS[scene]
        _validate_path(
            f"[{scene}] run_dir",
            _resolve_under_root(_run_relative_path(run_rel, run_namespace), run_root),
        )
        _validate_path(f"[{scene}] dataset_dir", _resolve_under_root(dataset_rel, data_root))

    with output_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    gpu_counts = {gpu: sum(int(entry["gpu"]) == gpu for entry in entries) for gpu in args.gpus}
    print(f"Wrote {len(entries)} query entries to {output_path}")
    print("  " + "  |  ".join(f"GPU {gpu}: {count}" for gpu, count in gpu_counts.items()))
    print(f"  Profile: {args.profile}")
    print(f"  Query set: {args.query_set}")
    print(f"  Output root: {output_root}")
    print(f"  Run root: {run_root}")
    print(f"  Run namespace: {run_namespace}")
    print(f"  Data root: {data_root}")
    for scene in sorted({str(entry["scene"]) for entry in entries}):
        print(f"    {scene}: {sum(entry['scene'] == scene for entry in entries)} queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
