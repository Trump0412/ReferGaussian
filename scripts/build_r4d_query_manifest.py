#!/usr/bin/env python3
"""
Build a JSONL query manifest from an R4D-Bench-QA benchmark file.

Reads the benchmark JSON (R4D-Bench-QA format), filters queries belonging to
the requested scenes (flexible matching against scene identifiers), and writes:

1. A JSONL manifest with query_id, query, run_dir, dataset_dir, output_root, gpu,
   scene, and the original benchmark query record.
2. ``query_root_map.json``   -- maps query_id -> output root subdirectory.
3. ``dataset_dir_map.json``  -- maps query_id -> dataset_dir.

The GPU assignment cycles over the indices supplied with ``--gpus``.

Usage:
    python build_r4d_query_manifest.py \
        --benchmark data/benchmarks/r4d_bench_qa/benchmark_all_queries.json \
        --scenes americano coffee_martini cook_spinach cut_lemon \
                 cut_roasted_beef espresso flame_salmon flame_steak \
                 keyboard sear_steak split_cookie torchchocolate \
        --output manifest.jsonl \
        --output-root /path/to/query/outputs
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import sys
from pathlib import Path

from query_text_utils import benchmark_record_with_english_query, choose_english_query_text, contains_cjk


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_REGISTRY_PATH = REPO_ROOT / "configs" / "benchmarks" / "release_protocols.json"
R4D_ENGLISH_QUERY_MAP_PATH = REPO_ROOT / "configs" / "benchmarks" / "r4d_query_text_en.json"
FORMAL_R4D_PROTOCOL = "release_r4d_dense89_renderer_consistent"
FORMAL_R4D_PROFILE = "r4d_renderer_consistent"
QUERY_TYPE_TO_CATEGORY = {
    "A": "temporal_single_target",
    "B": "multi_target_reasoning",
    "C": "zero_target_distractor",
}


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------

# Scene keys as used on the CLI -> (run path under GS_RUN_ROOT,
# dataset path under GS_DATA_ROOT). These are release layout hints,
# not algorithm branches.
SCENE_CONFIG: dict[str, tuple[str, str]] = {
    "americano": (
        "baseline_4dgs/hypernerf/americano",
        "hypernerf/misc/americano",
    ),
    "cut_lemon": (
        "baseline_4dgs/hypernerf/cut-lemon1",
        "hypernerf/interp/cut-lemon1",
    ),
    "split_cookie": (
        "baseline_4dgs/hypernerf/split-cookie",
        "hypernerf/misc/split-cookie",
    ),
    "espresso": (
        "baseline_4dgs/hypernerf/espresso",
        "hypernerf/misc/espresso",
    ),
    "keyboard": (
        "baseline_4dgs/hypernerf/keyboard",
        "hypernerf/misc/keyboard",
    ),
    "torchchocolate": (
        "baseline_4dgs/hypernerf/torchocolate",
        "hypernerf/interp/torchocolate",
    ),
    "coffee_martini": (
        "baseline_4dgs/dynerf/coffee_martini",
        "dynerf/coffee_martini",
    ),
    "cook_spinach": (
        "baseline_4dgs/dynerf/cook_spinach",
        "dynerf/cook_spinach",
    ),
    "flame_salmon": (
        "baseline_4dgs/dynerf/flame_salmon_1",
        "dynerf/flame_salmon_1",
    ),
    "flame_steak": (
        "baseline_4dgs/dynerf/flame_steak",
        "dynerf/flame_steak",
    ),
    "sear_steak": (
        "baseline_4dgs/dynerf/sear_steak",
        "dynerf/sear_steak",
    ),
    "cut_roasted_beef": (
        "baseline_4dgs/dynerf/cut_roasted_beef",
        "dynerf/cut_roasted_beef",
    ),
}


def _root_env_default(primary: str, deprecated: str, fallback: str) -> str:
    if os.environ.get(primary):
        return str(os.environ[primary])
    if os.environ.get(deprecated):
        print(
            f"[deprecated] {deprecated} is supported for compatibility; use {primary}",
            file=sys.stderr,
        )
        return str(os.environ[deprecated])
    return fallback


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol_registry() -> dict:
    payload = json.loads(PROTOCOL_REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("protocols"), dict):
        raise ValueError(f"Invalid release protocol registry: {PROTOCOL_REGISTRY_PATH}")
    return payload


def _query_list(payload: dict | list, *, label: str) -> list[dict]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = []
        for key in ("queries", "data", "predictions", "items", "entries"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
        if not rows:
            for value in payload.values():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    rows = value
                    break
    else:
        rows = []
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Could not locate a query-object list in {label}")
    return rows


def _normalise_scene_name(scene: object) -> str:
    return str(scene or "").strip().lower().replace("-", "_")


def _load_query_metadata(path: Path) -> dict[str, dict]:
    metadata_rows = _query_list(_read_json(path), label=str(path))
    metadata_by_id: dict[str, dict] = {}
    for row in metadata_rows:
        query_id = str(row.get("query_id") or "").strip()
        if not query_id:
            raise ValueError(f"Query metadata row lacks query_id: {row}")
        if query_id in metadata_by_id:
            raise ValueError(f"Duplicate query metadata id: {query_id}")
        metadata_by_id[query_id] = row
    return metadata_by_id


def _validate_formal_sources(
    *,
    benchmark_path: Path,
    query_metadata_path: Path,
    registry: dict,
) -> tuple[dict[str, dict], dict[str, str], dict[str, str]]:
    protocol = registry["protocols"].get(FORMAL_R4D_PROTOCOL)
    if not isinstance(protocol, dict):
        raise ValueError(f"Missing {FORMAL_R4D_PROTOCOL} in release registry")
    source_paths = {
        "benchmark": str(benchmark_path.resolve()),
        "query_metadata": str(query_metadata_path.resolve()),
        "english_query_map": str(R4D_ENGLISH_QUERY_MAP_PATH.resolve()),
    }
    source_hashes = {
        "benchmark_sha256": _sha256(benchmark_path),
        "query_metadata_sha256": _sha256(query_metadata_path),
        "english_query_map_sha256": _sha256(R4D_ENGLISH_QUERY_MAP_PATH),
    }
    expected_hashes = {
        "benchmark_sha256": str(protocol.get("dense_gt_sha256", "")),
        "query_metadata_sha256": str(protocol.get("query_metadata_sha256", "")),
        "english_query_map_sha256": str(protocol.get("english_query_map_sha256", "")),
    }
    for key, expected in expected_hashes.items():
        if source_hashes[key] != expected:
            raise ValueError(
                f"{FORMAL_R4D_PROTOCOL} {key} mismatch: expected {expected}, "
                f"got {source_hashes[key]}"
            )

    metadata_by_id = _load_query_metadata(query_metadata_path)

    english_map = _read_json(R4D_ENGLISH_QUERY_MAP_PATH)
    if not isinstance(english_map, dict):
        raise ValueError("R4D English query map must be a JSON object")
    english_by_id = {str(key): str(value) for key, value in english_map.items()}
    if set(metadata_by_id) != set(english_by_id):
        raise ValueError(
            "R4D query metadata IDs differ from the versioned English query map: "
            f"missing={sorted(set(english_by_id) - set(metadata_by_id))}, "
            f"unexpected={sorted(set(metadata_by_id) - set(english_by_id))}"
        )
    if len(metadata_by_id) != int(protocol.get("query_count", -1)):
        raise ValueError(
            f"{FORMAL_R4D_PROTOCOL} requires {protocol.get('query_count')} metadata rows; "
            f"got {len(metadata_by_id)}"
        )
    categories = Counter(
        QUERY_TYPE_TO_CATEGORY.get(str(row.get("query_type", "")), "unknown")
        for row in metadata_by_id.values()
    )
    expected_categories = {
        str(key): int(value) for key, value in protocol.get("category_counts", {}).items()
    }
    if dict(categories) != expected_categories:
        raise ValueError(
            f"R4D category counts differ: expected {expected_categories}, got {dict(categories)}"
        )
    return metadata_by_id, source_paths, source_hashes


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
    if path.is_absolute() or not path.parts or path.parts[0] not in {"baseline_4dgs", "4dgs", "refergaussian"}:
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
        default=None,
        help=(
            "Scene names to include. The formal dense protocol fixes all 12 scenes; "
            "an incomplete canary must list its subset."
        ),
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
        default=[0],
        help="GPU ids to assign round-robin in the manifest (portable default: 0).",
    )
    parser.add_argument(
        "--profile",
        default=FORMAL_R4D_PROFILE,
        help="Query evaluation profile recorded in every manifest row.",
    )
    parser.add_argument(
        "--protocol-id",
        choices=[FORMAL_R4D_PROTOCOL],
        default=None,
        help="Exact formal protocol identity. Required unless --allow-incomplete is used.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Build an explicitly incomplete, non-strict canary manifest.",
    )
    parser.add_argument(
        "--query-metadata",
        default=None,
        help=(
            "Pinned R4D-Bench_queries.json. Required by the formal protocol for "
            "query categories and source-hash validation."
        ),
    )
    parser.add_argument(
        "--run-root",
        default=_root_env_default("GS_RUN_ROOT", "REFERGAUSSIAN_RUN_ROOT", "runs"),
        help="Root containing ReferGaussian outputs (default: GS_RUN_ROOT or runs).",
    )
    parser.add_argument(
        "--data-root",
        default=_root_env_default("GS_DATA_ROOT", "REFERGAUSSIAN_DATA_ROOT", "data"),
        help="Root containing prepared datasets (default: GS_DATA_ROOT or data).",
    )
    parser.add_argument(
        "--run-namespace",
        default=os.environ.get("REFERGAUSSIAN_RUN_NAMESPACE", "baseline_4dgs"),
        help="4DGS run namespace beneath --run-root (default: baseline_4dgs).",
    )
    parser.add_argument(
        "--allow-non-english-query-text",
        action="store_true",
        help="Allow manifest rows whose final query text contains CJK characters. "
        "Release reproduction should not use this flag.",
    )
    args = parser.parse_args()

    if args.protocol_id and args.allow_incomplete:
        parser.error("--protocol-id and --allow-incomplete are mutually exclusive")
    if not args.protocol_id and not args.allow_incomplete:
        parser.error(
            f"choose --protocol-id {FORMAL_R4D_PROTOCOL} or explicitly pass --allow-incomplete"
        )
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must contain one or more unique GPU ids")
    try:
        registry = _load_protocol_registry()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    if args.protocol_id:
        if args.profile != FORMAL_R4D_PROFILE:
            parser.error(f"{FORMAL_R4D_PROTOCOL} requires {FORMAL_R4D_PROFILE}")
        protocol = registry["protocols"][FORMAL_R4D_PROTOCOL]
        formal_scenes = [str(scene) for scene in protocol.get("scenes", [])]
        if set(formal_scenes) != set(SCENE_CONFIG):
            parser.error(
                f"{FORMAL_R4D_PROTOCOL} registry scenes differ from the supported layout"
            )
        if args.scenes is not None and set(args.scenes) != set(formal_scenes):
            parser.error(
                f"{FORMAL_R4D_PROTOCOL} fixes scenes to {sorted(formal_scenes)}; "
                f"got {sorted(args.scenes)}"
            )
        args.scenes = formal_scenes
        if not args.query_metadata:
            parser.error(f"--query-metadata is required for {FORMAL_R4D_PROTOCOL}")
        if args.allow_non_english_query_text:
            parser.error("the formal R4D protocol does not allow non-English query text")
    elif not args.scenes:
        parser.error("--allow-incomplete requires an explicit --scenes subset")

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

    protocol_id = args.protocol_id or "incomplete_r4d_canary"
    protocol_complete = bool(args.protocol_id)
    metadata_by_id: dict[str, dict] = {}
    source_paths: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    if protocol_complete:
        query_metadata_path = Path(str(args.query_metadata))
        if not query_metadata_path.is_file():
            print(f"ERROR: query metadata file not found: {query_metadata_path}", file=sys.stderr)
            return 1
        try:
            metadata_by_id, source_paths, source_hashes = _validate_formal_sources(
                benchmark_path=benchmark_path,
                query_metadata_path=query_metadata_path,
                registry=registry,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    elif args.query_metadata:
        query_metadata_path = Path(str(args.query_metadata))
        if not query_metadata_path.is_file():
            print(f"ERROR: query metadata file not found: {query_metadata_path}", file=sys.stderr)
            return 1
        try:
            metadata_by_id = _load_query_metadata(query_metadata_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        source_paths = {
            "benchmark": str(benchmark_path.resolve()),
            "query_metadata": str(query_metadata_path.resolve()),
            "english_query_map": str(R4D_ENGLISH_QUERY_MAP_PATH.resolve()),
        }
        source_hashes = {
            "benchmark_sha256": _sha256(benchmark_path),
            "query_metadata_sha256": _sha256(query_metadata_path),
            "english_query_map_sha256": _sha256(R4D_ENGLISH_QUERY_MAP_PATH),
        }

    # The benchmark JSON may be structured in various ways.  We accept:
    #   - a list of query objects
    #   - a dict with a "queries" or "data" key containing a list
    try:
        queries_raw = _query_list(benchmark_data, label=str(benchmark_path))
    except ValueError:
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
        if query_id in query_root_map:
            print(f"ERROR: duplicate benchmark query id: {query_id}", file=sys.stderr)
            return 1
        metadata = metadata_by_id.get(query_id, {})
        query_category = QUERY_TYPE_TO_CATEGORY.get(str(metadata.get("query_type", "")), "")
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
            "query_category": query_category,
            "profile": args.profile,
            "protocol_id": protocol_id,
            "protocol_complete": protocol_complete,
            "protocol_registry_version": str(registry.get("registry_version", "")),
            "source_paths": source_paths,
            "source_hashes": source_hashes,
            "benchmark_record": benchmark_record_with_english_query(raw_query, str(query_text), query_text_source),
        }
        entries.append(entry)
        query_root_map[query_id] = query_output_dir
        dataset_dir_map[query_id] = dataset_dir

    if not entries:
        print("ERROR: no valid query entries produced after filtering.", file=sys.stderr)
        return 1

    if protocol_complete:
        protocol = registry["protocols"][FORMAL_R4D_PROTOCOL]
        actual_ids = {str(entry["query_id"]) for entry in entries}
        expected_ids = set(metadata_by_id)
        actual_scenes = {_normalise_scene_name(entry["scene"]) for entry in entries}
        expected_scenes = {
            _normalise_scene_name(scene) for scene in protocol.get("scenes", [])
        }
        category_counts = Counter(str(entry["query_category"]) for entry in entries)
        expected_categories = {
            str(key): int(value) for key, value in protocol.get("category_counts", {}).items()
        }
        formal_errors: list[str] = []
        if len(entries) != int(protocol.get("query_count", -1)):
            formal_errors.append(
                f"expected {protocol.get('query_count')} rows, got {len(entries)}"
            )
        if actual_ids != expected_ids:
            formal_errors.append(
                f"query IDs differ; missing={sorted(expected_ids - actual_ids)}, "
                f"unexpected={sorted(actual_ids - expected_ids)}"
            )
        if actual_scenes != expected_scenes:
            formal_errors.append(
                f"scenes differ; expected={sorted(expected_scenes)}, got={sorted(actual_scenes)}"
            )
        if dict(category_counts) != expected_categories:
            formal_errors.append(
                f"category counts differ; expected={expected_categories}, got={dict(category_counts)}"
            )
        for entry in entries:
            metadata_scene = _normalise_scene_name(metadata_by_id[str(entry["query_id"])].get("scene"))
            if metadata_scene != _normalise_scene_name(entry["scene"]):
                formal_errors.append(
                    f"{entry['query_id']}: metadata scene {metadata_scene!r} differs from "
                    f"manifest scene {_normalise_scene_name(entry['scene'])!r}"
                )
        if formal_errors:
            for error in formal_errors:
                print(f"ERROR: {FORMAL_R4D_PROTOCOL}: {error}", file=sys.stderr)
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
    print(f"Protocol:   {protocol_id} (complete={protocol_complete})")
    print(f"Profile:    {args.profile}")
    print(f"Run root:   {run_root}")
    print(f"Run namespace: {run_namespace}")
    print(f"Data root:  {data_root}")
    for scene_name in args.scenes:
        scene_count = sum(1 for e in entries if e["scene"] == scene_name)
        print(f"    {scene_name}: {scene_count} queries")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
