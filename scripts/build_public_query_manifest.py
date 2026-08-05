#!/usr/bin/env python3
"""Build JSONL manifests for public ReferGaussian queries.

The time-sensitive benchmark manifest is derived from the protocol generated
from ``video_annotations.json``. This keeps query text and query identifiers
aligned with the public evaluator instead of maintaining a second hand-written
copy of the temporal annotations. Time-agnostic manifests are derived from all
COCO categories that have at least one mask, matching 4D LangSplat evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from query_text_utils import contains_cjk


# These are dataset-layout hints, not algorithm branches.
SCENE_PATHS: dict[str, tuple[str, str]] = {
    "americano": ("baseline_4dgs/hypernerf/americano", "hypernerf/misc/americano"),
    "espresso": ("baseline_4dgs/hypernerf/espresso", "hypernerf/misc/espresso"),
    "split-cookie": ("baseline_4dgs/hypernerf/split-cookie", "hypernerf/misc/split-cookie"),
    "chickchicken": ("baseline_4dgs/hypernerf/chickchicken", "hypernerf/interp/chickchicken"),
}

QUERY_SETS = ("time_sensitive", "time_agnostic")
REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_REGISTRY_PATH = REPO_ROOT / "configs" / "benchmarks" / "release_protocols.json"
FORMAL_PUBLIC_PROTOCOLS = frozenset(
    {
        "paper_public3",
        "release_public4_extension",
        "paper_public3_time_agnostic",
        "release_public4_time_agnostic",
    }
)
PUBLIC_PROTOCOL_QUERY_IDS = {
    "paper_public3": frozenset(
        {
            "americano__the_glass_cup_that_is_glasses_contain_light_colored_liquid",
            "americano__the_glass_cup_that_is_liquid_become_darker_in_glasses",
            "espresso__the_empty_glass_cup",
            "espresso__the_full_glass_cup",
            "espresso__the_glass_cup_with_liquid_above_the_midpoint_of_the_cup",
            "split-cookie__the_cookie_broken_into_smaller_pieces",
            "split-cookie__the_complete_cookie",
        }
    ),
    "release_public4_extension": frozenset(
        {
            "americano__the_glass_cup_that_is_glasses_contain_light_colored_liquid",
            "americano__the_glass_cup_that_is_liquid_become_darker_in_glasses",
            "chickchicken__the_closed_chicken_container",
            "chickchicken__the_opened_chicken_container",
            "espresso__the_empty_glass_cup",
            "espresso__the_full_glass_cup",
            "espresso__the_glass_cup_with_liquid_above_the_midpoint_of_the_cup",
            "split-cookie__the_cookie_broken_into_smaller_pieces",
            "split-cookie__the_complete_cookie",
        }
    ),
    "paper_public3_time_agnostic": frozenset(
        {
            "americano__time_agnostic__coasters_braided_from_straw_and_black_thread",
            "americano__time_agnostic__glass_cup",
            "americano__time_agnostic__hands",
            "americano__time_agnostic__metal_cup",
            "americano__time_agnostic__tray",
            "espresso__time_agnostic__electronic_scales_with_cup",
            "espresso__time_agnostic__glass_cup",
            "espresso__time_agnostic__metal_cup",
            "espresso__time_agnostic__round_wooden_coasters",
            "espresso__time_agnostic__table",
            "espresso__time_agnostic__white_bottle",
            "split-cookie__time_agnostic__bare_hands",
            "split-cookie__time_agnostic__checkered_tablecloth",
            "split-cookie__time_agnostic__cookie",
            "split-cookie__time_agnostic__square_wooden_board",
        }
    ),
    "release_public4_time_agnostic": frozenset(
        {
            "americano__time_agnostic__coasters_braided_from_straw_and_black_thread",
            "americano__time_agnostic__glass_cup",
            "americano__time_agnostic__hands",
            "americano__time_agnostic__metal_cup",
            "americano__time_agnostic__tray",
            "chickchicken__time_agnostic__board",
            "chickchicken__time_agnostic__chicken_container",
            "chickchicken__time_agnostic__hands",
            "chickchicken__time_agnostic__white_chicken",
            "chickchicken__time_agnostic__yellow_chicken",
            "espresso__time_agnostic__electronic_scales_with_cup",
            "espresso__time_agnostic__glass_cup",
            "espresso__time_agnostic__metal_cup",
            "espresso__time_agnostic__round_wooden_coasters",
            "espresso__time_agnostic__table",
            "espresso__time_agnostic__white_bottle",
            "split-cookie__time_agnostic__bare_hands",
            "split-cookie__time_agnostic__checkered_tablecloth",
            "split-cookie__time_agnostic__cookie",
            "split-cookie__time_agnostic__square_wooden_board",
        }
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


def _validate_formal_identity(
    rows: list[tuple[str, str, str]],
    *,
    protocol_id: str,
    registry: dict,
) -> None:
    protocol = registry["protocols"].get(protocol_id)
    if not isinstance(protocol, dict):
        raise ValueError(f"Unknown public release protocol: {protocol_id}")
    expected_scenes = set(map(str, protocol.get("scenes", [])))
    expected_ids = set(PUBLIC_PROTOCOL_QUERY_IDS[protocol_id])
    actual_scenes = {scene for scene, _query_id, _query in rows}
    actual_ids = {query_id for _scene, query_id, _query in rows}
    if len(rows) != int(protocol.get("query_count", -1)):
        raise ValueError(
            f"{protocol_id} requires {protocol.get('query_count')} queries; got {len(rows)}"
        )
    if actual_scenes != expected_scenes:
        raise ValueError(
            f"{protocol_id} requires scenes {sorted(expected_scenes)}; got {sorted(actual_scenes)}"
        )
    if actual_ids != expected_ids:
        raise ValueError(
            f"{protocol_id} query IDs differ; missing={sorted(expected_ids - actual_ids)}, "
            f"unexpected={sorted(actual_ids - expected_ids)}"
        )


def _public_source_identity(
    annotation_root: Path,
    scene: str,
    *,
    registry: dict,
) -> tuple[dict[str, str], dict[str, str]]:
    annotation_dir = annotation_root / scene
    video_path = annotation_dir / "video_annotations.json"
    coco_path = annotation_dir / "train" / "_annotations.coco.json"
    expected = registry.get("public_annotation_sha256", {}).get(scene, {})
    for label, path, expected_hash in (
        ("video annotations", video_path, expected.get("video_annotations")),
        ("COCO annotations", coco_path, expected.get("coco_masks")),
    ):
        if not path.is_file():
            raise ValueError(f"Missing {label} for {scene}: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{scene} {label} hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
    return (
        {
            "video_annotations": str(video_path.resolve()),
            "coco_annotations": str(coco_path.resolve()),
        },
        {
            "video_annotations_sha256": str(expected["video_annotations"]),
            "coco_annotations_sha256": str(expected["coco_masks"]),
        },
    )


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
    if path.is_absolute() or not path.parts or path.parts[0] not in {"baseline_4dgs", "4dgs", "refergaussian"}:
        return str(path)
    return str(Path(namespace, *path.parts[1:]))


def _protocol_rows(
    protocol_path: Path,
    *,
    category: str,
) -> list[tuple[str, str, str]]:
    """Return ``(scene, query_id, query)`` rows from a generated protocol."""

    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    rows = payload.get("queries") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"Protocol has no query list: {protocol_path}")

    selected: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("category") != category:
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
        raise ValueError(f"Protocol contains no supported {category} queries: {protocol_path}")
    return selected


def _time_sensitive_protocol_rows(protocol_path: Path) -> list[tuple[str, str, str]]:
    """Backward-compatible helper for callers that only need state queries."""
    return _protocol_rows(protocol_path, category="temporal_state_reference")


def _filter_protocol_scenes(
    rows: list[tuple[str, str, str]],
    scenes: list[str] | None,
    *,
    category: str = "temporal_state_reference",
) -> list[tuple[str, str, str]]:
    if scenes is None:
        return rows
    requested = set(scenes)
    selected = [row for row in rows if row[0] in requested]
    found = {row[0] for row in selected}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"Protocol contains no {category} rows for: " + ", ".join(missing))
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
    profile: str,
    protocol_id: str,
    protocol_complete: bool,
    protocol_registry_version: str,
    annotation_dir: str = "",
    source_paths: dict[str, str] | None = None,
    source_hashes: dict[str, str] | None = None,
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
        "profile": profile,
        "protocol_id": protocol_id,
        "protocol_complete": bool(protocol_complete),
        "protocol_registry_version": protocol_registry_version,
        "annotation_dir": annotation_dir,
        "source_paths": dict(source_paths or {}),
        "source_hashes": dict(source_hashes or {}),
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
        required=True,
        help="Output of build_4dlangsplat_query_protocol.py.",
    )
    parser.add_argument(
        "--protocol-id",
        choices=sorted(FORMAL_PUBLIC_PROTOCOLS),
        default=None,
        help="Exact formal protocol identity. Required unless --allow-incomplete is used.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Build an explicitly incomplete, non-strict canary manifest.",
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
        "--annotation-root",
        default=None,
        help=(
            "4DLangSplat HyperNeRF-Annotation root. Formal protocols verify its "
            "video and COCO annotation hashes."
        ),
    )
    parser.add_argument(
        "--run-namespace",
        default=os.environ.get("REFERGAUSSIAN_RUN_NAMESPACE", "baseline_4dgs"),
        help="4DGS run namespace beneath --run-root (default: baseline_4dgs).",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=[0],
        help="GPU ids to assign round-robin in the manifest (portable default: 0).",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=sorted(SCENE_PATHS),
        default=None,
        help=(
            "Optional exact scene subset. Use americano split-cookie espresso "
            "for the three-scene paper protocol; omit for the four-scene extension."
        ),
    )
    args = parser.parse_args()

    if args.protocol_id and args.allow_incomplete:
        parser.error("--protocol-id and --allow-incomplete are mutually exclusive")
    if not args.protocol_id and not args.allow_incomplete:
        parser.error("choose a formal --protocol-id or explicitly pass --allow-incomplete")
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must contain one or more unique GPU ids")
    time_agnostic_protocol = bool(
        args.protocol_id and args.protocol_id.endswith("_time_agnostic")
    )
    if args.protocol_id and time_agnostic_protocol != (args.query_set == "time_agnostic"):
        parser.error(f"{args.protocol_id} is incompatible with --query-set {args.query_set}")
    allowed_profiles = (
        {"public_time_agnostic_v1"}
        if args.query_set == "time_agnostic"
        else {"public_time_boundary_gated_v5", "public_time_boundary_gated_v5_numeric"}
    )
    if args.protocol_id and args.profile not in allowed_profiles:
        parser.error(
            f"{args.protocol_id} requires one of: {', '.join(sorted(allowed_profiles))}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_root = str(args.output_root)
    run_root = str(args.run_root)
    data_root = str(args.data_root)
    run_namespace = str(args.run_namespace)
    entries: list[dict[str, object]] = []

    try:
        registry = _load_protocol_registry()
        protocol_path = Path(args.protocol_json)
        if not protocol_path.is_file():
            parser.error(f"protocol not found: {protocol_path}")
        if args.protocol_id:
            expected_scenes = list(registry["protocols"][args.protocol_id]["scenes"])
            if args.scenes is not None and set(args.scenes) != set(expected_scenes):
                raise ValueError(
                    f"{args.protocol_id} fixes scenes to {sorted(expected_scenes)}; "
                    f"got {sorted(args.scenes)}"
                )
            requested_scenes = expected_scenes
        else:
            requested_scenes = args.scenes
        protocol_category = (
            "time_agnostic_reference"
            if args.query_set == "time_agnostic"
            else "temporal_state_reference"
        )
        source_rows = _filter_protocol_scenes(
            _protocol_rows(protocol_path, category=protocol_category),
            requested_scenes,
            category=protocol_category,
        )
        if args.protocol_id:
            _validate_formal_identity(
                source_rows,
                protocol_id=args.protocol_id,
                registry=registry,
            )

        protocol_id = args.protocol_id or "incomplete_public_canary"
        protocol_complete = bool(args.protocol_id)
        annotation_root = Path(
            args.annotation_root
            or Path(data_root) / "benchmarks" / "4dlangsplat" / "HyperNeRF-Annotation"
        )
        protocol_hash = _sha256(protocol_path)
        scene_sources: dict[str, tuple[dict[str, str], dict[str, str]]] = {}
        if protocol_complete:
            for scene in sorted({row[0] for row in source_rows}):
                source_paths, source_hashes = _public_source_identity(
                    annotation_root,
                    scene,
                    registry=registry,
                )
                source_paths["protocol_json"] = str(protocol_path.resolve())
                source_hashes["protocol_json_sha256"] = protocol_hash
                scene_sources[scene] = (source_paths, source_hashes)

        for index, (scene, query_id, query) in enumerate(source_rows):
            source_paths, source_hashes = scene_sources.get(scene, ({}, {}))
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
                    profile=args.profile,
                    protocol_id=protocol_id,
                    protocol_complete=protocol_complete,
                    protocol_registry_version=str(registry.get("registry_version", "")),
                    annotation_dir=(str((annotation_root / scene).resolve()) if protocol_complete else ""),
                    source_paths=source_paths,
                    source_hashes=source_hashes,
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
    print(f"  Protocol: {protocol_id} (complete={protocol_complete})")
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
