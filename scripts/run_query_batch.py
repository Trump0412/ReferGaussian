#!/usr/bin/env python3
"""
Multi-GPU batch query runner.

Reads a JSONL manifest where each line describes a query to run on a specific GPU.
Each GPU processes its assigned queries serially; GPUs run in parallel.

Manifest fields per line (JSON):
    query_id      : str   — unique identifier for this query
    query         : str   — natural-language query text
    run_dir       : str   — path to the 3D scene run directory
    dataset_dir   : str   — path to the dataset directory
    output_root   : str   — directory where query outputs and traces are written
    gpu           : int   — preferred GPU index from the runner's allowed list
    annotation_dir: str   — (optional) path to annotation directory for diagnostics
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import platform
import selectors
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "run_query_specific_worldtube_pipeline.sh"
_RELEASE_CONFIG_PREFIXES = ("QUERY_", "GS_QUERY_", "GSAM2_")
EXPLORATORY_DEFAULT_PROFILE = "boundary_shape_v2"
PROTOCOL_REGISTRY_PATH = REPO_ROOT / "configs" / "benchmarks" / "release_protocols.json"
R4D_ENGLISH_QUERY_MAP_PATH = REPO_ROOT / "configs" / "benchmarks" / "r4d_query_text_en.json"
STRICT_RELEASE_PROFILES = frozenset(
    {
        "public_time_boundary_gated_v5",
        "public_time_boundary_gated_v5_numeric",
        "r4d_boundary_gated_v5",
        "r4d_renderer_consistent",
    }
)
R4D_RELEASE_PROTOCOLS = frozenset(
    {"release_r4d_dense89", "release_r4d_dense89_renderer_consistent"}
)
EXECUTABLE_RELEASE_PROTOCOLS = frozenset(
    {
        "paper_public3",
        "release_public4_extension",
        *R4D_RELEASE_PROTOCOLS,
    }
)
PROTOCOL_PROFILES = {
    "paper_public3": frozenset(
        {"public_time_boundary_gated_v5", "public_time_boundary_gated_v5_numeric"}
    ),
    "release_public4_extension": frozenset(
        {"public_time_boundary_gated_v5", "public_time_boundary_gated_v5_numeric"}
    ),
    "release_r4d_dense89": frozenset({"r4d_boundary_gated_v5"}),
    "release_r4d_dense89_renderer_consistent": frozenset(
        {"r4d_renderer_consistent"}
    ),
}
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
}
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run queries from a JSONL manifest on one or more GPUs in parallel."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to JSONL manifest file (one JSON object per line).",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help=(
            "Query eval profile name. Required with --strict-release; "
            f"non-strict runs default to {EXPLORATORY_DEFAULT_PROFILE}."
        ),
    )
    parser.add_argument(
        "--gpu",
        nargs="+",
        type=int,
        default=[0],
        help="GPU indices to use (portable default: 0).",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Set QUERY_FORCE_RERUN=1 so the pipeline ignores cached results.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-query timeout in seconds (default: 3600, i.e. 1 hour).",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Keep the batch process exit code at 0 even if some queries fail; failures are still logged.",
    )
    parser.add_argument(
        "--strict-release",
        action="store_true",
        help=(
            "Require --force-rerun, reject manifest environment overrides, and "
            "clear inherited query/GSAM tuning variables."
        ),
    )
    parser.add_argument(
        "--protocol-id",
        default=None,
        help=(
            "Release protocol declared by the manifest. Required with "
            "--strict-release (paper_public3, release_public4_extension, or "
            "release_r4d_dense89, or release_r4d_dense89_renderer_consistent)."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def load_manifest_with_audit(path: str, *, strict: bool = False) -> tuple[list[dict], dict]:
    """Load a JSONL manifest and retain an explicit record of skipped rows."""
    items: list[dict] = []
    required = {"query_id", "query", "run_dir", "dataset_dir", "output_root", "gpu"}
    issues: list[str] = []
    seen_query_ids: set[str] = set()
    nonempty_lines = 0

    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            nonempty_lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(f"line {lineno}: invalid JSON ({exc})")
                continue

            if not isinstance(obj, dict):
                issues.append(f"line {lineno}: expected a JSON object")
                continue

            missing = required - obj.keys()
            if missing:
                issues.append(f"line {lineno}: missing keys {sorted(missing)}")
                continue

            query_id = str(obj.get("query_id") or "").strip()
            if not query_id:
                issues.append(f"line {lineno}: query_id must be a non-empty string")
                continue
            if query_id in seen_query_ids:
                issues.append(f"line {lineno}: duplicate query_id {query_id!r}")
                continue
            if not str(obj.get("query") or "").strip():
                issues.append(f"line {lineno}: query text must be non-empty")
                continue
            if isinstance(obj.get("gpu"), bool) or not isinstance(obj.get("gpu"), int):
                issues.append(f"line {lineno}: gpu must be a JSON integer")
                continue

            seen_query_ids.add(query_id)
            items.append(obj)

    if issues and strict:
        raise ValueError("invalid strict-release manifest:\n" + "\n".join(issues))
    for issue in issues:
        print(f"[warn] skipping manifest row: {issue}", file=sys.stderr)
    return items, {
        "nonempty_rows": nonempty_lines,
        "loaded_rows": len(items),
        "skipped_rows": len(issues),
        "issues": issues,
    }


def load_manifest(path: str, *, strict: bool = False) -> list[dict]:
    """Compatibility helper returning only valid rows."""
    return load_manifest_with_audit(path, strict=strict)[0]


def group_by_gpu(
    items: list[dict],
    gpus: list[int],
    *,
    strict: bool = False,
) -> dict[int, list[dict]]:
    """Partition manifest items by their assigned GPU."""
    groups: dict[int, list[dict]] = {g: [] for g in gpus}
    for item in items:
        gpu = item["gpu"]
        if gpu not in groups:
            message = (
                f"query {item['query_id']!r} is assigned to GPU {gpu}, "
                f"outside active GPU list {sorted(groups)}"
            )
            if strict:
                raise ValueError(message)
            print(f"[warn] {message}; skipping", file=sys.stderr)
            continue
        groups[gpu].append(item)
    return groups


def _load_protocol_registry() -> dict:
    try:
        payload = json.loads(PROTOCOL_REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"release protocol registry is unreadable: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("protocols"), dict):
        raise ValueError("release protocol registry has no protocols object")
    return payload


def _sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_scene(scene: object) -> str:
    return str(scene or "").strip().lower().replace("-", "_")


def _expected_r4d_query_ids(registry: dict, protocol_id: str) -> set[str]:
    expected_hash = str(
        registry["protocols"][protocol_id].get("english_query_map_sha256", "")
    )
    actual_hash = _sha256_hex(R4D_ENGLISH_QUERY_MAP_PATH)
    if actual_hash != expected_hash:
        raise ValueError(
            "release R4D English query map does not match the protocol registry: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    payload = json.loads(R4D_ENGLISH_QUERY_MAP_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("release R4D English query map must be a JSON object")
    return {str(query_id) for query_id in payload}


def _source_identity_errors(
    items: list[dict],
    *,
    protocol_id: str,
    registry: dict,
    verify_source_files: bool,
) -> list[str]:
    errors: list[str] = []
    expected_registry_version = str(registry.get("registry_version", ""))
    for item in items:
        query_id = str(item.get("query_id", "<unknown>"))
        if str(item.get("protocol_registry_version", "")) != expected_registry_version:
            errors.append(
                f"{query_id}: protocol_registry_version differs from {expected_registry_version!r}"
            )

    if protocol_id in PUBLIC_PROTOCOL_QUERY_IDS:
        expected_by_scene = registry.get("public_annotation_sha256", {})
        checked_paths: set[tuple[str, str]] = set()
        for item in items:
            query_id = str(item.get("query_id", "<unknown>"))
            scene = str(item.get("scene", ""))
            expected = expected_by_scene.get(scene, {})
            hashes = item.get("source_hashes")
            paths = item.get("source_paths")
            if not isinstance(hashes, dict):
                errors.append(f"{query_id}: source_hashes must be an object")
                continue
            if hashes.get("video_annotations_sha256") != expected.get("video_annotations"):
                errors.append(f"{query_id}: video_annotations hash differs from registry")
            if hashes.get("coco_annotations_sha256") != expected.get("coco_masks"):
                errors.append(f"{query_id}: COCO annotation hash differs from registry")
            if not verify_source_files:
                continue
            if not isinstance(paths, dict):
                errors.append(f"{query_id}: source_paths must be an object")
                continue
            for key, hash_key in (
                ("video_annotations", "video_annotations_sha256"),
                ("coco_annotations", "coco_annotations_sha256"),
                ("protocol_json", "protocol_json_sha256"),
            ):
                source_path = Path(str(paths.get(key, "")))
                expected_hash = str(hashes.get(hash_key, ""))
                identity = (str(source_path), expected_hash)
                if identity in checked_paths:
                    continue
                checked_paths.add(identity)
                if not source_path.is_file():
                    errors.append(f"{query_id}: source file missing: {source_path}")
                elif not expected_hash or _sha256_hex(source_path) != expected_hash:
                    errors.append(f"{query_id}: source file hash mismatch: {source_path}")
    elif protocol_id in R4D_RELEASE_PROTOCOLS:
        protocol = registry["protocols"][protocol_id]
        expected_hashes = {
            "benchmark_sha256": str(protocol.get("dense_gt_sha256", "")),
            "query_metadata_sha256": str(protocol.get("query_metadata_sha256", "")),
            "english_query_map_sha256": str(protocol.get("english_query_map_sha256", "")),
        }
        checked_paths: set[tuple[str, str]] = set()
        for item in items:
            query_id = str(item.get("query_id", "<unknown>"))
            hashes = item.get("source_hashes")
            paths = item.get("source_paths")
            if not isinstance(hashes, dict):
                errors.append(f"{query_id}: source_hashes must be an object")
                continue
            for hash_key, expected_hash in expected_hashes.items():
                if hashes.get(hash_key) != expected_hash:
                    errors.append(f"{query_id}: {hash_key} differs from registry")
            if not verify_source_files:
                continue
            if not isinstance(paths, dict):
                errors.append(f"{query_id}: source_paths must be an object")
                continue
            for path_key, hash_key in (
                ("benchmark", "benchmark_sha256"),
                ("query_metadata", "query_metadata_sha256"),
                ("english_query_map", "english_query_map_sha256"),
            ):
                source_path = Path(str(paths.get(path_key, "")))
                expected_hash = expected_hashes[hash_key]
                identity = (str(source_path), expected_hash)
                if identity in checked_paths:
                    continue
                checked_paths.add(identity)
                if not source_path.is_file():
                    errors.append(f"{query_id}: source file missing: {source_path}")
                elif _sha256_hex(source_path) != expected_hash:
                    errors.append(f"{query_id}: source file hash mismatch: {source_path}")
    return errors


def validate_release_manifest(
    items: list[dict],
    profile: str,
    protocol_id: str | None = None,
    *,
    verify_source_files: bool = False,
) -> list[str]:
    """Return release-contract violations without changing exploratory manifests."""
    errors: list[str] = []
    if protocol_id not in EXECUTABLE_RELEASE_PROTOCOLS:
        return [
            "strict release requires --protocol-id to name an executable protocol: "
            + ", ".join(sorted(EXECUTABLE_RELEASE_PROTOCOLS))
        ]
    if profile not in PROTOCOL_PROFILES[protocol_id]:
        errors.append(f"protocol {protocol_id!r} does not allow profile {profile!r}")

    try:
        registry = _load_protocol_registry()
    except ValueError as exc:
        return [str(exc)]
    protocol = registry["protocols"].get(protocol_id)
    if not isinstance(protocol, dict):
        return [f"protocol {protocol_id!r} is absent from the release registry"]

    for item in items:
        query_id = str(item.get("query_id", "<unknown>"))
        if item.get("env") not in (None, {}):
            errors.append(f"{query_id}: strict release manifests cannot contain an env override")
        item_profile = str(item.get("profile") or "")
        if item_profile != profile:
            errors.append(
                f"{query_id}: manifest profile {item_profile!r} differs from --profile {profile!r}"
            )
        if item.get("protocol_id") != protocol_id:
            errors.append(
                f"{query_id}: manifest protocol {item.get('protocol_id')!r} "
                f"differs from --protocol-id {protocol_id!r}"
            )
        if item.get("protocol_complete") is not True:
            errors.append(f"{query_id}: formal protocol row is not marked complete")

    query_ids = [str(item.get("query_id", "")) for item in items]
    if len(query_ids) != int(protocol.get("query_count", -1)):
        errors.append(
            f"protocol {protocol_id!r} requires {protocol.get('query_count')} rows, "
            f"manifest has {len(query_ids)}"
        )
    if len(set(query_ids)) != len(query_ids):
        errors.append("strict release manifest contains duplicate query IDs")

    scenes = {_normalise_scene(item.get("scene")) for item in items}
    if protocol_id in PUBLIC_PROTOCOL_QUERY_IDS:
        expected_scenes = {_normalise_scene(scene) for scene in protocol.get("scenes", [])}
        expected_ids = set(PUBLIC_PROTOCOL_QUERY_IDS[protocol_id])
        if scenes != expected_scenes:
            errors.append(
                f"protocol {protocol_id!r} scenes differ: expected {sorted(expected_scenes)}, "
                f"got {sorted(scenes)}"
            )
        if set(query_ids) != expected_ids:
            errors.append(
                f"protocol {protocol_id!r} query IDs differ; "
                f"missing={sorted(expected_ids - set(query_ids))}, "
                f"unexpected={sorted(set(query_ids) - expected_ids)}"
            )
    elif protocol_id in R4D_RELEASE_PROTOCOLS:
        expected_scenes = {
            _normalise_scene(scene) for scene in protocol.get("scenes", [])
        }
        try:
            expected_ids = _expected_r4d_query_ids(registry, protocol_id)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(str(exc))
            expected_ids = set()
        if scenes != expected_scenes:
            errors.append(
                f"protocol {protocol_id!r} scenes differ: expected {sorted(expected_scenes)}, "
                f"got {sorted(scenes)}"
            )
        if set(query_ids) != expected_ids:
            errors.append(
                f"protocol {protocol_id!r} query IDs differ; "
                f"missing={sorted(expected_ids - set(query_ids))}, "
                f"unexpected={sorted(set(query_ids) - expected_ids)}"
            )
        category_counts = Counter(str(item.get("query_category", "")) for item in items)
        expected_categories = {
            str(name): int(count)
            for name, count in protocol.get("category_counts", {}).items()
        }
        if dict(category_counts) != expected_categories:
            errors.append(
                f"protocol {protocol_id!r} category counts differ: "
                f"expected {expected_categories}, got {dict(category_counts)}"
            )

    errors.extend(
        _source_identity_errors(
            items,
            protocol_id=protocol_id,
            registry=registry,
            verify_source_files=verify_source_files,
        )
    )
    return errors


def resolve_profile(profile: str | None, *, strict_release: bool) -> str:
    """Resolve the requested profile without silently weakening a release run."""

    if profile:
        if strict_release and profile not in STRICT_RELEASE_PROFILES:
            raise ValueError(
                f"--strict-release profile {profile!r} is not allowed; choose one of "
                + ", ".join(sorted(STRICT_RELEASE_PROFILES))
            )
        return profile
    if strict_release:
        raise ValueError("--strict-release requires an explicit --profile")
    return EXPLORATORY_DEFAULT_PROFILE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """Return a UTC timestamp string for logging."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _ensure_dir(path: str) -> None:
    """Create a directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def _write_json_atomic(path: str, payload: object) -> None:
    """Atomically write JSON to *path* via a temp file + rename."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _append_jsonl(path: str, record: dict) -> None:
    """Append a single JSON record as a line to a JSONL file."""
    _ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _clear_inherited_release_tuning(env: dict[str, str]) -> None:
    """Keep model/path variables but remove mutable query-pipeline tuning."""
    for key in list(env):
        if key.startswith(_RELEASE_CONFIG_PREFIXES):
            env.pop(key, None)


def _sha256_file(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": digest.hexdigest(),
        }
    except OSError:
        return None


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _safe_release_environment(env: dict[str, str]) -> dict[str, str]:
    prefixes = ("REFERGAUSSIAN_", "QWEN_", "GSAM2_", "GROUNDING_", "SAM2_", "HF_")
    forbidden_parts = ("TOKEN", "PASSWORD", "SECRET", "API_KEY", "ACCESS_KEY")
    return {
        key: value
        for key, value in sorted(env.items())
        if key.startswith(prefixes) and not any(part in key.upper() for part in forbidden_parts)
    }


def _resolve_qwen_model_dir(env: dict[str, str]) -> Path | None:
    configured = env.get("REFERGAUSSIAN_QWEN_MODEL") or env.get("QWEN_MODEL_PATH")
    candidates = [
        Path(configured).expanduser() if configured else None,
        REPO_ROOT / "models" / "Qwen3-VL-8B-Instruct",
        REPO_ROOT.parent / "models" / "Qwen3-VL-8B-Instruct",
        Path.home() / "models" / "Qwen3-VL-8B-Instruct",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            return candidate.resolve()
    return None


def _qwen_model_provenance(env: dict[str, str]) -> dict[str, object] | None:
    model_dir = _resolve_qwen_model_dir(env)
    if model_dir is None:
        return None
    metadata_names = (
        "refergaussian_snapshot.json",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "model.safetensors.index.json",
    )
    return {
        "root": str(model_dir),
        "metadata": [
            record
            for name in metadata_names
            if (record := _sha256_file(model_dir / name))
        ],
        "weight_files": [
            {"name": path.name, "bytes": int(path.stat().st_size)}
            for path in sorted(model_dir.glob("*.safetensors"))
        ],
    }


def _source_tree_release_errors(
    *,
    strict_release: bool,
    commit: str | None,
    status: str | None,
) -> list[str]:
    if strict_release and commit and status:
        return [
            "strict release requires a clean Git worktree; commit or remove "
            "the listed changes before running"
        ]
    return []


def _latest_point_cloud(run_dir: Path) -> Path | None:
    point_cloud_root = run_dir / "point_cloud"
    candidates: list[tuple[int, Path]] = []
    if not point_cloud_root.is_dir():
        return None
    for path in point_cloud_root.glob("iteration_*/point_cloud.ply"):
        try:
            iteration = int(path.parent.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        candidates.append((iteration, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def build_batch_provenance(
    items: list[dict],
    *,
    manifest_path: Path,
    profile: str,
    protocol_id: str,
    strict_release: bool,
    manifest_audit: dict | None = None,
) -> dict[str, object]:
    status = _git_output("status", "--porcelain=v1")
    diff = _git_output("diff", "--binary", "HEAD")
    source_files = [
        REPO_ROOT / "scripts" / "run_query_batch.py",
        REPO_ROOT / "scripts" / "run_query_specific_worldtube_pipeline.sh",
        REPO_ROOT / "scripts" / "query_eval_profiles.sh",
        REPO_ROOT / "scripts" / "evaluate_ours_benchmark.py",
        REPO_ROOT / "scripts" / "evaluate_public_query_protocol.py",
        REPO_ROOT / "configs" / "benchmarks" / "release_protocols.json",
        REPO_ROOT / "configs" / "benchmarks" / "r4d_query_text_en.json",
    ]
    runs = {}
    for run_dir_text in sorted({str(item["run_dir"]) for item in items}):
        run_dir = Path(run_dir_text)
        files = [run_dir / "config.yaml", run_dir / "metrics.json", run_dir / "results.json"]
        point_cloud = _latest_point_cloud(run_dir)
        if point_cloud is not None:
            files.append(point_cloud)
        runs[run_dir_text] = [record for path in files if (record := _sha256_file(path))]

    datasets = {}
    for dataset_dir_text in sorted({str(item["dataset_dir"]) for item in items}):
        dataset_dir = Path(dataset_dir_text)
        files = [
            dataset_dir / "metadata.json",
            dataset_dir / "dataset.json",
            dataset_dir / "scene.json",
            dataset_dir / "poses_bounds.npy",
        ]
        datasets[dataset_dir_text] = [
            record for path in files if (record := _sha256_file(path))
        ]

    return {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "profile": profile,
        "protocol_id": protocol_id,
        "strict_release": bool(strict_release),
        "manifest": _sha256_file(manifest_path),
        "manifest_audit": dict(manifest_audit or {}),
        "repository": {
            "root": str(REPO_ROOT),
            "commit": _git_output("rev-parse", "HEAD"),
            "dirty": bool(status),
            "status": status.splitlines() if status else [],
            "diff_sha256": hashlib.sha256((diff or "").encode("utf-8")).hexdigest(),
        },
        "runtime": {
            "hostname": socket.gethostname(),
            "python": sys.executable,
            "python_version": platform.python_version(),
        },
        "environment": _safe_release_environment(dict(os.environ)),
        "qwen_model": _qwen_model_provenance(dict(os.environ)),
        "source_files": [record for path in source_files if (record := _sha256_file(path))],
        "runs": runs,
        "datasets": datasets,
    }


# ---------------------------------------------------------------------------
# Single-query execution
# ---------------------------------------------------------------------------

def run_one_query(
    item: dict,
    *,
    profile: str,
    force_rerun: bool,
    timeout: int,
    strict_release: bool,
) -> dict:
    """Execute one query via the pipeline shell script.

    Returns an efficiency-trace dictionary.
    """
    query_id: str = item["query_id"]
    query_text: str = item["query"]
    run_dir: str = item["run_dir"]
    dataset_dir: str = item["dataset_dir"]
    output_root: str = item["output_root"]
    gpu: int = item["gpu"]
    annotation_dir: str = item.get("annotation_dir", "")

    query_output_root = os.path.join(output_root, query_id)
    mllm_trace_path = os.path.join(query_output_root, "mllm_trace.jsonl")
    log_dir = os.path.join(output_root, "logs")
    log_file = os.path.join(log_dir, f"gpu{gpu}_{query_id}.log")
    item_profile = str(item.get("profile") or profile)
    item_env = item.get("env") if isinstance(item.get("env"), dict) else {}

    _ensure_dir(log_dir)
    _ensure_dir(query_output_root)

    # Build environment
    env = dict(os.environ)
    if strict_release:
        _clear_inherited_release_tuning(env)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["QUERY_EVAL_PROFILE"] = item_profile
    if strict_release:
        env["REFERGAUSSIAN_STRICT_RELEASE"] = "1"
    if force_rerun:
        env["QUERY_FORCE_RERUN"] = "1"
    env["QUERY_OUTPUT_ROOT_OVERRIDE"] = query_output_root
    env["QUERY_GPU_LOCK_FILE"] = f"/tmp/refergaussian_query_gpu{gpu}.lock"
    env["MLLM_TRACE_PATH"] = mllm_trace_path
    if annotation_dir:
        env["QUERY_ANNOTATION_DIR"] = annotation_dir
    for key, value in item_env.items():
        if key:
            env[str(key)] = str(value)

    cmd = [
        "bash",
        str(PIPELINE_SCRIPT),
        run_dir,
        dataset_dir,
        query_text,
        query_id,
    ]

    # Log preamble
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"\n{'=' * 60}\n")
        lf.write(f"[{_utc_now()}] START  query_id={query_id}  gpu={gpu}\n")
        lf.write(f"[{_utc_now()}] CMD   {' '.join(cmd)}\n")
        lf.write(f"[{_utc_now()}] CWD   {REPO_ROOT}\n")
        lf.write(f"[{_utc_now()}] ENV   CUDA_VISIBLE_DEVICES={gpu}  "
                 f"QUERY_EVAL_PROFILE={item_profile}  force_rerun={force_rerun}  "
                 f"strict_release={strict_release}\n")
        if item_env:
            lf.write(f"[{_utc_now()}] ENV_OVERRIDES {json.dumps(item_env, ensure_ascii=False)}\n")
        lf.write(f"{'=' * 60}\n\n")
        lf.flush()

    start = time.monotonic()
    started_at_utc = _utc_now()
    exit_code: int = -1
    error_msg: str = ""

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        deadline = start + float(timeout)

        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write("--- PIPELINE OUTPUT ---\n")
            lf.flush()
            if proc.stdout is not None:
                selector = selectors.DefaultSelector()
                selector.register(proc.stdout, selectors.EVENT_READ)
                while True:
                    if time.monotonic() > deadline:
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(proc.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            proc.wait(timeout=10)
                        raise subprocess.TimeoutExpired(cmd, timeout)
                    ready = selector.select(timeout=0.5)
                    for key, _mask in ready:
                        line = key.fileobj.readline()
                        if line:
                            lf.write(line)
                            lf.flush()
                    if proc.poll() is not None:
                        while True:
                            line = proc.stdout.readline()
                            if not line:
                                break
                            lf.write(line)
                        break
                selector.close()
            exit_code = int(proc.wait())
            lf.write(f"\n[{_utc_now()}] EXIT  query_id={query_id}  "
                     f"exit_code={exit_code}  elapsed={time.monotonic() - start:.1f}s\n")

    except subprocess.TimeoutExpired:
        exit_code = -2
        error_msg = f"timeout after {timeout}s"
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"\n[{_utc_now()}] TIMEOUT  query_id={query_id}  "
                     f"limit={timeout}s\n")
        print(f"[{_utc_now()}] GPU{gpu} TIMEOUT {query_id} after {timeout}s",
              file=sys.stderr)

    except FileNotFoundError:
        exit_code = -3
        error_msg = f"pipeline script not found: {PIPELINE_SCRIPT}"
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"\n[{_utc_now()}] ERROR  query_id={query_id}: {error_msg}\n")

    except Exception as exc:
        exit_code = -4
        error_msg = str(exc)
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"\n[{_utc_now()}] EXCEPTION  query_id={query_id}: {error_msg}\n")

    elapsed = round(time.monotonic() - start, 3)

    trace: dict = {
        "query_id": query_id,
        "protocol_id": item.get("protocol_id", "unregistered_exploratory"),
        "gpu": gpu,
        "returncode": exit_code,
        "exit_code": exit_code,
        "elapsed_seconds": elapsed,
        "error": error_msg,
        "run_dir": run_dir,
        "dataset_dir": dataset_dir,
        "output_root": query_output_root,
        "log_path": log_file,
        "profile": item_profile,
        "force_rerun": bool(force_rerun),
        "strict_release": bool(strict_release),
        "env_overrides": item_env,
        "started_at_utc": started_at_utc,
        "finished_at_utc": _utc_now(),
    }
    return trace


# ---------------------------------------------------------------------------
# GPU worker thread
# ---------------------------------------------------------------------------

def _gpu_worker(
    gpu: int,
    items: list[dict],
    *,
    profile: str,
    force_rerun: bool,
    timeout: int,
    strict_release: bool,
    results: list[dict],
    lock: threading.Lock,
) -> None:
    """Process every query assigned to *gpu* serially."""
    for idx, item in enumerate(items, 1):
        query_id = item["query_id"]
        print(f"[{_utc_now()}] GPU{gpu} [{idx}/{len(items)}] starting {query_id}")

        trace = run_one_query(
            item,
            profile=profile,
            force_rerun=force_rerun,
            timeout=timeout,
            strict_release=strict_release,
        )

        with lock:
            results.append(trace)
            # Write efficiency trace into the manifest-level output_root
            _append_jsonl(
                os.path.join(item["output_root"], "efficiency_trace.jsonl"),
                trace,
            )

        status = "OK" if trace["exit_code"] == 0 else f"FAIL(rc={trace['exit_code']})"
        print(
            f"[{_utc_now()}] GPU{gpu} [{idx}/{len(items)}] finished {query_id}  "
            f"{status}  {trace['elapsed_seconds']:.1f}s"
            + (f"  error: {trace['error']}" if trace["error"] else "")
        )


def _execution_contract_errors(
    scheduled_items: list[dict],
    results: list[dict],
) -> list[str]:
    scheduled_ids = [str(item["query_id"]) for item in scheduled_items]
    executed_ids = [str(result.get("query_id", "")) for result in results]
    errors: list[str] = []
    if len(results) != len(scheduled_items):
        errors.append(f"executed {len(results)} of {len(scheduled_items)} scheduled rows")
    if len(executed_ids) != len(set(executed_ids)):
        errors.append("execution produced duplicate query IDs")
    if set(executed_ids) != set(scheduled_ids):
        errors.append(
            "executed query IDs differ from scheduled IDs: "
            f"missing={sorted(set(scheduled_ids) - set(executed_ids))}, "
            f"unexpected={sorted(set(executed_ids) - set(scheduled_ids))}"
        )
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    try:
        args.profile = resolve_profile(args.profile, strict_release=args.strict_release)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Pre-flight: does the pipeline script exist?
    if not PIPELINE_SCRIPT.exists():
        print(f"ERROR: pipeline script not found: {PIPELINE_SCRIPT}", file=sys.stderr)
        return 1

    # Load manifest. Strict mode treats every non-empty JSONL line as part of
    # the declared protocol; no malformed or duplicate row may disappear.
    try:
        items, manifest_audit = load_manifest_with_audit(
            args.manifest,
            strict=args.strict_release,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not items:
        print("ERROR: no valid items in manifest", file=sys.stderr)
        return 1
    if len(set(args.gpu)) != len(args.gpu):
        print("ERROR: --gpu contains duplicate indices", file=sys.stderr)
        return 1

    manifest_protocol_ids = {
        str(item.get("protocol_id"))
        for item in items
        if item.get("protocol_id") not in (None, "")
    }
    if args.strict_release and args.protocol_id is None:
        print("ERROR: --strict-release requires an explicit --protocol-id", file=sys.stderr)
        return 1
    if args.protocol_id is None and len(manifest_protocol_ids) == 1:
        args.protocol_id = next(iter(manifest_protocol_ids))
    if args.protocol_id is None:
        args.protocol_id = "unregistered_exploratory"

    if args.strict_release:
        if not args.force_rerun:
            print("ERROR: --strict-release requires --force-rerun", file=sys.stderr)
            return 1
        if args.allow_failures:
            print("ERROR: --strict-release cannot be combined with --allow-failures", file=sys.stderr)
            return 1
        release_errors = validate_release_manifest(
            items,
            args.profile,
            args.protocol_id,
            verify_source_files=True,
        )
        if release_errors:
            for error in release_errors:
                print(f"ERROR: release manifest violation: {error}", file=sys.stderr)
            return 1
    source_errors = _source_tree_release_errors(
        strict_release=args.strict_release,
        commit=_git_output("rev-parse", "HEAD"),
        status=_git_output("status", "--porcelain=v1"),
    )
    if source_errors:
        for error in source_errors:
            print(f"ERROR: release source violation: {error}", file=sys.stderr)
        return 1

    # Partition by GPU
    try:
        groups = group_by_gpu(items, args.gpu, strict=args.strict_release)
    except ValueError as exc:
        print(f"ERROR: release scheduling violation: {exc}", file=sys.stderr)
        return 1
    scheduled_items = [item for gpu_items in groups.values() for item in gpu_items]
    if args.strict_release and len(scheduled_items) != len(items):
        print(
            f"ERROR: strict release scheduled {len(scheduled_items)} of {len(items)} manifest rows",
            file=sys.stderr,
        )
        return 1

    print(f"Loaded {len(items)} queries from {args.manifest}")
    print(f"GPUs: {args.gpu}")
    print(f"Profile: {args.profile}")
    print(f"Protocol: {args.protocol_id}")
    print(f"Force rerun: {args.force_rerun}")
    print(f"Strict release: {args.strict_release}")
    print(f"Per-query timeout: {args.timeout}s")
    for gpu, gpu_items in groups.items():
        print(f"  GPU {gpu}: {len(gpu_items)} queries")
    print()

    unique_roots: set[str] = {item["output_root"] for item in items}
    provenance = build_batch_provenance(
        items,
        manifest_path=Path(args.manifest),
        profile=args.profile,
        protocol_id=args.protocol_id,
        strict_release=args.strict_release,
        manifest_audit=manifest_audit,
    )
    for root in unique_roots:
        _ensure_dir(root)
        _write_json_atomic(os.path.join(root, "batch_provenance.json"), provenance)

    results: list[dict] = []
    lock = threading.Lock()
    started_at_utc = _utc_now()
    t0 = time.monotonic()

    # Launch one thread per GPU
    threads: list[threading.Thread] = []
    for gpu, gpu_items in groups.items():
        if not gpu_items:
            continue
        t = threading.Thread(
            target=_gpu_worker,
            args=(gpu, gpu_items),
            kwargs={
                "profile": args.profile,
                "force_rerun": args.force_rerun,
                "timeout": args.timeout,
                "strict_release": args.strict_release,
                "results": results,
                "lock": lock,
            },
            daemon=False,
        )
        t.start()
        threads.append(t)

    # Wait for all workers to finish
    for t in threads:
        t.join()

    total_elapsed = round(time.monotonic() - t0, 1)

    execution_contract_errors = _execution_contract_errors(scheduled_items, results)

    # ---- Summary ----
    succeeded = sum(1 for r in results if r["exit_code"] == 0)
    failed = len(results) - succeeded

    print()
    print("=" * 60)
    print("BATCH COMPLETE")
    print(f"  Total queries : {len(results)}")
    print(f"  Succeeded     : {succeeded}")
    print(f"  Failed        : {failed}")
    print(f"  Wall time     : {total_elapsed:.1f}s")
    print(f"  Contract errs : {len(execution_contract_errors)}")
    print("=" * 60)
    for error in execution_contract_errors:
        print(f"ERROR: execution contract violation: {error}", file=sys.stderr)

    # Per-query table
    if results:
        print()
        print(f"{'Status':>6s}  {'Query ID':<40s}  {'GPU':>3s}  {'Time (s)':>10s}  Error")
        print("-" * 90)
        for r in sorted(results, key=lambda x: x["query_id"]):
            status = "OK" if r["exit_code"] == 0 else f"RC={r['exit_code']}"
            print(
                f"{status:>6s}  {r['query_id']:<40s}  {r['gpu']:>3d}  "
                f"{r['elapsed_seconds']:>10.1f}  {r['error'] or ''}"
            )

    # Write a batch-summary JSON per unique manifest-level output_root
    for root in unique_roots:
        _ensure_dir(root)
        summary = {
            "total_queries": len(results),
            "succeeded": succeeded,
            "failed": failed,
            "total_elapsed_seconds": total_elapsed,
            "gpus": args.gpu,
            "profile": args.profile,
            "protocol_id": args.protocol_id,
            "force_rerun": args.force_rerun,
            "strict_release": args.strict_release,
            "manifest_nonempty_rows": manifest_audit["nonempty_rows"],
            "manifest_loaded_rows": manifest_audit["loaded_rows"],
            "manifest_skipped_rows": manifest_audit["skipped_rows"],
            "scheduled_queries": len(scheduled_items),
            "execution_contract_errors": execution_contract_errors,
            "started_at_utc": started_at_utc,
            "finished_at_utc": _utc_now(),
            "allow_failures": bool(args.allow_failures),
            "results": sorted(results, key=lambda row: str(row.get("query_id", ""))),
        }
        _write_json_atomic(os.path.join(root, "batch_summary.json"), summary)

    if args.strict_release and execution_contract_errors:
        return 1
    return 0 if failed == 0 or args.allow_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
