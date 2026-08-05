#!/usr/bin/env python3
"""Run the frozen, matched ReferGaussian reconstruction release protocol.

The runner is deliberately fail-closed. It accepts no model-parameter extras,
requires a clean source checkout, validates the patched 4DGaussians checkout,
and records immutable provenance before launching either method.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "benchmarks" / "reconstruction_release_v1.json"
METRIC_KEYS = ("PSNR", "SSIM", "LPIPS-vgg", "LPIPS-alex")
METHODS = ("baseline_4dgs", "refergaussian")
EXECUTABLE_STATUS = "executable_frozen"
EXTERNAL_4DGS_COMMIT = "843d5ac636c37e4b611242287754f3d4ed150144"
EXPECTED_WRAPPERS = {
    "baseline_train": "scripts/train_baseline.sh",
    "baseline_eval": "scripts/eval_baseline.sh",
    "refergaussian_train": "scripts/train.sh",
    "refergaussian_eval": "scripts/eval.sh",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
METADATA_SUFFIXES = {".json", ".yaml", ".yml", ".txt", ".csv"}
FROZEN_PARAMETER_PREFIXES = ("WARP_", "TEMPORAL_")
FROZEN_PARAMETER_KEYS = {
    "REFERGAUSSIAN_SEED",
    "REFERGAUSSIAN_ITERATIONS",
    "GS_SKIP_FULL_METRICS",
    "GS_QUICK_METRIC_FRAMES",
    "GS_RUN_NAMESPACE",
    "GS_PORT",
}
RELEASE_SCENE_LAYOUT = {
    "americano": (
        "hypernerf",
        "misc/americano",
        "hypernerf/misc/americano",
        "external/4DGaussians/arguments/hypernerf/default.py",
        True,
    ),
    "coffee_martini": (
        "dynerf",
        "coffee_martini",
        "dynerf/coffee_martini",
        "external/4DGaussians/arguments/dynerf/coffee_martini.py",
        False,
    ),
    "cook_spinach": (
        "dynerf",
        "cook_spinach",
        "dynerf/cook_spinach",
        "external/4DGaussians/arguments/dynerf/cook_spinach.py",
        False,
    ),
    "cut_lemon": (
        "hypernerf",
        "interp/cut-lemon1",
        "hypernerf/interp/cut-lemon1",
        "external/4DGaussians/arguments/hypernerf/cut-lemon1.py",
        False,
    ),
    "cut_roasted_beef": (
        "dynerf",
        "cut_roasted_beef",
        "dynerf/cut_roasted_beef",
        "external/4DGaussians/arguments/dynerf/cut_roasted_beef.py",
        False,
    ),
    "espresso": (
        "hypernerf",
        "misc/espresso",
        "hypernerf/misc/espresso",
        "external/4DGaussians/arguments/hypernerf/default.py",
        True,
    ),
    "flame_salmon": (
        "dynerf",
        "flame_salmon_1",
        "dynerf/flame_salmon_1",
        "external/4DGaussians/arguments/dynerf/flame_salmon_1.py",
        False,
    ),
    "flame_steak": (
        "dynerf",
        "flame_steak",
        "dynerf/flame_steak",
        "external/4DGaussians/arguments/dynerf/flame_steak.py",
        False,
    ),
    "keyboard": (
        "hypernerf",
        "misc/keyboard",
        "hypernerf/misc/keyboard",
        "external/4DGaussians/arguments/hypernerf/default.py",
        True,
    ),
    "sear_steak": (
        "dynerf",
        "sear_steak",
        "dynerf/sear_steak",
        "external/4DGaussians/arguments/dynerf/sear_steak.py",
        False,
    ),
    "split_cookie": (
        "hypernerf",
        "misc/split-cookie",
        "hypernerf/misc/split-cookie",
        "external/4DGaussians/arguments/hypernerf/default.py",
        True,
    ),
    "torchchocolate": (
        "hypernerf",
        "interp/torchocolate",
        "hypernerf/interp/torchocolate",
        "external/4DGaussians/arguments/hypernerf/default.py",
        True,
    ),
}
RECONSTRUCTION_PROTOCOL_CONTRACTS = {
    "release_reconstruction_v1": {
        "seed": 6666,
        "tube": {
            "temporal_tube_samples": 3,
            "temporal_tube_span": 0.4,
            "temporal_tube_sigma": 0.34,
            "temporal_tube_covariance_mix": 0.05,
        },
        "environment": {
            "TEMPORAL_WARP_LR_INIT": "0.00012",
            "TEMPORAL_WARP_LR_SCHEDULE": "shared_exponential",
            "TEMPORAL_LR_INIT": "0.00012",
            "TEMPORAL_LR_FINAL": "0.000012",
            "TEMPORAL_TUBE_SIGMA": "0.34",
        },
    },
    "release_reconstruction_v2_paper_compat": {
        "seed": 0,
        "tube": {
            "temporal_tube_samples": 3,
            "temporal_tube_span": 0.4,
            "temporal_tube_sigma": 0.32,
            "temporal_tube_covariance_mix": 0.05,
        },
        "environment": {
            "TEMPORAL_WARP_LR_INIT": "0.00016",
            "TEMPORAL_WARP_LR_SCHEDULE": "constant",
            "TEMPORAL_LR_INIT": "0.00016",
            "TEMPORAL_LR_FINAL": "0.000016",
            "TEMPORAL_TUBE_SIGMA": "0.32",
        },
    },
}


class HarnessError(RuntimeError):
    """Raised when a release invariant is not satisfied."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"Cannot read JSON file {path}: {exc}") from exc


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise HarnessError(f"{label} must be a JSON object")
    return value


def _require_sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise HarnessError(f"{label} must be a JSON array")
    return value


def validate_default_config_declaration(
    scene: Mapping[str, Any], resolved_config_relpath: str
) -> None:
    """Validate both the exact config and any use of default.py."""
    expected = Path(str(scene.get("config_relpath", ""))).as_posix()
    actual = Path(resolved_config_relpath).as_posix()
    if not expected or expected != actual:
        raise HarnessError(
            f"Scene {scene.get('id', '<unknown>')} resolved config {actual!r}; "
            f"registry requires {expected!r}"
        )
    declared = scene.get("default_config_declared")
    if not isinstance(declared, bool):
        raise HarnessError(
            f"Scene {scene.get('id', '<unknown>')} must declare "
            "default_config_declared as a boolean"
        )
    is_default = Path(actual).name == "default.py"
    if is_default and not declared:
        raise HarnessError(
            f"Scene {scene.get('id', '<unknown>')} resolved default.py without "
            "an explicit registry declaration"
        )
    if declared and not is_default:
        raise HarnessError(
            f"Scene {scene.get('id', '<unknown>')} declares a default fallback "
            f"but resolves dedicated config {actual!r}"
        )


def _validate_scene_entry(scene: Mapping[str, Any]) -> None:
    required = ("id", "dataset", "scene", "source_relpath", "config_relpath")
    missing = [key for key in required if not str(scene.get(key, "")).strip()]
    if missing:
        raise HarnessError(f"Scene entry is missing fields: {missing}")
    dataset = str(scene["dataset"])
    wrapper_scene = str(scene["scene"])
    expected_source = Path(dataset, wrapper_scene).as_posix()
    if Path(str(scene["source_relpath"])).as_posix() != expected_source:
        raise HarnessError(
            f"Scene {scene['id']} source_relpath must match the wrapper pair "
            f"{dataset}/{wrapper_scene}"
        )
    config_relpath = Path(str(scene["config_relpath"])).as_posix()
    validate_default_config_declaration(scene, config_relpath)
    expected_prefix = f"external/4DGaussians/arguments/{dataset}/"
    if not config_relpath.startswith(expected_prefix):
        raise HarnessError(
            f"Scene {scene['id']} config is outside {expected_prefix}"
        )


def validate_protocol(protocol_id: str, protocol: Mapping[str, Any]) -> None:
    if protocol.get("status") != EXECUTABLE_STATUS or protocol.get("executable") is not True:
        raise HarnessError(
            f"Protocol {protocol_id!r} is unresolved or non-executable "
            f"(status={protocol.get('status')!r})"
        )
    if protocol.get("is_paper_reproduction") is not False:
        raise HarnessError(
            f"Protocol {protocol_id!r} must explicitly state that it is not a paper reproduction"
        )

    contract = RECONSTRUCTION_PROTOCOL_CONTRACTS.get(protocol_id)
    if contract is None:
        raise HarnessError(f"No executable parameter contract for {protocol_id!r}")

    shared = _require_mapping(protocol.get("shared"), f"{protocol_id}.shared")
    if shared.get("seed") != contract["seed"] or shared.get("iterations") != 14000:
        raise HarnessError(
            f"{protocol_id} requires seed {contract['seed']} and 14000 iterations"
        )
    if shared.get("metric_mode") != "full":
        raise HarnessError("Matched release metrics must use full mode")
    if shared.get("aggregation") != "scene_equal_arithmetic_mean":
        raise HarnessError("Matched release aggregation must be scene-equal")
    if shared.get("post_hoc_psnr_filtering") is not False:
        raise HarnessError("Post-hoc PSNR filtering must be disabled")
    if list(shared.get("headline_metrics", [])) != ["PSNR", "SSIM", "LPIPS-vgg"]:
        raise HarnessError("Headline metrics must be PSNR, SSIM, and LPIPS-vgg")
    if list(shared.get("diagnostic_metrics", [])) != ["LPIPS-alex"]:
        raise HarnessError("LPIPS-alex must be retained as a diagnostic metric")

    refer = _require_mapping(
        protocol.get("refergaussian"), f"{protocol_id}.refergaussian"
    )
    expected_tube = contract["tube"]
    for key, expected in expected_tube.items():
        if refer.get(key) != expected:
            raise HarnessError(f"Frozen ReferGaussian parameter {key} must be {expected}")
    frozen_environment = _require_mapping(
        refer.get("frozen_environment"), "refergaussian.frozen_environment"
    )
    for key, value in frozen_environment.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise HarnessError("Frozen environment keys and values must be strings")
    expected_tube_environment = {
        "TEMPORAL_TUBE_SAMPLES": "3",
        "TEMPORAL_TUBE_SPAN": "0.40",
        "TEMPORAL_TUBE_COVARIANCE_MIX": "0.05",
    }
    expected_tube_environment.update(contract["environment"])
    for key, expected in expected_tube_environment.items():
        if frozen_environment.get(key) != expected:
            raise HarnessError(f"Frozen environment value {key} must be {expected}")

    wrappers = _require_mapping(protocol.get("wrappers"), f"{protocol_id}.wrappers")
    if dict(wrappers) != EXPECTED_WRAPPERS:
        raise HarnessError("Wrapper paths do not match the bounded release harness")
    namespaces = _require_mapping(
        protocol.get("run_namespaces"), f"{protocol_id}.run_namespaces"
    )
    if dict(namespaces) != {
        "baseline_4dgs": "baseline_4dgs",
        "refergaussian": "refergaussian",
    }:
        raise HarnessError("Run namespaces must keep the two methods separate")

    scenes_raw = _require_sequence(protocol.get("scenes"), f"{protocol_id}.scenes")
    scenes = [_require_mapping(item, "scene") for item in scenes_raw]
    scene_ids = [str(scene.get("id", "")) for scene in scenes]
    declared_ids = [str(item) for item in protocol.get("scene_ids", [])]
    if len(scenes) != 12 or protocol.get("scene_count") != 12:
        raise HarnessError("Executable release protocol must contain exactly 12 scenes")
    if len(set(scene_ids)) != len(scene_ids):
        raise HarnessError("Scene ids must be unique")
    if scene_ids != declared_ids:
        raise HarnessError("scene_ids and ordered scene entries must match exactly")
    if set(scene_ids) != set(RELEASE_SCENE_LAYOUT):
        raise HarnessError("Executable release scene ids do not match the dense R4D roster")
    for scene in scenes:
        _validate_scene_entry(scene)
        actual_layout = (
            scene["dataset"],
            scene["scene"],
            Path(str(scene["source_relpath"])).as_posix(),
            Path(str(scene["config_relpath"])).as_posix(),
            scene["default_config_declared"],
        )
        if actual_layout != RELEASE_SCENE_LAYOUT[str(scene["id"])]:
            raise HarnessError(
                f"Scene {scene['id']} does not match the frozen dense R4D layout"
            )

    external = _require_mapping(
        protocol.get("external_4dgaussians"), "external_4dgaussians"
    )
    commit = str(external.get("commit", ""))
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise HarnessError("External 4DGaussians commit must be a full lowercase SHA")
    if commit != EXTERNAL_4DGS_COMMIT:
        raise HarnessError(f"External 4DGaussians commit must be {EXTERNAL_4DGS_COMMIT}")
    patches = _require_sequence(external.get("patches"), "external patches")
    if not patches:
        raise HarnessError("At least one frozen external patch is required")
    for patch in patches:
        item = _require_mapping(patch, "external patch")
        digest = str(item.get("sha256", ""))
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise HarnessError(f"Invalid patch SHA-256 for {item.get('path')}")
        if not item.get("marker_file") or not item.get("marker_text"):
            raise HarnessError(f"Patch {item.get('path')} lacks an applied-state marker")
    patched_diff_sha256 = str(external.get("patched_diff_sha256", ""))
    if len(patched_diff_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in patched_diff_sha256
    ):
        raise HarnessError("External patched diff must have a full SHA-256")
    untracked = _require_sequence(
        external.get("untracked_files"), "external untracked files"
    )
    untracked_paths: list[str] = []
    for record in untracked:
        item = _require_mapping(record, "external untracked file")
        path = Path(str(item.get("path", ""))).as_posix()
        digest = str(item.get("sha256", ""))
        if not path or path.startswith("../") or path.startswith("/"):
            raise HarnessError("External untracked file path must be repository-relative")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise HarnessError(f"Invalid external file SHA-256 for {path}")
        untracked_paths.append(path)
    if len(untracked_paths) != len(set(untracked_paths)):
        raise HarnessError("External untracked file paths must be unique")


def _merge_protocol(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(parent))
    for key, value in child.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_protocol(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_protocol(
    identities: Mapping[str, Any], protocol_id: str, stack: tuple[str, ...] = ()
) -> dict[str, Any]:
    if protocol_id in stack:
        raise HarnessError(f"Protocol inheritance cycle: {' -> '.join(stack + (protocol_id,))}")
    raw = identities.get(protocol_id)
    if raw is None:
        raise HarnessError(f"Unknown protocol identity: {protocol_id}")
    protocol = dict(_require_mapping(raw, protocol_id))
    parent_id = protocol.get("extends")
    if not parent_id:
        return copy.deepcopy(protocol)
    parent = _resolve_protocol(identities, str(parent_id), stack + (protocol_id,))
    return _merge_protocol(parent, protocol)


def load_protocol(registry_path: Path, protocol_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = _require_mapping(read_json(registry_path), "registry")
    identities = _require_mapping(registry.get("identities"), "registry.identities")
    protocol = _resolve_protocol(identities, protocol_id)
    validate_protocol(protocol_id, protocol)
    return dict(registry), protocol


def select_scene_ids(
    protocol: Mapping[str, Any], requested: Sequence[str] | None
) -> tuple[list[str], bool]:
    required = [str(item) for item in protocol["scene_ids"]]
    if not requested:
        return required, True
    selected = [str(item) for item in requested]
    if not selected:
        raise HarnessError("At least one scene is required")
    if len(set(selected)) != len(selected):
        raise HarnessError("Scene subset contains duplicates")
    unknown = sorted(set(selected) - set(required))
    if unknown:
        raise HarnessError(f"Unknown scene ids: {unknown}")
    selected_set = set(selected)
    ordered = [scene_id for scene_id in required if scene_id in selected_set]
    return ordered, ordered == required


def aggregate_scene_equal(
    per_scene: Mapping[str, Any], required_scene_ids: Sequence[str]
) -> dict[str, Any]:
    """Return an unfiltered scene-equal mean for a complete protocol only."""
    required = list(required_scene_ids)
    if set(per_scene) != set(required) or len(per_scene) != len(required):
        missing = sorted(set(required) - set(per_scene))
        extra = sorted(set(per_scene) - set(required))
        raise HarnessError(
            f"Final aggregation requires all declared scenes; missing={missing}, extra={extra}"
        )
    aggregate: dict[str, Any] = {
        "aggregation": "scene_equal_arithmetic_mean",
        "scene_count": len(required),
        "post_hoc_psnr_filtering": False,
        "methods": {},
    }
    for method in METHODS:
        values: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}
        for scene_id in required:
            scene = _require_mapping(per_scene[scene_id], f"per_scene.{scene_id}")
            if scene.get("status") != "complete":
                raise HarnessError(f"Scene {scene_id} is not complete")
            methods = _require_mapping(scene.get("metrics"), f"{scene_id}.metrics")
            metrics = _require_mapping(methods.get(method), f"{scene_id}.{method}")
            validated = validate_metric_payload(metrics, f"{scene_id}.{method}")
            for key in METRIC_KEYS:
                values[key].append(validated[key])
        aggregate["methods"][method] = {
            key: sum(metric_values) / len(metric_values)
            for key, metric_values in values.items()
        }
    aggregate["refergaussian_minus_baseline"] = {
        key: aggregate["methods"]["refergaussian"][key]
        - aggregate["methods"]["baseline_4dgs"][key]
        for key in METRIC_KEYS
    }
    return aggregate


def validate_metric_payload(metrics: Mapping[str, Any], label: str) -> dict[str, float]:
    validated: dict[str, float] = {}
    for key in METRIC_KEYS:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HarnessError(f"{label} lacks numeric metric {key}")
        number = float(value)
        if not math.isfinite(number):
            raise HarnessError(f"{label} metric {key} is not finite")
        validated[key] = number
    return validated


def _git_output(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise HarnessError(f"git {' '.join(args)} failed in {root}: {detail}")
    # Porcelain status uses leading columns to encode the index/worktree state.
    # Preserve those columns while removing only subprocess line terminators.
    return result.stdout.rstrip("\r\n")


def repository_snapshot(repo_root: Path) -> dict[str, Any]:
    commit = _git_output(repo_root, "rev-parse", "HEAD")
    status = _git_output(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise HarnessError(
            "Matched release execution requires a clean Git tree:\n" + status
        )
    return {
        "root": str(repo_root.resolve()),
        "commit": commit,
        "status": [],
        "clean": True,
    }


def expected_parameter_environment(protocol: Mapping[str, Any]) -> dict[str, str]:
    shared = _require_mapping(protocol["shared"], "shared")
    refer = _require_mapping(protocol["refergaussian"], "refergaussian")
    frozen = dict(_require_mapping(refer["frozen_environment"], "frozen_environment"))
    frozen.update(
        {
            "REFERGAUSSIAN_SEED": str(shared["seed"]),
            "REFERGAUSSIAN_ITERATIONS": str(shared["iterations"]),
            "GS_SKIP_FULL_METRICS": "0",
            "GS_PORT": "6021",
        }
    )
    return {str(key): str(value) for key, value in frozen.items()}


def validate_no_mutable_parameter_overrides(
    environ: Mapping[str, str], expected: Mapping[str, str]
) -> None:
    errors: list[str] = []
    for key, value in environ.items():
        is_parameter = key in FROZEN_PARAMETER_KEYS or key.startswith(
            FROZEN_PARAMETER_PREFIXES
        )
        if not is_parameter:
            continue
        if key not in expected:
            errors.append(f"{key} is not declared by the protocol")
        elif str(value) != str(expected[key]):
            errors.append(f"{key}={value!r}, expected {expected[key]!r}")
    if errors:
        raise HarnessError(
            "Mutable reconstruction parameter overrides are forbidden: "
            + "; ".join(sorted(errors))
        )


def _patch_targets(path: Path) -> set[str]:
    targets: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("+++ b/"):
                targets.add(line[6:].strip())
    return targets


def _porcelain_path(line: str) -> str:
    value = line[3:] if len(line) >= 4 else line
    if " -> " in value:
        value = value.split(" -> ", 1)[1]
    return value.strip().strip('"')


def external_snapshot(repo_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    external_spec = _require_mapping(
        protocol["external_4dgaussians"], "external_4dgaussians"
    )
    external_root = repo_root / "external" / "4DGaussians"
    if not (external_root / ".git").exists():
        raise HarnessError(
            f"Missing external 4DGaussians checkout: {external_root}. "
            "Run scripts/bootstrap_external.sh first."
        )
    commit = _git_output(external_root, "rev-parse", "HEAD")
    if commit != external_spec["commit"]:
        raise HarnessError(
            f"External 4DGaussians commit is {commit}, expected {external_spec['commit']}"
        )

    patch_records = []
    allowed_targets: set[str] = set()
    for patch_raw in _require_sequence(external_spec["patches"], "external patches"):
        patch = _require_mapping(patch_raw, "external patch")
        patch_path = repo_root / str(patch["path"])
        if not patch_path.is_file():
            raise HarnessError(f"Missing frozen patch: {patch_path}")
        digest = sha256_file(patch_path)
        if digest != patch["sha256"]:
            raise HarnessError(
                f"Patch hash mismatch for {patch_path}: {digest} != {patch['sha256']}"
            )
        allowed_targets.update(_patch_targets(patch_path))
        marker_path = external_root / str(patch["marker_file"])
        if not marker_path.is_file():
            raise HarnessError(f"Missing patch marker file: {marker_path}")
        marker_text = marker_path.read_text(encoding="utf-8", errors="replace")
        if str(patch["marker_text"]) not in marker_text:
            raise HarnessError(f"Patch is not applied: {patch_path.name}")
        patch_records.append(
            {
                "path": str(patch_path.relative_to(repo_root)),
                "sha256": digest,
                "targets": sorted(_patch_targets(patch_path)),
            }
        )

    status_text = _git_output(
        external_root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    status_lines = status_text.splitlines() if status_text else []
    unexpected = [
        line for line in status_lines if _porcelain_path(line) not in allowed_targets
    ]
    if unexpected:
        raise HarnessError(
            "External checkout has changes outside the frozen patches:\n"
            + "\n".join(unexpected)
        )
    diff = _git_output(external_root, "diff", "--binary", "HEAD")
    diff_digest = hashlib.sha256(diff.encode("utf-8")).hexdigest()
    expected_diff_digest = str(external_spec["patched_diff_sha256"])
    if diff_digest != expected_diff_digest:
        raise HarnessError(
            "External patched diff does not match the frozen patch chain: "
            f"{diff_digest} != {expected_diff_digest}"
        )
    untracked = []
    for line in status_lines:
        if not line.startswith("?? "):
            continue
        relpath = _porcelain_path(line)
        path = external_root / relpath
        if path.is_file():
            untracked.append(
                {"path": relpath, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    expected_untracked = {
        Path(str(item["path"])).as_posix(): str(item["sha256"])
        for item in _require_sequence(
            external_spec["untracked_files"], "external untracked files"
        )
    }
    actual_untracked = {item["path"]: item["sha256"] for item in untracked}
    if actual_untracked != expected_untracked:
        raise HarnessError(
            "External generated patch files do not match the frozen hashes: "
            f"actual={actual_untracked}, expected={expected_untracked}"
        )
    return {
        "root": str(external_root.resolve()),
        "repository": external_spec["repository"],
        "commit": commit,
        "status": status_lines,
        "patched_diff_sha256": diff_digest,
        "untracked_files": untracked,
        "patches": patch_records,
    }


def _wrapper_config_path(repo_root: Path, scene: Mapping[str, Any]) -> Path:
    dataset = str(scene["dataset"])
    config_scene = str(scene["scene"]).split("/")[-1]
    if dataset == "hypernerf" and config_scene == "slice-banana":
        config_scene = "banana"
    if dataset == "hypernerf" and config_scene == "chickchicken":
        config_scene = "chicken"
    root = repo_root / "external" / "4DGaussians" / "arguments" / dataset
    candidate = root / f"{config_scene}.py"
    return candidate if candidate.is_file() else root / "default.py"


def dataset_metadata_record(source: Path) -> dict[str, Any]:
    inventory = hashlib.sha256()
    metadata = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    metadata_count = 0
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = path.relative_to(source).as_posix()
        size = path.stat().st_size
        inventory.update(f"{relative}\0{size}\n".encode("utf-8"))
        file_count += 1
        total_bytes += size
        if path.suffix.lower() in METADATA_SUFFIXES:
            metadata.update(f"{relative}\0{sha256_file(path)}\n".encode("utf-8"))
            metadata_count += 1
    if file_count == 0:
        raise HarnessError(f"Dataset directory is empty: {source}")
    return {
        "root": str(source.resolve()),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "inventory_path_size_sha256": inventory.hexdigest(),
        "metadata_file_count": metadata_count,
        "metadata_content_sha256": metadata.hexdigest(),
    }


def validate_scene_paths(
    repo_root: Path, data_root: Path, scene: Mapping[str, Any]
) -> dict[str, Any]:
    source = data_root / str(scene["source_relpath"])
    if not source.is_dir():
        raise HarnessError(f"Missing dataset for scene {scene['id']}: {source}")
    expected_config = repo_root / str(scene["config_relpath"])
    if not expected_config.is_file():
        raise HarnessError(f"Missing config for scene {scene['id']}: {expected_config}")
    wrapper_config = _wrapper_config_path(repo_root, scene)
    try:
        wrapper_relative = wrapper_config.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise HarnessError(f"Resolved config is outside repository: {wrapper_config}") from exc
    validate_default_config_declaration(scene, wrapper_relative)
    if wrapper_config.resolve() != expected_config.resolve():
        raise HarnessError(
            f"Wrapper/config mismatch for scene {scene['id']}: "
            f"{wrapper_config} != {expected_config}"
        )
    return {
        "id": scene["id"],
        "dataset": scene["dataset"],
        "scene": scene["scene"],
        "source": dataset_metadata_record(source),
        "config": {
            "path": str(expected_config.resolve()),
            "bytes": expected_config.stat().st_size,
            "sha256": sha256_file(expected_config),
            "default_config_declared": scene["default_config_declared"],
        },
    }


def build_commands(
    repo_root: Path, protocol: Mapping[str, Any], scenes: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    wrappers = _require_mapping(protocol["wrappers"], "wrappers")
    namespaces = _require_mapping(protocol["run_namespaces"], "run_namespaces")
    commands: list[dict[str, Any]] = []
    definitions = (
        ("baseline_train", "baseline_4dgs", namespaces["baseline_4dgs"]),
        ("baseline_eval", "baseline_4dgs", namespaces["baseline_4dgs"]),
        ("refergaussian_train", "refergaussian", namespaces["refergaussian"]),
        ("refergaussian_eval", "refergaussian", namespaces["refergaussian"]),
    )
    for scene in scenes:
        for step, method, namespace in definitions:
            commands.append(
                {
                    "scene_id": scene["id"],
                    "step": step,
                    "method": method,
                    "namespace": namespace,
                    "environment": {"GS_RUN_NAMESPACE": namespace},
                    "argv": [
                        "bash",
                        str((repo_root / str(wrappers[step])).resolve()),
                        str(scene["dataset"]),
                        str(scene["scene"]),
                    ],
                }
            )
    return commands


def runtime_snapshot(gpu: int | None) -> dict[str, Any]:
    nvidia = None
    executable = shutil.which("nvidia-smi")
    if executable:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            nvidia = result.stdout.strip().splitlines()
    return {
        "created_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "selected_gpu": gpu,
        "cuda_visible_devices_before_runner": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": nvidia,
    }


def _safe_output_root(output_root: Path, repo_root: Path, data_root: Path) -> None:
    output = output_root.resolve()
    protected = (repo_root.resolve(), data_root.resolve())
    for path in protected:
        if output == path or output in path.parents:
            raise HarnessError(f"Output root may not contain protected path {path}")


def _run_dir(
    run_root: Path, protocol: Mapping[str, Any], method: str, scene: Mapping[str, Any]
) -> Path:
    namespace = str(protocol["run_namespaces"][method])
    return run_root / namespace / str(scene["dataset"]) / Path(str(scene["scene"])).name


def verify_iteration(run_dir: Path, iterations: int) -> dict[str, Any]:
    iteration_dir = run_dir / "point_cloud" / f"iteration_{iterations}"
    point_cloud = iteration_dir / "point_cloud.ply"
    if not iteration_dir.is_dir() or not point_cloud.is_file():
        raise HarnessError(f"Missing iteration_{iterations} checkpoint in {run_dir}")
    return {
        "path": str(iteration_dir),
        "point_cloud_bytes": point_cloud.stat().st_size,
        "point_cloud_sha256": sha256_file(point_cloud),
    }


def image_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(32)
        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
            return struct.unpack(">II", header[16:24])
        if header[:2] != b"\xff\xd8":
            raise HarnessError(f"Unsupported render image format: {path}")
        handle.seek(2)
        while True:
            marker_start = handle.read(1)
            if not marker_start:
                break
            if marker_start != b"\xff":
                continue
            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if marker in {b"\xd8", b"\xd9"}:
                continue
            length_raw = handle.read(2)
            if len(length_raw) != 2:
                break
            length = struct.unpack(">H", length_raw)[0]
            if marker and marker[0] in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                payload = handle.read(5)
                if len(payload) != 5:
                    break
                height, width = struct.unpack(">HH", payload[1:5])
                return width, height
            handle.seek(max(0, length - 2), os.SEEK_CUR)
    raise HarnessError(f"Cannot read image dimensions: {path}")


def render_inventory(run_dir: Path, iterations: int) -> dict[str, Any]:
    method_dir = run_dir / "test" / f"ours_{iterations}"
    renders_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    if not renders_dir.is_dir() or not gt_dir.is_dir():
        raise HarnessError(f"Missing full test render directories in {method_dir}")
    renders = {
        path.name: path
        for path in renders_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    ground_truth = {
        path.name: path
        for path in gt_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if not renders:
        raise HarnessError(f"No rendered test images in {renders_dir}")
    if set(renders) != set(ground_truth):
        raise HarnessError(f"Render/GT filenames differ in {method_dir}")
    dimensions: dict[str, list[int]] = {}
    for name in sorted(renders):
        render_size = image_dimensions(renders[name])
        gt_size = image_dimensions(ground_truth[name])
        if render_size != gt_size:
            raise HarnessError(
                f"Render/GT dimensions differ for {method_dir / name}: "
                f"{render_size} != {gt_size}"
            )
        dimensions[name] = [render_size[0], render_size[1]]
    return {
        "method_dir": str(method_dir),
        "count": len(renders),
        "filenames": sorted(renders),
        "dimensions": dimensions,
        "inventory_sha256": canonical_json_sha256(dimensions),
    }


def load_full_metrics(run_dir: Path, iterations: int) -> dict[str, float]:
    metrics_path = run_dir / "metrics.json"
    results_path = run_dir / "results.json"
    per_view_path = run_dir / "per_view.json"
    for path in (metrics_path, results_path, per_view_path):
        if not path.is_file():
            raise HarnessError(f"Full metrics artifact is missing: {path}")
    metrics = _require_mapping(read_json(metrics_path), str(metrics_path))
    if metrics.get("sample_count") is not None:
        raise HarnessError(f"Subset metrics are forbidden in matched release: {metrics_path}")
    if metrics.get("method") != f"ours_{iterations}":
        raise HarnessError(
            f"Metrics in {metrics_path} are not from ours_{iterations}: "
            f"{metrics.get('method')!r}"
        )
    results = _require_mapping(read_json(results_path), str(results_path))
    if f"ours_{iterations}" not in results:
        raise HarnessError(f"results.json lacks ours_{iterations}: {results_path}")
    _require_mapping(read_json(per_view_path), str(per_view_path))
    return validate_metric_payload(metrics, str(metrics_path))


def verify_scene_pair(
    run_root: Path,
    protocol: Mapping[str, Any],
    scene: Mapping[str, Any],
) -> dict[str, Any]:
    iterations = int(protocol["shared"]["iterations"])
    method_records: dict[str, Any] = {}
    inventories: dict[str, Any] = {}
    for method in METHODS:
        run_dir = _run_dir(run_root, protocol, method, scene)
        method_records[method] = {
            "run_dir": str(run_dir),
            "checkpoint": verify_iteration(run_dir, iterations),
            "metrics": load_full_metrics(run_dir, iterations),
        }
        inventories[method] = render_inventory(run_dir, iterations)
        method_records[method]["renders"] = inventories[method]
    baseline = inventories["baseline_4dgs"]
    refer = inventories["refergaussian"]
    if baseline["filenames"] != refer["filenames"]:
        raise HarnessError(f"Method render filenames differ for scene {scene['id']}")
    if baseline["dimensions"] != refer["dimensions"]:
        raise HarnessError(f"Method render dimensions differ for scene {scene['id']}")
    return {
        "status": "complete",
        "verified_at_utc": utc_now(),
        "metrics": {
            method: method_records[method]["metrics"] for method in METHODS
        },
        "methods": method_records,
        "matched_render_count": baseline["count"],
    }


def _step_artifact_check(
    step: str,
    run_root: Path,
    protocol: Mapping[str, Any],
    scene: Mapping[str, Any],
) -> None:
    method = "baseline_4dgs" if step.startswith("baseline_") else "refergaussian"
    run_dir = _run_dir(run_root, protocol, method, scene)
    iterations = int(protocol["shared"]["iterations"])
    verify_iteration(run_dir, iterations)
    if step.endswith("_eval"):
        render_inventory(run_dir, iterations)
        load_full_metrics(run_dir, iterations)


def _initial_state(
    selected_scene_ids: Sequence[str], complete_protocol: bool
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "planned",
        "complete_protocol": complete_protocol,
        "selected_scene_ids": list(selected_scene_ids),
        "updated_at_utc": utc_now(),
        "scenes": {
            scene_id: {
                "status": "pending",
                "steps": {
                    step: {"status": "pending"} for step in EXPECTED_WRAPPERS
                },
            }
            for scene_id in selected_scene_ids
        },
        "aggregate": None,
    }


def _build_execution_environment(
    protocol: Mapping[str, Any], data_root: Path, run_root: Path, gpu: int | None
) -> tuple[dict[str, str], dict[str, str]]:
    expected = expected_parameter_environment(protocol)
    validate_no_mutable_parameter_overrides(os.environ, expected)
    env = dict(os.environ)
    env.update(expected)
    env["GS_DATA_ROOT"] = str(data_root.resolve())
    env["GS_RUN_ROOT"] = str(run_root.resolve())
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    recorded = dict(expected)
    recorded["GS_DATA_ROOT"] = env["GS_DATA_ROOT"]
    recorded["GS_RUN_ROOT"] = env["GS_RUN_ROOT"]
    if gpu is not None:
        recorded["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env, recorded


def _resume_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "protocol_id",
        "protocol_sha256",
        "registry_sha256",
        "repository",
        "external_4dgaussians",
        "data_root",
        "run_root",
        "selected_scene_ids",
        "complete_protocol",
        "scene_inputs",
        "commands",
        "frozen_environment",
    )
    return {key: manifest[key] for key in keys}


def _validate_resume(
    output_root: Path, current_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    manifest_path = output_root / "run_manifest.json"
    state_path = output_root / "run_state.json"
    if not manifest_path.is_file() or not state_path.is_file():
        raise HarnessError("Resume requires run_manifest.json and run_state.json")
    previous = _require_mapping(read_json(manifest_path), str(manifest_path))
    recorded_previous_hash = previous.get("resume_identity_sha256")
    actual_previous_hash = canonical_json_sha256(_resume_identity_payload(previous))
    if recorded_previous_hash != actual_previous_hash:
        raise HarnessError("Stored run manifest failed its own resume identity check")
    expected_hash = canonical_json_sha256(_resume_identity_payload(current_manifest))
    if recorded_previous_hash != expected_hash:
        raise HarnessError(
            "Resume identity mismatch: protocol, commit, external patches, data, "
            "configuration, scene subset, or commands changed"
        )
    state = dict(_require_mapping(read_json(state_path), str(state_path)))
    if state.get("selected_scene_ids") != current_manifest["selected_scene_ids"]:
        raise HarnessError("Resume state scene subset does not match the manifest")
    return state


def _execute(
    output_root: Path,
    protocol_id: str,
    protocol: Mapping[str, Any],
    selected_scenes: Sequence[Mapping[str, Any]],
    commands: Sequence[Mapping[str, Any]],
    environment: Mapping[str, str],
    state: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    run_root = output_root / "runs"
    state_path = output_root / "run_state.json"
    results_path = output_root / "per_scene_results.json"
    logs_root = output_root / "driver_logs"
    command_by_scene: dict[str, list[Mapping[str, Any]]] = {}
    for command in commands:
        command_by_scene.setdefault(str(command["scene_id"]), []).append(command)
    per_scene: dict[str, Any] = {}
    if results_path.is_file():
        loaded = _require_mapping(read_json(results_path), str(results_path))
        per_scene = dict(loaded.get("scenes", {}))

    for scene in selected_scenes:
        scene_id = str(scene["id"])
        scene_state = state["scenes"][scene_id]
        scene_state["status"] = "running"
        state["status"] = "running"
        state["updated_at_utc"] = utc_now()
        write_json_atomic(state_path, state)
        try:
            for command in command_by_scene[scene_id]:
                step = str(command["step"])
                step_state = scene_state["steps"][step]
                if resume and step_state.get("status") == "complete":
                    _step_artifact_check(step, run_root, protocol, scene)
                    step_state["resume_verified_at_utc"] = utc_now()
                    write_json_atomic(state_path, state)
                    print(f"[resume-verified] {scene_id} {step}", flush=True)
                    continue
                step_env = dict(environment)
                step_env["GS_RUN_NAMESPACE"] = str(command["namespace"])
                log_path = logs_root / scene_id / f"{step}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                step_state.clear()
                step_state.update(
                    {
                        "status": "running",
                        "started_at_utc": utc_now(),
                        "argv": list(command["argv"]),
                        "log_path": str(log_path),
                    }
                )
                write_json_atomic(state_path, state)
                print(f"[run] {scene_id} {step}", flush=True)
                started = time.monotonic()
                with log_path.open("a", encoding="utf-8") as log_handle:
                    result = subprocess.run(
                        list(command["argv"]),
                        cwd=REPO_ROOT,
                        env=step_env,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                elapsed = time.monotonic() - started
                step_state.update(
                    {
                        "returncode": result.returncode,
                        "elapsed_seconds": elapsed,
                        "finished_at_utc": utc_now(),
                    }
                )
                if result.returncode != 0:
                    step_state["status"] = "failed"
                    raise HarnessError(
                        f"Command failed for {scene_id} {step} with return code "
                        f"{result.returncode}; see {log_path}"
                    )
                _step_artifact_check(step, run_root, protocol, scene)
                step_state["status"] = "complete"
                write_json_atomic(state_path, state)

            verification = verify_scene_pair(run_root, protocol, scene)
            scene_state["status"] = "complete"
            scene_state["verified_at_utc"] = verification["verified_at_utc"]
            per_scene[scene_id] = verification
            write_json_atomic(
                results_path,
                {
                    "protocol_id": protocol_id,
                    "complete_protocol": state["complete_protocol"],
                    "scenes": per_scene,
                },
            )
            write_json_atomic(state_path, state)
            print(f"[verified] {scene_id}", flush=True)
        except Exception as exc:
            scene_state["status"] = "failed"
            scene_state["error"] = str(exc)
            state["status"] = "failed"
            state["updated_at_utc"] = utc_now()
            write_json_atomic(state_path, state)
            raise

    required_ids = [str(item) for item in protocol["scene_ids"]]
    if state["complete_protocol"]:
        aggregate = aggregate_scene_equal(per_scene, required_ids)
        final = {
            "protocol_id": protocol_id,
            "is_paper_reproduction": False,
            "status": "complete",
            "completed_at_utc": utc_now(),
            "aggregate": aggregate,
            "per_scene_results": str(results_path),
        }
        write_json_atomic(output_root / "final_metrics.json", final)
        state["aggregate"] = aggregate
        state["status"] = "complete"
    else:
        subset = {
            "protocol_id": protocol_id,
            "status": "incomplete_canary_subset",
            "is_final_aggregate": False,
            "selected_scene_ids": state["selected_scene_ids"],
            "aggregate": None,
            "per_scene_results": str(results_path),
        }
        write_json_atomic(output_root / "incomplete_subset.json", subset)
        state["aggregate"] = None
        state["status"] = "incomplete_canary_subset"
    state["updated_at_utc"] = utc_now()
    write_json_atomic(state_path, state)
    return state


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", type=Path, default=DEFAULT_REGISTRY, help="Protocol registry JSON"
    )
    parser.add_argument(
        "--protocol", default="release_reconstruction_v1", help="Executable identity"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("GS_DATA_ROOT", REPO_ROOT / "data")),
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        help="Exact optional scene-id subset for an incomplete canary",
    )
    parser.add_argument("--gpu", type=int, help="Single physical GPU id to expose")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only an output with an identical immutable identity",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the exact plan without creating output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        if args.resume and args.dry_run:
            raise HarnessError("--resume and --dry-run cannot be combined")
        registry_path = args.registry.expanduser().resolve()
        output_root = args.output_root.expanduser().resolve()
        data_root = args.data_root.expanduser().resolve()
        _safe_output_root(output_root, REPO_ROOT, data_root)
        if args.resume:
            if not output_root.is_dir():
                raise HarnessError(f"Resume output root does not exist: {output_root}")
        elif output_root.exists():
            raise HarnessError(
                f"Output root already exists; choose a new path or use --resume: {output_root}"
            )

        registry, protocol = load_protocol(registry_path, args.protocol)
        selected_ids, complete_protocol = select_scene_ids(protocol, args.scenes)
        scene_map = {str(scene["id"]): scene for scene in protocol["scenes"]}
        selected_scenes = [scene_map[scene_id] for scene_id in selected_ids]
        repository = repository_snapshot(REPO_ROOT)
        external = external_snapshot(REPO_ROOT, protocol)
        run_root = output_root / "runs"
        environment, recorded_environment = _build_execution_environment(
            protocol, data_root, run_root, args.gpu
        )
        scene_inputs = [
            validate_scene_paths(REPO_ROOT, data_root, scene)
            for scene in selected_scenes
        ]
        commands = build_commands(REPO_ROOT, protocol, selected_scenes)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "protocol_id": args.protocol,
            "protocol_sha256": canonical_json_sha256(protocol),
            "registry_path": str(registry_path),
            "registry_sha256": sha256_file(registry_path),
            "registry_id": registry.get("registry_id"),
            "is_paper_reproduction": False,
            "repository": repository,
            "external_4dgaussians": external,
            "data_root": str(data_root),
            "run_root": str(run_root),
            "selected_scene_ids": selected_ids,
            "complete_protocol": complete_protocol,
            "result_label": (
                "full_protocol_execution"
                if complete_protocol
                else "incomplete_canary_subset"
            ),
            "scene_inputs": scene_inputs,
            "commands": commands,
            "frozen_environment": recorded_environment,
            "runtime": runtime_snapshot(args.gpu),
        }
        manifest["resume_identity_sha256"] = canonical_json_sha256(
            _resume_identity_payload(manifest)
        )

        if args.dry_run:
            print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True))
            return 0

        if args.resume:
            state = _validate_resume(output_root, manifest)
        else:
            output_root.mkdir(parents=True, exist_ok=False)
            state = _initial_state(selected_ids, complete_protocol)
            write_json_atomic(output_root / "run_manifest.json", manifest)
            write_json_atomic(output_root / "run_state.json", state)

        _execute(
            output_root,
            args.protocol,
            protocol,
            selected_scenes,
            commands,
            environment,
            state,
            args.resume,
        )
        print(f"[done] matched reconstruction output: {output_root}", flush=True)
        return 0
    except HarnessError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
