#!/usr/bin/env python3
"""Preflight checks for manifest-based query batches."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from refergaussian.run_identity import validate_query_ready_4dgs_run
from run_query_batch import resolve_profile, validate_release_manifest


REQUIRED_KEYS = {"query_id", "query", "run_dir", "dataset_dir", "output_root", "gpu"}
DEFAULT_IMPORTS = ["numpy", "scipy", "cv2", "PIL", "torch"]


def _load_manifest(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(f"line {lineno}: invalid JSON: {exc}")
                continue
            if not isinstance(row, dict):
                errors.append(f"line {lineno}: expected JSON object")
                continue
            missing = REQUIRED_KEYS - row.keys()
            if missing:
                errors.append(f"line {lineno}: missing keys {sorted(missing)}")
                continue
            rows.append(row)
    return rows, errors


def _check_imports(names: list[str]) -> tuple[list[str], dict[str, str]]:
    errors: list[str] = []
    versions: dict[str, str] = {}
    for name in names:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - preflight should expose import failures.
            errors.append(f"import failed: {name}: {exc}")
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return errors, versions


def _visible_gpu_count() -> tuple[int | None, str]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None, "nvidia-smi not found"
    try:
        proc = subprocess.run(
            ["bash", "-lc", f"{exe} -L"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"nvidia-smi -L failed: {exc}"
    lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("GPU ")]
    return len(lines), proc.stdout.strip()


def _is_writable_dir(path: Path, *, create: bool) -> bool:
    try:
        if create:
            path.mkdir(parents=True, exist_ok=True)
        probe_dir = path if path.exists() else path.parent
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe = probe_dir / ".refergaussian_preflight_write_test"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
    except OSError:
        return False
    return True


def _check_rows(
    rows: list[dict[str, Any]],
    *,
    active_gpus: set[int] | None,
    create_output_root: bool,
    check_camera: bool,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()

    for row in rows:
        query_id = str(row["query_id"])
        if query_id in seen:
            errors.append(f"{query_id}: duplicate query_id")
        seen.add(query_id)

        try:
            gpu = int(row["gpu"])
        except (TypeError, ValueError):
            errors.append(f"{query_id}: gpu is not an integer: {row['gpu']!r}")
            continue
        if active_gpus is not None and gpu not in active_gpus:
            errors.append(f"{query_id}: assigned GPU {gpu} is not in active GPU list {sorted(active_gpus)}")

        run_dir = Path(str(row["run_dir"]))
        dataset_dir = Path(str(row["dataset_dir"]))
        output_root = Path(str(row["output_root"]))
        if not run_dir.is_dir():
            errors.append(f"{query_id}: run_dir missing: {run_dir}")
        else:
            run_errors = validate_query_ready_4dgs_run(run_dir)
            for run_error in run_errors:
                errors.append(f"{query_id}: query-ready upstream 4DGS input required: {run_error}")
        if not dataset_dir.is_dir():
            errors.append(f"{query_id}: dataset_dir missing: {dataset_dir}")
        if not _is_writable_dir(output_root, create=create_output_root):
            errors.append(f"{query_id}: output_root is not writable: {output_root}")

        dataset_text = str(dataset_dir).lower()
        scene_text = str(row.get("scene") or "").lower()
        if check_camera and ("dynerf" in dataset_text or "coffee_martini" in scene_text):
            camera0 = dataset_dir / "camera" / "0000.json"
            if not camera0.is_file():
                errors.append(f"{query_id}: missing required DyNeRF camera file: {camera0}")

        query_text = str(row.get("query") or "").strip()
        if not query_text:
            errors.append(f"{query_id}: empty query text")
        elif len(query_text) < 3:
            warnings.append(f"{query_id}: very short query text: {query_text!r}")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSONL manifest to preflight.")
    parser.add_argument("--gpu", nargs="*", type=int, help="Active GPU ids expected by the batch runner.")
    parser.add_argument(
        "--imports",
        nargs="*",
        default=DEFAULT_IMPORTS,
        help="Python modules to import before running the batch.",
    )
    parser.add_argument(
        "--create-output-root",
        action="store_true",
        help="Create output roots while checking writability.",
    )
    parser.add_argument(
        "--no-camera-check",
        action="store_true",
        help="Disable DyNeRF camera/0000.json checks.",
    )
    parser.add_argument(
        "--require-visible-gpu",
        action="store_true",
        help="Fail if nvidia-smi reports fewer visible GPUs than requested.",
    )
    parser.add_argument(
        "--strict-release",
        action="store_true",
        help="Validate the complete registered protocol and its pinned source hashes.",
    )
    parser.add_argument(
        "--protocol-id",
        default=None,
        help="Registered protocol identity required by --strict-release.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Formal query profile required by --strict-release.",
    )
    parser.add_argument("--output-json", help="Optional path to write a machine-readable summary.")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    errors: list[str] = []
    warnings: list[str] = []
    manifest_errors: list[str] = []

    if not manifest.is_file():
        errors.append(f"manifest not found: {manifest}")
        rows: list[dict[str, Any]] = []
    else:
        rows, manifest_errors = _load_manifest(manifest)
        errors.extend(manifest_errors)

    if args.strict_release:
        if not args.protocol_id:
            errors.append("--strict-release requires --protocol-id")
        try:
            strict_profile = resolve_profile(args.profile, strict_release=True)
        except ValueError as exc:
            errors.append(str(exc))
            strict_profile = ""
        if args.protocol_id and strict_profile and not manifest_errors:
            errors.extend(
                validate_release_manifest(
                    rows,
                    strict_profile,
                    args.protocol_id,
                    verify_source_files=True,
                )
            )

    import_errors, versions = _check_imports(list(args.imports or []))
    errors.extend(import_errors)

    active_gpus = set(args.gpu) if args.gpu else None
    visible_gpu_count, gpu_output = _visible_gpu_count()
    if active_gpus and visible_gpu_count is not None:
        max_requested = max(active_gpus) if active_gpus else -1
        if visible_gpu_count == 0:
            msg = "nvidia-smi reports zero visible GPUs"
            (errors if args.require_visible_gpu else warnings).append(msg)
        elif max_requested >= visible_gpu_count:
            msg = f"requested GPU {max_requested}, but only {visible_gpu_count} visible GPUs reported"
            (errors if args.require_visible_gpu else warnings).append(msg)
    elif active_gpus and visible_gpu_count is None:
        msg = f"could not determine visible GPUs: {gpu_output}"
        (errors if args.require_visible_gpu else warnings).append(msg)

    row_errors, row_warnings = _check_rows(
        rows,
        active_gpus=active_gpus,
        create_output_root=bool(args.create_output_root),
        check_camera=not bool(args.no_camera_check),
    )
    errors.extend(row_errors)
    warnings.extend(row_warnings)

    summary = {
        "manifest": str(manifest),
        "rows": len(rows),
        "imports": versions,
        "active_gpus": sorted(active_gpus) if active_gpus is not None else None,
        "strict_release": bool(args.strict_release),
        "protocol_id": args.protocol_id,
        "profile": args.profile,
        "visible_gpu_count": visible_gpu_count,
        "gpu_probe": gpu_output,
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }

    print(f"manifest_rows: {len(rows)}")
    print("imports: " + ", ".join(f"{k}={v}" for k, v in versions.items()))
    print(f"visible_gpu_count: {visible_gpu_count}")
    if warnings:
        print("\nWARNINGS:")
        for item in warnings:
            print(f"- {item}")
    if errors:
        print("\nERRORS:")
        for item in errors:
            print(f"- {item}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if errors:
        print("\npreflight: FAILED")
        return 1
    print("\npreflight: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
