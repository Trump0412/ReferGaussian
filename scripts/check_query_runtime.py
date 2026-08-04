#!/usr/bin/env python3
"""Fail fast when a query run is missing the pinned VLM checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from refergaussian.semantics.qwen_query_planner import _resolve_qwen_model


DEFAULT_QWEN_REPO_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_QWEN_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def _has_model_weights(model_dir: Path) -> bool:
    patterns = (
        "*.safetensors",
        "*.bin",
        "*.pt",
        "*.ckpt",
        "*.safetensors.index.json",
        "*.bin.index.json",
    )
    return any(model_dir.glob(pattern) for pattern in patterns)


def _pinned_manifest_errors(
    model_dir: Path,
    *,
    expected_repo_id: str = DEFAULT_QWEN_REPO_ID,
    expected_revision: str = DEFAULT_QWEN_REVISION,
) -> list[str]:
    manifest_path = model_dir / "refergaussian_snapshot.json"
    if not manifest_path.is_file():
        return [f"Qwen checkpoint is missing {manifest_path.name}: {model_dir}"]
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Qwen snapshot manifest is unreadable: {manifest_path} ({exc})"]

    errors: list[str] = []
    if str(payload.get("repo_id", "")) != expected_repo_id:
        errors.append(
            f"Qwen snapshot repo_id is {payload.get('repo_id')!r}, expected {expected_repo_id!r}"
        )
    if str(payload.get("repo_type", "model")) != "model":
        errors.append("Qwen snapshot manifest repo_type must be 'model'")
    resolved_revision = str(payload.get("resolved_revision", ""))
    if resolved_revision != expected_revision:
        errors.append(
            f"Qwen resolved revision is {resolved_revision!r}, expected {expected_revision!r}"
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate external runtime assets before starting a query pipeline."
    )
    parser.add_argument("--require-qwen", action="store_true")
    parser.add_argument(
        "--require-pinned-manifest",
        action="store_true",
        help="Require the README-pinned Qwen snapshot manifest.",
    )
    parser.add_argument("--qwen-model", default=None)
    args = parser.parse_args()

    if not args.require_qwen:
        return 0

    try:
        model_dir = _resolve_qwen_model(args.qwen_model).resolve()
    except FileNotFoundError as exc:
        parser.error(
            f"{exc}\nSet REFERGAUSSIAN_QWEN_MODEL=/path/to/Qwen3-VL-8B-Instruct "
            "or follow the README model-download step before launching queries."
        )
    if not model_dir.is_dir():
        parser.error(f"Qwen path is not a directory: {model_dir}")
    if not (model_dir / "config.json").is_file():
        parser.error(f"Qwen checkpoint is missing config.json: {model_dir}")
    if not _has_model_weights(model_dir):
        parser.error(f"Qwen checkpoint has no model weight file: {model_dir}")

    strict_release = os.environ.get("REFERGAUSSIAN_STRICT_RELEASE", "0") == "1"
    if args.require_pinned_manifest or strict_release:
        expected_revision = os.environ.get(
            "REFERGAUSSIAN_QWEN_REVISION", DEFAULT_QWEN_REVISION
        )
        errors = _pinned_manifest_errors(
            model_dir,
            expected_revision=expected_revision,
        )
        if errors:
            parser.error("\n".join(errors))

    print(f"[ok] Qwen query model: {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
