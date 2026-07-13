#!/usr/bin/env python3
"""Fail fast when a query run is missing the pinned VLM checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from refergaussian.semantics.qwen_query_planner import _resolve_qwen_model


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate external runtime assets before starting a query pipeline."
    )
    parser.add_argument("--require-qwen", action="store_true")
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

    print(f"[ok] Qwen query model: {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
