#!/usr/bin/env python3
"""Download a pinned Hugging Face model or dataset snapshot with a manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download an immutable Hugging Face snapshot and record its resolved commit."
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    parser.add_argument("--revision", required=True, help="Git commit SHA or immutable revision.")
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--manifest-name", default="refergaussian_snapshot.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing huggingface_hub. Run scripts/setup_grounded_sam2.sh first, "
            "then invoke this script with gsam2_python."
        ) from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    api = HfApi()
    if args.repo_type == "dataset":
        info = api.dataset_info(args.repo_id, revision=args.revision, token=token)
    else:
        info = api.model_info(args.repo_id, revision=args.revision, token=token)
    resolved_revision = str(info.sha)

    local_dir = Path(args.local_dir).expanduser().resolve()
    snapshot_path = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=resolved_revision,
            local_dir=str(local_dir),
            token=token,
            max_workers=4,
            etag_timeout=60,
        )
    ).resolve()
    manifest = {
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "snapshot_path": str(snapshot_path),
        "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
    }
    manifest_path = local_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
