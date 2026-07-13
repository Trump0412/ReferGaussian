#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

REPO_ID="${1:-rpzhou/HyperNeRF-Annotation}"
OUTPUT_DIR="${2:-${GS_ROOT}/data/benchmarks/4dlangsplat/$(basename "${REPO_ID}")}"
SCENE_NAME="${3:-}"
REVISION="${4:-${FOURDLANGSPLAT_ANNOTATION_REVISION:-d127a280446206fc97887a304de790a1fe6af5ff}}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

mkdir -p "${OUTPUT_DIR}"

gsam2_python - <<'PY' "${REPO_ID}" "${OUTPUT_DIR}" "${SCENE_NAME}" "${REVISION}"
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

repo_id = sys.argv[1]
output_dir = Path(sys.argv[2]).resolve()
scene_name = sys.argv[3].strip()
requested_revision = sys.argv[4]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
resolved_revision = str(HfApi().dataset_info(repo_id, revision=requested_revision, token=token).sha)

allow_patterns = None
if scene_name:
    allow_patterns = [
        f"{scene_name}/**",
        f"**/{scene_name}/**",
        "*.json",
        "*.md",
        "*.txt",
    ]

snapshot_path = Path(
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=resolved_revision,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=allow_patterns,
        max_workers=1,
        etag_timeout=60,
    )
).resolve()

scene_dirs = sorted(
    [
        path
        for path in output_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".") and path.name not in {"refs", "blobs", "snapshots"}
    ]
)
manifest = {
    "repo_id": repo_id,
    "requested_revision": requested_revision,
    "resolved_revision": resolved_revision,
    "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
    "snapshot_path": str(snapshot_path),
    "scene_dirs": [path.name for path in scene_dirs],
}

if scene_name:
    matches = [path for path in scene_dirs if path.name == scene_name]
    manifest["requested_scene"] = scene_name
    manifest["scene_exists"] = bool(matches)
    if matches:
        scene_path = matches[0]
        scene_manifest = {
            "scene": scene_name,
            "files": sorted(str(path.relative_to(scene_path)) for path in scene_path.rglob("*") if path.is_file())[:400],
        }
        with open(output_dir / f"{scene_name}_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(scene_manifest, handle, indent=2, ensure_ascii=False)

with open(output_dir / "download_manifest.json", "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, ensure_ascii=False)

print(output_dir)
PY
