#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

REPO_ID="${1:-Trump0412/ReferGaussian-R4D-Bench-QA}"
OUTPUT_DIR="${2:-${GS_ROOT}/data/benchmarks/r4d_bench_qa}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

mkdir -p "${OUTPUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Python is required to download dataset snapshots." >&2
  exit 2
fi

"${PYTHON_BIN}" - <<'PY' "${REPO_ID}" "${OUTPUT_DIR}"
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

repo_id = sys.argv[1]
output_dir = Path(sys.argv[2]).resolve()

try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'huggingface_hub'. Install it first, e.g. pip install huggingface_hub"
    ) from exc

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

snapshot_path = Path(
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=token,
        max_workers=4,
        etag_timeout=60,
    )
).resolve()

manifest = {
    "repo_id": repo_id,
    "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
    "snapshot_path": str(snapshot_path),
    "token_provided": bool(token),
}

(output_dir / "download_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(output_dir)
PY

echo "R4D-Bench-QA prepared at ${OUTPUT_DIR}"
