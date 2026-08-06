#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

REPO_ID="${1:-${R4D_BENCH_REPO_ID:-LiYacheng/r4d-bench-qa}}"
OUTPUT_DIR="${2:-${GS_DATA_ROOT}/benchmarks/r4d_bench_qa}"
REVISION="${R4D_BENCH_REVISION:-0fe2b3a99a95632ea6d0bd1718723ac24804e49b}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

mkdir -p "${OUTPUT_DIR}"

gsam2_python - <<'PY' "${REPO_ID}" "${OUTPUT_DIR}" "${REVISION}"
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

repo_id = sys.argv[1]
output_dir = Path(sys.argv[2]).resolve()
requested_revision = sys.argv[3]

try:
    from huggingface_hub import HfApi, snapshot_download
except ModuleNotFoundError as exc:
    raise SystemExit("Missing huggingface_hub in the query environment; run scripts/setup.sh") from exc

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
resolved_revision = str(HfApi().dataset_info(repo_id, revision=requested_revision, token=token).sha)

snapshot_path = Path(
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=resolved_revision,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=token,
        max_workers=4,
        etag_timeout=60,
    )
).resolve()

def _link_or_copy(src: Path, dst: Path) -> None:
    src = src.resolve()
    if dst.exists() and dst.resolve() == src:
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _pick_first(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


# Canonical entrypoints used by README / pipeline.
benchmark_36 = _pick_first(
    [
        output_dir / "benchmark.json",
        output_dir / "scripts" / "new_predictions_ground_truth_final.json",
    ]
)
benchmark_89 = _pick_first(
    [
        output_dir / "benchmark_all_queries.json",
        output_dir / "scripts" / "new_predictions_ground_truth_all_queries.json",
    ]
)

if benchmark_36 is not None:
    _link_or_copy(benchmark_36, output_dir / "benchmark.json")
if benchmark_89 is not None:
    _link_or_copy(benchmark_89, output_dir / "benchmark_all_queries.json")

manifest = {
    "repo_id": repo_id,
    "requested_revision": requested_revision,
    "resolved_revision": resolved_revision,
    "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
    "snapshot_path": str(snapshot_path),
    "token_provided": bool(token),
    "benchmark_36": str((output_dir / "benchmark.json").resolve()) if (output_dir / "benchmark.json").exists() else "",
    "benchmark_89": str((output_dir / "benchmark_all_queries.json").resolve()) if (output_dir / "benchmark_all_queries.json").exists() else "",
}

(output_dir / "download_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(output_dir)
if benchmark_36 is not None:
    print(f"benchmark_36={output_dir / 'benchmark.json'}")
if benchmark_89 is not None:
    print(f"benchmark_89={output_dir / 'benchmark_all_queries.json'}")
PY

echo "R4D-Bench-QA prepared at ${OUTPUT_DIR}"
