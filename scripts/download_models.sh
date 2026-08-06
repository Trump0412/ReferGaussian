#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/common.sh"

QWEN_REPO_ID="${REFERGAUSSIAN_QWEN_REPO_ID:-Qwen/Qwen3-VL-8B-Instruct}"
QWEN_REVISION="${REFERGAUSSIAN_QWEN_REVISION:-0c351dd01ed87e9c1b53cbc748cba10e6187ff3b}"
QWEN_MODEL_DIR="${REFERGAUSSIAN_QWEN_MODEL:-${ROOT_DIR}/models/Qwen3-VL-8B-Instruct}"

gsam2_python "${ROOT_DIR}/scripts/download_hf_snapshot.py" \
  --repo-id "${QWEN_REPO_ID}" \
  --revision "${QWEN_REVISION}" \
  --local-dir "${QWEN_MODEL_DIR}"

export REFERGAUSSIAN_QWEN_MODEL="${QWEN_MODEL_DIR}"
export REFERGAUSSIAN_QWEN_REVISION="${QWEN_REVISION}"
gsam2_python "${ROOT_DIR}/scripts/check_query_runtime.py" \
  --require-qwen \
  --require-pinned-manifest

echo "Pinned ReferGaussian model weights are ready under ${QWEN_MODEL_DIR}."
