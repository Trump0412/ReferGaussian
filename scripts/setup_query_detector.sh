#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_ID="${1:-IDEA-Research/grounding-dino-base}"
MODEL_REVISION="${GSAM2_GROUNDING_MODEL_REVISION:-12bdfa3120f3e7ec7b434d90674b3396eccf88eb}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

echo "Prefetching detector model: ${MODEL_ID} @ ${MODEL_REVISION}"
gsam2_python - <<PY
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

model_id = "${MODEL_ID}"
model_revision = "${MODEL_REVISION}"
processor = AutoProcessor.from_pretrained(model_id, revision=model_revision)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, revision=model_revision)
print("processor", type(processor).__name__)
print("model", type(model).__name__)
print("cached", model_id)
PY
