#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PROTOCOL_JSON="${1:-}"
RUN_DIR="${2:-}"
DATASET_DIR="${3:-}"
CATEGORY="${4:-temporal_state_reference}"

if [[ -z "${PROTOCOL_JSON}" || -z "${RUN_DIR}" || -z "${DATASET_DIR}" ]]; then
  echo "Usage: $0 <protocol_json> <run_dir> <dataset_dir> [protocol_category]" >&2
  exit 2
fi

if [[ ! -f "${PROTOCOL_JSON}" ]]; then
  echo "Missing protocol json: ${PROTOCOL_JSON}" >&2
  exit 2
fi
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Missing run dir: ${RUN_DIR}" >&2
  exit 2
fi
if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Missing dataset dir: ${DATASET_DIR}" >&2
  exit 2
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

while IFS=$'\t' read -r query_slug query_text; do
  if [[ -z "${query_slug}" || -z "${query_text}" ]]; then
    continue
  fi
  echo "[query] ${query_slug}: ${query_text}"
  bash "${GS_ROOT}/scripts/run_query_specific_worldtube_pipeline.sh" \
    "${RUN_DIR}" \
    "${DATASET_DIR}" \
    "${query_text}" \
    "${query_slug}"
done < <(
  gs_python "${GS_ROOT}/scripts/list_public_protocol_queries.py" "${PROTOCOL_JSON}" --category "${CATEGORY}"
)
