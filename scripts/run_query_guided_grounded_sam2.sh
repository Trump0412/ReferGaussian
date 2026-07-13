#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_DIR="$1"
DATASET_DIR="$2"
QUERY_TEXT="$3"
QUERY_NAME="${4:-$(echo "${QUERY_TEXT}" | tr ' ' '_' | tr -cd '[:alnum:]_-' | cut -c1-80)}"
OUTPUT_ROOT="${RUN_DIR}/entitybank/query_guided/${QUERY_NAME}"
PLAN_PATH="${OUTPUT_ROOT}/query_plan.json"
TRACK_DIR="${OUTPUT_ROOT}/grounded_sam2"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

mkdir -p "${OUTPUT_ROOT}"

if [[ "${GSAM2_REUSE_QUERY_PLAN:-0}" == "1" && -f "${PLAN_PATH}" ]]; then
  echo "Reusing existing query plan: ${PLAN_PATH}"
else
  plan_args=(
    --query "${QUERY_TEXT}"
    --dataset-dir "${DATASET_DIR}"
    --output-path "${PLAN_PATH}"
    --frame-subsample-stride "${GSAM2_FRAME_SUBSAMPLE_STRIDE:-10}"
    --num-sampled-frames "${GSAM2_NUM_CONTEXT_FRAMES:-9}"
    --num-boundary-frames "${GSAM2_NUM_BOUNDARY_FRAMES:-15}"
  )
  if [[ "${GSAM2_QUERY_PLAN_STRICT:-1}" == "1" ]]; then
    if ! gsam2_python "${GS_ROOT}/scripts/plan_query_entities.py" "${plan_args[@]}" --strict; then
      if [[ "${GSAM2_QUERY_PLAN_STRICT_FALLBACK:-0}" == "1" ]]; then
        echo "[warn] strict query planner failed; retrying because non-strict mode was explicitly enabled"
        gsam2_python "${GS_ROOT}/scripts/plan_query_entities.py" "${plan_args[@]}"
      else
        echo "[error] strict query planner failed; inspect the planner diagnostics or explicitly enable GSAM2_QUERY_PLAN_STRICT_FALLBACK=1" >&2
        exit 1
      fi
    fi
  else
    gsam2_python "${GS_ROOT}/scripts/plan_query_entities.py" "${plan_args[@]}"
  fi
fi

TRACKS_PATH="${TRACK_DIR}/grounded_sam2_query_tracks.json"
if [[ "${GSAM2_REUSE_TRACKS:-0}" == "1" && -f "${TRACKS_PATH}" ]]; then
  echo "Reusing existing GSam2 tracks: ${TRACKS_PATH}"
else
LOCAL_ONLY_ARGS=(--local-files-only)
case "${GSAM2_LOCAL_FILES_ONLY:-1}" in
  1|true|TRUE|yes|YES) ;;
  0|false|FALSE|no|NO) LOCAL_ONLY_ARGS=(--no-local-files-only) ;;
  *)
    echo "GSAM2_LOCAL_FILES_ONLY must be 0 or 1" >&2
    exit 2
    ;;
esac
gsam2_python "${GS_ROOT}/scripts/run_grounded_sam2_query.py" \
  --dataset-dir "${DATASET_DIR}" \
  --query-plan-path "${PLAN_PATH}" \
  --output-dir "${TRACK_DIR}" \
  --grounding-model-id "${GSAM2_GROUNDING_MODEL_ID:-IDEA-Research/grounding-dino-base}" \
  --sam2-model-id "${GSAM2_SAM2_MODEL_ID:-facebook/sam2-hiera-large}" \
  --grounding-model-revision "${GSAM2_GROUNDING_MODEL_REVISION:-12bdfa3120f3e7ec7b434d90674b3396eccf88eb}" \
  --sam2-model-revision "${GSAM2_SAM2_MODEL_REVISION:-e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251}" \
  --prompt-type "${GSAM2_PROMPT_TYPE:-point}" \
  --detector-frame-stride "${GSAM2_DETECTOR_FRAME_STRIDE:-6}" \
  --max-detector-frames "${GSAM2_MAX_DETECTOR_FRAMES:-48}" \
  --detection-top-k "${GSAM2_DETECTION_TOP_K:-5}" \
  --box-threshold "${GSAM2_BOX_THRESHOLD:-0.25}" \
  --text-threshold "${GSAM2_TEXT_THRESHOLD:-0.20}" \
  --num-point-prompts "${GSAM2_NUM_POINT_PROMPTS:-16}" \
  --track-window-radius "${GSAM2_TRACK_WINDOW_RADIUS:-160}" \
  --frame-subsample-stride "${GSAM2_FRAME_SUBSAMPLE_STRIDE:-10}" \
  --num-anchor-seeds "${GSAM2_NUM_ANCHOR_SEEDS:-3}" \
  "${LOCAL_ONLY_ARGS[@]}"
fi

echo "${OUTPUT_ROOT}"
