#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/query_eval_profiles.sh"
require_4dgaussians
require_grounded_sam2

RUN_DIR="$1"
DATASET_DIR="$2"
QUERY_TEXT="$3"
QUERY_NAME="${4:-$(echo "${QUERY_TEXT}" | tr ' ' '_' | tr -cd '[:alnum:]_-' | cut -c1-80)}"

# Validate external, non-versioned assets before allocating detector/VLM GPU work.
gsam2_python "${GS_ROOT}/scripts/check_grounded_sam2_import.py" \
  --grounded-sam2-root "${GS_ROOT}/external/Grounded-SAM-2"
if [[ "${QUERY_SKIP_QWEN_EXPORT:-0}" != "1" || "${QUERY_SKIP_QWEN_SELECTION:-0}" != "1" ]]; then
  gsam2_python "${GS_ROOT}/scripts/check_query_runtime.py" --require-qwen
fi

if [[ -n "${QUERY_OUTPUT_ROOT_OVERRIDE:-}" ]]; then
  if [[ "${QUERY_OUTPUT_ROOT_OVERRIDE}" = /* ]]; then
    OUTPUT_ROOT="${QUERY_OUTPUT_ROOT_OVERRIDE}"
  else
    OUTPUT_ROOT="${GS_ROOT}/${QUERY_OUTPUT_ROOT_OVERRIDE}"
  fi
else
  OUTPUT_ROOT="${RUN_DIR}/entitybank/query_guided/${QUERY_NAME}"
fi
TRACK_DIR="${OUTPUT_ROOT}/grounded_sam2"
TRACKS_PATH="${TRACK_DIR}/grounded_sam2_query_tracks.json"
PROPOSAL_DIR="${OUTPUT_ROOT}/proposal_dir"
QUERY_ENTITYBANK_DIR="${OUTPUT_ROOT}/query_entitybank"
QUERY_RUN_DIR="${OUTPUT_ROOT}/query_worldtube_run"
ENTITY_LIBRARY_DIR="${OUTPUT_ROOT}/entity_library_qwen_sourcebg"
QWEN_ASSIGNMENTS_PATH="${QUERY_RUN_DIR}/entitybank/semantic_assignments_qwen.json"
QWEN_SELECTION_PATH="${QUERY_RUN_DIR}/entitybank/selected_query_qwen.json"
FINAL_RENDER_DIR="${OUTPUT_ROOT}/final_query_render_sourcebg"
DIAGNOSTIC_DIR="${OUTPUT_ROOT}/diagnostics"
FINAL_VALIDATION_PATH="${FINAL_RENDER_DIR}/validation.json"
EFFICIENCY_TRACE="${OUTPUT_ROOT}/efficiency_trace.jsonl"
EFFICIENCY_SUMMARY="${OUTPUT_ROOT}/efficiency_summary.json"
QUERY_START_TS="${QUERY_START_TS:-$(date +%s)}"

_record_stage() {
  local stage_name="$1"
  local exit_code="$2"
  local elapsed="$3"
  local end_ts
  end_ts="$(date +%s)"
  local peak_gpu_mb=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    peak_gpu_mb="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')" || peak_gpu_mb=0
  fi
  printf '{"stage":"%s","exit_code":%d,"elapsed_seconds":%d,"peak_gpu_mb":%d,"timestamp":%d}\n' \
    "${stage_name}" "${exit_code}" "${elapsed}" "${peak_gpu_mb:-0}" "${end_ts}" >> "${EFFICIENCY_TRACE}"
}

_stage_wrapper() {
  local stage_name="$1"
  shift
  echo "[stage] ${stage_name} start: $(date '+%Y-%m-%d %H:%M:%S')"
  local stage_start
  stage_start="$(date +%s)"
  local stage_status=0
  "$@" || stage_status=$?
  local stage_end
  stage_end="$(date +%s)"
  local stage_elapsed=$((stage_end - stage_start))
  _record_stage "${stage_name}" "${stage_status}" "${stage_elapsed}"
  echo "[stage] ${stage_name} done: elapsed=${stage_elapsed}s exit=${stage_status}"
  return "${stage_status}"
}

write_efficiency_summary() {
  QUERY_END_TS="$(date +%s)"
  QUERY_TOTAL_ELAPSED=$((QUERY_END_TS - QUERY_START_TS))
  QUERY_PEAK_GPU_MB=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    QUERY_PEAK_GPU_MB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')" || QUERY_PEAK_GPU_MB=0
  fi
  if [[ -f "${EFFICIENCY_TRACE}" ]]; then
    gs_python - "${EFFICIENCY_TRACE}" "${EFFICIENCY_SUMMARY}" "${QUERY_TOTAL_ELAPSED}" "${QUERY_PEAK_GPU_MB:-0}" "${QUERY_NAME}" <<'PY'
import json, sys
trace_path, summary_path, total_sec, peak_mb, query_name = sys.argv[1:]
stages = []
with open(trace_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            stages.append(json.loads(line))
stage_elapsed = sum(s.get('elapsed_seconds', 0) for s in stages)
summary = {
    'query_name': query_name,
    'total_elapsed_seconds': int(total_sec),
    'stage_elapsed_seconds': stage_elapsed,
    'peak_gpu_mb': int(peak_mb),
    'stage_count': len(stages),
    'stages': stages,
}
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
PY
  fi
}

run_final_render_and_summary() {
  # Release GPU lock before final rendering to overlap with next query's GPU phase.
  release_gpu_lock

  render_args=(
    "${GS_ROOT}/scripts/render_query_video.py"
    --run-dir "${QUERY_RUN_DIR}" \
    --dataset-dir "${DATASET_DIR}" \
    --selection-path "${QWEN_SELECTION_PATH}" \
    --output-dir "${FINAL_RENDER_DIR}" \
    --eval-profile "${QUERY_EVAL_PROFILE}" \
    --background-mode source \
    --fps "${QUERY_RENDER_FPS:-6}" \
    --stride "${QUERY_RENDER_STRIDE:-1}"
  )
  if [[ -n "${QUERY_RENDER_IMAGE_IDS_JSON:-}" ]]; then
    if [[ ! -f "${QUERY_RENDER_IMAGE_IDS_JSON}" ]]; then
      echo "[error] QUERY_RENDER_IMAGE_IDS_JSON is not readable: ${QUERY_RENDER_IMAGE_IDS_JSON}" >&2
      return 2
    fi
    render_args+=(--image-ids-json "${QUERY_RENDER_IMAGE_IDS_JSON}")
  fi
  run_gs_python_with_timeout "${QUERY_RENDER_STAGE_TIMEOUT:-0}" "${render_args[@]}"

  if [[ "${QUERY_SKIP_DIAGNOSTIC_EXPORT:-0}" != "1" \
      && -n "${QUERY_ANNOTATION_DIR:-}" \
      && -d "${QUERY_ANNOTATION_DIR}" ]]; then
    gs_python "${GS_ROOT}/scripts/export_query_diagnostics.py" \
      --query-root "${OUTPUT_ROOT}" \
      --dataset-dir "${DATASET_DIR}" \
      --annotation-dir "${QUERY_ANNOTATION_DIR}" \
      --output-dir "${DIAGNOSTIC_DIR}"
  fi

  write_efficiency_summary
  echo "${QUERY_RUN_DIR}"
}

QUERY_EVAL_PROFILE="${QUERY_EVAL_PROFILE:-${REFERGAUSSIAN_QUERY_EVAL_PROFILE:-default}}"
apply_query_eval_profile "${QUERY_EVAL_PROFILE}"
echo "[profile] QUERY_EVAL_PROFILE=${QUERY_EVAL_PROFILE}"
require_4dgs_run "${RUN_DIR}"

QUERY_PROPOSAL_BUILDER="${QUERY_PROPOSAL_BUILDER:-mask_supported_lifting}"
RENDERER_GEOMETRY_SUPPORT_DIR="${OUTPUT_ROOT}/renderer_geometry_support"
RENDERER_GEOMETRY_FINAL_DIR="${OUTPUT_ROOT}/renderer_geometry_final"

if [[ "${QUERY_PROPOSAL_BUILDER}" != "mask_supported_lifting" ]]; then
  echo "[error] the public release supports only QUERY_PROPOSAL_BUILDER=mask_supported_lifting." >&2
  exit 2
fi

run_gs_python_with_timeout() {
  local timeout_s="$1"
  shift
  if [[ "${timeout_s}" =~ ^[0-9]+$ && "${timeout_s}" -gt 0 ]] && command -v timeout >/dev/null 2>&1; then
    if [[ "${CONDA_PREFIX:-}" == "${GS_ENV_PATH}" ]]; then
      timeout --foreground --kill-after="${QUERY_STAGE_TIMEOUT_KILL_AFTER:-30}s" "${timeout_s}s" python "$@"
      return $?
    fi
    require_conda_bin
    timeout --foreground --kill-after="${QUERY_STAGE_TIMEOUT_KILL_AFTER:-30}s" "${timeout_s}s" \
      env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XDG_CACHE_HOME="${GS_CACHE_ROOT}" TORCH_HOME="${GS_TORCH_HOME}" MPLCONFIGDIR="${GS_MPLCONFIGDIR}" CONDA_PKGS_DIRS="${GS_CONDA_PKGS_DIRS}" PIP_CACHE_DIR="${GS_PIP_CACHE_DIR}" \
      "${GS_CONDA_BIN}" run --no-capture-output -p "${GS_ENV_PATH}" python "$@"
    return $?
  fi
  gs_python "$@"
}

# Serialize GPU-heavy phase across parallel query workers, but let final rendering overlap.
QUERY_SERIALIZE_GPU_STAGE="${QUERY_SERIALIZE_GPU_STAGE:-1}"
QUERY_GPU_LOCK_FILE="${QUERY_GPU_LOCK_FILE:-/tmp/refergaussian_query_gpu.lock}"
QUERY_GPU_LOCK_FD=203
QUERY_GPU_LOCK_HELD=0

release_gpu_lock() {
  if [[ "${QUERY_GPU_LOCK_HELD}" == "1" ]]; then
    flock -u "${QUERY_GPU_LOCK_FD}" || true
    QUERY_GPU_LOCK_HELD=0
    echo "[gpu-lock] released: ${QUERY_GPU_LOCK_FILE}"
  fi
}

acquire_gpu_lock() {
  if [[ "${QUERY_SERIALIZE_GPU_STAGE}" != "1" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "${QUERY_GPU_LOCK_FILE}")"
  eval "exec ${QUERY_GPU_LOCK_FD}>\"${QUERY_GPU_LOCK_FILE}\""
  echo "[gpu-lock] waiting: ${QUERY_GPU_LOCK_FILE}"
  flock "${QUERY_GPU_LOCK_FD}"
  QUERY_GPU_LOCK_HELD=1
  echo "[gpu-lock] acquired: ${QUERY_GPU_LOCK_FILE}"
}

trap release_gpu_lock EXIT

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

mkdir -p "${OUTPUT_ROOT}"

if [[ "${QUERY_FORCE_RERUN:-0}" != "1" && -f "${FINAL_VALIDATION_PATH}" ]]; then
  echo "[skip] existing final validation for ${QUERY_NAME}: ${FINAL_VALIDATION_PATH}"
  echo "${QUERY_RUN_DIR}"
  exit 0
fi

acquire_gpu_lock

if [[ "${QUERY_REUSE_GROUNDED_SAM2:-0}" == "1" && -f "${TRACKS_PATH}" && -f "${OUTPUT_ROOT}/query_plan.json" ]]; then
  echo "[reuse] existing grounded_sam2 tracks and query_plan for ${QUERY_NAME}"
else
  bash "${GS_ROOT}/scripts/run_query_guided_grounded_sam2.sh" \
    "${RUN_DIR}" \
    "${DATASET_DIR}" \
    "${QUERY_TEXT}" \
    "${QUERY_NAME}"
fi

if gs_python "${GS_ROOT}/scripts/write_empty_query_selection.py" \
  --query "${QUERY_TEXT}" \
  --query-plan-path "${OUTPUT_ROOT}/query_plan.json" \
  --check-only >/dev/null 2>&1; then
  echo "[empty-query] query plan is explicitly empty; skipping proposal/entitybank stages for ${QUERY_NAME}"
  mkdir -p "${QUERY_RUN_DIR}"
  ln -sfn "${RUN_DIR}/config.yaml" "${QUERY_RUN_DIR}/config.yaml"
  ln -sfn "${RUN_DIR}/point_cloud" "${QUERY_RUN_DIR}/point_cloud"
  ln -sfn "${RUN_DIR}/test" "${QUERY_RUN_DIR}/test"
  rm -rf "${QUERY_RUN_DIR}/entitybank"
  mkdir -p "${QUERY_RUN_DIR}/entitybank"
  gs_python "${GS_ROOT}/scripts/write_empty_query_selection.py" \
    --query "${QUERY_TEXT}" \
    --query-plan-path "${OUTPUT_ROOT}/query_plan.json" \
    --output-path "${QWEN_SELECTION_PATH}"
  run_final_render_and_summary
  exit 0
fi

run_export_renderer_geometry_support() {
  if [[ "${QUERY_USE_RENDERER_GEOMETRY:-0}" != "1" ]]; then
    return 0
  fi
  _stage_wrapper renderer_geometry_support \
    run_gs_python_with_timeout "${QUERY_RENDERER_GEOMETRY_SUPPORT_TIMEOUT:-1200}" \
      "${GS_ROOT}/scripts/export_renderer_geometry.py" \
      --run-dir "${RUN_DIR}" \
      --dataset-dir "${DATASET_DIR}" \
      --tracks-path "${TRACKS_PATH}" \
      --output-dir "${RENDERER_GEOMETRY_SUPPORT_DIR}"
  export QUERY_RENDERER_GEOMETRY_SUPPORT_PATH="${RENDERER_GEOMETRY_SUPPORT_DIR}"
}

run_export_renderer_geometry_final() {
  if [[ "${QUERY_USE_RENDERER_GEOMETRY:-0}" != "1" ]]; then
    return 0
  fi
  local frame_args=(--all-dataset-frames)
  if [[ -n "${QUERY_RENDER_IMAGE_IDS_JSON:-}" && -f "${QUERY_RENDER_IMAGE_IDS_JSON}" ]]; then
    frame_args=(--image-ids-json "${QUERY_RENDER_IMAGE_IDS_JSON}")
  fi
  _stage_wrapper renderer_geometry_final \
    run_gs_python_with_timeout "${QUERY_RENDERER_GEOMETRY_FINAL_TIMEOUT:-1800}" \
      "${GS_ROOT}/scripts/export_renderer_geometry.py" \
      --run-dir "${RUN_DIR}" \
      --dataset-dir "${DATASET_DIR}" \
      "${frame_args[@]}" \
      --entitybank-dir "${QUERY_ENTITYBANK_DIR}" \
      --output-dir "${RENDERER_GEOMETRY_FINAL_DIR}"
  rm -f "${QUERY_ENTITYBANK_DIR}/renderer_geometry"
  ln -s "${RENDERER_GEOMETRY_FINAL_DIR}" "${QUERY_ENTITYBANK_DIR}/renderer_geometry"
}

run_build_query_proposal() {
  run_gs_python_with_timeout "${QUERY_PROPOSAL_STAGE_TIMEOUT:-0}" \
    "${GS_ROOT}/scripts/build_mask_supported_proposal_dir.py" \
    --run-dir "${RUN_DIR}" \
    --dataset-dir "${DATASET_DIR}" \
    --tracks-path "${TRACKS_PATH}" \
    --output-dir "${PROPOSAL_DIR}" \
    --max-track-frames "${QUERY_MAX_TRACK_FRAMES:-24}" \
    --min-gaussians "${QUERY_LIFT_MIN_GAUSSIANS:-192}" \
    --max-gaussians "${QUERY_LIFT_MAX_GAUSSIANS:-1280}" \
    --max-gaussians-per-frame "${QUERY_LIFT_MAX_GAUSSIANS_PER_FRAME:-18000}" \
    --gate-threshold "${QUERY_LIFT_GATE_THRESHOLD:-0.01}" \
    --graph-knn "${QUERY_LIFT_GRAPH_KNN:-24}" \
    --graph-radius-scale "${QUERY_LIFT_GRAPH_RADIUS_SCALE:-1.35}"
}

run_build_query_proposal_with_relaxed_retry() {
  if run_build_query_proposal; then
    return 0
  fi
  if [[ "${QUERY_RETRY_RELAXED_GSAM2:-0}" != "1" ]]; then
    echo "[error] query proposal build failed for ${QUERY_NAME}; no retry profile is enabled" >&2
    return 1
  fi
  echo "[warn] query proposal build failed for ${QUERY_NAME}; retrying because the active profile explicitly enables it" >&2
  GSAM2_REUSE_QUERY_PLAN=1 \
  GSAM2_PROMPT_TYPE="${QUERY_RELAXED_GSAM2_PROMPT_TYPE:-box}" \
  GSAM2_DETECTOR_FRAME_STRIDE="${QUERY_RELAXED_GSAM2_DETECTOR_STRIDE:-4}" \
  GSAM2_MAX_DETECTOR_FRAMES="${QUERY_RELAXED_GSAM2_MAX_DETECTOR_FRAMES:-96}" \
  GSAM2_DETECTION_TOP_K="${QUERY_RELAXED_GSAM2_DETECTION_TOPK:-8}" \
  GSAM2_BOX_THRESHOLD="${QUERY_RELAXED_GSAM2_BOX_THRESHOLD:-0.18}" \
  GSAM2_TEXT_THRESHOLD="${QUERY_RELAXED_GSAM2_TEXT_THRESHOLD:-0.10}" \
  GSAM2_NUM_ANCHOR_SEEDS="${QUERY_RELAXED_GSAM2_NUM_ANCHOR_SEEDS:-5}" \
  bash "${GS_ROOT}/scripts/run_query_guided_grounded_sam2.sh" \
    "${RUN_DIR}" \
    "${DATASET_DIR}" \
    "${QUERY_TEXT}" \
    "${QUERY_NAME}"
  run_build_query_proposal
}

run_export_entitybank_with_proposal() {
  local min_gaussians_per_entity="${QUERY_MIN_GAUSSIANS_PER_ENTITY:-32}"
  local requested_entity_count
  requested_entity_count="$(query_requested_entity_count)"
  if [[ "${requested_entity_count}" -gt 1 ]]; then
    min_gaussians_per_entity="${QUERY_MULTI_ENTITY_MIN_GAUSSIANS_PER_ENTITY:-4}"
  elif query_is_static_set; then
    min_gaussians_per_entity="${QUERY_STATIC_SET_MIN_GAUSSIANS_PER_ENTITY:-4}"
  fi
  gs_python "${GS_ROOT}/scripts/export_entitybank.py" \
    --run-dir "${RUN_DIR}" \
    --proposal-dir "${PROPOSAL_DIR}" \
    --proposal-strict \
    --proposal-supervision-mode "${QUERY_PROPOSAL_SUPERVISION_MODE:-guided}" \
    --output-dir "${QUERY_ENTITYBANK_DIR}" \
    --max-entities "${QUERY_MAX_ENTITIES:-12}" \
    --min-gaussians-per-entity "${min_gaussians_per_entity}"
}

query_requested_entity_count() {
  local query_plan_path="${TRACK_DIR}/query_plan.json"
  gs_python - "${query_plan_path}" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {}
if path.is_file():
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}

phrases = payload.get("primary_subject_phrases") or payload.get("query_subject_phrases") or []
phrases = [str(phrase).strip().lower() for phrase in phrases if str(phrase).strip()]
count = max(len(phrases), 1)
for phrase in phrases:
    normalized = " ".join(re.findall(r"[a-z0-9]+", phrase))
    if re.match(r"^(?:both|two)\b", normalized) or re.match(r"^(?:a )?pair of\b", normalized):
        count = max(count, 2)
    elif re.match(r"^three\b", normalized):
        count = max(count, 3)
print(count)
PY
}

query_is_static_set() {
  local q_lc
  q_lc="$(printf '%s' "${QUERY_TEXT}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${q_lc}" == *"stationary"* || "${q_lc}" == *"static"* || "${q_lc}" == *"not moving"* || "${q_lc}" == *"never move"* ]]; then
    return 0
  fi
  return 1
}

query_entitybank_size() {
  local entities_path="${QUERY_ENTITYBANK_DIR}/entities.json"
  if [[ ! -f "${entities_path}" ]]; then
    echo 0
    return 0
  fi
  gs_python - "${entities_path}" <<'PY'
import json
import sys
path = sys.argv[1]
try:
    payload = json.load(open(path, "r", encoding="utf-8"))
except Exception:
    print(0)
    raise SystemExit(0)
if isinstance(payload, dict):
    rows = payload.get("entities", [])
elif isinstance(payload, list):
    rows = payload
else:
    rows = []
print(len(rows))
PY
}

proposal_dir_ready() {
  [[ -f "${PROPOSAL_DIR}/entities.json" && -f "${PROPOSAL_DIR}/query_proposal_summary.json" ]]
}

query_entitybank_ready() {
  [[ -f "${QUERY_ENTITYBANK_DIR}/entities.json" ]]
}

replace_query_run_link() {
  local target="$1"
  local destination="$2"
  local resolved_target
  if [[ ! -e "${target}" ]]; then
    echo "[error] cannot link missing query-run target: ${target}" >&2
    return 2
  fi
  resolved_target="$(realpath -e "${target}")"
  rm -rf "${destination}"
  ln -sfn "${resolved_target}" "${destination}"
}

proposal_ready=0
run_export_renderer_geometry_support
if [[ "${QUERY_REUSE_PROPOSAL_DIR:-0}" == "1" ]] && proposal_dir_ready; then
  echo "[reuse] existing query proposal dir: ${PROPOSAL_DIR}"
  proposal_ready=1
elif run_build_query_proposal_with_relaxed_retry; then
  proposal_ready=1
fi

if [[ "${proposal_ready}" == "1" ]]; then
  if [[ "${QUERY_REUSE_ENTITYBANK:-0}" == "1" ]] && query_entitybank_ready; then
    echo "[reuse] existing query entitybank: ${QUERY_ENTITYBANK_DIR}"
  else
    run_export_entitybank_with_proposal
  fi
  requested_entity_count="$(query_requested_entity_count)"
  if [[ "${requested_entity_count}" -gt 1 ]]; then
    entity_count="$(query_entitybank_size)"
    if [[ "${entity_count}" -lt "${requested_entity_count}" ]]; then
      echo "[error] multi-entity query requires ${requested_entity_count} independently supported entities, but lifting produced ${entity_count}" >&2
      exit 3
    fi
  fi
else
  echo "[error] mask-supported lifting failed for ${QUERY_NAME}; inspect grounded_sam2 and proposal diagnostics" >&2
  exit 2
fi

run_export_renderer_geometry_final

mkdir -p "${QUERY_RUN_DIR}"
replace_query_run_link "${RUN_DIR}/config.yaml" "${QUERY_RUN_DIR}/config.yaml"
replace_query_run_link "${RUN_DIR}/point_cloud" "${QUERY_RUN_DIR}/point_cloud"
replace_query_run_link "${RUN_DIR}/test" "${QUERY_RUN_DIR}/test"
replace_query_run_link "${QUERY_ENTITYBANK_DIR}" "${QUERY_RUN_DIR}/entitybank"

gs_python "${GS_ROOT}/scripts/export_semantic_slots.py" --run-dir "${QUERY_RUN_DIR}"
gs_python "${GS_ROOT}/scripts/export_semantic_tracks.py" --run-dir "${QUERY_RUN_DIR}"
gs_python "${GS_ROOT}/scripts/export_semantic_priors.py" --run-dir "${QUERY_RUN_DIR}"
gs_python "${GS_ROOT}/scripts/export_native_semantics.py" --run-dir "${QUERY_RUN_DIR}"

# A declared Stage-1 multi-instance group already fixes the selected Gaussian
# members structurally.  Only take this path after every member has been
# lifted; all other queries retain the normal Qwen assignment/selection path.
AUTO_DECLARED_MULTI_INSTANCE=0
if [[ "${QUERY_AUTO_SKIP_QWEN_FOR_DECLARED_MULTIHYPOTHESIS:-0}" == "1" ]]; then
  declared_multi_instance="$(gs_python "${GS_ROOT}/scripts/check_declared_multi_instance.py" \
    --query-plan-path "${OUTPUT_ROOT}/query_plan.json" \
    --tracks-path "${TRACKS_PATH}" \
    --entitybank-path "${QUERY_RUN_DIR}/entitybank/entities.json" \
    --probe)"
  if [[ "${declared_multi_instance}" == "1" ]]; then
    AUTO_DECLARED_MULTI_INSTANCE=1
    echo "[semantic-fast-path] using declared Stage-1 multi-instance membership"
  fi
fi

ASSIGNMENTS_PATH="${QWEN_ASSIGNMENTS_PATH}"
if [[ "${QUERY_SKIP_QWEN_EXPORT:-0}" == "1" || "${AUTO_DECLARED_MULTI_INSTANCE}" == "1" ]]; then
  ASSIGNMENTS_PATH="${QUERY_RUN_DIR}/entitybank/native_semantic_assignments.json"
elif [[ "${QUERY_REUSE_QWEN_EXPORT:-0}" == "1" && -f "${QWEN_ASSIGNMENTS_PATH}" ]]; then
  echo "Reusing existing Qwen assignments: ${QWEN_ASSIGNMENTS_PATH}"
else
  gsam2_python "${GS_ROOT}/scripts/export_qwen_semantics.py" \
    --run-dir "${QUERY_RUN_DIR}" \
    --query "${QUERY_TEXT}" \
    --max-entities "${QUERY_QWEN_MAX_ENTITIES:-12}"
fi

if [[ "${QUERY_SKIP_ENTITY_LIBRARY:-0}" != "1" ]]; then
  gs_python "${GS_ROOT}/scripts/export_entity_library.py" \
    --run-dir "${QUERY_RUN_DIR}" \
    --dataset-dir "${DATASET_DIR}" \
    --assignments-path "${ASSIGNMENTS_PATH}" \
    --output-root "${ENTITY_LIBRARY_DIR}" \
    --background-mode source \
    --fps "${QUERY_RENDER_FPS:-6}" \
    --stride "${QUERY_RENDER_STRIDE:-1}"
fi

if [[ "${QUERY_REUSE_QWEN_SELECTION:-0}" == "1" && -f "${QWEN_SELECTION_PATH}" ]]; then
  echo "Reusing existing Qwen selection: ${QWEN_SELECTION_PATH}"
else
  if [[ "${AUTO_DECLARED_MULTI_INSTANCE}" == "1" ]]; then
    QUERY_SKIP_QWEN_SELECTION=1 gsam2_python "${GS_ROOT}/scripts/select_qwen_query_entities.py" \
      --assignments-path "${ASSIGNMENTS_PATH}" \
      --query "${QUERY_TEXT}" \
      --query-plan-path "${OUTPUT_ROOT}/query_plan.json" \
      --output-path "${QWEN_SELECTION_PATH}"
  else
    gsam2_python "${GS_ROOT}/scripts/select_qwen_query_entities.py" \
      --assignments-path "${ASSIGNMENTS_PATH}" \
      --query "${QUERY_TEXT}" \
      --query-plan-path "${OUTPUT_ROOT}/query_plan.json" \
      --output-path "${QWEN_SELECTION_PATH}"
  fi
fi

run_final_render_and_summary
