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
SURFACE_FIELD_DIR="${OUTPUT_ROOT}/surface_mask_field"
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

  gs_python "${GS_ROOT}/scripts/render_query_video.py" \
    --run-dir "${QUERY_RUN_DIR}" \
    --dataset-dir "${DATASET_DIR}" \
    --selection-path "${QWEN_SELECTION_PATH}" \
    --output-dir "${FINAL_RENDER_DIR}" \
    --eval-profile "${QUERY_EVAL_PROFILE}" \
    --background-mode source \
    --fps "${QUERY_RENDER_FPS:-6}" \
    --stride "${QUERY_RENDER_STRIDE:-1}"

  if [[ -n "${QUERY_ANNOTATION_DIR:-}" && -d "${QUERY_ANNOTATION_DIR}" ]]; then
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

QUERY_PROPOSAL_BUILDER="${QUERY_PROPOSAL_BUILDER:-mask_supported_lifting}"
LEGACY_ENTITYBANK_FALLBACK_DISABLED="${DISABLE_LEGACY_ENTITYBANK_FALLBACK:-0}"

legacy_entitybank_fallback_disabled() {
  if [[ "${LEGACY_ENTITYBANK_FALLBACK_DISABLED}" == "1" ]]; then
    return 0
  fi
  if [[ "${QUERY_PROPOSAL_BUILDER}" == "surface_mask_field" ]]; then
    return 0
  fi
  return 1
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
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

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
  --check-only; then
  echo "[empty-query] query plan is explicitly empty; skipping proposal/entitybank stages for ${QUERY_NAME}"
  mkdir -p "${QUERY_RUN_DIR}"
  ln -sfn "${RUN_DIR}/config.yaml" "${QUERY_RUN_DIR}/config.yaml"
  ln -sfn "${RUN_DIR}/point_cloud" "${QUERY_RUN_DIR}/point_cloud"
  ln -sfn "${RUN_DIR}/test" "${QUERY_RUN_DIR}/test"
  mkdir -p "${QUERY_RUN_DIR}/entitybank"
  gs_python "${GS_ROOT}/scripts/write_empty_query_selection.py" \
    --query "${QUERY_TEXT}" \
    --query-plan-path "${OUTPUT_ROOT}/query_plan.json" \
    --output-path "${QWEN_SELECTION_PATH}"
  run_final_render_and_summary
  exit 0
fi

run_build_query_proposal() {
  local builder="${QUERY_PROPOSAL_BUILDER}"
  local legacy_cluster_mode="${QUERY_CLUSTER_MODE:-support_only}"
  local legacy_opacity_power="${QUERY_OPACITY_POWER:-0.0}"
  if [[ "${builder}" == "worldtube_consistency" && -z "${QUERY_CLUSTER_MODE:-}" ]]; then
    legacy_cluster_mode="worldtube_consistency"
  fi
  if [[ "${builder}" == "worldtube_consistency" && -z "${QUERY_OPACITY_POWER:-}" ]]; then
    legacy_opacity_power="1.0"
  fi
  case "${builder}" in
    mask_supported_lifting)
      gs_python "${GS_ROOT}/scripts/build_joint_query_proposal_dir.py" \
        --run-dir "${RUN_DIR}" \
        --dataset-dir "${DATASET_DIR}" \
        --tracks-path "${TRACKS_PATH}" \
        --output-dir "${PROPOSAL_DIR}" \
        --proposal-builder mask_supported_lifting \
        --max-track-frames "${QUERY_MAX_TRACK_FRAMES:-24}" \
        --min-gaussians "${QUERY_LIFT_MIN_GAUSSIANS:-192}" \
        --max-gaussians "${QUERY_LIFT_MAX_GAUSSIANS:-1280}" \
        --max-gaussians-per-frame "${QUERY_LIFT_MAX_GAUSSIANS_PER_FRAME:-18000}" \
        --gate-threshold "${QUERY_LIFT_GATE_THRESHOLD:-0.01}" \
        --graph-knn "${QUERY_LIFT_GRAPH_KNN:-24}" \
        --graph-radius-scale "${QUERY_LIFT_GRAPH_RADIUS_SCALE:-1.35}"
      ;;
    seeded_local_graph|joint|joint_embedding|joint_worldtube_embedding|mask_supervised_joint)
      gs_python "${GS_ROOT}/scripts/build_joint_query_proposal_dir.py" \
        --run-dir "${RUN_DIR}" \
        --dataset-dir "${DATASET_DIR}" \
        --tracks-path "${TRACKS_PATH}" \
        --output-dir "${PROPOSAL_DIR}" \
        --max-track-frames "${QUERY_MAX_TRACK_FRAMES:-16}" \
        --proposal-keep-ratio "${QUERY_JOINT_KEEP_RATIO:-0.03}" \
        --min-gaussians "${QUERY_JOINT_MIN_GAUSSIANS:-384}" \
        --max-gaussians "${QUERY_JOINT_MAX_GAUSSIANS:-1024}" \
        --chunk-size "${QUERY_JOINT_CHUNK_SIZE:-4096}" \
        --embed-dim "${QUERY_JOINT_EMBED_DIM:-16}" \
        --num-steps "${QUERY_JOINT_NUM_STEPS:-400}" \
        --lr "${QUERY_JOINT_LR:-0.01}" \
        --identity-cluster-mode "${QUERY_IDENTITY_CLUSTER_MODE:-seeded_local_graph}" \
        --graph-knn "${QUERY_GRAPH_KNN:-24}" \
        --graph-radius-scale "${QUERY_GRAPH_RADIUS_SCALE:-1.25}" \
        --graph-min-component-size "${QUERY_GRAPH_MIN_COMPONENT_SIZE:-24}"
      ;;
    support_only|worldtube_consistency|legacy_support)
      gs_python "${GS_ROOT}/scripts/build_query_proposal_dir.py" \
        --run-dir "${RUN_DIR}" \
        --dataset-dir "${DATASET_DIR}" \
        --tracks-path "${TRACKS_PATH}" \
        --output-dir "${PROPOSAL_DIR}" \
        --max-track-frames "${QUERY_MAX_TRACK_FRAMES:-16}" \
        --proposal-keep-ratio "${QUERY_PROPOSAL_KEEP_RATIO:-0.03}" \
        --min-gaussians "${QUERY_MIN_GAUSSIANS:-256}" \
        --max-gaussians "${QUERY_MAX_GAUSSIANS:-4096}" \
        --opacity-power "${legacy_opacity_power}" \
        --cluster-mode "${legacy_cluster_mode}" \
        --seed-ratio "${QUERY_SEED_RATIO:-0.05}" \
        --expansion-factor "${QUERY_EXPANSION_FACTOR:-4.0}"
      ;;
    *)
      echo "[error] unknown QUERY_PROPOSAL_BUILDER=${builder}" >&2
      return 2
      ;;
  esac
}

run_surface_mask_field_entitybank() {
  gs_python "${GS_ROOT}/scripts/train_surface_mask_field.py" \
    --run-dir "${RUN_DIR}" \
    --dataset-dir "${DATASET_DIR}" \
    --stage1-root "${TRACK_DIR}" \
    --output-dir "${SURFACE_FIELD_DIR}" \
    --num-steps "${QUERY_SURFACE_NUM_STEPS:-2000}" \
    --lr "${QUERY_SURFACE_LR:-0.05}" \
    --feature-dim "${QUERY_SURFACE_FEATURE_DIM:-32}" \
    --render-res-scale "${QUERY_SURFACE_RENDER_RES_SCALE:-1.0}" \
    --train-frame-stride "${QUERY_SURFACE_TRAIN_FRAME_STRIDE:-2}" \
    --val-frame-stride "${QUERY_SURFACE_VAL_FRAME_STRIDE:-8}" \
    --val-interval "${QUERY_SURFACE_VAL_INTERVAL:-100}" \
    --log-interval "${QUERY_SURFACE_LOG_INTERVAL:-20}" \
    --max-gaussians-per-frame "${QUERY_SURFACE_MAX_GAUSSIANS_PER_FRAME:-16000}" \
    --gate-threshold "${QUERY_SURFACE_GATE_THRESHOLD:-0.01}" \
    --core-kernel "${QUERY_SURFACE_CORE_KERNEL:-3}" \
    --outer-kernel "${QUERY_SURFACE_OUTER_KERNEL:-15}" \
    --knn-k "${QUERY_SURFACE_KNN_K:-8}" \
    --knn-radius "${QUERY_SURFACE_KNN_RADIUS:-0.08}" \
    --knn-max-pairs "${QUERY_SURFACE_KNN_MAX_PAIRS:-128}" \
    --relative-threshold "${QUERY_SURFACE_RELATIVE_THRESHOLD:-0.12}" \
    --absolute-threshold "${QUERY_SURFACE_ABSOLUTE_THRESHOLD:-0.008}" \
    --full-positive-weight "${QUERY_SURFACE_FULL_POSITIVE_WEIGHT:-1.5}" \
    --support-loss-weight "${QUERY_SURFACE_SUPPORT_LOSS_WEIGHT:-0.8}" \
    --temporal-consistency-weight "${QUERY_SURFACE_TEMPORAL_CONSISTENCY_WEIGHT:-0.55}" \
    --graph-same-weight "${QUERY_SURFACE_GRAPH_SAME_WEIGHT:-0.10}" \
    --graph-contrast-weight "${QUERY_SURFACE_GRAPH_CONTRAST_WEIGHT:-0.18}" \
    --feature-contrast-weight "${QUERY_SURFACE_FEATURE_CONTRAST_WEIGHT:-0.12}" \
    --contrast-temperature "${QUERY_SURFACE_CONTRAST_TEMPERATURE:-0.18}" \
    --device "${QUERY_SURFACE_DEVICE:-cuda}"
  if [[ ! -f "${SURFACE_FIELD_DIR}/field_best.pt" ]]; then
    echo "[error] Surface-Mask Field is required; legacy fallback is disabled." >&2
    return 2
  fi

  gs_python "${GS_ROOT}/scripts/export_surface_entitybank.py" \
    --run-dir "${RUN_DIR}" \
    --dataset-dir "${DATASET_DIR}" \
    --surface-field-dir "${SURFACE_FIELD_DIR}" \
    --stage1-root "${TRACK_DIR}" \
    --output-dir "${QUERY_ENTITYBANK_DIR}" \
    --threshold-search \
    --component-prune \
    --min-render-iou "${QUERY_SURFACE_MIN_RENDER_IOU:-0.10}" \
    --min-render-recall "${QUERY_SURFACE_MIN_RENDER_RECALL:-0.10}" \
    --min-render-precision "${QUERY_SURFACE_MIN_RENDER_PRECISION:-0.55}" \
    --max-area-ratio "${QUERY_SURFACE_MAX_AREA_RATIO:-1.20}" \
    --max-gaussians-per-frame "${QUERY_SURFACE_MAX_GAUSSIANS_PER_FRAME:-16000}" \
    --gate-threshold "${QUERY_SURFACE_GATE_THRESHOLD:-0.01}" \
    --knn-k "${QUERY_SURFACE_COMPONENT_KNN_K:-8}" \
    --component-radius-scale "${QUERY_SURFACE_COMPONENT_RADIUS_SCALE:-4.0}" \
    --graph-knn "${QUERY_SURFACE_EXPORT_GRAPH_KNN:-24}" \
    --graph-min-votes "${QUERY_SURFACE_EXPORT_GRAPH_MIN_VOTES:-4}" \
    --device "${QUERY_SURFACE_DEVICE:-cuda}"
}

run_build_query_proposal_with_fallback() {
  local primary_builder="${QUERY_PROPOSAL_BUILDER}"
  if run_build_query_proposal; then
    return 0
  fi
  if [[ "${primary_builder}" != "support_only" && "${primary_builder}" != "worldtube_consistency" && "${primary_builder}" != "legacy_support" && "${primary_builder}" != "mask_supported_lifting" && "${QUERY_PROPOSAL_ALLOW_FALLBACK_SUPPORT:-0}" == "1" ]] && ! legacy_entitybank_fallback_disabled; then
    echo "[warn] proposal builder ${primary_builder} failed for ${QUERY_NAME}; retrying legacy support-only proposal builder" >&2
    QUERY_PROPOSAL_BUILDER="support_only" run_build_query_proposal
    return $?
  fi
  return 1
}

run_build_query_proposal_with_relaxed_retry() {
  if run_build_query_proposal_with_fallback; then
    return 0
  fi
  echo "[warn] query proposal build failed for ${QUERY_NAME}; trying relaxed GSAM2 retry" >&2
  if [[ "${QUERY_RETRY_RELAXED_GSAM2:-1}" != "1" ]]; then
    return 1
  fi
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
  run_build_query_proposal_with_fallback
}

run_build_query_proposal_legacy() {
  gs_python "${GS_ROOT}/scripts/build_query_proposal_dir.py" \
    --run-dir "${RUN_DIR}" \
    --dataset-dir "${DATASET_DIR}" \
    --tracks-path "${TRACKS_PATH}" \
    --output-dir "${PROPOSAL_DIR}" \
    --max-track-frames "${QUERY_MAX_TRACK_FRAMES:-16}" \
    --proposal-keep-ratio "${QUERY_PROPOSAL_KEEP_RATIO:-0.03}" \
    --min-gaussians "${QUERY_MIN_GAUSSIANS:-256}" \
    --max-gaussians "${QUERY_MAX_GAUSSIANS:-4096}" \
    --opacity-power "${QUERY_OPACITY_POWER:-0.0}" \
    --cluster-mode "${QUERY_CLUSTER_MODE:-support_only}" \
    --seed-ratio "${QUERY_SEED_RATIO:-0.05}" \
    --expansion-factor "${QUERY_EXPANSION_FACTOR:-4.0}"
}

run_export_entitybank_with_proposal() {
  gs_python "${GS_ROOT}/scripts/export_entitybank.py" \
    --run-dir "${RUN_DIR}" \
    --proposal-dir "${PROPOSAL_DIR}" \
    --proposal-strict \
    --proposal-supervision-mode "${QUERY_PROPOSAL_SUPERVISION_MODE:-guided}" \
    --output-dir "${QUERY_ENTITYBANK_DIR}" \
    --max-entities "${QUERY_MAX_ENTITIES:-12}" \
    --min-gaussians-per-entity "${QUERY_MIN_GAUSSIANS_PER_ENTITY:-32}"
}

run_export_entitybank_fallback() {
  gs_python "${GS_ROOT}/scripts/export_entitybank.py" \
    --run-dir "${RUN_DIR}" \
    --output-dir "${QUERY_ENTITYBANK_DIR}" \
    --max-entities "${QUERY_MAX_ENTITIES:-12}" \
    --min-gaussians-per-entity "${QUERY_MIN_GAUSSIANS_PER_ENTITY:-32}"
}

query_requires_dual_hands() {
  local q_lc
  q_lc="$(printf '%s' "${QUERY_TEXT}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${q_lc}" == *"both hands"* || "${q_lc}" == *"two hands"* || "${q_lc}" == *"left hand and right hand"* || "${q_lc}" == *"right hand and left hand"* ]]; then
    return 0
  fi
  if [[ "${QUERY_TEXT}" == *"双手"* || "${QUERY_TEXT}" == *"两只手"* || "${QUERY_TEXT}" == *"两手"* || "${QUERY_TEXT}" == *"左右手"* ]]; then
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

proposal_ready=0
if [[ "${QUERY_PROPOSAL_BUILDER}" == "surface_mask_field" ]]; then
  run_surface_mask_field_entitybank
else
  if run_build_query_proposal_with_relaxed_retry; then
    proposal_ready=1
  fi

  if [[ "${proposal_ready}" == "1" ]]; then
    run_export_entitybank_with_proposal
    if query_requires_dual_hands; then
      entity_count="$(query_entitybank_size)"
      if [[ "${entity_count}" -lt 2 ]]; then
        if [[ "${QUERY_ALLOW_FULLSCENE_FALLBACK:-0}" == "1" ]] && ! legacy_entitybank_fallback_disabled; then
          echo "[warn] dual-hand query but proposal entitybank has ${entity_count} entities; switching to full-scene entitybank fallback" >&2
          run_export_entitybank_fallback
        else
          echo "[error] dual-hand query but proposal entitybank has only ${entity_count} entities; fallback is disabled" >&2
          exit 3
        fi
      fi
    fi
  else
    if [[ "${QUERY_ALLOW_FULLSCENE_FALLBACK:-0}" == "1" ]] && ! legacy_entitybank_fallback_disabled; then
      echo "[warn] proposal path unavailable for ${QUERY_NAME}; using full-scene entitybank fallback" >&2
      run_export_entitybank_fallback
    else
      echo "[error] proposal path unavailable for ${QUERY_NAME}; full-scene fallback is disabled" >&2
      exit 2
    fi
  fi
fi

mkdir -p "${QUERY_RUN_DIR}"
ln -sfn "${RUN_DIR}/config.yaml" "${QUERY_RUN_DIR}/config.yaml"
ln -sfn "${RUN_DIR}/point_cloud" "${QUERY_RUN_DIR}/point_cloud"
ln -sfn "${RUN_DIR}/test" "${QUERY_RUN_DIR}/test"
ln -sfn "${QUERY_ENTITYBANK_DIR}" "${QUERY_RUN_DIR}/entitybank"

gs_python "${GS_ROOT}/scripts/export_semantic_slots.py" --run-dir "${QUERY_RUN_DIR}"
gs_python "${GS_ROOT}/scripts/export_semantic_tracks.py" --run-dir "${QUERY_RUN_DIR}"
gs_python "${GS_ROOT}/scripts/export_semantic_priors.py" --run-dir "${QUERY_RUN_DIR}"
gs_python "${GS_ROOT}/scripts/export_native_semantics.py" --run-dir "${QUERY_RUN_DIR}"
ASSIGNMENTS_PATH="${QWEN_ASSIGNMENTS_PATH}"
if [[ "${QUERY_SKIP_QWEN_EXPORT:-0}" == "1" ]]; then
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
  gsam2_python "${GS_ROOT}/scripts/select_qwen_query_entities.py" \
    --assignments-path "${ASSIGNMENTS_PATH}" \
    --query "${QUERY_TEXT}" \
    --query-plan-path "${OUTPUT_ROOT}/query_plan.json" \
    --output-path "${QWEN_SELECTION_PATH}"
fi

run_final_render_and_summary
