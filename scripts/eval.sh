#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_4dgaussians

DATASET="${1:-dnerf}"
SCENE="${2:-mutant}"
shift $(( $# > 1 ? 2 : $# ))
PY_CMD="$(gs_python_cmd)"
EXTRA_ARGS="$(shell_join "$@")"
RUN_NAMESPACE="${GS_RUN_NAMESPACE:-refergaussian}"
TEMPORAL_GATE_MIX_VALUE="${TEMPORAL_GATE_MIX:-1.0}"
TEMPORAL_DRIFT_MIX_VALUE="${TEMPORAL_DRIFT_MIX:-1.0}"
TEMPORAL_ACCELERATION_ENABLED_VALUE="${TEMPORAL_ACCELERATION_ENABLED:-1}"

RUN_DIR="${GS_ROOT}/runs/${RUN_NAMESPACE}/${DATASET}/${SCENE##*/}"
LOG_PATH="${RUN_DIR}/render.log"
META_PATH="${RUN_DIR}/render_meta.json"

ACCEL_ARGS=""
if [[ "${TEMPORAL_ACCELERATION_ENABLED_VALUE}" == "1" ]]; then
  ACCEL_ARGS="--temporal_acceleration_enabled"
fi

run_with_gpu_monitor "${LOG_PATH}" "${META_PATH}" \
  bash -lc "cd '${GS_ROOT}' && export PYTHONPATH='${PYTHONPATH}' && ${PY_CMD} external/4DGaussians/render.py -m '${RUN_DIR}' --iteration -1 --warp_enabled --temporal_warp_type 'contextual' --temporal_extent_enabled --temporal_gate_sharpness '${TEMPORAL_GATE_SHARPNESS:-1.0}' --temporal_drift_scale '${TEMPORAL_DRIFT_SCALE:-1.0}' --temporal_gate_mix '${TEMPORAL_GATE_MIX_VALUE}' --temporal_drift_mix '${TEMPORAL_DRIFT_MIX_VALUE}' --temporal_tube_enabled --temporal_tube_samples '${TEMPORAL_TUBE_SAMPLES:-5}' --temporal_tube_span '${TEMPORAL_TUBE_SPAN:-1.0}' --temporal_tube_sigma '${TEMPORAL_TUBE_SIGMA:-0.75}' --temporal_tube_weight_power '${TEMPORAL_TUBE_WEIGHT_POWER:-1.0}' --temporal_tube_covariance_mix '${TEMPORAL_TUBE_COVARIANCE_MIX:-1.0}' ${ACCEL_ARGS} ${EXTRA_ARGS}"

if [[ "${GS_SKIP_FULL_METRICS:-0}" == "1" ]]; then
  gs_python "${GS_ROOT}/scripts/quick_subset_metrics.py" \
    --run-dir "${RUN_DIR}" \
    --max-frames "${GS_QUICK_METRIC_FRAMES:-32}" \
    --with-lpips
else
  bash -lc "cd '${GS_ROOT}' && export PYTHONPATH='${PYTHONPATH}' && ${PY_CMD} external/4DGaussians/metrics.py -m '${RUN_DIR}'" | tee -a "${RUN_DIR}/metrics.log"
fi
gs_python "${GS_ROOT}/scripts/plot_time_warp.py" --run-dir "${RUN_DIR}"
gs_python "${GS_ROOT}/scripts/export_entitybank.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/export_semantic_slots.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/export_semantic_tracks.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/export_semantic_priors.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/export_segmentation_bootstrap.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/collect_metrics.py" --run-dir "${RUN_DIR}" --write-summary
