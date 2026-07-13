#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

DATASET="${1:-hypernerf}"
SCENE="${2:-interp/cut-lemon1}"
shift $(( $# > 1 ? 2 : $# ))

RUN_NAMESPACE="${GS_RUN_NAMESPACE:-baseline_4dgs}"
RUN_DIR="${GS_RUN_ROOT}/${RUN_NAMESPACE}/${DATASET}/${SCENE##*/}"
LOG_PATH="${RUN_DIR}/render.log"
META_PATH="${RUN_DIR}/render_meta.json"
PY_CMD="$(gs_python_cmd)"
EXTRA_ARGS="$(shell_join "$@")"

run_with_gpu_monitor "${LOG_PATH}" "${META_PATH}" \
  bash -lc "cd '${GS_ROOT}' && export PYTHONPATH='${PYTHONPATH}' && ${PY_CMD} external/4DGaussians/render.py -m '${RUN_DIR}' --iteration -1 ${EXTRA_ARGS}"

gs_python "${GS_ROOT}/scripts/quick_subset_metrics.py" \
  --run-dir "${RUN_DIR}" \
  --max-frames "${GS_QUICK_METRIC_FRAMES:-32}" \
  --with-lpips
gs_python "${GS_ROOT}/scripts/collect_metrics.py" --run-dir "${RUN_DIR}" --write-summary
gs_python "${GS_ROOT}/scripts/export_entitybank.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/export_semantic_slots.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/export_semantic_tracks.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/export_semantic_priors.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/export_segmentation_bootstrap.py" --run-dir "${RUN_DIR}" || true
