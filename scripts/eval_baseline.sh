#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_4dgaussians

DATASET="${1:-dnerf}"
SCENE="${2:-bouncingballs}"
shift $(( $# > 1 ? 2 : $# ))

PY_CMD="$(gs_python_cmd)"
EXTRA_ARGS="$(shell_join "$@")"
RUN_NAMESPACE="${GS_RUN_NAMESPACE:-baseline_4dgs}"
RUN_DIR="${GS_RUN_ROOT}/${RUN_NAMESPACE}/${DATASET}/${SCENE##*/}"
LOG_PATH="${RUN_DIR}/render.log"
META_PATH="${RUN_DIR}/render_meta.json"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Missing baseline run directory: ${RUN_DIR}" >&2
  exit 2
fi

run_with_gpu_monitor "${LOG_PATH}" "${META_PATH}" \
  bash -lc "cd '${GS_ROOT}' && export PYTHONPATH='${PYTHONPATH}' && ${PY_CMD} external/4DGaussians/render.py -m '${RUN_DIR}' --iteration -1 ${EXTRA_ARGS}"

if [[ "${GS_SKIP_FULL_METRICS:-0}" == "1" ]]; then
  gs_python "${GS_ROOT}/scripts/quick_subset_metrics.py" \
    --run-dir "${RUN_DIR}" \
    --max-frames "${GS_QUICK_METRIC_FRAMES:-32}" \
    --with-lpips
else
  bash -lc "cd '${GS_ROOT}' && export PYTHONPATH='${PYTHONPATH}' && ${PY_CMD} external/4DGaussians/metrics.py -m '${RUN_DIR}'" | tee -a "${RUN_DIR}/metrics.log"
fi

gs_python "${GS_ROOT}/scripts/collect_metrics.py" --run-dir "${RUN_DIR}" --write-summary
