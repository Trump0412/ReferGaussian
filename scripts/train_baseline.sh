#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

DATASET="${1:-dnerf}"
SCENE="${2:-bouncingballs}"
shift $(( $# > 1 ? 2 : $# ))

SOURCE_PATH="$(dataset_source_path "${DATASET}" "${SCENE}")"
CONFIG_PATH="$(dataset_config_path "${DATASET}" "${SCENE}")"
RUN_NAMESPACE="${GS_RUN_NAMESPACE:-baseline_4dgs}"
RUN_DIR="${GS_ROOT}/runs/${RUN_NAMESPACE}/${DATASET}/${SCENE##*/}"
LOG_PATH="${RUN_DIR}/train.log"
META_PATH="${RUN_DIR}/train_meta.json"
PY_CMD="$(gs_python_cmd)"
EXTRA_ARGS="$(shell_join "$@")"
TRAIN_SEED="${REFERGAUSSIAN_SEED:-6666}"

mkdir -p "${RUN_DIR}"
cat > "${RUN_DIR}/config.yaml" <<EOF
phase: baseline
dataset: ${DATASET}
scene: ${SCENE}
source_path: ${SOURCE_PATH}
config_path: ${CONFIG_PATH}
seed: ${TRAIN_SEED}
warp_enabled: false
EOF

run_with_gpu_monitor "${LOG_PATH}" "${META_PATH}" \
  bash -lc "cd '${GS_ROOT}' && export PYTHONPATH='${PYTHONPATH}' && ${PY_CMD} external/4DGaussians/train.py -s '${SOURCE_PATH}' -m '${RUN_DIR}' --expname '${DATASET}/${SCENE##*/}' --configs '${CONFIG_PATH}' --port 6017 --seed '${TRAIN_SEED}' ${EXTRA_ARGS}"

gs_python "${GS_ROOT}/scripts/collect_metrics.py" --run-dir "${RUN_DIR}" --write-summary || true
