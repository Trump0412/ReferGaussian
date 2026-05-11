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
PORT_VALUE="${GS_PORT:-6021}"
TEMPORAL_GATE_MIX_VALUE="${TEMPORAL_GATE_MIX:-1.0}"
TEMPORAL_DRIFT_MIX_VALUE="${TEMPORAL_DRIFT_MIX:-1.0}"
TEMPORAL_ACCELERATION_ENABLED_VALUE="${TEMPORAL_ACCELERATION_ENABLED:-1}"

SOURCE_PATH="$(dataset_source_path "${DATASET}" "${SCENE}")"
CONFIG_PATH="$(dataset_config_path "${DATASET}" "${SCENE}")"
RUN_DIR="${GS_ROOT}/runs/${RUN_NAMESPACE}/${DATASET}/${SCENE##*/}"
LOG_PATH="${RUN_DIR}/train.log"
META_PATH="${RUN_DIR}/train_meta.json"

mkdir -p "${RUN_DIR}"
cat > "${RUN_DIR}/config.yaml" <<EOF
phase: refergaussian
dataset: ${DATASET}
scene: ${SCENE}
source_path: ${SOURCE_PATH}
config_path: ${CONFIG_PATH}
port: ${PORT_VALUE}
warp_enabled: true
temporal_warp_type: contextual
warp_hidden_dim: ${WARP_HIDDEN_DIM:-32}
warp_num_layers: ${WARP_NUM_LAYERS:-2}
warp_num_bins: ${WARP_NUM_BINS:-128}
warp_mono_weight: ${WARP_MONO_WEIGHT:-0.05}
warp_smooth_weight: ${WARP_SMOOTH_WEIGHT:-0.01}
warp_budget_weight: ${WARP_BUDGET_WEIGHT:-0.01}
warp_sample_count: ${WARP_SAMPLE_COUNT:-128}
temporal_lr_init: ${TEMPORAL_LR_INIT:-0.00016}
temporal_lr_final: ${TEMPORAL_LR_FINAL:-0.000016}
temporal_lr_delay_mult: ${TEMPORAL_LR_DELAY_MULT:-0.01}
temporal_extent_enabled: true
temporal_gate_sharpness: ${TEMPORAL_GATE_SHARPNESS:-1.0}
temporal_drift_scale: ${TEMPORAL_DRIFT_SCALE:-1.0}
temporal_gate_mix: ${TEMPORAL_GATE_MIX_VALUE}
temporal_drift_mix: ${TEMPORAL_DRIFT_MIX_VALUE}
temporal_acceleration_enabled: ${TEMPORAL_ACCELERATION_ENABLED_VALUE}
temporal_velocity_reg_weight: ${TEMPORAL_VELOCITY_REG_WEIGHT:-0.0}
temporal_acceleration_reg_weight: ${TEMPORAL_ACCELERATION_REG_WEIGHT:-0.0}
temporal_tube_enabled: true
temporal_tube_samples: ${TEMPORAL_TUBE_SAMPLES:-5}
temporal_tube_span: ${TEMPORAL_TUBE_SPAN:-1.0}
temporal_tube_sigma: ${TEMPORAL_TUBE_SIGMA:-0.75}
temporal_tube_weight_power: ${TEMPORAL_TUBE_WEIGHT_POWER:-1.0}
temporal_tube_covariance_mix: ${TEMPORAL_TUBE_COVARIANCE_MIX:-1.0}
EOF

ACCEL_ARGS=""
if [[ "${TEMPORAL_ACCELERATION_ENABLED_VALUE}" == "1" ]]; then
  ACCEL_ARGS="--temporal_acceleration_enabled --temporal_acceleration_reg_weight '${TEMPORAL_ACCELERATION_REG_WEIGHT:-0.0}'"
fi

run_with_gpu_monitor "${LOG_PATH}" "${META_PATH}" \
  bash -lc "cd '${GS_ROOT}' && export PYTHONPATH='${PYTHONPATH}' && ${PY_CMD} external/4DGaussians/train.py -s '${SOURCE_PATH}' -m '${RUN_DIR}' --expname 'refergaussian/${DATASET}/${SCENE##*/}' --configs '${CONFIG_PATH}' --port '${PORT_VALUE}' --warp_enabled --temporal_warp_type 'contextual' --warp_hidden_dim '${WARP_HIDDEN_DIM:-32}' --warp_num_layers '${WARP_NUM_LAYERS:-2}' --warp_num_bins '${WARP_NUM_BINS:-128}' --warp_mono_weight '${WARP_MONO_WEIGHT:-0.05}' --warp_smooth_weight '${WARP_SMOOTH_WEIGHT:-0.01}' --warp_budget_weight '${WARP_BUDGET_WEIGHT:-0.01}' --warp_sample_count '${WARP_SAMPLE_COUNT:-128}' --temporal_lr_init '${TEMPORAL_LR_INIT:-0.00016}' --temporal_lr_final '${TEMPORAL_LR_FINAL:-0.000016}' --temporal_lr_delay_mult '${TEMPORAL_LR_DELAY_MULT:-0.01}' --temporal_extent_enabled --temporal_gate_sharpness '${TEMPORAL_GATE_SHARPNESS:-1.0}' --temporal_drift_scale '${TEMPORAL_DRIFT_SCALE:-1.0}' --temporal_gate_mix '${TEMPORAL_GATE_MIX_VALUE}' --temporal_drift_mix '${TEMPORAL_DRIFT_MIX_VALUE}' --temporal_velocity_reg_weight '${TEMPORAL_VELOCITY_REG_WEIGHT:-0.0}' --temporal_tube_enabled --temporal_tube_samples '${TEMPORAL_TUBE_SAMPLES:-5}' --temporal_tube_span '${TEMPORAL_TUBE_SPAN:-1.0}' --temporal_tube_sigma '${TEMPORAL_TUBE_SIGMA:-0.75}' --temporal_tube_weight_power '${TEMPORAL_TUBE_WEIGHT_POWER:-1.0}' --temporal_tube_covariance_mix '${TEMPORAL_TUBE_COVARIANCE_MIX:-1.0}' ${ACCEL_ARGS} ${EXTRA_ARGS}"

gs_python "${GS_ROOT}/scripts/collect_metrics.py" --run-dir "${RUN_DIR}" --write-summary || true
gs_python "${GS_ROOT}/scripts/export_entitybank.py" --run-dir "${RUN_DIR}" || true
