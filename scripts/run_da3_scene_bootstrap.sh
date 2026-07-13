#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

SCENE_DIR="${1:?scene dir required}"
OUTPUT_DIR="${2:-${GS_DATA_ROOT}/downloads/da3_$(basename "${SCENE_DIR}")}"
INPUT_PATH="${3:-${SCENE_DIR}/val}"
MAX_POINTS="${DA3_BOOTSTRAP_MAX_POINTS:-200000}"
shift $(( $# > 2 ? 3 : $# ))
EXTRA_ARGS=("$@")

mkdir -p "${OUTPUT_DIR}"
bash "${GS_ROOT}/scripts/run_da3_bootstrap.sh" "${INPUT_PATH}" "${OUTPUT_DIR}" "${EXTRA_ARGS[@]}"
gs_python "${GS_ROOT}/scripts/convert_da3_gs_ply_to_fused.py"   --da3-output "${OUTPUT_DIR}"   --scene-dir "${SCENE_DIR}"   --max-points "${MAX_POINTS}"
