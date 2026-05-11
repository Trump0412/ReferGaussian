#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

GROUP="${1:-interp}"
SCENE="${2:-cut-lemon1}"
PHASES="${HYPERNERF_SUITE_PHASES:-baseline,stellar_worldtube}"
RUN_SUFFIX="${HYPERNERF_SUITE_SUFFIX:-smoke300}"
COARSE_ITERS="${HYPERNERF_SUITE_COARSE_ITERS:-50}"
TRAIN_ITERS="${HYPERNERF_SUITE_TRAIN_ITERS:-300}"
TEST_ITERS="${HYPERNERF_SUITE_TEST_ITERS:-300}"
SAVE_ITERS="${HYPERNERF_SUITE_SAVE_ITERS:-300}"
SOURCE_ROOT="${HYPERNERF_SOURCE_ROOT:-${HYPERNERF_RAW_ROOT:-${GS_ROOT}/data/raw/HyperNeRF}/${GROUP}/${SCENE}}"

bash "${GS_ROOT}/scripts/prepare_local_hypernerf_scene.sh" "${SOURCE_ROOT}" "${GROUP}" "${SCENE}"

IFS=',' read -r -a phase_array <<< "${PHASES}"
for phase in "${phase_array[@]}"; do
  phase="$(echo "${phase}" | xargs)"
  if [[ -z "${phase}" ]]; then
    continue
  fi
  export GS_RUN_NAMESPACE="${phase}_${SCENE}_${RUN_SUFFIX}"
  case "${phase}" in
    baseline)
      bash "${GS_ROOT}/scripts/train_baseline.sh" hypernerf "${GROUP}/${SCENE}" \
        --coarse_iterations "${COARSE_ITERS}" \
        --iterations "${TRAIN_ITERS}" \
        --test_iterations "${TEST_ITERS}" \
        --save_iterations "${SAVE_ITERS}"
      bash "${GS_ROOT}/scripts/eval_baseline.sh" hypernerf "${GROUP}/${SCENE}"
      ;;
    stellar_worldtube)
      export TEMPORAL_WORLDTUBE_SAMPLES="${TEMPORAL_WORLDTUBE_SAMPLES:-5}"
      export TEMPORAL_WORLDTUBE_SPAN="${TEMPORAL_WORLDTUBE_SPAN:-0.75}"
      export TEMPORAL_WORLDTUBE_SIGMA="${TEMPORAL_WORLDTUBE_SIGMA:-0.45}"
      export TEMPORAL_WORLDTUBE_OPACITY_MIX="${TEMPORAL_WORLDTUBE_OPACITY_MIX:-1.0}"
      export TEMPORAL_WORLDTUBE_SCALE_MIX="${TEMPORAL_WORLDTUBE_SCALE_MIX:-0.12}"
      export TEMPORAL_ACCELERATION_ENABLED="${TEMPORAL_ACCELERATION_ENABLED:-1}"
      export GS_SKIP_FULL_METRICS="${GS_SKIP_FULL_METRICS:-1}"
      bash "${GS_ROOT}/scripts/train_stellar_worldtube.sh" hypernerf "${GROUP}/${SCENE}" \
        --coarse_iterations "${COARSE_ITERS}" \
        --iterations "${TRAIN_ITERS}" \
        --test_iterations "${TEST_ITERS}" \
        --save_iterations "${SAVE_ITERS}"
      bash "${GS_ROOT}/scripts/eval_stellar_worldtube.sh" hypernerf "${GROUP}/${SCENE}"
      ;;
    *)
      echo "Unsupported phase: ${phase}" >&2
      exit 2
      ;;
  esac
done

REPORT_OUTPUT="${GS_ROOT}/reports/hypernerf_showcase/${GROUP}_${SCENE}_${RUN_SUFFIX}_benchmark.md"
mkdir -p "$(dirname "${REPORT_OUTPUT}")"
SUBTITLE="${HYPERNERF_SUITE_SUBTITLE:-Baseline vs ReferGaussian worldtube comparison for a local HyperNeRF scene.}"
build_args=(
  --title "HyperNeRF ${GROUP}/${SCENE} ${RUN_SUFFIX}"
  --subtitle "${SUBTITLE}"
  --output "${REPORT_OUTPUT}"
)
for phase in "${phase_array[@]}"; do
  phase="$(echo "${phase}" | xargs)"
  [[ -z "${phase}" ]] && continue
  build_args+=(--entry "${phase}=${GS_ROOT}/runs/${phase}_${SCENE}_${RUN_SUFFIX}/hypernerf/${SCENE}")
done
gs_python "${GS_ROOT}/scripts/build_benchmark_report.py" "${build_args[@]}"
echo "Wrote ${REPORT_OUTPUT}"
