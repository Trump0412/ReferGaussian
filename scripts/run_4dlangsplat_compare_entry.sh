#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

SCENE="${1:-}"
SOURCE_ROOT="${2:-}"
GROUP="${3:-interp}"
RUN_SUFFIX="${4:-compare5k}"

if [[ -z "${SCENE}" || -z "${SOURCE_ROOT}" ]]; then
  echo "Usage: $0 <scene> </abs/hypernerf_scene_root> [group=interp] [run_suffix=compare5k]" >&2
  exit 2
fi

ANNOT_ROOT="${GS_ROOT}/data/benchmarks/4dlangsplat/HyperNeRF-Annotation"
ANNOT_SCENE_DIR="${ANNOT_ROOT}/${SCENE}"
if [[ ! -d "${ANNOT_SCENE_DIR}" ]]; then
  echo "Missing 4DLangSplat annotation scene: ${ANNOT_SCENE_DIR}" >&2
  exit 2
fi

if [[ ! -f "${ANNOT_SCENE_DIR}/video_annotations.json" ]]; then
  echo "Missing annotation file: ${ANNOT_SCENE_DIR}/video_annotations.json" >&2
  exit 2
fi

echo "[1/4] Preparing HyperNeRF scene ${GROUP}/${SCENE}"
bash "${GS_ROOT}/scripts/prepare_local_hypernerf_scene.sh" "${SOURCE_ROOT}" "${GROUP}" "${SCENE}"

echo "[2/4] Running baseline + worldtube suite"
export HYPERNERF_SUITE_SUFFIX="${RUN_SUFFIX}"
export HYPERNERF_SUITE_PHASES="${HYPERNERF_SUITE_PHASES:-baseline,stellar_worldtube}"
export HYPERNERF_SUITE_COARSE_ITERS="${HYPERNERF_SUITE_COARSE_ITERS:-1000}"
export HYPERNERF_SUITE_TRAIN_ITERS="${HYPERNERF_SUITE_TRAIN_ITERS:-5000}"
export HYPERNERF_SUITE_TEST_ITERS="${HYPERNERF_SUITE_TEST_ITERS:-5000}"
export HYPERNERF_SUITE_SAVE_ITERS="${HYPERNERF_SUITE_SAVE_ITERS:-5000}"
bash "${GS_ROOT}/scripts/run_hypernerf_eval_suite.sh" "${GROUP}" "${SCENE}"

echo "[3/4] Writing comparison stub"
REPORT_DIR="${GS_ROOT}/reports/4dlangsplat_compare"
mkdir -p "${REPORT_DIR}"
REPORT_PATH="${REPORT_DIR}/${SCENE}_${RUN_SUFFIX}.md"
cat > "${REPORT_PATH}" <<EOF
# ReferGaussian vs 4DLangSplat Entry

- Scene: \`${SCENE}\`
- HyperNeRF group: \`${GROUP}\`
- Raw scene root: \`${SOURCE_ROOT}\`
- 4DLangSplat annotation: \`${ANNOT_SCENE_DIR}/video_annotations.json\`
- Baseline run: \`${GS_ROOT}/runs/baseline_${SCENE}_${RUN_SUFFIX}/hypernerf/${SCENE}\`
- Worldtube run: \`${GS_ROOT}/runs/stellar_worldtube_${SCENE}_${RUN_SUFFIX}/hypernerf/${SCENE}\`
- Benchmark table: \`${GS_ROOT}/reports/hypernerf_showcase/${GROUP}_${SCENE}_${RUN_SUFFIX}_benchmark.md\`

Next steps:

1. Map the scene's annotation queries to ReferGaussian query-guided planner inputs.
2. Run the query-guided worldtube pipeline on the annotation queries.
3. Compare query grounding/render outputs against 4DLangSplat annotations.
EOF

echo "[4/4] Done"
echo "Report: ${REPORT_PATH}"
