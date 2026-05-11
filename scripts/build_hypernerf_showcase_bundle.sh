#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

GROUP="${1:-interp}"
SCENE="${2:-cut-lemon1}"
RUN_SUFFIX="${3:-smoke300}"
OUTPUT_ROOT="${GS_ROOT}/reports/hypernerf_showcase"
SCENE_SLUG="${SCENE//-/_}"

mkdir -p "${OUTPUT_ROOT}"

resolve_run_dir() {
  local prefix="$1"
  local override="${2:-}"
  local candidate
  if [[ -n "${override}" ]]; then
    candidate="${GS_ROOT}/runs/${override}/hypernerf/${SCENE}"
    [[ -f "${candidate}/metrics.json" ]] && printf '%s' "${candidate}" && return
  fi
  for namespace in \
    "${prefix}_${SCENE}_${RUN_SUFFIX}" \
    "${prefix}_${SCENE_SLUG}_${RUN_SUFFIX}"
  do
    candidate="${GS_ROOT}/runs/${namespace}/hypernerf/${SCENE}"
    if [[ -f "${candidate}/metrics.json" ]]; then
      printf '%s' "${candidate}"
      return
    fi
  done
  return 1
}

BASELINE_RUN="$(resolve_run_dir baseline "${HYPERNERF_BASELINE_NAMESPACE:-}")" || {
  echo "Missing baseline run for ${GROUP}/${SCENE} ${RUN_SUFFIX}" >&2
  exit 2
}
WORLDTUBE_RUN="$(resolve_run_dir stellar_worldtube "${HYPERNERF_WORLDTUBE_NAMESPACE:-}")" || {
  echo "Missing worldtube run for ${GROUP}/${SCENE} ${RUN_SUFFIX}" >&2
  exit 2
}

gs_python "${GS_ROOT}/scripts/build_benchmark_report.py" \
  --title "HyperNeRF ${GROUP}/${SCENE} ${RUN_SUFFIX}" \
  --subtitle "${HYPERNERF_SHOWCASE_SUBTITLE:-Baseline vs ReferGaussian worldtube comparison.}" \
  --entry "baseline=${BASELINE_RUN}" \
  --entry "stellar_worldtube=${WORLDTUBE_RUN}" \
  --output "${OUTPUT_ROOT}/${GROUP}_${SCENE}_${RUN_SUFFIX}_benchmark.md"

gs_python "${GS_ROOT}/scripts/export_comparison_frames.py" \
  --title "${GROUP}/${SCENE} frame 00010" \
  --frame-name 00010.png \
  --columns 2 \
  --entry "baseline=${BASELINE_RUN}" \
  --entry "stellar_worldtube=${WORLDTUBE_RUN}" \
  --output "${OUTPUT_ROOT}/${GROUP}_${SCENE}_${RUN_SUFFIX}_frame_00010.png"

gs_python "${GS_ROOT}/scripts/export_comparison_gif.py" \
  --title "${GROUP}/${SCENE}" \
  --frame-step 4 \
  --max-frames 24 \
  --entry "baseline=${BASELINE_RUN}" \
  --entry "stellar_worldtube=${WORLDTUBE_RUN}" \
  --output "${OUTPUT_ROOT}/${GROUP}_${SCENE}_${RUN_SUFFIX}_compare.gif"

cat > "${OUTPUT_ROOT}/${GROUP}_${SCENE}_${RUN_SUFFIX}_assets.md" <<EOF
# HyperNeRF ${GROUP}/${SCENE} ${RUN_SUFFIX} Assets

- Benchmark: \`${OUTPUT_ROOT}/${GROUP}_${SCENE}_${RUN_SUFFIX}_benchmark.md\`
- Key frame: \`${OUTPUT_ROOT}/${GROUP}_${SCENE}_${RUN_SUFFIX}_frame_00010.png\`
- GIF: \`${OUTPUT_ROOT}/${GROUP}_${SCENE}_${RUN_SUFFIX}_compare.gif\`
- Baseline run: \`${BASELINE_RUN}\`
- Worldtube run: \`${WORLDTUBE_RUN}\`
EOF

echo "Wrote HyperNeRF showcase assets to ${OUTPUT_ROOT}"
