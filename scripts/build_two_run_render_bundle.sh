#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

BASELINE_RUN="${1:?baseline run dir required}"
OURS_RUN="${2:?ours run dir required}"
OUT_DIR="${3:?output dir required}"
TITLE="${4:-Render Comparison}"
BASELINE_LABEL="${5:-4DGS}"
OURS_LABEL="${6:-ReferGaussian}"
FRAME_A="${7:-00000.png}"
FRAME_B="${8:-00150.png}"
FRAME_C="${9:-00299.png}"

mkdir -p "${OUT_DIR}"

gs_python "${GS_ROOT}/scripts/fullframe_metrics.py" \
  --run-dir "${BASELINE_RUN}" \
  --with-lpips \
  --out-name "full_metrics_with_lpips.json"

gs_python "${GS_ROOT}/scripts/fullframe_metrics.py" \
  --run-dir "${OURS_RUN}" \
  --with-lpips \
  --out-name "full_metrics_with_lpips.json"

gs_python "${GS_ROOT}/scripts/export_comparison_gif.py" \
  --title "${TITLE}" \
  --frame-step 4 \
  --max-frames 40 \
  --entry "${BASELINE_LABEL}=${BASELINE_RUN}" \
  --entry "${OURS_LABEL}=${OURS_RUN}" \
  --output "${OUT_DIR}/render_compare.gif"

gs_python "${GS_ROOT}/scripts/export_keyframe_triptych.py" \
  --baseline-run "${BASELINE_RUN}" \
  --ours-run "${OURS_RUN}" \
  --baseline-label "${BASELINE_LABEL}" \
  --ours-label "${OURS_LABEL}" \
  --title "${TITLE} Keyframes" \
  --frame-name "${FRAME_A}" \
  --frame-name "${FRAME_B}" \
  --frame-name "${FRAME_C}" \
  --output "${OUT_DIR}/render_keyframes_poster.png"

gs_python "${GS_ROOT}/scripts/export_keyframe_triptych.py" \
  --baseline-run "${BASELINE_RUN}" \
  --ours-run "${OURS_RUN}" \
  --baseline-label "${BASELINE_LABEL}" \
  --ours-label "${OURS_LABEL}" \
  --title "${TITLE} ${FRAME_A}" \
  --frame-name "${FRAME_A}" \
  --output "${OUT_DIR}/render_triptych_${FRAME_A}"

gs_python "${GS_ROOT}/scripts/export_keyframe_triptych.py" \
  --baseline-run "${BASELINE_RUN}" \
  --ours-run "${OURS_RUN}" \
  --baseline-label "${BASELINE_LABEL}" \
  --ours-label "${OURS_LABEL}" \
  --title "${TITLE} ${FRAME_B}" \
  --frame-name "${FRAME_B}" \
  --output "${OUT_DIR}/render_triptych_${FRAME_B}"

gs_python "${GS_ROOT}/scripts/export_keyframe_triptych.py" \
  --baseline-run "${BASELINE_RUN}" \
  --ours-run "${OURS_RUN}" \
  --baseline-label "${BASELINE_LABEL}" \
  --ours-label "${OURS_LABEL}" \
  --title "${TITLE} ${FRAME_C}" \
  --frame-name "${FRAME_C}" \
  --output "${OUT_DIR}/render_triptych_${FRAME_C}"

gs_python - "${BASELINE_RUN}" "${OURS_RUN}" "${OUT_DIR}" "${BASELINE_LABEL}" "${OURS_LABEL}" "${FRAME_A}" "${FRAME_B}" "${FRAME_C}" "${TITLE}" <<'PY'
import json
import sys
from pathlib import Path

baseline_run = Path(sys.argv[1])
ours_run = Path(sys.argv[2])
out_dir = Path(sys.argv[3])
baseline_label = sys.argv[4]
ours_label = sys.argv[5]
frame_a = sys.argv[6]
frame_b = sys.argv[7]
frame_c = sys.argv[8]
title = sys.argv[9]

with open(baseline_run / "full_metrics_with_lpips.json", "r", encoding="utf-8") as handle:
    baseline = json.load(handle)
with open(ours_run / "full_metrics_with_lpips.json", "r", encoding="utf-8") as handle:
    ours = json.load(handle)

def delta(key):
    left = baseline.get(key)
    right = ours.get(key)
    if left is None or right is None:
        return None
    return right - left

def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)

report_lines = [
    f"# {title} Metrics",
    "",
    "Full-frame test-set metrics with LPIPS for both runs.",
    "",
    "| Method | PSNR | SSIM | MS-SSIM | LPIPS-vgg | Frames |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
    f"| {baseline_label} | {fmt(baseline.get('PSNR'))} | {fmt(baseline.get('SSIM'))} | {fmt(baseline.get('MS-SSIM'))} | {fmt(baseline.get('LPIPS-vgg'))} | {fmt(baseline.get('sample_count'))} |",
    f"| {ours_label} | {fmt(ours.get('PSNR'))} | {fmt(ours.get('SSIM'))} | {fmt(ours.get('MS-SSIM'))} | {fmt(ours.get('LPIPS-vgg'))} | {fmt(ours.get('sample_count'))} |",
    f"| delta ({ours_label} - {baseline_label}) | {fmt(delta('PSNR'))} | {fmt(delta('SSIM'))} | {fmt(delta('MS-SSIM'))} | {fmt(delta('LPIPS-vgg'))} | - |",
    "",
    "## Run Paths",
    "",
    f"- `{baseline_label}`: `{baseline_run}`",
    f"- `{ours_label}`: `{ours_run}`",
    "",
]
with open(out_dir / "metrics_report.md", "w", encoding="utf-8") as handle:
    handle.write("\n".join(report_lines))

payload = {
    "baseline_label": baseline_label,
    "ours_label": ours_label,
    "baseline_run": str(baseline_run),
    "ours_run": str(ours_run),
    "baseline_metrics": baseline,
    "ours_metrics": ours,
    "delta": {
        "PSNR": delta("PSNR"),
        "SSIM": delta("SSIM"),
        "MS-SSIM": delta("MS-SSIM"),
        "LPIPS-vgg": delta("LPIPS-vgg"),
    },
    "artifacts": {
        "gif": str(out_dir / "render_compare.gif"),
        "poster": str(out_dir / "render_keyframes_poster.png"),
        "triptych_a": str(out_dir / f"render_triptych_{Path(frame_a).name}"),
        "triptych_b": str(out_dir / f"render_triptych_{Path(frame_b).name}"),
        "triptych_c": str(out_dir / f"render_triptych_{Path(frame_c).name}"),
        "metrics_report": str(out_dir / "metrics_report.md"),
    },
}

with open(out_dir / "comparison_summary.json", "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
print(out_dir / "comparison_summary.json")
PY

echo "Wrote render bundle to ${OUT_DIR}"
