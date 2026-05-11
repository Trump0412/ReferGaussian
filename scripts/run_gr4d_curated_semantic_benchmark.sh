#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BENCHMARK_ROOT="${1:-${GS_ROOT}/data/benchmarks/gr4d_curated_v1}"
RESULT_ROOT="${2:-${GS_EXPERIMENT_ROOT:-${GS_ROOT}/experiments}/refergaussian_semantic_audit_$(date +%Y%m%d_%H%M%S)}"
EXTRA_QUERY_PACK="${3:-${GS_ROOT}/configs/benchmarks/gr4d_semantic_stress_queries.json}"

mkdir -p "${RESULT_ROOT}/queries" "${RESULT_ROOT}/logs" "${RESULT_ROOT}/reports"

resolve_run_dir() {
  local scene="$1"
  case "${scene}" in
    americano)
      printf '%s' "${GS_ROOT}/runs/stellar_worldtube_americano_compare5k/hypernerf/americano"
      ;;
    cut-lemon1)
      printf '%s' "${GS_ROOT}/runs/stellar_worldtube_cut_lemon1_smoke300_v6a/hypernerf/cut-lemon1"
      ;;
    split-cookie)
      printf '%s' "${GS_ROOT}/runs/stellar_tube_split-cookie_smoke300_tubecmp/hypernerf/split-cookie"
      ;;
    coffee_martini)
      printf '%s' "${GS_ROOT}/runs/stellar_tube_coffee_martini_smoke300_tubecmp/dynerf/coffee_martini"
      ;;
    flame_steak)
      printf '%s' "${GS_ROOT}/runs/stellar_tube_flame_steak_smoke300_tubecmp/dynerf/flame_steak"
      ;;
    *)
      echo "Unsupported scene: ${scene}" >&2
      return 1
      ;;
  esac
}

resolve_source_path() {
  python - <<'PY' "$1"
from pathlib import Path
import sys

config_path = Path(sys.argv[1]) / "config.yaml"
source_path = ""
for raw_line in config_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if line.startswith("source_path:"):
        source_path = line.split(":", 1)[1].strip()
        break
if not source_path:
    raise SystemExit(f"Unable to resolve source_path from {config_path}")
print(source_path)
PY
}

python - <<'PY' "${BENCHMARK_ROOT}" "${EXTRA_QUERY_PACK}" "${RESULT_ROOT}/query_jobs.tsv"
import json
import sys
from pathlib import Path

benchmark_root = Path(sys.argv[1])
extra_pack = Path(sys.argv[2])
output_path = Path(sys.argv[3])

rows = []
payload = json.loads((benchmark_root / "gr4d_curated_v1_queries.json").read_text(encoding="utf-8"))
for item in payload.get("queries", []):
    rows.append(
        {
            "scene": str(item["scene"]),
            "query_id": str(item["query_id"]),
            "text_en": str(item["text_en"]),
            "pack": "curated",
        }
    )

if extra_pack.exists():
    extra_payload = json.loads(extra_pack.read_text(encoding="utf-8"))
    for item in extra_payload.get("queries", []):
        rows.append(
            {
                "scene": str(item["scene"]),
                "query_id": str(item["query_id"]),
                "text_en": str(item["text_en"]),
                "pack": "stress",
            }
        )

output_path.write_text(
    "\n".join(
        "\t".join((row["scene"], row["query_id"], row["pack"], row["text_en"]))
        for row in rows
    )
    + "\n",
    encoding="utf-8",
)
PY

python - <<'PY' "${RESULT_ROOT}/run_map.json"
import json
import sys
from pathlib import Path

payload = {
    "americano": "${GS_ROOT}/runs/stellar_worldtube_americano_compare5k/hypernerf/americano",
    "cut-lemon1": "${GS_ROOT}/runs/stellar_worldtube_cut_lemon1_smoke300_v6a/hypernerf/cut-lemon1",
    "split-cookie": "${GS_ROOT}/runs/stellar_tube_split-cookie_smoke300_tubecmp/hypernerf/split-cookie",
    "coffee_martini": "${GS_ROOT}/runs/stellar_tube_coffee_martini_smoke300_tubecmp/dynerf/coffee_martini",
    "flame_steak": "${GS_ROOT}/runs/stellar_tube_flame_steak_smoke300_tubecmp/dynerf/flame_steak",
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

cp "${BENCHMARK_ROOT}/gr4d_curated_v1_queries.json" "${RESULT_ROOT}/curated_queries_snapshot.json"
if [[ -f "${EXTRA_QUERY_PACK}" ]]; then
  cp "${EXTRA_QUERY_PACK}" "${RESULT_ROOT}/stress_queries_snapshot.json"
fi

while IFS=$'\t' read -r scene query_id pack query_text; do
  [[ -z "${scene}" ]] && continue
  run_dir="$(resolve_run_dir "${scene}")"
  dataset_dir="$(resolve_source_path "${run_dir}")"
  query_root="${RESULT_ROOT}/queries/${scene}/${query_id}"
  query_plan_path="${query_root}/query_plan.json"
  track_dir="${query_root}/grounded_sam2"
  tracks_path="${track_dir}/grounded_sam2_query_tracks.json"
  proposal_dir="${query_root}/proposal_dir"
  query_entitybank_dir="${query_root}/query_entitybank"
  query_run_dir="${query_root}/query_run_dir"
  qwen_selection_path="${query_root}/selected_query_qwen.json"
  log_path="${RESULT_ROOT}/logs/${scene}__${query_id}.log"

  mkdir -p "${query_root}"
  printf '[semantic-benchmark] %s / %s (%s)\n' "${scene}" "${query_id}" "${pack}"

  if [[ -f "${qwen_selection_path}" ]]; then
    printf '  cached result found at %s\n' "${qwen_selection_path}"
    continue
  fi

  if (
    set -euo pipefail

    gsam2_python "${GS_ROOT}/scripts/plan_query_entities.py" \
      --query "${query_text}" \
      --dataset-dir "${dataset_dir}" \
      --output-path "${query_plan_path}" \
      --frame-subsample-stride "${GSAM2_FRAME_SUBSAMPLE_STRIDE:-12}" \
      --num-sampled-frames "${GSAM2_NUM_CONTEXT_FRAMES:-7}" \
      --num-boundary-frames "${GSAM2_NUM_BOUNDARY_FRAMES:-11}" \
      --strict

    gsam2_python "${GS_ROOT}/scripts/run_grounded_sam2_query.py" \
      --dataset-dir "${dataset_dir}" \
      --query-plan-path "${query_plan_path}" \
      --output-dir "${track_dir}" \
      --grounding-model-id "${GSAM2_GROUNDING_MODEL_ID:-IDEA-Research/grounding-dino-base}" \
      --sam2-model-id "${GSAM2_SAM2_MODEL_ID:-facebook/sam2-hiera-large}" \
      --prompt-type "${GSAM2_PROMPT_TYPE:-point}" \
      --detector-frame-stride "${GSAM2_DETECTOR_FRAME_STRIDE:-6}" \
      --max-detector-frames "${GSAM2_MAX_DETECTOR_FRAMES:-48}" \
      --detection-top-k "${GSAM2_DETECTION_TOP_K:-5}" \
      --box-threshold "${GSAM2_BOX_THRESHOLD:-0.25}" \
      --text-threshold "${GSAM2_TEXT_THRESHOLD:-0.20}" \
      --num-point-prompts "${GSAM2_NUM_POINT_PROMPTS:-16}" \
      --track-window-radius "${GSAM2_TRACK_WINDOW_RADIUS:-160}" \
      --frame-subsample-stride "${GSAM2_FRAME_SUBSAMPLE_STRIDE:-10}" \
      --num-anchor-seeds "${GSAM2_NUM_ANCHOR_SEEDS:-3}"

    gs_python "${GS_ROOT}/scripts/build_query_proposal_dir.py" \
      --run-dir "${run_dir}" \
      --dataset-dir "${dataset_dir}" \
      --tracks-path "${tracks_path}" \
      --output-dir "${proposal_dir}" \
      --max-track-frames "${QUERY_MAX_TRACK_FRAMES:-16}" \
      --proposal-keep-ratio "${QUERY_PROPOSAL_KEEP_RATIO:-0.03}" \
      --min-gaussians "${QUERY_MIN_GAUSSIANS:-256}" \
      --max-gaussians "${QUERY_MAX_GAUSSIANS:-4096}"

    gs_python "${GS_ROOT}/scripts/export_entitybank.py" \
      --run-dir "${run_dir}" \
      --proposal-dir "${proposal_dir}" \
      --proposal-strict \
      --output-dir "${query_entitybank_dir}" \
      --max-entities "${QUERY_MAX_ENTITIES:-12}" \
      --min-gaussians-per-entity "${QUERY_MIN_GAUSSIANS_PER_ENTITY:-32}"

    mkdir -p "${query_run_dir}"
    ln -sfn "${run_dir}/config.yaml" "${query_run_dir}/config.yaml"
    ln -sfn "${run_dir}/point_cloud" "${query_run_dir}/point_cloud"
    ln -sfn "${run_dir}/test" "${query_run_dir}/test"
    ln -sfn "${query_entitybank_dir}" "${query_run_dir}/entitybank"

    gs_python "${GS_ROOT}/scripts/export_semantic_slots.py" --run-dir "${query_run_dir}"
    gs_python "${GS_ROOT}/scripts/export_semantic_tracks.py" --run-dir "${query_run_dir}"
    gs_python "${GS_ROOT}/scripts/export_semantic_priors.py" --run-dir "${query_run_dir}"
    gs_python "${GS_ROOT}/scripts/export_native_semantics.py" --run-dir "${query_run_dir}"

    gsam2_python "${GS_ROOT}/scripts/export_qwen_semantics.py" \
      --run-dir "${query_run_dir}" \
      --query "${query_text}" \
      --max-entities "${QUERY_QWEN_MAX_ENTITIES:-12}"

    gsam2_python "${GS_ROOT}/scripts/select_qwen_query_entities.py" \
      --assignments-path "${query_run_dir}/entitybank/semantic_assignments_qwen.json" \
      --query "${query_text}" \
      --query-plan-path "${query_plan_path}" \
      --output-path "${qwen_selection_path}"
  ) >"${log_path}" 2>&1; then
    python - <<'PY' "${query_root}/status.json" "${scene}" "${query_id}" "${pack}" "${run_dir}" "${dataset_dir}" "${query_text}" "${log_path}"
import json
import sys
from pathlib import Path

payload = {
    "status": "ok",
    "scene": sys.argv[2],
    "query_id": sys.argv[3],
    "pack": sys.argv[4],
    "run_dir": sys.argv[5],
    "dataset_dir": sys.argv[6],
    "query": sys.argv[7],
    "log_path": sys.argv[8],
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
PY
  else
    python - <<'PY' "${query_root}/status.json" "${scene}" "${query_id}" "${pack}" "${run_dir}" "${dataset_dir}" "${query_text}" "${log_path}"
import json
import sys
from pathlib import Path

payload = {
    "status": "error",
    "scene": sys.argv[2],
    "query_id": sys.argv[3],
    "pack": sys.argv[4],
    "run_dir": sys.argv[5],
    "dataset_dir": sys.argv[6],
    "query": sys.argv[7],
    "log_path": sys.argv[8],
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
PY
    printf '  query failed, see %s\n' "${log_path}" >&2
  fi
done < "${RESULT_ROOT}/query_jobs.tsv"

python "${GS_ROOT}/scripts/build_gr4d_semantic_report.py" \
  --benchmark-root "${BENCHMARK_ROOT}" \
  --results-root "${RESULT_ROOT}" \
  --extra-query-pack "${EXTRA_QUERY_PACK}" \
  --output-json "${RESULT_ROOT}/reports/semantic_benchmark_summary.json" \
  --output-md "${RESULT_ROOT}/reports/semantic_benchmark_summary.md"

printf '%s\n' "${RESULT_ROOT}"
