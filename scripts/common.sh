#!/usr/bin/env bash
set -euo pipefail

GS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GS_REPO_PARENT="$(cd "${GS_ROOT}/.." && pwd)"
GS_HOME_DEFAULT="${HOME:-/tmp}"
GS_CACHE_ROOT="${GS4D_CACHE_ROOT:-${GS_HOME_DEFAULT}/.cache/refergaussian}"
GS_ENV_ROOT="${GS4D_ENV_ROOT:-${GS_CACHE_ROOT}/conda-envs}"
GS_LEGACY_ENV_ROOT="${GS4D_LEGACY_ENV_ROOT:-${GS_REPO_PARENT}/.conda-envs}"
GS_CONDA_PKGS_DIRS="${GS4D_CONDA_PKGS_DIRS:-${GS_CACHE_ROOT}/conda-pkgs}"
GS_PIP_CACHE_DIR="${GS4D_PIP_CACHE_DIR:-${GS_CACHE_ROOT}/pip}"
GS_TORCH_HOME="${GS4D_TORCH_HOME:-${GS_CACHE_ROOT}/torch}"
GS_MPLCONFIGDIR="${GS4D_MPLCONFIGDIR:-${GS_CACHE_ROOT}/matplotlib}"
GSAM2_ENV_PATH="${GS4D_GSAM2_ENV_PATH:-${GS_ENV_ROOT}/grounded-sam2-py310}"

resolve_conda_bin() {
  if [[ -n "${GS4D_CONDA_BIN:-}" && -x "${GS4D_CONDA_BIN}" ]]; then
    printf '%s' "${GS4D_CONDA_BIN}"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return
  fi
  local candidate
  for candidate in \
    "${GS_HOME_DEFAULT}/miniconda3/bin/conda" \
    /root/miniconda3/bin/conda \
    /opt/conda/bin/conda \
    /usr/local/miniconda3/bin/conda; do
    if [[ -x "${candidate}" ]]; then
      printf '%s' "${candidate}"
      return
    fi
  done
}

GS_CONDA_BIN="$(resolve_conda_bin || true)"

require_conda_bin() {
  if [[ -z "${GS_CONDA_BIN}" ]]; then
    echo "conda was not found. Set GS4D_CONDA_BIN or add conda to PATH." >&2
    return 127
  fi
}

require_external_dependency() {
  local check_path="$1"
  local dep_name="$2"
  if [[ -e "${check_path}" ]]; then
    return 0
  fi
  cat >&2 <<MSG
Missing external dependency: ${dep_name}
Expected path: ${check_path}

Run:
  bash ${GS_ROOT}/scripts/bootstrap_external.sh
MSG
  return 2
}

require_4dgaussians() {
  require_external_dependency "${GS_ROOT}/external/4DGaussians/train.py" "4DGaussians"
}

require_grounded_sam2() {
  require_external_dependency "${GS_ROOT}/external/Grounded-SAM-2/sam2/__init__.py" "Grounded-SAM-2"
}

detect_default_env_path() {
  if [[ -n "${GS4D_ENV_PATH:-}" ]]; then
    printf '%s' "${GS4D_ENV_PATH}"
    return
  fi
  if [[ -d "${GS_ENV_ROOT}/gs4d-cuda121-py310" ]]; then
    printf '%s' "${GS_ENV_ROOT}/gs4d-cuda121-py310"
    return
  fi
  if [[ -d "${GS_ENV_ROOT}/gs4d-baseline-py37" ]]; then
    printf '%s' "${GS_ENV_ROOT}/gs4d-baseline-py37"
    return
  fi
  if [[ -d "${GS_LEGACY_ENV_ROOT}/gs4d-cuda121-py310" ]]; then
    printf '%s' "${GS_LEGACY_ENV_ROOT}/gs4d-cuda121-py310"
    return
  fi
  if [[ -d "${GS_LEGACY_ENV_ROOT}/gs4d-baseline-py37" ]]; then
    printf '%s' "${GS_LEGACY_ENV_ROOT}/gs4d-baseline-py37"
    return
  fi
  if [[ -d "${GS_HOME_DEFAULT}/miniconda3/envs/gs4d-cuda121-py310" ]]; then
    printf '%s' "${GS_HOME_DEFAULT}/miniconda3/envs/gs4d-cuda121-py310"
    return
  fi
  if [[ -d "${GS_HOME_DEFAULT}/miniconda3/envs/gs4d-baseline-py37" ]]; then
    printf '%s' "${GS_HOME_DEFAULT}/miniconda3/envs/gs4d-baseline-py37"
    return
  fi
  printf '%s' "${GS_ENV_ROOT}/gs4d-cuda121-py310"
}

mkdir -p "${GS_ENV_ROOT}" "${GS_CONDA_PKGS_DIRS}" "${GS_PIP_CACHE_DIR}" "${GS_TORCH_HOME}" "${GS_MPLCONFIGDIR}"
GS_ENV_PATH="$(detect_default_env_path)"
GS_ENV_NAME="${GS4D_ENV_NAME:-$(basename "${GS_ENV_PATH}")}"
export GS_ROOT
export GS_ENV_ROOT
export GS_LEGACY_ENV_ROOT
export GS_CACHE_ROOT
export GS_ENV_PATH
export GS_ENV_NAME
export GS_CONDA_BIN
export GS_CONDA_PKGS_DIRS
export GS_PIP_CACHE_DIR
export GS_TORCH_HOME
export GS_MPLCONFIGDIR
export GSAM2_ENV_PATH
export XDG_CACHE_HOME="${GS_CACHE_ROOT}"
export TORCH_HOME="${GS_TORCH_HOME}"
export MPLCONFIGDIR="${GS_MPLCONFIGDIR}"
export PYTHONPATH="${GS_ROOT}:${GS_ROOT}/external/4DGaussians:${PYTHONPATH:-}"
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi

gs_python() {
  if [[ "${CONDA_PREFIX:-}" == "${GS_ENV_PATH}" ]]; then
    python "$@"
  else
    require_conda_bin
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XDG_CACHE_HOME="${GS_CACHE_ROOT}" TORCH_HOME="${GS_TORCH_HOME}" MPLCONFIGDIR="${GS_MPLCONFIGDIR}" CONDA_PKGS_DIRS="${GS_CONDA_PKGS_DIRS}" PIP_CACHE_DIR="${GS_PIP_CACHE_DIR}" \
      "${GS_CONDA_BIN}" run --no-capture-output -p "${GS_ENV_PATH}" python "$@"
  fi
}

gs_pip() {
  if [[ "${CONDA_PREFIX:-}" == "${GS_ENV_PATH}" ]]; then
    python -m pip "$@"
  else
    require_conda_bin
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XDG_CACHE_HOME="${GS_CACHE_ROOT}" TORCH_HOME="${GS_TORCH_HOME}" MPLCONFIGDIR="${GS_MPLCONFIGDIR}" CONDA_PKGS_DIRS="${GS_CONDA_PKGS_DIRS}" PIP_CACHE_DIR="${GS_PIP_CACHE_DIR}" \
      "${GS_CONDA_BIN}" run --no-capture-output -p "${GS_ENV_PATH}" python -m pip "$@"
  fi
}

gsam2_python() {
  if [[ "${CONDA_PREFIX:-}" == "${GSAM2_ENV_PATH}" ]]; then
    python "$@"
  else
    require_conda_bin
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XDG_CACHE_HOME="${GS_CACHE_ROOT}" TORCH_HOME="${GS_TORCH_HOME}" MPLCONFIGDIR="${GS_MPLCONFIGDIR}" CONDA_PKGS_DIRS="${GS_CONDA_PKGS_DIRS}" PIP_CACHE_DIR="${GS_PIP_CACHE_DIR}" \
      "${GS_CONDA_BIN}" run --no-capture-output -p "${GSAM2_ENV_PATH}" python "$@"
  fi
}

gsam2_pip() {
  if [[ "${CONDA_PREFIX:-}" == "${GSAM2_ENV_PATH}" ]]; then
    python -m pip "$@"
  else
    require_conda_bin
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XDG_CACHE_HOME="${GS_CACHE_ROOT}" TORCH_HOME="${GS_TORCH_HOME}" MPLCONFIGDIR="${GS_MPLCONFIGDIR}" CONDA_PKGS_DIRS="${GS_CONDA_PKGS_DIRS}" PIP_CACHE_DIR="${GS_PIP_CACHE_DIR}" \
      "${GS_CONDA_BIN}" run --no-capture-output -p "${GSAM2_ENV_PATH}" python -m pip "$@"
  fi
}

gs_python_cmd() {
  if [[ "${CONDA_PREFIX:-}" == "${GS_ENV_PATH}" ]]; then
    printf 'python'
  else
    require_conda_bin >/dev/null
    printf 'env OMP_NUM_THREADS=%q MKL_NUM_THREADS=%q XDG_CACHE_HOME=%q TORCH_HOME=%q MPLCONFIGDIR=%q CONDA_PKGS_DIRS=%q PIP_CACHE_DIR=%q %q run --no-capture-output -p %q python' "1" "1" "${GS_CACHE_ROOT}" "${GS_TORCH_HOME}" "${GS_MPLCONFIGDIR}" "${GS_CONDA_PKGS_DIRS}" "${GS_PIP_CACHE_DIR}" "${GS_CONDA_BIN}" "${GS_ENV_PATH}"
  fi
}

shell_join() {
  local quoted=""
  local arg
  for arg in "$@"; do
    printf -v quoted '%s%q ' "${quoted}" "${arg}"
  done
  printf '%s' "${quoted% }"
}

dataset_source_path() {
  local dataset="$1"
  local scene="$2"
  case "${dataset}" in
    dnerf)
      printf '%s/data/dnerf/%s' "${GS_ROOT}" "${scene}"
      ;;
    dynerf)
      printf '%s/data/dynerf/%s' "${GS_ROOT}" "${scene}"
      ;;
    hypernerf)
      if [[ "${scene}" == */* ]]; then
        printf '%s/data/hypernerf/%s' "${GS_ROOT}" "${scene}"
      else
        printf '%s/data/hypernerf/virg/%s' "${GS_ROOT}" "${scene}"
      fi
      ;;
    *)
      echo "Unsupported dataset: ${dataset}" >&2
      return 1
      ;;
  esac
}

dataset_config_path() {
  local dataset="$1"
  local scene="$2"
  local config_scene="${scene##*/}"
  case "${dataset}" in
    dnerf)
      printf '%s/external/4DGaussians/arguments/dnerf/%s.py' "${GS_ROOT}" "${config_scene}"
      ;;
    dynerf)
      local candidate="${GS_ROOT}/external/4DGaussians/arguments/dynerf/${config_scene}.py"
      if [[ -f "${candidate}" ]]; then
        printf '%s' "${candidate}"
      else
        printf '%s/external/4DGaussians/arguments/dynerf/default.py' "${GS_ROOT}"
      fi
      ;;
    hypernerf)
      case "${config_scene}" in
        slice-banana)
          config_scene="banana"
          ;;
        chickchicken)
          config_scene="chicken"
          ;;
      esac
      local candidate="${GS_ROOT}/external/4DGaussians/arguments/hypernerf/${config_scene}.py"
      if [[ -f "${candidate}" ]]; then
        printf '%s' "${candidate}"
      else
        printf '%s/external/4DGaussians/arguments/hypernerf/default.py' "${GS_ROOT}"
      fi
      ;;
    *)
      echo "Unsupported dataset: ${dataset}" >&2
      return 1
      ;;
  esac
}

run_with_gpu_monitor() {
  local log_path="$1"
  local meta_path="$2"
  shift 2

  mkdir -p "$(dirname "${log_path}")"
  mkdir -p "$(dirname "${meta_path}")"
  : > "${log_path}"

  local start_ts
  start_ts="$(date +%s)"

  ("$@" > >(tee -a "${log_path}") 2> >(tee -a "${log_path}" >&2)) &
  local cmd_pid=$!
  local peak_file
  peak_file="$(mktemp)"
  printf '0' > "${peak_file}"

  (
    while kill -0 "${cmd_pid}" 2>/dev/null; do
      local used
      used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1{print int($1)}')"
      local current_peak
      current_peak="$(cat "${peak_file}")"
      if [[ -n "${used}" && "${used}" =~ ^[0-9]+$ && "${used}" -gt "${current_peak}" ]]; then
        printf '%s' "${used}" > "${peak_file}"
      fi
      sleep 1
    done
  ) &
  local monitor_pid=$!

  local status=0
  wait "${cmd_pid}" || status=$?
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true

  local end_ts
  end_ts="$(date +%s)"
  local peak_mb
  peak_mb="$(cat "${peak_file}")"
  rm -f "${peak_file}"
  cat > "${meta_path}" <<META
{
  "status": ${status},
  "elapsed_seconds": $((end_ts - start_ts)),
  "gpu_peak_mb": ${peak_mb}
}
META
  return "${status}"
}
