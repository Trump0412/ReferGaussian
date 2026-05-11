#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-${GS4D_ENV_PROFILE:-cuda121}}"
CACHE_ROOT="${GS4D_CACHE_ROOT:-${HOME:-/tmp}/.cache/refergaussian}"
ENV_ROOT="${GS4D_ENV_ROOT:-${CACHE_ROOT}/conda-envs}"
CONDA_PKGS_DIRS="${GS4D_CONDA_PKGS_DIRS:-${CACHE_ROOT}/conda-pkgs}"
PIP_CACHE_DIR="${GS4D_PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"

require_external_4dgaussians() {
  if [[ -f "${ROOT_DIR}/external/4DGaussians/train.py" ]]; then
    return 0
  fi
  cat >&2 <<MSG
Missing dependency: external/4DGaussians
Run:
  bash ${ROOT_DIR}/scripts/bootstrap_external.sh
MSG
  exit 2
}

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
  for candidate in "${HOME:-/tmp}/miniconda3/bin/conda" /root/miniconda3/bin/conda /opt/conda/bin/conda /usr/local/miniconda3/bin/conda; do
    if [[ -x "${candidate}" ]]; then
      printf '%s' "${candidate}"
      return
    fi
  done
}

CONDA_BIN="$(resolve_conda_bin || true)"
if [[ -z "${CONDA_BIN}" ]]; then
  echo "conda is required but was not found. Set GS4D_CONDA_BIN or add conda to PATH." >&2
  exit 1
fi

mkdir -p "${ENV_ROOT}" "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}"

create_env_if_missing() {
  local env_prefix="$1"
  local python_version="$2"
  if [[ ! -d "${env_prefix}" ]]; then
    env CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS}" PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
      "${CONDA_BIN}" create -y -p "${env_prefix}" "python=${python_version}"
  fi
}

pip_run() {
  local env_prefix="$1"
  shift
  env CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS}" PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
    "${CONDA_BIN}" run --no-capture-output -p "${env_prefix}" python -m pip "$@"
}

conda_install() {
  local env_prefix="$1"
  shift
  env CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS}" PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
    "${CONDA_BIN}" install -y -p "${env_prefix}" "$@"
}

install_local_package() {
  local env_prefix="$1"
  local package_dir="$2"
  env CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS}" PIP_CACHE_DIR="${PIP_CACHE_DIR}" CUDA_HOME="${CUDA_HOME}" PATH="${PATH}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MAX_JOBS="${MAX_JOBS:-4}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}" \
    "${CONDA_BIN}" run --no-capture-output -p "${env_prefix}" env -C "${package_dir}" python setup.py install
}

ensure_cuda_headers() {
  local cuda_home="$1"
  if [[ ! -f "${cuda_home}/include/cuda_runtime.h" ]]; then
    echo "Missing ${cuda_home}/include/cuda_runtime.h" >&2
    echo "Provide a full CUDA toolkit via GS4D_CUDA_HOME before compiling extensions." >&2
    exit 2
  fi
}

install_official_profile() {
  require_external_4dgaussians

  local env_name="${GS4D_ENV_NAME:-gs4d-baseline-py37}"
  local env_prefix
  env_prefix="${ENV_ROOT}/${env_name}"
  create_env_if_missing "${env_prefix}" "3.7"
  conda_install "${env_prefix}" -c nvidia cuda-nvcc=11.7
  conda_install "${env_prefix}" ninja

  export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/external/4DGaussians:${PYTHONPATH:-}"
  export CUDA_HOME="${GS4D_CUDA_HOME:-${env_prefix}}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
  export OMP_NUM_THREADS=1
  export MAX_JOBS="${GS4D_MAX_JOBS:-4}"
  ensure_cuda_headers "${CUDA_HOME}"

  pip_run "${env_prefix}" install --upgrade pip "setuptools<81" wheel
  pip_run "${env_prefix}" install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
  pip_run "${env_prefix}" install -r "${ROOT_DIR}/external/4DGaussians/requirements.txt"
  pip_run "${env_prefix}" install matplotlib pandas pyyaml requests
  install_local_package "${env_prefix}" "${ROOT_DIR}/external/4DGaussians/submodules/depth-diff-gaussian-rasterization"
  install_local_package "${env_prefix}" "${ROOT_DIR}/external/4DGaussians/submodules/simple-knn"
  echo "Environment '${env_prefix}' is ready with the official 11.7 profile."
}

install_cuda121_profile() {
  require_external_4dgaussians

  local env_name="${GS4D_ENV_NAME:-gs4d-cuda121-py310}"
  local env_prefix
  env_prefix="${ENV_ROOT}/${env_name}"
  create_env_if_missing "${env_prefix}" "3.10"

  export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/external/4DGaussians:${PYTHONPATH:-}"
  export CUDA_HOME="${GS4D_CUDA_HOME:-/usr/local/cuda-12.1}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
  export OMP_NUM_THREADS=1
  export MAX_JOBS="${GS4D_MAX_JOBS:-4}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
  ensure_cuda_headers "${CUDA_HOME}"

  pip_run "${env_prefix}" install --upgrade pip "setuptools<81" wheel ninja "numpy<2"
  pip_run "${env_prefix}" install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
  pip_run "${env_prefix}" install "numpy<2" matplotlib pandas pyyaml requests tqdm scipy "opencv-python<4.10" plyfile lpips pytorch_msssim "imageio[ffmpeg]"
  install_local_package "${env_prefix}" "${ROOT_DIR}/external/4DGaussians/submodules/simple-knn"
  install_local_package "${env_prefix}" "${ROOT_DIR}/external/4DGaussians/submodules/depth-diff-gaussian-rasterization"
  echo "Environment '${env_prefix}' is ready with the CUDA 12.1 profile."
}

case "${PROFILE}" in
  official)
    install_official_profile
    ;;
  cuda121)
    install_cuda121_profile
    ;;
  *)
    echo "Unsupported profile: ${PROFILE}" >&2
    echo "Use one of: official, cuda121" >&2
    exit 1
    ;;
esac
