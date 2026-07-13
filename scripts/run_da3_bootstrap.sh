#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

INPUT_PATH="${1:?input path required}"
OUTPUT_DIR="${2:-${GS_DATA_ROOT}/downloads/da3_bootstrap}"
shift $(( $# > 1 ? 2 : $# ))
EXTRA_ARGS="$(shell_join "$@")"

DA3_ENV_PATH="${DA3_ENV_PATH:-${GS_ENV_ROOT}/da3-gs-py310}"
DA3_HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
DA3_HF_HOME="${HF_HOME:-${GS_CACHE_ROOT}/huggingface}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY
export HF_ENDPOINT="${DA3_HF_ENDPOINT}"
export HF_HOME="${DA3_HF_HOME}"
export XDG_CACHE_HOME="${GS_CACHE_ROOT}"
export TORCH_HOME="${GS_TORCH_HOME}"
export MPLCONFIGDIR="${GS_MPLCONFIGDIR}"
export CONDA_PKGS_DIRS="${GS_CONDA_PKGS_DIRS}"
export PIP_CACHE_DIR="${GS_PIP_CACHE_DIR}"

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"

conda run --no-capture-output -p "${DA3_ENV_PATH}" \
  python "${GS_ROOT}/scripts/run_da3_bootstrap.py" \
  --input "${INPUT_PATH}" \
  --output "${OUTPUT_DIR}" \
  ${EXTRA_ARGS}
