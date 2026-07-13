#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

DA3_ENV_PATH="${DA3_ENV_PATH:-${GS_ENV_ROOT}/da3-gs-py310}"
DA3_REPO_DIR="${DA3_REPO_DIR:-${GS_ROOT}/external/Depth-Anything-3}"
DA3_GSPLAT_REPO_DIR="${DA3_GSPLAT_REPO_DIR:-${GS_ROOT}/external/gsplat}"
DA3_INSTALL_GSPLAT="${DA3_INSTALL_GSPLAT:-0}"
DA3_HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
DA3_HF_HOME="${HF_HOME:-${GS_CACHE_ROOT}/huggingface}"
DA3_PIP_CACHE_DIR="${PIP_CACHE_DIR:-${GS_PIP_CACHE_DIR}}"
DA3_GIT_URL="${DA3_GIT_URL:-https://github.com/ByteDance-Seed/Depth-Anything-3.git}"
DA3_GIT_MIRROR_URL="${DA3_GIT_MIRROR_URL:-https://mirror.ghproxy.com/https://github.com/ByteDance-Seed/Depth-Anything-3.git}"
DA3_ARCHIVE_URL="${DA3_ARCHIVE_URL:-https://mirror.ghproxy.com/https://codeload.github.com/ByteDance-Seed/Depth-Anything-3/tar.gz/refs/heads/main}"
DA3_ARCHIVE_FALLBACK_URL="${DA3_ARCHIVE_FALLBACK_URL:-https://codeload.github.com/ByteDance-Seed/Depth-Anything-3/tar.gz/refs/heads/main}"
DA3_GSPLAT_GIT_URL="${DA3_GSPLAT_GIT_URL:-https://github.com/nerfstudio-project/gsplat.git}"
DA3_GSPLAT_GIT_MIRROR_URL="${DA3_GSPLAT_GIT_MIRROR_URL:-https://mirror.ghproxy.com/https://github.com/nerfstudio-project/gsplat.git}"
DA3_GSPLAT_ARCHIVE_URL="${DA3_GSPLAT_ARCHIVE_URL:-https://mirror.ghproxy.com/https://codeload.github.com/nerfstudio-project/gsplat/tar.gz/0b4dddf04cb687367602c01196913cde6a743d70}"
DA3_GSPLAT_ARCHIVE_FALLBACK_URL="${DA3_GSPLAT_ARCHIVE_FALLBACK_URL:-https://codeload.github.com/nerfstudio-project/gsplat/tar.gz/0b4dddf04cb687367602c01196913cde6a743d70}"
DA3_PIP_INDEX_URL="${DA3_PIP_INDEX_URL:-https://pypi.org/simple}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY
export HF_ENDPOINT="${DA3_HF_ENDPOINT}"
export HF_HOME="${DA3_HF_HOME}"
export XDG_CACHE_HOME="${GS_CACHE_ROOT}"
export TORCH_HOME="${GS_TORCH_HOME}"
export MPLCONFIGDIR="${GS_MPLCONFIGDIR}"
export CONDA_PKGS_DIRS="${GS_CONDA_PKGS_DIRS}"
export PIP_CACHE_DIR="${DA3_PIP_CACHE_DIR}"
export PIP_INDEX_URL="${DA3_PIP_INDEX_URL}"
export GIT_HTTP_VERSION=HTTP/1.1

mkdir -p "${GS_ENV_ROOT}" "${DA3_HF_HOME}" "${DA3_PIP_CACHE_DIR}"

download_repo_archive() {
  local archive_url="$1"
  local dest_dir="$2"
  local marker_relpath="$3"
  local tmp_dir
  local archive_path
  local extracted_dir

  tmp_dir="$(mktemp -d)"
  archive_path="${tmp_dir}/repo.tar.gz"
  curl -L --retry 5 --retry-delay 2 --connect-timeout 20 "${archive_url}" -o "${archive_path}"
  tar -xzf "${archive_path}" -C "${tmp_dir}"
  extracted_dir="$(find "${tmp_dir}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
  if [[ -z "${extracted_dir}" || ! -e "${extracted_dir}/${marker_relpath}" ]]; then
    rm -rf "${tmp_dir}"
    return 1
  fi
  rm -rf "${dest_dir}"
  mv "${extracted_dir}" "${dest_dir}"
  rm -rf "${tmp_dir}"
}

ensure_repo_dir() {
  local dest_dir="$1"
  local marker_relpath="$2"
  local archive_url="$3"
  local archive_fallback_url="$4"
  local git_mirror_url="$5"
  local git_url="$6"

  if [[ -e "${dest_dir}/${marker_relpath}" ]]; then
    return 0
  fi

  if download_repo_archive "${archive_url}" "${dest_dir}" "${marker_relpath}"; then
    return 0
  fi

  if download_repo_archive "${archive_fallback_url}" "${dest_dir}" "${marker_relpath}"; then
    return 0
  fi

  rm -rf "${dest_dir}"
  if git -c http.version=HTTP/1.1 clone --depth 1 "${git_mirror_url}" "${dest_dir}"; then
    return 0
  fi

  rm -rf "${dest_dir}"
  git -c http.version=HTTP/1.1 clone --depth 1 "${git_url}" "${dest_dir}"
}

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH" >&2
  exit 1
fi

if [[ ! -d "${DA3_ENV_PATH}" ]]; then
  conda create -y -p "${DA3_ENV_PATH}" python=3.10
fi

ensure_repo_dir \
  "${DA3_REPO_DIR}" \
  "pyproject.toml" \
  "${DA3_ARCHIVE_URL}" \
  "${DA3_ARCHIVE_FALLBACK_URL}" \
  "${DA3_GIT_MIRROR_URL}" \
  "${DA3_GIT_URL}"

conda run --no-capture-output -p "${DA3_ENV_PATH}" python -m pip install --upgrade pip "setuptools<81" wheel -i "${DA3_PIP_INDEX_URL}"
conda run --no-capture-output -p "${DA3_ENV_PATH}" python -m pip install \
  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121
conda run --no-capture-output -p "${DA3_ENV_PATH}" python -m pip install \
  "numpy<2" huggingface_hub safetensors xformers==0.0.23.post1 \
  addict einops imageio opencv-python omegaconf moviepy==1.0.3 plyfile pillow-heif \
  -i "${DA3_PIP_INDEX_URL}"
conda run --no-capture-output -p "${DA3_ENV_PATH}" python -m pip install --no-deps -e "${DA3_REPO_DIR}"

if [[ "${DA3_INSTALL_GSPLAT}" == "1" ]]; then
  ensure_repo_dir \
    "${DA3_GSPLAT_REPO_DIR}" \
    "setup.py" \
    "${DA3_GSPLAT_ARCHIVE_URL}" \
    "${DA3_GSPLAT_ARCHIVE_FALLBACK_URL}" \
    "${DA3_GSPLAT_GIT_MIRROR_URL}" \
    "${DA3_GSPLAT_GIT_URL}"

  conda run --no-capture-output -p "${DA3_ENV_PATH}" python -m pip install \
    --no-build-isolation \
    "${DA3_GSPLAT_REPO_DIR}"
else
  echo "Skipping gsplat install; set DA3_INSTALL_GSPLAT=1 for gs_video export."
fi

echo "DA3 environment ready: ${DA3_ENV_PATH}"
echo "HF mirror endpoint: ${HF_ENDPOINT}"
echo "Git mirror URL: ${DA3_GIT_URL}"
echo "Repository path: ${DA3_REPO_DIR}"
