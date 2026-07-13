#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"

FOURD_REPO_URL="${REFERGAUSSIAN_4DGS_REPO:-https://github.com/hustvl/4DGaussians.git}"
FOURD_REPO_REF="${REFERGAUSSIAN_4DGS_REF:-843d5ac636c37e4b611242287754f3d4ed150144}"
GSAM2_REPO_URL="${REFERGAUSSIAN_GSAM2_REPO:-https://github.com/IDEA-Research/Grounded-SAM-2.git}"
GSAM2_REPO_REF="${REFERGAUSSIAN_GSAM2_REF:-b7a9c29f196edff0eb54dbe14588d7ae5e3dde28}"

mkdir -p "${EXTERNAL_DIR}"

clone_checkout_repo() {
  local name="$1"
  local repo_url="$2"
  local repo_ref="$3"
  local marker="$4"
  local target="${EXTERNAL_DIR}/${name}"

  if [[ -f "${target}/${marker}" ]]; then
    echo "[ok] ${name} already present: ${target}"
    return 0
  fi

  if [[ -d "${target}" && -n "$(ls -A "${target}" 2>/dev/null || true)" ]]; then
    if [[ "${BOOTSTRAP_EXTERNAL_FORCE:-0}" != "1" ]]; then
      echo "[error] ${target} exists but ${marker} is missing." >&2
      echo "        Set BOOTSTRAP_EXTERNAL_FORCE=1 to replace it." >&2
      return 2
    fi
    rm -rf "${target}"
  fi

  echo "[clone] ${name} <- ${repo_url} @ ${repo_ref}"
  git clone "${repo_url}" "${target}"
  git -C "${target}" fetch --all --tags --prune
  git -C "${target}" checkout "${repo_ref}"

  if [[ -f "${target}/.gitmodules" ]]; then
    git -C "${target}" submodule update --init --recursive
  fi

  if [[ ! -f "${target}/${marker}" ]]; then
    echo "[error] ${name} checkout succeeded but marker missing: ${marker}" >&2
    return 3
  fi

  echo "[done] ${name} ready: ${target}"
}

clone_checkout_repo "4DGaussians" "${FOURD_REPO_URL}" "${FOURD_REPO_REF}" "train.py"
clone_checkout_repo "Grounded-SAM-2" "${GSAM2_REPO_URL}" "${GSAM2_REPO_REF}" "sam2/__init__.py"

REFERGAUSSIAN_4DGS_PATCH="${ROOT_DIR}/patches/4dgaussians_refergaussian.patch"
FOURD_TARGET="${EXTERNAL_DIR}/4DGaussians"
if git -C "${FOURD_TARGET}" apply --check --whitespace=nowarn "${REFERGAUSSIAN_4DGS_PATCH}"; then
  git -C "${FOURD_TARGET}" apply --whitespace=nowarn "${REFERGAUSSIAN_4DGS_PATCH}"
  echo "[done] applied ReferGaussian integration patch to 4DGaussians"
elif git -C "${FOURD_TARGET}" apply --reverse --check "${REFERGAUSSIAN_4DGS_PATCH}"; then
  echo "[ok] ReferGaussian integration patch already applied"
else
  echo "[error] 4DGaussians checkout does not match the released integration patch." >&2
  exit 4
fi

echo "All external dependencies are ready under ${EXTERNAL_DIR}."
