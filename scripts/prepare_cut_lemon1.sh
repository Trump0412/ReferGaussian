#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

SOURCE_ROOT="${CUT_LEMON1_SOURCE_ROOT:-${HYPERNERF_RAW_ROOT:-${GS_DATA_ROOT}/raw/HyperNeRF}/interp/cut-lemon1}"
TARGET_ROOT="${GS_DATA_ROOT}/hypernerf/interp/cut-lemon1"
LINK_MODE="${CUT_LEMON1_LINK_MODE:-symlink}"

mkdir -p "$(dirname "${TARGET_ROOT}")"

for required in dataset.json metadata.json scene.json points3D_downsample2.ply; do
  if [[ ! -e "${SOURCE_ROOT}/${required}" ]]; then
    echo "Missing required file: ${SOURCE_ROOT}/${required}" >&2
    exit 2
  fi
done

if [[ -L "${TARGET_ROOT}" || -d "${TARGET_ROOT}" ]]; then
  rm -rf "${TARGET_ROOT}"
fi

if [[ "${LINK_MODE}" == "copy" ]]; then
  cp -a "${SOURCE_ROOT}" "${TARGET_ROOT}"
else
  ln -s "${SOURCE_ROOT}" "${TARGET_ROOT}"
fi

echo "Prepared cut-lemon1 at ${TARGET_ROOT}"
