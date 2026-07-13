#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

SOURCE_ROOT="${1:-}"
TARGET_GROUP="${2:-}"
TARGET_SCENE="${3:-}"
LINK_MODE="${HYPERNERF_LINK_MODE:-symlink}"

if [[ -z "${SOURCE_ROOT}" || -z "${TARGET_GROUP}" || -z "${TARGET_SCENE}" ]]; then
  echo "Usage: $0 /abs/source_root <group> <scene>" >&2
  exit 2
fi

TARGET_ROOT="${GS_DATA_ROOT}/hypernerf/${TARGET_GROUP}/${TARGET_SCENE}"
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

echo "Prepared HyperNeRF scene at ${TARGET_ROOT}"
