#!/usr/bin/env bash
# Download all HyperNeRF scenes used in the ReferGaussian paper.
# Each scene requires COLMAP to generate the initial point cloud.
# Usage:
#   bash scripts/prepare_hypernerf.sh              # download all paper scenes
#   bash scripts/prepare_hypernerf.sh misc keyboard # download one scene
set -euo pipefail

source "$(dirname "$0")/common.sh"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ROOT_DIR}/data/hypernerf"
DOWNLOAD_ROOT="${ROOT_DIR}/data/downloads"

# All HyperNeRF scenes used in the paper (group/scene : asset_zip)
declare -A PAPER_SCENES=(
  ["misc/keyboard"]="misc_keyboard.zip"
  ["misc/espresso"]="misc_espresso.zip"
  ["misc/americano"]="misc_americano.zip"
  ["misc/split-cookie"]="misc_split-cookie.zip"
  ["misc/cut-lemon1"]="misc_cut-lemon1.zip"
  ["misc/torchocolate"]="misc_torchocolate.zip"
)

prepare_scene() {
  local group_scene="$1"
  local asset_name="$2"
  local group="${group_scene%%/*}"
  local scene="${group_scene##*/}"
  local target_dir="${DATA_ROOT}/${group}/${scene}"
  local zip_path="${DOWNLOAD_ROOT}/${asset_name}"
  local url="https://github.com/google/hypernerf/releases/download/v0.1/${asset_name}"

  mkdir -p "${DATA_ROOT}/${group}" "${DOWNLOAD_ROOT}"

  if [[ -f "${target_dir}/dataset.json" ]]; then
    echo "[skip] ${group_scene} already prepared"
    return
  fi

  if [[ ! -f "${zip_path}" ]]; then
    echo "[download] ${url}"
    wget -q --show-progress -O "${zip_path}" "${url}"
  fi

  echo "[extract] ${asset_name} -> ${target_dir}"
  ${PYTHON_FOR_EXTRACTION:-$(command -v python3 || command -v python)} - <<PY "${zip_path}" "${target_dir}"
import os, shutil, sys, tempfile, zipfile
zip_path, target_dir = sys.argv[1], sys.argv[2]
temp_dir = tempfile.mkdtemp(prefix="hypernerf_extract_", dir=os.path.dirname(target_dir))
with zipfile.ZipFile(zip_path) as archive:
    archive.extractall(temp_dir)
scene_root = None
for root, _dirs, files in os.walk(temp_dir):
    if {"dataset.json", "metadata.json", "scene.json"}.issubset(set(files)):
        scene_root = root
        break
if scene_root is None:
    raise SystemExit(f"Cannot find HyperNeRF scene root in {zip_path}")
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
shutil.move(scene_root, target_dir)
shutil.rmtree(temp_dir)
PY

  if [[ ! -f "${target_dir}/points3D_downsample2.ply" ]]; then
    if command -v colmap >/dev/null 2>&1; then
      echo "[colmap] generating point cloud for ${group_scene}"
      bash "${ROOT_DIR}/external/4DGaussians/colmap.sh" "${target_dir}" hypernerf
      gs_python "${ROOT_DIR}/external/4DGaussians/scripts/downsample_point.py" \
        "${target_dir}/colmap/dense/workspace/fused.ply" \
        "${target_dir}/points3D_downsample2.ply"
    else
      echo "WARNING: COLMAP not found. Install COLMAP and re-run, or place a pregenerated" >&2
      echo "         point cloud at ${target_dir}/points3D_downsample2.ply" >&2
    fi
  fi

  echo "[done] ${target_dir}"
}

# Single-scene mode: bash prepare_hypernerf.sh misc keyboard
if [[ $# -ge 2 ]]; then
  GROUP="$1"
  SCENE="$2"
  ASSET="${HYPERNERF_ASSET:-${GROUP}_${SCENE}.zip}"
  prepare_scene "${GROUP}/${SCENE}" "${ASSET}"
  exit 0
fi

# Full download: all paper scenes
echo "Downloading all HyperNeRF scenes used in the paper..."
for group_scene in "${!PAPER_SCENES[@]}"; do
  prepare_scene "${group_scene}" "${PAPER_SCENES[${group_scene}]}"
done
echo "All scenes ready under ${DATA_ROOT}"
