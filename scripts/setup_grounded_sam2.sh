#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_conda_bin
require_grounded_sam2

GSAM2_ROOT="${GS_ROOT}/external/Grounded-SAM-2"
PYTHON_VERSION="${1:-3.10}"
TORCH_VERSION="${GSAM2_TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${GSAM2_TORCHVISION_VERSION:-0.20.1}"
TORCHAUDIO_VERSION="${GSAM2_TORCHAUDIO_VERSION:-2.5.1}"
PIP_INDEX_URL="${GSAM2_PIP_INDEX_URL:-${GSAM2_PIP_MIRROR:-https://pypi.org/simple}}"
HF_MIRROR="${HF_ENDPOINT:-https://huggingface.co}"
SAM2_MODEL_ID="${GSAM2_SAM2_MODEL_ID:-facebook/sam2-hiera-large}"
GDINO_MODEL_ID="${GSAM2_GDINO_MODEL_ID:-IDEA-Research/grounding-dino-base}"
SAM2_MODEL_REVISION="${GSAM2_SAM2_MODEL_REVISION:-e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251}"
GDINO_MODEL_REVISION="${GSAM2_GROUNDING_MODEL_REVISION:-12bdfa3120f3e7ec7b434d90674b3396eccf88eb}"
# The query scripts must import the pinned checkout below, not an editable SAM2
# package left by another project in the same conda environment.
INSTALL_EDITABLE="${GSAM2_INSTALL_EDITABLE:-1}"
# Prefer a fully pinned local cache. Set this to 0 for an explicitly offline
# setup that should fail rather than attempting a first-time download.
DOWNLOAD_WEIGHTS="${GSAM2_DOWNLOAD_WEIGHTS:-1}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY
export HF_ENDPOINT="${HF_MIRROR}"
export PIP_CACHE_DIR="${GS_PIP_CACHE_DIR}"
export CONDA_PKGS_DIRS="${GS_CONDA_PKGS_DIRS}"
export XDG_CACHE_HOME="${GS_CACHE_ROOT}"
export TORCH_HOME="${GS_TORCH_HOME}"
export MPLCONFIGDIR="${GS_MPLCONFIGDIR}"
export CUDA_HOME="${GS4D_CUDA_HOME:-/usr/local/cuda-12.1}"
export SAM2_BUILD_ALLOW_ERRORS=1

mkdir -p "$(dirname "${GSAM2_ENV_PATH}")"

if [[ ! -d "${GSAM2_ENV_PATH}" ]]; then
  "${GS_CONDA_BIN}" create -y -p "${GSAM2_ENV_PATH}" "python=${PYTHON_VERSION}" pip
fi

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "${GS_CONDA_BIN}" run --no-capture-output -p "${GSAM2_ENV_PATH}" python -m pip install --upgrade pip setuptools wheel

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "${GS_CONDA_BIN}" run --no-capture-output -p "${GSAM2_ENV_PATH}" python -m pip install \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url https://download.pytorch.org/whl/cu121

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "${GS_CONDA_BIN}" run --no-capture-output -p "${GSAM2_ENV_PATH}" python -m pip install --index-url "${PIP_INDEX_URL}" \
    "numpy==1.26.4" "transformers==5.3.0" "huggingface_hub==1.7.2" "pillow==12.0.0" "tqdm==4.67.3" \
    "hydra-core==1.3.2" "iopath==0.1.10" "opencv-python==4.11.0.86" "supervision==0.27.0.post2" \
    "pyyaml==6.0.3" "matplotlib==3.10.8" "accelerate==1.13.0" "sentencepiece==0.2.1"

if [[ "${INSTALL_EDITABLE}" == "1" ]]; then
  pushd "${GSAM2_ROOT}" >/dev/null
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_HOME="${CUDA_HOME}" SAM2_BUILD_ALLOW_ERRORS=1 \
    "${GS_CONDA_BIN}" run --no-capture-output -p "${GSAM2_ENV_PATH}" python -m pip install --no-build-isolation -e .
  popd >/dev/null
fi

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_ENDPOINT="${HF_MIRROR}" \
  "${GS_CONDA_BIN}" run --no-capture-output -p "${GSAM2_ENV_PATH}" python - <<PY
import sys
from huggingface_hub import hf_hub_download
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
sys.path.insert(0, "${GSAM2_ROOT}")
from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES, build_sam2_video_predictor

gdino_model_id = "${GDINO_MODEL_ID}"
sam2_model_id = "${SAM2_MODEL_ID}"
gdino_model_revision = "${GDINO_MODEL_REVISION}"
sam2_model_revision = "${SAM2_MODEL_REVISION}"
allow_download = "${DOWNLOAD_WEIGHTS}" == "1"

config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[sam2_model_id]

def load_pinned_assets(*, local_files_only: bool):
    processor = AutoProcessor.from_pretrained(
        gdino_model_id,
        revision=gdino_model_revision,
        use_fast=True,
        local_files_only=local_files_only,
    )
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        gdino_model_id,
        revision=gdino_model_revision,
        local_files_only=local_files_only,
    )
    checkpoint_path = hf_hub_download(
        repo_id=sam2_model_id,
        filename=checkpoint_name,
        revision=sam2_model_revision,
        local_files_only=local_files_only,
    )
    return processor, grounding_model, checkpoint_path

# Avoid a network lookup when an exact local snapshot already exists. Besides
# making setup reliable on clusters, this sidesteps optional upstream files that
# newer Transformers versions may probe even though inference does not need them.
try:
    processor, grounding_model, checkpoint_path = load_pinned_assets(local_files_only=True)
except Exception as offline_error:
    if not allow_download:
        raise RuntimeError(
            "Pinned Grounded-SAM2 weights are missing from the local cache. "
            "Set GSAM2_DOWNLOAD_WEIGHTS=1 on a machine with Hub access."
        ) from offline_error
    print("[info] pinned Grounded-SAM2 cache incomplete; downloading required files")
    processor, grounding_model, checkpoint_path = load_pinned_assets(local_files_only=False)

predictor = build_sam2_video_predictor(config_file=config_name, ckpt_path=checkpoint_path)

from pathlib import Path
import importlib.util
package_file = Path(__import__("sam2").__file__).resolve()
source_root = Path("${GSAM2_ROOT}").resolve()
if source_root not in package_file.parents:
    raise RuntimeError(f"sam2 imported from another checkout: {package_file}")
extension = importlib.util.find_spec("sam2._C")
if extension and extension.origin:
    extension_path = Path(extension.origin).resolve()
    if source_root not in extension_path.parents:
        raise RuntimeError(f"sam2._C imported from another checkout: {extension_path}")

print("grounding processor", type(processor).__name__)
print("grounding model", type(grounding_model).__name__)
print("sam2 predictor", type(predictor).__name__)
print("gsam2 env ready:", "${GSAM2_ENV_PATH}")
PY
