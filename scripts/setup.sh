#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "${ROOT_DIR}/scripts/bootstrap_external.sh"
bash "${ROOT_DIR}/scripts/setup_4dgs_env.sh" cuda121
GSAM2_INSTALL_EDITABLE=1 bash "${ROOT_DIR}/scripts/setup_grounded_sam2.sh"

echo "ReferGaussian environments are ready."
echo "Next: bash scripts/download_models.sh"
