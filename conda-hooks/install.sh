#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-flux}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
ENV_PREFIX="${CONDA_BASE}/envs/${ENV_NAME}"

if [ ! -d "${ENV_PREFIX}" ]; then
  echo "Conda env not found: ${ENV_PREFIX}" >&2
  echo "Create it first, e.g.:" >&2
  echo "  conda env create -f environment.yml" >&2
  exit 1
fi

mkdir -p "${ENV_PREFIX}/etc/conda/activate.d" "${ENV_PREFIX}/etc/conda/deactivate.d"

install -m 755 "${SCRIPT_DIR}/activate.d/zz-cuda-toolkit-path.sh" \
  "${ENV_PREFIX}/etc/conda/activate.d/zz-cuda-toolkit-path.sh"
install -m 755 "${SCRIPT_DIR}/deactivate.d/zz-cuda-toolkit-path.sh" \
  "${ENV_PREFIX}/etc/conda/deactivate.d/zz-cuda-toolkit-path.sh"

echo "Installed CUDA toolkit PATH hooks into: ${ENV_PREFIX}"
echo "Re-activate the env to apply:"
echo "  conda deactivate && conda activate ${ENV_NAME}"
