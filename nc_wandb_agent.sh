#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

if [ ! -f ".venv/bin/activate" ]; then
  echo "Missing virtualenv activation file: ${REPO_DIR}/.venv/bin/activate"
  exit 1
fi
source .venv/bin/activate

export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
mkdir -p "${WANDB_DIR}" job_outputs

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
  if command -v module >/dev/null 2>&1; then
    module purge
    module load cuda/12.1
    module load cudnn/9.10.2
    module list
  fi
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
nvidia-smi -L || true

# Make cuPTI visible (often needed for CUDA/JAX/PyTorch tooling checks).
export LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"

SWEEP_PATH="${1:?need sweep path like entity/project/sweepid}"
COUNT="${2:-1}"

echo "Starting W&B agent: sweep=${SWEEP_PATH}, count=${COUNT}"
wandb agent --count "${COUNT}" "${SWEEP_PATH}"
