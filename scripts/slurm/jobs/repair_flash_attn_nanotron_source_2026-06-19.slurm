#!/usr/bin/env bash
#SBATCH --job-name=repair_flash_nanotron
#SBATCH --partition=a40,a100,rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

export http_proxy="${http_proxy:-${HTTP_PROXY:-http://proxy:80}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY:-http://proxy:80}}"
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORK}/.cache/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${WORK}/.cache/pip}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${WORK}/.cache/torch_extensions}"
export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
export FLASH_ATTENTION_FORCE_BUILD="${FLASH_ATTENTION_FORCE_BUILD:-TRUE}"

if [[ ! -f "${WORK}/nanotron/bin/activate" ]]; then
  echo "missing nanotron env: ${WORK}/nanotron/bin/activate" >&2
  exit 2
fi

source "${WORK}/nanotron/bin/activate"
python -V
which python
which uv

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
PY

uv pip uninstall flash-attn || true
uv pip install ninja triton
uv pip install --reinstall --no-cache --no-binary flash-attn "flash-attn>=2.5.0,<2.8.0" --no-build-isolation

python - <<'PY'
import flash_attn
print("flash_attn", getattr(flash_attn, "__version__", "unknown"))
PY
