#!/usr/bin/env bash
#SBATCH --job-name=install_grouped_gemm
#SBATCH --partition=a40,a100,rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
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
export MAX_JOBS="${MAX_JOBS:-4}"
export CXXFLAGS="${CXXFLAGS:-} -D_GLIBCXX_USE_CXX11_ABI=0"
export CFLAGS="${CFLAGS:-} -D_GLIBCXX_USE_CXX11_ABI=0"

if [[ ! -f "${WORK}/nanotron/bin/activate" ]]; then
  echo "missing nanotron env: ${WORK}/nanotron/bin/activate" >&2
  exit 2
fi

source "${WORK}/nanotron/bin/activate"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
PY

uv pip uninstall nv-grouped-gemm || true
uv pip install --reinstall --no-cache --no-deps --no-binary nv-grouped-gemm --no-build-isolation \
  "git+https://github.com/fanshiqing/grouped_gemm@main"

python - <<'PY'
import grouped_gemm.ops as ops
print("grouped_gemm import ok", ops)
PY
