#!/usr/bin/env bash
#SBATCH --job-name=longwin_rebuild_nanoenv
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --exclude=a0531,a0532,a0533,a0934
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Rebuild the deleted $WORK/nanotron training venv (it vanished in the
# undocumented 2026-08-24..26 storage cleanup along with the vault nanosets
# and nanotron_checkpoints; atuin user file count dropped ~574k -> ~82k).
#
# Accepted-environment spec recovered from logs/repair_flash_nanotron_3760067:
#   Python 3.11.14, torch 2.6.0+cu124, flash_attn 2.7.4.post1
# plus the local patched checkout /home/hpc/c107fa/c107fa12/nanotron
# (HEAD 13625f34, includes padding-label masking dd6aae2e) installed editable,
# and datatrove for the TokenizedBytes loader.
#
# NOTE: exact transitive versions of the old env are not recoverable; the
# tiny-nanoset smoke (nanotron_qwen25_tiny_nanoset_smoke_2026-07-03.slurm)
# must be re-run before any midtrain is released on this env.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"

source ./scripts/env.sh

export UV_LINK_MODE=copy
# Memory model on alex: RAM is per-CPU and capped (a40 3750 MB/CPU, a100
# 7500 MB/CPU) and cpus-per-gpu is capped too, so --mem is rejected and the
# ONLY levers are partition + MAX_JOBS. Job 4109435 died OUT_OF_MEMORY with
# the SLURM_CPUS_PER_TASK default (16 parallel nvcc compiles, MaxRSS 113 GB
# against a 60000M a40 allocation). a100 x 16 CPUs = 120 GB; MAX_JOBS=6 peaks
# near 45 GB. Never restore the cpus-per-task default here.
export MAX_JOBS="${MAX_JOBS:-6}"
export FLASH_ATTENTION_FORCE_BUILD="${FLASH_ATTENTION_FORCE_BUILD:-TRUE}"

# flash-attn source build needs nvcc. Batch shells on alex have NO `module`
# function (4107718 looked for /usr/local/cuda/bin/nvcc; 4107741 died on
# "module: command not found"), so resolve the Spack CUDA 12.8.1 toolkit --
# the same version scripts/env.sh module-loads -- by path.
NVCC_PATH="${NVCC_PATH:-$(command -v nvcc || true)}"
if [[ -z "${NVCC_PATH}" ]]; then
  for CANDIDATE in /apps/SPACK/*/opt/spack/*/*/cuda-12.8.1-*/bin/nvcc; do
    [[ -x "${CANDIDATE}" ]] && NVCC_PATH="${CANDIDATE}" && break
  done
fi
if [[ -z "${NVCC_PATH}" ]]; then
  echo "no nvcc found; cannot source-build flash-attn" >&2
  exit 2
fi
export CUDA_HOME="$(dirname "$(dirname "${NVCC_PATH}")")"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
echo "using nvcc: ${NVCC_PATH}"
UV="${UV:-/home/hpc/c107fa/c107fa12/.local/bin/uv}"
ENV_DIR="${ENV_DIR:-${WORK}/nanotron}"
NANOTRON_REPO="/home/hpc/c107fa/c107fa12/nanotron"

if [[ "${FORCE_ENV_REBUILD:-0}" == "1" || ! -f "${ENV_DIR}/bin/activate" ]]; then
  rm -rf "${ENV_DIR}"
  "${UV}" venv --python 3.11 "${ENV_DIR}"
fi
PY="${ENV_DIR}/bin/python"
"${UV}" pip install --python "${PY}" \
  torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
"${UV}" pip install --python "${PY}" \
  -e "${NANOTRON_REPO}" \
  transformers "datatrove[io,processing]" numba pybind11 ninja wandb zstandard huggingface_hub \
  "numpy<2"
# flash-attn matching the accepted env (source build; needs nvcc + GPU node).
"${UV}" pip install --python "${PY}" flash-attn==2.7.4.post1 --no-build-isolation

source "${ENV_DIR}/bin/activate"
python - <<'PY'
import sys
print("python", sys.version)
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
import flash_attn
print("flash_attn", flash_attn.__version__)
import nanotron
from nanotron.helpers import lr_scheduler_builder
import inspect
assert 'param_group.get("initial_lr", param_group["lr"])' in inspect.getsource(lr_scheduler_builder), "LR resume fix missing"
from nanotron.data.clm_collator import DataCollatorForCLMWithPositionIds
from datatrove.utils.dataset import DatatroveFolderDataset
print("nanotron + datatrove loader imports OK; LR resume fix present")
PY
python -m pip freeze > "logs/nanotron_env_freeze_${SLURM_JOB_ID}.txt" 2>/dev/null || pip freeze > "logs/nanotron_env_freeze_${SLURM_JOB_ID}.txt"
echo "nanotron venv rebuilt at ${ENV_DIR}"
