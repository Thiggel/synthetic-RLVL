#!/usr/bin/env bash
#SBATCH --job-name=rl_filter
#SBATCH --partition=gpu-staff
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Per-prompt GRPO reward rate + harvest of passing completions (scripts/rl_prompt_filter.py).
# Required env: MODEL (absolute path), OUT (json path). Optional: FILTER_ARGS (passed through).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
ROOT=/vol/tmp2/laitenbf
VENV="${ROOT}/.venv_rlvl_grpo"
export PATH="${VENV}/bin:${PATH}" VIRTUAL_ENV="${VENV}"
export HF_HOME="${ROOT}" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}" PATH="${CUDA_HOME:-/usr/local/cuda-13.2}/bin:${PATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0 TRITON_CACHE_DIR=/tmp/laitenbf_triton_${SLURM_JOB_ID:-0}
export PYTHONPATH="${RLVL_PYTHONPATH:-/vol/home-vol2/ml/laitenbf/RLVL-next/gen:/vol/home-vol2/ml/laitenbf/RLVL-next/rlvl/python}"  # RLVL_PYTHONPATH: frozen checker
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=index,name,memory.free --format=csv,noheader
python scripts/rl_prompt_filter.py --model "${MODEL:?}" --out "${OUT:?}" ${FILTER_ARGS:-}
