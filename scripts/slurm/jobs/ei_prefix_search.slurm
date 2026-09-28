#!/usr/bin/env bash
#SBATCH --job-name=ei_search
#SBATCH --partition=gpu-staff
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Expert-iteration collection with checker-guided prefix resampling
# (scripts/ei_prefix_search.py). Required env: RUN_NAME (formal-mixture SFT run).
# Optional: MODEL, OUT_NAME (default ei_search_r1), PER_BENCH, ROUNDS, GPU_MEM, EXTRA_ARGS.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
ROOT=/vol/tmp2/laitenbf
VENV="${ROOT}/.venv_rlvl_vllm"
OUT_ROOT="${ROOT}/rlvl_data/formal_mixture_sft_20260925"
export PATH="${VENV}/bin:${PATH}" VIRTUAL_ENV="${VENV}" HF_HOME="${ROOT}" HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export PYTHONPATH=/vol/home-vol2/ml/laitenbf/RLVL-next/gen:/vol/home-vol2/ml/laitenbf/RLVL-next/rlvl/python
RUN_NAME="${RUN_NAME:?}"
MODEL="${MODEL:-${OUT_ROOT}/${RUN_NAME}/final}"
OUT_DIR="${OUT_ROOT}/${RUN_NAME}/${OUT_NAME:-ei_search_r1}"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
python scripts/ei_prefix_search.py --model "${MODEL}" --out-dir "${OUT_DIR}" \
  --per-bench "${PER_BENCH:-1500}" --rounds "${ROUNDS:-3}" --gpu-mem "${GPU_MEM:-0.85}" ${EXTRA_ARGS:-}
