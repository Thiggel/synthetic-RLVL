#!/usr/bin/env bash
#SBATCH --job-name=rl_probe
#SBATCH --partition=gpu-staff
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Sampling version of the Stage-2 gate (scripts/rl_signal_probe.py): K samples per
# gate item at several temperatures; reports pass@K and the fraction of GRPO groups
# with nonzero reward variance per arm. Valid samples are kept for a rejection-
# sampling bootstrap. Required env: RUN_NAME (formal-mixture SFT run). Optional:
# K, TEMPS, PER_BENCH, NO_TAG=1, OUT_NAME, MODEL, GPU_MEM, EXTRA_ARGS (e.g. "--source train").
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
OUT_DIR="${OUT_ROOT}/${RUN_NAME}/${OUT_NAME:-rl_signal_probe}"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
python scripts/rl_signal_probe.py --model "${MODEL}" --out-dir "${OUT_DIR}" \
  --k "${K:-16}" --temperatures "${TEMPS:-0.6,1.0}" --per-bench "${PER_BENCH:-100}" \
  --gpu-mem "${GPU_MEM:-0.85}" ${NO_TAG:+--no-tag} ${EXTRA_ARGS:-}
