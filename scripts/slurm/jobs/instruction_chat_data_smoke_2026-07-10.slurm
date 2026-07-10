#!/usr/bin/env bash
#SBATCH --job-name=instr_chat_smoke
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100small:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
hash -r 2>/dev/null || true

python scripts/train_instruction_sft.py \
  --model "${MODEL_ID:-Qwen/Qwen2.5-7B}" \
  --dataset "${INSTRUCTION_DATASET:-HuggingFaceH4/ultrachat_200k}" \
  --train-split "${TRAIN_SPLIT:-train_sft}" \
  --eval-split "${EVAL_SPLIT:-test_sft}" \
  --train-samples "${TRAIN_SAMPLES:-32}" \
  --eval-samples "${EVAL_SAMPLES:-16}" \
  --max-length "${SFT_MAX_LENGTH:-4096}" \
  --format-mode chat \
  --dry-run
