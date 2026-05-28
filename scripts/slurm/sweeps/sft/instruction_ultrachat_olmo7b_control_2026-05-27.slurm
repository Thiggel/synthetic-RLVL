#!/usr/bin/env bash
#SBATCH --job-name=sft_instr_ctrl
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-0%1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then VENV_PATH="${WORK}/.venv"; fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

SEED="${SEED:-3407}"
RUN_NAME="sft_instruction_ultrachat200k_olmo3_7b_10k_seed${SEED}"
OUT_DIR="${WORK}/synthetic-RLVL/runs/${RUN_NAME}"
GROUP="sft_instruction_control/ultrachat200k_answer_tags"

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="${GROUP}"
export WANDB_RUN_GROUP="${GROUP}"

if [[ -f "${OUT_DIR}/final/adapter_config.json" && "${FORCE_SFT:-0}" != "1" ]]; then
  echo "final checkpoint exists for ${RUN_NAME}; skipping"
  exit 0
fi

python scripts/train_instruction_sft.py \
  --model "${MODEL_NAME:-allenai/Olmo-3-1025-7B}" \
  --dataset "${INSTRUCTION_DATASET:-HuggingFaceH4/ultrachat_200k}" \
  --train-split "${TRAIN_SPLIT:-train_sft}" \
  --eval-split "${EVAL_SPLIT:-test_sft}" \
  --run-name "${RUN_NAME}" \
  --output-dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --train-samples "${TRAIN_SAMPLES:-50000}" \
  --eval-samples "${EVAL_SAMPLES:-512}" \
  --max-length "${SFT_MAX_LENGTH:-4096}" \
  --max-steps "${MAX_STEPS:-10000}" \
  --lr "${LR:-1e-4}" \
  --warmup-steps "${WARMUP_STEPS:-50}" \
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE:-1}" \
  --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}" \
  --grad-accum "${GRAD_ACCUM:-1}" \
  --eval-steps "${EVAL_STEPS:-1000}" \
  --save-steps "${SAVE_STEPS:-1000}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT:-12}" \
  ${GRADIENT_CHECKPOINTING:+--gradient-checkpointing}
