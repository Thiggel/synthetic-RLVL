#!/usr/bin/env bash
#SBATCH --job-name=sft_hard_v5_short_lr1e4
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then
  VENV_PATH="${WORK}/.venv"
fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

SEEDS=(3407 3408 3409)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID}]}
RUN_NAME="sft_hard_v5_short_lr1e-4_seed${SEED}"
GROUP="sft_hard_v5_short/lr1e-4_depth1to3_10k"

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="${GROUP}"
export WANDB_RUN_GROUP="${GROUP}"

sleep $((SLURM_ARRAY_TASK_ID * ${STARTUP_STAGGER_SECONDS:-300} + RANDOM % ${STARTUP_JITTER_SECONDS:-60}))

python train_sft.py \
  --config-name sft_hard_v5_short \
  seed="${SEED}" \
  run_name="${RUN_NAME}" \
  train.lr="${LR:-1e-4}" \
  train.max_steps="${MAX_STEPS:-1000}" \
  train.eval_steps="${EVAL_STEPS:-250}" \
  train.save_steps="${SAVE_STEPS:-250}" \
  data.train_samples="${TRAIN_SAMPLES:-10000}"
