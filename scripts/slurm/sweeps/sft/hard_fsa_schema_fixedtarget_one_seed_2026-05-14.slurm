#!/usr/bin/env bash
#SBATCH --job-name=sft_hfsa_fix
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
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
TAGS=(sft1to3 sft1to5)
TRAIN_MAXES=(3 5)
TRAIN_SUBSETS=(train_fixedtarget_up_to_3_50k train_fixedtarget_up_to_5_50k)
IDX=${SLURM_ARRAY_TASK_ID}
TAG=${TAGS[$IDX]}
TRAIN_MAX=${TRAIN_MAXES[$IDX]}
TRAIN_SUBSET=${TRAIN_SUBSETS[$IDX]}
RUN_NAME="sft_hfsa_fixedtarget_${TAG}_lr1e-4_seed${SEED}"
GROUP="sft_hard_fsa_schema_fixedtarget/${TAG}/seed${SEED}"

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="${GROUP}"
export WANDB_RUN_GROUP="${GROUP}"
sleep $((SLURM_ARRAY_TASK_ID * ${STARTUP_STAGGER_SECONDS:-300} + RANDOM % ${STARTUP_JITTER_SECONDS:-60}))

python train_sft.py \
  --config-name sft_hard_fsa_schema_fixedtarget \
  seed="${SEED}" \
  run_name="${RUN_NAME}" \
  task.train_max_step="${TRAIN_MAX}" \
  data.materialized.train_subset="${TRAIN_SUBSET}" \
  data.train_samples="${TRAIN_SAMPLES:-50000}" \
  train.lr="${LR:-1e-4}" \
  train.max_steps="${MAX_STEPS:-3000}" \
  train.eval_steps="${EVAL_STEPS:-500}" \
  train.save_steps="${SAVE_STEPS:-500}"
