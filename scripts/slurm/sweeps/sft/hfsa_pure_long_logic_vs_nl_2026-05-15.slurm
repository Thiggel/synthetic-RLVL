#!/usr/bin/env bash
#SBATCH --job-name=sft_hfsa_pure_long
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
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
TEMPLATES=(logic nl_exact logic nl_exact)
TRAIN_MAXES=(10 10 15 15)
TRAIN_SUBSETS=(
  train_fixedtarget_up_to_10_50k
  train_fixedtarget_up_to_10_50k
  train_fixedtarget_up_to_15_50k
  train_fixedtarget_up_to_15_50k
)
IDX=${SLURM_ARRAY_TASK_ID}
TEMPLATE=${TEMPLATES[$IDX]}
TRAIN_MAX=${TRAIN_MAXES[$IDX]}
TRAIN_SUBSET=${TRAIN_SUBSETS[$IDX]}
TAG="${TEMPLATE}_train1to${TRAIN_MAX}"
RUN_NAME="sft_hfsa_pure_long_${TAG}_10k_seed${SEED}"
GROUP="sft_hfsa_pure_long/${TAG}"

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="${GROUP}"
export WANDB_RUN_GROUP="${GROUP}"
sleep $((SLURM_ARRAY_TASK_ID * ${STARTUP_STAGGER_SECONDS:-300} + RANDOM % ${STARTUP_JITTER_SECONDS:-60}))

python train_sft.py \
  --config-name sft_hard_fsa_schema_fixedtarget \
  seed="${SEED}" \
  run_name="${RUN_NAME}" \
  task.template="${TEMPLATE}" \
  task.train_max_step="${TRAIN_MAX}" \
  task.val_max_step="${VAL_MAX:-25}" \
  data.materialized.train_subset="${TRAIN_SUBSET}" \
  data.train_samples="${TRAIN_SAMPLES:-50000}" \
  data.eval_samples="${DATA_EVAL_SAMPLES:-256}" \
  train.lr="${LR:-1e-4}" \
  train.max_steps="${MAX_STEPS:-10000}" \
  train.eval_steps="${EVAL_STEPS:-1000}" \
  train.save_steps="${SAVE_STEPS:-1000}" \
  validation.samples_per_step="${ONLINE_SAMPLES_PER_STEP:-16}" \
  validation.max_new_tokens="${ONLINE_MAX_NEW_TOKENS:-2048}"
