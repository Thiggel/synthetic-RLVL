#!/usr/bin/env bash
#SBATCH --job-name=sft_hfsa_dscale
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-29%6
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

SEEDS=(3407 3408 3409)
TEMPLATES=(logic nl_exact)
TRAIN_MAXES=(5 10 15 20 25)
TRAIN_SUBSETS=(
  train_fixedtarget_up_to_5_50k
  train_fixedtarget_up_to_10_50k
  train_fixedtarget_up_to_15_50k
  train_fixedtarget_up_to_20_50k
  train_fixedtarget_up_to_25_50k
)
IDX=${SLURM_ARRAY_TASK_ID}
SEED_IDX=$((IDX % 3))
TRAIN_IDX=$(((IDX / 3) % 5))
TEMPLATE_IDX=$((IDX / 15))
SEED=${SEEDS[$SEED_IDX]}
TEMPLATE=${TEMPLATES[$TEMPLATE_IDX]}
TRAIN_MAX=${TRAIN_MAXES[$TRAIN_IDX]}
TRAIN_SUBSET=${TRAIN_SUBSETS[$TRAIN_IDX]}
TAG="${TEMPLATE}_train1to${TRAIN_MAX}"
RUN_NAME="sft_hfsa_depth_scaling_${TAG}_10k_seed${SEED}"
GROUP="sft_hfsa_depth_scaling/${TAG}"
DATASET_ID="${DATASET_ID:-flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget-depth50}"

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="${GROUP}"
export WANDB_RUN_GROUP="${GROUP}"

sleep $((SLURM_ARRAY_TASK_ID * ${STARTUP_STAGGER_SECONDS:-180} + RANDOM % ${STARTUP_JITTER_SECONDS:-90}))

if [[ -f "${WORK}/synthetic-RLVL/runs/${RUN_NAME}/final/adapter_config.json" && "${FORCE_SFT:-0}" != "1" ]]; then
  echo "final checkpoint exists for ${RUN_NAME}; skipping"
  exit 0
fi

python train_sft.py \
  --config-name sft_hard_fsa_schema_fixedtarget \
  seed="${SEED}" \
  run_name="${RUN_NAME}" \
  task.template="${TEMPLATE}" \
  task.train_max_step="${TRAIN_MAX}" \
  task.val_max_step="${ONLINE_VAL_MAX:-25}" \
  data.materialized.dataset_id="${DATASET_ID}" \
  data.materialized.train_subset="${TRAIN_SUBSET}" \
  data.train_samples="${TRAIN_SAMPLES:-50000}" \
  data.eval_samples="${DATA_EVAL_SAMPLES:-256}" \
  data.max_length="${SFT_MAX_LENGTH:-8192}" \
  train.per_device_batch_size="${PER_DEVICE_BATCH_SIZE:-1}" \
  train.grad_accum="${GRAD_ACCUM:-1}" \
  train.lr="${LR:-1e-4}" \
  train.max_steps="${MAX_STEPS:-10000}" \
  train.eval_steps="${EVAL_STEPS:-1000}" \
  train.save_steps="${SAVE_STEPS:-1000}" \
  train.save_total_limit="${SAVE_TOTAL_LIMIT:-12}" \
  train.gradient_checkpointing="${GRADIENT_CHECKPOINTING:-auto}" \
  validation.step_max="${ONLINE_VAL_MAX:-25}" \
  validation.samples_per_step="${ONLINE_SAMPLES_PER_STEP:-4}" \
  validation.max_new_tokens="${ONLINE_MAX_NEW_TOKENS:-3072}"
