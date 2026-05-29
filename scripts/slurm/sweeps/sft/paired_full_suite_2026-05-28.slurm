#!/usr/bin/env bash
#SBATCH --job-name=sft_pair_full
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-89%6
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

FAMILIES=(official_igsm maze_navigation attribute_constraints_hard)
TASK_DIFFICULTIES=(official_igsm maze_navigation attribute_constraints)
TRAIN_MAXES=(5 10 15 20 25)
TEMPLATES=(logic nl_exact)
SEEDS=(3407 3408 3409)

IDX="${SLURM_ARRAY_TASK_ID}"
SEED_IDX=$((IDX % 3))
TEMPLATE_IDX=$(((IDX / 3) % 2))
TRAIN_IDX=$(((IDX / 6) % 5))
FAMILY_IDX=$((IDX / 30))

FAMILY="${FAMILIES[${FAMILY_IDX}]}"
TASK_DIFFICULTY="${TASK_DIFFICULTIES[${FAMILY_IDX}]}"
TRAIN_MAX="${TRAIN_MAXES[${TRAIN_IDX}]}"
TEMPLATE="${TEMPLATES[${TEMPLATE_IDX}]}"
SEED="${SEEDS[${SEED_IDX}]}"

TRAIN_SUBSET="train_${FAMILY}_up_to_${TRAIN_MAX}_50k"
LOCAL_ROOT="${PAIRED_FULL_LOCAL_ROOT:-${WORK}/synthetic-RLVL/datasets/materialized_paired_${FAMILY}_full_20260528}"
TAG="${FAMILY}_${TEMPLATE}_train1to${TRAIN_MAX}"
RUN_NAME="sft_paired_full_${TAG}_10k_seed${SEED}"
GROUP="sft_paired_full/${FAMILY}/${TEMPLATE}_train1to${TRAIN_MAX}"
MAX_STEPS_VALUE="${MAX_STEPS:-10000}"
EVAL_STEPS_VALUE="${EVAL_STEPS:-$((MAX_STEPS_VALUE + 1))}"

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="${GROUP}"
export WANDB_RUN_GROUP="${GROUP}"

STARTUP_STAGGER_MOD_VALUE="${STARTUP_STAGGER_MOD:-6}"
STARTUP_STAGGER_INDEX=$((SLURM_ARRAY_TASK_ID % STARTUP_STAGGER_MOD_VALUE))
STARTUP_JITTER_SECONDS_VALUE="${STARTUP_JITTER_SECONDS:-90}"
if (( STARTUP_JITTER_SECONDS_VALUE > 0 )); then
  sleep $((STARTUP_STAGGER_INDEX * ${STARTUP_STAGGER_SECONDS:-240} + RANDOM % STARTUP_JITTER_SECONDS_VALUE))
else
  sleep $((STARTUP_STAGGER_INDEX * ${STARTUP_STAGGER_SECONDS:-240}))
fi

if [[ -f "${WORK}/synthetic-RLVL/runs/${RUN_NAME}/final/adapter_config.json" && "${FORCE_SFT:-0}" != "1" ]]; then
  echo "final checkpoint exists for ${RUN_NAME}; skipping"
  exit 0
fi

python train_sft.py \
  --config-name sft_hard_fsa_schema_fixedtarget \
  seed="${SEED}" \
  run_name="${RUN_NAME}" \
  task.difficulty="${TASK_DIFFICULTY}" \
  task.template="${TEMPLATE}" \
  task.train_max_step="${TRAIN_MAX}" \
  task.val_max_step="${ONLINE_VAL_MAX:-25}" \
  data.materialized.dataset_id=null \
  data.materialized.local_root="${LOCAL_ROOT}" \
  data.materialized.train_subset="${TRAIN_SUBSET}" \
  data.train_samples="${TRAIN_SAMPLES:-50000}" \
  data.eval_samples="${DATA_EVAL_SAMPLES:-256}" \
  data.max_length="${SFT_MAX_LENGTH:-16384}" \
  train.per_device_batch_size="${PER_DEVICE_BATCH_SIZE:-1}" \
  train.grad_accum="${GRAD_ACCUM:-1}" \
  train.lr="${LR:-1e-4}" \
  train.max_steps="${MAX_STEPS_VALUE}" \
  train.eval_steps="${EVAL_STEPS_VALUE}" \
  train.save_steps="${SAVE_STEPS:-5000}" \
  train.save_total_limit="${SAVE_TOTAL_LIMIT:-3}" \
  train.gradient_checkpointing="${GRADIENT_CHECKPOINTING:-true}" \
  validation.step_max="${ONLINE_VAL_MAX:-25}" \
  validation.samples_per_step="${ONLINE_SAMPLES_PER_STEP:-4}" \
  validation.max_new_tokens="${ONLINE_MAX_NEW_TOKENS:-8192}"
