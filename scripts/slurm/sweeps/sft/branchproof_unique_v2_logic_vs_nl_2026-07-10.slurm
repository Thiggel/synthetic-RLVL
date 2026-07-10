#!/usr/bin/env bash
#SBATCH --job-name=sft_bp_unique
#SBATCH --partition=a40,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-29%12
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

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
IDX="${SLURM_ARRAY_TASK_ID}"
SEED_IDX=$((IDX % 3))
TRAIN_IDX=$(((IDX / 3) % 5))
TEMPLATE_IDX=$((IDX / 15))
SEED="${SEEDS[$SEED_IDX]}"
TEMPLATE="${TEMPLATES[$TEMPLATE_IDX]}"
TRAIN_MAX="${TRAIN_MAXES[$TRAIN_IDX]}"
TRAIN_SUBSET="${TRAIN_SUBSETS[$TRAIN_IDX]}"
RUN_NAME="sft_branchproof_unique_v2_${TEMPLATE}_train1to${TRAIN_MAX}_10k_seed${SEED}"
OUTPUT_DIR="${HPCVAULT}/synthetic-RLVL/runs/${RUN_NAME}"
DATASET_ID="${DATASET_ID:-flaitenberger/BranchProof-unique-v2}"

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="branchproof_unique_v2/${TEMPLATE}_train1to${TRAIN_MAX}"
export WANDB_RUN_GROUP="${WANDB_GROUP}"

sleep $((SLURM_ARRAY_TASK_ID * ${STARTUP_STAGGER_SECONDS:-20} + RANDOM % ${STARTUP_JITTER_SECONDS:-45}))

if [[ -f "${OUTPUT_DIR}/final/adapter_config.json" && "${FORCE_SFT:-0}" != "1" ]]; then
  echo "final checkpoint exists for ${RUN_NAME}; skipping"
  exit 0
fi

python train_sft.py \
  --config-name sft_hard_fsa_schema_fixedtarget \
  seed="${SEED}" \
  run_name="${RUN_NAME}" \
  output_dir="${OUTPUT_DIR}" \
  task.template="${TEMPLATE}" \
  task.train_max_step="${TRAIN_MAX}" \
  task.val_max_step=50 \
  data.materialized.dataset_id="${DATASET_ID}" \
  data.materialized.train_subset="${TRAIN_SUBSET}" \
  data.train_samples=50000 \
  data.eval_samples=256 \
  data.max_length=8192 \
  train.per_device_batch_size=1 \
  train.grad_accum=1 \
  train.lr=1e-4 \
  train.max_steps=10000 \
  train.eval_steps=20000 \
  train.save_steps=5000 \
  train.save_total_limit=2 \
  train.gradient_checkpointing=true \
  validation.eval_steps=20000 \
  validation.step_max=50 \
  validation.samples_per_step=4 \
  validation.max_new_tokens=7168
