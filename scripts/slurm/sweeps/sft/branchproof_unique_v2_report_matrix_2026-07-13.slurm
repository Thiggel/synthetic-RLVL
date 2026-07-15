#!/usr/bin/env bash
#SBATCH --job-name=sft_bp_v2_report
#SBATCH --partition=a40,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-0%3
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
hash -r 2>/dev/null || true

MATRIX_GROUP="${MATRIX_GROUP:?Set MATRIX_GROUP to a report-matrix group}"
ROW="$(${VENV_PATH}/bin/python scripts/experiments/branchproof_unique_v2_report_matrix.py \
  --group "${MATRIX_GROUP}" --kind train --index "${SLURM_ARRAY_TASK_ID}")"
IFS=$'\t' read -r FAMILY RUN_NAME TRAIN_TEMPLATE EVAL_TEMPLATES SEED TRAIN_MAX TRAIN_SUBSET \
  MODEL_NAME MAX_STEPS PER_DEVICE_BATCH_SIZE GRAD_ACCUM BALANCED DATA_KIND SHORTCUT_RATE SHORTCUT_KIND MAX_LENGTH <<< "${ROW}"

OUTPUT_DIR="${HPCVAULT}/synthetic-RLVL/runs/${RUN_NAME}"
DATASET_ID="flaitenberger/BranchProof-unique-v2"
LOCAL_ROOT="null"
if [[ "${DATA_KIND}" != "base" ]]; then
  DATASET_ID="null"
  LOCAL_ROOT="${HPCVAULT}/synthetic-RLVL/datasets/branchproof_unique_v2_${DATA_KIND}_20260713"
  if [[ ! -s "${LOCAL_ROOT}/${TRAIN_SUBSET}/train.parquet" ]]; then
    echo "Missing corrected shortcut data: ${LOCAL_ROOT}/${TRAIN_SUBSET}/train.parquet" >&2
    exit 2
  fi
fi

TARGET_STEP="${MAX_STEPS}"
SAVE_STEPS_VALUE="${SAVE_STEPS:-20000}"
SAVE_TOTAL_LIMIT_VALUE="${SAVE_TOTAL_LIMIT:-1}"
if [[ "${MATRIX_GROUP}" == "conditioned50k" ]]; then
  SAVE_STEPS_VALUE="${SAVE_STEPS:-5000}"
  SAVE_TOTAL_LIMIT_VALUE="${SAVE_TOTAL_LIMIT:-3}"
elif [[ "${MATRIX_GROUP}" == "batch" ]]; then
  # The batch-size-4 rows run slightly longer than the a100 partition's
  # 24-hour ceiling, so retain a recent resumable checkpoint.
  SAVE_STEPS_VALUE="${SAVE_STEPS:-1000}"
fi

TARGET_CHECKPOINT="${OUTPUT_DIR}/checkpoint-${TARGET_STEP}/trainer_state.json"
if [[ -f "${OUTPUT_DIR}/final/adapter_config.json" && "${MATRIX_GROUP}" != "conditioned50k" && "${FORCE_SFT:-0}" != "1" ]]; then
  echo "Final checkpoint exists for ${RUN_NAME}; skipping"
  exit 0
fi
if [[ -f "${TARGET_CHECKPOINT}" && -f "${OUTPUT_DIR}/final/adapter_config.json" && "${FORCE_SFT:-0}" != "1" ]]; then
  echo "Target step ${TARGET_STEP} exists for ${RUN_NAME}; skipping"
  exit 0
fi

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="branchproof_unique_v2_report/${FAMILY}/${TRAIN_TEMPLATE}_train1to${TRAIN_MAX}"
export WANDB_RUN_GROUP="${WANDB_GROUP}"

STAGGER_SLOTS="${STARTUP_STAGGER_SLOTS:-4}"
sleep $(((SLURM_ARRAY_TASK_ID % STAGGER_SLOTS) * ${STARTUP_STAGGER_SECONDS:-30} + RANDOM % ${STARTUP_JITTER_SECONDS:-30}))

python train_sft.py \
  --config-name sft_hard_fsa_schema_fixedtarget \
  hydra.run.dir="${OUTPUT_DIR}/hydra/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
  hydra.job.chdir=false \
  seed="${SEED}" \
  run_name="${RUN_NAME}" \
  output_dir="${OUTPUT_DIR}" \
  model.name="${MODEL_NAME}" \
  task.template="${TRAIN_TEMPLATE}" \
  task.shortcut_rate="${SHORTCUT_RATE}" \
  task.shortcut_kind="${SHORTCUT_KIND}" \
  task.train_max_step="${TRAIN_MAX}" \
  task.val_max_step=50 \
  data.materialized.dataset_id="${DATASET_ID}" \
  data.materialized.local_root="${LOCAL_ROOT}" \
  data.materialized.train_subset="${TRAIN_SUBSET}" \
  data.train_samples=50000 \
  data.eval_samples=256 \
  data.max_length="${SFT_MAX_LENGTH:-${MAX_LENGTH}}" \
  train.per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  train.grad_accum="${GRAD_ACCUM}" \
  +train.balanced_modality_batches="${BALANCED}" \
  train.lr="${LR:-1e-4}" \
  train.max_steps="${TARGET_STEP}" \
  train.eval_steps="$((TARGET_STEP + 1))" \
  train.save_steps="${SAVE_STEPS_VALUE}" \
  train.save_total_limit="${SAVE_TOTAL_LIMIT_VALUE}" \
  train.gradient_checkpointing="${GRADIENT_CHECKPOINTING:-true}" \
  train.resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-auto}" \
  validation.eval_steps="$((TARGET_STEP + 1))" \
  validation.step_max=50 \
  validation.samples_per_step=4 \
  validation.max_new_tokens=7168
