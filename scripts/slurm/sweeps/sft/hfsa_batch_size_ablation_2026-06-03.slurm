#!/usr/bin/env bash
#SBATCH --job-name=sft_hfsa_bsz
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-11%3
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

# The default $WORK filesystem is currently unable to create new directories
# for this user, so keep this new ablation's outputs/caches on vault.
BATCH_WORK="${BATCH_WORK:-${HPCVAULT:-${WORK}}}"
export WORK="${BATCH_WORK}"
export SYNTHRLVL_WORK_ROOT="${WORK}/synthetic-RLVL"
export XDG_CACHE_HOME="${BATCH_XDG_CACHE_HOME:-${WORK}/.cache_batch_size_20260603}"
export HF_HOME="${BATCH_HF_HOME:-${XDG_CACHE_HOME}/hf}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export WANDB_DIR="${SYNTHRLVL_WORK_ROOT}/wandb"
export WANDB_ARTIFACT_DIR="${SYNTHRLVL_WORK_ROOT}/wandb_artifacts"
export WANDB_DATA_DIR="${SYNTHRLVL_WORK_ROOT}/wandb_data"
export WANDB_CACHE_DIR="${SYNTHRLVL_WORK_ROOT}/wandb_cache"
mkdir -p "${SYNTHRLVL_WORK_ROOT}" "${HF_HOME}" "${WANDB_DIR}" "${WANDB_ARTIFACT_DIR}" "${WANDB_DATA_DIR}" "${WANDB_CACHE_DIR}"

SEED="${SEED:-3407}"
BATCH_SIZES=(2 4 8 16)
CONDITIONS=(logic nl_exact conditioned_dual)
TRAIN_MAX="${TRAIN_MAX:-20}"
TRAIN_SUBSET="${TRAIN_SUBSET:-train_fixedtarget_up_to_20_50k}"
DATASET_ID="${DATASET_ID:-flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget-depth50}"

IDX=${SLURM_ARRAY_TASK_ID}
BATCH_IDX=$((IDX % 4))
COND_IDX=$((IDX / 4))
BATCH_SIZE=${BATCH_SIZES[$BATCH_IDX]}
CONDITION=${CONDITIONS[$COND_IDX]}
BALANCED=false
if [[ "${CONDITION}" == "conditioned_dual" ]]; then
  BALANCED=true
fi
PER_DEVICE_BATCH_SIZE="${BATCH_SIZE}"
GRAD_ACCUM_VALUE="${GRAD_ACCUM:-1}"
if [[ "${BATCH_SIZE}" -ge 16 && "${BSZ16_USE_GRAD_ACCUM:-1}" == "1" ]]; then
  PER_DEVICE_BATCH_SIZE="${BSZ16_PER_DEVICE_BATCH_SIZE:-8}"
  GRAD_ACCUM_VALUE="${BSZ16_GRAD_ACCUM:-2}"
fi

TAG="batch_bsz${BATCH_SIZE}_${CONDITION}_train1to${TRAIN_MAX}"
RUN_NAME="sft_hfsa_${TAG}_10k_seed${SEED}"
GROUP="sft_hfsa_batch_size_ablation_20260603/${TAG}"
MAX_STEPS_VALUE="${MAX_STEPS:-10000}"
EVAL_STEPS_VALUE="${EVAL_STEPS:-$((MAX_STEPS_VALUE + 1))}"

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="${GROUP}"
export WANDB_RUN_GROUP="${GROUP}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

sleep $((SLURM_ARRAY_TASK_ID * ${STARTUP_STAGGER_SECONDS:-180} + RANDOM % ${STARTUP_JITTER_SECONDS:-90}))

if [[ -f "${WORK}/synthetic-RLVL/runs/${RUN_NAME}/final/adapter_config.json" && "${FORCE_SFT:-0}" != "1" ]]; then
  echo "final checkpoint exists for ${RUN_NAME}; skipping"
  exit 0
fi

python train_sft.py \
  --config-name sft_hard_fsa_schema_fixedtarget \
  seed="${SEED}" \
  run_name="${RUN_NAME}" \
  task.template="${CONDITION}" \
  task.train_max_step="${TRAIN_MAX}" \
  task.val_max_step="${ONLINE_VAL_MAX:-25}" \
  data.materialized.dataset_id="${DATASET_ID}" \
  data.materialized.train_subset="${TRAIN_SUBSET}" \
  data.train_samples="${TRAIN_SAMPLES:-50000}" \
  data.eval_samples="${DATA_EVAL_SAMPLES:-256}" \
  data.max_length="${SFT_MAX_LENGTH:-8192}" \
  train.per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  train.grad_accum="${GRAD_ACCUM_VALUE}" \
  train.lr="${LR:-1e-4}" \
  train.max_steps="${MAX_STEPS_VALUE}" \
  train.eval_steps="${EVAL_STEPS_VALUE}" \
  train.save_steps="${SAVE_STEPS:-5000}" \
  train.save_total_limit="${SAVE_TOTAL_LIMIT:-3}" \
  train.gradient_checkpointing="${GRADIENT_CHECKPOINTING:-true}" \
  train.resume_from_checkpoint="${SFT_RESUME_FROM_CHECKPOINT:-null}" \
  +train.balanced_modality_batches="${BALANCED}" \
  validation.step_max="${ONLINE_VAL_MAX:-25}" \
  validation.samples_per_step="${ONLINE_SAMPLES_PER_STEP:-4}" \
  validation.max_new_tokens="${ONLINE_MAX_NEW_TOKENS:-6144}"
