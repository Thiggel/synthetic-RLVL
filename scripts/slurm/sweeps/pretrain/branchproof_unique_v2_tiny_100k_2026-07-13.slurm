#!/usr/bin/env bash
#SBATCH --job-name=pt_bp_v2_tiny
#SBATCH --partition=a40,a100
#SBATCH --exclude=a0934
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-17%3
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
hash -r 2>/dev/null || true

SIZES=(50m 100m 200m)
TEMPLATES=(logic nl_exact)
SEEDS=(3407 3408 3409)
IDX="${SLURM_ARRAY_TASK_ID}"
SEED="${SEEDS[$((IDX % 3))]}"
SIZE="${SIZES[$(((IDX / 3) % 3))]}"
TEMPLATE="${TEMPLATES[$((IDX / 9))]}"
TARGET_EXAMPLES=100000
EFFECTIVE_BATCH=16
TARGET_STEP=$((TARGET_EXAMPLES / EFFECTIVE_BATCH))
RUN_NAME="pretrain_branchproof_unique_v2_llama3_${SIZE}_${TEMPLATE}_train1to10_100kexamples_seed${SEED}"
OUTPUT_DIR="${HPCVAULT}/synthetic-RLVL/runs/${RUN_NAME}"
TARGET_CHECKPOINT="${OUTPUT_DIR}/checkpoint-${TARGET_STEP}/trainer_state.json"

if [[ -s "${TARGET_CHECKPOINT}" && -s "${OUTPUT_DIR}/final/config.json" && "${FORCE_PRETRAIN:-0}" != "1" ]]; then
  echo "Target checkpoint ${TARGET_STEP} exists for ${RUN_NAME}; skipping"
  exit 0
fi

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="branchproof_unique_v2_tiny_100kexamples/${SIZE}/${TEMPLATE}"
export WANDB_RUN_GROUP="${WANDB_GROUP}"

python scripts/train_tiny_llama_pretrain.py \
  --run-name "${RUN_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --size "${SIZE}" \
  --tokenizer NousResearch/Meta-Llama-3-8B \
  --dataset-id flaitenberger/BranchProof-unique-v2 \
  --train-subset train_fixedtarget_up_to_10_100k \
  --eval-subset val_step_10_1k \
  --template "${TEMPLATE}" \
  --train-max-step 10 \
  --train-samples "${TARGET_EXAMPLES}" \
  --eval-samples 512 \
  --max-length 4096 \
  --max-position-embeddings 16384 \
  --seed "${SEED}" \
  --max-steps "${TARGET_STEP}" \
  --warmup-steps 100 \
  --lr 3e-4 \
  --per-device-batch-size 2 \
  --grad-accum 8 \
  --eval-steps "$((TARGET_STEP + 1))" \
  --save-steps 1250 \
  --save-total-limit 6 \
  --logging-steps 100 \
  --require-unique-train-examples \
  --resume-from-checkpoint auto \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}"
