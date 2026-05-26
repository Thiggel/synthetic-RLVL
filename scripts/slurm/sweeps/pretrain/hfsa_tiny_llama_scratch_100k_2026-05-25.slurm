#!/usr/bin/env bash
#SBATCH --job-name=pt_hfsa_llama100k
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-5%3
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

SIZES=(50m 100m 200m)
TEMPLATES=(logic nl_exact)
IDX="${SLURM_ARRAY_TASK_ID}"
SIZE="${SIZES[$((IDX % 3))]}"
TEMPLATE="${TEMPLATES[$((IDX / 3))]}"
SEED="${SEED:-3407}"
TRAIN_MAX="${TRAIN_MAX:-10}"
RUN_NAME="pretrain_hfsa_llama3_${SIZE}_${TEMPLATE}_train1to${TRAIN_MAX}_100k_seed${SEED}"
GROUP="pretrain_hfsa_llama3_scratch_100k/${SIZE}/${TEMPLATE}_train1to${TRAIN_MAX}"

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="${GROUP}"
export WANDB_RUN_GROUP="${GROUP}"

python scripts/train_tiny_llama_pretrain.py \
  --run-name "${RUN_NAME}" \
  --size "${SIZE}" \
  --tokenizer "${TOKENIZER_NAME:-NousResearch/Meta-Llama-3-8B}" \
  --dataset-id "${DATASET_ID:-flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget-depth50}" \
  --train-subset "${TRAIN_SUBSET:-train_fixedtarget_up_to_10_50k}" \
  --eval-subset "${EVAL_SUBSET:-val_step_10_1k}" \
  --template "${TEMPLATE}" \
  --train-max-step "${TRAIN_MAX}" \
  --train-samples "${TRAIN_SAMPLES:-50000}" \
  --eval-samples "${EVAL_SAMPLES:-512}" \
  --max-length "${MAX_LENGTH:-2048}" \
  --seed "${SEED}" \
  --max-steps "${MAX_STEPS:-100000}" \
  --warmup-steps "${WARMUP_STEPS:-1000}" \
  --lr "${LR:-3e-4}" \
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE:-2}" \
  --grad-accum "${GRAD_ACCUM:-8}" \
  --eval-steps "${EVAL_STEPS:-5000}" \
  --save-steps "${SAVE_STEPS:-20000}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT:-6}" \
  --logging-steps "${LOGGING_STEPS:-100}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${GROUP}"
