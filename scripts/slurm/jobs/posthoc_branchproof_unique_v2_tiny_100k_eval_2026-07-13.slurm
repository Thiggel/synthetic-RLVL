#!/usr/bin/env bash
#SBATCH --job-name=eval_bp_v2_tiny
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --exclude=a0934
#SBATCH --gres=gpu:a100:1
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
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
hash -r 2>/dev/null || true

SIZES=(50m 100m 200m)
TEMPLATES=(logic nl_exact)
SEEDS=(3407 3408 3409)
EVAL_KIND="${TINY_EVAL_KIND:-final}"
IDX="${SLURM_ARRAY_TASK_ID}"
if [[ "${EVAL_KIND}" == "checkpoint" ]]; then
  BASE_IDX=$((IDX / 5))
  CHECKPOINT_STEPS=(1250 2500 3750 5000 6250)
  CHECKPOINT_STEP="${CHECKPOINT_STEPS[$((IDX % 5))]}"
else
  BASE_IDX="${IDX}"
  CHECKPOINT_STEP=""
fi
SEED="${SEEDS[$((BASE_IDX % 3))]}"
SIZE="${SIZES[$(((BASE_IDX / 3) % 3))]}"
TEMPLATE="${TEMPLATES[$((BASE_IDX / 9))]}"
RUN_NAME="pretrain_branchproof_unique_v2_llama3_${SIZE}_${TEMPLATE}_train1to10_100kexamples_seed${SEED}"
RUN_ROOT="${HPCVAULT}/synthetic-RLVL/runs/${RUN_NAME}"
if [[ "${EVAL_KIND}" == "checkpoint" ]]; then
  CHECKPOINT="${RUN_ROOT}/checkpoint-${CHECKPOINT_STEP}"
  OUTPUT_TAG="${RUN_NAME}_checkpoint-${CHECKPOINT_STEP}"
else
  CHECKPOINT="${RUN_ROOT}/final"
  OUTPUT_TAG="${RUN_NAME}"
fi
OUT_DIR="${HPCVAULT}/synthetic-RLVL/passk_eval/branchproof_unique_v2_tiny_100kexamples_20260713/${EVAL_KIND}"
OUT="${OUT_DIR}/${OUTPUT_TAG}_passk.json"
SAMPLES_OUT="${OUT_DIR}/${OUTPUT_TAG}_samples.jsonl"

if [[ ! -s "${CHECKPOINT}/config.json" ]]; then
  echo "Missing tiny checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi
if [[ -s "${OUT}" && "${FORCE_PASSK_EVAL:-0}" != "1" ]]; then
  echo "Output exists: ${OUT}; skipping"
  exit 0
fi
mkdir -p "${OUT_DIR}"

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="branchproof_unique_v2_tiny_eval/${SIZE}/${TEMPLATE}/${EVAL_KIND}"
export WANDB_RUN_GROUP="${WANDB_GROUP}"

python scripts/evaluate_checkpoint_passk.py \
  --checkpoint "${CHECKPOINT}" \
  --config conf/sft_hard_fsa_schema_fixedtarget.yaml \
  --profile sft \
  --backend vllm \
  --steps "${PASSK_STEPS:-1,2,5,10,12,15,18,20,25,30,40,50}" \
  --samples-per-step "${PASSK_SAMPLES_PER_STEP:-16}" \
  --num-generations "${PASSK_NUM_GENERATIONS:-8}" \
  --k-values "${PASSK_K_VALUES:-1,2,4,8}" \
  --temperature 1.0 \
  --max-new-tokens "${PASSK_MAX_NEW_TOKENS:-7168}" \
  --batch-size "${PASSK_BATCH_SIZE:-64}" \
  --gpu-memory-utilization 0.88 \
  --vllm-max-model-len 16384 \
  --stop-strings '</answer>' \
  --scoring-log-interval 100 \
  --collect-samples 96 \
  --collect-sampled-samples 192 \
  --output "${OUT}" \
  --samples-output "${SAMPLES_OUT}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-run-name "${OUTPUT_TAG}_passk" \
  task.template="${TEMPLATE}" \
  task.train_max_step=10 \
  task.val_max_step=50 \
  data.materialized.dataset_id=flaitenberger/BranchProof-unique-v2 \
  data.materialized.train_subset=train_fixedtarget_up_to_10_100k \
  validation.max_new_tokens="${PASSK_MAX_NEW_TOKENS:-7168}" \
  validation.vllm_max_model_len=16384
