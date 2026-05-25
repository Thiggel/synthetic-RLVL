#!/usr/bin/env bash
#SBATCH --job-name=eval_pt_llama
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
RUN_NAME="pretrain_hfsa_llama3_${SIZE}_${TEMPLATE}_train1to${TRAIN_MAX}_seed${SEED}"
CHECKPOINT="${WORK}/synthetic-RLVL/runs/${RUN_NAME}/final"
GROUP="pretrain_hfsa_llama3_scratch_eval/${SIZE}/${TEMPLATE}_train1to${TRAIN_MAX}"
OUT_DIR="${WORK}/synthetic-RLVL/passk_eval/hfsa_tiny_llama_pretrain_sparse"
OUT="${OUT_DIR}/${RUN_NAME}_passk.json"
SAMPLES_OUT="${OUT_DIR}/${RUN_NAME}_samples.jsonl"

mkdir -p "${OUT_DIR}"
if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  echo "missing pretrain checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi

if [[ -s "${OUT}" && "${FORCE_PASSK_EVAL:-0}" != "1" ]]; then
  echo "output already exists: ${OUT}; skipping eval"
  exit 0
fi

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="${GROUP}"
export WANDB_RUN_GROUP="${GROUP}"

SKIP_GREEDY_ARGS=()
case "${PASSK_SKIP_GREEDY:-1}" in
  1|true|TRUE|yes|YES) SKIP_GREEDY_ARGS=(--skip-greedy) ;;
esac

python scripts/evaluate_checkpoint_passk.py \
  --checkpoint "${CHECKPOINT}" \
  --config conf/sft_hard_fsa_schema_fixedtarget.yaml \
  --profile sft \
  --backend vllm \
  --step-min 1 \
  --step-max "${PASSK_STEP_MAX:-50}" \
  --steps "${PASSK_STEPS:-1,2,5,10,12,15,18,20,25,30,40,50}" \
  --samples-per-step "${PASSK_SAMPLES_PER_STEP:-16}" \
  --num-generations "${PASSK_NUM_GENERATIONS:-8}" \
  --k-values "${PASSK_K_VALUES:-1,2,4,8}" \
  --temperature "${PASSK_TEMPERATURE:-1.0}" \
  --max-new-tokens "${PASSK_MAX_NEW_TOKENS:-3072}" \
  --batch-size "${PASSK_BATCH_SIZE:-64}" \
  --gpu-memory-utilization "${PASSK_GPU_MEMORY_UTILIZATION:-0.80}" \
  --vllm-max-model-len "${PASSK_VLLM_MAX_MODEL_LEN:-8192}" \
  --stop-strings "${PASSK_STOP_STRINGS:-</answer>}" \
  --scoring-log-interval "${PASSK_SCORING_LOG_INTERVAL:-100}" \
  --collect-samples "${PASSK_COLLECT_SAMPLES:-80}" \
  --output "${OUT}" \
  --samples-output "${SAMPLES_OUT}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${GROUP}" \
  --wandb-run-name "${RUN_NAME}_passk" \
  "${SKIP_GREEDY_ARGS[@]}" \
  task.template="${TEMPLATE}" \
  task.train_max_step="${TRAIN_MAX}" \
  task.val_max_step="${PASSK_STEP_MAX:-50}" \
  data.materialized.dataset_id="${DATASET_ID:-flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget-depth50}" \
  data.materialized.train_subset="${TRAIN_SUBSET:-train_fixedtarget_up_to_10_50k}" \
  validation.max_new_tokens="${PASSK_MAX_NEW_TOKENS:-3072}" \
  validation.vllm_max_model_len="${PASSK_VLLM_MAX_MODEL_LEN:-8192}"
