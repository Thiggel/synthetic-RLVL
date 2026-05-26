#!/usr/bin/env bash
#SBATCH --job-name=pair_t10_eval
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs tmp
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then VENV_PATH="${WORK}/.venv"; fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

KINDS=(maze_navigation attribute_constraints)
TEMPLATES=(logic nl_exact)
IDX="${SLURM_ARRAY_TASK_ID}"
KIND="${KINDS[$((IDX / 2))]}"
TEMPLATE="${TEMPLATES[$((IDX % 2))]}"
SEED="${SEED:-3407}"
TRAIN_MAX=10
TRAIN_SUBSET="train_${KIND}_up_to_${TRAIN_MAX}_50k"
LOCAL_ROOT="${PAIRED_LOCAL_ROOT:-${WORK}/synthetic-RLVL/datasets/materialized_paired_${KIND}_train10_20260524}"
TAG="${KIND}_${TEMPLATE}_train1to${TRAIN_MAX}"
RUN_NAME="sft_paired_followup_${TAG}_10k_seed${SEED}"
GROUP="sft_paired_followup_eval/${KIND}/${TEMPLATE}_train1to${TRAIN_MAX}"
SFT_FINAL="${WORK}/synthetic-RLVL/runs/${RUN_NAME}/final"
MERGED="${WORK}/synthetic-RLVL/tmp/merged_${RUN_NAME}"
OUT_DIR="${WORK}/synthetic-RLVL/passk_eval/paired_followup_train10_sparse"
OUT="${OUT_DIR}/${RUN_NAME}_passk.json"
SAMPLES_OUT="${OUT_DIR}/${RUN_NAME}_samples.jsonl"

DEFAULT_PASSK_MAX_NEW_TOKENS=4096
if [[ "${TEMPLATE}" == "nl_exact" ]]; then
  DEFAULT_PASSK_MAX_NEW_TOKENS=6144
fi
DEFAULT_PASSK_BATCH_SIZE=128
DEFAULT_PASSK_VLLM_MAX_MODEL_LEN=16384
if [[ "${KIND}" == "maze_navigation" ]]; then
  DEFAULT_PASSK_BATCH_SIZE=64
  DEFAULT_PASSK_VLLM_MAX_MODEL_LEN=32768
fi
PASSK_MAX_NEW_TOKENS_VALUE="${PASSK_MAX_NEW_TOKENS:-${DEFAULT_PASSK_MAX_NEW_TOKENS}}"
PASSK_BATCH_SIZE_VALUE="${PASSK_BATCH_SIZE:-${DEFAULT_PASSK_BATCH_SIZE}}"
PASSK_VLLM_MAX_MODEL_LEN_VALUE="${PASSK_VLLM_MAX_MODEL_LEN:-${DEFAULT_PASSK_VLLM_MAX_MODEL_LEN}}"
PASSK_STEPS_VALUE="${PASSK_STEPS:-1,2,5,10,12,15,18,20,25,30,35,40,45,50}"

cleanup_merged() {
  rm -rf "${MERGED}.tmp"
  if [[ "${CLEANUP_MERGED_AFTER_EVAL:-1}" == "1" ]]; then
    rm -rf "${MERGED}"
  fi
}
trap cleanup_merged EXIT

PASSK_STAGGER_SECONDS_VALUE="${PASSK_STAGGER_SECONDS:-240}"
PASSK_JITTER_SECONDS_VALUE="${PASSK_JITTER_SECONDS:-90}"
if (( PASSK_JITTER_SECONDS_VALUE > 0 )); then
  STARTUP_SLEEP_SECONDS=$((SLURM_ARRAY_TASK_ID * PASSK_STAGGER_SECONDS_VALUE + RANDOM % PASSK_JITTER_SECONDS_VALUE))
else
  STARTUP_SLEEP_SECONDS=$((SLURM_ARRAY_TASK_ID * PASSK_STAGGER_SECONDS_VALUE))
fi
sleep "${STARTUP_SLEEP_SECONDS}"
mkdir -p "${OUT_DIR}"

if [[ ! -f "${SFT_FINAL}/adapter_config.json" ]]; then
  echo "missing SFT LoRA checkpoint: ${SFT_FINAL}" >&2
  exit 2
fi

if [[ -s "${OUT}" && "${FORCE_PASSK_EVAL:-0}" != "1" ]]; then
  echo "output already exists: ${OUT}; skipping eval"
  exit 0
fi

if [[ ! -f "${MERGED}/config.json" || "${FORCE_SFT_MERGE:-0}" == "1" ]]; then
  rm -rf "${MERGED}.tmp"
  python scripts/merge_lora_checkpoint.py --checkpoint "${SFT_FINAL}" --output "${MERGED}.tmp" --dtype bf16 --overwrite
  rm -rf "${MERGED}"
  mv "${MERGED}.tmp" "${MERGED}"
fi

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="${GROUP}"
export WANDB_RUN_GROUP="${GROUP}"

SKIP_GREEDY_ARGS=()
case "${PASSK_SKIP_GREEDY:-1}" in
  1|true|TRUE|yes|YES) SKIP_GREEDY_ARGS=(--skip-greedy) ;;
esac

python scripts/evaluate_checkpoint_passk.py \
  --checkpoint "${MERGED}" \
  --config conf/sft_hard_fsa_schema_fixedtarget.yaml \
  --profile sft \
  --backend vllm \
  --step-min 1 \
  --step-max "${PASSK_STEP_MAX:-50}" \
  --steps "${PASSK_STEPS_VALUE}" \
  --samples-per-step "${PASSK_SAMPLES_PER_STEP:-32}" \
  --num-generations "${PASSK_NUM_GENERATIONS:-16}" \
  --k-values "${PASSK_K_VALUES:-1,2,4,8,16}" \
  --temperature "${PASSK_TEMPERATURE:-1.0}" \
  --max-new-tokens "${PASSK_MAX_NEW_TOKENS_VALUE}" \
  --batch-size "${PASSK_BATCH_SIZE_VALUE}" \
  --gpu-memory-utilization "${PASSK_GPU_MEMORY_UTILIZATION:-0.80}" \
  --vllm-max-model-len "${PASSK_VLLM_MAX_MODEL_LEN_VALUE}" \
  --stop-strings "${PASSK_STOP_STRINGS:-</answer>}" \
  --scoring-log-interval "${PASSK_SCORING_LOG_INTERVAL:-100}" \
  --collect-samples "${PASSK_COLLECT_SAMPLES:-80}" \
  --output "${OUT}" \
  --samples-output "${SAMPLES_OUT}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${GROUP}" \
  --wandb-run-name "${RUN_NAME}_passk" \
  "${SKIP_GREEDY_ARGS[@]}" \
  task.difficulty="${KIND}" \
  task.template="${TEMPLATE}" \
  task.train_max_step="${TRAIN_MAX}" \
  task.val_max_step="${PASSK_STEP_MAX:-50}" \
  data.materialized.dataset_id=null \
  data.materialized.local_root="${LOCAL_ROOT}" \
  data.materialized.train_subset="${TRAIN_SUBSET}" \
  validation.max_new_tokens="${PASSK_MAX_NEW_TOKENS_VALUE}" \
  validation.vllm_max_model_len="${PASSK_VLLM_MAX_MODEL_LEN_VALUE}"

cleanup_merged
trap - EXIT
