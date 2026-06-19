#!/usr/bin/env bash
#SBATCH --job-name=eval_hfsa_32b_t25
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --array=0-11%1
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

SEEDS=(3407 3408 3409)
TEMPLATES=(logic nl_exact)
MODEL_TAGS=(olmo3_1125_32b qwen3_32b)
TRAIN_MAX="${TRAIN_MAX:-25}"
TRAIN_SUBSET="${TRAIN_SUBSET:-train_fixedtarget_up_to_25_50k}"
DATASET_ID="${DATASET_ID:-flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget-depth50}"

IDX="${SLURM_ARRAY_TASK_ID}"
SEED="${SEEDS[$((IDX % 3))]}"
TEMPLATE="${TEMPLATES[$(((IDX / 3) % 2))]}"
MODEL_IDX=$((IDX / 6))
MODEL_TAG="${MODEL_TAGS[${MODEL_IDX}]}"
TAG="${TEMPLATE}_train1to${TRAIN_MAX}"
RUN_NAME="sft_hfsa_modelablate_${MODEL_TAG}_${TAG}_10k_seed${SEED}"
SFT_FINAL="${SFT_ROOT:-${HPCVAULT:-${WORK}}/synthetic-RLVL/runs}/${RUN_NAME}/final"
MERGE_WORK="${MODEL32_EVAL_MERGE_WORK:-${HPCVAULT:-${WORK}}}"
MERGED="${MERGE_WORK}/synthetic-RLVL/tmp/model32_eval_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${RUN_NAME}"
OUT_DIR="${PASSK_OUTPUT_ROOT:-${HPCVAULT:-${WORK}}/synthetic-RLVL/passk_eval/hfsa_model_ablation_32b_train25_20260619}"
OUT="${OUT_DIR}/${RUN_NAME}_passk.json"
SAMPLES_OUT="${OUT_DIR}/${RUN_NAME}_samples.jsonl"
GROUP="sft_hfsa_model_ablation_eval/${MODEL_TAG}/${TAG}"

DEFAULT_PASSK_MAX_NEW_TOKENS=4096
if [[ "${TEMPLATE}" == "nl_exact" ]]; then
  DEFAULT_PASSK_MAX_NEW_TOKENS=6144
fi
PASSK_MAX_NEW_TOKENS_VALUE="${PASSK_MAX_NEW_TOKENS:-${DEFAULT_PASSK_MAX_NEW_TOKENS}}"
PASSK_VLLM_MAX_MODEL_LEN_VALUE="${PASSK_VLLM_MAX_MODEL_LEN:-16384}"
PASSK_STEP_MAX_VALUE="${PASSK_STEP_MAX:-50}"
PASSK_STEPS_VALUE="${PASSK_STEPS:-1,2,5,10,12,15,18,20,25,30,35,40,45,50}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"

cleanup_merged() {
  rm -rf "${MERGED}.tmp"
  if [[ "${CLEANUP_MERGED_AFTER_EVAL:-1}" == "1" ]]; then
    rm -rf "${MERGED}"
  fi
}
trap cleanup_merged EXIT

sleep $((SLURM_ARRAY_TASK_ID * ${PASSK_STAGGER_SECONDS:-300} + RANDOM % ${PASSK_JITTER_SECONDS:-90}))
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
  --step-max "${PASSK_STEP_MAX_VALUE}" \
  --steps "${PASSK_STEPS_VALUE}" \
  --samples-per-step "${PASSK_SAMPLES_PER_STEP:-32}" \
  --num-generations "${PASSK_NUM_GENERATIONS:-16}" \
  --k-values "${PASSK_K_VALUES:-1,2,4,8,16}" \
  --temperature "${PASSK_TEMPERATURE:-1.0}" \
  --max-new-tokens "${PASSK_MAX_NEW_TOKENS_VALUE}" \
  --batch-size "${PASSK_BATCH_SIZE:-64}" \
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
  task.template="${TEMPLATE}" \
  task.train_max_step="${TRAIN_MAX}" \
  task.val_max_step="${PASSK_STEP_MAX_VALUE}" \
  data.materialized.dataset_id="${DATASET_ID}" \
  data.materialized.train_subset="${TRAIN_SUBSET}" \
  validation.max_new_tokens="${PASSK_MAX_NEW_TOKENS_VALUE}" \
  validation.vllm_max_model_len="${PASSK_VLLM_MAX_MODEL_LEN_VALUE}"

cleanup_merged
trap - EXIT
