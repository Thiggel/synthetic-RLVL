#!/usr/bin/env bash
# Targeted dense checkpoint eval for the matched seed-3407 train-1-to-25
# logic/NL pair. This avoids passing comma-separated CHECKPOINT_STEPS via
# sbatch --export, which Slurm treats as export separators.
#SBATCH --job-name=hfsa_t25_ckpt_eval
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-1%2
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

SEED="${SEED:-3407}"
TRAIN_MAX=25
TRAIN_SUBSET="train_fixedtarget_up_to_25_50k"
TEMPLATES=(logic nl_exact)
IDX=${SLURM_ARRAY_TASK_ID}
TEMPLATE=${TEMPLATES[$IDX]}
TAG="${TEMPLATE}_train1to${TRAIN_MAX}"
RUN_NAME="sft_hfsa_depth_scaling_${TAG}_10k_seed${SEED}"
GROUP="sft_hfsa_depth_scaling_intermediate_dense_train25/${TAG}"
DATASET_ID="${DATASET_ID:-flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget-depth50}"
RUN_DIR="${WORK}/synthetic-RLVL/runs/${RUN_NAME}"
MERGED_ROOT="${WORK}/synthetic-RLVL/tmp/train25_dense_eval_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OUT_DIR="${WORK}/synthetic-RLVL/passk_eval/${PASSK_OUTPUT_SUBDIR:-hfsa_depth_scaling_intermediate_sparse}"
CHECKPOINT_STEPS_VALUE="${CHECKPOINT_STEPS:-1000,2000,3000,4000,5000,6000,7000,8000,9000,10000}"
DEFAULT_PASSK_MAX_NEW_TOKENS=4096
if [[ "${TEMPLATE}" == "nl_exact" ]]; then
  DEFAULT_PASSK_MAX_NEW_TOKENS=6144
fi
PASSK_MAX_NEW_TOKENS_VALUE="${PASSK_MAX_NEW_TOKENS:-${DEFAULT_PASSK_MAX_NEW_TOKENS}}"
PASSK_VLLM_MAX_MODEL_LEN_VALUE="${PASSK_VLLM_MAX_MODEL_LEN:-16384}"
PASSK_STEPS_VALUE="${PASSK_STEPS:-1,5,10,15,20,25,30,40,50}"
DEFAULT_PASSK_STOP_STRINGS='</answer>'
PASSK_STOP_STRINGS_VALUE="${PASSK_STOP_STRINGS:-${DEFAULT_PASSK_STOP_STRINGS}}"

cleanup_merged() {
  rm -rf "${MERGED_ROOT}"
}
trap cleanup_merged EXIT

STAGGER_SECONDS="${PASSK_STAGGER_SECONDS:-60}"
JITTER_SECONDS="${PASSK_JITTER_SECONDS:-30}"
SLEEP_SECONDS=$((SLURM_ARRAY_TASK_ID * STAGGER_SECONDS))
if [[ "${JITTER_SECONDS}" -gt 0 ]]; then
  SLEEP_SECONDS=$((SLEEP_SECONDS + RANDOM % JITTER_SECONDS))
fi
sleep "${SLEEP_SECONDS}"
mkdir -p "${MERGED_ROOT}" "${OUT_DIR}"

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="${GROUP}"
export WANDB_RUN_GROUP="${GROUP}"

SKIP_GREEDY_ARGS=()
case "${PASSK_SKIP_GREEDY:-1}" in
  1|true|TRUE|yes|YES) SKIP_GREEDY_ARGS=(--skip-greedy) ;;
esac

STOP_STRING_ARGS=()
if [[ -n "${PASSK_STOP_STRINGS_VALUE}" ]]; then
  STOP_STRING_ARGS=(--stop-strings "${PASSK_STOP_STRINGS_VALUE}")
fi

IFS=',' read -r -a CKPT_STEPS <<< "${CHECKPOINT_STEPS_VALUE}"
for RAW_STEP in "${CKPT_STEPS[@]}"; do
  STEP="$(echo "${RAW_STEP}" | xargs)"
  [[ -z "${STEP}" ]] && continue
  CKPT_DIR="${RUN_DIR}/checkpoint-${STEP}"
  CKPT_TAG="checkpoint-${STEP}"
  if [[ ! -f "${CKPT_DIR}/adapter_config.json" ]]; then
    if [[ "${SKIP_MISSING_CHECKPOINTS:-1}" == "1" ]]; then
      echo "missing checkpoint; skipping: ${CKPT_DIR}"
      continue
    fi
    echo "missing checkpoint: ${CKPT_DIR}" >&2
    exit 2
  fi

  OUT="${OUT_DIR}/${RUN_NAME}_${CKPT_TAG}_passk.json"
  SAMPLES_OUT="${OUT_DIR}/${RUN_NAME}_${CKPT_TAG}_samples.jsonl"
  if [[ -s "${OUT}" && "${FORCE_PASSK_EVAL:-0}" != "1" ]]; then
    echo "output already exists: ${OUT}; skipping"
    continue
  fi

  MERGED="${MERGED_ROOT}/merged_${RUN_NAME}_${CKPT_TAG}"
  rm -rf "${MERGED}.tmp" "${MERGED}"
  python scripts/merge_lora_checkpoint.py --checkpoint "${CKPT_DIR}" --output "${MERGED}.tmp" --dtype bf16 --overwrite
  mv "${MERGED}.tmp" "${MERGED}"

  python scripts/evaluate_checkpoint_passk.py \
    --checkpoint "${MERGED}" \
    --config conf/sft_hard_fsa_schema_fixedtarget.yaml \
    --profile sft \
    --backend vllm \
    --step-min 1 \
    --step-max "${PASSK_STEP_MAX:-50}" \
    --steps "${PASSK_STEPS_VALUE}" \
    --samples-per-step "${PASSK_SAMPLES_PER_STEP:-16}" \
    --num-generations "${PASSK_NUM_GENERATIONS:-16}" \
    --k-values "${PASSK_K_VALUES:-1,2,4,8,16}" \
    --temperature "${PASSK_TEMPERATURE:-1.0}" \
    --max-new-tokens "${PASSK_MAX_NEW_TOKENS_VALUE}" \
    --batch-size "${PASSK_BATCH_SIZE:-128}" \
    --gpu-memory-utilization "${PASSK_GPU_MEMORY_UTILIZATION:-0.80}" \
    --vllm-max-model-len "${PASSK_VLLM_MAX_MODEL_LEN_VALUE}" \
    --scoring-log-interval "${PASSK_SCORING_LOG_INTERVAL:-100}" \
    --collect-samples "${PASSK_COLLECT_SAMPLES:-40}" \
    --output "${OUT}" \
    --samples-output "${SAMPLES_OUT}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${GROUP}" \
    --wandb-run-name "${RUN_NAME}_${CKPT_TAG}_passk" \
    "${SKIP_GREEDY_ARGS[@]}" \
    "${STOP_STRING_ARGS[@]}" \
    task.template="${TEMPLATE}" \
    task.train_max_step="${TRAIN_MAX}" \
    task.val_max_step="${PASSK_STEP_MAX:-50}" \
    data.materialized.dataset_id="${DATASET_ID}" \
    data.materialized.train_subset="${TRAIN_SUBSET}" \
    validation.max_new_tokens="${PASSK_MAX_NEW_TOKENS_VALUE}" \
    validation.vllm_max_model_len="${PASSK_VLLM_MAX_MODEL_LEN_VALUE}"

  rm -rf "${MERGED}"
done

cleanup_merged
trap - EXIT
