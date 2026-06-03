#!/usr/bin/env bash
#SBATCH --job-name=hfsa_bsz_eval
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-15%4
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

# Match the SFT sweep: this ablation writes new outputs/caches on vault because
# the default $WORK filesystem currently rejects new directories for this user.
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
mkdir -p "${SYNTHRLVL_WORK_ROOT}" "${HF_HOME}" "${WANDB_DIR}" "${WANDB_ARTIFACT_DIR}" "${WANDB_DATA_DIR}"

SEED="${SEED:-3407}"
BATCH_SIZES=(2 4 8 16)
EVAL_CONDITIONS=(logic nl_exact conditioned_logic conditioned_nl)
TRAIN_MAX="${TRAIN_MAX:-20}"
TRAIN_SUBSET="${TRAIN_SUBSET:-train_fixedtarget_up_to_20_50k}"
DATASET_ID="${DATASET_ID:-flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget-depth50}"

IDX=${SLURM_ARRAY_TASK_ID}
BATCH_IDX=$((IDX % 4))
EVAL_IDX=$((IDX / 4))
BATCH_SIZE=${BATCH_SIZES[$BATCH_IDX]}
EVAL_CONDITION=${EVAL_CONDITIONS[$EVAL_IDX]}
if [[ "${EVAL_CONDITION}" == conditioned_* ]]; then
  TRAIN_CONDITION="conditioned_dual"
else
  TRAIN_CONDITION="${EVAL_CONDITION}"
fi

TAG="batch_bsz${BATCH_SIZE}_${TRAIN_CONDITION}_train1to${TRAIN_MAX}"
RUN_NAME="sft_hfsa_${TAG}_10k_seed${SEED}"
GROUP="sft_hfsa_batch_size_ablation_eval_20260603/${TAG}/${EVAL_CONDITION}"
SFT_FINAL="${WORK}/synthetic-RLVL/runs/${RUN_NAME}/final"
MERGED="${WORK}/synthetic-RLVL/tmp/merged_${RUN_NAME}"
OUT_DIR="${WORK}/synthetic-RLVL/passk_eval/${PASSK_OUTPUT_SUBDIR:-hfsa_batch_size_ablation_20260603}"
OUT="${OUT_DIR}/${RUN_NAME}_${EVAL_CONDITION}_passk.json"
SAMPLES_OUT="${OUT_DIR}/${RUN_NAME}_${EVAL_CONDITION}_samples.jsonl"

DEFAULT_PASSK_MAX_NEW_TOKENS=4096
if [[ "${EVAL_CONDITION}" == "nl_exact" || "${EVAL_CONDITION}" == "conditioned_nl" ]]; then
  DEFAULT_PASSK_MAX_NEW_TOKENS=6144
fi
PASSK_MAX_NEW_TOKENS_VALUE="${PASSK_MAX_NEW_TOKENS:-${DEFAULT_PASSK_MAX_NEW_TOKENS}}"
PASSK_VLLM_MAX_MODEL_LEN_VALUE="${PASSK_VLLM_MAX_MODEL_LEN:-16384}"
PASSK_STEPS_VALUE="${PASSK_STEPS:-1,2,5,10,12,15,18,20,25,30,35,40,45,50}"
DEFAULT_PASSK_STOP_STRINGS='</answer>'
PASSK_STOP_STRINGS_VALUE="${PASSK_STOP_STRINGS:-${DEFAULT_PASSK_STOP_STRINGS}}"

cleanup_merged() {
  rm -rf "${MERGED}.tmp"
  if [[ "${CLEANUP_MERGED_AFTER_EVAL:-1}" == "1" ]]; then
    rm -rf "${MERGED}"
  fi
}
trap cleanup_merged EXIT

sleep $((SLURM_ARRAY_TASK_ID * ${PASSK_STAGGER_SECONDS:-120} + RANDOM % ${PASSK_JITTER_SECONDS:-90}))
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

STOP_STRING_ARGS=()
if [[ -n "${PASSK_STOP_STRINGS_VALUE}" ]]; then
  STOP_STRING_ARGS=(--stop-strings "${PASSK_STOP_STRINGS_VALUE}")
fi

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
  --batch-size "${PASSK_BATCH_SIZE:-128}" \
  --gpu-memory-utilization "${PASSK_GPU_MEMORY_UTILIZATION:-0.80}" \
  --vllm-max-model-len "${PASSK_VLLM_MAX_MODEL_LEN_VALUE}" \
  --scoring-log-interval "${PASSK_SCORING_LOG_INTERVAL:-100}" \
  --collect-samples "${PASSK_COLLECT_SAMPLES:-80}" \
  --output "${OUT}" \
  --samples-output "${SAMPLES_OUT}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${GROUP}" \
  --wandb-run-name "${RUN_NAME}_${EVAL_CONDITION}_passk" \
  "${SKIP_GREEDY_ARGS[@]}" \
  "${STOP_STRING_ARGS[@]}" \
  task.template="${EVAL_CONDITION}" \
  task.train_max_step="${TRAIN_MAX}" \
  task.val_max_step="${PASSK_STEP_MAX:-50}" \
  data.materialized.dataset_id="${DATASET_ID}" \
  data.materialized.train_subset="${TRAIN_SUBSET}" \
  validation.max_new_tokens="${PASSK_MAX_NEW_TOKENS_VALUE}" \
  validation.vllm_max_model_len="${PASSK_VLLM_MAX_MODEL_LEN_VALUE}"

cleanup_merged
trap - EXIT
