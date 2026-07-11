#!/usr/bin/env bash
#SBATCH --job-name=eval_bp_unique
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-29%6
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
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
hash -r 2>/dev/null || true

SEEDS=(3407 3408 3409)
TEMPLATES=(logic nl_exact)
TRAIN_MAXES=(5 10 15 20 25)
TRAIN_SUBSETS=(
  train_fixedtarget_up_to_5_50k
  train_fixedtarget_up_to_10_50k
  train_fixedtarget_up_to_15_50k
  train_fixedtarget_up_to_20_50k
  train_fixedtarget_up_to_25_50k
)
IDX="${SLURM_ARRAY_TASK_ID}"
SEED_IDX=$((IDX % 3))
TRAIN_IDX=$(((IDX / 3) % 5))
TEMPLATE_IDX=$((IDX / 15))
SEED="${SEEDS[$SEED_IDX]}"
TEMPLATE="${TEMPLATES[$TEMPLATE_IDX]}"
TRAIN_MAX="${TRAIN_MAXES[$TRAIN_IDX]}"
TRAIN_SUBSET="${TRAIN_SUBSETS[$TRAIN_IDX]}"
RUN_NAME="sft_branchproof_unique_v2_${TEMPLATE}_train1to${TRAIN_MAX}_10k_seed${SEED}"
OUTPUT_SUFFIX="${PASSK_OUTPUT_SUFFIX:-}"
SFT_FINAL="${HPCVAULT}/synthetic-RLVL/runs/${RUN_NAME}/final"
MERGED_ROOT="${HPCVAULT}/synthetic-RLVL/tmp/branchproof_unique_v2_${RUN_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
MERGED="${MERGED_ROOT}/merged"
OUT_DIR="${HPCVAULT}/synthetic-RLVL/passk_eval/branchproof_unique_v2_20260710"
OUT="${OUT_DIR}/${RUN_NAME}${OUTPUT_SUFFIX}_passk.json"
SAMPLES_OUT="${OUT_DIR}/${RUN_NAME}${OUTPUT_SUFFIX}_samples.jsonl"
DATASET_ID="${DATASET_ID:-flaitenberger/BranchProof-unique-v2}"
PASSK_K_VALUES_RESOLVED="${PASSK_K_VALUES_COLON:-${PASSK_K_VALUES:-1,2,4,8,16}}"
PASSK_K_VALUES_RESOLVED="${PASSK_K_VALUES_RESOLVED//:/,}"
PASSK_STEPS_RESOLVED="${PASSK_STEPS_COLON:-${PASSK_STEPS:-1,2,5,10,12,15,18,20,25,30,35,40,45,50}}"
PASSK_STEPS_RESOLVED="${PASSK_STEPS_RESOLVED//:/,}"
SKIP_GREEDY_ARGS=()
case "${PASSK_SKIP_GREEDY:-0}" in
  1|true|TRUE|yes|YES) SKIP_GREEDY_ARGS=(--skip-greedy) ;;
esac

cleanup() {
  rm -rf "${MERGED_ROOT}"
}
trap cleanup EXIT

if [[ ! -f "${SFT_FINAL}/adapter_config.json" ]]; then
  echo "Missing SFT adapter: ${SFT_FINAL}" >&2
  exit 2
fi
if [[ -s "${OUT}" && "${FORCE_PASSK_EVAL:-0}" != "1" ]]; then
  echo "Output already exists: ${OUT}; skipping"
  exit 0
fi

mkdir -p "${MERGED_ROOT}" "${OUT_DIR}"
python scripts/merge_lora_checkpoint.py \
  --checkpoint "${SFT_FINAL}" \
  --output "${MERGED}.tmp" \
  --dtype bf16 \
  --overwrite
mv "${MERGED}.tmp" "${MERGED}"

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="branchproof_unique_v2_eval/${TEMPLATE}_train1to${TRAIN_MAX}"
export WANDB_RUN_GROUP="${WANDB_GROUP}"

python scripts/evaluate_checkpoint_passk.py \
  --checkpoint "${MERGED}" \
  --config conf/sft_hard_fsa_schema_fixedtarget.yaml \
  --profile sft \
  --backend vllm \
  --steps "${PASSK_STEPS_RESOLVED}" \
  --samples-per-step "${PASSK_SAMPLES_PER_STEP:-32}" \
  --num-generations "${PASSK_NUM_GENERATIONS:-16}" \
  --k-values "${PASSK_K_VALUES_RESOLVED}" \
  --temperature "${PASSK_TEMPERATURE:-1.0}" \
  --max-new-tokens "${PASSK_MAX_NEW_TOKENS:-7168}" \
  --batch-size "${PASSK_BATCH_SIZE:-64}" \
  --gpu-memory-utilization "${PASSK_GPU_MEMORY_UTILIZATION:-0.88}" \
  --vllm-max-model-len "${PASSK_VLLM_MAX_MODEL_LEN:-16384}" \
  --scoring-log-interval 100 \
  --collect-samples "${PASSK_COLLECT_SAMPLES:-128}" \
  --collect-sampled-samples "${PASSK_COLLECT_SAMPLED_SAMPLES:-0}" \
  "${SKIP_GREEDY_ARGS[@]}" \
  --stop-strings '</answer>' \
  --output "${OUT}" \
  --samples-output "${SAMPLES_OUT}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-run-name "${RUN_NAME}${OUTPUT_SUFFIX}_passk" \
  task.template="${TEMPLATE}" \
  task.train_max_step="${TRAIN_MAX}" \
  task.val_max_step=50 \
  data.materialized.dataset_id="${DATASET_ID}" \
  data.materialized.train_subset="${TRAIN_SUBSET}" \
  validation.max_new_tokens="${PASSK_MAX_NEW_TOKENS:-7168}" \
  validation.vllm_max_model_len="${PASSK_VLLM_MAX_MODEL_LEN:-16384}"

cleanup
trap - EXIT
