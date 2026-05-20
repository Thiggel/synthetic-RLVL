#!/usr/bin/env bash
#SBATCH --job-name=hfsa_1k_eval
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-9%5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs passk_eval/hfsa_depth_scaling_1k tmp
source ./scripts/env.sh
VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then VENV_PATH="${WORK}/.venv"; fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

SEED="${SEED:-3407}"
TEMPLATES=(logic nl_exact)
TRAIN_MAXES=(5 10 15 20 25)
TRAIN_SUBSETS=(
  train_fixedtarget_up_to_5_50k
  train_fixedtarget_up_to_10_50k
  train_fixedtarget_up_to_15_50k
  train_fixedtarget_up_to_20_50k
  train_fixedtarget_up_to_25_50k
)
IDX=${SLURM_ARRAY_TASK_ID}
TEMPLATE_IDX=$((IDX / 5))
TRAIN_IDX=$((IDX % 5))
TEMPLATE=${TEMPLATES[$TEMPLATE_IDX]}
TRAIN_MAX=${TRAIN_MAXES[$TRAIN_IDX]}
TRAIN_SUBSET=${TRAIN_SUBSETS[$TRAIN_IDX]}
TAG="${TEMPLATE}_train1to${TRAIN_MAX}"
RUN_NAME="sft_hfsa_depth_scaling_${TAG}_1k_seed${SEED}"
GROUP="sft_hfsa_depth_scaling_1k_eval/${TAG}"
DATASET_ID="${DATASET_ID:-flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget-depth50}"
SFT_FINAL="${WORK}/synthetic-RLVL/runs/${RUN_NAME}/final"
MERGED="${WORK}/synthetic-RLVL/tmp/merged_${RUN_NAME}"
OUT="${WORK}/synthetic-RLVL/passk_eval/hfsa_depth_scaling_1k/${RUN_NAME}_passk.json"
SAMPLES_OUT="${WORK}/synthetic-RLVL/passk_eval/hfsa_depth_scaling_1k/${RUN_NAME}_samples.jsonl"

cleanup_merged() {
  rm -rf "${MERGED}.tmp"
  if [[ "${CLEANUP_MERGED_AFTER_EVAL:-1}" == "1" ]]; then
    rm -rf "${MERGED}"
  fi
}
trap cleanup_merged EXIT

sleep $((SLURM_ARRAY_TASK_ID * ${PASSK_STAGGER_SECONDS:-120} + RANDOM % ${PASSK_JITTER_SECONDS:-60}))

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

python scripts/evaluate_checkpoint_passk.py \
  --checkpoint "${MERGED}" \
  --config conf/sft_hard_fsa_schema_fixedtarget.yaml \
  --profile sft \
  --backend vllm \
  --step-min 1 \
  --step-max "${PASSK_STEP_MAX:-50}" \
  --samples-per-step "${PASSK_SAMPLES_PER_STEP:-64}" \
  --num-generations "${PASSK_NUM_GENERATIONS:-16}" \
  --k-values "${PASSK_K_VALUES:-1,2,4,8,16}" \
  --temperature "${PASSK_TEMPERATURE:-1.0}" \
  --max-new-tokens "${PASSK_MAX_NEW_TOKENS:-6144}" \
  --batch-size "${PASSK_BATCH_SIZE:-256}" \
  --gpu-memory-utilization "${PASSK_GPU_MEMORY_UTILIZATION:-0.80}" \
  --collect-samples "${PASSK_COLLECT_SAMPLES:-80}" \
  --output "${OUT}" \
  --samples-output "${SAMPLES_OUT}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${GROUP}" \
  --wandb-run-name "${RUN_NAME}_passk" \
  task.template="${TEMPLATE}" \
  task.train_max_step="${TRAIN_MAX}" \
  task.val_max_step="${PASSK_STEP_MAX:-50}" \
  data.materialized.dataset_id="${DATASET_ID}" \
  data.materialized.train_subset="${TRAIN_SUBSET}" \
  validation.max_new_tokens="${PASSK_MAX_NEW_TOKENS:-6144}"

cleanup_merged
trap - EXIT
