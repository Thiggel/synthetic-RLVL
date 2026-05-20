#!/usr/bin/env bash
#SBATCH --job-name=hfsa_dscale_ckpt_eval
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-29%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs passk_eval/hfsa_depth_scaling_intermediate tmp
source ./scripts/env.sh
VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then VENV_PATH="${WORK}/.venv"; fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
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
IDX=${SLURM_ARRAY_TASK_ID}
SEED_IDX=$((IDX % 3))
TRAIN_IDX=$(((IDX / 3) % 5))
TEMPLATE_IDX=$((IDX / 15))
SEED=${SEEDS[$SEED_IDX]}
TEMPLATE=${TEMPLATES[$TEMPLATE_IDX]}
TRAIN_MAX=${TRAIN_MAXES[$TRAIN_IDX]}
TRAIN_SUBSET=${TRAIN_SUBSETS[$TRAIN_IDX]}
TAG="${TEMPLATE}_train1to${TRAIN_MAX}"
RUN_NAME="sft_hfsa_depth_scaling_${TAG}_10k_seed${SEED}"
GROUP="sft_hfsa_depth_scaling_intermediate/${TAG}"
DATASET_ID="${DATASET_ID:-flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget-depth50}"
RUN_DIR="${WORK}/synthetic-RLVL/runs/${RUN_NAME}"
MERGED_ROOT="${WORK}/synthetic-RLVL/tmp/intermediate_eval_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OUT_DIR="${WORK}/synthetic-RLVL/passk_eval/hfsa_depth_scaling_intermediate"

cleanup_merged() {
  rm -rf "${MERGED_ROOT}"
}
trap cleanup_merged EXIT

sleep $((SLURM_ARRAY_TASK_ID * ${PASSK_STAGGER_SECONDS:-120} + RANDOM % ${PASSK_JITTER_SECONDS:-90}))
mkdir -p "${MERGED_ROOT}" "${OUT_DIR}"

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="${GROUP}"
export WANDB_RUN_GROUP="${GROUP}"

IFS=',' read -r -a CKPT_STEPS <<< "${CHECKPOINT_STEPS:-1000,3000,10000}"
for RAW_STEP in "${CKPT_STEPS[@]}"; do
  STEP="$(echo "${RAW_STEP}" | xargs)"
  [[ -z "${STEP}" ]] && continue
  if [[ "${STEP}" == "final" ]]; then
    CKPT_DIR="${RUN_DIR}/final"
    CKPT_TAG="final"
  else
    CKPT_DIR="${RUN_DIR}/checkpoint-${STEP}"
    CKPT_TAG="checkpoint-${STEP}"
  fi
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
    --samples-per-step "${PASSK_SAMPLES_PER_STEP:-64}" \
    --num-generations "${PASSK_NUM_GENERATIONS:-16}" \
    --k-values "${PASSK_K_VALUES:-1,2,4,8,16}" \
    --temperature "${PASSK_TEMPERATURE:-1.0}" \
    --max-new-tokens "${PASSK_MAX_NEW_TOKENS:-6144}" \
    --batch-size "${PASSK_BATCH_SIZE:-256}" \
    --gpu-memory-utilization "${PASSK_GPU_MEMORY_UTILIZATION:-0.80}" \
    --collect-samples "${PASSK_COLLECT_SAMPLES:-40}" \
    --output "${OUT}" \
    --samples-output "${SAMPLES_OUT}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${GROUP}" \
    --wandb-run-name "${RUN_NAME}_${CKPT_TAG}_passk" \
    task.template="${TEMPLATE}" \
    task.train_max_step="${TRAIN_MAX}" \
    task.val_max_step="${PASSK_STEP_MAX:-50}" \
    data.materialized.dataset_id="${DATASET_ID}" \
    data.materialized.train_subset="${TRAIN_SUBSET}" \
    validation.max_new_tokens="${PASSK_MAX_NEW_TOKENS:-6144}"

  rm -rf "${MERGED}"
done

cleanup_merged
trap - EXIT
