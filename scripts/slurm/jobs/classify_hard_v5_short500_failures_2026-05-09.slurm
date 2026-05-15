#!/usr/bin/env bash
#SBATCH --job-name=classify_hard_v5_short500
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --array=0-14%5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs analysis/hard_v5_short500_failures tmp
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then
  VENV_PATH="${WORK}/.venv"
fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

SEEDS=(3407 3408 3409)
SCHEMAS=(
  correct_plus_0p1_format
  correct_plus_citation_free_valid_plus_0p1_format
  correct_times_citation_free_valid_plus_0p1_format
  correct_plus_citation_free_line_valid_plus_0p1_format
  correct_times_citation_free_line_valid_plus_0p1_format
)

IDX=${SLURM_ARRAY_TASK_ID}
SEED=${SEEDS[$((IDX / ${#SCHEMAS[@]}))]}
SCHEMA=${SCHEMAS[$((IDX % ${#SCHEMAS[@]}))]}
RUN_NAME="rl_hard_v5_short500_${SCHEMA}_seed${SEED}_mrg"
RUN_ROOT="${WORK}/synthetic-RLVL/runs/${RUN_NAME}"

sleep $((SLURM_ARRAY_TASK_ID * ${CLASSIFY_STAGGER_SECONDS:-120} + RANDOM % ${CLASSIFY_JITTER_SECONDS:-30}))

ACTOR=""
for STEP in 501 500 250; do
  CANDIDATE="${RUN_ROOT}/global_step_${STEP}/actor"
  if [[ -f "${CANDIDATE}/lora_adapter/adapter_config.json" || -f "${CANDIDATE}/adapter_config.json" ]]; then
    ACTOR="${CANDIDATE}"
    break
  fi
done

if [[ -z "${ACTOR}" ]]; then
  echo "missing actor checkpoint for ${RUN_NAME}; skipping"
  exit 0
fi

MERGED="${WORK}/synthetic-RLVL/tmp/merged_actor_${RUN_NAME}_for_failure_classification"
OUT="${WORK}/synthetic-RLVL/analysis/hard_v5_short500_failures/${RUN_NAME}.json"

if [[ ! -f "${MERGED}/config.json" || "${FORCE_CLASSIFY_MERGE:-0}" == "1" ]]; then
  rm -rf "${MERGED}.tmp"
  python scripts/merge_lora_checkpoint.py --checkpoint "${ACTOR}" --output "${MERGED}.tmp" --dtype bf16 --overwrite
  rm -rf "${MERGED}"
  mv "${MERGED}.tmp" "${MERGED}"
fi

python scripts/analysis/classify_hard_v5_failures.py \
  --checkpoint "${MERGED}" \
  --config conf/posttrain_grpo_hard_v5_500.yaml \
  --step-min "${CLASSIFY_STEP_MIN:-4}" \
  --step-max "${CLASSIFY_STEP_MAX:-20}" \
  --samples-per-step "${CLASSIFY_SAMPLES_PER_STEP:-20}" \
  --max-new-tokens "${CLASSIFY_MAX_NEW_TOKENS:-1024}" \
  --output "${OUT}"
