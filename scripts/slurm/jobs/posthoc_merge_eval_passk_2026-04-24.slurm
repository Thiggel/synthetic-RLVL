#!/usr/bin/env bash
# Merge final VERL actor LoRA checkpoints and run full post-hoc pass@k evaluation.
# Submit example:
#   sbatch --dependency=afterany:<posttrain_jobids> --array=0-47%2 scripts/slurm/jobs/posthoc_merge_eval_passk_2026-04-24.slurm

#SBATCH --job-name=posthoc_passk_final
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --array=0-47%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs passk_eval tmp
source ./scripts/env.sh

WORK_ROOT="${WORK:?WORK must be set}"
VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then
  VENV_PATH="${WORK_ROOT}/.venv"
fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

MANIFEST="docs/manifests/passk_final_eval_2026-04-24.tsv"
ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
if [[ -z "${ROW}" ]]; then
  echo "No manifest row for array task ${SLURM_ARRAY_TASK_ID}; exiting."
  exit 0
fi
IFS=$'\t' read -r SWEEP KIND VAL SEED RUN_NAME WANDB_GROUP_VALUE <<< "${ROW}"

echo "[passk] row=${SLURM_ARRAY_TASK_ID} sweep=${SWEEP} kind=${KIND} value=${VAL} seed=${SEED} run=${RUN_NAME} group=${WANDB_GROUP_VALUE}"

RUN_ROOT="/home/atuin/c107fa/c107fa12/synthetic-RLVL/runs/${RUN_NAME}"
ACTOR=""
for STEP in 1501 1500; do
  CANDIDATE="${RUN_ROOT}/global_step_${STEP}/actor"
  if [[ -f "${CANDIDATE}/lora_adapter/adapter_config.json" || -f "${CANDIDATE}/adapter_config.json" ]]; then
    ACTOR="${CANDIDATE}"
    FINAL_STEP="${STEP}"
    break
  fi
done

if [[ -z "${ACTOR}" ]]; then
  echo "[passk] missing final actor for ${RUN_NAME}; expected global_step_1500 or 1501. Skipping for oversight/resubmission."
  exit 0
fi

MERGED="/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/merged_actor_${RUN_NAME}_final"
OUT="/home/atuin/c107fa/c107fa12/synthetic-RLVL/passk_eval/${SWEEP}_${RUN_NAME}_passk.json"
SAMPLES_OUT="/home/atuin/c107fa/c107fa12/synthetic-RLVL/passk_eval/${SWEEP}_${RUN_NAME}_samples.jsonl"

if [[ ! -f "${MERGED}/config.json" || "${FORCE_PASSK_MERGE:-0}" == "1" ]]; then
  rm -rf "${MERGED}.tmp"
  python scripts/merge_lora_checkpoint.py \
    --checkpoint "${ACTOR}" \
    --output "${MERGED}.tmp" \
    --dtype bf16 \
    --overwrite
  rm -rf "${MERGED}"
  mv "${MERGED}.tmp" "${MERGED}"
else
  echo "[passk] using existing merged actor ${MERGED}"
fi

if [[ -s "${OUT}" && "${FORCE_PASSK_EVAL:-0}" != "1" ]]; then
  echo "[passk] output already exists: ${OUT}; skipping eval."
  exit 0
fi

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="${WANDB_GROUP_VALUE}"
export WANDB_RUN_GROUP="${WANDB_GROUP_VALUE}"

EXTRA_PASSK_ARGS=()
if [[ "${CONSTRAINED_PASSK_ENABLED:-0}" == "1" ]]; then
  EXTRA_PASSK_ARGS+=(
    --constrained-enabled
    --constrained-samples-per-step "${CONSTRAINED_PASSK_SAMPLES_PER_STEP:-4}"
    --constrained-num-generations "${CONSTRAINED_PASSK_NUM_GENERATIONS:-8}"
    --constrained-candidates-per-line "${CONSTRAINED_PASSK_CANDIDATES_PER_LINE:-8}"
    --constrained-k-values "${CONSTRAINED_PASSK_K_VALUES:-1,2,4,8}"
    --constrained-max-lines "${CONSTRAINED_PASSK_MAX_LINES:-32}"
    --constrained-max-line-tokens "${CONSTRAINED_PASSK_MAX_LINE_TOKENS:-48}"
    --constrained-suffix-max-tokens "${CONSTRAINED_PASSK_SUFFIX_MAX_TOKENS:-256}"
    --constrained-temperature "${CONSTRAINED_PASSK_TEMPERATURE:-1.0}"
  )
fi

python scripts/evaluate_checkpoint_passk.py \
  --checkpoint "${MERGED}" \
  --config conf/posttrain_grpo.yaml \
  --profile grpo \
  --backend vllm \
  --step-min 1 \
  --step-max 20 \
  --samples-per-step 20 \
  --num-generations 64 \
  --k-values 1,2,4,8,16,32,64 \
  --temperature 1.0 \
  --max-new-tokens 768 \
  --batch-size 1024 \
  --gpu-memory-utilization 0.80 \
  --disable-external \
  --collect-samples 8 \
  --output "${OUT}" \
  --samples-output "${SAMPLES_OUT}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP_VALUE}" \
  --wandb-run-name "${SWEEP}_${RUN_NAME}_passk_step${FINAL_STEP}" \
  "${EXTRA_PASSK_ARGS[@]}"
