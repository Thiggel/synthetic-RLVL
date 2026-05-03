#!/usr/bin/env bash
# Run constrained/rejection-sampling pass@k eval for selected reward checkpoints.
# Submit:
#   sbatch --array=0-5%1 scripts/slurm/jobs/reward_constrained_passk_eval_2026-04-27.slurm

#SBATCH --job-name=reward_constrained_passk
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --array=0-5%1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs passk_eval/reward_constrained tmp
source ./scripts/env.sh

WORK_ROOT="${WORK:?WORK must be set}"
VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then
  VENV_PATH="${WORK_ROOT}/.venv"
fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

MANIFEST="docs/manifests/reward_constrained_passk_eval_2026-04-27.tsv"
ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
if [[ -z "${ROW}" ]]; then
  echo "No manifest row for array task ${SLURM_ARRAY_TASK_ID}; exiting."
  exit 0
fi
IFS=$'\t' read -r SWEEP KIND VAL SEED RUN_NAME WANDB_GROUP_VALUE <<< "${ROW}"

MERGED="/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/merged_actor_${RUN_NAME}_final"
if [[ ! -f "${MERGED}/config.json" ]]; then
  echo "[constrained-passk] missing merged checkpoint: ${MERGED}" >&2
  exit 2
fi

OUT="/home/atuin/c107fa/c107fa12/synthetic-RLVL/passk_eval/reward_constrained/${SWEEP}_${RUN_NAME}_constrained_N${CONSTRAINED_PASSK_CANDIDATES_PER_LINE:-8}_K${CONSTRAINED_PASSK_NUM_GENERATIONS:-8}.json"
SAMPLES_OUT="/home/atuin/c107fa/c107fa12/synthetic-RLVL/passk_eval/reward_constrained/${SWEEP}_${RUN_NAME}_constrained_N${CONSTRAINED_PASSK_CANDIDATES_PER_LINE:-8}_K${CONSTRAINED_PASSK_NUM_GENERATIONS:-8}_samples.jsonl"

if [[ -s "${OUT}" && "${FORCE_CONSTRAINED_PASSK_EVAL:-0}" != "1" ]]; then
  echo "[constrained-passk] output already exists: ${OUT}; skipping eval."
  exit 0
fi

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="${WANDB_GROUP_VALUE}"
export WANDB_RUN_GROUP="${WANDB_GROUP_VALUE}"

echo "[constrained-passk] schema=${VAL} seed=${SEED} checkpoint=${MERGED}"
echo "[constrained-passk] output=${OUT}"

python scripts/evaluate_checkpoint_passk.py \
  --checkpoint "${MERGED}" \
  --config conf/posttrain_grpo.yaml \
  --profile grpo \
  --backend vllm \
  --step-min 1 \
  --step-max 20 \
  --samples-per-step "${PASSK_SAMPLES_PER_STEP:-20}" \
  --num-generations "${BASE_PASSK_NUM_GENERATIONS:-1}" \
  --k-values "${BASE_PASSK_K_VALUES:-1}" \
  --temperature 1.0 \
  --max-new-tokens 768 \
  --batch-size 1024 \
  --gpu-memory-utilization 0.80 \
  --disable-external \
  --collect-samples "${COLLECT_SAMPLES:-8}" \
  --constrained-enabled \
  --constrained-samples-per-step "${CONSTRAINED_PASSK_SAMPLES_PER_STEP:-4}" \
  --constrained-num-generations "${CONSTRAINED_PASSK_NUM_GENERATIONS:-8}" \
  --constrained-candidates-per-line "${CONSTRAINED_PASSK_CANDIDATES_PER_LINE:-8}" \
  --constrained-k-values "${CONSTRAINED_PASSK_K_VALUES:-1,2,4,8}" \
  --constrained-max-lines "${CONSTRAINED_PASSK_MAX_LINES:-32}" \
  --constrained-max-line-tokens "${CONSTRAINED_PASSK_MAX_LINE_TOKENS:-48}" \
  --constrained-suffix-max-tokens "${CONSTRAINED_PASSK_SUFFIX_MAX_TOKENS:-256}" \
  --constrained-temperature "${CONSTRAINED_PASSK_TEMPERATURE:-1.0}" \
  --output "${OUT}" \
  --samples-output "${SAMPLES_OUT}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP_VALUE}" \
  --wandb-run-name "${SWEEP}_${RUN_NAME}_constrained_N${CONSTRAINED_PASSK_CANDIDATES_PER_LINE:-8}_K${CONSTRAINED_PASSK_NUM_GENERATIONS:-8}"
