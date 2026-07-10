#!/usr/bin/env bash
#SBATCH --job-name=audit_bp_eval
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100small:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs analysis
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

RUN_NAME="${RUN_NAME:-sft_branchproof_unique_v2_logic_train1to25_10k_seed3407}"
OUT_DIR="${OUT_DIR:-${HPCVAULT}/synthetic-RLVL/passk_eval/branchproof_unique_v2_20260710}"
PILOT_EVAL_JOB_ID="${PILOT_EVAL_JOB_ID:-3832945}"
PILOT_EVAL_TASK_ID="${PILOT_EVAL_TASK_ID:-12}"

python scripts/analysis/audit_branchproof_unique_v2_pilot_eval.py \
  --metrics "${OUT_DIR}/${RUN_NAME}_passk.json" \
  --samples "${OUT_DIR}/${RUN_NAME}_samples.jsonl" \
  --output analysis/branchproof_unique_v2_pilot_eval_audit_2026-07-10.json \
  --steps "${PILOT_AUDIT_STEPS:-1,2,5,10,12,15,18,20,25,30,35,40,45,50}" \
  --k-values "${PILOT_AUDIT_K_VALUES:-1,2,4,8}" \
  --samples-per-step "${PILOT_AUDIT_SAMPLES_PER_STEP:-16}" \
  --generations-per-prompt "${PILOT_AUDIT_GENERATIONS_PER_PROMPT:-8}" \
  --expected-retained-samples "${PILOT_AUDIT_RETAINED_SAMPLES:-128}" \
  --eval-log "${PILOT_EVAL_LOG:-logs/eval_bp_unique_${PILOT_EVAL_JOB_ID}_${PILOT_EVAL_TASK_ID}.out}" \
  --batch-size "${PILOT_AUDIT_BATCH_SIZE:-64}" \
  --generation-cap "${PILOT_AUDIT_GENERATION_CAP:-7168}"
