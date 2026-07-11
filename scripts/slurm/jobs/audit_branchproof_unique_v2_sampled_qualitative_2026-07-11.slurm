#!/usr/bin/env bash
#SBATCH --job-name=audit_bp_sampled
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

RUN_NAME="sft_branchproof_unique_v2_logic_train1to25_10k_seed3407"
OUT_DIR="${HPCVAULT}/synthetic-RLVL/passk_eval/branchproof_unique_v2_20260710"

python scripts/analysis/audit_branchproof_unique_v2_pilot_eval.py \
  --metrics "${OUT_DIR}/${RUN_NAME}_sampled_qualitative_passk.json" \
  --samples "${OUT_DIR}/${RUN_NAME}_sampled_qualitative_samples.jsonl" \
  --output analysis/branchproof_unique_v2_sampled_qualitative_audit_2026-07-11.json \
  --steps 1,25,30,50 \
  --k-values 1,2,4,8 \
  --samples-per-step 4 \
  --generations-per-prompt 8 \
  --expected-retained-samples 128 \
  --expected-sample-source synthetic_sampled \
  --expected-samples-per-step 32 \
  --expected-sample-indices 0,1,2,3,4,5,6,7 \
  --greedy-optional
