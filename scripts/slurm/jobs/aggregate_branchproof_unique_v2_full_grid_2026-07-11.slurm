#!/usr/bin/env bash
#SBATCH --job-name=agg_bp_full
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100small:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

EVAL_ARRAY_JOB_ID="${EVAL_ARRAY_JOB_ID:-3834582}"

VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

AUDIT_DIR="analysis/branchproof_unique_v2_full_grid_audits"
AUDIT_COUNT=$(find "${AUDIT_DIR}" -maxdepth 1 -type f -name '*.json' | wc -l)
if [[ "${AUDIT_COUNT}" -ne 30 ]]; then
  echo "Expected 30 accepted row audits, found ${AUDIT_COUNT}" >&2
  exit 2
fi

python scripts/analysis/aggregate_hfsa_depth_scaling.py \
  --final-dir "${HPCVAULT}/synthetic-RLVL/passk_eval/branchproof_unique_v2_20260710" \
  --skip-intermediate \
  --strict-final-grid \
  --out-dir analysis/branchproof_unique_v2_20260711

python scripts/analysis/audit_branchproof_unique_v2_qualitative_grid.py \
  --final-dir "${HPCVAULT}/synthetic-RLVL/passk_eval/branchproof_unique_v2_20260710" \
  --log-dir logs \
  --eval-array-job-id "${EVAL_ARRAY_JOB_ID}" \
  --output-json analysis/branchproof_unique_v2_20260711/qualitative_grid_audit.json \
  --output-markdown analysis/branchproof_unique_v2_20260711/qualitative_grid_audit.md
