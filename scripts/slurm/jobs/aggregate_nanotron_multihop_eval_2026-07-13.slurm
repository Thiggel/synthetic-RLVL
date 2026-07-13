#!/usr/bin/env bash
#SBATCH --job-name=agg_q25_mhop
#SBATCH --partition=a100mig
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

ROOT="${HPCVAULT}/synthetic-RLVL/lm_eval_results/qwen25_branchproof_unique_v2_multihop_20260713"
OUT="${HPCVAULT}/synthetic-RLVL/analysis/qwen25_branchproof_unique_v2_multihop_20260713"
python scripts/analysis/aggregate_nanotron_multihop_eval.py --root "${ROOT}" --output-dir "${OUT}"
