#!/usr/bin/env bash
#SBATCH --job-name=agg_nano_p15
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

VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"

python scripts/analysis/aggregate_nanotron_downstream_pilot.py \
  --root "${HPCVAULT}/synthetic-RLVL/lm_eval_results/qwen25_branchproof_unique_v2_pilot_20260710" \
  --output-dir analysis/nanotron_branchproof_unique_v2_p15_20260711 \
  --checkpoint-step 8192
