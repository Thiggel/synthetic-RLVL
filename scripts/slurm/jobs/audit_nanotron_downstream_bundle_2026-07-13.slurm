#!/usr/bin/env bash
#SBATCH --job-name=audit_nano_eval
#SBATCH --partition=a100mig
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

: "${RUN_DIR:?RUN_DIR is required}"
: "${MODE:?MODE is required}"

VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
TASKS="${REVIEWER_TASKS_COLON:-gsm8k:arc_challenge:hellaswag:winogrande:piqa:agieval_logiqa_en:hendrycks_math500:bbh:mmlu:mmlu_pro}"

"${VENV_PATH}/bin/python" scripts/analysis/audit_nanotron_downstream_eval.py \
  --run-dir "${RUN_DIR}" \
  --mode "${MODE}" \
  --tasks "${TASKS}"
