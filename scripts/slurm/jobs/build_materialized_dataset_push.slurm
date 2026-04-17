#!/usr/bin/env bash
#SBATCH --job-name=build_materialized_dataset_push
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

WORK_ROOT="${WORK:?WORK must be set}"
VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then
  VENV_PATH="${WORK_ROOT}/.venv"
fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

python -V

OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/synthetic-RLVL/datasets/materialized_logic}"
HF_REPO_ID="${HF_REPO_ID:?Set HF_REPO_ID, e.g. user/synthetic-rlvl-materialized-logic}"

python -m synthrlvl.datasets.materialize \
  --output-root "${OUTPUT_ROOT}" \
  --train-up-to-5-rows 1000000 \
  --train-up-to-10-rows 1000000 \
  --val-rows-per-step 1000 \
  --chunk-size 10000 \
  --push-to-hub \
  --hf-repo-id "${HF_REPO_ID}"
