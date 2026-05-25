#!/usr/bin/env bash
#SBATCH --job-name=build_attr_hard
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs tmp
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then VENV_PATH="${WORK}/.venv"; fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

OUT_ROOT="${ATTRIBUTE_HARD_OUT_ROOT:-${WORK}/synthetic-RLVL/datasets/materialized_paired_attribute_constraints_hard_train10_20260525}"
TRAIN_ROWS="${TRAIN_ROWS:-50000}"
VAL_ROWS_PER_DEPTH="${VAL_ROWS_PER_DEPTH:-1000}"
VAL_MAX_DEPTH="${VAL_MAX_DEPTH:-50}"
VALIDATE_EXAMPLES="${VALIDATE_EXAMPLES:--1}"
CHUNK_SIZE="${CHUNK_SIZE:-10000}"
SEED="${DATASET_SEED:-3407}"

python scripts/data/build_paired_synthetic_dataset.py \
  --kind attribute_constraints \
  --output-root "${OUT_ROOT}" \
  --train-subset "train_attribute_constraints_hard_up_to_10_50k" \
  --train-rows "${TRAIN_ROWS}" \
  --train-max-depth 10 \
  --val-rows-per-depth "${VAL_ROWS_PER_DEPTH}" \
  --val-max-depth "${VAL_MAX_DEPTH}" \
  --validate-examples "${VALIDATE_EXAMPLES}" \
  --chunk-size "${CHUNK_SIZE}" \
  --seed "${SEED}"
