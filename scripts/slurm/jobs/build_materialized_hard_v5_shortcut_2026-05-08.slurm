#!/usr/bin/env bash
#SBATCH --job-name=build_hard_v5_shortcut
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then
  VENV_PATH="${WORK}/.venv"
fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

OUT_ROOT="${HARD_V5_DATA_ROOT:-${WORK}/synthetic-RLVL/datasets/materialized_logic_hard_v5_shortcut_20260508}"

ARGS=(
  --output-root "${OUT_ROOT}"
  --train-up-to-3-subset train_up_to_3_10k
  --train-up-to-15-subset train_up_to_15_50k
  --train-up-to-3-rows "${TRAIN_UP_TO_3_ROWS:-10000}"
  --train-up-to-5-rows 0
  --train-up-to-10-rows 0
  --train-up-to-15-rows "${TRAIN_UP_TO_15_ROWS:-50000}"
  --val-rows-per-step "${VAL_ROWS_PER_STEP:-1000}"
  --seed "${DATASET_SEED:-3407}"
  --distractor-ratio "${DISTRACTOR_RATIO:-0.5}"
  --difficulty hard_v5
  --train-shortcut-rate "${TRAIN_SHORTCUT_RATE:-0.8}"
  --val-shortcut-rate "${VAL_SHORTCUT_RATE:-0.0}"
  --chunk-size "${CHUNK_SIZE:-10000}"
)
if [[ -n "${PUSH_TO_HUB:-}" ]]; then
  ARGS+=(--push-to-hub)
fi
if [[ -n "${HF_REPO_ID:-}" ]]; then
  ARGS+=(--hf-repo-id "${HF_REPO_ID}")
fi
if [[ -n "${HF_PRIVATE:-}" ]]; then
  ARGS+=(--hf-private)
fi
python -m synthrlvl.datasets.materialize "${ARGS[@]}"
