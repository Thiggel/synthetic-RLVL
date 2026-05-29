#!/usr/bin/env bash
#SBATCH --job-name=build_hfsa_shkind
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs tmp
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then VENV_PATH="${WORK}/.venv"; fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

IFS=',' read -r -a SHORTCUT_KINDS <<< "${SHORTCUT_KINDS_CSV:-position,initial_marker}"
IFS=',' read -r -a RATE_TAGS <<< "${RATE_TAGS_CSV:-0p5,0p8}"
IFS=',' read -r -a RATE_VALUES <<< "${RATE_VALUES_CSV:-0.5,0.8}"
if [[ "${#RATE_TAGS[@]}" -ne "${#RATE_VALUES[@]}" ]]; then
  echo "RATE_TAGS_CSV and RATE_VALUES_CSV must have the same length" >&2
  exit 2
fi
IDX=${SLURM_ARRAY_TASK_ID}
RATE_IDX=$((IDX % ${#RATE_TAGS[@]}))
KIND_IDX=$((IDX / ${#RATE_TAGS[@]}))
SHORTCUT_KIND=${SHORTCUT_KINDS[$KIND_IDX]}
RATE_TAG=${RATE_TAGS[$RATE_IDX]}
RATE_VALUE=${RATE_VALUES[$RATE_IDX]}

OUT_ROOT="${HFSA_SHORTCUT_DATA_ROOT:-${WORK}/synthetic-RLVL/datasets/materialized_hfsa_fixedtarget_depth50_shortcut_${SHORTCUT_KIND}_${RATE_TAG}_20260529}"
export OUT_ROOT
DATASET_SEED="${DATASET_SEED:-3407}"
BRANCHING_FACTOR="${BRANCHING_FACTOR:-4}"
TRAIN_ROWS="${TRAIN_ROWS:-50000}"
VAL_ROWS_PER_STEP="${VAL_ROWS_PER_STEP:-1000}"
VAL_MAX_STEP="${VAL_MAX_STEP:-50}"

python scripts/analysis/probe_hard_fsa_schema.py \
  --shortcut-kind "${SHORTCUT_KIND}" \
  --train-shortcut-rate "${RATE_VALUE}" \
  --eval-shortcut-rate 0.0 \
  --seed "${DATASET_SEED}" \
  --branching-factor "${BRANCHING_FACTOR}" \
  --train-depth-max 25 \
  --eval-depth-max "${VAL_MAX_STEP}" \
  --n-per-depth "${PROBE_N_PER_DEPTH:-40}" \
  --output "tmp/hard_fsa_schema_${SHORTCUT_KIND}_${RATE_TAG}_probe_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

python -m synthrlvl.datasets.materialize \
  --output-root "${OUT_ROOT}" \
  --train-up-to-3-subset train_fixedtarget_up_to_3_50k \
  --train-up-to-5-subset train_fixedtarget_up_to_5_50k \
  --train-up-to-10-subset train_fixedtarget_up_to_10_50k \
  --train-up-to-15-subset train_fixedtarget_up_to_15_50k \
  --train-up-to-20-subset train_fixedtarget_up_to_20_50k \
  --train-up-to-25-subset train_fixedtarget_up_to_25_50k \
  --train-up-to-3-rows 0 \
  --train-up-to-5-rows 0 \
  --train-up-to-10-rows 0 \
  --train-up-to-15-rows 0 \
  --train-up-to-20-rows 0 \
  --train-up-to-25-rows "${TRAIN_ROWS}" \
  --val-rows-per-step "${VAL_ROWS_PER_STEP}" \
  --val-max-step "${VAL_MAX_STEP}" \
  --seed "${DATASET_SEED}" \
  --distractor-ratio "${DISTRACTOR_RATIO:-0.5}" \
  --difficulty hard_fsa_schema \
  --branching-factor "${BRANCHING_FACTOR}" \
  --train-shortcut-rate "${RATE_VALUE}" \
  --val-shortcut-rate 0.0 \
  --shortcut-kind "${SHORTCUT_KIND}" \
  --chunk-size "${CHUNK_SIZE:-10000}"

python - <<'PY'
import os
from datasets import load_dataset

root = os.environ["OUT_ROOT"]
for subset in ["train_fixedtarget_up_to_25_50k", "val_step_50_1k"]:
    ds = load_dataset("parquet", data_files={"train": f"{root}/{subset}/train.parquet"}, split="train[:1]")
    row = ds[0]
    print(subset, row["depth"], row["answer"], row["metadata"])
PY
