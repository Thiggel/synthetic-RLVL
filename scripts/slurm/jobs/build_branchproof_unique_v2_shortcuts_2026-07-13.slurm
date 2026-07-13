#!/usr/bin/env bash
#SBATCH --job-name=build_bp_v2_short
#SBATCH --partition=a100mig
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --array=0-6%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs tmp
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
hash -r 2>/dev/null || true

KINDS=(schema schema schema position position initial_marker initial_marker)
RATE_TAGS=(0p3 0p5 0p8 0p5 0p8 0p5 0p8)
RATE_VALUES=(0.3 0.5 0.8 0.5 0.8 0.5 0.8)
IDX="${SLURM_ARRAY_TASK_ID}"
SHORTCUT_KIND="${KINDS[$IDX]}"
RATE_TAG="${RATE_TAGS[$IDX]}"
RATE_VALUE="${RATE_VALUES[$IDX]}"
DATA_KIND="shortcut_${SHORTCUT_KIND}_${RATE_TAG}"
OUT_ROOT="${HPCVAULT}/synthetic-RLVL/datasets/branchproof_unique_v2_${DATA_KIND}_20260713"
PROBE_OUT="tmp/branchproof_unique_v2_${DATA_KIND}_probe_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

if [[ -s "${OUT_ROOT}/train_fixedtarget_up_to_25_50k/train.parquet" \
      && -s "${OUT_ROOT}/val_step_50_1k/train.parquet" \
      && "${FORCE_DATA_BUILD:-0}" != "1" ]]; then
  echo "Corrected shortcut dataset exists: ${OUT_ROOT}; skipping"
  exit 0
fi

python scripts/analysis/probe_hard_fsa_schema.py \
  --shortcut-kind "${SHORTCUT_KIND}" \
  --train-shortcut-rate "${RATE_VALUE}" \
  --eval-shortcut-rate 0.0 \
  --seed 3407 \
  --branching-factor 4 \
  --train-depth-max 25 \
  --eval-depth-max 50 \
  --n-per-depth "${PROBE_N_PER_DEPTH:-40}" \
  --output "${PROBE_OUT}"

rm -rf "${OUT_ROOT}.tmp"
python -m synthrlvl.datasets.materialize \
  --output-root "${OUT_ROOT}.tmp" \
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
  --train-up-to-25-rows "${TRAIN_ROWS:-50000}" \
  --val-rows-per-step "${VAL_ROWS_PER_STEP:-1000}" \
  --val-max-step 50 \
  --seed 3407 \
  --distractor-ratio 0.5 \
  --difficulty hard_fsa_schema \
  --branching-factor 4 \
  --train-shortcut-rate "${RATE_VALUE}" \
  --val-shortcut-rate 0.0 \
  --shortcut-kind "${SHORTCUT_KIND}" \
  --chunk-size 10000

python - "${OUT_ROOT}.tmp" <<'PY'
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
for relative, expected_rows in (
    ("train_fixedtarget_up_to_25_50k/train.parquet", 50_000),
    ("val_step_50_1k/train.parquet", 1_000),
):
    path = root / relative
    if not path.is_file() or pq.ParquetFile(path).metadata.num_rows != expected_rows:
        raise SystemExit(f"bad materialized artifact: {path}")
    row = pq.read_table(path).slice(0, 1).to_pylist()[0]
    metadata = row.get("metadata")
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if int(row["depth"]) > len(metadata.get("path_constants", [])) - 1:
        raise SystemExit(f"missing fresh path constants in {path}")
print("Shortcut materialization artifact gate passed")
PY

rm -rf "${OUT_ROOT}"
mv "${OUT_ROOT}.tmp" "${OUT_ROOT}"
