#!/usr/bin/env bash
#SBATCH --job-name=build_pair_full
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
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

FAMILIES=(official_igsm maze_navigation attribute_constraints_hard)
BUILD_KINDS=(official_igsm maze_navigation attribute_constraints)
IDX="${SLURM_ARRAY_TASK_ID}"
FAMILY="${FAMILIES[${IDX}]}"
BUILD_KIND="${BUILD_KINDS[${IDX}]}"
OUT_ROOT="${PAIRED_FULL_OUT_ROOT:-${WORK}/synthetic-RLVL/datasets/materialized_paired_${FAMILY}_full_20260528}"
TRAIN_ROWS="${TRAIN_ROWS:-50000}"
VAL_ROWS_PER_DEPTH="${VAL_ROWS_PER_DEPTH:-1000}"
VAL_MAX_DEPTH="${VAL_MAX_DEPTH:-50}"
VALIDATE_EXAMPLES="${VALIDATE_EXAMPLES:--1}"
CHUNK_SIZE="${CHUNK_SIZE:-10000}"
SEED="${DATASET_SEED:-3407}"
TRAIN_MAXES=(25 5 10 15 20)

for TRAIN_MAX in "${TRAIN_MAXES[@]}"; do
  SUBSET="train_${FAMILY}_up_to_${TRAIN_MAX}_50k"
  THIS_VAL_MAX=0
  THIS_VAL_ROWS=0
  if [[ "${TRAIN_MAX}" == "25" ]]; then
    THIS_VAL_MAX="${VAL_MAX_DEPTH}"
    THIS_VAL_ROWS="${VAL_ROWS_PER_DEPTH}"
  fi

  python scripts/data/build_paired_synthetic_dataset.py \
    --kind "${BUILD_KIND}" \
    --output-root "${OUT_ROOT}" \
    --train-subset "${SUBSET}" \
    --train-rows "${TRAIN_ROWS}" \
    --train-max-depth "${TRAIN_MAX}" \
    --val-rows-per-depth "${THIS_VAL_ROWS}" \
    --val-max-depth "${THIS_VAL_MAX}" \
    --validate-examples "${VALIDATE_EXAMPLES}" \
    --chunk-size "${CHUNK_SIZE}" \
    --seed "${SEED}"
done

python - "${OUT_ROOT}" "${FAMILY}" "${BUILD_KIND}" "${TRAIN_ROWS}" "${VAL_ROWS_PER_DEPTH}" "${VAL_MAX_DEPTH}" "${SEED}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
family, build_kind = sys.argv[2], sys.argv[3]
train_rows, val_rows, val_max, seed = map(int, sys.argv[4:8])
train_maxes = [5, 10, 15, 20, 25]
subsets = []
for train_max in train_maxes:
    path = root / f"train_{family}_up_to_{train_max}_50k" / "train.parquet"
    subsets.append({"subset": f"train_{family}_up_to_{train_max}_50k", "rows": train_rows, "path": str(path)})
for step in range(1, val_max + 1):
    path = root / f"val_step_{step:02d}_1k" / "train.parquet"
    subsets.append({"subset": f"val_step_{step:02d}_1k", "rows": val_rows, "path": str(path)})
manifest = {
    "family": family,
    "build_kind": build_kind,
    "train_rows_per_subset": train_rows,
    "train_max_depths": train_maxes,
    "val_rows_per_depth": val_rows,
    "val_max_depth": val_max,
    "seed": seed,
    "subsets": subsets,
}
(root / "full_suite_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2), flush=True)
PY
