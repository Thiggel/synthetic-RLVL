#!/usr/bin/env bash
#SBATCH --job-name=build_maze_typed
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

EXP_WORK="${PAIRED_MAZE_TYPED_WORK:-${HPCVAULT:-${WORK}}}"
export WORK="${EXP_WORK}"
export SYNTHRLVL_WORK_ROOT="${WORK}/synthetic-RLVL"
export XDG_CACHE_HOME="${PAIRED_MAZE_TYPED_XDG_CACHE_HOME:-${WORK}/.cache_paired_maze_typed_20260603}"
export HF_HOME="${PAIRED_MAZE_TYPED_HF_HOME:-${XDG_CACHE_HOME}/hf}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export WANDB_DIR="${SYNTHRLVL_WORK_ROOT}/wandb"
export WANDB_ARTIFACT_DIR="${SYNTHRLVL_WORK_ROOT}/wandb_artifacts"
export WANDB_DATA_DIR="${SYNTHRLVL_WORK_ROOT}/wandb_data"
export WANDB_CACHE_DIR="${SYNTHRLVL_WORK_ROOT}/wandb_cache"
mkdir -p "${SYNTHRLVL_WORK_ROOT}" "${HF_HOME}" "${WANDB_DIR}" "${WANDB_ARTIFACT_DIR}" "${WANDB_DATA_DIR}" "${WANDB_CACHE_DIR}"

FAMILY="maze_navigation"
BUILD_KIND="maze_navigation"
OUT_ROOT="${PAIRED_MAZE_TYPED_OUT_ROOT:-${WORK}/synthetic-RLVL/datasets/materialized_paired_maze_navigation_typed_20260603}"
TRAIN_ROWS="${TRAIN_ROWS:-50000}"
VAL_ROWS_PER_DEPTH="${VAL_ROWS_PER_DEPTH:-1000}"
VAL_MAX_DEPTH="${VAL_MAX_DEPTH:-50}"
VALIDATE_EXAMPLES="${VALIDATE_EXAMPLES:--1}"
CHUNK_SIZE="${CHUNK_SIZE:-10000}"
SEED="${DATASET_SEED:-3407}"
TRAIN_MAXES=(25 5 10 15 20)

for TRAIN_MAX in "${TRAIN_MAXES[@]}"; do
  SUBSET="train_${FAMILY}_typed_up_to_${TRAIN_MAX}_50k"
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
    subset = f"train_{family}_typed_up_to_{train_max}_50k"
    subsets.append({"subset": subset, "rows": train_rows, "path": str(root / subset / "train.parquet")})
for step in range(1, val_max + 1):
    subset = f"val_step_{step:02d}_1k"
    subsets.append({"subset": subset, "rows": val_rows, "path": str(root / subset / "train.parquet")})
manifest = {
    "family": family,
    "build_kind": build_kind,
    "typed_symbols": True,
    "train_rows_per_subset": train_rows,
    "train_max_depths": train_maxes,
    "val_rows_per_depth": val_rows,
    "val_max_depth": val_max,
    "seed": seed,
    "subsets": subsets,
}
(root / "typed_maze_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2), flush=True)
PY
