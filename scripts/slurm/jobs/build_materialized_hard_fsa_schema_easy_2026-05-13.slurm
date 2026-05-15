#!/usr/bin/env bash
#SBATCH --job-name=build_hfsa_easy
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs tmp
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then
  VENV_PATH="${WORK}/.venv"
fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

OUT_ROOT="${HFSA_EASY_DATA_ROOT:-${WORK}/synthetic-RLVL/datasets/materialized_logic_hard_fsa_schema_easy_20260513}"
HF_REPO_ID="${HF_REPO_ID:-flaitenberger/LogicalReasoning-hard-fsa-schema-easy}"
DATASET_SEED="${DATASET_SEED:-3407}"
BRANCHING_FACTOR="${BRANCHING_FACTOR:-2}"
VAL_ROWS_PER_STEP="${VAL_ROWS_PER_STEP:-1000}"
TRAIN_ROWS="${TRAIN_ROWS:-50000}"

python scripts/analysis/probe_hard_fsa_schema.py \
  --train-shortcut-rate 0.0 \
  --eval-shortcut-rate 0.0 \
  --seed "${DATASET_SEED}" \
  --branching-factor "${BRANCHING_FACTOR}" \
  --train-depth-max 5 \
  --eval-depth-max 20 \
  --n-per-depth "${PROBE_N_PER_DEPTH:-100}" \
  --output "tmp/hard_fsa_schema_easy0p0_probe_${SLURM_JOB_ID}.json"

python scripts/analysis/probe_hard_fsa_schema.py \
  --train-shortcut-rate 0.5 \
  --eval-shortcut-rate 0.0 \
  --seed "${DATASET_SEED}" \
  --branching-factor "${BRANCHING_FACTOR}" \
  --train-depth-max 5 \
  --eval-depth-max 20 \
  --n-per-depth "${PROBE_N_PER_DEPTH:-100}" \
  --output "tmp/hard_fsa_schema_easy0p5_probe_${SLURM_JOB_ID}.json"

COMMON=(
  --output-root "${OUT_ROOT}"
  --train-up-to-3-rows 0
  --train-up-to-10-rows 0
  --train-up-to-15-rows 0
  --seed "${DATASET_SEED}"
  --distractor-ratio "${DISTRACTOR_RATIO:-0.5}"
  --difficulty hard_fsa_schema
  --branching-factor "${BRANCHING_FACTOR}"
  --val-shortcut-rate 0.0
  --chunk-size "${CHUNK_SIZE:-10000}"
)

PUSH_ARGS=()
if [[ "${PUSH_TO_HUB:-1}" == "1" ]]; then
  PUSH_ARGS+=(--push-to-hub --hf-repo-id "${HF_REPO_ID}")
  if [[ "${HF_PRIVATE:-1}" == "1" ]]; then
    PUSH_ARGS+=(--hf-private)
  fi
fi

python -m synthrlvl.datasets.materialize \
  "${COMMON[@]}" \
  --train-up-to-5-subset train_schema_easy0p0_up_to_5_50k \
  --train-up-to-5-rows "${TRAIN_ROWS}" \
  --val-rows-per-step "${VAL_ROWS_PER_STEP}" \
  --train-shortcut-rate 0.0 \
  "${PUSH_ARGS[@]}"

python -m synthrlvl.datasets.materialize \
  "${COMMON[@]}" \
  --train-up-to-5-subset train_schema_easy0p5_up_to_5_50k \
  --train-up-to-5-rows "${TRAIN_ROWS}" \
  --val-rows-per-step 0 \
  --train-shortcut-rate 0.5 \
  "${PUSH_ARGS[@]}"

python - <<'PY'
from datasets import load_dataset
for subset in ["train_schema_easy0p0_up_to_5_50k", "train_schema_easy0p5_up_to_5_50k", "val_step_05_1k", "val_step_20_1k"]:
    ds = load_dataset("flaitenberger/LogicalReasoning-hard-fsa-schema-easy", subset, split="train[:2]")
    print(subset, len(ds), ds[0]["metadata"])
PY
