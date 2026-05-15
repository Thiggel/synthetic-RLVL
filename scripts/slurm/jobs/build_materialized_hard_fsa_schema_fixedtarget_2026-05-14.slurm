#!/usr/bin/env bash
#SBATCH --job-name=build_hfsa_fix
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
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

OUT_ROOT="${HFSA_FIXEDTARGET_DATA_ROOT:-${WORK}/synthetic-RLVL/datasets/materialized_logic_hard_fsa_schema_fixedtarget_20260514}"
HF_REPO_ID="${HF_REPO_ID:-flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget}"
DATASET_SEED="${DATASET_SEED:-3407}"
BRANCHING_FACTOR="${BRANCHING_FACTOR:-4}"
TRAIN_ROWS="${TRAIN_ROWS:-50000}"
VAL_ROWS_PER_STEP="${VAL_ROWS_PER_STEP:-1000}"
VAL_MAX_STEP="${VAL_MAX_STEP:-25}"

python scripts/analysis/probe_hard_fsa_schema.py \
  --train-shortcut-rate 0.0 \
  --eval-shortcut-rate 0.0 \
  --seed "${DATASET_SEED}" \
  --branching-factor "${BRANCHING_FACTOR}" \
  --train-depth-max 15 \
  --eval-depth-max "${VAL_MAX_STEP}" \
  --n-per-depth "${PROBE_N_PER_DEPTH:-100}" \
  --output "tmp/hard_fsa_schema_fixedtarget_probe_${SLURM_JOB_ID}.json"

PUSH_ARGS=()
if [[ "${PUSH_TO_HUB:-1}" == "1" ]]; then
  PUSH_ARGS+=(--push-to-hub --hf-repo-id "${HF_REPO_ID}")
  if [[ "${HF_PRIVATE:-1}" == "1" ]]; then
    PUSH_ARGS+=(--hf-private)
  fi
fi

python -m synthrlvl.datasets.materialize \
  --output-root "${OUT_ROOT}" \
  --train-up-to-3-subset train_fixedtarget_up_to_3_50k \
  --train-up-to-5-subset train_fixedtarget_up_to_5_50k \
  --train-up-to-10-subset train_fixedtarget_up_to_10_50k \
  --train-up-to-15-subset train_fixedtarget_up_to_15_50k \
  --train-up-to-3-rows "${TRAIN_ROWS}" \
  --train-up-to-5-rows "${TRAIN_ROWS}" \
  --train-up-to-10-rows "${TRAIN_ROWS}" \
  --train-up-to-15-rows "${TRAIN_ROWS}" \
  --val-rows-per-step "${VAL_ROWS_PER_STEP}" \
  --val-max-step "${VAL_MAX_STEP}" \
  --seed "${DATASET_SEED}" \
  --distractor-ratio "${DISTRACTOR_RATIO:-0.5}" \
  --difficulty hard_fsa_schema \
  --branching-factor "${BRANCHING_FACTOR}" \
  --train-shortcut-rate 0.0 \
  --val-shortcut-rate 0.0 \
  --chunk-size "${CHUNK_SIZE:-10000}" \
  "${PUSH_ARGS[@]}"

python - <<'PY'
from datasets import load_dataset
repo = "flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget"
for subset in ["train_fixedtarget_up_to_3_50k", "train_fixedtarget_up_to_5_50k", "train_fixedtarget_up_to_10_50k", "train_fixedtarget_up_to_15_50k", "val_step_25_1k"]:
    ds = load_dataset(repo, subset, split="train[:1]")
    row = ds[0]
    print(subset, row["answer"], row["metadata"])
PY
