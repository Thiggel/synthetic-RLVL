#!/usr/bin/env bash
#SBATCH --job-name=build_bp_unique
#SBATCH --partition=a40,rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs tmp
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

OUT_ROOT="${OUT_ROOT:-${HPCVAULT}/synthetic-RLVL/datasets/branchproof_unique_v2_20260710}"
HF_REPO_ID="${HF_REPO_ID:-flaitenberger/BranchProof-unique-v2}"
DATASET_SEED="${DATASET_SEED:-3407}"
TRAIN_ROWS="${TRAIN_ROWS:-50000}"
VAL_ROWS_PER_STEP="${VAL_ROWS_PER_STEP:-1000}"
VAL_MAX_STEP="${VAL_MAX_STEP:-50}"

python scripts/analysis/probe_hard_fsa_schema.py \
  --train-shortcut-rate 0.0 \
  --eval-shortcut-rate 0.0 \
  --seed "${DATASET_SEED}" \
  --branching-factor 4 \
  --train-depth-max 25 \
  --eval-depth-max "${VAL_MAX_STEP}" \
  --n-per-depth "${PROBE_N_PER_DEPTH:-40}" \
  --output "tmp/branchproof_unique_v2_probe_${SLURM_JOB_ID}.json"

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
  --train-up-to-20-subset train_fixedtarget_up_to_20_50k \
  --train-up-to-25-subset train_fixedtarget_up_to_25_50k \
  --train-up-to-3-rows 0 \
  --train-up-to-5-rows "${TRAIN_ROWS}" \
  --train-up-to-10-rows "${TRAIN_ROWS}" \
  --train-up-to-15-rows "${TRAIN_ROWS}" \
  --train-up-to-20-rows "${TRAIN_ROWS}" \
  --train-up-to-25-rows "${TRAIN_ROWS}" \
  --val-rows-per-step "${VAL_ROWS_PER_STEP}" \
  --val-max-step "${VAL_MAX_STEP}" \
  --seed "${DATASET_SEED}" \
  --distractor-ratio 0.5 \
  --difficulty hard_fsa_schema \
  --branching-factor 4 \
  --train-shortcut-rate 0.0 \
  --val-shortcut-rate 0.0 \
  --chunk-size "${CHUNK_SIZE:-10000}" \
  "${PUSH_ARGS[@]}"

if [[ "${PUSH_TO_HUB:-1}" == "1" ]]; then
  HF_REPO_ID_CHECK="${HF_REPO_ID}" python - <<'PY'
import os
from datasets import load_dataset

repo = os.environ["HF_REPO_ID_CHECK"]
for subset in (
    "train_fixedtarget_up_to_5_50k",
    "train_fixedtarget_up_to_25_50k",
    "val_step_50_1k",
):
    row = load_dataset(repo, subset, split="train[:1]")[0]
    constants = list(row["constants"])
    assert len(constants) == int(row["depth"]) + 1
    assert constants[0].startswith("c0 = c0")
    print(subset, row["depth"], row["answer"], constants[0], constants[-1])
PY
fi
