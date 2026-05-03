#!/usr/bin/env bash
# Build the hard-v2 materialized logic dataset locally, optionally push to HF.
# Submit:
#   sbatch scripts/slurm/jobs/build_materialized_hard_v2.slurm

#SBATCH --job-name=build_hard_v2_data
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
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

OUT_ROOT="${HARD_V2_DATA_ROOT:-${WORK}/synthetic-RLVL/datasets/materialized_logic_hard_v2}"

ARGS=(
  --output-root "${OUT_ROOT}"
  --train-up-to-5-rows "${TRAIN_UP_TO_5_ROWS:-1000000}"
  --train-up-to-10-rows "${TRAIN_UP_TO_10_ROWS:-1000000}"
  --val-rows-per-step "${VAL_ROWS_PER_STEP:-1000}"
  --seed "${DATASET_SEED:-3407}"
  --distractor-ratio "${DISTRACTOR_RATIO:-0.5}"
  --difficulty hard_v2
  --branching-factor "${BRANCHING_FACTOR:-4}"
  --decoy-chains "${DECOY_CHAINS:-3}"
  --near-miss-ratio "${NEAR_MISS_RATIO:-0.75}"
  --side-chain-depth "${SIDE_CHAIN_DEPTH:-2}"
  --entity-decoy-ratio "${ENTITY_DECOY_RATIO:-1.0}"
  --answer-decoy-ratio "${ANSWER_DECOY_RATIO:-1.0}"
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
