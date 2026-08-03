#!/usr/bin/env bash
#SBATCH --job-name=prep_dolci_sft
#SBATCH --partition=a100mig
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true
export HF_HUB_ENABLE_HF_TRANSFER=0
export WANDB_MODE=disabled

AUDIT_ROOT="${HPCVAULT}/synthetic-RLVL/analysis/dolci_control_sft_20260803"
PREPARED_DATASET="${HPCVAULT}/synthetic-RLVL/datasets/dolci_no_tools_single_turn_100k_seed3407_20260803"
mkdir -p "${AUDIT_ROOT}"
rm -rf "${PREPARED_DATASET}.tmp"

python scripts/train_instruction_sft.py \
  --model Qwen/Qwen2.5-7B \
  --dataset allenai/Dolci-Instruct-SFT-No-Tools \
  --dataset-revision 9156a5a542b2503100d4f1fabbf50be5eb25d977 \
  --train-split train \
  --eval-split train \
  --seed 3407 \
  --train-samples 100000 \
  --eval-samples 2048 \
  --max-length 4096 \
  --format-mode chat \
  --single-turn-only \
  --no-wrap-answer-tags \
  --prepared-dataset-output "${PREPARED_DATASET}.tmp" \
  --audit-output "${AUDIT_ROOT}/subset_and_tokenization_audit.json" \
  --dry-run

test -s "${AUDIT_ROOT}/subset_and_tokenization_audit.json"
rm -rf "${PREPARED_DATASET}"
mv "${PREPARED_DATASET}.tmp" "${PREPARED_DATASET}"
test -s "${PREPARED_DATASET}/dataset_dict.json"
