#!/usr/bin/env bash
#SBATCH --job-name=audit_bp_v2
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100small:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs tmp
source ./scripts/env.sh
source "${WORK}/nanotron/bin/activate"

DATA_ROOT="${DATA_ROOT:-${HPCVAULT}/synthetic-RLVL/nanotron_data/qwen25_branchproof_unique_v2}"
NANOSET_ROOT="${NANOSET_ROOT:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25_branchproof_unique_v2}"

python scripts/analysis/audit_branchproof_unique_v2_corpora.py \
  --data-root "${DATA_ROOT}" \
  --nanoset-root "${NANOSET_ROOT}" \
  --tokenizer "${MODEL_ID:-Qwen/Qwen2.5-7B}" \
  --full-pair-check \
  --output "analysis/branchproof_unique_v2_corpus_audit_2026-07-10.json"
