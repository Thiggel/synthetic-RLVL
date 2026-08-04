#!/usr/bin/env bash
#SBATCH --job-name=bp_verify_select
#SBATCH --partition=a100mig
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

ROOT="${HPCVAULT}/synthetic-RLVL/passk_eval/branchproof_unique_v2_20260710"
SUFFIX="${PASSK_OUTPUT_SUFFIX:-_verifier_fullsamples}"
FILES=()
for template in logic nl_exact; do
  for seed in 3407 3408 3409; do
    file="${ROOT}/sft_branchproof_unique_v2_${template}_train1to25_10k_seed${seed}${SUFFIX}_samples.jsonl"
    test -s "${file}"
    FILES+=("${file}")
  done
done

VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
"${VENV_PATH}/bin/python" scripts/analysis/analyze_verifier_selection.py \
  --samples "${FILES[@]}" \
  --expected-k 16 \
  --train-max 25 \
  --output analysis/branchproof_unique_v2_20260711/verifier_selection_train25.json
