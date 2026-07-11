#!/usr/bin/env bash
#SBATCH --job-name=audit_bp_full
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100small:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --array=0-29%8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs analysis/branchproof_unique_v2_full_grid_audits
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

SEEDS=(3407 3408 3409)
TEMPLATES=(logic nl_exact)
TRAIN_MAXES=(5 10 15 20 25)
IDX="${SLURM_ARRAY_TASK_ID}"
SEED_IDX=$((IDX % 3))
TRAIN_IDX=$(((IDX / 3) % 5))
TEMPLATE_IDX=$((IDX / 15))
SEED="${SEEDS[$SEED_IDX]}"
TEMPLATE="${TEMPLATES[$TEMPLATE_IDX]}"
TRAIN_MAX="${TRAIN_MAXES[$TRAIN_IDX]}"
RUN_NAME="sft_branchproof_unique_v2_${TEMPLATE}_train1to${TRAIN_MAX}_10k_seed${SEED}"
OUT_DIR="${HPCVAULT}/synthetic-RLVL/passk_eval/branchproof_unique_v2_20260710"
EVAL_ARRAY_JOB_ID="${EVAL_ARRAY_JOB_ID:-3834582}"
CONSTANT_ARGS=()
if [[ "${TEMPLATE}" == "nl_exact" ]]; then
  CONSTANT_ARGS=(--skip-fresh-constant-check)
fi

python scripts/analysis/audit_branchproof_unique_v2_pilot_eval.py \
  --metrics "${OUT_DIR}/${RUN_NAME}_passk.json" \
  --samples "${OUT_DIR}/${RUN_NAME}_samples.jsonl" \
  --output "analysis/branchproof_unique_v2_full_grid_audits/${RUN_NAME}.json" \
  --steps 1,2,5,10,12,15,18,20,25,30,35,40,45,50 \
  --k-values 1,2,4,8,16 \
  --samples-per-step 32 \
  --generations-per-prompt 16 \
  --expected-retained-samples 896 \
  --expected-total-samples 1024 \
  --sample-source-filter synthetic_sampled \
  --expected-sample-source synthetic_sampled \
  --expected-samples-per-step 64 \
  --expected-unique-prompts-per-step 32 \
  --expected-sample-indices 0,1 \
  --eval-log "logs/eval_bp_unique_${EVAL_ARRAY_JOB_ID}_${IDX}.out" \
  --batch-size 64 \
  --generation-cap 7168 \
  "${CONSTANT_ARGS[@]}"
