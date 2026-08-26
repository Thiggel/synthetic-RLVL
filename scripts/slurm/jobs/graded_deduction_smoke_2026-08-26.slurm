#!/usr/bin/env bash
#SBATCH --job-name=graded_deduction_smoke
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=a0532,a0934,a0534
#SBATCH --cpus-per-task=16
#SBATCH --time=00:59:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Smoke test for the graded deduction eval family (2026-08-26):
# 10 new tasks, --limit 8, one checkpoint.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"
source ./scripts/env.sh

CHECKPOINT="${HPCVAULT}/synthetic-RLVL/post_sft_reasoning_mixture_20260821/qwen25_7b_mixdepth_control_seed3407/final"
OUT_DIR="${HPCVAULT}/synthetic-RLVL/lm_eval_results/graded_deduction_smoke_20260826"

VENV_PATH="${HPCVAULT}/.venv_rlvl_posttrain"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

TASKS="synthrlvl_deduction_pw_d0,synthrlvl_deduction_pw_d1,synthrlvl_deduction_pw_d2,synthrlvl_deduction_pw_d3,synthrlvl_deduction_pw_d5,synthrlvl_deduction_bp_d5,synthrlvl_deduction_bp_d10,synthrlvl_deduction_bp_d15,synthrlvl_deduction_bp_d20,synthrlvl_deduction_bp_d25"

test -s "${CHECKPOINT}/config.json"
python scripts/validate_lm_eval_tasks.py --include-path lm_eval_tasks/synthrlvl_ood --tasks "${TASKS}"

python scripts/evaluate_lm_eval.py \
  --checkpoint "${CHECKPOINT}" \
  --model vllm \
  --tasks "${TASKS}" \
  --include-path lm_eval_tasks/synthrlvl_ood \
  --output-path "${OUT_DIR}" \
  --batch-size auto \
  --limit 8 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --model-arg "max_model_len=8192" \
  --trust-remote-code \
  --confirm-run-unsafe-code \
  --apply-chat-template \
  --log-samples
