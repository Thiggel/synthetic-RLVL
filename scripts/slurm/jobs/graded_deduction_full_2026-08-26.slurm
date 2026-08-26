#!/usr/bin/env bash
#SBATCH --job-name=graded_deduction_full
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=a0532,a0934,a0534
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-9%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Full graded-deduction readout (2026-08-26): 15 tasks (10 direct + 5 BP CoT)
# over the 10 mixdepth SFT checkpoints, no --limit.
# Array map: idx % 5 -> condition, idx / 5 -> seed.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"
source ./scripts/env.sh

CONDITIONS=(control logic_band15 nl_exact_band15 logic_band25 nl_exact_band25)
SEEDS=(3407 3408)
CONDITION="${CONDITIONS[$((SLURM_ARRAY_TASK_ID % 5))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID / 5))]}"
RUN_NAME="qwen25_7b_mixdepth_${CONDITION}_seed${SEED}"
CHECKPOINT="${HPCVAULT}/synthetic-RLVL/post_sft_reasoning_mixture_20260821/${RUN_NAME}/final"
OUT_DIR="${HPCVAULT}/synthetic-RLVL/lm_eval_results/qwen25_mixdepth_graded_deduction_20260826/${RUN_NAME}"

VENV_PATH="${HPCVAULT}/.venv_rlvl_posttrain"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="qwen25_mixdepth_graded_deduction_20260826"
export WANDB_RUN_GROUP="${WANDB_GROUP}"

TASKS="synthrlvl_deduction_pw_d0,synthrlvl_deduction_pw_d1,synthrlvl_deduction_pw_d2,synthrlvl_deduction_pw_d3,synthrlvl_deduction_pw_d5,synthrlvl_deduction_bp_d5,synthrlvl_deduction_bp_d10,synthrlvl_deduction_bp_d15,synthrlvl_deduction_bp_d20,synthrlvl_deduction_bp_d25,synthrlvl_deduction_bp_cot_d5,synthrlvl_deduction_bp_cot_d10,synthrlvl_deduction_bp_cot_d15,synthrlvl_deduction_bp_cot_d20,synthrlvl_deduction_bp_cot_d25"

test -s "${CHECKPOINT}/config.json"
python scripts/validate_lm_eval_tasks.py --include-path lm_eval_tasks/synthrlvl_ood --tasks "${TASKS}"

python scripts/evaluate_lm_eval.py \
  --checkpoint "${CHECKPOINT}" \
  --model vllm \
  --tasks "${TASKS}" \
  --include-path lm_eval_tasks/synthrlvl_ood \
  --output-path "${OUT_DIR}" \
  --batch-size auto \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --model-arg "max_model_len=8192" \
  --trust-remote-code \
  --confirm-run-unsafe-code \
  --apply-chat-template \
  --log-samples \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-run-name "${RUN_NAME}_graded_deduction"
