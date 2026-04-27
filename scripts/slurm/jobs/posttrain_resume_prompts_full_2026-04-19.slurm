#!/usr/bin/env bash
# Auto-resume safety net for merged prompts ablation.
# Submit with:
#   sbatch --dependency=afterany:<base_job_id> scripts/slurm/jobs/posttrain_resume_prompts_full_2026-04-19.slurm

#SBATCH --job-name=posttrain_resume_prompts
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

WORK_ROOT="${WORK:?WORK must be set}"
VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then
  VENV_PATH="${WORK_ROOT}/.venv"
fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

SEEDS=(3407 3408 3409)
PROMPTS=(8 16 32 64)
SEED_IDX=$((SLURM_ARRAY_TASK_ID / ${#PROMPTS[@]}))
PROMPT_IDX=$((SLURM_ARRAY_TASK_ID % ${#PROMPTS[@]}))
SEED="${SEEDS[$SEED_IDX]}"
NUM_PROMPTS="${PROMPTS[$PROMPT_IDX]}"
SFT_CKPT="${WORK_ROOT}/synthetic-RLVL/tmp/merged_sft_lr1e-4_seed${SEED}"
if [[ ! -f "${SFT_CKPT}/config.json" ]]; then
  echo "Missing pre-merged checkpoint: ${SFT_CKPT}" >&2
  exit 1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="posttrain/grpo_np${NUM_PROMPTS}_nr8"
export WANDB_RUN_GROUP="${WANDB_GROUP}"

python posttrain_grpo_verl.py \
  run_name="rl_grpo_merge_ablate_prompts${NUM_PROMPTS}_rollouts8_seed${SEED}" \
  seed="${SEED}" \
  model.path="${SFT_CKPT}" \
  model.merge_adapter_for_rollout=false \
  task.template=logic \
  task.prefill=none \
  reward.schema=correct_plus_valid_plus_0p1_format \
  grpo.max_num_batched_tokens=16384 \
  optim.micro_batch_size=4 \
  optim.logprob_micro_batch_size=4 \
  grpo.num_prompts="${NUM_PROMPTS}" \
  grpo.num_rollouts=8 \
  resume.mode=auto
