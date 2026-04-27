#!/usr/bin/env bash
# 3-seed posttrain run for partial line-valid reward:
#   R = correctness + line_valid_fraction + 0.1 * format
#
# Submit:
#   sbatch scripts/slurm/jobs/posttrain_reward_line_valid_fraction_3seed_2026-04-22.slurm

#SBATCH --job-name=posttrain_rw_line_valid
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
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
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
SCHEMA="correct_plus_line_valid_plus_0p1_format"

# Use re-merged checkpoints already verified in prior sanity checks.
SFT_CKPT="/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/merged_sft_lr1e-4_seed${SEED}_remerge_20260421"
if [[ ! -f "${SFT_CKPT}/config.json" ]]; then
  echo "Missing merged checkpoint: ${SFT_CKPT}" >&2
  exit 1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="posttrain/reward_${SCHEMA}"
export WANDB_RUN_GROUP="${WANDB_GROUP}"

python posttrain_grpo_verl.py \
  run_name="rl_reward_${SCHEMA}_seed${SEED}_mrg" \
  seed="${SEED}" \
  model.path="${SFT_CKPT}" \
  model.merge_adapter_for_rollout=false \
  reward.schema="${SCHEMA}" \
  grpo.num_prompts=8 \
  grpo.num_rollouts=8 \
  grpo.max_num_batched_tokens=16384 \
  optim.micro_batch_size=4 \
  optim.logprob_micro_batch_size=4 \
  task.template=logic \
  task.prefill=none \
  task.train_min_step=1 \
  task.train_max_step=10 \
  task.val_min_step=1 \
  task.val_max_step=20
