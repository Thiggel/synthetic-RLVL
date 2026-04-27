#!/usr/bin/env bash
#SBATCH --job-name=posttrain_reward_ablation
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-20
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
SCHEMAS=(
  correct_plus_0p1_format
  indicator_correct_and_format
  correct_plus_valid_plus_0p1_format
  correct_plus_0p75_valid_plus_0p1_format
  correct_plus_0p5_valid_plus_0p1_format
  correct_plus_0p25_valid_plus_0p1_format
  indicator_all
)

SEED_IDX=$((SLURM_ARRAY_TASK_ID / ${#SCHEMAS[@]}))
SCHEMA_IDX=$((SLURM_ARRAY_TASK_ID % ${#SCHEMAS[@]}))
SEED="${SEEDS[$SEED_IDX]}"
SCHEMA="${SCHEMAS[$SCHEMA_IDX]}"

MERGED_DIR="${WORK_ROOT}/synthetic-RLVL/tmp/merged_sft_lr1e-4_seed${SEED}"
if [[ ! -f "${MERGED_DIR}/config.json" ]]; then
  echo "Missing pre-merged checkpoint for seed ${SEED}: ${MERGED_DIR}" >&2
  echo "Please pre-merge SFT LoRA checkpoints before submitting this sweep." >&2
  exit 1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="posttrain/reward_${SCHEMA}"
export WANDB_RUN_GROUP="${WANDB_GROUP}"

python posttrain_grpo_verl.py \
  run_name="rl_reward_${SCHEMA}_seed${SEED}_mrg" \
  seed="${SEED}" \
  model.path="${MERGED_DIR}" \
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
