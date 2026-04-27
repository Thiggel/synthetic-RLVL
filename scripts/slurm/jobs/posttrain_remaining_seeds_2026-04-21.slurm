#!/usr/bin/env bash
#SBATCH --job-name=posttrain_remain_2seed
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-29
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

SEEDS=(3407 3408)
CFGS=(
  prompts:8
  prompts:16
  prompts:32
  prompts:64
  rollouts:8
  rollouts:16
  rollouts:32
  rollouts:64
  reward:correct_plus_0p1_format
  reward:indicator_correct_and_format
  reward:correct_plus_valid_plus_0p1_format
  reward:correct_plus_0p75_valid_plus_0p1_format
  reward:correct_plus_0p5_valid_plus_0p1_format
  reward:correct_plus_0p25_valid_plus_0p1_format
  reward:indicator_all
)

N_CFGS=${#CFGS[@]}
SEED_IDX=$((SLURM_ARRAY_TASK_ID / N_CFGS))
CFG_IDX=$((SLURM_ARRAY_TASK_ID % N_CFGS))
SEED="${SEEDS[$SEED_IDX]}"
CFG="${CFGS[$CFG_IDX]}"
KIND="${CFG%%:*}"
VAL="${CFG#*:}"

SFT_CKPT="/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/merged_sft_lr1e-4_seed${SEED}_remerge_20260421"
if [[ ! -f "${SFT_CKPT}/config.json" ]]; then
  echo "Missing merged checkpoint: ${SFT_CKPT}" >&2
  exit 1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"

COMMON=(
  seed="${SEED}"
  model.path="${SFT_CKPT}"
  model.merge_adapter_for_rollout=false
  task.template=logic
  task.prefill=none
  grpo.max_num_batched_tokens=16384
  optim.micro_batch_size=4
  optim.logprob_micro_batch_size=4
)

if [[ "${KIND}" == "prompts" ]]; then
  export WANDB_GROUP="posttrain/grpo_np${VAL}_nr8"
  export WANDB_RUN_GROUP="${WANDB_GROUP}"
  python posttrain_grpo_verl.py \
    run_name="rl_grpo_merge_ablate_prompts${VAL}_rollouts8_seed${SEED}" \
    reward.schema=correct_plus_valid_plus_0p1_format \
    grpo.num_prompts="${VAL}" \
    grpo.num_rollouts=8 \
    "${COMMON[@]}"
elif [[ "${KIND}" == "rollouts" ]]; then
  export WANDB_GROUP="posttrain/grpo_np8_nr${VAL}"
  export WANDB_RUN_GROUP="${WANDB_GROUP}"
  python posttrain_grpo_verl.py \
    run_name="rl_grpo_merge_ablate_prompts8_rollouts${VAL}_seed${SEED}" \
    reward.schema=correct_plus_valid_plus_0p1_format \
    grpo.num_prompts=8 \
    grpo.num_rollouts="${VAL}" \
    "${COMMON[@]}"
else
  export WANDB_GROUP="posttrain/reward_${VAL}"
  export WANDB_RUN_GROUP="${WANDB_GROUP}"
  python posttrain_grpo_verl.py \
    run_name="rl_reward_${VAL}_seed${SEED}_mrg" \
    reward.schema="${VAL}" \
    grpo.num_prompts=8 \
    grpo.num_rollouts=8 \
    task.train_min_step=1 \
    task.train_max_step=10 \
    task.val_min_step=1 \
    task.val_max_step=20 \
    "${COMMON[@]}"
fi
