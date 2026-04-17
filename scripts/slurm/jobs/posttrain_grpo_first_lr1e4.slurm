#!/usr/bin/env bash
#SBATCH --job-name=posttrain_grpo_first
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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

SFT_CKPT_DEFAULT="/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/merged_final_lr1e4_smoke"
SFT_CKPT="${SFT_CHECKPOINT:-$SFT_CKPT_DEFAULT}"
if [[ ! -d "${SFT_CKPT}" ]]; then
  echo "SFT checkpoint not found: ${SFT_CKPT}" >&2
  echo "Set SFT_CHECKPOINT=/path/to/sft/final and retry." >&2
  exit 1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"

python posttrain_grpo_verl.py \
  run_name="rl_grpo_logic_lr1e4_seed3407_first" \
  seed=3407 \
  model.path="${SFT_CKPT}" \
  task.template=logic \
  task.prefill=none \
  reward.schema=correct_plus_0p1_format \
  grpo.num_prompts=8 \
  grpo.num_rollouts=8
