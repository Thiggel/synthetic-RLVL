#!/usr/bin/env bash
#SBATCH --job-name=posttrain_grpo_base_smoke
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=04:00:00
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

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"

python posttrain_grpo_verl.py \
  run_name="rl_grpo_base_smoke_${SLURM_JOB_ID}" \
  seed=3407 \
  allow_base_model=true \
  model.path=allenai/Olmo-3-1025-7B \
  task.template=logic \
  task.prefill=none \
  reward.schema=correct_plus_valid_plus_0p1_format \
  data.train_samples=1024 \
  data.val_samples=128 \
  grpo.train_steps=120 \
  grpo.num_prompts=8 \
  grpo.num_rollouts=4 \
  validation.before_train=true \
  validation.eval_every=40 \
  validation.save_every=40 \
  eval.enabled=false \
  log_generations.num_samples=4 \
  log_generations.train_every=20 \
  log_generations.train_num_samples=6 \
  log_generations.train_max_chars=1600 \
  log_generations.train_dump_jsonl=false
