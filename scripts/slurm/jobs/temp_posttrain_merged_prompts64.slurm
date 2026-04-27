#!/usr/bin/env bash
# Temporary OOM smoke test: merged-init checkpoint with high num_prompts.
# Sweep target: num_prompts in {8,16,32,64} at fixed num_rollouts=8.
# This script pins the max case (64) for quick capacity validation.
#
# Usage:
#   sbatch scripts/slurm/jobs/temp_posttrain_merged_prompts64.slurm
# Optional overrides:
#   MERGED_CHECKPOINT=/path/to/merged_or_adapter_ckpt
#   SEED=3407 WANDB_PROJECT=synthetic-rlvl
#
#SBATCH --job-name=tmp_grpo_mrg_p64_r8
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: bash ${BASH_SOURCE[0]}" >&2
  return 1
fi

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

MERGED_CKPT_DEFAULT="${WORK_ROOT}/synthetic-RLVL/tmp/merged_final_lr1e4_smoke"
SFT_CKPT="${MERGED_CHECKPOINT:-$MERGED_CKPT_DEFAULT}"
if [[ ! -d "${SFT_CKPT}" ]]; then
  echo "Merged checkpoint not found: ${SFT_CKPT}" >&2
  echo "Set MERGED_CHECKPOINT=/path/to/merged_or_adapter_ckpt and retry." >&2
  exit 1
fi

SEED="${SEED:-3407}"
JOB_TAG="${SLURM_JOB_ID:-interactive}"
export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"

python posttrain_grpo_verl.py \
  run_name="tmp_rl_grpo_merge_p64_r8_seed${SEED}_${JOB_TAG}" \
  seed="${SEED}" \
  model.path="${SFT_CKPT}" \
  task.template=logic \
  task.prefill=none \
  reward.schema=correct_plus_valid_plus_0p1_format \
  grpo.train_steps=120 \
  grpo.num_prompts=64 \
  grpo.num_rollouts=8 \
  validation.before_train=true \
  validation.eval_every=40 \
  validation.save_every=40 \
  eval.enabled=false \
  log_generations.num_samples=4 \
  log_generations.train_every=20 \
  log_generations.train_num_samples=6 \
  log_generations.train_max_chars=1600 \
  log_generations.train_dump_jsonl=false \
  grpo.max_num_batched_tokens=16384
