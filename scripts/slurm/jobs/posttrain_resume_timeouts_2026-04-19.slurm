#!/usr/bin/env bash
# Resume (or restart if no checkpoint) timed-out posttrain runs from 2026-04-19.
#
# Usage:
#   sbatch scripts/slurm/jobs/posttrain_resume_timeouts_2026-04-19.slurm
#
#SBATCH --job-name=posttrain_resume_0419
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-5
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

SEED=3409
RUN_NAME=""
SCHEMA="correct_plus_valid_plus_0p1_format"
NUM_ROLLOUTS=8

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    SEED=3407
    RUN_NAME="rl_grpo_merge_ablate_prompts8_rollouts64_seed3407"
    NUM_ROLLOUTS=64
    SCHEMA="correct_plus_valid_plus_0p1_format"
    ;;
  1)
    SEED=3408
    RUN_NAME="rl_grpo_merge_ablate_prompts8_rollouts64_seed3408"
    NUM_ROLLOUTS=64
    SCHEMA="correct_plus_valid_plus_0p1_format"
    ;;
  2)
    SEED=3409
    RUN_NAME="rl_grpo_merge_ablate_prompts8_rollouts32_seed3409"
    NUM_ROLLOUTS=32
    SCHEMA="correct_plus_valid_plus_0p1_format"
    ;;
  3)
    SEED=3409
    RUN_NAME="rl_grpo_merge_ablate_prompts8_rollouts64_seed3409"
    NUM_ROLLOUTS=64
    SCHEMA="correct_plus_valid_plus_0p1_format"
    ;;
  4)
    SEED=3409
    RUN_NAME="rl_reward_correct_plus_valid_plus_0p1_format_seed3409_mrg"
    NUM_ROLLOUTS=8
    SCHEMA="correct_plus_valid_plus_0p1_format"
    ;;
  5)
    SEED=3409
    RUN_NAME="rl_reward_indicator_all_seed3409_mrg"
    NUM_ROLLOUTS=8
    SCHEMA="indicator_all"
    ;;
  *)
    echo "Unsupported task id: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
    ;;
esac

SFT_CKPT="/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/merged_sft_lr1e-4_seed${SEED}"
if [[ ! -f "${SFT_CKPT}/config.json" ]]; then
  echo "Missing pre-merged checkpoint: ${SFT_CKPT}" >&2
  exit 1
fi

RUN_DIR="/home/atuin/c107fa/c107fa12/synthetic-RLVL/runs/${RUN_NAME}"
if [[ -d "${RUN_DIR}" ]]; then
  echo "Existing run dir detected: ${RUN_DIR}"
  if [[ -f "${RUN_DIR}/latest_checkpointed_iteration.txt" ]]; then
    echo "latest checkpoint step: $(cat "${RUN_DIR}/latest_checkpointed_iteration.txt")"
  else
    echo "no latest_checkpointed_iteration.txt (resume auto may restart from scratch)"
  fi
else
  echo "Run dir not found, will start fresh: ${RUN_DIR}"
fi

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"

python posttrain_grpo_verl.py \
  run_name="${RUN_NAME}" \
  seed="${SEED}" \
  model.path="${SFT_CKPT}" \
  model.merge_adapter_for_rollout=false \
  task.template=logic \
  task.prefill=none \
  reward.schema="${SCHEMA}" \
  grpo.max_num_batched_tokens=16384 \
  optim.micro_batch_size=4 \
  optim.logprob_micro_batch_size=4 \
  grpo.num_prompts=8 \
  grpo.num_rollouts="${NUM_ROLLOUTS}" \
  resume.mode=auto
