#!/usr/bin/env bash
# Three-seed rollout-count ablation for dense citation-free line-valid reward.
# Settings:
#   - 8 prompts x 16 rollouts
#   - 8 prompts x 32 rollouts
# Reward:
#   - R = correct * citation_free_line_valid_fraction + 0.1 * format
# Usage:
#   sbatch scripts/slurm/sweeps/posttrain_hard_fsa_schema_short500_linevalid_rollouts_2026-05-11.slurm
# Optional env overrides:
#   TRAIN_STEPS=500 STARTUP_STAGGER_SECONDS=300 STARTUP_JITTER_SECONDS=60 sbatch ...
#SBATCH --job-name=posttrain_hfsa_linevalid_rollouts
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-5%6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then
  VENV_PATH="${WORK}/.venv"
fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

SEEDS=(3407 3408 3409)
ROLLOUTS=(16 32)
NUM_PROMPTS=8
SCHEMA="correct_times_citation_free_line_valid_plus_0p1_format"

IDX=${SLURM_ARRAY_TASK_ID}
ROLLOUT_IDX=$((IDX % ${#ROLLOUTS[@]}))
SEED=${SEEDS[$((IDX / ${#ROLLOUTS[@]}))]}
NUM_ROLLOUTS=${ROLLOUTS[$ROLLOUT_IDX]}

RUN_NAME="rl_hard_fsa_schema_short500_np${NUM_PROMPTS}_nr${NUM_ROLLOUTS}_${SCHEMA}_seed${SEED}_mrg"
GROUP="posttrain_hard_fsa_schema_short500_rollouts/${SCHEMA}_np${NUM_PROMPTS}_nr${NUM_ROLLOUTS}"

SFT_CHECKPOINT_DEFAULT="${WORK}/synthetic-RLVL/tmp/merged_sft_hard_fsa_schema_short_seed${SEED}"
export SFT_CHECKPOINT="${SFT_CHECKPOINT:-${SFT_CHECKPOINT_DEFAULT}}"
if [[ ! -f "${SFT_CHECKPOINT}/config.json" ]]; then
  echo "Missing merged SFT checkpoint: ${SFT_CHECKPOINT}" >&2
  exit 2
fi

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="${GROUP}"
export WANDB_RUN_GROUP="${GROUP}"
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-120}"

sleep $((SLURM_ARRAY_TASK_ID * ${STARTUP_STAGGER_SECONDS:-300} + RANDOM % ${STARTUP_JITTER_SECONDS:-60}))

python posttrain_grpo_verl.py \
  --config-name posttrain_grpo_hard_fsa_schema_500 \
  seed="${SEED}" \
  run_name="${RUN_NAME}" \
  model.path="${SFT_CHECKPOINT}" \
  model.merge_adapter_for_rollout=false \
  reward.schema="${SCHEMA}" \
  logging.project="${WANDB_PROJECT}" \
  data.train_samples="${TRAIN_SAMPLES:-30000}" \
  grpo.train_steps="${TRAIN_STEPS:-500}" \
  grpo.num_prompts="${NUM_PROMPTS}" \
  grpo.num_rollouts="${NUM_ROLLOUTS}" \
  resume.mode="${RESUME_MODE:-disable}" \
  validation.eval_every="${EVAL_EVERY:-100}" \
  validation.save_every="${SAVE_EVERY:-250}" \
  eval.synthetic_samples_per_step="${EVAL_SAMPLES_PER_STEP:-10}" \
  grpo.max_response_length="${MAX_RESPONSE_LENGTH:-1024}"
