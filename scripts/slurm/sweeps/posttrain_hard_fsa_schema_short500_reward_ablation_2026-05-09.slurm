#!/usr/bin/env bash
#SBATCH --job-name=posttrain_hard_fsa_schema500
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
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then VENV_PATH="${WORK}/.venv"; fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true
SEEDS=(3407 3408 3409)
SCHEMAS=(
  correct_plus_0p1_format
  indicator_correct_and_citation_free_valid_plus_0p1_format
)
IDX=${SLURM_ARRAY_TASK_ID}
SEED=${SEEDS[$((IDX / ${#SCHEMAS[@]}))]}
SCHEMA=${SCHEMAS[$((IDX % ${#SCHEMAS[@]}))]}
RUN_NAME="rl_hard_fsa_schema_short500_${SCHEMA}_seed${SEED}_mrg"
GROUP="posttrain_hard_fsa_schema_short500/${SCHEMA}"
SFT_CHECKPOINT_DEFAULT="${WORK}/synthetic-RLVL/tmp/merged_sft_hard_fsa_schema_short_seed${SEED}"
export SFT_CHECKPOINT="${SFT_CHECKPOINT:-${SFT_CHECKPOINT_DEFAULT}}"
if [[ ! -f "${SFT_CHECKPOINT}/config.json" ]]; then echo "Missing merged SFT checkpoint: ${SFT_CHECKPOINT}" >&2; exit 2; fi
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
  resume.mode="${RESUME_MODE:-disable}" \
  validation.eval_every="${EVAL_EVERY:-100}" \
  validation.save_every="${SAVE_EVERY:-250}" \
  eval.synthetic_samples_per_step="${EVAL_SAMPLES_PER_STEP:-10}" \
  grpo.max_response_length="${MAX_RESPONSE_LENGTH:-1024}"
