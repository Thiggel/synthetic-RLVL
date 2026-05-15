#!/usr/bin/env bash
# Easy curriculum K=2 hard-FSA-schema GRPO.
# Axes:
#   - train shortcut_rate: 0.0 sanity vs 0.5 shortcut-rich
#   - reward: answer-only vs validity-gated answer
#   - seeds: 3407,3408,3409
#SBATCH --job-name=posttrain_hfsa_easy500
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-11%12
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
SHORTCUT_TAGS=(0p0 0p5)
SHORTCUT_VALUES=(0.0 0.5)
TRAIN_SUBSETS=(train_schema_easy0p0_up_to_5_50k train_schema_easy0p5_up_to_5_50k)
SCHEMAS=(
  correct_plus_0p1_format
  indicator_correct_and_citation_free_valid_plus_0p1_format
)

IDX=${SLURM_ARRAY_TASK_ID}
SCHEMA_IDX=$((IDX % ${#SCHEMAS[@]}))
TMP=$((IDX / ${#SCHEMAS[@]}))
SHORTCUT_IDX=$((TMP % ${#SHORTCUT_TAGS[@]}))
SEED=${SEEDS[$((TMP / ${#SHORTCUT_TAGS[@]}))]}
SCHEMA=${SCHEMAS[$SCHEMA_IDX]}
SHORTCUT_TAG=${SHORTCUT_TAGS[$SHORTCUT_IDX]}
SHORTCUT_VALUE=${SHORTCUT_VALUES[$SHORTCUT_IDX]}
TRAIN_SUBSET=${TRAIN_SUBSETS[$SHORTCUT_IDX]}

RUN_NAME="rl_hfsa_easy500_train${SHORTCUT_TAG}_${SCHEMA}_seed${SEED}_mrg"
GROUP="posttrain_hard_fsa_schema_easy500/train${SHORTCUT_TAG}/${SCHEMA}"
SFT_CHECKPOINT_DEFAULT="${WORK}/synthetic-RLVL/tmp/merged_sft_hard_fsa_schema_easy_seed${SEED}"
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
  --config-name posttrain_grpo_hard_fsa_schema_easy_500 \
  seed="${SEED}" \
  run_name="${RUN_NAME}" \
  model.path="${SFT_CHECKPOINT}" \
  model.merge_adapter_for_rollout=false \
  reward.schema="${SCHEMA}" \
  logging.project="${WANDB_PROJECT}" \
  task.shortcut_rate="${SHORTCUT_VALUE}" \
  data.materialized.train_subset="${TRAIN_SUBSET}" \
  data.train_samples="${TRAIN_SAMPLES:-30000}" \
  grpo.train_steps="${TRAIN_STEPS:-500}" \
  grpo.num_prompts="${NUM_PROMPTS:-8}" \
  grpo.num_rollouts="${NUM_ROLLOUTS:-8}" \
  resume.mode="${RESUME_MODE:-disable}" \
  validation.eval_every="${EVAL_EVERY:-100}" \
  validation.save_every="${SAVE_EVERY:-100}" \
  validation.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPTS:-3}" \
  eval.synthetic_samples_per_step="${EVAL_SAMPLES_PER_STEP:-10}" \
  grpo.max_response_length="${MAX_RESPONSE_LENGTH:-1024}"
