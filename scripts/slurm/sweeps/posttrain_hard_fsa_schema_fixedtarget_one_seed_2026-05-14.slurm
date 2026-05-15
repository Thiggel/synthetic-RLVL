#!/usr/bin/env bash
# One-seed fixed-target HFSA GRPO matrix.
# Regimes:
#   0: SFT 1-3, RL 1-10, eval 1-25
#   1: SFT 1-5, RL 1-15, eval 1-25
# Rewards:
#   - correctness-only baseline: correct + 0.1 format
#   - validity reward: correct + citation-free-valid + 0.1 format
#SBATCH --job-name=posttrain_hfsa_fix
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
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

SEED="${SEED:-3407}"
REGIME_TAGS=(sft1to3_rl1to10 sft1to5_rl1to15)
SFT_TAGS=(sft1to3 sft1to5)
TRAIN_MAXES=(10 15)
TRAIN_SUBSETS=(train_fixedtarget_up_to_10_50k train_fixedtarget_up_to_15_50k)
SCHEMAS=(
  correct_plus_0p1_format
  correct_plus_citation_free_valid_plus_0p1_format
)

IDX=${SLURM_ARRAY_TASK_ID}
SCHEMA_IDX=$((IDX % ${#SCHEMAS[@]}))
REGIME_IDX=$((IDX / ${#SCHEMAS[@]}))
SCHEMA=${SCHEMAS[$SCHEMA_IDX]}
REGIME_TAG=${REGIME_TAGS[$REGIME_IDX]}
SFT_TAG=${SFT_TAGS[$REGIME_IDX]}
TRAIN_MAX=${TRAIN_MAXES[$REGIME_IDX]}
TRAIN_SUBSET=${TRAIN_SUBSETS[$REGIME_IDX]}

RUN_NAME="rl_hfsa_fixedtarget_${REGIME_TAG}_${SCHEMA}_seed${SEED}_mrg"
GROUP="posttrain_hard_fsa_schema_fixedtarget/${REGIME_TAG}/${SCHEMA}"
SFT_CHECKPOINT_DEFAULT="${WORK}/synthetic-RLVL/tmp/merged_sft_hfsa_fixedtarget_${SFT_TAG}_seed${SEED}"
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
  --config-name posttrain_grpo_hard_fsa_schema_fixedtarget_500 \
  seed="${SEED}" \
  run_name="${RUN_NAME}" \
  model.path="${SFT_CHECKPOINT}" \
  model.merge_adapter_for_rollout=false \
  reward.schema="${SCHEMA}" \
  logging.project="${WANDB_PROJECT}" \
  task.train_max_step="${TRAIN_MAX}" \
  task.shortcut_rate=0.0 \
  data.materialized.train_subset="${TRAIN_SUBSET}" \
  data.train_samples="${TRAIN_SAMPLES:-30000}" \
  grpo.train_steps="${TRAIN_STEPS:-500}" \
  grpo.num_prompts="${NUM_PROMPTS:-8}" \
  grpo.num_rollouts="${NUM_ROLLOUTS:-8}" \
  grpo.max_response_length="${MAX_RESPONSE_LENGTH:-2048}" \
  grpo.max_prompt_length="${MAX_PROMPT_LENGTH:-8192}" \
  grpo.max_model_len="${MAX_MODEL_LEN:-12288}" \
  grpo.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-32768}" \
  resume.mode="${RESUME_MODE:-disable}" \
  validation.eval_every="${EVAL_EVERY:-100}" \
  validation.save_every="${SAVE_EVERY:-100}" \
  validation.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPTS:-3}" \
  eval.synthetic_samples_per_step="${EVAL_SAMPLES_PER_STEP:-10}" \
  eval.max_new_tokens="${EVAL_MAX_NEW_TOKENS:-2048}"
