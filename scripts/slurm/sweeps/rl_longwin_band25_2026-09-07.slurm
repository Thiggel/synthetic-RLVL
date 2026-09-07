#!/usr/bin/env bash
#SBATCH --job-name=rl_longwin
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=a0531,a0532,a0533,a0934,a0534
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-5%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# GRPO RL post-training of the long-window band-25 arms: logic vs nl_exact
# across three reward definitions.
#
# Array map: idx % 2 -> arm (logic_band25, nl_exact_band25)
#            idx / 2 -> reward schema
#   0 correct_only        1 (correct)                       -- the plain RLVR recipe
#   1 correct_times_valid 1 only if correct AND proof-valid -- validity as a gate
#   2 correct_plus_valid  1 for correct + 1 for valid       -- validity as a bonus
#
# The question: pass@16 says formal has more coverage than prose but loses on
# greedy. RL with a verifiable reward is the mechanism that should convert
# coverage into pass@1. Reward 0 tests that with answer-correctness alone;
# rewards 1 and 2 add the answer-blind proof verifier, which is the paper's
# verifier dividend made causal. Validity-based rewards are run SECOND on
# purpose: they are more hackable than correctness, so the clean recipe is the
# reference point.
#
# LoRA rank 16 / alpha 16 on all-linear, 1xA100, per the accepted sweep config.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

ARMS=(logic_band25 nl_exact_band25)
TEMPLATES=(logic nl_exact)
SCHEMAS=(correct_only correct_times_valid correct_plus_valid)

IDX="${SLURM_ARRAY_TASK_ID}"
ARM_IDX=$((IDX % 2))
SCHEMA_IDX=$((IDX / 2))
ARM="${ARMS[$ARM_IDX]}"
TEMPLATE="${TEMPLATES[$ARM_IDX]}"
SCHEMA="${SCHEMAS[$SCHEMA_IDX]}"
SEED="${SEED:-3407}"

SFT_ROOT="${WORK}/synthetic-RLVL/post_sft_dolci_longwin_20260826"
export SFT_CHECKPOINT="${SFT_ROOT}/qwen25_7b_longwin_${ARM}_2p5b_dolci_100k_lr5em6/final"
RUN_NAME="rl_longwin_${ARM}_${SCHEMA}_seed${SEED}"
GROUP="rl_longwin_band25/${SCHEMA}"

# Fail closed: never RL-train from a missing or half-written SFT checkpoint,
# and never from a corpus that has not been materialised.
test -s "${SFT_CHECKPOINT}/config.json"
test -s "${WORK}/synthetic-RLVL/datasets/branchproof_unique_v2_rl_20260907/train_fixedtarget_up_to_25_30000/train.parquet"

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="${GROUP}"
export WANDB_RUN_GROUP="${GROUP}"
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-120}"
sleep $((SLURM_ARRAY_TASK_ID * ${STARTUP_STAGGER_SECONDS:-180} + RANDOM % 60))

echo "arm=${ARM} template=${TEMPLATE} schema=${SCHEMA} seed=${SEED} ckpt=${SFT_CHECKPOINT}"

python posttrain_grpo_verl.py \
  --config-name posttrain_grpo_longwin_band25_300 \
  seed="${SEED}" \
  run_name="${RUN_NAME}" \
  model.path="${SFT_CHECKPOINT}" \
  model.merge_adapter_for_rollout=false \
  task.template="${TEMPLATE}" \
  reward.schema="${SCHEMA}" \
  logging.project="${WANDB_PROJECT}" \
  grpo.train_steps="${TRAIN_STEPS:-300}" \
  validation.eval_every="${EVAL_EVERY:-50}" \
  validation.save_every="${SAVE_EVERY:-100}"
