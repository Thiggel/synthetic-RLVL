#!/usr/bin/env bash
# Smoke test: hard-FSA-schema GRPO with 8 prompts x 16 rollouts on 2 GPUs.
# This is intentionally not a real experiment run.
#SBATCH --job-name=smoke_hfsa_np8_nr16_2gpu
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=24
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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

SEED=3407
NUM_PROMPTS=8
NUM_ROLLOUTS=16
NGPUS=2
SCHEMA="indicator_correct_and_citation_free_valid_plus_0p1_format"
RUN_NAME="smoke_hard_fsa_schema_np${NUM_PROMPTS}_nr${NUM_ROLLOUTS}_${NGPUS}gpu_${SCHEMA}_seed${SEED}_${SLURM_JOB_ID}"
SFT_CHECKPOINT="${WORK}/synthetic-RLVL/tmp/merged_sft_hard_fsa_schema_short_seed${SEED}"

if [[ ! -f "${SFT_CHECKPOINT}/config.json" ]]; then
  echo "Missing merged SFT checkpoint: ${SFT_CHECKPOINT}" >&2
  exit 2
fi

export WANDB_MODE=disabled
export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="smoke/hard_fsa_schema_multigpu"
export WANDB_RUN_GROUP="${WANDB_GROUP}"
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-120}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

python posttrain_grpo_verl.py \
  --config-name posttrain_grpo_hard_fsa_schema_500 \
  seed="${SEED}" \
  run_name="${RUN_NAME}" \
  model.path="${SFT_CHECKPOINT}" \
  model.merge_adapter_for_rollout=false \
  reward.schema="${SCHEMA}" \
  logging.project="${WANDB_PROJECT}" \
  data.train_samples=512 \
  data.val_samples=32 \
  grpo.train_steps=5 \
  grpo.num_prompts="${NUM_PROMPTS}" \
  grpo.num_rollouts="${NUM_ROLLOUTS}" \
  resume.mode=disable \
  validation.enabled=false \
  validation.before_train=false \
  validation.eval_every=1000 \
  validation.save_every=1000 \
  eval.enabled=false \
  log_generations.num_samples=0 \
  log_generations.train_every=0 \
  system.n_gpus_per_node="${NGPUS}" \
  system.ray_max_colocate_count=1 \
  system.ray_cpus="${SLURM_CPUS_PER_TASK}"
