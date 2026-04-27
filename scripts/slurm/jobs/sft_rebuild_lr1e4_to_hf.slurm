#!/usr/bin/env bash
# Rebuild the required SFT checkpoints (seed 3407/3408/3409, lr=1e-4),
# upload final adapters to Hugging Face, and clean local run outputs.
#
# Usage:
#   HF_TOKEN=hf_xxx sbatch scripts/slurm/jobs/sft_rebuild_lr1e4_to_hf.slurm
#
# Optional:
#   HF_REPO_PREFIX=flaitenberger/synthetic-rlvl-sft-lr1e4-seed
#   WANDB_PROJECT=synthetic-rlvr-sft

#SBATCH --job-name=sft_lr1e4_hf
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-2
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

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "Missing HF token. Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) before sbatch." >&2
  exit 1
fi

SEEDS=(3407 3408 3409)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
LR="1e-4"
RUN_NAME="sft_lr_lr${LR}_seed${SEED}"

HF_REPO_PREFIX="${HF_REPO_PREFIX:-flaitenberger/synthetic-rlvl-sft-lr1e4-seed}"
HF_REPO_ID="${HF_REPO_PREFIX}${SEED}"

export WANDB_RUN_GROUP="sft_lr/lr_${LR}"
export WANDB_GROUP="${WANDB_RUN_GROUP}"
export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvr-sft}"

python train_sft.py \
  run_name="${RUN_NAME}" \
  seed="${SEED}" \
  task.template=logic \
  task.prefill=none \
  task.train_min_step=1 \
  task.train_max_step=5 \
  task.val_min_step=1 \
  task.val_max_step=10 \
  data.source=materialized \
  data.materialized.dataset_id=flaitenberger/LogicalReasoning \
  data.materialized.local_root=null \
  data.train_samples=10000 \
  train.lr="${LR}" \
  train.per_device_batch_size=1 \
  train.grad_accum=1 \
  train.num_epochs=1 \
  train.max_steps=-1 \
  train.eval_steps=1000 \
  train.save_steps=10000 \
  train.save_total_limit=1 \
  external_eval.enabled=false \
  constrained_eval.enabled=false \
  hub.push_final=true \
  hub.repo_id="${HF_REPO_ID}" \
  hub.private=true \
  hub.token_env=HF_TOKEN \
  hub.path_in_repo="" \
  hub.cleanup_local_output=true
