#!/usr/bin/env bash
#SBATCH --job-name=sft_logic_cot
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

SEEDS=(3407 3408 3409)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

export WANDB_RUN_GROUP="sft_template/logic_cot_lr_1e-4"
export WANDB_GROUP="${WANDB_RUN_GROUP}"
export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvr-sft}"

python train_sft.py \
  run_name="sft_logic_cot_lr1e-4_b1_seed${SEED}" \
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
  train.lr=1e-4 \
  train.per_device_batch_size=1 \
  train.grad_accum=1 \
  train.num_epochs=1 \
  train.max_steps=-1 \
  train.eval_steps=1000 \
  train.save_steps=1000 \
  external_eval.enabled=false \
  constrained_eval.enabled=false
