#!/usr/bin/env bash
#SBATCH --job-name=sft_sweep_bsz
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-8
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
BSZS=(4 8 16)
TRAIN_SAMPLES=(40000 80000 160000)

NUM_SEEDS=${#SEEDS[@]}
NUM_BSZS=${#BSZS[@]}
HP_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

BSZ="${BSZS[$HP_IDX]}"
N_SAMPLES="${TRAIN_SAMPLES[$HP_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

export WANDB_RUN_GROUP="sft_bsz/bsz_${BSZ}"
export WANDB_GROUP="${WANDB_RUN_GROUP}"
export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvr-sft}"

python train_sft.py \
  run_name="sft_bsz_b${BSZ}_n${N_SAMPLES}_seed${SEED}" \
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
  data.train_samples="${N_SAMPLES}" \
  train.lr=3e-5 \
  train.per_device_batch_size="${BSZ}" \
  train.grad_accum=1 \
  train.num_epochs=1 \
  train.max_steps=-1 \
  train.eval_steps=1000 \
  train.save_steps=1000 \
  external_eval.enabled=false \
  constrained_eval.enabled=false
