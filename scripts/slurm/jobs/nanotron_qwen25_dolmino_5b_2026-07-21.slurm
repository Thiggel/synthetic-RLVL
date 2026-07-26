#!/usr/bin/env bash
#SBATCH --job-name=q25_dolmino_5b
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./scripts/env.sh

CONDITIONS=(control logic nl_exact)
INNER_TASK_IDS=(0 1 6)
OUTER_TASK_ID="${SLURM_ARRAY_TASK_ID}"
CONDITION="${CONDITIONS[$OUTER_TASK_ID]}"
export SLURM_ARRAY_TASK_ID="${INNER_TASK_IDS[$OUTER_TASK_ID]}"
if [[ "${CONDITION}" == "control" ]]; then
  MIX_PCT_TAG=0
else
  MIX_PCT_TAG=5
fi

export TRAIN_STEPS=9537
export CHECKPOINT_INTERVAL=500
export LR=1.0e-5
export MIN_LR=1.0e-6
export WARMUP_STEPS=256
export LR_DECAY_STARTING_STEP=256
export LR_DECAY_STEPS=37891
export NORMAL_DATASET_FOLDER="${NORMAL_DATASET_FOLDER:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino}"
export LOGIC_DATASET_FOLDER="${LOGIC_DATASET_FOLDER:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1/logic}"
export NL_DATASET_FOLDER="${NL_DATASET_FOLDER:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1/nl_exact}"
export RUN_TAG_OVERRIDE="${RUN_TAG_OVERRIDE:-qwen25_7b_dolmino_${CONDITION}_p${MIX_PCT_TAG}_5b}"
export RUN_ROOT_OVERRIDE="${RUN_ROOT_OVERRIDE:-${HPCVAULT}/synthetic-RLVL/nanotron_dolmino_5b/${RUN_TAG_OVERRIDE}}"
export FINAL_CHECKPOINT_ALIAS=5b
export PRUNE_SUPERSEDED_RUN_CHECKPOINTS=1

bash scripts/slurm/jobs/nanotron_qwen25_midtrain_grid_2026-06-24.slurm
