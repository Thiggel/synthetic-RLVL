#!/usr/bin/env bash
#SBATCH --job-name=q25_dolmino_lr_seq
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

for lr_index in 0 1 2; do
  echo "Starting Dolmino LR sweep row ${lr_index} at $(date --iso-8601=seconds)"
  SLURM_ARRAY_TASK_ID="${lr_index}" \
    bash scripts/slurm/jobs/nanotron_qwen25_dolmino_lr_sweep_2026-07-15.slurm
done

echo "Completed all Dolmino LR sweep rows at $(date --iso-8601=seconds)"
