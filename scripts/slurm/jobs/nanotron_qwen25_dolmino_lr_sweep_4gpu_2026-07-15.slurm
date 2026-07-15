#!/usr/bin/env bash
#SBATCH --job-name=q25_dolmino_lr_4g
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export NPROC_PER_NODE=4
export TP=4
export DP=1
export PP=1
export MICRO_BATCH_SIZE=4
export BATCH_ACCUMULATION=32
export TOPOLOGY_TAG=tp4dp1

bash scripts/slurm/jobs/nanotron_qwen25_dolmino_lr_sweep_2026-07-15.slurm
