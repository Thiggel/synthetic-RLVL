#!/usr/bin/env bash
#SBATCH --job-name=q25_mixdepth_sft
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:4
#SBATCH --exclude=a0532,a0934,a0534
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --array=0-9%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Depth-threshold transfer test at SFT scale.
#
# Question: the controlled study says formal traces only beat prose ABOVE a
# depth threshold (greedy crossover at band 25; sampled crossover already at
# band 15). The midtraining transfer study could never test above the threshold
# because deep proofs did not fit its 4,096-token window. SFT is cheap enough to
# afford a longer window, so we test it here.
#
# 5 conditions x 2 seeds. Replacement design: every condition trains on exactly
# 100k examples; only the composition of the replaced 10% differs.
#   0 control       100k Dolci
#   1 logic band15  90k Dolci + 10k formal traces, depths 1-15  (BELOW threshold)
#   2 nl    band15  90k Dolci + 10k prose  traces, depths 1-15  (BELOW threshold)
#   3 logic band25  90k Dolci + 10k formal traces, depths 1-25  (ABOVE threshold)
#   4 nl    band25  90k Dolci + 10k prose  traces, depths 1-25  (ABOVE threshold)
#
# Window is 8192, set FROM MEASUREMENT: band-25 traces reach 7,695 tokens, and
# 0% exceed 8192, while 43-47% exceed the historical 4,096 default. Training
# those at 4,096 would silently right-truncate the proof conclusion and the
# <answer> -- reproducing, inside SFT, the mid-proof splitting defect that
# invalidated the midtraining run. The job is therefore FAIL-CLOSED on
# truncation via --max-truncated-frac.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"
source ./scripts/env.sh

MIX_ROOT="${MIX_ROOT:-${HPCVAULT}/synthetic-RLVL/datasets/reasoning_mixture_20260821}"
DOLCI="${DOLCI:-${HPCVAULT}/synthetic-RLVL/datasets/dolci_no_tools_single_turn_100k_seed3407_20260803}"
OUT_ROOT="${MIXDEPTH_OUT_ROOT:-${HPCVAULT}/synthetic-RLVL/post_sft_reasoning_mixture_20260821}"

CONDITIONS=(control logic_band15 nl_exact_band15 logic_band25 nl_exact_band25)
SEEDS=(3407 3408)
IDX="${SLURM_ARRAY_TASK_ID}"
COND="${CONDITIONS[$(( IDX % 5 ))]}"
SEED="${SEEDS[$(( IDX / 5 ))]}"

if [[ "${COND}" == "control" ]]; then
  PREPARED="${DOLCI}"
else
  PREPARED="${MIX_ROOT}/dolci_bp_${COND}_p10"
fi
test -s "${PREPARED}/dataset_dict.json"

RUN_NAME="qwen25_7b_mixdepth_${COND}_seed${SEED}"
OUT_DIR="${OUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUT_ROOT}"
if [[ -s "${OUT_DIR}/final/config.json" ]]; then
  echo "Complete checkpoint exists: ${OUT_DIR}/final; skipping"
  exit 0
fi

VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true
export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="qwen25_mixdepth_sft"
export WANDB_RUN_GROUP="${WANDB_GROUP}"
export WANDB_NAME="${RUN_NAME}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"

echo "condition=${COND} seed=${SEED} prepared=${PREPARED} out=${OUT_DIR}"

"${VENV_PATH}/bin/python" -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train_instruction_sft.py \
  --model Qwen/Qwen2.5-7B \
  --prepared-dataset "${PREPARED}" \
  --train-split train \
  --eval-split eval \
  --run-name "${RUN_NAME}" \
  --output-dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --train-samples 100000 \
  --eval-samples 2048 \
  --max-length "${MAX_LENGTH:-8192}" \
  --max-truncated-frac "${MAX_TRUNCATED_FRAC:-0.005}" \
  --max-steps "${MAX_STEPS:--1}" \
  --num-train-epochs 1 \
  --lr "${LR:-5e-6}" \
  --warmup-steps 0 \
  --warmup-ratio 0.03 \
  --per-device-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --grad-accum 32 \
  --eval-steps 250 \
  --save-steps 250 \
  --save-total-limit 1 \
  --format-mode chat \
  --single-turn-only
