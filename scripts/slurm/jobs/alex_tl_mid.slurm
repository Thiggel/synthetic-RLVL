#!/usr/bin/env bash
#SBATCH --job-name=tl_mid
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:2
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/atuin/c107fa/c107fa12/rlvl_data/logs/%x_%j.out
#SBATCH --error=/home/atuin/c107fa/c107fa12/rlvl_data/logs/%x_%j.err

# alex (rtxpro6k, 96 GB) TL token-logic midtraining (RLVL-next/TLANG.md §8): full-parameter SFT of a Qwen3.5 base
# extended with the 59 TL symbol tokens (--tl-root) on a tl/build_sft.py mixture (generator traces + Dolci replay).
# Same trainer as gruenau_formal_mix_train_2026-09-25.slurm, env from alex_grpo_env.sh (transformers 5.17, fla 0.5.2).
# Env: MODEL (abs or relative to RLVL_DATA_ROOT), MIX_DIR (same), TL_ROOT (same), RUN_NAME (out dir under
# ${RLVL_DATA_ROOT}/tl/mid), LR (2e-5), GLOBAL_BATCH (64), PER_DEVICE_BATCH (4),
# MAX_STEPS (-1), GRAD_CKPT (0), TL_LOSS_HEAD (full), DS_CONFIG (unset = DDP; 9B: configs/deepspeed/zero2.json),
# SAVE_STEPS (500). Submit with --export=ALL.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export RLVL_DATA_ROOT="${RLVL_DATA_ROOT:-/home/atuin/c107fa/c107fa12/rlvl_data}"
source scripts/slurm/jobs/alex_grpo_env.sh
unset PYTHONPATH
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}" WANDB_MODE=offline \
  WANDB_DIR="${TMPDIR}/wandb" OMP_NUM_THREADS=8
abs() { [[ "$1" = /* ]] && echo "$1" || echo "${RLVL_DATA_ROOT}/$1"; }
MODEL=$(abs "${MODEL:?}"); MIX_DIR=$(abs "${MIX_DIR:?}"); TL_ROOT=$(abs "${TL_ROOT:?}")
OUT_DIR="${RLVL_DATA_ROOT}/tl/mid/${RUN_NAME:?}"
test -s "${MODEL}/config.json" && test -s "${MIX_DIR}/mixture_manifest.json" && test -d "${TL_ROOT}/tl"
[[ -s "${OUT_DIR}/final/config.json" ]] && { echo "complete: ${OUT_DIR}/final"; exit 0; }
NGPU=$(nvidia-smi -L | wc -l)
GLOBAL_BATCH="${GLOBAL_BATCH:-64}"; PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM=$(( GLOBAL_BATCH / (NGPU * PER_DEVICE_BATCH) ))
(( GRAD_ACCUM * NGPU * PER_DEVICE_BATCH == GLOBAL_BATCH )) || { echo "batch not divisible" >&2; exit 2; }
echo "model=${MODEL} mix=${MIX_DIR} ngpu=${NGPU} pdb=${PER_DEVICE_BATCH} accum=${GRAD_ACCUM} out=${OUT_DIR} host=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
EXTRA=(--tl-root "${TL_ROOT}")
[[ "${GRAD_CKPT:-0}" == 1 ]] && EXTRA+=(--gradient-checkpointing)
[[ -n "${EVAL_ROWS:-}" ]] && EXTRA+=(--eval-rows "${EVAL_ROWS}")
[[ -n "${TL_LOSS_HEAD:-}" ]] && EXTRA+=(--tl-loss-head "${TL_LOSS_HEAD}")  # PLAN W3.3: full|rows60|legal
[[ -n "${DS_CONFIG:-}" ]] && EXTRA+=(--deepspeed "${DS_CONFIG}")  # ZeRO-2, bf16 load + fp32 master weights
mkdir -p "${OUT_DIR}"
python -m torch.distributed.run --standalone --nproc_per_node="${NGPU}" scripts/train_formal_mixture_sft.py \
  --model "${MODEL}" --prepared-dataset "${MIX_DIR}" --output-dir "${OUT_DIR}" --run-name "${RUN_NAME}" \
  --seed "${SEED:-3407}" --max-length "${MAX_LENGTH:-4096}" --max-steps "${MAX_STEPS:--1}" --lr "${LR:-2e-5}" \
  --per-device-batch-size "${PER_DEVICE_BATCH}" --per-device-eval-batch-size "${PER_DEVICE_BATCH}" \
  --grad-accum "${GRAD_ACCUM}" --logging-steps "${LOGGING_STEPS:-10}" --eval-steps "${EVAL_STEPS:-250}" \
  --save-steps "${SAVE_STEPS:-500}" --save-total-limit 1 --audit-output "${OUT_DIR}/tokenization_audit.json" \
  --max-truncated-frac "${MAX_TRUNCATED_FRAC:-0.01}" --report-to none "${EXTRA[@]}"
test -s "${OUT_DIR}/final/config.json"
du -sh "${OUT_DIR}/final"
echo "done ${RUN_NAME}"
