#!/usr/bin/env bash
#SBATCH --job-name=gru_notation_sft_a100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a10080gb:2
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=gruenau9
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3%1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Unconditioned per-notation instruction tuning on gruenau9 (2x A100 80GB).
#
# Why this variant exists: the H100 run (job 1844_0) diverged at step 3 on
# torch 2.9.0, the version the handoff believed safe, and gruenau11 went DOWN
# on 2026-09-10 21:09. gruenau9 carries the same GPU class as alex, where the
# identical recipe trains cleanly, so it separates "gruenau software" from
# "H100 numerics". Set MAX_STEPS=12 SMOKE=1 for a divergence smoke test that
# writes to a throwaway directory.
#
# Why here: alex sits at its account-wide 24-GPU cap, held entirely by the
# three 30-percent midtrain arms, so every instruction-tuning and evaluation
# job there is blocked on AssocGrpGRES. Gruenau's H100 and A100 nodes are idle
# and carry a 14-day walltime instead of 24 hours. The two filesystems are
# separate, so the bases and mixtures are copied over first.
#
# The recipe matches the accepted alex post-SFT wrapper: full-parameter FSDP,
# lr 5e-6, one epoch over 100k examples, chat format, single turn, no answer
# tags. Library versions are pinned to alex's (transformers 4.57.3, trl 0.24.0,
# peft 0.17.1, accelerate 1.13.0) so the runs are comparable.
#
# Two deliberate differences from the 4096-token alex default, both about the
# same hazard: the window is 8192, because a band-25 formal derivation runs to
# about 7.7k tokens and 43 to 47 percent of them exceed 4096, and truncation is
# fail-closed. Silently right-truncating a derivation removes its conclusion
# and answer field and yields a clean-looking null.
#
#   index  base                  mixture     purpose
#     0    control               formal      the clean notation contrast,
#     1    control               English       identical base, one variable
#     2    logic_band25          formal      matched-notation interaction
#     3    nl_exact_band25       English

# Measured on gruenau11, 781 steps, seconds per optimizer step:
#   2 GPUs, batch 1, accum 64   210 s   (~45 h)
#   4 GPUs, batch 1, accum 32   110 s   (~24 h)
#   4 GPUs, batch 2, accum 16    91 s   (~20 h)  <- used
#   4 GPUs, batch 4              OOM
#   4 GPUs, batch 2, ZeRO-2      OOM
# Alex does the same work in about 18 s per step on 4 A100 SXM, where every GPU
# sees every other over NVSwitch. Here the two NVLink pairs meet only over the
# host PCIe bus, so ZeRO-3 pays for that link on every micro-batch. Scaling from
# 2 to 4 GPUs is still near-linear, so taking the whole node and raising the
# per-device batch is the win available; ZeRO-2 would cut the traffic further
# but its replicated parameters do not fit in 82 GB.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

ROOT=/vol/home-vol2/ml/laitenbf
VENV="${ROOT}/.venv_rlvl_gruenau"
DATA="${ROOT}/rlvl_data"
export PATH="${VENV}/bin:${PATH}"
export VIRTUAL_ENV="${VENV}"
export HF_HOME="${HF_HOME:-${DATA}/hf_home}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
# no wandb account is configured on gruenau; keep runs local
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${DATA}/wandb"
hash -r 2>/dev/null || true

BASES=(control control logic_band25 nl_exact_band25)
MIXTURES=(logic nl_exact logic nl_exact)
IDX="${SLURM_ARRAY_TASK_ID}"
BASE="${BASES[$IDX]}"
MIXTURE="${MIXTURES[$IDX]}"
SEED="${SEED:-3407}"

declare -A MIX_DIR=(
  [logic]="${DATA}/datasets/reasoning_mixture_20260821/dolci_bp_logic_band25_p10"
  [nl_exact]="${DATA}/datasets/reasoning_mixture_20260821/dolci_bp_nl_exact_band25_p10"
)
MODEL="${DATA}/hf_midtrain/qwen25_7b_longwin_${BASE}_2p5b_hf"
RUN_NAME="qwen25_7b_longwin_${BASE}_2p5b_sft${MIXTURE}_100k_lr5em6_seed${SEED}${SMOKE:+_smoke}"
OUT_DIR="${DATA}/post_sft_notation/${RUN_NAME}"

# fail closed on both inputs before spending a GPU-hour
test -s "${MODEL}/config.json" || { echo "missing converted base: ${MODEL}" >&2; exit 2; }
test -d "${MIX_DIR[$MIXTURE]}" || { echo "missing mixture: ${MIX_DIR[$MIXTURE]}" >&2; exit 2; }
if [[ -s "${OUT_DIR}/final/config.json" ]]; then
  echo "complete run already exists: ${OUT_DIR}/final; skipping"
  exit 0
fi

NGPU=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr "," "\n" | wc -l)
# Hold the global batch at alex's 128 sequences whatever the GPU count is, or
# the runs are not comparable with the ones trained there.
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
if (( GLOBAL_BATCH % NGPU != 0 )); then
  echo "global batch ${GLOBAL_BATCH} is not divisible by ${NGPU} GPUs" >&2; exit 2
fi
# FSDP_STRATEGY: gruenau11 NVLinks GPU0-1 and GPU2-3 but joins the two pairs
# only over the host PCIe bus, which makes ZeRO-3 (full_shard) all-gather
# every layer across a slow link on every accumulation micro-batch.
# shard_grad_op (ZeRO-2) replicates parameters and only reduce-scatters
# gradients, trading memory for far less traffic.
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
DENOM=$(( NGPU * PER_DEVICE_BATCH ))
if (( GLOBAL_BATCH % DENOM != 0 )); then
  echo "global batch ${GLOBAL_BATCH} is not divisible by ${NGPU} GPUs x ${PER_DEVICE_BATCH}" >&2; exit 2
fi
GRAD_ACCUM=$(( GLOBAL_BATCH / DENOM ))
echo "base=${BASE} mixture=${MIXTURE} seed=${SEED} gpus=${NGPU} per_device=${PER_DEVICE_BATCH} grad_accum=${GRAD_ACCUM} global_batch=${GLOBAL_BATCH}"
echo "model=${MODEL}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m torch.distributed.run --standalone --nproc_per_node="${NGPU}" \
  scripts/train_instruction_sft.py \
  --model "${MODEL}" \
  --prepared-dataset "${MIX_DIR[$MIXTURE]}" \
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
  --per-device-batch-size "${PER_DEVICE_BATCH}" \
  --per-device-eval-batch-size 1 \
  --grad-accum "${GRAD_ACCUM}" \
  --eval-steps 250 \
  --logging-steps "${LOGGING_STEPS:-10}" \
  --save-steps "${SAVE_STEPS:-100000}" \
  --save-total-limit 1 \
  --format-mode chat \
  --single-turn-only \
  --no-wrap-answer-tags \
  --full-parameter \
  --fsdp "${FSDP_STRATEGY:-full_shard auto_wrap}" \
  --fsdp-transformer-layer-cls-to-wrap Qwen2DecoderLayer \
  --gradient-checkpointing \
  --bf16 \
  --report-to none \
  --resume-from-checkpoint auto

test -s "${OUT_DIR}/final/config.json"
find "${OUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf -- {} +
du -sh "${OUT_DIR}/final"
echo "done ${RUN_NAME}"
