#!/usr/bin/env bash
#SBATCH --job-name=q25_depth_sft
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --exclude=a0531,a0532,a0533,a0934
#SBATCH --array=0-7%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Chain-length axis. The share is fixed at ten percent and the depth of the
# training proofs varies over bands 5 and 15; band 25 comes from the dose
# sweep, so the three points together give a depth curve at constant dose.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"
source ./scripts/env.sh
VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"; export PATH="${VENV_PATH}/bin:${PATH}"
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
hash -r 2>/dev/null || true

CONFIGS=()
for seed in 3407 3408; do
  for tmpl in logic nl_exact; do
    for band in 5 15; do CONFIGS+=("${tmpl} ${band} ${seed}"); done
  done
done
read -r TMPL BAND SEED <<<"${CONFIGS[${SLURM_ARRAY_TASK_ID}]}"
RUN_NAME="qwen25_7b_depth_${TMPL}_b${BAND}_seed${SEED}"
MIX="${HPCVAULT}/synthetic-RLVL/datasets/depth_sweep_20260915/${TMPL}_b${BAND}"
OUT_DIR="${WORK}/synthetic-RLVL/post_sft_depth_20260915/${RUN_NAME}"

test -d "${MIX}" || { echo "missing mixture ${MIX}" >&2; exit 2; }
[[ -s "${OUT_DIR}/final/config.json" ]] && { echo "already done ${RUN_NAME}"; exit 0; }
echo "run=${RUN_NAME} mixture=${MIX}"

python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train_instruction_sft.py \
  --model "${BASE_MODEL:-Qwen/Qwen2.5-7B}" --prepared-dataset "${MIX}" \
  --train-split train --eval-split eval \
  --run-name "${RUN_NAME}" --output-dir "${OUT_DIR}" --seed "${SEED}" \
  --train-samples 100000 --eval-samples 2048 \
  --max-length 8192 --max-truncated-frac 0.005 \
  --max-steps -1 --num-train-epochs 1 --lr 5e-6 \
  --warmup-steps 0 --warmup-ratio 0.03 \
  --per-device-batch-size 2 --per-device-eval-batch-size 1 --grad-accum 16 \
  --eval-steps 2000 --save-steps 200 --save-total-limit 1 \
  --format-mode chat --single-turn-only --no-wrap-answer-tags \
  --full-parameter --fsdp "full_shard auto_wrap" \
  --fsdp-transformer-layer-cls-to-wrap Qwen2DecoderLayer \
  --gradient-checkpointing --bf16 --report-to none \
  --resume-from-checkpoint auto

test -s "${OUT_DIR}/final/config.json"
find "${OUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf -- {} +
echo "done ${RUN_NAME}"
