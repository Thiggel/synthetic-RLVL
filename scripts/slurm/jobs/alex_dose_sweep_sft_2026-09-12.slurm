#!/usr/bin/env bash
#SBATCH --job-name=q25_dose_sft
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --exclude=a0531,a0532,a0533,a0934
#SBATCH --array=0-26%6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# The headline sweep: how much synthetic derivation data does an ordinary
# instruction mixture need before the model reasons better?
#
# Stock Qwen2.5-7B (no midtraining), 100k instruction examples, REPLACEMENT
# design so every arm sees the same number of examples and only the content of
# the replaced slice differs. Nine configurations x three seeds:
#
#   share  0, 1, 5, 10, 25 percent   notation  formal / English   band 25
#
# The 0 percent arm is the control and exists once per seed, so the grid is
# 2 notations x 4 shares + 1 control = 9 per seed.
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
for seed in 3407 3408 3409; do
  CONFIGS+=("control 0 ${seed}")
  for tmpl in logic nl_exact; do
    for frac in 0.01 0.05 0.10 0.25; do CONFIGS+=("${tmpl} ${frac} ${seed}"); done
  done
done
read -r TMPL FRAC SEED <<<"${CONFIGS[${SLURM_ARRAY_TASK_ID}]}"
TAG=$(python3 -c "print('p%02d' % round(float('${FRAC}')*100))")
RUN_NAME="qwen25_7b_dose_${TMPL}_${TAG}_seed${SEED}"
MIX_ROOT="${HPCVAULT}/synthetic-RLVL/datasets/dose_sweep_20260912"
if [[ "${TMPL}" == "control" ]]; then MIX="${MIX_ROOT}/control"; else MIX="${MIX_ROOT}/${TMPL}_${TAG}"; fi
OUT_DIR="${WORK}/synthetic-RLVL/post_sft_dose_20260912/${RUN_NAME}"
BASE="${BASE_MODEL:-Qwen/Qwen2.5-7B}"

test -d "${MIX}" || { echo "missing mixture ${MIX}" >&2; exit 2; }
[[ -s "${OUT_DIR}/final/config.json" ]] && { echo "already done ${RUN_NAME}"; exit 0; }
echo "run=${RUN_NAME} base=${BASE} mixture=${MIX}"

python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train_instruction_sft.py \
  --model "${BASE}" --prepared-dataset "${MIX}" \
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
du -sh "${OUT_DIR}/final"; echo "done ${RUN_NAME}"
