#!/usr/bin/env bash
#SBATCH --job-name=fam_sft
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:H200:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --array=0-11%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Does the effect hold outside Qwen2.5-7B?
#
# The headline sweep trains one model family at one scale, which is its main
# limitation. This job repeats three conditions of that sweep, the control,
# five percent formal and ten percent English, on two further families and two
# further scales. Every hyperparameter matches the headline sweep so that the
# only differences are the base model and its size.
#
# Base models that ship no chat template receive the plain role-prefixed
# format, which adds no vocabulary, and the template is saved with the
# checkpoint so evaluation applies the same one.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

R=/scratch/usr/bemflait/synthetic-RLVL
export HF_HOME="${R}/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1
# Compute nodes have no route to the internet, so every weight and dataset is
# fetched on the login node first and read from the cache here.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
source "${R}/.venv/bin/activate"

MODELS=(
  "NousResearch/Meta-Llama-3.1-8B LlamaDecoderLayer llama31_8b plain"
  "allenai/OLMo-2-1124-7B Olmo2DecoderLayer olmo2_7b plain"
  "Qwen/Qwen2.5-3B Qwen2DecoderLayer qwen25_3b none"
  "Qwen/Qwen2.5-14B Qwen2DecoderLayer qwen25_14b none"
)
CONDITIONS=("control control" "logic_p05 logic_p05" "nl_exact_p10 nl_exact_p10")

M_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
C_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
read -r BASE WRAP TAG TEMPLATE <<<"${MODELS[$M_IDX]}"
read -r MIX_NAME COND <<<"${CONDITIONS[$C_IDX]}"
SEED="${SEED:-3407}"

RUN_NAME="${TAG}_${COND}_seed${SEED}"
MIX="${R}/datasets/dose_sweep_20260912/${MIX_NAME}"
OUT_DIR="${R}/post_sft_family_20260915/${RUN_NAME}"

test -d "${MIX}" || { echo "missing mixture ${MIX}" >&2; exit 2; }
[[ -s "${OUT_DIR}/final/config.json" ]] && { echo "already done ${RUN_NAME}"; exit 0; }
echo "run=${RUN_NAME} base=${BASE} wrap=${WRAP} mixture=${MIX}"

TEMPLATE_ARGS=()
if [[ "${TEMPLATE}" != "none" ]]; then TEMPLATE_ARGS=(--chat-template "${TEMPLATE}"); fi

# SMOKE_STEPS runs a few optimizer steps so that a configuration error surfaces
# in minutes instead of after a night of training. It writes to its own output
# directory and never touches the real one.
SMOKE_ARGS=()
if [[ -n "${SMOKE_STEPS:-}" ]]; then
  SMOKE_ARGS=(--max-steps "${SMOKE_STEPS}" --train-samples 256 --eval-samples 8)
  OUT_DIR="${R}/post_sft_family_smoke/${RUN_NAME}"
  rm -rf "${OUT_DIR}"
fi

python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train_instruction_sft.py \
  --model "${BASE}" --prepared-dataset "${MIX}" \
  --train-split train --eval-split eval \
  --run-name "${RUN_NAME}" --output-dir "${OUT_DIR}" --seed "${SEED}" \
  --train-samples 100000 --eval-samples 2048 --max-steps -1 \
  --max-length 8192 --max-truncated-frac 0.005 \
  --num-train-epochs 1 --lr 5e-6 \
  "${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"}" \
  --warmup-steps 0 --warmup-ratio 0.03 \
  --per-device-batch-size 2 --per-device-eval-batch-size 1 --grad-accum 16 \
  --eval-steps 2000 --save-steps 500 --save-total-limit 1 \
  --format-mode chat --single-turn-only --no-wrap-answer-tags \
  "${TEMPLATE_ARGS[@]+"${TEMPLATE_ARGS[@]}"}" \
  --full-parameter --fsdp "full_shard auto_wrap" \
  --fsdp-transformer-layer-cls-to-wrap "${WRAP}" \
  --gradient-checkpointing --bf16 --report-to none \
  --resume-from-checkpoint auto

test -s "${OUT_DIR}/final/config.json"
find "${OUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf -- {} +
du -sh "${OUT_DIR}/final"; echo "done ${RUN_NAME}"
