#!/usr/bin/env bash
#SBATCH --job-name=hfsa_dscale_lmeval
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-29%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs lm_eval_results/hfsa_depth_scaling tmp
source ./scripts/env.sh
VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then VENV_PATH="${WORK}/.venv"; fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

SEEDS=(3407 3408 3409)
TEMPLATES=(logic nl_exact)
TRAIN_MAXES=(5 10 15 20 25)
IDX=${SLURM_ARRAY_TASK_ID}
SEED_IDX=$((IDX % 3))
TRAIN_IDX=$(((IDX / 3) % 5))
TEMPLATE_IDX=$((IDX / 15))
SEED=${SEEDS[$SEED_IDX]}
TEMPLATE=${TEMPLATES[$TEMPLATE_IDX]}
TRAIN_MAX=${TRAIN_MAXES[$TRAIN_IDX]}
TAG="${TEMPLATE}_train1to${TRAIN_MAX}"
RUN_NAME="sft_hfsa_depth_scaling_${TAG}_10k_seed${SEED}"
GROUP="hfsa_depth_scaling_lmeval/${TAG}"
SFT_FINAL="${WORK}/synthetic-RLVL/runs/${RUN_NAME}/final"
MERGED_ROOT="${WORK}/synthetic-RLVL/tmp/lmeval_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
MERGED="${MERGED_ROOT}/merged_${RUN_NAME}"
OUT_DIR="${WORK}/synthetic-RLVL/lm_eval_results/hfsa_depth_scaling/${RUN_NAME}"

cleanup_merged() { rm -rf "${MERGED_ROOT}"; }
trap cleanup_merged EXIT

if [[ ! -f "${SFT_FINAL}/adapter_config.json" ]]; then
  echo "missing SFT LoRA checkpoint: ${SFT_FINAL}" >&2
  exit 2
fi
if [[ -d "${OUT_DIR}" && "${FORCE_LM_EVAL:-0}" != "1" ]]; then
  echo "lm-eval output dir exists: ${OUT_DIR}; skipping"
  exit 0
fi

sleep $((SLURM_ARRAY_TASK_ID * ${LMEVAL_STAGGER_SECONDS:-120} + RANDOM % ${LMEVAL_JITTER_SECONDS:-90}))
mkdir -p "${MERGED_ROOT}" "${OUT_DIR}"
python scripts/merge_lora_checkpoint.py --checkpoint "${SFT_FINAL}" --output "${MERGED}.tmp" --dtype bf16 --overwrite
mv "${MERGED}.tmp" "${MERGED}"

LIMIT_ARGS=()
if [[ -n "${LM_EVAL_LIMIT:-100}" ]]; then
  LIMIT_ARGS=(--limit "${LM_EVAL_LIMIT:-100}")
fi

python scripts/evaluate_lm_eval.py \
  --checkpoint "${MERGED}" \
  --model "${LM_EVAL_MODEL:-vllm}" \
  --tasks ${LM_EVAL_TASKS:-gsm8k} \
  --output-path "${OUT_DIR}" \
  --batch-size "${LM_EVAL_BATCH_SIZE:-auto}" \
  "${LIMIT_ARGS[@]}" \
  --dtype "${LM_EVAL_DTYPE:-bfloat16}" \
  --gpu-memory-utilization "${LM_EVAL_GPU_MEMORY_UTILIZATION:-0.80}" \
  --log-samples \
  --wandb-project "${WANDB_PROJECT:-synthetic-rlvl}" \
  --wandb-group "${GROUP}" \
  --wandb-run-name "${RUN_NAME}_lmeval"

cleanup_merged
trap - EXIT
