#!/usr/bin/env bash
#SBATCH --job-name=ood_tiny_cotbare
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --array=0-17%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then VENV_PATH="${WORK}/.venv"; fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

SIZES=(50m 100m 200m)
TEMPLATES=(logic nl_exact)
SEEDS=(3407 3408 3409)

IDX="${SLURM_ARRAY_TASK_ID}"
SEED="${SEEDS[$((IDX / 6))]}"
LOCAL_IDX="$((IDX % 6))"
SIZE="${SIZES[$((LOCAL_IDX % 3))]}"
TEMPLATE="${TEMPLATES[$((LOCAL_IDX / 3))]}"
TRAIN_MAX="${TRAIN_MAX:-10}"

RUN_NAME="pretrain_hfsa_llama3_${SIZE}_${TEMPLATE}_train1to${TRAIN_MAX}_seed${SEED}"
CHECKPOINT="${WORK}/synthetic-RLVL/runs/${RUN_NAME}/final"
OUT_DIR="${WORK}/synthetic-RLVL/lm_eval_results/ood_tiny_llama_cot_bare_2026-05-27/${RUN_NAME}"
GROUP="ood_lmeval_tiny_llama_cot_bare"

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  echo "missing tiny Llama checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi
if [[ -d "${OUT_DIR}" && "${FORCE_LM_EVAL:-0}" != "1" ]]; then
  echo "lm-eval output dir exists: ${OUT_DIR}; skipping"
  exit 0
fi

sleep $((SLURM_ARRAY_TASK_ID * ${LMEVAL_STAGGER_SECONDS:-20} + RANDOM % ${LMEVAL_JITTER_SECONDS:-30}))
mkdir -p "${OUT_DIR}"

LIMIT_ARGS=()
if [[ -n "${LM_EVAL_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LM_EVAL_LIMIT}")
fi

export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"

python scripts/evaluate_lm_eval.py \
  --checkpoint "${CHECKPOINT}" \
  --model vllm \
  --suite synthrlvl_ood_cot_bare \
  --include-path lm_eval_tasks/synthrlvl_ood \
  --output-path "${OUT_DIR}" \
  --batch-size "${LM_EVAL_BATCH_SIZE:-8}" \
  "${LIMIT_ARGS[@]}" \
  --dtype "${LM_EVAL_DTYPE:-bfloat16}" \
  --gpu-memory-utilization "${LM_EVAL_GPU_MEMORY_UTILIZATION:-0.70}" \
  --model-arg "max_model_len=${LM_EVAL_MAX_MODEL_LEN:-8192}" \
  --model-arg "max_num_seqs=${LM_EVAL_MAX_NUM_SEQS:-8}" \
  --confirm-run-unsafe-code \
  --log-samples \
  --wandb-project "${WANDB_PROJECT:-synthetic-rlvl}" \
  --wandb-group "${GROUP}" \
  --wandb-run-name "${RUN_NAME}_ood_cot_bare"
