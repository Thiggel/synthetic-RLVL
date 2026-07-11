#!/usr/bin/env bash
#SBATCH --job-name=nano_eval_smoke
#SBATCH --partition=a40,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
hash -r 2>/dev/null || true

TASKS="gsm8k,arc_challenge,hellaswag,winogrande,piqa,logiqa2,fld_default,fld_logical_formula_default,bbh,mmlu,mmlu_pro"
OUT_ROOT="${HPCVAULT}/synthetic-RLVL/lm_eval_results/qwen25_downstream_suite_smoke_20260711"
rm -rf "${OUT_ROOT}"

python scripts/evaluate_lm_eval.py \
  --checkpoint Qwen/Qwen2.5-0.5B \
  --model vllm \
  --tasks "${TASKS}" \
  --output-path "${OUT_ROOT}/direct" \
  --batch-size auto \
  --limit 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --model-arg max_model_len=8192 \
  --trust-remote-code \
  --confirm-run-unsafe-code \
  --log-samples

python scripts/evaluate_lm_eval.py \
  --checkpoint Qwen/Qwen2.5-0.5B \
  --model vllm \
  --tasks "${TASKS}" \
  --output-path "${OUT_ROOT}/chat" \
  --batch-size auto \
  --limit 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --model-arg max_model_len=8192 \
  --trust-remote-code \
  --confirm-run-unsafe-code \
  --apply-chat-template \
  --log-samples
