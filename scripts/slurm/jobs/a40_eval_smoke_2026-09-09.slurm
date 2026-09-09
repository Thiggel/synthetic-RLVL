#!/usr/bin/env bash
#SBATCH --job-name=a40_eval_smoke
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:50:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Can the 1-GPU evaluation stack run on a40 (48 GB, sm_86) instead of a100?
#
# Motivation: the REINST-CONFLUENT maintenance reservation (2026-09-08 to
# 2026-12-16, 56 nodes) has left the a100 partition with 34-53 h queue waits,
# while a40 is nearly empty. The midtrains need 80 GB cards and cannot move,
# but a 7B bf16 vLLM eval needs about 38 GB at gpu_memory_utilization 0.80 and
# should fit on a 48 GB a40. If it does, the readouts stop competing with the
# midtrains for the project GPU budget.
#
# v2: the v1 attempt failed for a reason unrelated to the hardware. It drove
# vllm from a heredoc, and vllm v1 spawns its EngineCore worker with the
# "spawn" start method, which re-imports __main__ from its path; with a
# heredoc that path is "<stdin>" and the worker dies with FileNotFoundError.
# The production evaluators are real files and never hit this, so v2 simply
# tests the production evaluator itself.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"
source ./scripts/env.sh

VENV_PATH="${HPCVAULT}/.venv_rlvl_posttrain"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

CKPT="${WORK}/synthetic-RLVL/post_sft_dolci_longwin_20260826/qwen25_7b_longwin_control_2p5b_dolci_100k_lr5em6/final"
test -s "${CKPT}/config.json"
OUT="${WORK}/synthetic-RLVL/tmp/a40_smoke_${SLURM_JOB_ID}"

echo "=== stage 1: the production graded-deduction evaluator, 16 items ==="
python scripts/evaluate_lm_eval.py \
  --checkpoint "${CKPT}" \
  --model vllm \
  --tasks synthrlvl_deduction_pw_d3 \
  --include-path lm_eval_tasks/synthrlvl_ood \
  --output-path "${OUT}" \
  --batch-size auto \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --model-arg "max_model_len=8192" \
  --trust-remote-code \
  --confirm-run-unsafe-code \
  --apply-chat-template \
  --log-samples \
  --limit 16

echo "=== stage 2: the production sampled evaluator, 4 docs x 16 samples ==="
# This is the path that generates at 8192 context with a 2048-token cap, so it
# is the memory-hungriest thing the readouts do on one GPU.
python scripts/analysis/evaluate_real_passk.py run \
  --condition a40_smoke \
  --task-group deduction_cot \
  --checkpoint "${CKPT}" \
  --accepted-run-dir "${HPCVAULT}/synthetic-RLVL/lm_eval_results/qwen25_longwin_graded_deduction_20260906/qwen25_7b_longwin_control_2p5b_dolci_100k_lr5em6" \
  --output-root "${OUT}_passk" \
  --limit 4

echo "=== peak GPU memory ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

echo "A40 EVAL SMOKE PASSED"
