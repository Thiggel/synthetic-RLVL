#!/usr/bin/env bash
#SBATCH --job-name=q25_3sft_passk
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=a0532,a0934
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-5%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Real-benchmark pass@k / maj@k readout for the three Dolmino post-SFT
# checkpoints (control / logic / nl_exact), replaying the accepted greedy
# lm-eval prompts with n=16 sampled generations + 1 greedy via vLLM.
#
# Array map: idx % 3 -> condition (control, logic, nl_exact)
#            idx / 3 -> task group (0 = standard [gsm8k+math500, 8k window],
#                                   1 = multihop [3 tagged LongBench, 32k window])
# %3 keeps concurrent GPU usage <= 3.
#
# Smoke override: sbatch --array=1,4 --time=02:00:00 \
#   --export=ALL,PASSK_LIMIT=8,PASSK_OUT_ROOT=$WORK/synthetic-RLVL/tmp/passk_smoke \
#   scripts/slurm/jobs/qwen25_dolmino_threeway_passk_eval_2026-08-06.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

CONDITIONS=(control logic nl_exact)
GROUPS_LIST=(standard multihop)
CONDITION="${CONDITIONS[$((SLURM_ARRAY_TASK_ID % 3))]}"
TASK_GROUP="${GROUPS_LIST[$((SLURM_ARRAY_TASK_ID / 3))]}"

OUT_ROOT="${PASSK_OUT_ROOT:-${HPCVAULT}/synthetic-RLVL/lm_eval_results/qwen25_dolmino_post_sft_passk_20260806}"

VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

LIMIT_ARGS=()
if [[ -n "${PASSK_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${PASSK_LIMIT}")
fi

echo "condition=${CONDITION} task_group=${TASK_GROUP} out_root=${OUT_ROOT} limit=${PASSK_LIMIT:-none}"

python scripts/analysis/evaluate_real_passk.py run \
  --condition "${CONDITION}" \
  --task-group "${TASK_GROUP}" \
  --output-root "${OUT_ROOT}" \
  "${LIMIT_ARGS[@]}"
