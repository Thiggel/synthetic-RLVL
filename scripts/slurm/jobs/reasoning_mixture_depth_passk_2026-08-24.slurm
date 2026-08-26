#!/usr/bin/env bash
#SBATCH --job-name=q25_mixdepth_passk
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=a0532,a0934,a0534
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-19%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Sampled pass@k / maj@k readout for the depth-threshold transfer test
# (post_sft_reasoning_mixture_20260821, 5 conditions x 2 seeds).
# Identical sampler, seed, n=16+greedy and audits as the accepted rerun
# readout (scripts/slurm/jobs/qwen25_docpack_rerun_threeway_passk_eval_2026-08-18.slurm);
# only the checkpoint root, run names and greedy prompt source change.
#
# Array map: idx % 10 -> model (5 conditions x 2 seeds, condition-major)
#            idx / 10 -> task group (0 = standard, 1 = multihop)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"

source ./scripts/env.sh

CONDITIONS=(control logic_band15 nl_exact_band15 logic_band25 nl_exact_band25)
SEEDS=(3407 3408)
GROUPS_LIST=(standard multihop)
MODEL_IDX=$((SLURM_ARRAY_TASK_ID % 10))
CONDITION="${CONDITIONS[$((MODEL_IDX % 5))]}"
SEED="${SEEDS[$((MODEL_IDX / 5))]}"
TASK_GROUP="${GROUPS_LIST[$((SLURM_ARRAY_TASK_ID / 10))]}"

RUN_NAME="qwen25_7b_mixdepth_${CONDITION}_seed${SEED}"
LABEL="${CONDITION}_seed${SEED}"
POST_SFT_ROOT="${POST_SFT_OUT_ROOT:-${HPCVAULT}/synthetic-RLVL/post_sft_reasoning_mixture_20260821}"
CHECKPOINT="${POST_SFT_ROOT}/${RUN_NAME}/final"
ACCEPTED_ROOT="${ACCEPTED_ROOT_OVERRIDE:-${HPCVAULT}/synthetic-RLVL/lm_eval_results/qwen25_mixdepth_post_sft_20260824}"
ACCEPTED_RUN_DIR="${ACCEPTED_ROOT}/${RUN_NAME}_${TASK_GROUP}"
OUT_ROOT="${PASSK_OUT_ROOT:-${HPCVAULT}/synthetic-RLVL/lm_eval_results/qwen25_mixdepth_post_sft_passk_20260824}"

VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Fail closed: the sampled readout replays the accepted greedy prompts, so the
# greedy bundle for this exact model/suite must already exist.
test -s "${CHECKPOINT}/config.json"
test -d "${ACCEPTED_RUN_DIR}"

LIMIT_ARGS=()
if [[ -n "${PASSK_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${PASSK_LIMIT}")
fi

echo "condition=${LABEL} task_group=${TASK_GROUP} ckpt=${CHECKPOINT} accepted=${ACCEPTED_RUN_DIR} out_root=${OUT_ROOT} limit=${PASSK_LIMIT:-none}"

python scripts/analysis/evaluate_real_passk.py run \
  --condition "${LABEL}" \
  --task-group "${TASK_GROUP}" \
  --checkpoint "${CHECKPOINT}" \
  --accepted-run-dir "${ACCEPTED_RUN_DIR}" \
  --output-root "${OUT_ROOT}" \
  "${LIMIT_ARGS[@]}"
