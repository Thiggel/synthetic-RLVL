#!/usr/bin/env bash
#SBATCH --job-name=q25_docpack_passk
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=a0532,a0934
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-5%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Sampled pass@k / maj@k readout for the document-preserving 2.5B rerun,
# post-SFT.  Identical sampler, seed, n=16+greedy and audits as the accepted
# readout (scripts/slurm/jobs/qwen25_dolmino_threeway_passk_eval_2026-08-06.slurm);
# only the checkpoint root and the accepted-greedy prompt source change, so the
# rerun is reported on the same footing as the 5B full-document run.
#
# Array map: idx % 3 -> condition (control, logic, nl_exact)
#            idx / 3 -> task group (0 = standard, 1 = multihop)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"

source ./scripts/env.sh

CONDITIONS=(control logic nl_exact)
GROUPS_LIST=(standard multihop)
CONDITION="${CONDITIONS[$((SLURM_ARRAY_TASK_ID % 3))]}"
TASK_GROUP="${GROUPS_LIST[$((SLURM_ARRAY_TASK_ID / 3))]}"

RUN_NAME="qwen25_7b_dolmino_${CONDITION}_docpack_2p5b_dolci_100k_lr5em6"
POST_SFT_ROOT="${POST_SFT_OUT_ROOT:-${WORK}/synthetic-RLVL/post_sft_dolci_docpack_rerun_20260814}"
CHECKPOINT="${POST_SFT_ROOT}/${RUN_NAME}/final"
ACCEPTED_ROOT="${ACCEPTED_ROOT_OVERRIDE:-${HPCVAULT}/synthetic-RLVL/lm_eval_results/qwen25_docpack_rerun_post_sft_20260814}"
ACCEPTED_RUN_DIR="${ACCEPTED_ROOT}/${RUN_NAME}_${TASK_GROUP}"
OUT_ROOT="${PASSK_OUT_ROOT:-${HPCVAULT}/synthetic-RLVL/lm_eval_results/qwen25_docpack_rerun_post_sft_passk_20260818}"

VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Fail closed: the sampled readout replays the accepted greedy prompts, so the
# greedy bundle for this exact condition/suite must already exist.
test -s "${CHECKPOINT}/config.json"
test -d "${ACCEPTED_RUN_DIR}"

LIMIT_ARGS=()
if [[ -n "${PASSK_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${PASSK_LIMIT}")
fi

echo "condition=${CONDITION} task_group=${TASK_GROUP} ckpt=${CHECKPOINT} accepted=${ACCEPTED_RUN_DIR} out_root=${OUT_ROOT} limit=${PASSK_LIMIT:-none}"

python scripts/analysis/evaluate_real_passk.py run \
  --condition "${CONDITION}" \
  --task-group "${TASK_GROUP}" \
  --checkpoint "${CHECKPOINT}" \
  --accepted-run-dir "${ACCEPTED_RUN_DIR}" \
  --output-root "${OUT_ROOT}" \
  "${LIMIT_ARGS[@]}"
