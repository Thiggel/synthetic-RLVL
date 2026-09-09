#!/usr/bin/env bash
#SBATCH --job-name=longwin_passk
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=a0531,a0532,a0533,a0934,a0534
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-4%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Sampled pass@k / maj@k readout for the LONG-WINDOW (8192) depth-25 arms,
# BranchProof-CoT deduction only.
#
# WHY CoT ONLY: the greedy readout (2026-09-06) put nl_exact above logic on
# bp_cot by -0.21/-0.17/-0.10 at d5/d10/d25. Greedy is known to understate the
# formal condition (synthetic crossover is at band 15 sampled vs band 25
# greedy), and the mixdepth campaign showed a greedy crater 0.085 -> 0.455 at
# pass@16. This run tests whether the formal deficit is a greedy artifact.
# The pw_* tasks are excluded on purpose: max_gen_toks=16 with until=["\n"]
# makes them short-answer classification, where pass@16 over a 3-class label
# saturates and carries no information.
#
# Same sampler/seed/N_SAMPLES=16 as the accepted readouts. Prompts are replayed
# verbatim from the accepted greedy samples, and the deduction scorer imports
# process_deduction_bp_cot from the task utils (verified 2026-09-07 to
# reproduce all 15 published greedy exact_match values bit-for-bit).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"

source ./scripts/env.sh

CONDITIONS=(control longdoc logic_band25 nl_exact_band25 condensed_logic_band25)
CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
TASK_GROUP=deduction_cot

# Seed replication (2026-09-08): SEED=3407 keeps the original names; any other
# seed appends _seed<SEED> to the checkpoint name and to every output root.
SEED="${SEED:-3407}"
if [[ "${SEED}" == "3407" ]]; then RUN_TAG_SUFFIX=""; else RUN_TAG_SUFFIX="_seed${SEED}"; fi
# MIX_TAG is empty for the 10-percent arms and "_p30" for the 30-percent
# arms. It goes into the checkpoint name and into every output root, so the
# two sweeps can never write to the same place.
MIX_TAG="${MIX_TAG:-}"
RUN_NAME="qwen25_7b_longwin_${CONDITION}${MIX_TAG}_2p5b_dolci_100k_lr5em6${RUN_TAG_SUFFIX}"
LABEL="${CONDITION}"
CHECKPOINT="${WORK}/synthetic-RLVL/post_sft_dolci_longwin_20260826/${RUN_NAME}/final"
ACCEPTED_RUN_DIR="${HPCVAULT}/synthetic-RLVL/lm_eval_results/qwen25_longwin_graded_deduction_20260906${MIX_TAG}${RUN_TAG_SUFFIX}/${RUN_NAME}"
OUT_ROOT="${PASSK_OUT_ROOT:-${HPCVAULT}/synthetic-RLVL/lm_eval_results/qwen25_longwin_passk_20260907${MIX_TAG}${RUN_TAG_SUFFIX}}"

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
