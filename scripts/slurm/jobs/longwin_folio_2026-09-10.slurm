#!/usr/bin/env bash
# FOLIO greedy readout (2026-09-10): human-written first-order-logic premises
# with a balanced three-way label, as an external deduction benchmark next to
# ProofWriter. Same wrapper, arms and gates as the graded-deduction readout.
#SBATCH --job-name=longwin_folio
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=a0531,a0532,a0533,a0934,a0534
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-4%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Graded-deduction readout for the LONG-WINDOW (8192) depth-25 midtrain arms.
#
# Cloned verbatim from the accepted graded_deduction_full_2026-08-26.slurm
# (which produced the first positive transfer result on the mixdepth SFT
# models). Deliberate diffs vs that reference, everything else identical:
#   - five long-window arms, ONE seed, instead of 5 conditions x 2 seeds.
#   - checkpoints come from $WORK/post_sft_dolci_longwin_20260826/ (the
#     long-window post-SFT finals), not $HPCVAULT/post_sft_reasoning_mixture.
#   - adds a0531/a0533 to --exclude (repeated NODE_FAILs without requeue).
# Same 15 tasks, same dataset (graded_deduction_eval_20260826), same metric
# key `exact_match,none`, same vllm settings -> results are directly
# comparable to the mixdepth readout.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"
source ./scripts/env.sh

CONDITIONS=(control longdoc logic_band25 nl_exact_band25 condensed_logic_band25 nl_scrambled_band25)  # index 5 is the shuffled-order control
CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
# Seed replication (2026-09-08): SEED=3407 keeps the original names; any other
# seed appends _seed<SEED> to the checkpoint name and to every output root.
SEED="${SEED:-3407}"
if [[ "${SEED}" == "3407" ]]; then RUN_TAG_SUFFIX=""; else RUN_TAG_SUFFIX="_seed${SEED}"; fi
# MIX_TAG is empty for the 10-percent arms and "_p30" for the 30-percent
# arms. It goes into the checkpoint name and into every output root, so the
# two sweeps can never write to the same place.
MIX_TAG="${MIX_TAG:-}"
RUN_NAME="qwen25_7b_longwin_${CONDITION}${MIX_TAG}_2p5b_dolci_100k_lr5em6${RUN_TAG_SUFFIX}"
CHECKPOINT="${WORK}/synthetic-RLVL/post_sft_dolci_longwin_20260826/${RUN_NAME}/final"
OUT_DIR="${HPCVAULT}/synthetic-RLVL/lm_eval_results/qwen25_longwin_folio_20260910${MIX_TAG}${RUN_TAG_SUFFIX}/${RUN_NAME}"

VENV_PATH="${HPCVAULT}/.venv_rlvl_posttrain"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="qwen25_longwin_folio_20260910"
export WANDB_RUN_GROUP="${WANDB_GROUP}"

TASKS="synthrlvl_folio"

# Fail closed: never evaluate a checkpoint that is missing or half-written.
test -s "${CHECKPOINT}/config.json"
python scripts/validate_lm_eval_tasks.py --include-path lm_eval_tasks/synthrlvl_ood --tasks "${TASKS}"

python scripts/evaluate_lm_eval.py \
  --checkpoint "${CHECKPOINT}" \
  --model vllm \
  --tasks "${TASKS}" \
  --include-path lm_eval_tasks/synthrlvl_ood \
  --output-path "${OUT_DIR}" \
  --batch-size auto \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --model-arg "max_model_len=8192" \
  --trust-remote-code \
  --confirm-run-unsafe-code \
  --apply-chat-template \
  --log-samples \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-run-name "${RUN_NAME}_folio"
