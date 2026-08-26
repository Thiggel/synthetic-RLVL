#!/usr/bin/env bash
#SBATCH --job-name=q25_mixdepth_eval
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=a0532,a0934,a0534
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-9%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Greedy readout for the depth-threshold transfer test at SFT scale
# (post_sft_reasoning_mixture_20260821, 5 conditions x 2 seeds).
#
# Same suites, tasks and audits as the accepted docpack-rerun readout
# (scripts/slurm/jobs/qwen25_docpack_rerun_threeway_post_sft_eval_2026-08-14.slurm);
# only the checkpoint root, run names and result root change.
# Set EVAL_SUITE=standard or multihop.
#
# Array map: idx % 5 -> condition (control, logic_band15, nl_exact_band15,
#                                  logic_band25, nl_exact_band25)
#            idx / 5 -> seed (3407, 3408)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"

source ./scripts/env.sh

CONDITIONS=(control logic_band15 nl_exact_band15 logic_band25 nl_exact_band25)
SEEDS=(3407 3408)
CONDITION="${CONDITIONS[$((SLURM_ARRAY_TASK_ID % 5))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID / 5))]}"
RUN_NAME="qwen25_7b_mixdepth_${CONDITION}_seed${SEED}"
POST_SFT_ROOT="${POST_SFT_OUT_ROOT:-${HPCVAULT}/synthetic-RLVL/post_sft_reasoning_mixture_20260821}"
CHECKPOINT="${POST_SFT_ROOT}/${RUN_NAME}/final"
SUITE="${EVAL_SUITE:?Set EVAL_SUITE to standard or multihop}"
OUT_ROOT="${OUT_ROOT_OVERRIDE:-${HPCVAULT}/synthetic-RLVL/lm_eval_results/qwen25_mixdepth_post_sft_20260824}"
OUT_DIR="${OUT_ROOT}/${RUN_NAME}_${SUITE}"

VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="qwen25_mixdepth_post_sft/${SUITE}"
export WANDB_RUN_GROUP="${WANDB_GROUP}"

test -s "${CHECKPOINT}/config.json"
python scripts/nanotron/verify_qwen2_hf_checkpoint.py \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUT_DIR}_checkpoint_audit.json"

if [[ "${SUITE}" == "standard" ]]; then
  TASKS="${STANDARD_TASKS:-gsm8k,arc_challenge,hellaswag,winogrande,piqa,agieval_logiqa_en,hendrycks_math500,bbh,mmlu,mmlu_pro}"
  INCLUDE_ARGS=()
  MAX_MODEL_LEN="${STANDARD_MAX_MODEL_LEN:-8192}"
  AUDIT_SCRIPT="scripts/analysis/audit_nanotron_downstream_eval.py"
elif [[ "${SUITE}" == "multihop" ]]; then
  TASKS="${MULTIHOP_TASKS:-synthrlvl_longbench_hotpotqa_tagged,synthrlvl_longbench_2wikimqa_tagged,synthrlvl_longbench_musique_tagged,synthrlvl_longbench_hotpotqa_standard,synthrlvl_longbench_2wikimqa_standard,synthrlvl_longbench_musique_standard}"
  INCLUDE_ARGS=(--include-path lm_eval_tasks/synthrlvl_ood)
  MAX_MODEL_LEN="${MULTIHOP_MAX_MODEL_LEN:-32768}"
  AUDIT_SCRIPT="scripts/analysis/audit_nanotron_multihop_eval.py"
else
  echo "Unknown EVAL_SUITE=${SUITE}; expected standard or multihop" >&2
  exit 2
fi

python scripts/validate_lm_eval_tasks.py "${INCLUDE_ARGS[@]}" --tasks "${TASKS}"
if [[ -d "${OUT_DIR}" && "${FORCE_LM_EVAL:-0}" != "1" ]]; then
  if python "${AUDIT_SCRIPT}" --run-dir "${OUT_DIR}" --mode instruction --tasks "${TASKS}"; then
    echo "Audited result exists: ${OUT_DIR}; skipping"
    exit 0
  fi
  mv "${OUT_DIR}" "${OUT_DIR}.incomplete.${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi

LIMIT_ARGS=()
AUDIT_LIMIT_ARGS=()
if [[ -n "${LM_EVAL_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LM_EVAL_LIMIT}")
  AUDIT_LIMIT_ARGS=(--allow-limit)
fi

python scripts/evaluate_lm_eval.py \
  --checkpoint "${CHECKPOINT}" \
  --model vllm \
  --tasks "${TASKS}" \
  "${INCLUDE_ARGS[@]}" \
  --output-path "${OUT_DIR}" \
  --batch-size "${LM_EVAL_BATCH_SIZE:-auto}" \
  "${LIMIT_ARGS[@]}" \
  --dtype bfloat16 \
  --gpu-memory-utilization "${LM_EVAL_GPU_MEMORY_UTILIZATION:-0.80}" \
  --model-arg "max_model_len=${MAX_MODEL_LEN}" \
  --trust-remote-code \
  --confirm-run-unsafe-code \
  --apply-chat-template \
  --log-samples \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-run-name "${RUN_NAME}_${SUITE}"

python "${AUDIT_SCRIPT}" \
  --run-dir "${OUT_DIR}" \
  --mode instruction \
  --tasks "${TASKS}" \
  "${AUDIT_LIMIT_ARGS[@]}"
