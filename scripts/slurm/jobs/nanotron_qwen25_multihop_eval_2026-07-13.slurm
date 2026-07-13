#!/usr/bin/env bash
#SBATCH --job-name=nano_q25_mhop
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-2%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

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

CHECKPOINT_STEP=8192
HF_NAMESPACE="${HF_NAMESPACE:-flaitenberger}"
EVAL_BRANCH="${EVAL_BRANCH:?Set EVAL_BRANCH to direct or instruction}"
TASKS="${MULTIHOP_TASKS:-synthrlvl_longbench_hotpotqa_tagged,synthrlvl_longbench_2wikimqa_tagged,synthrlvl_longbench_musique_tagged,synthrlvl_longbench_hotpotqa_standard,synthrlvl_longbench_2wikimqa_standard,synthrlvl_longbench_musique_standard}"
INCLUDE_PATH="${MULTIHOP_INCLUDE_PATH:-lm_eval_tasks/synthrlvl_ood}"

MIX_KINDS=(control logic nl_exact)
MIX_PCTS=(0 15 15)
IDX="${SLURM_ARRAY_TASK_ID}"
MIX_KIND="${MIX_KINDS[$IDX]}"
MIX_PCT="${MIX_PCTS[$IDX]}"
if [[ "${MIX_KIND}" == "control" ]]; then
  BASE_RUN_TAG="qwen25_7b_midtrain_control_p0_4p3b"
  BASE_REPO_ID="${HF_NAMESPACE}/qwen25-7b-logic-cot-midtrain-control-p0-step${CHECKPOINT_STEP}"
else
  BASE_RUN_TAG="qwen25_7b_midtrain_${MIX_KIND}_p${MIX_PCT}_bp_unique_v2_4p3b"
  BASE_REPO_ID="${HF_NAMESPACE}/qwen25-7b-branchproof-unique-v2-midtrain-${MIX_KIND}-p${MIX_PCT}-step${CHECKPOINT_STEP}"
fi
OUT_ROOT="${OUT_ROOT_OVERRIDE:-${HPCVAULT}/synthetic-RLVL/lm_eval_results/qwen25_branchproof_unique_v2_multihop_20260713}"

if [[ "${EVAL_BRANCH}" == "direct" ]]; then
  CHECKPOINT="${BASE_REPO_ID}"
  OUT_DIR="${OUT_ROOT}/${BASE_RUN_TAG}_step${CHECKPOINT_STEP}_direct"
  PROMPT_FORMAT_ARGS=()
elif [[ "${EVAL_BRANCH}" == "instruction" ]]; then
  RUN_NAME="sft_instr_${BASE_RUN_TAG}_step${CHECKPOINT_STEP}_ultrachat"
  ADAPTER="${HPCVAULT}/synthetic-RLVL/runs/${RUN_NAME}/final"
  if [[ ! -s "${ADAPTER}/adapter_config.json" ]]; then
    echo "Missing instruction adapter: ${ADAPTER}" >&2
    exit 2
  fi
  MERGED_ROOT="${HPCVAULT}/synthetic-RLVL/tmp/multihop_${RUN_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  CHECKPOINT="${MERGED_ROOT}/merged"
  cleanup() { rm -rf "${MERGED_ROOT}"; }
  trap cleanup EXIT
  mkdir -p "${MERGED_ROOT}"
  python scripts/merge_lora_checkpoint.py --checkpoint "${ADAPTER}" --output "${CHECKPOINT}.tmp" --dtype bf16 --overwrite
  mv "${CHECKPOINT}.tmp" "${CHECKPOINT}"
  OUT_DIR="${OUT_ROOT}/${BASE_RUN_TAG}_step${CHECKPOINT_STEP}_instruction"
  PROMPT_FORMAT_ARGS=(--apply-chat-template)
else
  echo "Unknown EVAL_BRANCH=${EVAL_BRANCH}" >&2
  exit 2
fi

python scripts/nanotron/verify_qwen2_rope_config.py --checkpoint "${CHECKPOINT}"
python scripts/validate_lm_eval_tasks.py --include-path "${INCLUDE_PATH}" --tasks "${TASKS}"

if [[ -d "${OUT_DIR}" && "${FORCE_LM_EVAL:-0}" != "1" ]]; then
  if python scripts/analysis/audit_nanotron_multihop_eval.py \
      --run-dir "${OUT_DIR}" --mode "${EVAL_BRANCH}" --tasks "${TASKS}"; then
    echo "Accepted multi-hop bundle exists: ${OUT_DIR}; skipping"
    exit 0
  fi
  INCOMPLETE_DIR="${OUT_DIR}.incomplete.${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  rm -rf "${INCOMPLETE_DIR}"
  mv "${OUT_DIR}" "${INCOMPLETE_DIR}"
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
  --include-path "${INCLUDE_PATH}" \
  --output-path "${OUT_DIR}" \
  --batch-size "${LM_EVAL_BATCH_SIZE:-auto}" \
  "${LIMIT_ARGS[@]}" \
  --dtype bfloat16 \
  --gpu-memory-utilization "${LM_EVAL_GPU_MEMORY_UTILIZATION:-0.80}" \
  --model-arg "max_model_len=${LM_EVAL_MAX_MODEL_LEN:-32768}" \
  --trust-remote-code \
  --confirm-run-unsafe-code \
  "${PROMPT_FORMAT_ARGS[@]}" \
  --log-samples \
  --wandb-project "${WANDB_PROJECT:-synthetic-rlvl}" \
  --wandb-group "qwen25_midtrain_multihop/${EVAL_BRANCH}" \
  --wandb-run-name "${BASE_RUN_TAG}_step${CHECKPOINT_STEP}_${EVAL_BRANCH}_multihop"

python scripts/analysis/audit_nanotron_multihop_eval.py \
  --run-dir "${OUT_DIR}" \
  --mode "${EVAL_BRANCH}" \
  --tasks "${TASKS}" \
  "${AUDIT_LIMIT_ARGS[@]}"

if [[ "${EVAL_BRANCH}" == "instruction" ]]; then
  cleanup
  trap - EXIT
fi
