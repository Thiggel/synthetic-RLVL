#!/usr/bin/env bash
#SBATCH --job-name=nano_q25_eval
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-10%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"

CHECKPOINT_STEP="${CHECKPOINT_STEP:-8192}"
HF_NAMESPACE="${HF_NAMESPACE:-flaitenberger}"
HF_REPO_PREFIX="${HF_REPO_PREFIX:-qwen25-7b-logic-cot-midtrain}"
EVAL_BRANCH="${EVAL_BRANCH:-direct}"
REVIEWER_TASKS="${REVIEWER_TASKS:-gsm8k,arc_challenge,hellaswag,winogrande,piqa,logiqa,folio,bbh,mmlu}"

MIX_KINDS=(control logic logic logic logic logic nl_exact nl_exact nl_exact nl_exact nl_exact)
MIX_PCTS=(0 5 10 15 20 25 5 10 15 20 25)

IDX="${SLURM_ARRAY_TASK_ID}"
MIX_KIND="${MIX_KINDS[$IDX]}"
MIX_PCT="${MIX_PCTS[$IDX]}"
BASE_RUN_TAG="${BASE_RUN_TAG_OVERRIDE:-qwen25_7b_midtrain_${MIX_KIND}_p${MIX_PCT}_4p3b}"
BASE_REPO_ID="${HF_NAMESPACE}/${HF_REPO_PREFIX}-${MIX_KIND}-p${MIX_PCT}-step${CHECKPOINT_STEP}"
OUT_ROOT="${OUT_ROOT_OVERRIDE:-${HPCVAULT}/synthetic-RLVL/lm_eval_results/qwen25_midtrain_downstream_20260624}"

if [[ "${EVAL_BRANCH}" == "direct" ]]; then
  CHECKPOINT="${BASE_REPO_ID}"
  OUT_DIR="${OUT_ROOT}/${BASE_RUN_TAG}_step${CHECKPOINT_STEP}_direct"
  WANDB_GROUP_VALUE="qwen25_midtrain_downstream/direct"
elif [[ "${EVAL_BRANCH}" == "instruction" ]]; then
  RUN_NAME="sft_instr_${BASE_RUN_TAG}_step${CHECKPOINT_STEP}_ultrachat"
  ADAPTER="${HPCVAULT}/synthetic-RLVL/runs/${RUN_NAME}/final"
  if [[ ! -f "${ADAPTER}/adapter_config.json" ]]; then
    echo "Missing instruction adapter: ${ADAPTER}" >&2
    exit 2
  fi
  MERGED_ROOT="${HPCVAULT}/synthetic-RLVL/tmp/eval_${RUN_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  CHECKPOINT="${MERGED_ROOT}/merged"
  cleanup_merged() { rm -rf "${MERGED_ROOT}"; }
  trap cleanup_merged EXIT
  mkdir -p "${MERGED_ROOT}"
  python scripts/merge_lora_checkpoint.py --checkpoint "${ADAPTER}" --output "${CHECKPOINT}.tmp" --dtype bf16 --overwrite
  mv "${CHECKPOINT}.tmp" "${CHECKPOINT}"
  OUT_DIR="${OUT_ROOT}/${BASE_RUN_TAG}_step${CHECKPOINT_STEP}_instruction"
  WANDB_GROUP_VALUE="qwen25_midtrain_downstream/instruction"
else
  echo "Unknown EVAL_BRANCH=${EVAL_BRANCH}; expected direct or instruction" >&2
  exit 2
fi

if [[ -d "${OUT_DIR}" && "${FORCE_LM_EVAL:-0}" != "1" ]]; then
  echo "lm-eval output dir exists: ${OUT_DIR}; skipping"
  exit 0
fi

LIMIT_ARGS=()
if [[ -n "${LM_EVAL_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LM_EVAL_LIMIT}")
fi

python scripts/evaluate_lm_eval.py \
  --checkpoint "${CHECKPOINT}" \
  --model "${LM_EVAL_MODEL:-vllm}" \
  --tasks "${REVIEWER_TASKS}" \
  --output-path "${OUT_DIR}" \
  --batch-size "${LM_EVAL_BATCH_SIZE:-auto}" \
  "${LIMIT_ARGS[@]}" \
  --dtype "${LM_EVAL_DTYPE:-bfloat16}" \
  --gpu-memory-utilization "${LM_EVAL_GPU_MEMORY_UTILIZATION:-0.80}" \
  --model-arg "max_model_len=${LM_EVAL_MAX_MODEL_LEN:-8192}" \
  --trust-remote-code \
  --confirm-run-unsafe-code \
  --log-samples \
  --wandb-project "${WANDB_PROJECT:-synthetic-rlvl}" \
  --wandb-group "${WANDB_GROUP_VALUE}" \
  --wandb-run-name "${BASE_RUN_TAG}_step${CHECKPOINT_STEP}_${EVAL_BRANCH}_reviewer"

if [[ "${EVAL_BRANCH}" == "instruction" ]]; then
  cleanup_merged
  trap - EXIT
fi
