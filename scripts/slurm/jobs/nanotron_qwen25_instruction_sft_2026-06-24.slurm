#!/usr/bin/env bash
#SBATCH --job-name=nano_q25_instr
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-10%2
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
export HF_HUB_ENABLE_HF_TRANSFER=0

CHECKPOINT_STEP="${CHECKPOINT_STEP:-8192}"
HF_NAMESPACE="${HF_NAMESPACE:-flaitenberger}"
HF_REPO_PREFIX="${HF_REPO_PREFIX:-qwen25-7b-logic-cot-midtrain}"
HF_PRIVATE="${HF_PRIVATE:-1}"

MIX_KINDS=(control logic logic logic logic logic nl_exact nl_exact nl_exact nl_exact nl_exact)
MIX_PCTS=(0 5 10 15 20 25 5 10 15 20 25)

IDX="${SLURM_ARRAY_TASK_ID}"
MIX_KIND="${MIX_KINDS[$IDX]}"
MIX_PCT="${MIX_PCTS[$IDX]}"
BASE_RUN_TAG="${BASE_RUN_TAG_OVERRIDE:-qwen25_7b_midtrain_${MIX_KIND}_p${MIX_PCT}_4p3b}"
BASE_REPO_ID="${HF_NAMESPACE}/${HF_REPO_PREFIX}-${MIX_KIND}-p${MIX_PCT}-step${CHECKPOINT_STEP}"
RUN_NAME="sft_instr_${BASE_RUN_TAG}_step${CHECKPOINT_STEP}_ultrachat"
OUT_DIR="${HPCVAULT}/synthetic-RLVL/runs/${RUN_NAME}"
ADAPTER_REPO_ID="${HF_NAMESPACE}/${HF_REPO_PREFIX}-${MIX_KIND}-p${MIX_PCT}-step${CHECKPOINT_STEP}-ultrachat-adapter"

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="qwen25_midtrain_instruction_sft"
export WANDB_RUN_GROUP="${WANDB_GROUP}"

if [[ -f "${OUT_DIR}/final/adapter_config.json" && "${FORCE_INSTRUCTION_SFT:-0}" != "1" ]]; then
  echo "instruction adapter exists for ${RUN_NAME}; skipping SFT"
else
  python scripts/train_instruction_sft.py \
    --model "${BASE_REPO_ID}" \
    --dataset "${INSTRUCTION_DATASET:-HuggingFaceH4/ultrachat_200k}" \
    --train-split "${TRAIN_SPLIT:-train_sft}" \
    --eval-split "${EVAL_SPLIT:-test_sft}" \
    --run-name "${RUN_NAME}" \
    --output-dir "${OUT_DIR}" \
    --seed "${SEED:-3407}" \
    --train-samples "${TRAIN_SAMPLES:-50000}" \
    --eval-samples "${EVAL_SAMPLES:-512}" \
    --max-length "${SFT_MAX_LENGTH:-4096}" \
    --max-steps "${INSTRUCTION_MAX_STEPS:-10000}" \
    --lr "${INSTRUCTION_LR:-1e-4}" \
    --warmup-steps "${INSTRUCTION_WARMUP_STEPS:-50}" \
    --per-device-batch-size "${PER_DEVICE_BATCH_SIZE:-1}" \
    --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}" \
    --grad-accum "${INSTRUCTION_GRAD_ACCUM:-1}" \
    --eval-steps "${EVAL_STEPS:-1000}" \
    --save-steps "${SAVE_STEPS:-1000}" \
    --save-total-limit "${SAVE_TOTAL_LIMIT:-3}" \
    --format-mode "${INSTRUCTION_FORMAT_MODE:-chat}" \
    --resume-from-checkpoint auto \
    --gradient-checkpointing
fi

if [[ "${PUSH_INSTRUCTION_ADAPTER:-1}" == "1" ]]; then
  python - <<PY
import os
from pathlib import Path
from huggingface_hub import HfApi

repo_id = "${ADAPTER_REPO_ID}"
folder = Path("${OUT_DIR}") / "final"
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not token:
    raise SystemExit("Missing HF_TOKEN/HUGGINGFACE_HUB_TOKEN for adapter upload")
api = HfApi(token=token)
api.create_repo(repo_id=repo_id, private=bool(int("${HF_PRIVATE}")), exist_ok=True)
api.upload_folder(repo_id=repo_id, folder_path=str(folder), path_in_repo="", commit_message="Upload instruction adapter")
print(f"Uploaded instruction adapter to {repo_id}")
PY
fi
