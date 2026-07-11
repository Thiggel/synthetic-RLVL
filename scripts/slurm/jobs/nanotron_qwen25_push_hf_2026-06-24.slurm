#!/usr/bin/env bash
#SBATCH --job-name=nano_q25_pushhf
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-10%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export NANOTRON_ROOT="${NANOTRON_ROOT:-$(cd ../nanotron && pwd)}"
source "${WORK}/nanotron/bin/activate"

CHECKPOINT_STEP="${CHECKPOINT_STEP:-8192}"
HF_NAMESPACE="${HF_NAMESPACE:-flaitenberger}"
HF_REPO_PREFIX="${HF_REPO_PREFIX:-qwen25-7b-logic-cot-midtrain}"
HF_PRIVATE="${HF_PRIVATE:-1}"

MIX_KINDS=(control logic logic logic logic logic nl_exact nl_exact nl_exact nl_exact nl_exact)
MIX_PCTS=(0 5 10 15 20 25 5 10 15 20 25)

IDX="${SLURM_ARRAY_TASK_ID}"
MIX_KIND="${MIX_KINDS[$IDX]}"
MIX_PCT="${MIX_PCTS[$IDX]}"
RUN_TAG="${RUN_TAG_OVERRIDE:-qwen25_7b_midtrain_${MIX_KIND}_p${MIX_PCT}_4p3b}"
RUN_ROOT="${RUN_ROOT_OVERRIDE:-${HPCVAULT}/synthetic-RLVL/nanotron_midtrain/${RUN_TAG}}"
NANOTRON_CKPT="${RUN_ROOT}/checkpoints/${CHECKPOINT_STEP}"
HF_STAGING="${HPCVAULT}/synthetic-RLVL/hf_staging/${RUN_TAG}_step${CHECKPOINT_STEP}"
REPO_ID="${HF_NAMESPACE}/${HF_REPO_PREFIX}-${MIX_KIND}-p${MIX_PCT}-step${CHECKPOINT_STEP}"

if [[ ! -d "${NANOTRON_CKPT}/model" ]]; then
  echo "Missing Nanotron model checkpoint: ${NANOTRON_CKPT}" >&2
  exit 2
fi
MODEL_FILE_COUNT="$(find "${NANOTRON_CKPT}/model" -type f 2>/dev/null | wc -l)"
MODEL_SIZE_BYTES="$(du -sb "${NANOTRON_CKPT}/model" 2>/dev/null | cut -f1)"
if [[ ! -s "${NANOTRON_CKPT}/model_config.json" \
      || ! -s "${NANOTRON_CKPT}/checkpoint_metadata.json" \
      || "${MODEL_FILE_COUNT}" -ne "${EXPECTED_MODEL_FILES:-625}" \
      || "${MODEL_SIZE_BYTES:-0}" -lt "${MIN_MODEL_SIZE_BYTES:-15000000000}" ]] \
    || find "${NANOTRON_CKPT}/model" -type f -size 0 -print -quit 2>/dev/null | grep -q .; then
  echo "Missing or incomplete Nanotron model checkpoint: ${NANOTRON_CKPT}" >&2
  exit 2
fi

rm -rf "${HF_STAGING}.tmp"
python scripts/nanotron/convert_qwen2_nanotron_to_hf.py \
  --checkpoint-path "${NANOTRON_CKPT}" \
  --save-path "${HF_STAGING}.tmp" \
  --tokenizer-name "${MODEL_ID:-Qwen/Qwen2.5-7B}" \
  --repo-id "${REPO_ID}" \
  --token-env "${HF_TOKEN_ENV:-HF_TOKEN}" \
  $(if [[ "${HF_PRIVATE}" == "1" ]]; then echo "--private"; else echo "--no-private"; fi)

VERIFY_REPORT="${RUN_ROOT}/hf_verify_step${CHECKPOINT_STEP}.json"
python scripts/nanotron/verify_qwen2_hf_checkpoint.py \
  --checkpoint "${HF_STAGING}.tmp" \
  --repo-id "${REPO_ID}" \
  --token-env "${HF_TOKEN_ENV:-HF_TOKEN}" \
  --output "${VERIFY_REPORT}"

rm -rf "${HF_STAGING}"
mv "${HF_STAGING}.tmp" "${HF_STAGING}"
if [[ "${CLEANUP_LOCAL_HF_STAGING:-1}" == "1" ]]; then
  rm -rf "${HF_STAGING}"
fi

echo "${REPO_ID}" > "${RUN_ROOT}/hf_repo_step${CHECKPOINT_STEP}.txt"
if [[ "${CLEANUP_NANOTRON_RUN_CHECKPOINTS_AFTER_UPLOAD:-0}" == "1" ]]; then
  EXPECTED_RUN_PREFIX="${HPCVAULT}/synthetic-RLVL/nanotron_midtrain/"
  if [[ "${RUN_ROOT}" != "${EXPECTED_RUN_PREFIX}"* ]]; then
    echo "Refusing checkpoint cleanup outside ${EXPECTED_RUN_PREFIX}: ${RUN_ROOT}" >&2
    exit 2
  fi
  rm -rf -- "${RUN_ROOT}/checkpoints"
elif [[ "${CLEANUP_NANOTRON_AFTER_UPLOAD:-0}" == "1" ]]; then
  rm -rf "${NANOTRON_CKPT}"
fi
echo "Uploaded ${RUN_TAG} step ${CHECKPOINT_STEP} to ${REPO_ID}"
