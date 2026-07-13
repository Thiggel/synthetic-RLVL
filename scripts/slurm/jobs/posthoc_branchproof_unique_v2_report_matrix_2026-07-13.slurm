#!/usr/bin/env bash
#SBATCH --job-name=eval_bp_v2_report
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-0%3
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

MATRIX_GROUP="${MATRIX_GROUP:?Set MATRIX_GROUP to a report-matrix group}"
ROW="$(${VENV_PATH}/bin/python scripts/experiments/branchproof_unique_v2_report_matrix.py \
  --group "${MATRIX_GROUP}" --kind eval --index "${SLURM_ARRAY_TASK_ID}")"
IFS=$'\t' read -r TRAIN_INDEX FAMILY RUN_NAME TRAIN_TEMPLATE EVAL_TEMPLATES SEED TRAIN_MAX TRAIN_SUBSET \
  MODEL_NAME MAX_STEPS PER_DEVICE_BATCH_SIZE GRAD_ACCUM BALANCED DATA_KIND SHORTCUT_RATE SHORTCUT_KIND \
  MAX_LENGTH EVAL_TEMPLATE OUTPUT_TAG <<< "${ROW}"

SFT_FINAL="${HPCVAULT}/synthetic-RLVL/runs/${RUN_NAME}/final"
MERGED_ROOT="${HPCVAULT}/synthetic-RLVL/tmp/branchproof_unique_v2_report_${RUN_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
MERGED="${MERGED_ROOT}/merged"
OUT_DIR="${HPCVAULT}/synthetic-RLVL/passk_eval/branchproof_unique_v2_report_20260713/${MATRIX_GROUP}"
OUT="${OUT_DIR}/${OUTPUT_TAG}_passk.json"
SAMPLES_OUT="${OUT_DIR}/${OUTPUT_TAG}_samples.jsonl"
DATASET_ID="flaitenberger/BranchProof-unique-v2"
LOCAL_ROOT="null"
if [[ "${DATA_KIND}" != "base" ]]; then
  DATASET_ID="null"
  LOCAL_ROOT="${HPCVAULT}/synthetic-RLVL/datasets/branchproof_unique_v2_${DATA_KIND}_20260713"
fi

PASSK_STEPS_RESOLVED="${PASSK_STEPS_COLON:-1:2:5:10:12:15:18:20:25:30:35:40:45:50}"
PASSK_STEPS_RESOLVED="${PASSK_STEPS_RESOLVED//:/,}"
PASSK_K_VALUES_RESOLVED="${PASSK_K_VALUES_COLON:-1:2:4:8:16}"
PASSK_K_VALUES_RESOLVED="${PASSK_K_VALUES_RESOLVED//:/,}"
EXPECTED_PROMPTS=$((14 * ${PASSK_SAMPLES_PER_STEP:-32}))
EXPECTED_SAMPLED=$((EXPECTED_PROMPTS * ${PASSK_RETAINED_SAMPLES_PER_PROMPT:-1}))
EXPECTED_ROWS=$((EXPECTED_SAMPLED + ${PASSK_COLLECT_SAMPLES:-128}))

cleanup() {
  rm -rf "${MERGED_ROOT}"
}
trap cleanup EXIT

if [[ ! -s "${SFT_FINAL}/adapter_config.json" ]]; then
  echo "Missing SFT adapter: ${SFT_FINAL}" >&2
  exit 2
fi
mkdir -p "${OUT_DIR}" "${MERGED_ROOT}"

if [[ -s "${OUT}" && -s "${SAMPLES_OUT}" && "${FORCE_PASSK_EVAL:-0}" != "1" ]]; then
  if python - "${OUT}" "${SAMPLES_OUT}" "${EXPECTED_PROMPTS}" "${PASSK_NUM_GENERATIONS:-16}" \
      "${EXPECTED_ROWS}" "${EXPECTED_SAMPLED}" <<'PY'
import json
import sys
from pathlib import Path

metrics_path, samples_path = map(Path, sys.argv[1:3])
expected_prompts, expected_generations, expected_rows, expected_sampled = map(int, sys.argv[3:])
payload = json.loads(metrics_path.read_text(encoding="utf-8"))
metrics = payload.get("metrics", {})
rows = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
sampled = [row for row in rows if row.get("source") == "synthetic_sampled"]
if metrics.get("posthoc/prompts") != expected_prompts:
    raise SystemExit(1)
if metrics.get("posthoc/sampled_generations_per_prompt") != expected_generations:
    raise SystemExit(1)
if len(rows) != expected_rows or len(sampled) != expected_sampled:
    raise SystemExit(1)
PY
  then
    echo "Complete corrected report output exists: ${OUT}; skipping"
    exit 0
  fi
  echo "Existing output failed the corrected artifact gate: ${OUT}" >&2
  exit 3
fi

python scripts/merge_lora_checkpoint.py \
  --checkpoint "${SFT_FINAL}" \
  --output "${MERGED}.tmp" \
  --dtype bf16 \
  --overwrite
mv "${MERGED}.tmp" "${MERGED}"

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="branchproof_unique_v2_report_eval/${FAMILY}/${EVAL_TEMPLATE}_train1to${TRAIN_MAX}"
export WANDB_RUN_GROUP="${WANDB_GROUP}"

python scripts/evaluate_checkpoint_passk.py \
  --checkpoint "${MERGED}" \
  --config conf/sft_hard_fsa_schema_fixedtarget.yaml \
  --profile sft \
  --backend vllm \
  --steps "${PASSK_STEPS_RESOLVED}" \
  --samples-per-step "${PASSK_SAMPLES_PER_STEP:-32}" \
  --num-generations "${PASSK_NUM_GENERATIONS:-16}" \
  --k-values "${PASSK_K_VALUES_RESOLVED}" \
  --temperature "${PASSK_TEMPERATURE:-1.0}" \
  --max-new-tokens "${PASSK_MAX_NEW_TOKENS:-7168}" \
  --batch-size "${PASSK_BATCH_SIZE:-64}" \
  --gpu-memory-utilization "${PASSK_GPU_MEMORY_UTILIZATION:-0.88}" \
  --vllm-max-model-len "${PASSK_VLLM_MAX_MODEL_LEN:-16384}" \
  --scoring-log-interval 100 \
  --collect-samples "${PASSK_COLLECT_SAMPLES:-128}" \
  --collect-sampled-samples "${EXPECTED_SAMPLED}" \
  --stop-strings '</answer>' \
  --output "${OUT}" \
  --samples-output "${SAMPLES_OUT}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-run-name "${OUTPUT_TAG}_passk" \
  task.template="${EVAL_TEMPLATE}" \
  task.shortcut_rate=0.0 \
  task.shortcut_kind="${SHORTCUT_KIND}" \
  task.train_max_step="${TRAIN_MAX}" \
  task.val_max_step=50 \
  data.materialized.dataset_id="${DATASET_ID}" \
  data.materialized.local_root="${LOCAL_ROOT}" \
  data.materialized.train_subset="${TRAIN_SUBSET}" \
  validation.max_new_tokens="${PASSK_MAX_NEW_TOKENS:-7168}" \
  validation.vllm_max_model_len="${PASSK_VLLM_MAX_MODEL_LEN:-16384}"

python - "${OUT}" "${SAMPLES_OUT}" "${EXPECTED_PROMPTS}" "${PASSK_NUM_GENERATIONS:-16}" \
    "${EXPECTED_ROWS}" "${EXPECTED_SAMPLED}" <<'PY'
import json
import sys
from pathlib import Path

metrics_path, samples_path = map(Path, sys.argv[1:3])
expected_prompts, expected_generations, expected_rows, expected_sampled = map(int, sys.argv[3:])
payload = json.loads(metrics_path.read_text(encoding="utf-8"))
metrics = payload.get("metrics", {})
rows = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
sampled = [row for row in rows if row.get("source") == "synthetic_sampled"]
if metrics.get("posthoc/prompts") != expected_prompts:
    raise SystemExit("wrong prompt count")
if metrics.get("posthoc/sampled_generations_per_prompt") != expected_generations:
    raise SystemExit("wrong generation count")
if len(rows) != expected_rows or len(sampled) != expected_sampled:
    raise SystemExit("wrong retained-sample shape")
if not any(row.get("generation") for row in sampled):
    raise SystemExit("sample artifact has no generated text")
print(f"Accepted {metrics_path}: {expected_prompts} prompts, {expected_sampled} retained samples")
PY

cleanup
trap - EXIT
