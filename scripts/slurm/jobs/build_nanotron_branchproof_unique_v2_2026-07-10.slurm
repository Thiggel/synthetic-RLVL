#!/usr/bin/env bash
#SBATCH --job-name=build_nano_bp_v2
#SBATCH --partition=a40,rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs tmp
source ./scripts/env.sh

export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
source "${WORK}/nanotron/bin/activate"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}"
PROOF_TOKEN_BUDGET="${PROOF_TOKEN_BUDGET:-1200000000}"
PREPROCESS_TASKS="${PREPROCESS_TASKS:-16}"
DATA_ROOT="${DATA_ROOT:-${HPCVAULT}/synthetic-RLVL/nanotron_data/qwen25_branchproof_unique_v2}"
NANOSET_ROOT="${NANOSET_ROOT:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25_branchproof_unique_v2}"
TEMPLATES=(logic nl_exact)
TEMPLATE="${TEMPLATES[$SLURM_ARRAY_TASK_ID]}"
JSONL="${DATA_ROOT}/${TEMPLATE}.jsonl"
NANOSET="${NANOSET_ROOT}/${TEMPLATE}"

mkdir -p "${DATA_ROOT}" "${NANOSET_ROOT}"

python scripts/analysis/probe_hard_fsa_schema.py \
  --train-shortcut-rate 0.0 \
  --eval-shortcut-rate 0.0 \
  --seed 3407 \
  --branching-factor 4 \
  --train-depth-max 25 \
  --eval-depth-max 50 \
  --n-per-depth "${PROBE_N_PER_DEPTH:-8}" \
  --output "tmp/nanotron_branchproof_unique_v2_probe_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

python scripts/data/export_generated_proof_jsonl.py \
  --template "${TEMPLATE}" \
  --output "${JSONL}" \
  --tokenizer "${MODEL_ID}" \
  --target-tokens "${PROOF_TOKEN_BUDGET}" \
  --seed 3407 \
  --train-min-step 1 \
  --train-max-step 25 \
  --difficulty hard_fsa_schema \
  --branching-factor 4 \
  --resume \
  --progress-every 5000

MODEL_ID="${MODEL_ID}" JSONL="${JSONL}" python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["JSONL"])
manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
assert manifest["done"]
first = json.loads(path.open().readline())["text"]
assert "c0" in first
print(json.dumps({"manifest": manifest, "first_record_prefix": first[:2000]}, indent=2))
PY

EOS_TOKEN="$(MODEL_ID="${MODEL_ID}" python - <<'PY'
import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_ID"])
print(tokenizer.eos_token or "")
PY
)"
EOS_ARGS=()
if [[ -n "${EOS_TOKEN}" ]]; then
  EOS_ARGS=(--eos-token "${EOS_TOKEN}")
fi

rm -rf "${NANOSET}.tmp"
mkdir -p "${NANOSET}.tmp"
(
  cd ../nanotron
  python tools/preprocess_data.py \
    --tokenizer-name-or-path "${MODEL_ID}" \
    "${EOS_ARGS[@]}" \
    --output-folder "${NANOSET}.tmp" \
    --logging-dir "${NANOSET}.tmp/logs" \
    --n-tasks "${PREPROCESS_TASKS}" \
    jsonl --dataset "${JSONL}" --column text
)
rm -rf "${NANOSET}"
mv "${NANOSET}.tmp" "${NANOSET}"
compgen -G "${NANOSET}/*.metadata" >/dev/null
