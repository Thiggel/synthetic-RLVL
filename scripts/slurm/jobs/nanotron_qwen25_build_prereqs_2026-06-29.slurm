#!/usr/bin/env bash
#SBATCH --job-name=nano_q25_prereq
#SBATCH --partition=a40,rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

source "${WORK}/nanotron/bin/activate"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}"
NORMAL_DATASET="${NORMAL_DATASET:-HuggingFaceFW/fineweb-edu}"
NORMAL_CONFIG="${NORMAL_CONFIG:-sample-10BT}"
NORMAL_TOKEN_BUDGET="${NORMAL_TOKEN_BUDGET:-4800000000}"
PROOF_TOKEN_BUDGET="${PROOF_TOKEN_BUDGET:-1200000000}"
PREPROCESS_TASKS="${PREPROCESS_TASKS:-16}"

DATA_ROOT="${DATA_ROOT:-${HPCVAULT}/synthetic-RLVL/nanotron_data/qwen25}"
JSONL_ROOT="${JSONL_ROOT:-${DATA_ROOT}/jsonl}"
NANOSET_ROOT="${NANOSET_ROOT:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25}"
CONVERTED_CKPT="${CONVERTED_CKPT:-${HPCVAULT}/synthetic-RLVL/nanotron_checkpoints/qwen25_7b_tp1}"

NORMAL_JSONL="${JSONL_ROOT}/normal_continuation.jsonl"
LOGIC_JSONL="${JSONL_ROOT}/logical_deduction_logic.jsonl"
NL_JSONL="${JSONL_ROOT}/logical_deduction_nl_exact.jsonl"

mkdir -p "${JSONL_ROOT}" "${NANOSET_ROOT}" "$(dirname "${CONVERTED_CKPT}")"

jsonl_done() {
  local manifest="$1"
  python - "$manifest" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
payload = json.loads(path.read_text())
raise SystemExit(0 if payload.get("done") else 1)
PY
}

nanoset_done() {
  local path="$1"
  compgen -G "${path}/*.metadata" >/dev/null
}

echo "### Exporting token-budgeted JSONL sources"
if ! jsonl_done "${NORMAL_JSONL}.manifest.json"; then
  python scripts/data/export_hf_text_jsonl_token_budget.py \
    --dataset "${NORMAL_DATASET}" \
    --config "${NORMAL_CONFIG}" \
    --split train \
    --text-field text \
    --output "${NORMAL_JSONL}" \
    --tokenizer "${MODEL_ID}" \
    --target-tokens "${NORMAL_TOKEN_BUDGET}" \
    --resume \
    --progress-every 5000
else
  echo "Normal JSONL already complete: ${NORMAL_JSONL}"
fi

if ! jsonl_done "${LOGIC_JSONL}.manifest.json"; then
  python scripts/data/export_generated_proof_jsonl.py \
    --template logic \
    --output "${LOGIC_JSONL}" \
    --tokenizer "${MODEL_ID}" \
    --target-tokens "${PROOF_TOKEN_BUDGET}" \
    --seed 3407 \
    --train-min-step 1 \
    --train-max-step 25 \
    --difficulty hard_fsa_schema \
    --branching-factor 4 \
    --resume \
    --progress-every 5000
else
  echo "Logic JSONL already complete: ${LOGIC_JSONL}"
fi

if ! jsonl_done "${NL_JSONL}.manifest.json"; then
  python scripts/data/export_generated_proof_jsonl.py \
    --template nl_exact \
    --output "${NL_JSONL}" \
    --tokenizer "${MODEL_ID}" \
    --target-tokens "${PROOF_TOKEN_BUDGET}" \
    --seed 3407 \
    --train-min-step 1 \
    --train-max-step 25 \
    --difficulty hard_fsa_schema \
    --branching-factor 4 \
    --resume \
    --progress-every 5000
else
  echo "NL JSONL already complete: ${NL_JSONL}"
fi

EOS_TOKEN="$(MODEL_ID="${MODEL_ID}" python - <<'PY'
import os
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.environ["MODEL_ID"])
print(tok.eos_token or "")
PY
)"
EOS_ARGS=()
if [[ -n "${EOS_TOKEN}" ]]; then
  EOS_ARGS=(--eos-token "${EOS_TOKEN}")
fi

preprocess_jsonl() {
  local jsonl="$1"
  local output="$2"
  if nanoset_done "${output}"; then
    echo "Nanoset already complete: ${output}"
    return
  fi
  rm -rf "${output}.tmp"
  mkdir -p "${output}.tmp"
  (
    cd ../nanotron
    python tools/preprocess_data.py \
      --tokenizer-name-or-path "${MODEL_ID}" \
      "${EOS_ARGS[@]}" \
      --output-folder "${output}.tmp" \
      --logging-dir "${output}.tmp/logs" \
      --n-tasks "${PREPROCESS_TASKS}" \
      jsonl --dataset "${jsonl}" --column text
  )
  rm -rf "${output}"
  mv "${output}.tmp" "${output}"
}

echo "### Tokenizing JSONLs into Nanotron Nanosets"
preprocess_jsonl "${NORMAL_JSONL}" "${NANOSET_ROOT}/normal_continuation"
preprocess_jsonl "${LOGIC_JSONL}" "${NANOSET_ROOT}/logical_deduction_logic"
preprocess_jsonl "${NL_JSONL}" "${NANOSET_ROOT}/logical_deduction_nl_exact"

echo "### Converting ${MODEL_ID} to Nanotron checkpoint if needed"
if [[ ! -f "${CONVERTED_CKPT}/model_config.json" ]]; then
  rm -rf "${CONVERTED_CKPT}.tmp"
  mkdir -p "${CONVERTED_CKPT}.tmp"
  (
    cd ../nanotron
    torchrun --standalone --nproc_per_node=1 -m examples.llama.convert_hf_to_nanotron \
      --checkpoint_path "${MODEL_ID}" \
      --save_path "${CONVERTED_CKPT}.tmp"
  )
  rm -rf "${CONVERTED_CKPT}"
  mv "${CONVERTED_CKPT}.tmp" "${CONVERTED_CKPT}"
else
  echo "Converted checkpoint already complete: ${CONVERTED_CKPT}"
fi

echo "### Verifying prerequisites"
test -f "${CONVERTED_CKPT}/model_config.json"
nanoset_done "${NANOSET_ROOT}/normal_continuation"
nanoset_done "${NANOSET_ROOT}/logical_deduction_logic"
nanoset_done "${NANOSET_ROOT}/logical_deduction_nl_exact"
python - <<PY
import json
from pathlib import Path
for path in [Path("${NORMAL_JSONL}.manifest.json"), Path("${LOGIC_JSONL}.manifest.json"), Path("${NL_JSONL}.manifest.json")]:
    payload = json.loads(path.read_text())
    print(json.dumps({"manifest": str(path), "records": payload["records"], "tokens": payload["tokens"], "done": payload["done"]}), flush=True)
PY
echo "Nanotron Qwen2.5 prerequisites are complete."
