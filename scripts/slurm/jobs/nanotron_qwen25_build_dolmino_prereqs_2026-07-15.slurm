#!/usr/bin/env bash
#SBATCH --job-name=build_q25_dolmino
#SBATCH --partition=rtxpro6k,a40,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh
source "${WORK}/nanotron/bin/activate"

export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}"
DATASET_ID="${DATASET_ID:-allenai/dolma3_dolmino_mix-10B-1025}"
NORMAL_TOKEN_BUDGET="${NORMAL_TOKEN_BUDGET:-4800000000}"
PROOF_TOKEN_BUDGET="${PROOF_TOKEN_BUDGET:-550000000}"
PREPROCESS_TASKS="${PREPROCESS_TASKS:-16}"
DATA_ROOT="${DATA_ROOT:-${HPCVAULT}/synthetic-RLVL/nanotron_data/qwen25_dolmino_neutral_v1}"
NANOSET_ROOT="${NANOSET_ROOT:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1}"

KINDS=(dolmino logic nl_exact)
KIND="${KINDS[$SLURM_ARRAY_TASK_ID]}"
JSONL="${DATA_ROOT}/${KIND}.jsonl"
NANOSET="${NANOSET_ROOT}/${KIND}"
mkdir -p "${DATA_ROOT}" "${NANOSET_ROOT}"

manifest_done() {
  python - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
payload = json.loads(path.read_text())
raise SystemExit(0 if payload.get("done") else 1)
PY
}

if ! manifest_done "${JSONL}.manifest.json"; then
  if [[ "${KIND}" == "dolmino" ]]; then
    python scripts/data/export_hf_zst_repo_text_token_budget.py \
      --dataset "${DATASET_ID}" \
      --output "${JSONL}" \
      --tokenizer "${MODEL_ID}" \
      --target-tokens "${NORMAL_TOKEN_BUDGET}" \
      --shuffle-seed 42 \
      --resume \
      --progress-every 5000
  else
    python scripts/data/export_generated_proof_jsonl.py \
      --template "${KIND}" \
      --document-format neutral_solution \
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
  fi
fi

KIND="${KIND}" JSONL="${JSONL}" python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["JSONL"])
manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
assert manifest["done"]
rows = []
with path.open(encoding="utf-8") as handle:
    for _ in range(8):
        rows.append(json.loads(next(handle)))
assert all(row["text"].strip() for row in rows)
if os.environ["KIND"] != "dolmino":
    for row in rows:
        text = row["text"]
        assert "\n\nSolution:\nContext:\n" in text
        assert "\n\nDerivation:\n" in text
        assert "\n\nConclusion:\n" in text
        assert "\n\nFinal answer: " in text
        assert "<formal>" not in text and "<think>" not in text
    assert manifest["document_format"] == "neutral_solution"
else:
    assert manifest["dataset"] == "allenai/dolma3_dolmino_mix-10B-1025"
    assert manifest["shuffle_seed"] == 42
    assert manifest["kind"] == "hf_zst_repo_text"
print(json.dumps({"kind": os.environ["KIND"], "manifest": manifest, "sample": rows[0]["text"][:2000]}, indent=2))
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
find "${NANOSET}" -maxdepth 1 -type f -name '*.ds' -size +0 -print -quit | grep -q .
echo "Dolmino prerequisite complete: ${KIND} -> ${NANOSET}"
