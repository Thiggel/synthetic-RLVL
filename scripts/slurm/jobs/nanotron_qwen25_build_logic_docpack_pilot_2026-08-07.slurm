#!/usr/bin/env bash
#SBATCH --job-name=build_q25_docpack
#SBATCH --partition=a100mig
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000M
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Build + audit the document-preserving compact logic corpus for the P0 pilot.
#
# 1. Export compact_solution logic documents at depths 1-14 (the deepest band
#    whose compact documents fit one 4,096-token window; measured depth-15 max
#    is 4,377 tokens and cannot be document-preserved at this window).
# 2. Pack whole documents into 4,097-token loader-aligned windows
#    (greedy first-fit, <|fim_pad|> tail padding, loss-masked at train time).
# 3. Run the four audit gates (zero-overlength, decoded-batch,
#    padding-loss-mask, exact-mixture) against the packed folder and the
#    unchanged 5.1B Dolmino nanoset, writing the audit bundle into the repo.
#
# CPU-only (no GRES), mirroring the oversight watcher's a100mig usage.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# Robust to non-login submission shells (ssh sbatch): NHR@FAU cluster vars may be absent.
export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"

source ./scripts/env.sh
source "${WORK}/nanotron/bin/activate"

export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}"
SEQ_LEN="${SEQ_LEN:-4096}"
PROOF_TOKEN_BUDGET="${PROOF_TOKEN_BUDGET:-40000000}"
TRAIN_MAX_STEP="${TRAIN_MAX_STEP:-14}"
DATA_ROOT="${DATA_ROOT:-${HPCVAULT}/synthetic-RLVL/nanotron_data/qwen25_dolmino_compact_docpack_v1}"
NANOSET_ROOT="${NANOSET_ROOT:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25_dolmino_compact_docpack_v1}"
DOLMINO_NANOSET="${DOLMINO_NANOSET:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino}"
TEMPLATE="${TEMPLATE:-logic}"
JSONL="${DATA_ROOT}/${TEMPLATE}_compact_d${TRAIN_MAX_STEP}.jsonl"
NANOSET="${NANOSET_ROOT}/${TEMPLATE}"
AUDIT_DIR="${AUDIT_DIR:-analysis/docpack_pilot_audit_20260807}"
PILOT_TRAIN_STEPS="${PILOT_TRAIN_STEPS:-954}"
PILOT_GLOBAL_BATCH_SIZE="${PILOT_GLOBAL_BATCH_SIZE:-128}"

mkdir -p "${DATA_ROOT}" "${NANOSET_ROOT}" "${AUDIT_DIR}"

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
  python scripts/data/export_generated_proof_jsonl.py \
    --template "${TEMPLATE}" \
    --document-format compact_solution \
    --output "${JSONL}" \
    --tokenizer "${MODEL_ID}" \
    --target-tokens "${PROOF_TOKEN_BUDGET}" \
    --seed 3407 \
    --train-min-step 1 \
    --train-max-step "${TRAIN_MAX_STEP}" \
    --difficulty hard_fsa_schema \
    --branching-factor 4 \
    --resume \
    --progress-every 5000
fi

JSONL="${JSONL}" python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["JSONL"])
manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
assert manifest["done"]
assert manifest["document_format"] == "compact_solution"
rows = []
with path.open(encoding="utf-8") as handle:
    for _ in range(8):
        rows.append(json.loads(next(handle)))
for row in rows:
    text = row["text"]
    assert "\n\nSolution:\n" in text
    assert "\n\nFinal answer: " in text
    assert "Derivation:\n" in text
    assert "Context:\n" not in text and "Conclusion:\n" not in text
    assert "<formal>" not in text and "<think>" not in text
print(json.dumps({"manifest": manifest, "sample": rows[0]["text"][:1200]}, indent=2))
PY

rm -rf "${NANOSET}.tmp"
python scripts/data/pack_document_preserving_nanoset.py \
  --input "${JSONL}" \
  --output-folder "${NANOSET}.tmp" \
  --tokenizer "${MODEL_ID}" \
  --seq-len "${SEQ_LEN}" \
  --pad-token "<|fim_pad|>" \
  --pool-size 64 \
  --shuffle-seed 42
rm -rf "${NANOSET}"
mv "${NANOSET}.tmp" "${NANOSET}"
compgen -G "${NANOSET}/*.metadata" >/dev/null

python scripts/nanotron/audit_docpack_training_path.py \
  --packed-folder "${NANOSET}" \
  --dolmino-folder "${DOLMINO_NANOSET}" \
  --tokenizer "${MODEL_ID}" \
  --template "${TEMPLATE}" \
  --seq-len "${SEQ_LEN}" \
  --train-steps "${PILOT_TRAIN_STEPS}" \
  --global-batch-size "${PILOT_GLOBAL_BATCH_SIZE}" \
  --target-ratio 0.05 \
  --out-json "${AUDIT_DIR}/audit_docpack_${TEMPLATE}.json" \
  --out-md "${AUDIT_DIR}/audit_docpack_${TEMPLATE}.md" \
  --examples-out "${AUDIT_DIR}/decoded_windows_${TEMPLATE}.md"

echo "Docpack build + audit complete: ${NANOSET}"
