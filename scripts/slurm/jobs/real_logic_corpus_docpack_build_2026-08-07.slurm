#!/usr/bin/env bash
#SBATCH --job-name=build_real_logic_docpack
#SBATCH --partition=a100mig
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Pack the deduplicated real-data logic corpus (pararule + prontoqa +
# proofwriter, 451k formal/NL pairs) with the document-preserving packer.
#
# Produces two docpacks (formal condition from `formal_doc`, NL condition
# from `nl_doc`), reports token counts / overlength rates / packing
# efficiency / depth histogram (incl. ProofWriter depth-0 fraction, which is
# a known open down-weighting question -- reported, NOT filtered), then runs
# the four-gate auditor on each pack. Real docs use the Solution/Derivation/
# Final-answer surface without the synthetic "Definitions:/Formal premises:"
# markers, so both packs are audited with --template nl_exact (structure
# markers only); audit verdicts are informational for real data and do not
# gate any submitted GPU job.
#
# CPU-only (no GRES) on a100mig.

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
CORPUS_DIR="${CORPUS_DIR:-${WORK}/synthetic-RLVL/data/real_logic_corpus}"
DATA_ROOT="${DATA_ROOT:-${HPCVAULT}/synthetic-RLVL/nanotron_data/qwen25_real_logic_docpack_v1}"
NANOSET_ROOT="${NANOSET_ROOT:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25_real_logic_docpack_v1}"
DOLMINO_NANOSET="${DOLMINO_NANOSET:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino}"
AUDIT_DIR="${AUDIT_DIR:-analysis/real_logic_docpack_audit_20260807}"
RERUN_TRAIN_STEPS="${RERUN_TRAIN_STEPS:-4770}"
RERUN_GLOBAL_BATCH_SIZE="${RERUN_GLOBAL_BATCH_SIZE:-128}"

mkdir -p "${DATA_ROOT}" "${NANOSET_ROOT}" "${AUDIT_DIR}"

# --- 1. Flatten the paired corpus into per-condition {text, depth} JSONL and report composition ---
CORPUS_DIR="${CORPUS_DIR}" DATA_ROOT="${DATA_ROOT}" AUDIT_DIR="${AUDIT_DIR}" python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

corpus_dir = Path(os.environ["CORPUS_DIR"])
data_root = Path(os.environ["DATA_ROOT"])
audit_dir = Path(os.environ["AUDIT_DIR"])

sources = ["pararule", "prontoqa", "proofwriter"]
outs = {
    "formal": (data_root / "real_logic_formal.jsonl").open("w", encoding="utf-8"),
    "nl": (data_root / "real_logic_nl.jsonl").open("w", encoding="utf-8"),
}
counts = Counter()
depth_by_source = {s: Counter() for s in sources}
n = 0
for src in sources:
    with (corpus_dir / f"{src}.dedup.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            assert row["source"] == src
            depth = int(row["depth"])
            depth_by_source[src][depth] += 1
            counts[src] += 1
            n += 1
            for cond, field in (("formal", "formal_doc"), ("nl", "nl_doc")):
                text = row[field]
                assert text and "\n\nSolution:\n" in text and "\n\nFinal answer: " in text
                assert "<|fim_pad|>" not in text
                outs[cond].write(json.dumps({
                    "text": text,
                    "depth": depth,
                    "source": src,
                    "id": row["id"],
                }, ensure_ascii=False) + "\n")
for f in outs.values():
    f.close()

pw = depth_by_source["proofwriter"]
summary = {
    "records_total": n,
    "records_by_source": dict(counts),
    "depth_hist_by_source": {s: dict(sorted(c.items())) for s, c in depth_by_source.items()},
    "proofwriter_depth0_count": pw.get(0, 0),
    "proofwriter_depth0_fraction_of_proofwriter": pw.get(0, 0) / max(sum(pw.values()), 1),
    "proofwriter_depth0_fraction_of_corpus": pw.get(0, 0) / n,
    "note": "depth-0 restatement proofs reported only; down-weighting is an open question, NOT filtered here",
}
(audit_dir / "real_corpus_composition.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

# --- 2. Pack both conditions ---
for COND in formal nl; do
  NANOSET="${NANOSET_ROOT}/${COND}"
  rm -rf "${NANOSET}.tmp"
  python scripts/data/pack_document_preserving_nanoset.py \
    --input "${DATA_ROOT}/real_logic_${COND}.jsonl" \
    --output-folder "${NANOSET}.tmp" \
    --tokenizer "${MODEL_ID}" \
    --seq-len "${SEQ_LEN}" \
    --pad-token "<|fim_pad|>" \
    --pool-size 64 \
    --shuffle-seed 42
  rm -rf "${NANOSET}"
  mv "${NANOSET}.tmp" "${NANOSET}"
  compgen -G "${NANOSET}/*.metadata" >/dev/null
done

# --- 3. Audit both packs (informational; do not abort the job on a failed verdict) ---
AUDIT_RC=0
for COND in formal nl; do
  python scripts/nanotron/audit_docpack_training_path.py \
    --packed-folder "${NANOSET_ROOT}/${COND}" \
    --dolmino-folder "${DOLMINO_NANOSET}" \
    --tokenizer "${MODEL_ID}" \
    --template nl_exact \
    --seq-len "${SEQ_LEN}" \
    --train-steps "${RERUN_TRAIN_STEPS}" \
    --global-batch-size "${RERUN_GLOBAL_BATCH_SIZE}" \
    --target-ratio 0.05 \
    --out-json "${AUDIT_DIR}/audit_real_docpack_${COND}.json" \
    --out-md "${AUDIT_DIR}/audit_real_docpack_${COND}.md" \
    --examples-out "${AUDIT_DIR}/decoded_windows_real_${COND}.md" || AUDIT_RC=$?
done
if [[ "${AUDIT_RC}" -ne 0 ]]; then
  echo "NOTE: at least one real-corpus audit reported a failed gate (informational; see ${AUDIT_DIR})"
fi

echo "Real-corpus docpack build complete: ${NANOSET_ROOT}"
