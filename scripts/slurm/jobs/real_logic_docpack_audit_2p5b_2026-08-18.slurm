#!/usr/bin/env bash
#SBATCH --job-name=build_q25_real_audit
#SBATCH --partition=a100mig
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Re-audit the already-built real-logic docpacks (ProofWriter/PARARULE/PrOntoQA,
# built by job 3964789) at the 2.5B blend horizon so each fail-closed real-corpus
# train job reads an audit solved for its own blend size.
#
# Design is matched to the accepted synthetic rerun: 5% replacement, 4,770 steps
# x GBS 128 x 4,096 seq = 2,500,853,760 tokens, same Dolmino nanoset. The
# Dolmino-only control at step 4770 is shared with the synthetic rerun and is
# NOT retrained.
#
# Does NOT rebuild the packs -- they are accepted (formal 299.34M real tokens /
# 78,497 windows / 93.08% efficiency; nl 213.83M / 55,113 / 94.70%; zero
# overlength in both).
#
# CPU-only (no GRES) on a100mig, mirroring build jobs 3964788/3964789.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"

RERUN_TRAIN_STEPS="${RERUN_TRAIN_STEPS:-4770}"
RERUN_GLOBAL_BATCH_SIZE="${RERUN_GLOBAL_BATCH_SIZE:-128}"
AUDIT_DIR="${AUDIT_DIR:-analysis/real_logic_docpack_audit_2p5b_20260818}"

source ./scripts/env.sh
source "${WORK}/nanotron/bin/activate"
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

REAL_ROOT="${REAL_ROOT:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25_real_logic_docpack_v1}"
DOLMINO_NANOSET="${DOLMINO_NANOSET:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino}"

mkdir -p "${AUDIT_DIR}"

# formal pack -> logic condition
python scripts/nanotron/audit_docpack_training_path.py \
  --packed-folder "${REAL_ROOT}/formal" \
  --dolmino-folder "${DOLMINO_NANOSET}" \
  --tokenizer "Qwen/Qwen2.5-7B" \
  --template real_logic \
  --seq-len 4096 \
  --train-steps "${RERUN_TRAIN_STEPS}" \
  --global-batch-size "${RERUN_GLOBAL_BATCH_SIZE}" \
  --target-ratio 0.05 \
  --out-json "${AUDIT_DIR}/audit_docpack_logic.json" \
  --out-md "${AUDIT_DIR}/audit_docpack_logic.md" \
  --examples-out "${AUDIT_DIR}/decoded_windows_real_logic.md"

# nl pack -> nl_exact condition
python scripts/nanotron/audit_docpack_training_path.py \
  --packed-folder "${REAL_ROOT}/nl" \
  --dolmino-folder "${DOLMINO_NANOSET}" \
  --tokenizer "Qwen/Qwen2.5-7B" \
  --template nl_exact \
  --seq-len 4096 \
  --train-steps "${RERUN_TRAIN_STEPS}" \
  --global-batch-size "${RERUN_GLOBAL_BATCH_SIZE}" \
  --target-ratio 0.05 \
  --out-json "${AUDIT_DIR}/audit_docpack_nl_exact.json" \
  --out-md "${AUDIT_DIR}/audit_docpack_nl_exact.md" \
  --examples-out "${AUDIT_DIR}/decoded_windows_real_nl_exact.md"

echo "Real-corpus 2.5B-horizon audits complete: ${AUDIT_DIR}"
python - "${AUDIT_DIR}" <<"PY"
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
for name in ("audit_docpack_logic.json", "audit_docpack_nl_exact.json"):
    j = json.loads((d / name).read_text())
    print(name, "all_pass=", j.get("all_pass"), "proof_weight=", j.get("proof_weight"),
          "realized=", j.get("realized_ratio"), "epochs=", j.get("synthetic_epochs"))
PY
