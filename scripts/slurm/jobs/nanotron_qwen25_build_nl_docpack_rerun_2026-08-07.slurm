#!/usr/bin/env bash
#SBATCH --job-name=build_q25_docpack_rerun
#SBATCH --partition=a100mig
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Clean three-way 2.5B rerun data prep:
# 1. Build + audit the nl_exact compact docpack (depths 1-14, same generator
#    seed 3407 as the logic pilot pack, so documents are the NL twins of the
#    logic corpus) by reusing the parameterized pilot build script.
# 2. Re-audit the already-built logic docpack at the 2.5B horizon
#    (4,770 steps x GBS 128) so each fail-closed train job reads an audit
#    solved for its own blend size.
# Audit bundle: analysis/docpack_rerun_audit_20260807/
#
# CPU-only (no GRES) on a100mig, mirroring build job 3964215.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# Robust to non-login submission shells (ssh sbatch): NHR@FAU cluster vars may be absent.
export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"

export RERUN_TRAIN_STEPS="${RERUN_TRAIN_STEPS:-4770}"
export RERUN_GLOBAL_BATCH_SIZE="${RERUN_GLOBAL_BATCH_SIZE:-128}"
export AUDIT_DIR="${AUDIT_DIR:-analysis/docpack_rerun_audit_20260807}"

# --- 1. nl_exact docpack build + audit (reuse the pilot build script) ---
TEMPLATE=nl_exact \
PILOT_TRAIN_STEPS="${RERUN_TRAIN_STEPS}" \
PILOT_GLOBAL_BATCH_SIZE="${RERUN_GLOBAL_BATCH_SIZE}" \
AUDIT_DIR="${AUDIT_DIR}" \
  bash scripts/slurm/jobs/nanotron_qwen25_build_logic_docpack_pilot_2026-08-07.slurm

# --- 2. logic re-audit at the 2.5B horizon (do NOT rebuild: pilot 3964239 trains from this pack) ---
source ./scripts/env.sh
source "${WORK}/nanotron/bin/activate"
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

LOGIC_NANOSET="${LOGIC_NANOSET:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25_dolmino_compact_docpack_v1/logic}"
DOLMINO_NANOSET="${DOLMINO_NANOSET:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino}"

python scripts/nanotron/audit_docpack_training_path.py \
  --packed-folder "${LOGIC_NANOSET}" \
  --dolmino-folder "${DOLMINO_NANOSET}" \
  --tokenizer "Qwen/Qwen2.5-7B" \
  --template logic \
  --seq-len 4096 \
  --train-steps "${RERUN_TRAIN_STEPS}" \
  --global-batch-size "${RERUN_GLOBAL_BATCH_SIZE}" \
  --target-ratio 0.05 \
  --out-json "${AUDIT_DIR}/audit_docpack_logic.json" \
  --out-md "${AUDIT_DIR}/audit_docpack_logic.md" \
  --examples-out "${AUDIT_DIR}/decoded_windows_logic.md"

echo "Rerun docpack build + audits complete: ${AUDIT_DIR}"
