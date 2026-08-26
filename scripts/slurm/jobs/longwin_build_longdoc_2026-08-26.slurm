#!/usr/bin/env bash
#SBATCH --job-name=longwin_build_longdoc
#SBATCH --partition=a40,a100,a100mig
#SBATCH --gres=gpu:1
#SBATCH --exclude=a0531,a0532,a0533,a0934
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Long-window midtrain prep, Arm 2 corpus: long Dolmino documents WITHOUT deep
# deductive content, length-matched to the band-25 logic trace token-length
# histogram, packed document-preservingly at seq 8192.
#
# Depends on: longwin_rebuild_prereqs (Dolmino nanoset) and
# longwin_build_band25 (band-25 logic length histogram).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"

source ./scripts/env.sh

VENV_PY="${HPCVAULT}/.venv_rlvl_posttrain/bin/python"
SEQ_LEN="${LONGWIN_SEQ_LEN:-8192}"
DOLMINO="${DOLMINO:-${WORK}/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino}"
JSONL_ROOT="${JSONL_ROOT:-${WORK}/synthetic-RLVL/nanotron_data/longwin_band25_20260826}"
PACK_ROOT="${PACK_ROOT:-${WORK}/synthetic-RLVL/nanosets_longwin_20260826}"
PREP_DIR="analysis/longwin_midtrain_prep_20260826"

compgen -G "${DOLMINO}/*.metadata" >/dev/null
test -s "${JSONL_ROOT}/logic_band25.stats.json"

"${VENV_PY}" scripts/data/build_longdoc_control_docpack.py \
  --dolmino-folder "${DOLMINO}" \
  --band25-stats-json "${JSONL_ROOT}/logic_band25.stats.json" \
  --output-folder "${PACK_ROOT}/longdoc_control" \
  --seq-len "${SEQ_LEN}" \
  --target-real-tokens "${LONGDOC_TARGET_TOKENS:-262000000}" \
  --seed 20260830

cp "${PACK_ROOT}/longdoc_control/docpack_stats.json" "${PREP_DIR}/longdoc_control_docpack_stats.json"
echo "longdoc control docpack complete: ${PACK_ROOT}/longdoc_control"
