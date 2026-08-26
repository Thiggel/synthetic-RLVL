#!/usr/bin/env bash
#SBATCH --job-name=longwin_prereq_rebuild
#SBATCH --partition=a40,a100
#SBATCH --gres=gpu:1
#SBATCH --exclude=a0531,a0532,a0533,a0934
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Rebuild the two deleted midtrain prerequisites for the long-window (8192)
# midtrain prep, onto $WORK (vault is over its soft quota):
#
#   1. The 5.1B-token Dolmino Qwen2.5 nanoset (identical recipe to accepted
#      build 3875824: allenai/dolma3_dolmino_mix-10B-1025, shuffle seed 42,
#      5.1B token budget, tools/preprocess_data with Qwen2.5-7B tokenizer) ->
#      $WORK/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino
#   2. The Qwen2.5-7B HF->Nanotron tp1 base checkpoint ->
#      $WORK/synthetic-RLVL/nanotron_checkpoints/qwen25_7b_tp1
#
# Background: $HPCVAULT/synthetic-RLVL/nanosets/ and the contents of
# $HPCVAULT/synthetic-RLVL/nanotron_checkpoints/ disappeared between
# 2026-08-24 and 2026-08-26 with no project-log entry (nanotron_checkpoints/
# is an empty dir with mtime Aug 24). Both artifacts are reproducible from
# committed recipes; this job reproduces them.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"

source ./scripts/env.sh

# --- 1. Dolmino 5.1B nanoset ---
# The 5p1b wrapper hardcodes (exports) HPCVAULT roots, so replicate its body:
# call the prereqs builder (which honors DATA_ROOT/NANOSET_ROOT env) for the
# dolmino row, then apply the same 5B packed-token gate + raw-JSONL cleanup.
NANOSET_ROOT_TARGET="${WORK}/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b"
DATA_ROOT_TARGET="${WORK}/synthetic-RLVL/nanotron_data/qwen25_dolmino_neutral_v1_5p1b"
if compgen -G "${NANOSET_ROOT_TARGET}/dolmino/*.metadata" >/dev/null; then
  echo "Dolmino nanoset already present: ${NANOSET_ROOT_TARGET}/dolmino"
else
  SLURM_ARRAY_TASK_ID=0 \
  NORMAL_TOKEN_BUDGET=5100000000 \
  DATA_ROOT="${DATA_ROOT_TARGET}" \
  NANOSET_ROOT="${NANOSET_ROOT_TARGET}" \
    bash scripts/slurm/jobs/nanotron_qwen25_build_dolmino_prereqs_2026-07-15.slurm

  STATS="${NANOSET_ROOT_TARGET}/dolmino/logs/stats.json"
  test -s "${STATS}"
  PACKED_TOKENS="$("${WORK}/nanotron/bin/python" - "${STATS}" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    payload = json.load(handle)
tokenizer_stats = next(row for row in payload if "tokens" in row.get("stats", {}))
print(tokenizer_stats["stats"]["tokens"]["total"])
PY
)"
  if (( PACKED_TOKENS < 5000134656 )); then
    echo "Packed Dolmino stream is too short for the 5B gate: ${PACKED_TOKENS}" >&2
    exit 3
  fi
  # Raw JSONL is reproducible and is no longer needed after the packed audit.
  find "${DATA_ROOT_TARGET}" -depth -delete
  echo "Accepted 5.1B Dolmino Nanoset with ${PACKED_TOKENS} packed tokens"
fi

# --- 2. Qwen2.5-7B tp1 Nanotron base checkpoint (conversion step lifted from
#        nanotron_qwen25_build_prereqs_2026-06-29.slurm) ---
source "${WORK}/nanotron/bin/activate"
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}"
CONVERTED_CKPT="${CONVERTED_CKPT:-${WORK}/synthetic-RLVL/nanotron_checkpoints/qwen25_7b_tp1}"
mkdir -p "$(dirname "${CONVERTED_CKPT}")"
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

test -f "${CONVERTED_CKPT}/model_config.json"
compgen -G "${NANOSET_ROOT_TARGET}/dolmino/*.metadata" >/dev/null
echo "Long-window midtrain prerequisites rebuilt."
