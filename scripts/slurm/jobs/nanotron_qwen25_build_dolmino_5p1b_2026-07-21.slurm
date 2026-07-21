#!/usr/bin/env bash
#SBATCH --job-name=build_q25_dolmino_5p1b
#SBATCH --partition=rtxpro6k,a40,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ./scripts/env.sh

export SLURM_ARRAY_TASK_ID=0
export NORMAL_TOKEN_BUDGET=5100000000
export DATA_ROOT="${HPCVAULT}/synthetic-RLVL/nanotron_data/qwen25_dolmino_neutral_v1_5p1b"
export NANOSET_ROOT="${HPCVAULT}/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b"

bash scripts/slurm/jobs/nanotron_qwen25_build_dolmino_prereqs_2026-07-15.slurm

STATS="${NANOSET_ROOT}/dolmino/logs/stats.json"
test -s "${STATS}"
PACKED_TOKENS="$(${WORK}/nanotron/bin/python - "${STATS}" <<'PY'
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
find "${DATA_ROOT}" -depth -delete
echo "Accepted 5.1B Dolmino Nanoset with ${PACKED_TOKENS} packed tokens"
