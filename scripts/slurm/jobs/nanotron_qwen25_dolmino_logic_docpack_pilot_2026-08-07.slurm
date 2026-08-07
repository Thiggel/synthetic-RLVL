#!/usr/bin/env bash
#SBATCH --job-name=q25_docpack_pilot
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:8
#SBATCH --exclude=a0532,a0934
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --array=0-0%1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# P0 document-preserving (docpack) compact-objective pilot, logic condition.
#
# Identical to the accepted 5B three-way recipe (same base checkpoint, LR
# schedule, parallelism, sequence length, and unchanged 5.1B Dolmino nanoset);
# the ONLY change is the synthetic data path: compact_solution documents at
# depths 1-14 packed whole into 4,097-token loader-aligned windows with
# loss-masked <|fim_pad|> tail padding (NANOSET_PADDING_LABEL_ID), and a
# proof-dataset weight solved so the realized synthetic share of loss tokens
# is exactly 5% despite padding overhead.
#
# Fail-closed: refuses to start unless the committed audit bundle reports
# all four gates (zero-overlength, decoded-batch, padding-loss-mask,
# exact-mixture) passing.
#
# ~0.5B tokens: 954 steps x 128 global batch x 4,096 seq = 500,170,752.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# Robust to non-login submission shells (ssh sbatch): NHR@FAU cluster vars may be absent.
export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"

source ./scripts/env.sh

AUDIT_JSON="${AUDIT_JSON:-analysis/docpack_pilot_audit_20260807/audit_docpack_logic.json}"
read -r AUDIT_OK PROOF_WEIGHT <<<"$("${WORK}/nanotron/bin/python" - "${AUDIT_JSON}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
print(str(bool(payload["all_pass"])).lower(), payload["gates"]["exact_mixture"]["proof_weight"])
PY
)"
if [[ "${AUDIT_OK}" != "true" ]]; then
  echo "Refusing to train: audit bundle ${AUDIT_JSON} does not report all_pass=true" >&2
  exit 2
fi

# Grid row 1 = logic p5 (only the synthetic loader/objective differs from the
# accepted 5B run; PROOF_WEIGHT_OVERRIDE replaces the nominal 0.05 sample
# weight with the padding-compensated value from the exact-mixture audit).
export SLURM_ARRAY_TASK_ID=1
export TRAIN_STEPS="${TRAIN_STEPS:-954}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
export LR=1.0e-5
export MIN_LR=1.0e-6
export WARMUP_STEPS=256
export LR_DECAY_STARTING_STEP=256
export LR_DECAY_STEPS=37891
export NORMAL_DATASET_FOLDER="${NORMAL_DATASET_FOLDER:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino}"
export LOGIC_DATASET_FOLDER="${LOGIC_DATASET_FOLDER:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25_dolmino_compact_docpack_v1/logic}"
export PROOF_WEIGHT_OVERRIDE="${PROOF_WEIGHT_OVERRIDE:-${PROOF_WEIGHT}}"
export NANOSET_PADDING_LABEL_ID="${NANOSET_PADDING_LABEL_ID:-151662}"
export NANOTRON_PG_TIMEOUT_MINUTES="${NANOTRON_PG_TIMEOUT_MINUTES:-120}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export RUN_TAG_OVERRIDE="${RUN_TAG_OVERRIDE:-qwen25_7b_dolmino_logic_docpack_p5_0p5b}"
export RUN_ROOT_OVERRIDE="${RUN_ROOT_OVERRIDE:-${HPCVAULT}/synthetic-RLVL/nanotron_docpack_pilot/${RUN_TAG_OVERRIDE}}"
export FINAL_CHECKPOINT_ALIAS=0p5b
export PRUNE_SUPERSEDED_RUN_CHECKPOINTS=1

echo "docpack pilot: proof_weight=${PROOF_WEIGHT_OVERRIDE} padding_label_id=${NANOSET_PADDING_LABEL_ID} steps=${TRAIN_STEPS}"

bash scripts/slurm/jobs/nanotron_qwen25_midtrain_grid_2026-06-24.slurm
