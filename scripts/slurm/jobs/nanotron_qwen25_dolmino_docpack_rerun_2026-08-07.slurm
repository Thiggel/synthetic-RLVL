#!/usr/bin/env bash
#SBATCH --job-name=q25_docpack_rerun
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:8
#SBATCH --exclude=a0532,a0934
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Clean three-way 2.5B rerun (document-preserving objective, depths 1-14).
#
# One condition per submission, selected by CONDITION=control|logic|nl_exact.
# Submit all three with the same job name and --dependency=singleton so they
# serialize (never more than 8 GPUs from this line at once).
#
# Identical to the accepted three-way recipe (base checkpoint, LR schedule,
# parallelism, seq len, unchanged 5.1B Dolmino nanoset); only the synthetic
# data path changed: compact_solution documents at depths 1-14 packed whole
# into 4,097-token windows with loss-masked <|fim_pad|> tail padding and a
# padding-compensated proof weight so the realized synthetic share of loss
# tokens is exactly 5%.
#
# Fail-closed (logic/nl_exact): refuses to train unless the committed rerun
# audit bundle reports all four gates passing at the 2.5B blend size.
# Control uses Dolmino only (grid row 0) and needs no synthetic docpack.
#
# 2.5B tokens: 4,770 steps x 128 global batch x 4,096 seq = 2,500,853,760.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# Robust to non-login submission shells (ssh sbatch): NHR@FAU cluster vars may be absent.
export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"

source ./scripts/env.sh

CONDITION="${CONDITION:?set CONDITION=control|logic|nl_exact}"
AUDIT_DIR="${AUDIT_DIR:-analysis/docpack_rerun_audit_20260807}"

case "${CONDITION}" in
  control)
    GRID_IDX=0
    MIX_TAG=p0
    ;;
  logic)
    GRID_IDX=1
    MIX_TAG=p5
    AUDIT_JSON="${AUDIT_DIR}/audit_docpack_logic.json"
    ;;
  nl_exact)
    GRID_IDX=6
    MIX_TAG=p5
    AUDIT_JSON="${AUDIT_DIR}/audit_docpack_nl_exact.json"
    ;;
  *)
    echo "Unknown CONDITION=${CONDITION}" >&2
    exit 2
    ;;
esac

export TRAIN_STEPS="${TRAIN_STEPS:-4770}"

if [[ "${CONDITION}" != "control" ]]; then
  read -r AUDIT_OK PROOF_WEIGHT AUDIT_STEPS <<<"$("${WORK}/nanotron/bin/python" - "${AUDIT_JSON}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
gate = payload["gates"]["exact_mixture"]
steps = gate["blend_size_samples"] // 128
print(str(bool(payload["all_pass"])).lower(), gate["proof_weight"], steps)
PY
)"
  if [[ "${AUDIT_OK}" != "true" ]]; then
    echo "Refusing to train: audit bundle ${AUDIT_JSON} does not report all_pass=true" >&2
    exit 2
  fi
  if [[ "${AUDIT_STEPS}" != "${TRAIN_STEPS}" ]]; then
    echo "Refusing to train: audit blend size (${AUDIT_STEPS} steps) != TRAIN_STEPS (${TRAIN_STEPS})" >&2
    exit 2
  fi
  export PROOF_WEIGHT_OVERRIDE="${PROOF_WEIGHT_OVERRIDE:-${PROOF_WEIGHT}}"
  export NANOSET_PADDING_LABEL_ID="${NANOSET_PADDING_LABEL_ID:-151662}"
  export LOGIC_DATASET_FOLDER="${LOGIC_DATASET_FOLDER:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25_dolmino_compact_docpack_v1/logic}"
  export NL_DATASET_FOLDER="${NL_DATASET_FOLDER:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25_dolmino_compact_docpack_v1/nl_exact}"
  echo "docpack rerun ${CONDITION}: proof_weight=${PROOF_WEIGHT_OVERRIDE} padding_label_id=${NANOSET_PADDING_LABEL_ID} steps=${TRAIN_STEPS}"
fi

export SLURM_ARRAY_TASK_ID="${GRID_IDX}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
export LR=1.0e-5
export MIN_LR=1.0e-6
export WARMUP_STEPS=256
export LR_DECAY_STARTING_STEP=256
export LR_DECAY_STEPS=37891
export NORMAL_DATASET_FOLDER="${NORMAL_DATASET_FOLDER:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino}"
export NANOTRON_PG_TIMEOUT_MINUTES="${NANOTRON_PG_TIMEOUT_MINUTES:-120}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export RUN_TAG_OVERRIDE="${RUN_TAG_OVERRIDE:-qwen25_7b_dolmino_${CONDITION}_docpack_${MIX_TAG}_2p5b}"
export RUN_ROOT_OVERRIDE="${RUN_ROOT_OVERRIDE:-${HPCVAULT}/synthetic-RLVL/nanotron_docpack_rerun/${RUN_TAG_OVERRIDE}}"
export FINAL_CHECKPOINT_ALIAS=2p5b
export PRUNE_SUPERSEDED_RUN_CHECKPOINTS=1

# Mid-run checkpoint pruner: ~199G per checkpoint at interval 500 accumulates
# past the HPCVAULT hard quota within a single 24h pass (this truncated the
# logic 4770 save on 2026-08-12). The grid script only prunes at start/end, so
# reclaim superseded states while training runs: keep the two newest numeric
# checkpoint dirs (the newest may still be mid-write; the second-newest is the
# resume state), delete anything older.
#
# The subshell must NOT inherit errexit/pipefail: before the first checkpoint
# exists the grep finds no numeric directories and exits 1, which silently
# killed the pruner on 2026-08-14 and let nl_exact fill the quota again.
(
  set +e +o pipefail
  while sleep 600; do
    CKPT_DIR="${RUN_ROOT_OVERRIDE}/checkpoints"
    [[ -d "${CKPT_DIR}" ]] || continue
    KEEP_COUNT=$(find "${CKPT_DIR}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null | grep -cE '^[0-9]+$')
    echo "mid-run prune heartbeat: ${KEEP_COUNT} checkpoint(s) present"
    find "${CKPT_DIR}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null \
      | grep -E '^[0-9]+$' | sort -n | head -n -2 \
      | while IFS= read -r OLD_STEP; do
          echo "mid-run prune: removing superseded checkpoint ${CKPT_DIR}/${OLD_STEP}"
          find "${CKPT_DIR}/${OLD_STEP}" -depth -delete
        done
  done
) &
PRUNER_PID=$!
trap 'kill "${PRUNER_PID}" 2>/dev/null || true' EXIT

bash scripts/slurm/jobs/nanotron_qwen25_midtrain_grid_2026-06-24.slurm
