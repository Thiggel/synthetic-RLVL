#!/usr/bin/env bash
#SBATCH --job-name=q25_longwin_midtrain
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:8
#SBATCH --exclude=a0531,a0532,a0533,a0934
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Long-window depth-25 midtrain: five-arm 2.5B-token study at seq 8192.
#
# One condition per submission, selected by
#   CONDITION=control|longdoc|logic_band25|nl_exact_band25|condensed_logic_band25
# Submit all with the same job name and --dependency=singleton so they
# serialize (never more than 8 GPUs from this line at once).
#
# Cloned from the accepted docpack rerun wrapper
# (nanotron_qwen25_dolmino_docpack_rerun_2026-08-07.slurm). Deliberate changes:
#   - SEQ_LEN 4096 -> 8192; identical token budget preserved by halving steps:
#     2385 steps x GBS 128 x 8192 = 2,500,853,760 = 4770 x 128 x 4096.
#   - GPU memory: micro-batch halved 4 -> 2 and grad accumulation doubled
#     16 -> 32, keeping the effective global batch at 128 sequences with
#     identical tokens per microbatch (2 x 8192 = 4 x 4096).
#   - LR schedule kept token-equivalent, not step-equivalent: WARMUP 256 -> 128
#     steps, LR_DECAY_STARTING_STEP 256 -> 128, LR_DECAY_STEPS 37891 -> 18946
#     (each step now covers 2x the tokens).
#   - Checkpoint interval 500 -> 250 (same token cadence, ~262M tokens).
#   - 10% replacement instead of 5%; docpacks are the SFT-style band-25
#     renderings (or long-doc / condensed variants) packed at 8193 windows.
#   - Run roots and data on $WORK (vault is over its soft quota).
#
# Fail-closed (all non-control arms): refuses to train unless the committed
# long-window audit bundle reports all four gates passing at the 2.5B/8192
# blend size (analysis/longwin_midtrain_prep_20260826/).
#
# DO NOT SUBMIT until the audit bundle exists and passes; see
# analysis/longwin_midtrain_prep_20260826/PREP_REPORT.md for the exact
# sbatch commands and review checklist.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"

source ./scripts/env.sh

CONDITION="${CONDITION:?set CONDITION=control|longdoc|logic_band25|nl_exact_band25|condensed_logic_band25}"
AUDIT_DIR="${AUDIT_DIR:-analysis/longwin_midtrain_prep_20260826}"
PACK_ROOT="${PACK_ROOT:-${WORK}/synthetic-RLVL/nanosets_longwin_20260826}"

case "${CONDITION}" in
  control)
    GRID_IDX=0
    MIX_TAG=p0
    ;;
  longdoc)
    GRID_IDX=2  # logic p10 grid row; folder + weight fully overridden below
    MIX_TAG=p10
    DOCPACK_FOLDER="${PACK_ROOT}/longdoc_control"
    AUDIT_JSON="${AUDIT_DIR}/audit_docpack_longdoc_control.json"
    ;;
  logic_band25)
    GRID_IDX=2
    MIX_TAG=p10
    DOCPACK_FOLDER="${PACK_ROOT}/logic_band25"
    AUDIT_JSON="${AUDIT_DIR}/audit_docpack_logic_band25.json"
    ;;
  nl_exact_band25)
    GRID_IDX=7  # nl_exact p10 grid row
    MIX_TAG=p10
    DOCPACK_FOLDER="${PACK_ROOT}/nl_exact_band25"
    AUDIT_JSON="${AUDIT_DIR}/audit_docpack_nl_exact_band25.json"
    ;;
  condensed_logic_band25)
    GRID_IDX=2
    MIX_TAG=p10
    DOCPACK_FOLDER="${PACK_ROOT}/condensed_logic_band25"
    AUDIT_JSON="${AUDIT_DIR}/audit_docpack_condensed_logic_band25.json"
    ;;
  *)
    echo "Unknown CONDITION=${CONDITION}" >&2
    exit 2
    ;;
esac

export TRAIN_STEPS="${TRAIN_STEPS:-2385}"
export SEQ_LEN="${SEQ_LEN:-8192}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-32}"

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
  export LOGIC_DATASET_FOLDER="${DOCPACK_FOLDER}"
  export NL_DATASET_FOLDER="${DOCPACK_FOLDER}"
  echo "longwin midtrain ${CONDITION}: proof_weight=${PROOF_WEIGHT_OVERRIDE} padding_label_id=${NANOSET_PADDING_LABEL_ID} steps=${TRAIN_STEPS} folder=${DOCPACK_FOLDER}"
fi

export SLURM_ARRAY_TASK_ID="${GRID_IDX}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-250}"
export LR=1.0e-5
export MIN_LR=1.0e-6
export WARMUP_STEPS=128
export LR_DECAY_STARTING_STEP=128
export LR_DECAY_STEPS=18946
export NORMAL_DATASET_FOLDER="${NORMAL_DATASET_FOLDER:-${WORK}/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino}"
export PRETRAINED_NANOTRON_CKPT="${PRETRAINED_NANOTRON_CKPT:-${WORK}/synthetic-RLVL/nanotron_checkpoints/qwen25_7b_tp1}"
export NANOTRON_PG_TIMEOUT_MINUTES="${NANOTRON_PG_TIMEOUT_MINUTES:-120}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export RUN_TAG_OVERRIDE="${RUN_TAG_OVERRIDE:-qwen25_7b_longwin_${CONDITION}_${MIX_TAG}_2p5b_s8192}"
export RUN_ROOT_OVERRIDE="${RUN_ROOT_OVERRIDE:-${WORK}/synthetic-RLVL/nanotron_longwin_midtrain/${RUN_TAG_OVERRIDE}}"
export FINAL_CHECKPOINT_ALIAS=2p5b
export PRUNE_SUPERSEDED_RUN_CHECKPOINTS=1

# Mid-run checkpoint pruner (same rationale as the docpack rerun wrapper):
# keep the two newest numeric checkpoint dirs, delete anything older. The
# subshell must NOT inherit errexit/pipefail (see 2026-08-14 incident).
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
