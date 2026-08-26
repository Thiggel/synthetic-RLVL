#!/usr/bin/env bash
#SBATCH --job-name=longwin_env_smoke
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100small:1
#SBATCH --exclude=a0531,a0532,a0533,a0934
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# HARD GATE before any long-window midtrain is released.
#
# The $WORK/nanotron venv was rebuilt from scratch (longwin_rebuild_nanotron_env
# _2026-08-26.slurm) and its transitive dependency versions are NOT identical to
# the accepted env, so the training path must be re-smoked. This wraps the
# accepted tiny-nanoset smoke (nanotron_qwen25_tiny_nanoset_smoke_2026-07-03)
# but points it at the ACTUAL long-window artifacts and geometry:
#   - the rebuilt Dolmino nanoset as the normal stream
#   - the real logic_band25 docpack as the proof stream
#   - padding-label masking on (the docpack path), padded-compensated weight
#   - SEQ_LEN 8192 (not the smoke's 256 default)
# so it exercises the 8193-window loader, the padding mask, and a real
# optimizer step on the rebuilt env.
#
# Passing this does NOT validate throughput or memory at full 8-GPU scale; it
# validates that the rebuilt env can import, load, mask, and step.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"

PACK_ROOT="${PACK_ROOT:-${WORK}/synthetic-RLVL/nanosets_longwin_20260826}"

export SEQ_LEN="${SEQ_LEN:-8192}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export TRAIN_STEPS="${TRAIN_STEPS:-3}"
export MIX_KIND="${MIX_KIND:-logic}"
export MIX_PCT="${MIX_PCT:-10}"
export NORMAL_DATASET_FOLDER="${NORMAL_DATASET_FOLDER:-${WORK}/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino}"
export LOGIC_DATASET_FOLDER="${LOGIC_DATASET_FOLDER:-${PACK_ROOT}/logic_band25}"
export NL_DATASET_FOLDER="${NL_DATASET_FOLDER:-${PACK_ROOT}/nl_exact_band25}"
export NANOSET_PADDING_LABEL_ID="${NANOSET_PADDING_LABEL_ID:-151662}"

bash scripts/slurm/jobs/nanotron_qwen25_tiny_nanoset_smoke_2026-07-03.slurm

echo "long-window env smoke PASSED (seq ${SEQ_LEN}, ${TRAIN_STEPS} steps, rebuilt \$WORK/nanotron)"
