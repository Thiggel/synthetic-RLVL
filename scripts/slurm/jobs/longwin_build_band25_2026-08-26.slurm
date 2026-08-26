#!/usr/bin/env bash
#SBATCH --job-name=longwin_build_band25
#SBATCH --partition=a40,a100,a100mig
#SBATCH --gres=gpu:1
#SBATCH --exclude=a0531,a0532,a0533,a0934
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Long-window (8192) midtrain prep: build the band-25 trace corpora.
#
#  1. Materialize a FRESH 72k-row BranchProof-unique-v2 band-25 dataset
#     (depths 1-25 round-robin, hard_fsa_schema, branching 4, distractor 0.5)
#     with seed 20260830 -- disjoint from the 3407-family training seeds and
#     from 20260826 (reserved for the concurrent graded-deduction eval build)
#     and from 20260806 (pass@k sampling).
#  2. Render three same-latent-proof corpora as midtrain documents in the
#     SFT-style rendering (prompt+target, identical to reasoning_mixture_20260821;
#     NOT the old compact midtrain rendering):
#       logic / nl_exact / condensed_logic (see condensed_formal_rendering.py)
#     with fail-closed 8192-window length gates + token-length stats.
#  3. Round-trip validate 100 condensed traces through the citation-free
#     LogicEngine path.
#  4. Pack each corpus with the document-preserving packer at seq 8192.
#
# ~72k docs x ~3.9k tokens (logic) ~= 280M real tokens >= the 250M needed for
# a 10% slice of 2.5B.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"

source ./scripts/env.sh

VENV_PY="${HPCVAULT}/.venv_rlvl_posttrain/bin/python"
SEED="${LONGWIN_SEED:-20260830}"
ROWS="${LONGWIN_ROWS:-72000}"
SEQ_LEN="${LONGWIN_SEQ_LEN:-8192}"
MAT_ROOT="${MAT_ROOT:-${WORK}/synthetic-RLVL/datasets/branchproof_unique_v2_longwin_20260826}"
JSONL_ROOT="${JSONL_ROOT:-${WORK}/synthetic-RLVL/nanotron_data/longwin_band25_20260826}"
PACK_ROOT="${PACK_ROOT:-${WORK}/synthetic-RLVL/nanosets_longwin_20260826}"
PREP_DIR="analysis/longwin_midtrain_prep_20260826"
SUBSET="train_fixedtarget_up_to_25_${ROWS}"
mkdir -p "${PREP_DIR}"

# 1. Materialize (skip if complete).
if [[ ! -s "${MAT_ROOT}/${SUBSET}/train.parquet" ]]; then
  "${VENV_PY}" -m synthrlvl.datasets.materialize \
    --output-root "${MAT_ROOT}" \
    --train-up-to-25-subset "${SUBSET}" \
    --train-up-to-25-rows "${ROWS}" \
    --train-up-to-3-rows 0 --train-up-to-5-rows 0 --train-up-to-10-rows 0 \
    --train-up-to-15-rows 0 --train-up-to-20-rows 0 \
    --val-rows-per-step 0 --val-max-step 1 \
    --seed "${SEED}" \
    --distractor-ratio 0.5 \
    --difficulty hard_fsa_schema \
    --branching-factor 4 \
    --train-shortcut-rate 0.0 \
    --val-shortcut-rate 0.0
fi

# 2+3. Condensed round-trip validation (100 rows, fresh dataset) + renderings.
"${VENV_PY}" scripts/data/condensed_formal_rendering.py \
  --parquet "${MAT_ROOT}/${SUBSET}/train.parquet" \
  --sample 400 --validate 100 --seed "${SEED}" \
  --out-json "${PREP_DIR}/condensed_rendering_measurement_fresh72k.json"

"${VENV_PY}" scripts/data/build_longwin_trace_jsonls.py \
  --parquet "${MAT_ROOT}/${SUBSET}/train.parquet" \
  --out-root "${JSONL_ROOT}" \
  --seed "${SEED}" \
  --window "${SEQ_LEN}" \
  --renderings logic,nl_exact,condensed_logic

cp "${JSONL_ROOT}"/*_band25.stats.json "${JSONL_ROOT}/manifest.json" "${PREP_DIR}/" 2>/dev/null || true

# 4. Pack each rendering at seq 8192 (windows are seq+1).
for NAME in logic nl_exact condensed_logic; do
  OUT="${PACK_ROOT}/${NAME}_band25"
  if [[ -s "${OUT}/docpack_stats.json" ]]; then
    echo "docpack exists: ${OUT}"
    continue
  fi
  "${VENV_PY}" scripts/data/pack_document_preserving_nanoset.py \
    --input "${JSONL_ROOT}/${NAME}_band25.jsonl" \
    --output-folder "${OUT}" \
    --tokenizer "Qwen/Qwen2.5-7B" \
    --seq-len "${SEQ_LEN}" \
    --shuffle-seed 42
done

echo "band-25 long-window corpora complete under ${PACK_ROOT}"
