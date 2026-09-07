#!/usr/bin/env bash
#SBATCH --job-name=rl_corpus_build
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Fresh BranchProof band-25 corpus for the long-window RL study (2026-09-07).
#
# WHY A NEW SEED: the band-25 corpus at seed 20260830 was consumed in full
# (all 72,000 rows) as the 10% midtrain replacement slice for the logic /
# nl_exact / condensed arms. Running RL on those same latent proofs would be
# training on data the policy already saw during midtraining, so reward would
# partly measure memorisation -- and the logic-vs-NL contrast is exactly what
# that would confound. Seed 20260907 is disjoint from 20260830 (midtrain),
# 20260826 (graded-deduction eval), 20260806 (pass@k sampling) and the
# 3407-family SFT seeds.
#
# Generation parameters are otherwise IDENTICAL to the midtrain corpus
# (hard_fsa_schema, branching 4, distractor 0.5, depths 1-25 round robin),
# so the RL task matches the training distribution in form, not in content.
# CPU-bound; a40 is used only because alex has no CPU-only partition.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"
source ./scripts/env.sh
VENV_PY="${HPCVAULT}/.venv_rlvl_posttrain/bin/python"

SEED="${RL_SEED:-20260907}"
ROWS="${RL_ROWS:-30000}"
MAT_ROOT="${MAT_ROOT:-${WORK}/synthetic-RLVL/datasets/branchproof_unique_v2_rl_20260907}"
SUBSET="train_fixedtarget_up_to_25_${ROWS}"

if [[ -s "${MAT_ROOT}/${SUBSET}/train.parquet" ]]; then
  echo "RL corpus already present: ${MAT_ROOT}/${SUBSET}; skipping"
  exit 0
fi

"${VENV_PY}" -m synthrlvl.datasets.materialize \
  --output-root "${MAT_ROOT}" \
  --train-up-to-25-subset "${SUBSET}" \
  --train-up-to-25-rows "${ROWS}" \
  --train-up-to-3-rows 0 --train-up-to-5-rows 0 --train-up-to-10-rows 0 \
  --train-up-to-15-rows 0 --train-up-to-20-rows 0 \
  --val-rows-per-step 40 --val-max-step 25 \
  --seed "${SEED}" \
  --distractor-ratio 0.5 \
  --difficulty hard_fsa_schema \
  --branching-factor 4 \
  --train-shortcut-rate 0.0 \
  --val-shortcut-rate 0.0

echo "=== built:"
ls -la "${MAT_ROOT}" "${MAT_ROOT}/${SUBSET}"
"${VENV_PY}" - <<'PY'
import glob, os, json
import pyarrow.parquet as pq
root = os.environ.get("MAT_ROOT", "")
for f in sorted(glob.glob(os.path.join(root, "*", "*.parquet")))[:6]:
    t = pq.read_table(f)
    print(os.path.relpath(f, root), "rows=", t.num_rows, "cols=", t.column_names[:8])
PY
