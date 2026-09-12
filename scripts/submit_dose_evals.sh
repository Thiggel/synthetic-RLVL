#!/usr/bin/env bash
# Submit every missing dose-sweep evaluation cell. Safe to re-run.
set -uo pipefail
cd "$(dirname "$0")/.."
W="${WORK:-/home/atuin/c107fa/c107fa12}"; V="${HPCVAULT:-/home/vault/c107fa/c107fa12}"
n=0
for d in "$W"/synthetic-RLVL/post_sft_dose_20260912/*/; do
  run=$(basename "$d"); [ -s "$d/final/config.json" ] || continue
  for s in deduction folio_gpqa native standard; do
    [ -e "$V/synthetic-RLVL/lm_eval_results/dose_sweep_20260912/$s/$run/.complete" ] && continue
    sbatch --export=ALL,RUN_NAME="$run",SUITE="$s" scripts/slurm/jobs/alex_dose_eval_2026-09-12.slurm >/dev/null && { echo "submitted $run $s"; n=$((n+1)); }
  done
done
echo "submitted $n cells"
