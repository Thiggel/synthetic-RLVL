#!/usr/bin/env bash
# Build the dose-sweep instruction mixtures on alex: replacement design,
# 100k examples per arm, band 25, shares 1/5/10/25 percent in both notations
# plus a zero-percent control. One fixed data seed; the three training seeds
# vary only the optimiser, so a difference cannot come from a different draw.
set -euo pipefail
cd "$(dirname "$0")/../.."
source ./scripts/env.sh
export PATH="${HPCVAULT}/.venv_rlvl_posttrain/bin:${PATH}"
export HF_HOME="${HPCVAULT}/.cache/huggingface"
# the builder imports the project package from the checkout
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
D="${HPCVAULT}/synthetic-RLVL/datasets/dolci_no_tools_single_turn_100k_seed3407_20260803"
B="${HPCVAULT}/synthetic-RLVL/datasets/branchproof_unique_v2_20260710"
O="${HPCVAULT}/synthetic-RLVL/datasets/dose_sweep_20260912"
mkdir -p "$O"
build(){ # template frac name
  local tmpl=$1 frac=$2 name=$3
  [ -d "$O/$name" ] && { echo "have $name"; return 0; }
  echo "=== building $name (template=$tmpl frac=$frac)"
  python scripts/data/build_reasoning_mixture_sft.py \
    --dolci "$D" --bp-root "$B" --out-root "$O/tmp_$name" \
    --band 25 --template "$tmpl" --frac "$frac" --total 100000 --seed 20260912
  # the builder names its output directory itself; hoist whatever it made
  inner=$(find "$O/tmp_$name" -maxdepth 1 -mindepth 1 -type d | head -1)
  mv "${inner:-$O/tmp_$name}" "$O/$name"
  rm -rf "$O/tmp_$name"
}
for tmpl in logic nl_exact; do
  for frac in 0.01 0.05 0.10 0.25; do
    tag=$(python3 -c "print('p%02d' % round(${frac}*100))")
    build "$tmpl" "$frac" "${tmpl}_${tag}"
  done
done
build logic 0.0 control
ls -d "$O"/*/ | sed 's#.*/\([^/]*\)/#\1#'
