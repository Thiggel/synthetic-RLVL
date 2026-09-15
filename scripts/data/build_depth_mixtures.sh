#!/usr/bin/env bash
# Chain-length axis: fix the share at ten percent and vary the depth of the
# training proofs. Band 25 already exists from the dose sweep.
set -euo pipefail
cd "$(dirname "$0")/../.."
source ./scripts/env.sh
export PATH="${HPCVAULT}/.venv_rlvl_posttrain/bin:${PATH}"
export HF_HOME="${HPCVAULT}/.cache/huggingface"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
D="${HPCVAULT}/synthetic-RLVL/datasets/dolci_no_tools_single_turn_100k_seed3407_20260803"
B="${HPCVAULT}/synthetic-RLVL/datasets/branchproof_unique_v2_20260710"
O="${HPCVAULT}/synthetic-RLVL/datasets/depth_sweep_20260915"
mkdir -p "$O"
for tmpl in logic nl_exact; do
  for band in 5 15; do
    name="${tmpl}_b${band}"
    [ -d "$O/$name" ] && { echo "have $name"; continue; }
    python scripts/data/build_reasoning_mixture_sft.py \
      --dolci "$D" --bp-root "$B" \
      --out-root "$O/tmp_$name" --band "$band" --template "$tmpl" \
      --frac 0.10 --total 100000 --seed 20260912
    inner=$(find "$O/tmp_$name" -maxdepth 1 -mindepth 1 -type d | head -1)
    mv "${inner:-$O/tmp_$name}" "$O/$name"
    rm -rf "$O/tmp_$name"
    echo "built $name"
  done
done
ls "$O"
