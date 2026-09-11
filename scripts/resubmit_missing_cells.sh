#!/usr/bin/env bash
# Resubmit every evaluation cell that has no .complete marker. Safe to run
# repeatedly: finished cells exit in seconds. Needed because jobs submitted
# while the home volume was full never started and left no logs.
set -uo pipefail
cd "$(dirname "$0")/.."
D=/vol/home-vol2/ml/laitenbf/rlvl_data
R=$D/lm_eval_results
sub(){
  local kind=$1 arm=$2 suite=$3 seed=$4 ckpt=${5:-} run=${6:-}
  local suf=""; [ "$seed" = 3407 ] || suf="_seed$seed"
  local name
  if [ -n "$run" ]; then name=$run
  elif [ "$kind" = base ]; then name="qwen25_7b_longwin_${arm}_2p5b_base"
  else name="qwen25_7b_longwin_${arm}_2p5b_dolci_100k_lr5em6${suf}"; fi
  [ -e "$R/gruenau_${kind}_${suite}_20260910${suf}/$name/.complete" ] && return 0
  if [ -n "$ckpt" ]; then
    sbatch --export=ALL,KIND=$kind,ARM=$arm,SUITE=$suite,SEED=$seed,CKPT_OVERRIDE=$ckpt,RUN_NAME_OVERRIDE=$run scripts/slurm/jobs/gruenau_full_eval.slurm >/dev/null && echo "submitted $kind $arm $suite $seed"
  else
    sbatch --export=ALL,KIND=$kind,ARM=$arm,SUITE=$suite,SEED=$seed scripts/slurm/jobs/gruenau_full_eval.slurm >/dev/null && echo "submitted $kind $arm $suite $seed"
  fi
}
CORE="deduction deduction_mc deduction_cot deduction_pert folio_gpqa standard multihop deduction_native deduction_native_long"
for a in nl_exact_band25 logic_band25 condensed_logic_band25; do
  run="qwen25_7b_longwin_${a}_p30_2p5b_dolci_100k_lr5em6"; ck="$D/post_sft_p30_alex/$run/final"
  [ -s "$ck/config.json" ] || continue
  for s in $CORE deduction_deep; do sub it "${a}_p30" "$s" 3407 "$ck" "$run"; done
done
for a in logic_band25 nl_exact_band25 control; do
  run="qwen25_7b_longwin_${a}_2p5b_modecond_100k_lr5em6"; ck="$D/post_sft_modecond_alex/$run/final"
  [ -s "$ck/config.json" ] || continue
  for s in switch_formal switch_english $CORE deduction_deep; do sub it "${a}_2p5b_modecond" "$s" 3407 "$ck" "$run"; done
done
for spec in "control sftlogic" "control sftnl_exact" "logic_band25 sftlogic" "nl_exact_band25 sftnl_exact"; do
  set -- $spec; run="qwen25_7b_longwin_${1}_2p5b_${2}_100k_lr5em6"; ck="$D/post_sft_notation_alex/$run/final"
  [ -s "$ck/config.json" ] || continue
  for s in $CORE deduction_deep; do sub it "${1}_2p5b_${2}" "$s" 3407 "$ck" "$run"; done
done
for a in control logic_band25 nl_exact_band25; do for s in deduction standard multihop deduction_native deduction_mc deduction_cot deduction_pert deduction_deep; do sub base "$a" "$s" 3407; done; done
