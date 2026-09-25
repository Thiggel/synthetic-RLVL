#!/usr/bin/env bash
# Submit the formal-CoT mixture sweep (2026-09-25) in serial "lanes" so that
# user laitenbf always leaves >= 2 actually-free GPUs on every node (other
# users hold cards outside Slurm: gruenau11 H100 0,2,3 and gruenau12 L40 0,6
# on 2026-09-25 17:15, so 9B on gruenau11 is deferred):
#   lanes A, B  gruenau7 / gruenau8  2 x RTX A6000 each   2B training
#   lanes C, D  gruenau12            2 x L40 each         0.8B training
#   eval lanes E1 (A, B) and E2 (C, D), 1 x L40 each on gruenau12
# => 2/4 on gruenau7, 2/4 on gruenau8, 6 of the 8 free L40s on gruenau12.
# Jobs in a lane chain with afterany (a failure does not stall the lane);
# evals need afterok on their training job and are killed if it fails.
# On the 46-48 GB cards training uses per-device batch 1 + gradient
# checkpointing; the global batch stays 128, so the recipe is unchanged.
# Cells with a finished run / eval are skipped by the job scripts.
#
# Usage: scripts/submit_formal_mixture_sweep_lanes.sh [--dry-run]
set -euo pipefail
cd "$(dirname "$0")/.."

DRY=0; [[ "${1:-}" == --dry-run ]] && DRY=1
TRAIN=scripts/slurm/jobs/gruenau_formal_mix_train_2026-09-25.slurm
EVAL=scripts/slurm/jobs/gruenau_formal_mix_eval_2026-09-25.slurm
OUT_ROOT=/vol/tmp2/laitenbf/rlvl_data/formal_mixture_sft_20260925

sub() {
  if (( DRY )); then echo "sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch --parsable "$@"; fi
}
dep() { [[ -n "$1" ]] && echo "--dependency=afterany:$1" || true; }

declare -A PREV=()   # last job id per lane
declare -A LANE_RES=(
  [A]="-p gpu --nodelist=gruenau7 --gres=gpu:rtxa6000:2"
  [B]="-p gpu --nodelist=gruenau8 --gres=gpu:rtxa6000:2"
  [C]="-p gpu-staff --nodelist=gruenau12 --gres=gpu:l40:2"
  [D]="-p gpu-staff --nodelist=gruenau12 --gres=gpu:l40:2"
)
declare -A TLIM=([0.8b]=0-09:00:00 [2b]=0-16:00:00)

submit_cell() {  # lane eval_lane model frac
  local lane=$1 elane=$2 m=$3 x=$4 t e
  local mix; mix=$(printf "dolci_rlvlgen_p%02d" "$x")
  local run="qwen35_${m}_${mix}_lr5em6_seed3407"
  t=$(sub --job-name="fmix_${m}_p${x}" ${LANE_RES[$lane]} --cpus-per-task=32 --time="${TLIM[$m]}" \
        $(dep "${PREV[$lane]:-}") \
        --export=ALL,MODEL_KEY="${m}",MIX="${mix}",PER_DEVICE_BATCH=1,GRAD_CKPT=1,MIN_FREE_MIB=40000 "$TRAIN")
  PREV[$lane]=$t
  local edep="afterok:${t}"
  [[ -n "${PREV[$elane]:-}" ]] && edep+=",afterany:${PREV[$elane]}"
  e=$(sub --job-name="fmix_eval_${m}_p${x}" -p gpu-staff --nodelist=gruenau12 --gres=gpu:l40:1 \
        --kill-on-invalid-dep=yes --dependency="${edep}" --export=ALL,CKPT="${OUT_ROOT}/${run}" "$EVAL")
  PREV[$elane]=$e
  echo "${m} p${x}: lane ${lane} train ${t} eval ${e}"
}

# 2B on A/B, 0.8B on C/D (0.8B p50 is done); key points first, lanes alternated
# so the shared eval lanes follow completion order.
A=(0 50 10 30 40 20); B=(25 5 15 35 45)
C=(0 10 30 40 20);    D=(25 5 15 35 45)
for k in 0 1 2 3 4 5; do
  [[ -n "${A[$k]:-}" ]] && submit_cell A E1 2b "${A[$k]}"
  [[ -n "${B[$k]:-}" ]] && submit_cell B E1 2b "${B[$k]}"
  [[ -n "${C[$k]:-}" ]] && submit_cell C E2 0.8b "${C[$k]}"
  [[ -n "${D[$k]:-}" ]] && submit_cell D E2 0.8b "${D[$k]}"
done
exit 0
