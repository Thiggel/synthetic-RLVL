#!/usr/bin/env bash
# Submit the formal-CoT data-mixture sweep (2026-09-25) on gruenau gpu-staff.
#   Qwen3.5-0.8B-Base and 2B-Base at X = 0,5,...,50 percent synthetic (22 runs)
#   Qwen3.5-9B-Base at X = 0,10,25,50                                  (4 runs)
# Each training job gets an eval job with --dependency=afterok.
#
# Usage: scripts/submit_formal_mixture_sweep.sh [--dry-run] [--models "0.8b 2b 9b"] [--fracs "0 25"]
# The trainer and eval jobs skip cells whose outputs exist, so re-running is safe.
set -euo pipefail
cd "$(dirname "$0")/.."

DRY=0
MODELS="0.8b 2b 9b"
FRACS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --models) MODELS="$2"; shift 2 ;;
    --fracs) FRACS="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

TRAIN=scripts/slurm/jobs/gruenau_formal_mix_train_2026-09-25.slurm
EVAL=scripts/slurm/jobs/gruenau_formal_mix_eval_2026-09-25.slurm
OUT_ROOT=/vol/tmp2/laitenbf/rlvl_data/formal_mixture_sft_20260925
# Walltimes are roughly 2-3x the smoke-measured estimate on 2 H100 NVL
# (0.8B ~1.4 h, 2B ~2.3 h, 9B ~7.2 h plus ~4 min per FSDP checkpoint save;
# see docs/running_experiments.md).
declare -A TLIM=([0.8b]=0-04:00:00 [2b]=0-06:00:00 [9b]=0-16:00:00)

sub() {
  if (( DRY )); then echo "sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch --parsable "$@"; fi
}

for m in ${MODELS}; do
  if [[ -n "${FRACS}" ]]; then fr="${FRACS}"
  elif [[ "$m" == 9b ]]; then fr="0 10 25 50"
  else fr="0 5 10 15 20 25 30 35 40 45 50"; fi
  for x in ${fr}; do
    mix=$(printf "dolci_rlvlgen_p%02d" "$x")
    run="qwen35_${m}_${mix}_lr5em6_seed3407"
    t=$(sub --job-name="fmix_${m}_p${x}" --time="${TLIM[$m]}" \
          --export=ALL,MODEL_KEY="$m",MIX="$mix" "$TRAIN")
    egres=()
    # 9B bf16 weights (~18 GB) plus KV cache do not fit the partly occupied L40s.
    # Take all of gruenau11: two of its "h100nvl" cards are occupied PCIe cards,
    # and the eval job picks the freest allocated GPU.
    [[ "$m" == 9b ]] && egres=(--gres=gpu:h100nvl:4 --cpus-per-task=64)
    e=$(sub --job-name="fmix_eval_${m}_p${x}" --dependency="afterok:${t}" "${egres[@]}" \
          --export=ALL,CKPT="${OUT_ROOT}/${run}" "$EVAL")
    echo "${m} p${x}: train ${t} eval ${e}"
  done
done
