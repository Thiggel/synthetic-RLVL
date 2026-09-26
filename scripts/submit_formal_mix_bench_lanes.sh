#!/usr/bin/env bash
# Downstream benchmarks ("Lead into Gold" suite) for every checkpoint of the
# 2026-09-25 formal-mixture sweep, in serial lanes of 1 GPU each so that user
# laitenbf leaves >= 2 GPUs free per node.
#   default lanes (2026-09-26, idle gpu-wbiml nodes, RTX 3090 24 GB):
#     guppi5 x2, guppi8 x2 (4 cards each), guppi6 x1, guppi7 x1 (3 cards each)
#   9B (18 GB weights, 32k multihop) needs a bigger card, e.g.
#     scripts/submit_formal_mix_bench_lanes.sh --models 9b --lanes gruenau12:1
# A job waits (afterok) for its model's training job when that is still in
# the queue (looked up by job name fmix_<m>_p<x>) and is killed if it fails;
# lane jobs chain with afterany. With --smoke, a LIMIT=10 run of all suites
# (results study formal_mix_smoke) gates the first job of every lane.
# Finished suites (.complete) are skipped by the job script.
# --tagged submits the format-tagged eval (gruenau_formal_mix_tagged_2026-09-26.slurm,
# job names fmix_tag_<m>_p<x>) instead; --chain "<id> ..." makes lane i start
# after job i (afterany), to append to lanes that are already busy.
#
# Usage: scripts/submit_formal_mix_bench_lanes.sh [--dry-run] [--smoke] [--tagged]
#          [--models "0.8b 2b"] [--lanes "guppi5:2 guppi8:2 guppi6:1 guppi7:1"] [--chain "<ids>"]
set -euo pipefail
cd "$(dirname "$0")/.."

DRY=0; SMOKE=0; MODELS="0.8b 2b"; LANES="guppi5:2 guppi8:2 guppi6:1 guppi7:1"; CHAIN=""
BENCH=scripts/slurm/jobs/gruenau_formal_mix_bench_2026-09-26.slurm; PREFIX=fmix_bench
while (( $# )); do
  case "$1" in
    --dry-run) DRY=1 ;;
    --smoke) SMOKE=1 ;;
    --models) MODELS=$2; shift ;;
    --lanes) LANES=$2; shift ;;
    --chain) CHAIN=$2; shift ;;
    --tagged) BENCH=scripts/slurm/jobs/gruenau_formal_mix_tagged_2026-09-26.slurm; PREFIX=fmix_tag ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
  shift
done
OUT_ROOT=/vol/tmp2/laitenbf/rlvl_data/formal_mixture_sft_20260925
declare -A XS=([0.8b]="0 5 10 15 20 25 30 35 40 45 50" [2b]="0 5 10 15 20 25 30 35 40 45 50" [9b]="0 10 25 50")

sub() {
  if (( DRY )); then echo "sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch --parsable "$@"; fi
}
res() {  # node -> sbatch resource args
  case "$1" in
    guppi*) echo "-p gpu-wbiml --nodelist=$1 --gres=gpu:rtx3090:1 --cpus-per-task=8 --mem=96G --export=ALL,MIN_FREE_MIB=20000" ;;
    gruenau12) echo "-p gpu-staff --nodelist=gruenau12 --gres=gpu:l40:1 --cpus-per-task=16 --mem=128G" ;;
    gruenau9|gruenau10) echo "-p gpu --nodelist=$1 --gres=gpu:a10080gb:1 --cpus-per-task=16 --mem=128G" ;;
    *) echo "no resource spec for node $1" >&2; exit 2 ;;
  esac
}

LANE_NODES=()
for spec in ${LANES}; do
  for _ in $(seq 1 "${spec#*:}"); do LANE_NODES+=("${spec%%:*}"); done
done
declare -A PREV=()
i=0; for c in ${CHAIN}; do PREV[$i]=$c; i=$((i + 1)); done

SMOKE_ID=""
if (( SMOKE )); then
  m0=${MODELS%% *}
  r=$(res "${LANE_NODES[0]}"); r="${r%% --export=*}"
  SMOKE_ID=$(sub --job-name="${PREFIX}_smoke_${m0}" ${r} \
    --export=ALL,MIN_FREE_MIB=20000,RUN_NAME="qwen35_${m0}_dolci_rlvlgen_p00_lr5em6_seed3407",LIMIT=10,PER_BENCH_LIMIT=5,RESULTS=formal_mix_smoke \
    "$BENCH")
  echo "smoke ${m0} p00: ${SMOKE_ID}"
fi

# finished checkpoints first, then those still training (in queue order)
ready=(); later=()
for m in ${MODELS}; do
  for x in ${XS[$m]}; do
    run=$(printf "qwen35_%s_dolci_rlvlgen_p%02d_lr5em6_seed3407" "$m" "$x")
    tj=$(squeue -u "$USER" -h -n "fmix_${m}_p${x}" -o "%i" | head -1)
    if [[ -n "${tj}" ]]; then later+=("${m}|${x}|${run}|${tj}")
    elif [[ -s "${OUT_ROOT}/${run}/final/config.json" ]]; then ready+=("${m}|${x}|${run}|")
    else echo "skip ${run}: no checkpoint and no training job in queue" >&2
    fi
  done
done

k=0
for cell in "${ready[@]}" "${later[@]}"; do
  IFS='|' read -r m x run tj <<<"${cell}"
  lane=$(( k % ${#LANE_NODES[@]} )); node=${LANE_NODES[$lane]}; k=$(( k + 1 ))
  deps=()
  [[ -n "${tj}" ]] && deps+=("afterok:${tj}")
  if [[ -n "${PREV[$lane]:-}" ]]; then deps+=("afterany:${PREV[$lane]}")
  elif [[ -n "${SMOKE_ID}" ]]; then deps+=("afterok:${SMOKE_ID}")
  fi
  depflag=(); (( ${#deps[@]} )) && depflag=(--kill-on-invalid-dep=yes "--dependency=$(IFS=,; echo "${deps[*]}")")
  r=$(res "${node}")
  # res() may carry its own --export; merge RUN_NAME into it
  if [[ "${r}" == *--export=ALL,* ]]; then r="${r/--export=ALL,/--export=ALL,RUN_NAME=${run},}"
  else r="${r} --export=ALL,RUN_NAME=${run}"; fi
  j=$(sub --job-name="${PREFIX}_${m}_p${x}" ${r} "${depflag[@]}" "$BENCH")
  PREV[$lane]=$j
  echo "${m} p${x}: lane ${lane} (${node}) ${PREFIX} ${j}${tj:+ after train ${tj}}"
done
exit 0
