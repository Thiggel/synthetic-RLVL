#!/usr/bin/env bash
#SBATCH --job-name=grpo_formal
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:4
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/atuin/c107fa/c107fa12/rlvl_data/logs/%x_%j.out
#SBATCH --error=/home/atuin/c107fa/c107fa12/rlvl_data/logs/%x_%j.err

# alex (rtxpro6k: 8x RTX PRO 6000 Blackwell 96 GB, 1-day walltime) version of grpo_formal.slurm:
# same env interface, env from alex_grpo_env.sh (unpacked to node-local $TMPDIR), data under
# RLVL_DATA_ROOT=/home/atuin/c107fa/c107fa12/rlvl_data (mirror of gruenau's rlvl_data).
# CPUs/memory follow the GPU count (alex: 32 cores and ~92 GB per GPU); --gres=gpu:rtxpro6k:N overrides.
# Required env: POLICY (SFT run dir name under formal_mixture_sft_20260925), ARM, RUN_NAME.
# Optional: NO_TAG=1, EXTRA_ARGS (passed to grpo_formal.py), POLICY_CKPT (default final),
# STOP_MARGIN_MIN (default 30): grpo_formal.py --stop-at saves a checkpoint and stops that many minutes
# before the walltime; with --resumable in EXTRA_ARGS the next job continues from it. Chain 1-day jobs
# with a unique --job-name and --dependency=singleton; a job exits at once when ${OUT}/final exists.
# Export the variables in the shell and submit with --export=ALL (never put comma values in --export=).
# Output: ${RLVL_DATA_ROOT}/grpo_formal_20260928/${RUN_NAME}/

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export RLVL_DATA_ROOT="${RLVL_DATA_ROOT:-/home/atuin/c107fa/c107fa12/rlvl_data}"
OUT="${RLVL_DATA_ROOT}/grpo_formal_20260928/${RUN_NAME:?}"
if [[ -d "${OUT}/final" ]]; then echo "done: ${OUT}/final exists"; exit 0; fi
source scripts/slurm/jobs/alex_grpo_env.sh

MODEL="${RLVL_DATA_ROOT}/formal_mixture_sft_20260925/${POLICY:?}/${POLICY_CKPT:-final}"
mkdir -p "${OUT}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then NGPU=$(tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES}" | wc -l)
else NGPU=$(nvidia-smi -L | wc -l); fi
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
# stop (checkpoint + exit 0 without final) STOP_MARGIN_MIN before the job's end time
END_TS=$(date -d "$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %e 2>/dev/null)" +%s 2>/dev/null || echo $(( $(date +%s) + 86400 )))
STOP_AT=$(( END_TS - ${STOP_MARGIN_MIN:-30} * 60 ))
echo "policy=${POLICY} arm=${ARM:?} ngpu=${NGPU} out=${OUT} host=$(hostname) stop_at=$(date -d "@${STOP_AT}" +%FT%T)"
ACC=(--num_processes "${NGPU}" --mixed_precision bf16)
if (( NGPU > 1 )); then ACC+=(--use_deepspeed --zero_stage 2); fi
# vLLM's startup memory profile can assert on a transient memory change (gruenau G9 7365); retry once.
ERRLOG="${TMPDIR}/grpo_${SLURM_JOB_ID:-0}.err"
for attempt in 1 2; do
  if "${VENV}/bin/python" -m accelerate.commands.launch "${ACC[@]}" scripts/grpo_formal.py --model "${MODEL}" --out-dir "${OUT}" \
      --arm "${ARM}" ${NO_TAG:+--no-tag} --stop-at "${STOP_AT}" ${EXTRA_ARGS:-} 2> >(tee "${ERRLOG}" >&2); then exit 0; fi
  if (( attempt == 1 )) && grep -q "Error in memory profiling" "${ERRLOG}"; then
    echo "vLLM memory-profiling race; retrying in 60 s" >&2; sleep 60; else exit 1; fi
done
