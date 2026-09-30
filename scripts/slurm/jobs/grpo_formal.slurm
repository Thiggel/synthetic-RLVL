#!/usr/bin/env bash
#SBATCH --job-name=grpo_formal
#SBATCH --partition=gpu-staff
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Stage-2 GRPO arm (docs/research_plan.md): scripts/grpo_formal.py with checker
# rewards (scripts/formal_rewards.py), TRL 1.14 + vLLM 0.30 colocate in
# .venv_rlvl_grpo. One process per GPU via accelerate (DeepSpeed ZeRO-2 if >1).
# Required env: POLICY (SFT run dir name under the formal-mixture sweep), ARM, RUN_NAME.
# Optional: NO_TAG=1, EXTRA_ARGS (passed to grpo_formal.py), POLICY_CKPT (checkpoint dir under POLICY, default final),
# RLVL_PYTHONPATH, STOP_MARGIN_MIN (with --resumable in EXTRA_ARGS), MIN_FREE_MIB + WAIT_FREE_MIN.
# Export the variables in the shell and submit with --export=ALL: `sbatch --export=ALL,EXTRA_ARGS="--benches a,b"`
# splits at the comma and silently drops everything after it.
# Output: rlvl_data/grpo_formal_20260928/${RUN_NAME}/

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
ROOT=/vol/tmp2/laitenbf
VENV="${ROOT}/.venv_rlvl_grpo"
export PATH="${VENV}/bin:${PATH}" VIRTUAL_ENV="${VENV}"
export HF_HOME="${ROOT}" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}" PATH="${CUDA_HOME:-/usr/local/cuda-13.2}/bin:${PATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0 TRITON_CACHE_DIR=/tmp/laitenbf_triton_${SLURM_JOB_ID:-0}
# RLVL_PYTHONPATH: a frozen checker, e.g. rlvl_data/checker_snapshot_pre_libext_20260930/{gen,rlvl_python}, so the
# reward cannot change when RLVL-next's library is rebuilt during a multi-day (resumed) run
export PYTHONPATH="${RLVL_PYTHONPATH:-/vol/home-vol2/ml/laitenbf/RLVL-next/gen:/vol/home-vol2/ml/laitenbf/RLVL-next/rlvl/python}"
hash -r 2>/dev/null || true

MODEL="${ROOT}/rlvl_data/formal_mixture_sft_20260925/${POLICY:?}/${POLICY_CKPT:-final}"
OUT="${ROOT}/rlvl_data/grpo_formal_20260928/${RUN_NAME:?}"
if [[ -d "${OUT}/final" && "${EXTRA_ARGS:-}" == *--resumable* ]]; then echo "done: ${OUT}/final exists"; exit 0; fi
mkdir -p "${OUT}"
# CUDA_VISIBLE_DEVICES (set by Slurm, or by grpo_formal_multi.slurm) wins over nvidia-smi -L
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then NGPU=$(tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES}" | wc -l)
else NGPU=$(nvidia-smi -L | wc -l); fi
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
# Fail fast when a process outside Slurm already holds the GPU (gruenau9/10/12, 2026-09-28)
# WAIT_FREE_MIN: first poll up to this many minutes for the memory to free up. In a singleton chain a fail-fast exit
# lets the next link start on the same busy GPU a second later, so L1_cvf 7955-7958 all died within a minute (2026-09-30).
free_mib() { nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1; }
FREE_MIB=$(free_mib)
for (( waited = 0; FREE_MIB < ${MIN_FREE_MIB:-0} && waited < ${WAIT_FREE_MIN:-0}; waited += 5 )); do
  echo "only ${FREE_MIB} MiB free; waiting (${waited}/${WAIT_FREE_MIN} min)" >&2; sleep 300; FREE_MIB=$(free_mib)
done
if (( FREE_MIB < ${MIN_FREE_MIB:-0} )); then echo "only ${FREE_MIB} MiB free < MIN_FREE_MIB=${MIN_FREE_MIB}: GPU shared with a foreign process" >&2; exit 3; fi
# --resumable runs: checkpoint and stop STOP_MARGIN_MIN (default 30) before the walltime; the next chained job
# (same --job-name, --dependency=singleton) resumes from it
STOP=()
if [[ "${EXTRA_ARGS:-}" == *--resumable* ]]; then
  END_TS=$(date -d "$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %e 2>/dev/null)" +%s 2>/dev/null || echo $(( $(date +%s) + 172800 )))
  STOP=(--stop-at $(( END_TS - ${STOP_MARGIN_MIN:-30} * 60 )))
fi
echo "policy=${POLICY} arm=${ARM:?} ngpu=${NGPU} out=${OUT} host=$(hostname) ${STOP[*]} pythonpath=${PYTHONPATH}"
ACC=(--num_processes "${NGPU}" --mixed_precision bf16)
if (( NGPU > 1 )); then ACC+=(--use_deepspeed --zero_stage 2); fi
# vLLM's startup memory profile asserts when a foreign process frees GPU memory mid-profile (G9 7365,
# 2026-09-29); that is transient, so retry once.
ERRLOG="/tmp/laitenbf_grpo_${SLURM_JOB_ID:-0}.err"
for attempt in 1 2; do
  if accelerate launch "${ACC[@]}" scripts/grpo_formal.py --model "${MODEL}" --out-dir "${OUT}" --arm "${ARM}" \
      ${NO_TAG:+--no-tag} "${STOP[@]}" ${EXTRA_ARGS:-} 2> >(tee "${ERRLOG}" >&2); then exit 0; fi
  if (( attempt == 1 )) && grep -q "Error in memory profiling" "${ERRLOG}"; then
    echo "vLLM memory-profiling race; retrying in 60 s" >&2; sleep 60; else exit 1; fi
done
