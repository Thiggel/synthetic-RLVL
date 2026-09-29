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
# Optional: NO_TAG=1, EXTRA_ARGS (passed to grpo_formal.py), POLICY_CKPT (checkpoint dir under POLICY, default final).
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
export PYTHONPATH=/vol/home-vol2/ml/laitenbf/RLVL-next/gen:/vol/home-vol2/ml/laitenbf/RLVL-next/rlvl/python
hash -r 2>/dev/null || true

MODEL="${ROOT}/rlvl_data/formal_mixture_sft_20260925/${POLICY:?}/${POLICY_CKPT:-final}"
OUT="${ROOT}/rlvl_data/grpo_formal_20260928/${RUN_NAME:?}"
mkdir -p "${OUT}"
# CUDA_VISIBLE_DEVICES (set by Slurm, or by grpo_formal_multi.slurm) wins over nvidia-smi -L
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then NGPU=$(tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES}" | wc -l)
else NGPU=$(nvidia-smi -L | wc -l); fi
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
# Fail fast when a process outside Slurm already holds the GPU (gruenau9/10/12, 2026-09-28)
FREE_MIB=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
if (( FREE_MIB < ${MIN_FREE_MIB:-0} )); then echo "only ${FREE_MIB} MiB free < MIN_FREE_MIB=${MIN_FREE_MIB}: GPU shared with a foreign process" >&2; exit 3; fi
echo "policy=${POLICY} arm=${ARM:?} ngpu=${NGPU} out=${OUT}"
ACC=(--num_processes "${NGPU}" --mixed_precision bf16)
if (( NGPU > 1 )); then ACC+=(--use_deepspeed --zero_stage 2); fi
# vLLM's startup memory profile asserts when a foreign process frees GPU memory mid-profile (G9 7365,
# 2026-09-29); that is transient, so retry once.
ERRLOG="/tmp/laitenbf_grpo_${SLURM_JOB_ID:-0}.err"
for attempt in 1 2; do
  if accelerate launch "${ACC[@]}" scripts/grpo_formal.py --model "${MODEL}" --out-dir "${OUT}" --arm "${ARM}" \
      ${NO_TAG:+--no-tag} ${EXTRA_ARGS:-} 2> >(tee "${ERRLOG}" >&2); then exit 0; fi
  if (( attempt == 1 )) && grep -q "Error in memory profiling" "${ERRLOG}"; then
    echo "vLLM memory-profiling race; retrying in 60 s" >&2; sleep 60; else exit 1; fi
done
