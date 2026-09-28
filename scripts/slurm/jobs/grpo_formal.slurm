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
# Optional: NO_TAG=1, EXTRA_ARGS (passed to grpo_formal.py).
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

MODEL="${ROOT}/rlvl_data/formal_mixture_sft_20260925/${POLICY:?}/final"
OUT="${ROOT}/rlvl_data/grpo_formal_20260928/${RUN_NAME:?}"
mkdir -p "${OUT}"
NGPU=$(nvidia-smi -L | wc -l)
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "policy=${POLICY} arm=${ARM:?} ngpu=${NGPU} out=${OUT}"
ACC=(--num_processes "${NGPU}" --mixed_precision bf16)
if (( NGPU > 1 )); then ACC+=(--use_deepspeed --zero_stage 2); fi
accelerate launch "${ACC[@]}" scripts/grpo_formal.py --model "${MODEL}" --out-dir "${OUT}" --arm "${ARM}" \
  ${NO_TAG:+--no-tag} ${EXTRA_ARGS:-}
