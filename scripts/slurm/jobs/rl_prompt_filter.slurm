#!/usr/bin/env bash
#SBATCH --job-name=rl_filter
#SBATCH --partition=gpu-staff
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Per-prompt GRPO reward rate + harvest of passing completions (scripts/rl_prompt_filter.py).
# Required env: MODEL (absolute path), OUT (json path). Optional: FILTER_ARGS (passed through), MIN_FREE_MIB,
# WAIT_FREE_MIN.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
ROOT=/vol/tmp2/laitenbf
VENV="${ROOT}/.venv_rlvl_grpo"
export PATH="${VENV}/bin:${PATH}" VIRTUAL_ENV="${VENV}"
export HF_HOME="${ROOT}" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}" PATH="${CUDA_HOME:-/usr/local/cuda-13.2}/bin:${PATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0 TRITON_CACHE_DIR=/tmp/laitenbf_triton_${SLURM_JOB_ID:-0}
export PYTHONPATH="${RLVL_PYTHONPATH:-/vol/home-vol2/ml/laitenbf/RLVL-next/gen:/vol/home-vol2/ml/laitenbf/RLVL-next/rlvl/python}"  # RLVL_PYTHONPATH: frozen checker
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=index,name,memory.free --format=csv,noheader
# gruenau12 L40s are shared with foreign (non-Slurm) processes: wait until MIN_FREE_MIB are free (WAIT_FREE_MIN),
# then size vLLM to the free memory instead of a fixed 0.85 (2026-10-01: two harvests died at 26 GiB free).
q() { nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu="$1" --format=csv,noheader,nounits | head -1; }
FREE=$(q memory.free); TOTAL=$(q memory.total); waited=0
while (( FREE < ${MIN_FREE_MIB:-16000} && waited < ${WAIT_FREE_MIN:-360} )); do
  echo "only ${FREE} MiB free; waiting (${waited} min)" >&2; sleep 300; waited=$(( waited + 5 )); FREE=$(q memory.free)
done
if (( FREE < ${MIN_FREE_MIB:-16000} )); then echo "only ${FREE} MiB free: GPU shared with a foreign process" >&2; exit 3; fi
GPU_MEM=$(awk -v f="${FREE}" -v t="${TOTAL}" 'BEGIN { x = (f - 2500) / t; printf "%.2f", (x > .85 ? .85 : x) }')
echo "free ${FREE}/${TOTAL} MiB -> --gpu-mem ${GPU_MEM}"
python scripts/rl_prompt_filter.py --model "${MODEL:?}" --out "${OUT:?}" --gpu-mem "${GPU_MEM}" ${FILTER_ARGS:-}
