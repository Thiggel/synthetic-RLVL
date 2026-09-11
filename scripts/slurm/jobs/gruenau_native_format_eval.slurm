#!/usr/bin/env bash
#SBATCH --job-name=gru_native_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
trap 'kill -- -$$ 2>/dev/null || true' EXIT
ROOT=/vol/home-vol2/ml/laitenbf
export PATH="${ROOT}/.venv_rlvl_gruenau_eval/bin:${PATH}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 TOKENIZERS_PARALLELISM=false HF_HOME="${ROOT}/rlvl_data/hf_home"
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
# shared cards: wait for room, size the fraction to what is free, retry the race
for round in 1 2 3 4 5; do
  for attempt in $(seq 1 30); do free=$(nvidia-smi -i "${GPU_ID:-0}" --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ $((free/1024)) -ge 30 ] && break; echo "waiting: $((free/1024)) GiB free"; sleep 120; done
  total=$(nvidia-smi -i "${GPU_ID:-0}" --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  export GPU_UTIL=$(python3 -c "print(round(min(0.65, ($free-4096)/$total),3))")
  echo "round ${round}: ${free} MiB free, gpu_memory_utilization=${GPU_UTIL}"
  if python scripts/analysis/native_format_eval.py --checkpoint "${MODEL_DIR:?}" --out "${ROOT}/rlvl_data/native_eval_20260911/${OUT_NAME:?}"; then exit 0; fi
  echo "attempt ${round} failed; retrying" >&2; sleep 180
done
exit 4
