#!/usr/bin/env bash
#SBATCH --job-name=gru_native_passk
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=16
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Sampled native-format derivations (n=16, T=0.8) for checker selection.
# MODEL_DIR: HF checkpoint; OUT_NAME: output directory name.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
trap 'kill -- -$$ 2>/dev/null || true' EXIT
ROOT=/vol/tmp2/laitenbf
export PATH="${ROOT}/.venv_rlvl_gruenau_eval/bin:${PATH}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 TOKENIZERS_PARALLELISM=false HF_HOME="${ROOT}/rlvl_data/hf_home"
MODEL_DIR="${MODEL_DIR:?}"; OUT_NAME="${OUT_NAME:?}"
OUT="${ROOT}/rlvl_data/native_passk_20260911/${OUT_NAME}"
[ -e "${OUT}/.complete" ] && { echo "already done"; exit 0; }
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
for attempt in $(seq 1 60); do
  free=$(nvidia-smi -i "${GPU_ID:-0}" --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ $((free/1024)) -ge 36 ] && break; echo "waiting: $((free/1024)) GiB free"; sleep 120
done
total=$(nvidia-smi -i "${GPU_ID:-0}" --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
UTIL=$(python3 -c "print(round(min(0.72, ($free-4096)/$total),3))")
python scripts/analysis/sample_native_derivations.py --checkpoint "${MODEL_DIR}" --out "${OUT}" \
  --n "${N:-16}" --max-tokens "${MAX_TOKENS:-9000}" --max-model-len 16384 --gpu-memory-utilization "${UTIL}" ${LIMIT:+--limit $LIMIT}
touch "${OUT}/.complete"; echo "done ${OUT_NAME}"
