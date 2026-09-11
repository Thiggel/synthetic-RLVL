#!/usr/bin/env bash
#SBATCH --job-name=gru_rl_checker
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --nodelist=gruenau9
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# GRPO with a program checker as the reward. REWARD=correct|valid|both,
# MODEL_DIR, OUT_NAME, and FORMAL=1 when the model writes formal notation.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
trap 'kill -- -$$ 2>/dev/null || true' EXIT
ROOT=/vol/home-vol2/ml/laitenbf
export PATH="${ROOT}/.venv_rlvl_gruenau/bin:${PATH}"
export VIRTUAL_ENV="${ROOT}/.venv_rlvl_gruenau"
export HF_HOME="${ROOT}/rlvl_data/hf_home"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
OUT="${ROOT}/rlvl_data/rl_checker_20260911/${OUT_NAME:?}"
[ -s "${OUT}/final/adapter_model.safetensors" ] && { echo "already done"; exit 0; }
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
for attempt in $(seq 1 60); do
  free=$(nvidia-smi -i "${GPU_ID:-0}" --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ $((free/1024)) -ge 70 ] && break; echo "waiting: $((free/1024)) GiB free"; sleep 180
done
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
python scripts/rl_checker_grpo.py --model "${MODEL_DIR:?}" --reward "${REWARD:?}" \
  --output-dir "${OUT}" --max-steps "${MAX_STEPS:-300}" \
  --num-generations "${NUM_GEN:-8}" --per-device-batch-size "${PDBS:-8}" --grad-accum "${GA:-2}" \
  --max-completion-length "${MAXCOMP:-9000}" --vllm-gpu-memory-utilization "${VLLMUTIL:-0.30}" \
  ${FORMAL:+--formal} ${DEPTHS:+--depths $DEPTHS}
echo "done ${OUT_NAME}"
