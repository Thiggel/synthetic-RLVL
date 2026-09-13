#!/usr/bin/env bash
#SBATCH --job-name=gru_rl_checker
#SBATCH --partition=gpu-staff
#SBATCH --gres=gpu:h100nvl:2
#SBATCH --nodelist=gruenau11
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
ROOT=/vol/tmp2/laitenbf
export PATH="${ROOT}/.venv_rlvl_gruenau/bin:${PATH}"
export VIRTUAL_ENV="${ROOT}/.venv_rlvl_gruenau"
export HF_HOME="${ROOT}/rlvl_data/hf_home"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
OUT="${ROOT}/rlvl_data/rl_checker_20260912/${OUT_NAME:?}"
[ -s "${OUT}/final/adapter_model.safetensors" ] && { echo "already done"; exit 0; }
# Under this scheduler nvidia-smi inside the job still lists every card on the
# node, and CUDA_VISIBLE_DEVICES does not always name the one we were given, so
# wait on the largest free card rather than on a guessed index.
# Two cards are requested because the node is MPS-shared and one of the pair is
# often mostly occupied by a neighbour. Train on whichever card is free.
for attempt in $(seq 1 40); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | tail -1)
  [ $((free/1024)) -ge 45 ] && break
  echo "waiting: largest free card has $((free/1024)) GiB"; sleep 180
done
[ $((free/1024)) -ge 45 ] || { echo "no card with 45 GiB free" >&2; exit 3; }
BEST=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -n | tail -1 | cut -d, -f1)
export CUDA_VISIBLE_DEVICES="${BEST}"
echo "training on card ${BEST} with $((free/1024)) GiB free"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
python scripts/rl_checker_grpo.py --model "${MODEL_DIR:?}" --reward "${REWARD:?}" --task "${TASK:-mixture}" \
  --output-dir "${OUT}" --max-steps "${MAX_STEPS:-300}" \
  --num-generations "${NUM_GEN:-8}" --per-device-batch-size "${PDBS:-8}" --grad-accum "${GA:-2}" \
  --max-completion-length "${MAXCOMP:-9000}" --vllm-gpu-memory-utilization "${VLLMUTIL:-0.30}" \
  ${FORMAL:+--formal} ${DEPTHS:+--depths $DEPTHS}
echo "done ${OUT_NAME}"
