#!/usr/bin/env bash
#SBATCH --job-name=q25_rl_checker
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --exclude=a0531,a0532,a0533,a0934
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# GRPO with a program checker as the reward. REWARD=correct|valid|both,
# MODEL_DIR, OUT_NAME, and FORMAL=1 when the model writes formal notation.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
trap 'kill -- -$$ 2>/dev/null || true' EXIT
ROOT=/vol/tmp2/laitenbf   # unused on alex; paths below are alex-local
export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"
source ./scripts/env.sh
export PATH="${HPCVAULT}/.venv_rlvl_posttrain/bin:${PATH}"
export VIRTUAL_ENV="${HPCVAULT}/.venv_rlvl_posttrain"
export HF_HOME="${HPCVAULT}/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
OUT="${WORK}/synthetic-RLVL/rl_checker_20260913/${OUT_NAME:?}"
[ -s "${OUT}/final/adapter_model.safetensors" ] && { echo "already done"; exit 0; }
# Under this scheduler nvidia-smi inside the job still lists every card on the
# node, and CUDA_VISIBLE_DEVICES does not always name the one we were given, so
# wait on the largest free card rather than on a guessed index.
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
python scripts/rl_checker_grpo.py --model "${MODEL_DIR:?}" --reward "${REWARD:?}" --task "${TASK:-mixture}" \
  --mixture "${HPCVAULT}/synthetic-RLVL/datasets/rlvr_mixture_20260912/train.jsonl" \
  --output-dir "${OUT}" --max-steps "${MAX_STEPS:-300}" \
  --num-generations "${NUM_GEN:-8}" --per-device-batch-size "${PDBS:-8}" --grad-accum "${GA:-2}" \
  --max-completion-length "${MAXCOMP:-9000}" --vllm-gpu-memory-utilization "${VLLMUTIL:-0.30}" \
  ${FORMAL:+--formal} ${DEPTHS:+--depths $DEPTHS}
echo "done ${OUT_NAME}"
