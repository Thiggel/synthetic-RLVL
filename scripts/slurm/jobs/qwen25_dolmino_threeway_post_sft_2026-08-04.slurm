#!/usr/bin/env bash
#SBATCH --job-name=q25_dolm_3sft
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --array=0-2%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

CONDITIONS=(control logic nl_exact)
RUN_DIR_TAGS=(
  qwen25_7b_dolmino_control_p5_5b
  qwen25_7b_dolmino_logic_p5_5b
  qwen25_7b_dolmino_nl_exact_p5_5b
)
IDX="${SLURM_ARRAY_TASK_ID}"
CONDITION="${CONDITIONS[$IDX]}"
RUN_DIR_TAG="${RUN_DIR_TAGS[$IDX]}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-9537}"
LR="${LR:-5e-6}"
RUN_NAME="qwen25_7b_dolmino_${CONDITION}_5b_dolci_100k_lr5em6"
RUN_ROOT="${HPCVAULT}/synthetic-RLVL/nanotron_dolmino_5b/${RUN_DIR_TAG}"
NANOTRON_CKPT="${RUN_ROOT}/checkpoints/${CHECKPOINT_STEP}"
OUT_ROOT="${POST_SFT_OUT_ROOT:-${WORK}/synthetic-RLVL/post_sft_dolci_20260804}"
OUT_DIR="${OUT_ROOT}/${RUN_NAME}"
AUDIT="${HPCVAULT}/synthetic-RLVL/analysis/dolci_control_sft_20260803/subset_and_tokenization_audit.json"
PREPARED_DATASET="${HPCVAULT}/synthetic-RLVL/datasets/dolci_no_tools_single_turn_100k_seed3407_20260803"

test -s "${AUDIT}"
test -s "${PREPARED_DATASET}/dataset_dict.json"
mkdir -p "${OUT_ROOT}"
if [[ -s "${OUT_DIR}/final/config.json" ]]; then
  echo "Complete post-SFT checkpoint exists: ${OUT_DIR}/final; skipping"
  exit 0
fi

python scripts/nanotron/verify_training_checkpoint.py \
  --checkpoint "${NANOTRON_CKPT}" \
  --expected-step "${CHECKPOINT_STEP}" \
  --sequence-length 4096 \
  --global-batch-samples 128 \
  --tp 4 --dp 2 --model-files 625 --optimizer-shards 4 \
  --lr-scheduler-shards 4 --rng-shards 8 \
  --output "${OUT_DIR}_base_checkpoint_audit.json"

LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
HF_WORK="$(mktemp -d "${LOCAL_BASE%/}/dolmino_3sft_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_XXXXXX")"
HF_CHECKPOINT="${HF_WORK}/hf"
cleanup() {
  case "${HF_WORK}" in
    "${LOCAL_BASE%/}"/dolmino_3sft_"${SLURM_JOB_ID}"_"${SLURM_ARRAY_TASK_ID}"_*) rm -rf -- "${HF_WORK}" ;;
    *) echo "Refusing unexpected temporary cleanup target: ${HF_WORK}" >&2 ;;
  esac
}
trap cleanup EXIT

source "${WORK}/nanotron/bin/activate"
export NANOTRON_ROOT="${NANOTRON_ROOT:-$(cd ../nanotron && pwd)}"
torchrun --standalone --nproc_per_node=1 scripts/nanotron/convert_qwen2_nanotron_to_hf.py \
  --checkpoint-path "${NANOTRON_CKPT}" \
  --save-path "${HF_CHECKPOINT}" \
  --tokenizer-name Qwen/Qwen2.5-7B

VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true
export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="qwen25_dolmino_threeway_post_sft_5em6"
export WANDB_RUN_GROUP="${WANDB_GROUP}"
export WANDB_NAME="${RUN_NAME}"
export HF_HUB_ENABLE_HF_TRANSFER=0
# Full-parameter FSDP checkpoint serialization can hold the Python GIL for
# longer than PyTorch's eight-minute NCCL watchdog default on this filesystem.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"

"${VENV_PATH}/bin/python" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/train_instruction_sft.py \
  --model "${HF_CHECKPOINT}" \
  --dataset allenai/Dolci-Instruct-SFT-No-Tools \
  --dataset-revision 9156a5a542b2503100d4f1fabbf50be5eb25d977 \
  --prepared-dataset "${PREPARED_DATASET}" \
  --train-split train \
  --eval-split train \
  --run-name "${RUN_NAME}" \
  --output-dir "${OUT_DIR}" \
  --seed 3407 \
  --train-samples 100000 \
  --eval-samples 2048 \
  --max-length 4096 \
  --max-steps -1 \
  --num-train-epochs 1 \
  --lr "${LR}" \
  --warmup-steps 0 \
  --warmup-ratio 0.03 \
  --per-device-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --grad-accum 32 \
  --eval-steps 250 \
  --save-steps 250 \
  --save-total-limit 1 \
  --format-mode chat \
  --single-turn-only \
  --no-wrap-answer-tags \
  --full-parameter \
  --fsdp "full_shard auto_wrap" \
  --fsdp-transformer-layer-cls-to-wrap Qwen2DecoderLayer \
  --gradient-checkpointing \
  --resume-from-checkpoint auto

test -s "${OUT_DIR}/final/config.json"
find "${OUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf -- {} +
cleanup
trap - EXIT
