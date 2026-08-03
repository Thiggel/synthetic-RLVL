#!/usr/bin/env bash
#SBATCH --job-name=q25_dolm_postsft
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

LRS=(2e-6 5e-6)
LR="${LRS[$SLURM_ARRAY_TASK_ID]}"
LR_TAG="${LR//-/m}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-9537}"
RUN_ROOT="${HPCVAULT}/synthetic-RLVL/nanotron_dolmino_5b/qwen25_7b_dolmino_control_p5_5b"
NANOTRON_CKPT="${RUN_ROOT}/checkpoints/${CHECKPOINT_STEP}"
RUN_SUFFIX="${RUN_SUFFIX:-100k}"
RUN_NAME="qwen25_7b_dolmino_control_5b_dolci_${RUN_SUFFIX}_lr${LR_TAG}"
OUT_ROOT="${POST_SFT_OUT_ROOT:-${HPCVAULT}/synthetic-RLVL/post_sft_dolci_20260803}"
OUT_DIR="${OUT_ROOT}/${RUN_NAME}"
AUDIT="${HPCVAULT}/synthetic-RLVL/analysis/dolci_control_sft_20260803/subset_and_tokenization_audit.json"
PREPARED_DATASET="${HPCVAULT}/synthetic-RLVL/datasets/dolci_no_tools_single_turn_100k_seed3407_20260803"

test -s "${AUDIT}"
test -s "${PREPARED_DATASET}/dataset_dict.json"
mkdir -p "${OUT_ROOT}"
python scripts/nanotron/verify_training_checkpoint.py \
  --checkpoint "${NANOTRON_CKPT}" \
  --expected-step "${CHECKPOINT_STEP}" \
  --sequence-length 4096 \
  --global-batch-samples 128 \
  --tp 4 --dp 2 --model-files 625 --optimizer-shards 4 \
  --lr-scheduler-shards 4 --rng-shards 8 \
  --output "${OUT_DIR}_base_checkpoint_audit.json"

LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
HF_WORK="$(mktemp -d "${LOCAL_BASE%/}/dolmino_postsft_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_XXXXXX")"
HF_CHECKPOINT="${HF_WORK}/hf"
cleanup() {
  case "${HF_WORK}" in
    "${LOCAL_BASE%/}"/dolmino_postsft_"${SLURM_JOB_ID}"_"${SLURM_ARRAY_TASK_ID}"_*) rm -rf -- "${HF_WORK}" ;;
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
export WANDB_GROUP="qwen25_dolmino_control_post_sft_lr"
export WANDB_RUN_GROUP="${WANDB_GROUP}"
export WANDB_NAME="${RUN_NAME}"
export HF_HUB_ENABLE_HF_TRANSFER=0

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
  --train-samples "${TRAIN_SAMPLES:-100000}" \
  --eval-samples "${EVAL_SAMPLES:-2048}" \
  --max-length 4096 \
  --max-steps "${MAX_STEPS:--1}" \
  --num-train-epochs 1 \
  --lr "${LR}" \
  --warmup-steps 0 \
  --warmup-ratio 0.03 \
  --per-device-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --grad-accum 32 \
  --eval-steps "${EVAL_STEPS:-250}" \
  --save-steps "${SAVE_STEPS:-10000}" \
  --save-total-limit 1 \
  --format-mode chat \
  --single-turn-only \
  --no-wrap-answer-tags \
  --full-parameter \
  --fsdp "full_shard auto_wrap" \
  --fsdp-transformer-layer-cls-to-wrap Qwen2DecoderLayer \
  --gradient-checkpointing

test -s "${OUT_DIR}/final/config.json"
if [[ "${CLEANUP_POST_SFT_OUTPUT:-0}" == "1" ]]; then
  OUT_DIR="${OUT_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUT_DIR"])
files = [path for path in root.rglob("*") if path.is_file()]
payload = {
    "accepted": bool((root / "final" / "config.json").is_file()),
    "file_count": len(files),
    "size_bytes": sum(path.stat().st_size for path in files),
    "output_removed_after_audit": True,
}
Path(str(root) + "_smoke_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
  test -s "${OUT_DIR}_smoke_audit.json"
  rm -rf -- "${OUT_DIR}"
fi
cleanup
trap - EXIT
