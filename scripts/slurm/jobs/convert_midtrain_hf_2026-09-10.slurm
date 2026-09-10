#!/usr/bin/env bash
#SBATCH --job-name=q25_convert_hf
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=a0531,a0532,a0533,a0934,a0534
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Convert the midtrained nanotron checkpoints to Hugging Face format, once,
# so they can be (a) evaluated directly as base models, before any instruction
# tuning, and (b) shipped to gruenau, whose H100 and A100 nodes are idle while
# alex sits at its account GPU cap.
#
# The post-SFT wrapper already does this conversion internally into a scratch
# directory that it deletes afterwards, so nothing persists. Here the output is
# kept.
#
# Array index: 0 control, 1 logic_band25, 2 nl_exact_band25.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"
source ./scripts/env.sh

CONDITIONS=(control logic_band25 nl_exact_band25)
RUN_DIR_TAGS=(
  qwen25_7b_longwin_control_p0_2p5b_s8192
  qwen25_7b_longwin_logic_band25_p10_2p5b_s8192
  qwen25_7b_longwin_nl_exact_band25_p10_2p5b_s8192
)
IDX="${SLURM_ARRAY_TASK_ID}"
CONDITION="${CONDITIONS[$IDX]}"
RUN_DIR_TAG="${RUN_DIR_TAGS[$IDX]}"
STEP="${CHECKPOINT_STEP:-2385}"

NANOTRON_CKPT="${WORK}/synthetic-RLVL/nanotron_longwin_midtrain/${RUN_DIR_TAG}/checkpoints/${STEP}"
OUT="${WORK}/synthetic-RLVL/hf_midtrain_longwin_20260910/qwen25_7b_longwin_${CONDITION}_2p5b_hf"

test -d "${NANOTRON_CKPT}"
if [[ -s "${OUT}/config.json" ]]; then
  echo "already converted: ${OUT}; skipping"
  exit 0
fi

source "${WORK}/nanotron/bin/activate"
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

mkdir -p "$(dirname "${OUT}")"
python scripts/nanotron/convert_qwen2_nanotron_to_hf.py \
  --checkpoint-path "${NANOTRON_CKPT}" \
  --save-path "${OUT}.tmp" \
  --tokenizer-name "Qwen/Qwen2.5-7B"

# fail closed: a half-written conversion must not be published under the
# final name, because everything downstream keys off config.json existing
test -s "${OUT}.tmp/config.json"
python scripts/nanotron/verify_qwen2_hf_checkpoint.py --checkpoint "${OUT}.tmp" \
  --output "${OUT}.tmp/conversion_audit.json" || {
  echo "verification failed for ${OUT}.tmp" >&2
  exit 2
}
mv "${OUT}.tmp" "${OUT}"
du -sh "${OUT}"
echo "converted ${CONDITION} -> ${OUT}"
