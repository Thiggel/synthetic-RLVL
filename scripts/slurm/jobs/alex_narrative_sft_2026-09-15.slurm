#!/usr/bin/env bash
#SBATCH --job-name=q25_narr_sft
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --exclude=a0531,a0532,a0533,a0934
#SBATCH --array=0-6%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Three studies the argument rests on, run as one array.
#
#   0,1  a second instruction corpus. The claim that instruction tuning
#        installs the answer prior rests on one corpus, so it is repeated on
#        Tulu 3, which has a different source and task distribution.
#   2-4  shares below one percent, which locates the threshold for installing
#        checkable derivations more precisely than the existing grid.
#   5,6  no instruction data at all, the endpoint of the replacement design,
#        which asks whether the instruction corpus is needed for the transfer.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"
source ./scripts/env.sh
VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"; export PATH="${VENV_PATH}/bin:${PATH}"
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
hash -r 2>/dev/null || true

D="${HPCVAULT}/synthetic-RLVL/datasets"
NAMES=(tulu3_control tulu3_logic_p05 dose_logic_p001 dose_logic_p002 dose_logic_p005 synthonly_logic synthonly_nl)
MIXES=(
  "${D}/tulu3_single_turn_100k_20260915"
  "${D}/tulu3_mix_20260915/dolci_bp_logic_band25_p5"
  "${D}/subdose_20260915/p001/dolci_bp_logic_band25_p0"
  "${D}/subdose_20260915/p002/dolci_bp_logic_band25_p0"
  "${D}/subdose_20260915/p005/dolci_bp_logic_band25_p0"
  "${D}/synthonly_20260915/logic"
  "${D}/synthonly_20260915/nl_exact"
)
IDX="${SLURM_ARRAY_TASK_ID}"
SEED="${SEED:-3407}"
RUN_NAME="qwen25_7b_${NAMES[$IDX]}_seed${SEED}"
MIX="${MIXES[$IDX]}"
OUT_DIR="${WORK}/synthetic-RLVL/post_sft_narrative_20260915/${RUN_NAME}"

test -d "${MIX}" || { echo "missing mixture ${MIX}" >&2; exit 2; }
[[ -s "${OUT_DIR}/final/config.json" ]] && { echo "already done ${RUN_NAME}"; exit 0; }
echo "run=${RUN_NAME} mixture=${MIX}"

python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train_instruction_sft.py \
  --model "${BASE_MODEL:-Qwen/Qwen2.5-7B}" --prepared-dataset "${MIX}" \
  --train-split train --eval-split eval \
  --run-name "${RUN_NAME}" --output-dir "${OUT_DIR}" --seed "${SEED}" \
  --train-samples 100000 --eval-samples 2048 \
  --max-length 8192 --max-truncated-frac 0.005 \
  --max-steps -1 --num-train-epochs 1 --lr 5e-6 \
  --warmup-steps 0 --warmup-ratio 0.03 \
  --per-device-batch-size 2 --per-device-eval-batch-size 1 --grad-accum 16 \
  --eval-steps 2000 --save-steps 500 --save-total-limit 1 \
  --format-mode chat --single-turn-only --no-wrap-answer-tags \
  --full-parameter --fsdp "full_shard auto_wrap" \
  --fsdp-transformer-layer-cls-to-wrap Qwen2DecoderLayer \
  --gradient-checkpointing --bf16 --report-to none \
  --resume-from-checkpoint auto

test -s "${OUT_DIR}/final/config.json"
find "${OUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf -- {} +
du -sh "${OUT_DIR}/final"; echo "done ${RUN_NAME}"
