#!/usr/bin/env bash
#SBATCH --job-name=large_lora_sft
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --exclude=a0531,a0532,a0533,a0934
#SBATCH --array=0-5%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# The two largest models of the family study, trained with LoRA on alex.
#
# Full-parameter training of Qwen2.5-32B and Qwen3-30B-A3B needs two H200
# nodes on Lise, which a maintenance reservation withholds. This job trains
# the same three conditions with a low-rank adapter on four A100 80GB instead:
# the base is sharded across the ranks with FSDP, activations are checkpointed
# and everything runs in bfloat16. The adapter is merged into the base after
# training so the evaluation jobs read an ordinary checkpoint. The 32B control
# already trains on Lise at full parameter, so its LoRA counterpart is
# included here too, since a treated LoRA arm is only comparable to a LoRA
# control.
#
#   0  qwen25_32b   control        3  qwen3_30b_moe logic_p05
#   1  qwen25_32b   logic_p05      4  qwen3_30b_moe nl_exact_p10
#   2  qwen25_32b   nl_exact_p10   5  qwen3_30b_moe control
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"
source ./scripts/env.sh
VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"; export PATH="${VENV_PATH}/bin:${PATH}"
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline OMP_NUM_THREADS=8
hash -r 2>/dev/null || true

ARMS=(
  "Qwen/Qwen2.5-32B Qwen2DecoderLayer qwen25_32b control"
  "Qwen/Qwen2.5-32B Qwen2DecoderLayer qwen25_32b logic_p05"
  "Qwen/Qwen2.5-32B Qwen2DecoderLayer qwen25_32b nl_exact_p10"
  "Qwen/Qwen3-30B-A3B-Base Qwen3MoeDecoderLayer qwen3_30b_moe logic_p05"
  "Qwen/Qwen3-30B-A3B-Base Qwen3MoeDecoderLayer qwen3_30b_moe nl_exact_p10"
  "Qwen/Qwen3-30B-A3B-Base Qwen3MoeDecoderLayer qwen3_30b_moe control"
)
read -r BASE WRAP TAG COND <<<"${ARMS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEED:-3407}"
RUN_NAME="${TAG}_${COND}_lora_seed${SEED}"
MIX="${HPCVAULT}/synthetic-RLVL/datasets/dose_sweep_20260912/${COND}"
STUDY_DIR="${WORK}/synthetic-RLVL/post_sft_large_lora_20260917"
OUT_DIR="${STUDY_DIR}/${RUN_NAME}"

test -d "${MIX}" || { echo "missing mixture ${MIX}" >&2; exit 2; }
[[ -s "${OUT_DIR}/final/config.json" ]] && { echo "already done ${RUN_NAME}"; exit 0; }

# The compute nodes reach the hub, so the weights are fetched here on first
# use. A lock keeps two array tasks from downloading the same model at once.
mkdir -p "${HF_HOME}/locks"
# The hf CLI entry point carries the shebang of the venv's old location, so
# the download goes through the interpreter on PATH instead.
BASE_PATH=$( ( flock 9; python -c "from huggingface_hub import snapshot_download; print(snapshot_download('${BASE}'))" ) 9>"${HF_HOME}/locks/${BASE//\//--}.lock" | tail -1 )
test -s "${BASE_PATH}/config.json" || { echo "incomplete snapshot ${BASE_PATH}" >&2; exit 2; }
echo "run=${RUN_NAME} base=${BASE_PATH} wrap=${WRAP} mixture=${MIX}"

SMOKE_ARGS=()
if [[ -n "${SMOKE_STEPS:-}" ]]; then
  SMOKE_ARGS=(--max-steps "${SMOKE_STEPS}" --train-samples 256 --eval-samples 8)
  OUT_DIR="${WORK}/synthetic-RLVL/post_sft_large_lora_smoke/${RUN_NAME}"
  rm -rf "${OUT_DIR}"
fi

# Same data, schedule and effective batch (128) as the full-parameter runs.
# The learning rate is the usual one for a rank-16 adapter, since the
# full-parameter rate of 5e-6 leaves an adapter essentially untrained.
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train_instruction_sft.py \
  --model "${BASE_PATH}" --prepared-dataset "${MIX}" \
  --train-split train --eval-split eval \
  --run-name "${RUN_NAME}" --output-dir "${OUT_DIR}" --seed "${SEED}" \
  --train-samples 100000 --eval-samples 2048 --max-steps -1 \
  --max-length 8192 --max-truncated-frac 0.005 \
  --num-train-epochs 1 --lr "${LR:-1e-4}" \
  "${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"}" \
  --warmup-steps 0 --warmup-ratio 0.03 \
  --per-device-batch-size "${PDBS:-1}" --per-device-eval-batch-size 1 --grad-accum "${GA:-32}" \
  --eval-steps 2000 --save-steps 500 --save-total-limit 1 \
  --format-mode chat --single-turn-only --no-wrap-answer-tags \
  --lora-r 16 --lora-alpha 32 \
  --fsdp "full_shard auto_wrap" \
  --fsdp-transformer-layer-cls-to-wrap "${WRAP}" \
  --gradient-checkpointing --bf16 --report-to none \
  --resume-from-checkpoint auto

test -s "${OUT_DIR}/final/adapter_config.json"
find "${OUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf -- {} +
mv "${OUT_DIR}/final" "${OUT_DIR}/adapter"
python scripts/merge_lora.py --base "${BASE_PATH}" --adapter "${OUT_DIR}/adapter" --out "${OUT_DIR}/final"
test -s "${OUT_DIR}/final/config.json"
du -sh "${OUT_DIR}/final"; echo "done ${RUN_NAME}"
