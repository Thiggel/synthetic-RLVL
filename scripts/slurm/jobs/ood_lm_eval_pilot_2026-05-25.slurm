#!/usr/bin/env bash
#SBATCH --job-name=ood_lmeval_pilot
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=04:00:00
#SBATCH --array=0-1%1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs tmp
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then VENV_PATH="${WORK}/.venv"; fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

RUN_NAMES=(
  sft_hfsa_modelablate_qwen2p5_1p5b_logic_train1to10_10k_seed3407
  sft_hfsa_modelablate_qwen2p5_1p5b_nl_exact_train1to10_10k_seed3407
)

RUN_NAME="${RUN_NAMES[${SLURM_ARRAY_TASK_ID}]}"
SFT_FINAL="${WORK}/synthetic-RLVL/runs/${RUN_NAME}/final"
MERGED_ROOT="${WORK}/synthetic-RLVL/tmp/ood_lmeval_pilot_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
MERGED="${MERGED_ROOT}/merged_${RUN_NAME}"
OUT_DIR="${WORK}/synthetic-RLVL/lm_eval_results/ood_pilot_2026-05-25/${RUN_NAME}"
GROUP="ood_lmeval_pilot/qwen2p5_1p5b"

cleanup_merged() { rm -rf "${MERGED_ROOT}"; }
trap cleanup_merged EXIT

if [[ ! -f "${SFT_FINAL}/adapter_config.json" ]]; then
  echo "missing SFT LoRA checkpoint: ${SFT_FINAL}" >&2
  exit 2
fi
if [[ -d "${OUT_DIR}" && "${FORCE_LM_EVAL:-0}" != "1" ]]; then
  echo "lm-eval output dir exists: ${OUT_DIR}; skipping"
  exit 0
fi

sleep $((SLURM_ARRAY_TASK_ID * ${LMEVAL_STAGGER_SECONDS:-120} + RANDOM % ${LMEVAL_JITTER_SECONDS:-60}))
mkdir -p "${MERGED_ROOT}" "${OUT_DIR}"
python scripts/merge_lora_checkpoint.py --checkpoint "${SFT_FINAL}" --output "${MERGED}.tmp" --dtype bf16 --overwrite
mv "${MERGED}.tmp" "${MERGED}"

python scripts/evaluate_lm_eval.py \
  --checkpoint "${MERGED}" \
  --model vllm \
  --suite synthrlvl_ood \
  --include-path lm_eval_tasks/synthrlvl_ood \
  --output-path "${OUT_DIR}" \
  --batch-size "${LM_EVAL_BATCH_SIZE:-auto}" \
  --limit "${LM_EVAL_LIMIT:-3}" \
  --dtype "${LM_EVAL_DTYPE:-bfloat16}" \
  --gpu-memory-utilization "${LM_EVAL_GPU_MEMORY_UTILIZATION:-0.80}" \
  --model-arg "max_model_len=${LM_EVAL_MAX_MODEL_LEN:-32768}" \
  --confirm-run-unsafe-code \
  --log-samples \
  --wandb-project "${WANDB_PROJECT:-synthetic-rlvl}" \
  --wandb-group "${GROUP}" \
  --wandb-run-name "${RUN_NAME}_ood_lmeval_pilot"

cleanup_merged
trap - EXIT
