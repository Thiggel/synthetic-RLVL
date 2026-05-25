#!/usr/bin/env bash
#SBATCH --job-name=ood_lmeval_1gpu
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-89%4
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

RUN_NAMES=()

for template in logic nl_exact; do
  for train_max in 5 10 15 20 25; do
    for seed in 3407 3408 3409; do
      RUN_NAMES+=("sft_hfsa_depth_scaling_${template}_train1to${train_max}_10k_seed${seed}")
    done
  done
done

for kind in attribute_constraints maze_navigation; do
  for template in logic nl_exact; do
    RUN_NAMES+=("sft_paired_followup_${kind}_${template}_train1to10_10k_seed3407")
  done
done

for template in logic nl_exact; do
  RUN_NAMES+=("sft_paired_followup_attribute_constraints_hard_${template}_train1to10_10k_seed3407")
done

for template in logic nl_exact; do
  for train_max in 10 20 25; do
    for seed in 3407 3408 3409; do
      RUN_NAMES+=("sft_hfsa_modelablate_qwen2p5_7b_${template}_train1to${train_max}_10k_seed${seed}")
    done
  done
done

for model_tag in qwen2p5_1p5b gemma3_4b_pt; do
  for template in logic nl_exact; do
    for train_max in 10 20 25; do
      for seed in 3407 3408 3409; do
        RUN_NAMES+=("sft_hfsa_modelablate_${model_tag}_${template}_train1to${train_max}_10k_seed${seed}")
      done
    done
  done
done

IDX="${SLURM_ARRAY_TASK_ID}"
if (( IDX >= ${#RUN_NAMES[@]} )); then
  echo "array index ${IDX} outside run map length ${#RUN_NAMES[@]}; skipping"
  exit 0
fi

RUN_NAME="${RUN_NAMES[$IDX]}"
SFT_FINAL="${WORK}/synthetic-RLVL/runs/${RUN_NAME}/final"
MERGED_ROOT="${WORK}/synthetic-RLVL/tmp/ood_lmeval_1gpu_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
MERGED="${MERGED_ROOT}/merged_${RUN_NAME}"
OUT_DIR="${WORK}/synthetic-RLVL/lm_eval_results/ood_large_2026-05-25/${RUN_NAME}"
GROUP="ood_lmeval_large_1gpu"

cleanup_merged() { rm -rf "${MERGED_ROOT}"; }
trap cleanup_merged EXIT

if [[ ! -f "${SFT_FINAL}/adapter_config.json" ]]; then
  echo "missing SFT LoRA checkpoint: ${SFT_FINAL}; skipping"
  if [[ "${MISSING_IS_ERROR:-0}" == "1" ]]; then exit 2; fi
  exit 0
fi
if [[ -d "${OUT_DIR}" && "${FORCE_LM_EVAL:-0}" != "1" ]]; then
  echo "lm-eval output dir exists: ${OUT_DIR}; skipping"
  exit 0
fi

sleep $((SLURM_ARRAY_TASK_ID * ${LMEVAL_STAGGER_SECONDS:-45} + RANDOM % ${LMEVAL_JITTER_SECONDS:-60}))
mkdir -p "${MERGED_ROOT}" "${OUT_DIR}"
python scripts/merge_lora_checkpoint.py --checkpoint "${SFT_FINAL}" --output "${MERGED}.tmp" --dtype bf16 --overwrite
mv "${MERGED}.tmp" "${MERGED}"

LIMIT_ARGS=()
if [[ -n "${LM_EVAL_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LM_EVAL_LIMIT}")
fi

export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"

python scripts/evaluate_lm_eval.py \
  --checkpoint "${MERGED}" \
  --model vllm \
  --suite synthrlvl_ood \
  --include-path lm_eval_tasks/synthrlvl_ood \
  --output-path "${OUT_DIR}" \
  --batch-size "${LM_EVAL_BATCH_SIZE:-auto}" \
  "${LIMIT_ARGS[@]}" \
  --dtype "${LM_EVAL_DTYPE:-bfloat16}" \
  --gpu-memory-utilization "${LM_EVAL_GPU_MEMORY_UTILIZATION:-0.80}" \
  --model-arg "max_model_len=${LM_EVAL_MAX_MODEL_LEN:-32768}" \
  --confirm-run-unsafe-code \
  --log-samples \
  --wandb-project "${WANDB_PROJECT:-synthetic-rlvl}" \
  --wandb-group "${GROUP}" \
  --wandb-run-name "${RUN_NAME}_ood_lmeval"

cleanup_merged
trap - EXIT
