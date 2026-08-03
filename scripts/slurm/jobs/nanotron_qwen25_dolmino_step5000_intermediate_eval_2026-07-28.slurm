#!/usr/bin/env bash
#SBATCH --job-name=q25_dolm5k_eval
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-2%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

CHECKPOINT_STEP="${CHECKPOINT_STEP:-5000}"
CONDITIONS=(control logic nl_exact)
MIX_PCTS=(0 5 5)
RUN_DIR_TAGS=(
  qwen25_7b_dolmino_control_p5_5b
  qwen25_7b_dolmino_logic_p5_5b
  qwen25_7b_dolmino_nl_exact_p5_5b
)
IDX="${SLURM_ARRAY_TASK_ID}"
CONDITION="${CONDITIONS[$IDX]}"
MIX_PCT="${MIX_PCTS[$IDX]}"
RUN_DIR_TAG="${RUN_DIR_TAGS[$IDX]}"
EVAL_TAG="qwen25_7b_dolmino_${CONDITION}_p${MIX_PCT}_5b"
RUN_ROOT="${HPCVAULT}/synthetic-RLVL/nanotron_dolmino_5b/${RUN_DIR_TAG}"
NANOTRON_CKPT="${RUN_ROOT}/checkpoints/${CHECKPOINT_STEP}"
OUT_ROOT="${OUT_ROOT_OVERRIDE:-${HPCVAULT}/synthetic-RLVL/lm_eval_results/qwen25_dolmino_step5000_intermediate_20260728}"
OUT_PREFIX="${OUT_ROOT}/${EVAL_TAG}_step${CHECKPOINT_STEP}_direct"
MULTIHOP_OUT="${OUT_PREFIX}_multihop"
STANDARD_OUT="${OUT_PREFIX}_standard"
mkdir -p "${OUT_ROOT}"

STANDARD_TASKS="${STANDARD_TASKS:-gsm8k,arc_challenge,hellaswag,winogrande,piqa,agieval_logiqa_en,hendrycks_math500,bbh,mmlu,mmlu_pro}"
MULTIHOP_TASKS="${MULTIHOP_TASKS:-synthrlvl_longbench_hotpotqa_tagged,synthrlvl_longbench_2wikimqa_tagged,synthrlvl_longbench_musique_tagged,synthrlvl_longbench_hotpotqa_standard,synthrlvl_longbench_2wikimqa_standard,synthrlvl_longbench_musique_standard}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-100}"

export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export NANOTRON_ROOT="${NANOTRON_ROOT:-$(cd ../nanotron && pwd)}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export WANDB_MODE=disabled
export WANDB_GROUP="qwen25_dolmino_step5000_intermediate"
export WANDB_RUN_GROUP="${WANDB_GROUP}"

if [[ ! -s "${NANOTRON_CKPT}/model_config.json" \
      || ! -s "${NANOTRON_CKPT}/checkpoint_metadata.json" \
      || ! -d "${NANOTRON_CKPT}/model" ]] \
    || find "${NANOTRON_CKPT}/model" -type f -size 0 -print -quit | grep -q .; then
  echo "Missing or incomplete audited checkpoint: ${NANOTRON_CKPT}" >&2
  exit 2
fi

python scripts/nanotron/verify_training_checkpoint.py \
  --checkpoint "${NANOTRON_CKPT}" \
  --expected-step "${CHECKPOINT_STEP}" \
  --sequence-length 4096 \
  --global-batch-samples 128 \
  --tp 4 \
  --dp 2 \
  --model-files 625 \
  --optimizer-shards 4 \
  --lr-scheduler-shards 4 \
  --rng-shards 8 \
  --output "${OUT_PREFIX}_nanotron_audit.json"

LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
HF_WORK="$(mktemp -d "${LOCAL_BASE%/}/dolmino_step${CHECKPOINT_STEP}_${CONDITION}_${SLURM_JOB_ID}_XXXXXX")"
HF_CHECKPOINT="${HF_WORK}/hf"
cleanup() {
  case "${HF_WORK}" in
    "${LOCAL_BASE%/}"/dolmino_step"${CHECKPOINT_STEP}"_"${CONDITION}"_"${SLURM_JOB_ID}"_*) rm -rf -- "${HF_WORK}" ;;
    *) echo "Refusing unexpected temporary cleanup target: ${HF_WORK}" >&2 ;;
  esac
}
trap cleanup EXIT

source "${WORK}/nanotron/bin/activate"
torchrun --standalone --nproc_per_node=1 scripts/nanotron/convert_qwen2_nanotron_to_hf.py \
  --checkpoint-path "${NANOTRON_CKPT}" \
  --save-path "${HF_CHECKPOINT}" \
  --tokenizer-name Qwen/Qwen2.5-7B

VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

python scripts/nanotron/verify_qwen2_hf_checkpoint.py \
  --checkpoint "${HF_CHECKPOINT}" \
  --output "${OUT_PREFIX}_local_hf_audit.json"
python scripts/validate_lm_eval_tasks.py \
  --include-path lm_eval_tasks/synthrlvl_ood \
  --tasks "${MULTIHOP_TASKS}"
python scripts/validate_lm_eval_tasks.py --tasks "${STANDARD_TASKS}"

# Run context-provided QA first so the requested HotpotQA/2Wiki/MuSiQue
# comparison survives even if the broader grouped suite consumes the time limit.
python scripts/evaluate_lm_eval.py \
  --checkpoint "${HF_CHECKPOINT}" \
  --model vllm \
  --tasks "${MULTIHOP_TASKS}" \
  --include-path lm_eval_tasks/synthrlvl_ood \
  --output-path "${MULTIHOP_OUT}" \
  --batch-size "${LM_EVAL_BATCH_SIZE:-auto}" \
  --limit "${LM_EVAL_LIMIT}" \
  --dtype bfloat16 \
  --gpu-memory-utilization "${LM_EVAL_GPU_MEMORY_UTILIZATION:-0.80}" \
  --model-arg "max_model_len=${LM_EVAL_MAX_MODEL_LEN:-32768}" \
  --trust-remote-code \
  --confirm-run-unsafe-code \
  --log-samples

python scripts/analysis/audit_nanotron_multihop_eval.py \
  --run-dir "${MULTIHOP_OUT}" \
  --mode direct \
  --tasks "${MULTIHOP_TASKS}" \
  --allow-limit

python scripts/evaluate_lm_eval.py \
  --checkpoint "${HF_CHECKPOINT}" \
  --model vllm \
  --tasks "${STANDARD_TASKS}" \
  --output-path "${STANDARD_OUT}" \
  --batch-size "${LM_EVAL_BATCH_SIZE:-auto}" \
  --limit "${LM_EVAL_LIMIT}" \
  --dtype bfloat16 \
  --gpu-memory-utilization "${LM_EVAL_GPU_MEMORY_UTILIZATION:-0.80}" \
  --model-arg "max_model_len=${STANDARD_MAX_MODEL_LEN:-8192}" \
  --trust-remote-code \
  --confirm-run-unsafe-code \
  --log-samples

python scripts/analysis/audit_nanotron_downstream_eval.py \
  --run-dir "${STANDARD_OUT}" \
  --mode direct \
  --tasks "${STANDARD_TASKS}" \
  --allow-limit

cleanup
trap - EXIT
