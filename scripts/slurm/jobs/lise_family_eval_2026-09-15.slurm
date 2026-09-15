#!/usr/bin/env bash
#SBATCH --job-name=fam_eval
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Evaluate one model of the family and scale study. RUN_NAME selects it and
# SUITE selects what to measure. The task files carry absolute paths from the
# cluster they were written on, so the job copies them into a scratch directory
# and rewrites the dataset root before lm-eval reads them.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

R=/scratch/usr/bemflait/synthetic-RLVL
export HF_HOME="${R}/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
source "${R}/.venv/bin/activate"

RUN_NAME="${RUN_NAME:?}"; SUITE="${SUITE:?}"
CKPT="${R}/post_sft_family_20260915/${RUN_NAME}/final"
OUT="${R}/lm_eval_results/family_20260915/${SUITE}/${RUN_NAME}"
test -s "${CKPT}/config.json" || { echo "missing ${CKPT}" >&2; exit 2; }
[ -e "${OUT}/.complete" ] && { echo "already done"; exit 0; }
mkdir -p "$(dirname "${OUT}")"
CLAIM="${OUT}.claim"
mkdir "${CLAIM}" 2>/dev/null || { echo "claimed by another job"; exit 0; }
trap 'rmdir "${CLAIM}" 2>/dev/null || true' EXIT

case "${SUITE}" in
  deduction) TASKS="synthrlvl_deduction_pw_d0,synthrlvl_deduction_pw_d1,synthrlvl_deduction_pw_d2,synthrlvl_deduction_pw_d3,synthrlvl_deduction_pw_d5"; MML=8192 ;;
  folio) TASKS="synthrlvl_folio"; MML=8192 ;;
  native) TASKS="synthrlvl_deduction_bp_native_d5,synthrlvl_deduction_bp_native_d15,synthrlvl_deduction_bp_native_d25"; MML=16384 ;;
  standard) TASKS="gsm8k,mmlu,arc_challenge,agieval_logiqa_en,hellaswag,winogrande,piqa"; MML=8192 ;;
  *) echo "unknown suite"; exit 2 ;;
esac

TASK_DIR="${TMPDIR:-/tmp}/synthrlvl_ood_${SLURM_JOB_ID}"
rm -rf "${TASK_DIR}"; cp -r lm_eval_tasks/synthrlvl_ood "${TASK_DIR}"
sed -i "s#/vol/tmp2/laitenbf/rlvl_data/datasets#${R}/datasets#g" "${TASK_DIR}"/*.yaml
sed -i "s#/home/vault/c107fa/c107fa12/synthetic-RLVL/datasets#${R}/datasets#g" "${TASK_DIR}"/*.yaml
python scripts/validate_lm_eval_tasks.py --include-path "${TASK_DIR}" --tasks "${TASKS}"
python scripts/evaluate_lm_eval.py --checkpoint "${CKPT}" --model vllm --tasks "${TASKS}" \
  --include-path "${TASK_DIR}" --output-path "${OUT}" --batch-size auto --dtype bfloat16 \
  --gpu-memory-utilization 0.85 --model-arg "max_model_len=${MML}" \
  --trust-remote-code --confirm-run-unsafe-code --apply-chat-template --log-samples
touch "${OUT}/.complete"; echo "done ${RUN_NAME} ${SUITE}"
