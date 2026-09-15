#!/usr/bin/env bash
#SBATCH --job-name=q25_dose_eval
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --exclude=a0531,a0532,a0533,a0934
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Evaluate one dose-sweep model where it was trained. RUN_NAME selects it and
# SUITE selects what to measure; the four suites that carry the paper's claims
# come first so a seed can be read before the sweep finishes.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"
source ./scripts/env.sh
VENV_PATH="${VENV_PATH:-${HPCVAULT}/.venv_rlvl_posttrain}"
export VIRTUAL_ENV="${VENV_PATH}"; export PATH="${VENV_PATH}/bin:${PATH}"
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 TOKENIZERS_PARALLELISM=false
hash -r 2>/dev/null || true

RUN_NAME="${RUN_NAME:?}"; SUITE="${SUITE:?}"
CKPT="${WORK}/synthetic-RLVL/post_sft_dose_20260912/${RUN_NAME}/final"
OUT="${HPCVAULT}/synthetic-RLVL/lm_eval_results/dose_sweep_20260912/${SUITE}/${RUN_NAME}"
test -s "${CKPT}/config.json" || { echo "missing ${CKPT}" >&2; exit 2; }
[ -e "${OUT}/.complete" ] && { echo "already done"; exit 0; }
mkdir -p "$(dirname "${OUT}")"
CLAIM="${OUT}.claim"
mkdir "${CLAIM}" 2>/dev/null || { echo "claimed by another job"; exit 0; }
rm -f "${OUT}.queued"
trap 'rmdir "${CLAIM}" 2>/dev/null || true' EXIT

case "${SUITE}" in
  deduction) TASKS="synthrlvl_deduction_pw_d0,synthrlvl_deduction_pw_d1,synthrlvl_deduction_pw_d2,synthrlvl_deduction_pw_d3,synthrlvl_deduction_pw_d5,synthrlvl_deduction_bp_cot_d5,synthrlvl_deduction_bp_cot_d15,synthrlvl_deduction_bp_cot_d25"; MML=8192 ;;
  folio_gpqa) TASKS="synthrlvl_folio,synthrlvl_gpqa_diamond"; MML=8192 ;;
  standard) TASKS="gsm8k,bbh,mmlu,arc_challenge,agieval_logiqa_en,hellaswag,winogrande,piqa"; MML=8192 ;;
  native) TASKS="synthrlvl_deduction_bp_native_d5,synthrlvl_deduction_bp_native_d15,synthrlvl_deduction_bp_native_d25"; MML=16384 ;;
  multihop) TASKS="synthrlvl_longbench_hotpotqa_standard,synthrlvl_longbench_2wikimqa_standard,synthrlvl_longbench_musique_standard,synthrlvl_longbench_hotpotqa_tagged,synthrlvl_longbench_2wikimqa_tagged,synthrlvl_longbench_musique_tagged"; MML=32768 ;;
  cot) TASKS="synthrlvl_deduction_pw_cot_d0,synthrlvl_deduction_pw_cot_d1,synthrlvl_deduction_pw_cot_d2,synthrlvl_deduction_pw_cot_d3,synthrlvl_deduction_pw_cot_d5"; MML=8192 ;;
  *) echo "unknown suite"; exit 2 ;;
esac

TASK_DIR="${TMPDIR:-/tmp}/synthrlvl_ood_${SLURM_JOB_ID}"
rm -rf "${TASK_DIR}"; cp -r lm_eval_tasks/synthrlvl_ood "${TASK_DIR}"
sed -i "s#/vol/tmp2/laitenbf/rlvl_data/datasets#${HPCVAULT}/synthetic-RLVL/datasets#g" "${TASK_DIR}"/*.yaml
python scripts/validate_lm_eval_tasks.py --include-path "${TASK_DIR}" --tasks "${TASKS}"
python scripts/evaluate_lm_eval.py --checkpoint "${CKPT}" --model vllm --tasks "${TASKS}" \
  --include-path "${TASK_DIR}" --output-path "${OUT}" --batch-size auto --dtype bfloat16 \
  --gpu-memory-utilization 0.85 --model-arg "max_model_len=${MML}" \
  --trust-remote-code --confirm-run-unsafe-code --apply-chat-template --log-samples
touch "${OUT}/.complete"; echo "done ${RUN_NAME} ${SUITE}"
