#!/usr/bin/env bash
#SBATCH --job-name=fmix_bench
#SBATCH --partition=gpu-staff
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Downstream benchmarks of the "Lead into Gold" paper for one formal-mixture
# SFT checkpoint (Qwen3.5-Base + Dolci with X% rlvlgen; X=0 is the pure-Dolci
# baseline). Suites, tasks, max_model_len and chat-template use are copied from
# gruenau_table1_eval_2026-09-22.slurm / alex_dose_eval_2026-09-12.slurm, with
# these differences:
#   - venv .venv_rlvl_lmeval35 (lm_eval 0.4.11 on vllm 0.30 / transformers
#     5.17): the paper's eval venv cannot load Qwen3.5.
#   - enable_thinking=False: training renders the empty <think></think> block.
#   - dropped: bp_cot / bp_native (old BranchProof format; the in-domain metric
#     is formal_eval/summary.json) and the *_tagged LongBench variants.
#   - `standard` drops bbh (bbh_nochat is the paper's BBH number).
# Prompts carry no <formal> tag: this measures transfer to ordinary prompting.
#
# Required env: RUN_NAME (dir under OUT_ROOT). Optional: SUITES, LIMIT,
# RESULTS (results study name), MIN_FREE_MIB.
# Each suite writes ${DATA}/lm_eval_results/${RESULTS}/${SUITE}/${RUN_NAME}
# and a .complete marker; finished suites are skipped.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

ROOT=/vol/tmp2/laitenbf
VENV="${ROOT}/.venv_rlvl_lmeval35"
DATA="${ROOT}/rlvl_data"
OUT_ROOT="${DATA}/formal_mixture_sft_20260925"
export PATH="${VENV}/bin:${PATH}"
export VIRTUAL_ENV="${VENV}"
export HF_HOME="${HF_HOME:-${ROOT}}"
export HPCVAULT="${DATA}"
export HF_ALLOW_CODE_EVAL=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TOKENIZERS_PARALLELISM=false
# as in gruenau_formal_mix_eval: flashinfer JIT needs the CUDA headers; the
# flashinfer sampler is unneeded for greedy decoding.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0
hash -r 2>/dev/null || true

RUN_NAME="${RUN_NAME:?}"
RESULTS="${RESULTS:-formal_mix_20260926}"
SUITES="${SUITES:-deduction cot folio_gpqa bbh_nochat multihop standard code_nochat}"
CKPT="${OUT_ROOT}/${RUN_NAME}/final"
test -s "${CKPT}/config.json" || { echo "missing checkpoint ${CKPT}" >&2; exit 2; }

TASK_DIR="${SLURM_TMPDIR:-/tmp}/synthrlvl_ood_${SLURM_JOB_ID:-$$}"
rm -rf "${TASK_DIR}"; cp -r lm_eval_tasks/synthrlvl_ood "${TASK_DIR}"
sed -i "s#/home/vault/c107fa/c107fa12/synthetic-RLVL/datasets#${DATA}/datasets#g; s#/home/vault/c107fa/c107fa12/synthetic-RLVL#${DATA}#g" "${TASK_DIR}"/*.yaml
grep -l "/home/vault" "${TASK_DIR}"/*.yaml && { echo "unrewritten vault path" >&2; exit 2; }
trap 'rm -rf "${TASK_DIR}"' EXIT

# Other users hold gruenau cards outside Slurm; wait until ours is really free.
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
gpu_free() { nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=memory.free,memory.total --format=csv,noheader,nounits | tr -d ',' | sort -n | head -1; }
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

for SUITE in ${SUITES}; do
  case "${SUITE}" in
    deduction) TASKS="synthrlvl_deduction_pw_d0,synthrlvl_deduction_pw_d1,synthrlvl_deduction_pw_d2,synthrlvl_deduction_pw_d3,synthrlvl_deduction_pw_d5"; MML=8192 ;;
    cot) TASKS="synthrlvl_deduction_pw_cot_d0,synthrlvl_deduction_pw_cot_d1,synthrlvl_deduction_pw_cot_d2,synthrlvl_deduction_pw_cot_d3,synthrlvl_deduction_pw_cot_d5"; MML=8192 ;;
    folio_gpqa) TASKS="synthrlvl_folio,synthrlvl_gpqa_diamond"; MML=8192 ;;
    bbh_nochat) TASKS="bbh"; MML=8192 ;;
    multihop) TASKS="synthrlvl_longbench_hotpotqa_standard,synthrlvl_longbench_2wikimqa_standard,synthrlvl_longbench_musique_standard"; MML=32768 ;;
    standard) TASKS="gsm8k,mmlu,arc_challenge,agieval_logiqa_en,hellaswag,winogrande,piqa"; MML=8192 ;;
    code_nochat) TASKS="humaneval,mbpp"; MML=8192 ;;
    *) echo "unknown suite ${SUITE}" >&2; exit 2 ;;
  esac
  CHAT_FLAG="--apply-chat-template"
  case "${SUITE}" in *_nochat) CHAT_FLAG="" ;; esac
  OUT="${DATA}/lm_eval_results/${RESULTS}/${SUITE}/${RUN_NAME}"
  if [[ -e "${OUT}/.complete" ]]; then echo "already done: ${OUT}"; continue; fi
  mkdir -p "$(dirname "${OUT}")"
  python scripts/validate_lm_eval_tasks.py --include-path "${TASK_DIR}" --tasks "${TASKS}"
  echo "run=${RUN_NAME} suite=${SUITE} mml=${MML} tasks=${TASKS}"
  ok=0
  for round in 1 2 3; do
    for a in $(seq 1 90); do
      read -r free_mib total_mib < <(gpu_free)
      (( free_mib >= MIN_FREE_MIB )) && break
      echo "waiting for GPU memory: ${free_mib} MiB free"; sleep 120
    done
    UTIL=$(python3 -c "print(round(min(0.85, (${free_mib}-3072)/${total_mib}), 3))")
    echo "round ${round}: gpu_memory_utilization=${UTIL}"
    if python scripts/evaluate_lm_eval.py --checkpoint "${CKPT}" --model vllm --tasks "${TASKS}" \
      --include-path "${TASK_DIR}" --output-path "${OUT}" --batch-size auto --dtype bfloat16 \
      --gpu-memory-utilization "${UTIL}" --model-arg "tensor_parallel_size=1" --model-arg "max_model_len=${MML}" \
      --model-arg enable_thinking=False ${LIMIT:+--limit "${LIMIT}"} \
      --trust-remote-code --confirm-run-unsafe-code ${CHAT_FLAG} --log-samples; then ok=1; break; fi
    echo "attempt ${round} failed; retrying in 120 s" >&2; sleep 120
  done
  [ "${ok}" = 1 ] || { echo "suite ${SUITE} failed" >&2; exit 4; }
  [[ -n "${LIMIT:-}" ]] || touch "${OUT}/.complete"
  echo "done ${RUN_NAME} ${SUITE}"
done
