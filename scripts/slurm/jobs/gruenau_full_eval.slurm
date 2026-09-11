#!/usr/bin/env bash
#SBATCH --job-name=gru_full_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# One evaluation entry point for every checkpoint kind and suite, run on
# gruenau because alex sits at its account GPU cap.
#
#   KIND=it     the instruction-tuned checkpoints, the ones the paper reports.
#               Prompts go through the chat template.
#   KIND=base   the midtrained checkpoints before any instruction tuning. These
#               have never been evaluated. No chat template is applied, because
#               a base model has not been taught one; numbers are therefore not
#               comparable to the instruction-tuned table and are reported
#               separately. The point is to see whether the notation survives
#               midtraining and is only washed out later.
#
#   SUITE=deduction   ProofWriter d0-d5 and BranchProof-CoT d5-d25
#   SUITE=folio_gpqa  FOLIO and GPQA-Diamond
#   SUITE=standard    GSM8K, MATH-500, BBH, MMLU, ARC, HellaSwag, WinoGrande,
#                     PIQA, LogiQA
#   SUITE=multihop    HotpotQA, 2Wiki, MuSiQue, standard and tagged
#   SUITE=deduction_cot  ProofWriter d0-d5 with a derive-then-answer prompt
#   SUITE=deduction_mc   ProofWriter d0-d5 scored by choice log-likelihood
#   SUITE=deduction_native BranchProof d5-d25 in the midtraining document format
#   SUITE=deduction_pert ProofWriter decidable items with the claim polarity
#                        flipped, and with the proof's premises removed
#
# ARM selects the condition. Everything fails closed on a missing checkpoint.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

ROOT=/vol/home-vol2/ml/laitenbf
VENV="${ROOT}/.venv_rlvl_gruenau_eval"
DATA="${ROOT}/rlvl_data"
export PATH="${VENV}/bin:${PATH}"
export VIRTUAL_ENV="${VENV}"
export HF_HOME="${HF_HOME:-${DATA}/hf_home}"
export HPCVAULT="${DATA}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TOKENIZERS_PARALLELISM=false
hash -r 2>/dev/null || true

KIND="${KIND:?set KIND=it or base}"
ARM="${ARM:?set ARM}"
SUITE="${SUITE:?set SUITE}"
SEED="${SEED:-3407}"

case "${SUITE}" in
  deduction)
    TASKS="synthrlvl_deduction_pw_d0,synthrlvl_deduction_pw_d1,synthrlvl_deduction_pw_d2,synthrlvl_deduction_pw_d3,synthrlvl_deduction_pw_d5,synthrlvl_deduction_bp_cot_d5,synthrlvl_deduction_bp_cot_d10,synthrlvl_deduction_bp_cot_d15,synthrlvl_deduction_bp_cot_d20,synthrlvl_deduction_bp_cot_d25" ;;
  folio_gpqa) TASKS="synthrlvl_folio,synthrlvl_gpqa_diamond" ;;
  deduction_cot) TASKS="synthrlvl_deduction_pw_cot_d0,synthrlvl_deduction_pw_cot_d1,synthrlvl_deduction_pw_cot_d2,synthrlvl_deduction_pw_cot_d3,synthrlvl_deduction_pw_cot_d5" ;;
  deduction_native) TASKS="synthrlvl_deduction_bp_native_d5,synthrlvl_deduction_bp_native_d10,synthrlvl_deduction_bp_native_d15,synthrlvl_deduction_bp_native_d20,synthrlvl_deduction_bp_native_d25" ;;
  deduction_native_long) TASKS="synthrlvl_deduction_bp_native_long_d25"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"; NEED_GIB="${NEED_GIB:-34}" ;;
  deduction_pert) TASKS="synthrlvl_deduction_pw_flip,synthrlvl_deduction_pw_ablate" ;;
  deduction_mc)  TASKS="synthrlvl_deduction_pw_mc_d0,synthrlvl_deduction_pw_mc_d1,synthrlvl_deduction_pw_mc_d2,synthrlvl_deduction_pw_mc_d3,synthrlvl_deduction_pw_mc_d5" ;;
  standard)   TASKS="gsm8k,hendrycks_math500,bbh,mmlu,arc_challenge,hellaswag,winogrande,piqa,agieval_logiqa_en" ;;
  multihop)   TASKS="synthrlvl_longbench_hotpotqa_standard,synthrlvl_longbench_hotpotqa_tagged,synthrlvl_longbench_2wikimqa_standard,synthrlvl_longbench_2wikimqa_tagged,synthrlvl_longbench_musique_standard,synthrlvl_longbench_musique_tagged" ;;
  *) echo "unknown SUITE=${SUITE}" >&2; exit 2 ;;
esac

# The multihop contexts are LongBench documents of 15-75k characters. Alex ran
# them at max_model_len 32768 (longwin_downstream_eval_2026-09-07.slurm); at
# 8192 lm-eval left-truncates the prompt and the first gruenau multihop
# bundles came out 15-30 points below alex. Match alex per suite.
if [[ "${SUITE}" == "multihop" ]]; then
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"; NEED_GIB="${NEED_GIB:-40}"
else
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
fi
if [[ "${SEED}" == "3407" ]]; then SUF=""; else SUF="_seed${SEED}"; fi
if [[ "${KIND}" == "it" ]]; then
  RUN_NAME="qwen25_7b_longwin_${ARM}_2p5b_dolci_100k_lr5em6${SUF}"
  CKPT="${DATA}/post_sft_alex/${RUN_NAME}/final"
  CHAT_ARGS=(--apply-chat-template)
else
  RUN_NAME="qwen25_7b_longwin_${ARM}_2p5b_base"
  CKPT="${DATA}/hf_midtrain/qwen25_7b_longwin_${ARM}_2p5b_hf"
  CHAT_ARGS=()   # a base model has no chat template
fi
OUT_DIR="${DATA}/lm_eval_results/gruenau_${KIND}_${SUITE}_20260910${SUF}/${RUN_NAME}"

test -s "${CKPT}/config.json" || { echo "missing checkpoint ${CKPT}" >&2; exit 2; }
[ "$(ls "${CKPT}"/*.safetensors 2>/dev/null | wc -l)" -ge 1 ] || { echo "no shards in ${CKPT}" >&2; exit 2; }
# .complete is an empty marker, so test for existence, not size
if [[ -e "${OUT_DIR}/.complete" ]]; then echo "already done: ${OUT_DIR}"; exit 0; fi

# These GPUs are shareable (the partition advertises MPS), so a neighbouring
# job can already hold most of the card when we start. vllm refuses to launch
# if the requested fraction is not free, which killed six jobs outright. Wait
# for room rather than dying, then size the fraction to what is actually free.
NEED_GIB="${NEED_GIB:-26}"
wait_for_gpu() {
  for attempt in $(seq 1 60); do
    GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"; GPU_ID="${GPU_ID:-0}"
    free_mib=$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    total_mib=$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    free_gib=$(( free_mib / 1024 ))
    if [ "${free_gib}" -ge "${NEED_GIB}" ]; then return 0; fi
    echo "waiting for GPU memory: ${free_gib} GiB free, need ${NEED_GIB} (attempt ${attempt}/60)"
    sleep 120
  done
  echo "gave up waiting: only ${free_gib} GiB free" >&2; return 3
}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "kind=${KIND} arm=${ARM} suite=${SUITE} seed=${SEED}"
echo "ckpt=${CKPT}"

# The task yamls hardcode alex's vault path for their data files. Rewrite them
# into a scratch include directory so the same yamls serve both clusters.
TASK_DIR="${SLURM_TMPDIR:-/tmp}/synthrlvl_ood_${SLURM_JOB_ID}"
rm -rf "${TASK_DIR}"; cp -r lm_eval_tasks/synthrlvl_ood "${TASK_DIR}"
sed -i "s#/home/vault/c107fa/c107fa12/synthetic-RLVL#${DATA}#g" "${TASK_DIR}"/*.yaml
grep -l "/home/vault" "${TASK_DIR}"/*.yaml && { echo "unrewritten vault path" >&2; exit 2; }

python scripts/validate_lm_eval_tasks.py --include-path "${TASK_DIR}" --tasks "${TASKS}"

# A neighbour can grab memory between the check and vLLM's own check, which
# then refuses to start ("Free memory on device ... is less than desired").
# Re-measure and retry a few times instead of dying on the race.
ok=0
for round in 1 2 3 4 5; do
  wait_for_gpu || exit 3
  # leave a small margin below what is free so a neighbour growing slightly does
  # not abort us
  GPU_UTIL="${GPU_UTIL_OVERRIDE:-$(python3 -c "print(round(min(0.70, (${free_mib}-4096)/${total_mib}), 3))")}"
  echo "round ${round}: GPU has ${free_gib} GiB free; using gpu_memory_utilization=${GPU_UTIL}"
  if python scripts/evaluate_lm_eval.py \
    --checkpoint "${CKPT}" \
    --model vllm \
    --tasks "${TASKS}" \
    --include-path "${TASK_DIR}" \
    --output-path "${OUT_DIR}" \
    --batch-size auto \
    --dtype bfloat16 \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --model-arg "max_model_len=${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --confirm-run-unsafe-code \
    "${CHAT_ARGS[@]}" \
    --log-samples; then ok=1; break; fi
  echo "evaluation attempt ${round} failed; retrying after 180 s" >&2
  sleep 180
done
[ "${ok}" = 1 ] || { echo "evaluation failed after 5 attempts" >&2; exit 4; }

touch "${OUT_DIR}/.complete"
echo "done kind=${KIND} arm=${ARM} suite=${SUITE}"
