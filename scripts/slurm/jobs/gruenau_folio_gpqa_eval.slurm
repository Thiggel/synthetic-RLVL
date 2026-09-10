#!/usr/bin/env bash
#SBATCH --job-name=gru_folio_gpqa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Two benchmarks the campaign never had, run on gruenau because alex is at its
# account GPU cap:
#
#   FOLIO          human-written first-order logic, 203 items, labels close to
#                  balanced (72 true / 69 uncertain / 62 false). This is the
#                  out-of-family deduction test. ProofWriter shares a genre with
#                  the generator; FOLIO does not, so it answers the referee who
#                  suspects the gain is template matching.
#   GPQA-Diamond   198 four-way multiple-choice science questions, chance 25
#                  percent. A regression check: replacing a tenth of the mixture
#                  should not damage hard reasoning outside deduction.
#
# GPQA's canonical repository is gated and this machine has no Hub token, so the
# fingertap mirror is used; absolute numbers are therefore not comparable to
# published GPQA scores, but the control-versus-treatment contrast is, since
# every arm sees the identical prompt.
#
# Array index: 0 control, 1 longdoc, 2 logic_band25, 3 nl_exact_band25.

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

ARMS=(control longdoc logic_band25 nl_exact_band25)
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEED:-3407}"
RUN_NAME="qwen25_7b_longwin_${ARM}_2p5b_dolci_100k_lr5em6"
CKPT="${DATA}/post_sft_alex/${RUN_NAME}/final"
OUT_DIR="${DATA}/lm_eval_results/qwen25_longwin_folio_gpqa_20260910/${RUN_NAME}"

TASKS="${TASKS:-synthrlvl_folio,synthrlvl_gpqa_diamond}"

# fail closed rather than evaluate half a transfer
test -s "${CKPT}/config.json" || { echo "missing checkpoint ${CKPT}" >&2; exit 2; }
n_shards=$(ls "${CKPT}"/*.safetensors 2>/dev/null | wc -l)
[ "${n_shards}" -ge 1 ] || { echo "no safetensors in ${CKPT}" >&2; exit 2; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "arm=${ARM} seed=${SEED} shards=${n_shards} ckpt=${CKPT}"

python scripts/validate_lm_eval_tasks.py --include-path lm_eval_tasks/synthrlvl_ood --tasks "${TASKS}"

python scripts/evaluate_lm_eval.py \
  --checkpoint "${CKPT}" \
  --model vllm \
  --tasks "${TASKS}" \
  --include-path lm_eval_tasks/synthrlvl_ood \
  --output-path "${OUT_DIR}" \
  --batch-size auto \
  --dtype bfloat16 \
  --gpu-memory-utilization "${GPU_UTIL:-0.70}" \
  --model-arg "max_model_len=8192" \
  --trust-remote-code \
  --confirm-run-unsafe-code \
  --apply-chat-template \
  --log-samples

echo "done ${ARM}"
