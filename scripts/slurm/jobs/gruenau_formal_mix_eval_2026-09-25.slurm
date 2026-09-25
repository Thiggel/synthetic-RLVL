#!/usr/bin/env bash
#SBATCH --job-name=gru_fmix_eval
#SBATCH --partition=gpu-staff
#SBATCH --gres=gpu:l40:1
# L40 (48 GB) on gruenau12: gruenau11 H100s are for training, and two of them
# are held outside Slurm by another user (see the train script).
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Formal-CoT mixture sweep eval (2026-09-25): vLLM generation with the
# simulated tool loop on the 2000-problem rlvlgen held-out test set, scored
# for faithful premises, grammaticality (rlvl.check strict=False) and validity
# (strict), with answer accuracy separately. scripts/eval_formal_vllm.py in
# .venv_rlvl_vllm (vllm 0.30.0).
#
# Required env: CKPT (a trained run dir containing final/, or any HF model
# dir). Optional: LIMIT, OUT_DIR, MAX_NEW_TOKENS.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

ROOT=/vol/tmp2/laitenbf
VENV="${ROOT}/.venv_rlvl_vllm"
export PATH="${VENV}/bin:${PATH}"
export VIRTUAL_ENV="${VENV}"
export HF_HOME="${ROOT}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# flashinfer JIT-compiles its sampling kernels and needs CUDA headers; the
# compute nodes have no /usr/local/cuda symlink. Use the CUDA 13 toolkit
# (matches torch cu130) and skip the flashinfer sampler anyway.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=/vol/home-vol2/ml/laitenbf/RLVL-next/gen:/vol/home-vol2/ml/laitenbf/RLVL-next/rlvl/python
hash -r 2>/dev/null || true

: "${CKPT:?set CKPT=<run dir or model dir>}"
MODEL="${CKPT}"
[[ -s "${CKPT}/final/config.json" ]] && MODEL="${CKPT}/final"
test -s "${MODEL}/config.json" || { echo "missing model: ${MODEL}" >&2; exit 2; }
TEST="${ROOT}/rlvl_data/datasets/formal_mixture_20260925/pool/test.jsonl"
test -s "${TEST}" || { echo "missing test set: ${TEST}" >&2; exit 2; }
OUT_DIR="${OUT_DIR:-${CKPT}/formal_eval${LIMIT:+_n${LIMIT}}}"
if [[ -s "${OUT_DIR}/summary.json" ]]; then
  echo "eval exists: ${OUT_DIR}/summary.json; skipping"; exit 0
fi

# The allocated GPU may be partly held by processes outside Slurm (seen on
# gruenau11 H100s and gruenau12 L40s, 2026-09-25), and their usage changes
# between our check and vLLM's startup check. Size vLLM's memory fraction to
# what is free now, leave a margin, and retry a few times on startup failure.
# The 9B eval should run on an H100 NVL. Slurm labels all four gruenau11 cards
# h100nvl, including the two occupied H100 PCIe, so submit_formal_mixture_sweep.sh
# requests the whole node for it and we run on the allocated GPU with the most
# free memory (smoke 6598 was handed an occupied PCIe card with 8 GB free).
MIN_GPU_MEM="${MIN_GPU_MEM:-0.3}"
GPU_IDX=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
export CUDA_VISIBLE_DEVICES="${GPU_IDX}"
echo "using GPU ${GPU_IDX}"
# Low memory waits (up to ~6 h, 5 min apart); a failed eval run is retried 4 times.
attempt=0; polls=0
while (( attempt < 4 && polls < 72 )); do
  read -r FREE TOTAL < <(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits -i "${GPU_IDX}" | tr -d ',')
  GPU_MEM=$(python -c "print(round(min(0.85, (${FREE} - 4096) / ${TOTAL}), 3))")
  echo "attempt $((attempt + 1)): gpu free ${FREE}/${TOTAL} MiB -> --gpu-mem ${GPU_MEM}"
  if python -c "import sys; sys.exit(0 if ${GPU_MEM} >= ${MIN_GPU_MEM} else 1)"; then
    if python scripts/eval_formal_vllm.py \
        --gpu-mem "${GPU_MEM}" \
        --test-jsonl "${TEST}" \
        --model "${MODEL}" \
        --out-dir "${OUT_DIR}" \
        --max-new-tokens "${MAX_NEW_TOKENS:-2048}" \
        --no-mm \
        ${LIMIT:+--limit "${LIMIT}"}; then
      echo "done ${OUT_DIR}"; exit 0
    fi
    attempt=$(( attempt + 1 )); sleep 60
  else
    echo "too little free GPU memory (< ${MIN_GPU_MEM}); waiting" >&2
    polls=$(( polls + 1 )); sleep 300
  fi
done
echo "eval failed (${attempt} failed runs, ${polls} low-memory waits)" >&2
exit 3
