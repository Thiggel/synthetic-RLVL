#!/usr/bin/env bash
#SBATCH --job-name=fmix_tag
#SBATCH --partition=gpu-staff
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Format-tagged downstream benchmarks for one formal-mixture SFT checkpoint:
# every item of rlvl_data/datasets/formal_bench_tagged_20260926/test.jsonl
# (ProofWriter, FOLIO, BBH, GPQA, ARC-C, LogiQA, MMLU, GSM8K, LongBench
# multi-hop; scripts/build_formal_bench_tagged.py) is prompted with the
# <formal> tag and the proof is scored with rlvl.check for grammaticality,
# validity, grounding and answer correctness (scripts/eval_formal_bench_vllm.py,
# .venv_rlvl_vllm). The untagged counterpart is gruenau_formal_mix_bench_2026-09-26.slurm.
#
# Required env: RUN_NAME (dir under OUT_ROOT). Optional: PER_BENCH_LIMIT (smoke;
# writes formal_bench_tagged_n<N>), BENCHES, MIN_FREE_MIB, MAX_NEW_TOKENS.
# Output: ${OUT_ROOT}/${RUN_NAME}/formal_bench_tagged/{summary.json,generations.jsonl}.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

ROOT=/vol/tmp2/laitenbf
VENV="${ROOT}/.venv_rlvl_vllm"
OUT_ROOT="${ROOT}/rlvl_data/formal_mixture_sft_20260925"
export PATH="${VENV}/bin:${PATH}"
export VIRTUAL_ENV="${VENV}"
export HF_HOME="${ROOT}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# as in gruenau_formal_mix_eval_2026-09-25.slurm
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=/vol/home-vol2/ml/laitenbf/RLVL-next/gen:/vol/home-vol2/ml/laitenbf/RLVL-next/rlvl/python
hash -r 2>/dev/null || true

RUN_NAME="${RUN_NAME:?}"
MODEL="${OUT_ROOT}/${RUN_NAME}/final"
test -s "${MODEL}/config.json" || { echo "missing checkpoint ${MODEL}" >&2; exit 2; }
OUT_DIR="${OUT_ROOT}/${RUN_NAME}/formal_bench_tagged${PER_BENCH_LIMIT:+_n${PER_BENCH_LIMIT}}"
if [[ -s "${OUT_DIR}/summary.json" ]]; then echo "exists: ${OUT_DIR}/summary.json"; exit 0; fi

# Other users hold cards outside Slurm; wait until ours is really free.
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
gpu_free() { nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=memory.free,memory.total --format=csv,noheader,nounits | tr -d ',' | sort -n | head -1; }
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "run=${RUN_NAME} out=${OUT_DIR}"
for round in 1 2 3; do
  for a in $(seq 1 90); do
    read -r free_mib total_mib < <(gpu_free)
    (( free_mib >= MIN_FREE_MIB )) && break
    echo "waiting for GPU memory: ${free_mib} MiB free"; sleep 120
  done
  UTIL=$(python3 -c "print(round(min(0.85, (${free_mib}-3072)/${total_mib}), 3))")
  echo "round ${round}: gpu_mem=${UTIL}"
  if python scripts/eval_formal_bench_vllm.py --model "${MODEL}" --out-dir "${OUT_DIR}" --no-mm \
      --gpu-mem "${UTIL}" --max-new-tokens "${MAX_NEW_TOKENS:-2048}" --max-model-len 32768 \
      ${PER_BENCH_LIMIT:+--per-bench-limit "${PER_BENCH_LIMIT}"} ${BENCHES:+--benches "${BENCHES}"}; then
    echo "done ${RUN_NAME}"; exit 0
  fi
  echo "attempt ${round} failed; retrying in 120 s" >&2; sleep 120
done
exit 4
