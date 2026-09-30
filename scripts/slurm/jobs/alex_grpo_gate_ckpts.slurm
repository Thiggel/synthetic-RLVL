#!/usr/bin/env bash
#SBATCH --job-name=grpo_gate
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/atuin/c107fa/c107fa12/rlvl_data/logs/%x_%j.out
#SBATCH --error=/home/atuin/c107fa/c107fa12/rlvl_data/logs/%x_%j.err

# alex version of grpo_gate_ckpts.slurm: gate eval (rl_gate_dolci, 950 items, greedy) of GRPO checkpoints,
# one after another on one GPU, with eval_formal_bench_vllm.py run as in
# gruenau_formal_mix_tagged_2026-09-26.slurm (but in the GRPO env of alex_grpo_env.sh: same vLLM 0.30,
# torch and transformers as .venv_rlvl_vllm). Args: run:ckpt pairs under
# ${RLVL_DATA_ROOT}/grpo_formal_20260928 (or $GATE_ROOT, e.g. .../formal_mixture_sft_20260925 for SFT runs).
# Waits until each checkpoint is completely written; skips checkpoints that already have a gate summary.
# OUT_NAME + EVAL_ARGS: e.g. OUT_NAME=rl_gate_dolci_k16 EVAL_ARGS="--n-samples 16 --temperature 1.0"
# (sampled pass@k). Export them in the shell and submit with --export=ALL.
# Output: ${GATE_ROOT}/${run}/${ckpt}/${OUT_NAME}/{summary.json,generations.jsonl}

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export RLVL_DATA_ROOT="${RLVL_DATA_ROOT:-/home/atuin/c107fa/c107fa12/rlvl_data}"
source scripts/slurm/jobs/alex_grpo_env.sh
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
OUT_NAME="${OUT_NAME:-rl_gate_dolci}"
TEST_JSONL="${RLVL_DATA_ROOT}/datasets/rl_gate_dolci_instruct_20260928/test.jsonl"
OUT_ROOT="${GATE_ROOT:-${RLVL_DATA_ROOT}/grpo_formal_20260928}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
for rc in "$@"; do
  d="${OUT_ROOT}/${rc%%:*}/${rc##*:}"
  [[ -s "${d}/${OUT_NAME}/summary.json" ]] && { echo "gated: ${rc}"; continue; }
  until ls "${d}"/model*.safetensors >/dev/null 2>&1 && [[ -f "${d}/config.json" ]]; do sleep 60; done
  # a Trainer checkpoint is complete once trainer_state.json is written (last); final/ has none
  if [[ "${d}" == */checkpoint-* ]]; then
    for i in $(seq 1 60); do [[ -f "${d}/trainer_state.json" ]] && break; sleep 30; done
  else sleep 120; fi
  echo "gate ${rc} -> ${d}/${OUT_NAME}"
  ok=0
  for round in 1 2 3; do
    if "${VENV}/bin/python" scripts/eval_formal_bench_vllm.py --model "${d}" --out-dir "${d}/${OUT_NAME}" --no-mm \
        --gpu-mem 0.85 --max-new-tokens "${MAX_NEW_TOKENS:-2048}" --max-model-len 32768 ${EVAL_ARGS:-} \
        --test-jsonl "${TEST_JSONL}"; then ok=1; break; fi
    echo "attempt ${round} failed; retrying in 120 s" >&2; sleep 120
  done
  (( ok )) || echo "gate failed: ${rc}" >&2
done
