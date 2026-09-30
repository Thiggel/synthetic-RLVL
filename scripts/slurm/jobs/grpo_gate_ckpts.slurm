#!/usr/bin/env bash
#SBATCH --job-name=grpo_gate
#SBATCH --partition=gpu-staff
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Gate eval (rl_gate_dolci, 950 items, greedy) of GRPO checkpoints, one after another on one GPU.
# Args: run:ckpt pairs under rlvl_data/grpo_formal_20260928 (or $GATE_ROOT). Waits until each checkpoint's
# weights are written; skips checkpoints that already have a gate summary.
# Standalone form of grpo_gate_ckpts_in_6964.sh (2026-09-28).
cd /vol/tmp2/laitenbf/synthetic-RLVL
export MIN_FREE_MIB="${MIN_FREE_MIB:-20000}" OUT_NAME="${OUT_NAME:-rl_gate_dolci}"  # OUT_NAME + EVAL_ARGS: e.g. rl_gate_dolci_k16 + "--n-samples 16 --temperature 1.0"
export TEST_JSONL=/vol/tmp2/laitenbf/rlvl_data/datasets/rl_gate_dolci_instruct_20260928/test.jsonl
export OUT_ROOT="${GATE_ROOT:-/vol/tmp2/laitenbf/rlvl_data/grpo_formal_20260928}"  # GATE_ROOT: e.g. formal_mixture_sft_20260925 for SFT runs
for rc in "$@"; do
  d="${OUT_ROOT}/${rc%%:*}/${rc##*:}"
  [[ -s "${d}/${OUT_NAME}/summary.json" ]] && { echo "gated: ${rc}"; continue; }
  until ls "${d}"/model*.safetensors >/dev/null 2>&1 && [[ -f "${d}/config.json" ]]; do sleep 60; done
  sleep 120  # let the save finish
  RUN_NAME=${rc%%:*} CKPT=${rc##*:} bash scripts/slurm/jobs/gruenau_formal_mix_tagged_2026-09-26.slurm || echo "gate failed: ${rc}" >&2
done
