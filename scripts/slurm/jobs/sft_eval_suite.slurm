#!/usr/bin/env bash
#SBATCH --job-name=sft_eval_suite
#SBATCH --partition=gpu-staff
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Held-out evals of one continued-SFT run (2026-09-30, EI x lib-extension arms), one after another on one GPU:
#   ${RUN}/final/rl_gate_dolci      Dolci gate, 950 real prompts, greedy (valid@1, correct@1)
#   ${RUN}/formal_eval              in-domain generator test, 2000 items (default families)
#   ${RUN}/formal_eval_math         new-family generator test, 1000 items (formal_libext_20260930 test_math)
#   ${RUN}/final/rl_gate_dolci_k16  Dolci gate, 16 samples at T=1.0 (valid@k, pass@k)
# Scored with the checker on PYTHONPATH: the live RLVL-next (extended lemma library) unless RLVL_PYTHONPATH
# is set. Each eval is skipped when its summary.json exists. Required env: RUN_NAME (dir under OUT_ROOT).
set -uo pipefail
cd /vol/tmp2/laitenbf/synthetic-RLVL
DATA=/vol/tmp2/laitenbf/rlvl_data
export OUT_ROOT="${OUT_ROOT:-${DATA}/formal_mixture_sft_20260925}"
RUN="${OUT_ROOT}/${RUN_NAME:?}"
test -s "${RUN}/final/config.json" || { echo "missing model: ${RUN}/final" >&2; exit 2; }
GATE=${DATA}/datasets/rl_gate_dolci_instruct_20260928/test.jsonl
rc=0
OUT_NAME=rl_gate_dolci TEST_JSONL=${GATE} CKPT=final \
  bash scripts/slurm/jobs/gruenau_formal_mix_tagged_2026-09-26.slurm || { echo "gate failed" >&2; rc=1; }
CKPT="${RUN}" OUT_DIR="${RUN}/formal_eval" \
  bash scripts/slurm/jobs/gruenau_formal_mix_eval_2026-09-25.slurm || { echo "formal_eval failed" >&2; rc=1; }
CKPT="${RUN}" OUT_DIR="${RUN}/formal_eval_math" TEST=${DATA}/datasets/formal_libext_20260930/pool/test_math.jsonl \
  bash scripts/slurm/jobs/gruenau_formal_mix_eval_2026-09-25.slurm || { echo "formal_eval_math failed" >&2; rc=1; }
OUT_NAME=rl_gate_dolci_k16 TEST_JSONL=${GATE} CKPT=final EVAL_ARGS="--n-samples 16 --temperature 1.0" \
  bash scripts/slurm/jobs/gruenau_formal_mix_tagged_2026-09-26.slurm || { echo "gate k16 failed" >&2; rc=1; }
exit ${rc}
