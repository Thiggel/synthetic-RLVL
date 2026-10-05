#!/usr/bin/env bash
# G19 (2026-10-05): the G18 recipe (cvf_fmt, gen + Dolci prompts, overlong + no-proof penalties, 1000 steps) from the
# best EI SFT arm, plus a KL anchor to the init (--beta, default 0.02). G16/G17 lost generator-test answer accuracy
# (.90 -> .76 greedy, `tools` valid -> .01) between steps 250 and 1000 although 3,000 of the 7,643 RL prompts are
# generator prompts: saturated gen groups carry no advantage, so nothing holds them in place.
# Usage: bash scripts/submit_g19.sh <ei arm, e.g. e5 or e6> [beta]. One L40 on gruenau12 (needs an empty card).
set -euo pipefail
cd /vol/tmp2/laitenbf/synthetic-RLVL
ARM_EI=${1:?ei arm}; BETA=${2:-0.02}
# --vllm-mem 0.15 (G18: 0.2): with the KL reference model the fp32 logits of the reference pass overflow the L40 at
# 0.2 (first G19 attempt, job 12165, OOM at step 1 by ~0.3 GB). Only the vLLM KV cache shrinks; the recipe is unchanged.
# At 0.15 vLLM cannot hold one 262144-token sequence (job 12174), so max_model_len is capped at 16384 (prompt + 2048).
VLLM_MEM=${VLLM_MEM:-0.15}
DATA=/vol/tmp2/laitenbf/rlvl_data
S=${DATA}/checker_snapshot_libext_20261001
export POLICY=qwen35_2b_lc_libext_${ARM_EI}_lr5em6_seed3407 ARM=cvf_fmt \
  RUN_NAME=2b_${ARM_EI}_G19_cvffmt_overlong_noproof_kl${BETA#0.} RLVL_PYTHONPATH=${S}/gen:${S}/rlvl_python \
  MIN_FREE_MIB=37000 WAIT_FREE_MIN=1440
export EXTRA_ARGS="--resumable --benches gen,dolci_math,dolci_wordprob,dolci_yesno --max-per-bench 3000 --max-steps 1000 \
--save-steps 50 --keep-every 250 --per-device-batch 1 --vllm-mem ${VLLM_MEM} --vllm-max-model-len 16384 --beta ${BETA} \
--exclude-ids ${DATA}/datasets/rl_gate_dolci_instruct_20260928/contamination.json --no-mask-truncated \
--overlong-penalty 0.5 --no-proof-penalty 0.5"
j=$(sbatch --parsable --export=ALL -p gpu-staff --gres=gpu:l40:1 -w gruenau12 --cpus-per-task=16 --mem=160G \
  --time=2-00:00:00 --job-name=grpo_G19 scripts/slurm/jobs/grpo_formal.slurm)
j2=$(sbatch --parsable --export=ALL -p gpu-staff --gres=gpu:l40:1 -w gruenau12 --cpus-per-task=16 --mem=160G \
  --time=2-00:00:00 --job-name=grpo_G19 --dependency=singleton scripts/slurm/jobs/grpo_formal.slurm)
echo "G19 ${RUN_NAME}: ${j} -> ${j2}"
