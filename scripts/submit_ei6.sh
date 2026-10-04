#!/usr/bin/env bash
# EI round 6 (2026-10-04): the e5 student samples 32 proofs on 3,000 gsm8k_train (de-contaminated against the gate)
# and the 1,000 round-1 Dolci prompts -> build e6 (e5's data + these, hardened filter) -> continued SFT from the L1 base
# on one gruenau12 L40 -> eval suite. Env vars go through `export` + --export=ALL (sbatch splits --export on commas).
# Thresholds sized for the shared gruenau12 L40s (foreign processes hold ~27 GB on several cards).
set -euo pipefail
cd /vol/tmp2/laitenbf/synthetic-RLVL
DATA=/vol/tmp2/laitenbf/rlvl_data
A=${DATA}/az/selfplay_r1
S=${DATA}/checker_snapshot_libext_20261001
CONTAM=${DATA}/datasets/rl_gate_dolci_instruct_20260928/contamination.json
BASE=qwen35_2b_p50_cont_lc_fp32m_lr5em6_seed3407
TEACHER=qwen35_2b_lc_libext_e5_lr5em6_seed3407
RUN=qwen35_2b_lc_libext_e6_lr5em6_seed3407
NODE=(-p gpu-staff --gres=gpu:l40:1 -w gruenau12 --cpus-per-task=16)
export MODEL=${DATA}/formal_mixture_sft_20260925/${TEACHER}/final RLVL_PYTHONPATH=${S}/gen:${S}/rlvl_python \
  MIN_FREE_MIB=16000 WAIT_FREE_MIN=720
export OUT=${A}/sample_e5_n32.json FILTER_ARGS="--arm cvf --n 32 --benches gsm8k_train --max-per-bench 3000 --exclude-ids ${CONTAM}"
s1=$(sbatch --parsable --export=ALL "${NODE[@]}" --job-name=az_sample_e5_gsm scripts/slurm/jobs/rl_prompt_filter.slurm)
export OUT=${A}/sample_e5_n32_dolci.json FILTER_ARGS="--arm cvf --n 32 --benches dolci_wordprob,dolci_math --max-per-bench 1000000 --ids ${A}/dolci_ids.jsonl --exclude-ids ${CONTAM}"
s2=$(sbatch --parsable --export=ALL "${NODE[@]}" --job-name=az_sample_e5_dolci scripts/slurm/jobs/rl_prompt_filter.slurm)
b=$(sbatch --parsable --export=ALL -p compute --cpus-per-task=8 --mem=64G --time=04:00:00 --job-name=build_ei6 \
  --output=logs/%x_%j.out --dependency=afterok:${s1}:${s2} \
  --wrap "cd /vol/tmp2/laitenbf/synthetic-RLVL && HF_HOME=/vol/tmp2/laitenbf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
/vol/tmp2/laitenbf/.venv_rlvl_grpo/bin/python scripts/data/build_libext_ei_mixture.py --arms e6")
export MODEL_KEY=2b MIX=libext_e6 MIX_DIR=${DATA}/datasets/formal_libext_ei_20260930/e6 \
  INIT_MODEL=${DATA}/formal_mixture_sft_20260925/${BASE}/final RUN_NAME=${RUN} \
  DS_CONFIG=configs/deepspeed/zero2_offload_optim.json GRAD_CKPT=1 LIGER_FLCE=1 PER_DEVICE_BATCH=1 MIN_FREE_MIB=17000 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
t=$(sbatch --parsable --export=ALL "${NODE[@]}" --job-name=sft_libext_e6 --dependency=afterok:${b} \
  scripts/slurm/jobs/gruenau_formal_mix_train_2026-09-25.slurm)
export MIN_FREE_MIB=15000
e=$(sbatch --parsable --export=ALL "${NODE[@]}" --job-name=eval_libext_e6 --dependency=afterok:${t} \
  scripts/slurm/jobs/sft_eval_suite.slurm)
echo "sample ${s1} ${s2} build_ei6 ${b} sft_libext_e6 ${t} eval_libext_e6 ${e}"
