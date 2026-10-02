#!/usr/bin/env bash
# EI round 3 (2026-10-03): teacher G16 checkpoint-250 (copied to rlvl_data/ei3_teacher_G16_ckpt250), harvest checked
# with the NEW checker snapshot (the teacher is a new-library model). Otherwise identical to submit_ei2.sh:
# harvest (cvf, n=16, gate near-duplicates excluded) on GSM8K train + Dolci wordprob/math -> build e3 (CPU)
# -> continued SFT from the L1 base on one gruenau12 L40 -> eval suite.
set -euo pipefail
cd /vol/tmp2/laitenbf/synthetic-RLVL
DATA=/vol/tmp2/laitenbf/rlvl_data
S=${DATA}/checker_snapshot_libext_20261001
T=${DATA}/ei3_teacher_G16_ckpt250
O=${DATA}/rl_filter_20261003
C=${DATA}/datasets/rl_gate_dolci_instruct_20260928/contamination.json
BASE=qwen35_2b_p50_cont_lc_fp32m_lr5em6_seed3407
NODE=(-p gpu-staff --gres=gpu:l40:1 -w gruenau12 --cpus-per-task=16)
RUN=qwen35_2b_lc_libext_e3_lr5em6_seed3407
mkdir -p "${O}"
a=$(env RLVL_PYTHONPATH=${S}/gen:${S}/rlvl_python MODEL=${T} MIN_FREE_MIB=13000 OUT=${O}/G16c250_cvf_n16_gsm8k.json \
  FILTER_ARGS="--arm cvf --benches gsm8k_train --max-per-bench 10000 --exclude-ids ${C}" \
  sbatch --parsable --export=ALL "${NODE[@]}" ${HARVEST_DEP:+--dependency=${HARVEST_DEP}} --job-name=ei3_harvest_gsm8k \
  scripts/slurm/jobs/rl_prompt_filter.slurm)
d=$(env RLVL_PYTHONPATH=${S}/gen:${S}/rlvl_python MODEL=${T} MIN_FREE_MIB=13000 OUT=${O}/G16c250_cvf_n16_dolci.json \
  FILTER_ARGS="--arm cvf --benches dolci_math,dolci_wordprob --max-per-bench 3000 --exclude-ids ${C}" \
  sbatch --parsable --export=ALL "${NODE[@]}" ${HARVEST_DEP:+--dependency=${HARVEST_DEP}} --job-name=ei3_harvest_dolci \
  scripts/slurm/jobs/rl_prompt_filter.slurm)
b=$(sbatch --parsable --export=ALL -p compute --cpus-per-task=8 --mem=64G --time=04:00:00 --job-name=build_ei3 \
  --output=logs/%x_%j.out --dependency=afterok:${a}:${d} \
  --wrap "cd /vol/tmp2/laitenbf/synthetic-RLVL && HF_HOME=/vol/tmp2/laitenbf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
/vol/tmp2/laitenbf/.venv_rlvl_grpo/bin/python scripts/data/build_libext_ei_mixture.py --arms e3")
t=$(MODEL_KEY=2b MIX=libext_e3 MIX_DIR=${DATA}/datasets/formal_libext_ei_20260930/e3 \
  INIT_MODEL=${DATA}/formal_mixture_sft_20260925/${BASE}/final RUN_NAME=${RUN} \
  DS_CONFIG=configs/deepspeed/zero2_offload_optim.json GRAD_CKPT=1 LIGER_FLCE=1 PER_DEVICE_BATCH=1 MIN_FREE_MIB=40000 \
  sbatch --parsable --export=ALL "${NODE[@]}" --job-name=sft_libext_e3 --dependency=afterok:${b} \
    scripts/slurm/jobs/gruenau_formal_mix_train_2026-09-25.slurm)
e=$(RUN_NAME=${RUN} sbatch --parsable --export=ALL "${NODE[@]}" --job-name=eval_libext_e3 --dependency=afterok:${t} \
  scripts/slurm/jobs/sft_eval_suite.slurm)
echo "ei3_harvest_gsm8k ${a} ei3_harvest_dolci ${d} build_ei3 ${b} sft_libext_e3 ${t} eval_libext_e3 ${e}"
