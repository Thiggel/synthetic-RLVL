#!/usr/bin/env bash
# EI round 2 (2026-10-01): arm e2 = arm e of scripts/data/build_libext_ei_mixture.py with the harvest from an RL'd
# teacher (G12 checkpoint-100, copied to rlvl_data/ei2_teacher_G12_ckpt100; harvest jobs scripts/rl_prompt_filter.py).
# Chain: build the e2 mixture (CPU, after the harvests) -> continued SFT from the L1 base on one gruenau12 L40
# -> eval suite (Dolci gate greedy + k16, generator test, new-family test), exactly as arm e.
# Usage: HARVEST_DEP=afterok:9778:9779 bash scripts/submit_ei2.sh
set -euo pipefail
cd /vol/tmp2/laitenbf/synthetic-RLVL
DATA=/vol/tmp2/laitenbf/rlvl_data
BASE=qwen35_2b_p50_cont_lc_fp32m_lr5em6_seed3407
NODE=(-p gpu-staff --gres=gpu:l40:1 -w gruenau12 --cpus-per-task=16)
RUN=qwen35_2b_lc_libext_e2_lr5em6_seed3407

b=$(sbatch --parsable --export=ALL -p compute --cpus-per-task=8 --mem=64G --time=04:00:00 --job-name=build_ei2 \
  --output=logs/%x_%j.out --dependency=${HARVEST_DEP:?} \
  --wrap "cd /vol/tmp2/laitenbf/synthetic-RLVL && HF_HOME=/vol/tmp2/laitenbf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
/vol/tmp2/laitenbf/.venv_rlvl_grpo/bin/python scripts/data/build_libext_ei_mixture.py --arms e2")
t=$(MODEL_KEY=2b MIX=libext_e2 MIX_DIR=${DATA}/datasets/formal_libext_ei_20260930/e2 \
  INIT_MODEL=${DATA}/formal_mixture_sft_20260925/${BASE}/final RUN_NAME=${RUN} \
  DS_CONFIG=configs/deepspeed/zero2_offload_optim.json GRAD_CKPT=1 LIGER_FLCE=1 PER_DEVICE_BATCH=1 MIN_FREE_MIB=40000 \
  sbatch --parsable --export=ALL "${NODE[@]}" --job-name=sft_libext_e2 --dependency=afterok:${b} \
    scripts/slurm/jobs/gruenau_formal_mix_train_2026-09-25.slurm)
e=$(RUN_NAME=${RUN} sbatch --parsable --export=ALL "${NODE[@]}" --job-name=eval_libext_e2 --dependency=afterok:${t} \
  scripts/slurm/jobs/sft_eval_suite.slurm)
echo "build_ei2 ${b} sft_libext_e2 ${t} eval_libext_e2 ${e}"
