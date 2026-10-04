#!/usr/bin/env bash
# EI round 5 (2026-10-04, Stage 3 decision): self-distillation from the e4 student's own 32 samples on the round-1
# self-play prompts (sampling beat MCTS at matched tokens, analysis/az_vs_sampling.md). Harvests already exist
# (az/selfplay_r1/sample_e4_n32{,_dolci}.json.passing.jsonl); build e5 (CPU) -> continued SFT from the L1 base on one
# gruenau12 L40 -> eval suite. HARVEST_DEP: dependency on the Dolci sampling job.
set -euo pipefail
cd /vol/tmp2/laitenbf/synthetic-RLVL
DATA=/vol/tmp2/laitenbf/rlvl_data
BASE=qwen35_2b_p50_cont_lc_fp32m_lr5em6_seed3407
NODE=(-p gpu-staff --gres=gpu:l40:1 -w gruenau12 --cpus-per-task=16)
RUN=qwen35_2b_lc_libext_e5_lr5em6_seed3407
b=$(sbatch --parsable --export=ALL -p compute --cpus-per-task=8 --mem=64G --time=04:00:00 --job-name=build_ei5 \
  --output=logs/%x_%j.out ${HARVEST_DEP:+--dependency=afterok:${HARVEST_DEP}} \
  --wrap "cd /vol/tmp2/laitenbf/synthetic-RLVL && HF_HOME=/vol/tmp2/laitenbf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
/vol/tmp2/laitenbf/.venv_rlvl_grpo/bin/python scripts/data/build_libext_ei_mixture.py --arms e5")
t=$(MODEL_KEY=2b MIX=libext_e5 MIX_DIR=${DATA}/datasets/formal_libext_ei_20260930/e5 \
  INIT_MODEL=${DATA}/formal_mixture_sft_20260925/${BASE}/final RUN_NAME=${RUN} \
  DS_CONFIG=configs/deepspeed/zero2_offload_optim.json GRAD_CKPT=1 LIGER_FLCE=1 PER_DEVICE_BATCH=1 MIN_FREE_MIB=40000 \
  sbatch --parsable --export=ALL "${NODE[@]}" --job-name=sft_libext_e5 --dependency=afterok:${b} \
    scripts/slurm/jobs/gruenau_formal_mix_train_2026-09-25.slurm)
e=$(RUN_NAME=${RUN} sbatch --parsable --export=ALL "${NODE[@]}" --job-name=eval_libext_e5 --dependency=afterok:${t} \
  scripts/slurm/jobs/sft_eval_suite.slurm)
echo "build_ei5 ${b} sft_libext_e5 ${t} eval_libext_e5 ${e}"
