#!/usr/bin/env bash
# EI round 2 ablation e2s (2026-10-02): e2's harvest cut to e's size (same prompts per bench, same EI rows per
# bench; scripts/data/build_libext_ei_mixture.py --arms e2s), so e2s - e is teacher quality at fixed quantity and
# e2 - e2s is quantity. Chain as scripts/submit_ei2.sh, but on gruenau7 (no gruenau12 L40 had 40 GB free):
# 2 of its 4 A6000s (half-node cap) because Slurm hands out IDX 0 first and other users hold ~40 GB on 0 and 2;
# the train script uses only cards with MIN_FREE_MIB free, the eval wrapper the card with the most free memory.
set -euo pipefail
cd /vol/tmp2/laitenbf/synthetic-RLVL
DATA=/vol/tmp2/laitenbf/rlvl_data
BASE=qwen35_2b_p50_cont_lc_fp32m_lr5em6_seed3407
NODE=(-p gpu --gres=gpu:rtxa6000:2 -w gruenau7 --cpus-per-task=16)  # the train script sets --mem-per-cpu=8G
RUN=qwen35_2b_lc_libext_e2s_lr5em6_seed3407

b=${BUILD_JOB:-}
[[ -n ${b} ]] || b=$(sbatch --parsable --export=ALL -p compute --cpus-per-task=8 --mem=64G --time=04:00:00 --job-name=build_ei2s \
  --output=logs/%x_%j.out \
  --wrap "cd /vol/tmp2/laitenbf/synthetic-RLVL && HF_HOME=/vol/tmp2/laitenbf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
/vol/tmp2/laitenbf/.venv_rlvl_grpo/bin/python scripts/data/build_libext_ei_mixture.py --arms e2s")
t=$(MODEL_KEY=2b MIX=libext_e2s MIX_DIR=${DATA}/datasets/formal_libext_ei_20260930/e2s \
  INIT_MODEL=${DATA}/formal_mixture_sft_20260925/${BASE}/final RUN_NAME=${RUN} \
  DS_CONFIG=configs/deepspeed/zero2_offload_optim.json GRAD_CKPT=1 LIGER_FLCE=1 PER_DEVICE_BATCH=1 MIN_FREE_MIB=40000 \
  sbatch --parsable --export=ALL "${NODE[@]}" --job-name=sft_libext_e2s --dependency=afterok:${b} \
    scripts/slurm/jobs/gruenau_formal_mix_train_2026-09-25.slurm)
e=$(sbatch --parsable --export=ALL "${NODE[@]}" --mem=128G --time=1-00:00:00 --job-name=eval_libext_e2s --dependency=afterok:${t} \
  --output=logs/%x_%j.out --error=logs/%x_%j.err \
  --wrap "cd /vol/tmp2/laitenbf/synthetic-RLVL && export CUDA_VISIBLE_DEVICES=\$(nvidia-smi --query-gpu=index,memory.free \
--format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1) && RUN_NAME=${RUN} bash scripts/slurm/jobs/sft_eval_suite.slurm")
echo "build_ei2s ${b} sft_libext_e2s ${t} eval_libext_e2s ${e}"
