#!/usr/bin/env bash
# Continued-SFT arms c / l / e / le from the L1 base (scripts/data/build_libext_ei_mixture.py, 2026-09-30),
# each trained on one gruenau12 L40 and then evaluated by scripts/slurm/jobs/sft_eval_suite.slurm
# (Dolci gate greedy + k16, generator test, new-family test; new checker).
# Two serial lanes keep laitenbf within half of gruenau12's GPUs:
#   lane A (after LANE_A_DEP):  base-model suite (fills formal_eval_math) -> train l -> eval l -> train le -> eval le
#   lane B (after LANE_B_DEP):  train c -> eval c -> train e -> eval e
# Arms e and le need their mixture dirs built before their train jobs start (the GSM8K harvest must finish first).
# Usage: LANE_A_DEP=afterany:8176 LANE_B_DEP=afterany:7088 bash scripts/submit_libext_ei_arms.sh
set -euo pipefail
cd /vol/tmp2/laitenbf/synthetic-RLVL
DATA=/vol/tmp2/laitenbf/rlvl_data
BASE=qwen35_2b_p50_cont_lc_fp32m_lr5em6_seed3407
NODE=(-p gpu-staff --gres=gpu:l40:1 -w gruenau12 --cpus-per-task=16)

train() {  # arm dep -> job id
  MODEL_KEY=2b MIX=libext_$1 MIX_DIR=${DATA}/datasets/formal_libext_ei_20260930/$1 \
  INIT_MODEL=${DATA}/formal_mixture_sft_20260925/${BASE}/final RUN_NAME=qwen35_2b_lc_libext_$1_lr5em6_seed3407 \
  DS_CONFIG=configs/deepspeed/zero2_offload_optim.json GRAD_CKPT=1 LIGER_FLCE=1 PER_DEVICE_BATCH=1 MIN_FREE_MIB=40000 \
    sbatch --parsable --export=ALL "${NODE[@]}" --job-name=sft_libext_$1 --dependency=$2 \
      scripts/slurm/jobs/gruenau_formal_mix_train_2026-09-25.slurm
}
suite() {  # run dep -> job id
  RUN_NAME=$1 sbatch --parsable --export=ALL "${NODE[@]}" --job-name=eval_$2 --dependency=$3 \
    scripts/slurm/jobs/sft_eval_suite.slurm
}

a0=$(suite ${BASE} base ${LANE_A_DEP:?})
a1=$(train l afterany:${a0}); a2=$(suite qwen35_2b_lc_libext_l_lr5em6_seed3407 libext_l afterok:${a1})
a3=$(train le afterany:${a2}); a4=$(suite qwen35_2b_lc_libext_le_lr5em6_seed3407 libext_le afterok:${a3})
b1=$(train c ${LANE_B_DEP:?}); b2=$(suite qwen35_2b_lc_libext_c_lr5em6_seed3407 libext_c afterok:${b1})
b3=$(train e afterany:${b2}); b4=$(suite qwen35_2b_lc_libext_e_lr5em6_seed3407 libext_e afterok:${b3})
echo "lane A: base-suite ${a0} train_l ${a1} eval_l ${a2} train_le ${a3} eval_le ${a4}"
echo "lane B: train_c ${b1} eval_c ${b2} train_e ${b3} eval_e ${b4}"
