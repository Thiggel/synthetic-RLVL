#!/bin/bash
# runs inside job 6964's allocation (srun --overlap) on its third L40
cd /vol/tmp2/laitenbf/synthetic-RLVL
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 MIN_FREE_MIB=20000 OUT_NAME=rl_gate_dolci
export TEST_JSONL=/vol/tmp2/laitenbf/rlvl_data/datasets/rl_gate_dolci_instruct_20260928/test.jsonl
export OUT_ROOT=/vol/tmp2/laitenbf/rlvl_data/grpo_formal_20260928
for rc in 2b_p50_G5_lines_bal:checkpoint-50 2b_p50_G1_correct_bal:checkpoint-100 2b_p0_G0_correct_bal:checkpoint-50; do
  RUN_NAME=${rc%%:*} CKPT=${rc##*:} bash scripts/slurm/jobs/gruenau_formal_mix_tagged_2026-09-26.slurm
done
