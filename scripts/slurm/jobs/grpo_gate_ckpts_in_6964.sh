#!/bin/bash
# Gate eval (rl_gate_dolci, greedy) of GRPO checkpoints, run inside job 6964's allocation
# (srun --jobid 6964 --overlap) on its third L40. Args: run:ckpt pairs (default: the first
# gate of 2026-09-28). Waits until each checkpoint's weights are written.
cd /vol/tmp2/laitenbf/synthetic-RLVL
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 MIN_FREE_MIB=20000 OUT_NAME=rl_gate_dolci
export TEST_JSONL=/vol/tmp2/laitenbf/rlvl_data/datasets/rl_gate_dolci_instruct_20260928/test.jsonl
export OUT_ROOT=/vol/tmp2/laitenbf/rlvl_data/grpo_formal_20260928
PAIRS=("$@")
(( ${#PAIRS[@]} )) || PAIRS=(2b_p50_G5_lines_bal:checkpoint-50 2b_p50_G1_correct_bal:checkpoint-100 2b_p0_G0_correct_bal:checkpoint-50)
for rc in "${PAIRS[@]}"; do
  d="${OUT_ROOT}/${rc%%:*}/${rc##*:}"
  until ls "${d}"/model*.safetensors >/dev/null 2>&1 && [[ -f "${d}/config.json" ]]; do sleep 60; done
  sleep 120  # let the save finish
  RUN_NAME=${rc%%:*} CKPT=${rc##*:} bash scripts/slurm/jobs/gruenau_formal_mix_tagged_2026-09-26.slurm
done
