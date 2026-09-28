#!/usr/bin/env bash
#SBATCH --job-name=grpo_formal_multi
#SBATCH --partition=gpu-staff
#SBATCH --gres=gpu:l40:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Several single-GPU GRPO arms in one allocation, each on one of the allocated GPUs
# with the most free memory. The gruenau nodes carry non-Slurm processes of other users,
# and Slurm cannot pick a device index, so allocate one spare GPU and use the free ones.
# Env: ARMS="lines gvc" RUN_NAMES="a b" POLICY, EXTRA_ARGS (as grpo_formal.slurm), optional NO_TAG.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
read -ra arms <<< "${ARMS:?}"; read -ra names <<< "${RUN_NAMES:?}"
mapfile -t gpus < <(nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | sort -t, -k2 -nr | cut -d, -f1 | head -n "${#arms[@]}")
echo "allocated=${CUDA_VISIBLE_DEVICES} using=${gpus[*]}"
pids=()
for i in "${!arms[@]}"; do
  CUDA_VISIBLE_DEVICES="${gpus[$i]// /}" ARM="${arms[$i]}" RUN_NAME="${names[$i]}" \
    bash scripts/slurm/jobs/grpo_formal.slurm > "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${arms[$i]}.log" 2>&1 &
  pids+=($!)
  sleep 90  # stagger vLLM start-up
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done; exit $rc
