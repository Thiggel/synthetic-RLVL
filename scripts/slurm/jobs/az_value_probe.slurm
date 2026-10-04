#!/usr/bin/env bash
#SBATCH --job-name=az_value_probe
#SBATCH --partition=gpu-staff
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Stage 3 value probe (scripts/az/value_probe.py). ARGS: extra arguments.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
ROOT=/vol/tmp2/laitenbf
S=${ROOT}/rlvl_data/checker_snapshot_libext_20261001
export PATH="${ROOT}/.venv_rlvl_grpo/bin:${PATH}" HF_HOME="${ROOT}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${S}/gen:${S}/rlvl_python:scripts:scripts/analysis"
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=index,name,memory.free --format=csv,noheader
# gruenau12 GPUs are shared with foreign non-Slurm processes: wait for MIN_FREE_MIB (default 12000)
for a in $(seq 1 120); do
  f=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=memory.free --format=csv,noheader,nounits)
  (( f >= ${MIN_FREE_MIB:-12000} )) && break; echo "waiting for GPU memory: ${f} MiB free"; sleep 120
done
python scripts/az/value_probe.py ${ARGS:-}
