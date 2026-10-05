#!/usr/bin/env bash
#SBATCH --job-name=az_train
#SBATCH --partition=gpu-staff
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Stage 3 online AlphaZero (scripts/az/az_train.py): MCTS self-play + joint policy/value-head training, one GPU.
# Env: MODEL (init checkpoint), OUT_DIR, PROMPTS, ARGS (extra az_train.py flags), GPU_MEM (vLLM share, default 0.28).
# Resumable: a chained job with the same name (--dependency=singleton) continues from OUT_DIR/latest.
set -euo pipefail
cd /vol/tmp2/laitenbf/synthetic-RLVL
ROOT=/vol/tmp2/laitenbf
VENV="${ROOT}/.venv_rlvl_grpo"
export PATH="${VENV}/bin:${PATH}" VIRTUAL_ENV="${VENV}" HF_HOME="${ROOT}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"; export PATH="${CUDA_HOME}/bin:${PATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0 RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
S=${ROOT}/rlvl_data/checker_snapshot_libext_20261001
export PYTHONPATH="${RLVL_PYTHONPATH:-$S/gen:$S/rlvl_python}"
hash -r 2>/dev/null || true
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
END_TS=$(( $(date +%s) + $(squeue -h -j "${SLURM_JOB_ID}" -o %L | awk -F'[-:]' '{ if (NF==4) print $1*86400+$2*3600+$3*60+$4; else if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2 }') ))
python scripts/az/az_train.py --model "${MODEL}" --out-dir "${OUT_DIR}" --prompts "${PROMPTS}" \
  --gpu-mem "${GPU_MEM:-0.28}" --stop-at $(( END_TS - 20 * 60 )) --resume ${ARGS:-}
