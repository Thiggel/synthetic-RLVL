#!/usr/bin/env bash
#SBATCH --job-name=az_mcts
#SBATCH --partition=gpu-staff
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Stage 3 line-level PUCT search (scripts/az/mcts_decode.py). Env: MODEL, OUT_DIR, ARGS, RESERVE_MIB (GPU memory left for the HF value model, default 10000).
# Env: MODEL (checkpoint dir), OUT_DIR, ARGS (extra az_mcts.py flags, e.g. "--repair --limit 40").
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
MIN_FREE_MIB="${MIN_FREE_MIB:-24000}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
for a in $(seq 1 90); do
  read -r free_mib total_mib < <(nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=memory.free,memory.total --format=csv,noheader,nounits | tr -d ',')
  (( free_mib >= MIN_FREE_MIB )) && break
  echo "waiting for GPU memory: ${free_mib} MiB free"; sleep 120
done
UTIL=$(python3 -c "print(round(min(0.85, (${free_mib}-3072-${RESERVE_MIB:-10000})/${total_mib}), 3))")
echo "gpu_mem=${UTIL} model=${MODEL} out=${OUT_DIR}"
python scripts/az/mcts_decode.py --model "${MODEL}" --out-dir "${OUT_DIR}" --gpu-mem "${UTIL}" ${ARGS:-}
