#!/usr/bin/env bash
#SBATCH --job-name=tl_eval
#SBATCH --partition=gpu-staff
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Decoding eval of a TL-extended model (RLVL-next/tl/eval_hf.py): guided and free decoding of held-out generator
# prompts, valid / correct / both. Env: MODEL, DATA (tl_eval.jsonl), OUT (dir; one subdir per mode),
# TL_ROOT (dir holding the tl package), MIN_FREE_MIB (default 20000), MODES (default "guided free"), N (default 500), BATCH (default 64).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
ROOT=/vol/tmp2/laitenbf
export PATH="${ROOT}/.venv_rlvl_tf5/bin:${PATH}" HF_HOME="${ROOT}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
: "${MODEL:?}" "${DATA:?}" "${OUT:?}" "${TL_ROOT:?}"
# Other users hold cards outside Slurm: use an allocated GPU with enough free memory.
FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v m="${MIN_FREE_MIB:-20000}" '$2 >= m {print $1}' | head -1)
[[ -n "${FREE}" ]] || { echo "no allocated GPU with ${MIN_FREE_MIB:-20000} MiB free" >&2; nvidia-smi; exit 3; }
export CUDA_VISIBLE_DEVICES="${FREE}"
cd "${TL_ROOT}"
for mode in ${MODES:-guided free}; do
  python -m tl.eval_hf --model "${MODEL}" --data "${DATA}" --out "${OUT}/${mode}" --mode "${mode}" \
    --n "${N:-500}" --batch "${BATCH:-64}"
done
