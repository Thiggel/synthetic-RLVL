#!/usr/bin/env bash
#SBATCH --job-name=az_train
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:4
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/atuin/c107fa/c107fa12/rlvl_data/logs/%x_%j.out
#SBATCH --error=/home/atuin/c107fa/c107fa12/rlvl_data/logs/%x_%j.err

# alex (rtxpro6k, 96 GB GPUs, 1-day walltime) version of az_train.slurm: scripts/az/az_train.py data parallel under
# torchrun, one rank (own vLLM + bf16 model + CPU AdamW) per GPU. Env from alex_grpo_env.sh, checker = the libext
# snapshot (as on gruenau). Env: MODEL (path relative to RLVL_DATA_ROOT or absolute), RUN_NAME (out dir under
# ${RLVL_DATA_ROOT}/az), ARGS (extra az_train.py flags; --prompts-per-iter / --train-moves / --moves-per-step are
# global), GPU_MEM (vLLM share, default 0.45). Resumable: chain 1-day jobs with a unique --job-name and
# --dependency=singleton; each continues from OUT_DIR/latest. Submit with --export=ALL.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export RLVL_DATA_ROOT="${RLVL_DATA_ROOT:-/home/atuin/c107fa/c107fa12/rlvl_data}"
source scripts/slurm/jobs/alex_grpo_env.sh
S="${RLVL_DATA_ROOT}/checker_snapshot_libext_20261001"
export PYTHONPATH="$S/gen:$S/rlvl_python"
export RAYON_NUM_THREADS=8 OMP_NUM_THREADS=8
[[ "${MODEL:?}" = /* ]] || MODEL="${RLVL_DATA_ROOT}/${MODEL}"
OUT="${RLVL_DATA_ROOT}/az/${RUN_NAME:?}"
mkdir -p "${OUT}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then NGPU=$(tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES}" | wc -l)
else NGPU=$(nvidia-smi -L | wc -l); fi
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
END_TS=$(date -d "$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %e 2>/dev/null)" +%s 2>/dev/null || echo $(( $(date +%s) + 86400 )))
echo "model=${MODEL} ngpu=${NGPU} out=${OUT} host=$(hostname)"
"${VENV}/bin/python" -m torch.distributed.run --standalone --nproc_per_node "${NGPU}" scripts/az/az_train.py \
  --model "${MODEL}" --out-dir "${OUT}" --prompts "${RLVL_DATA_ROOT}/az/online_pool_gatematched/pool.jsonl" \
  --eval-jsonl "${RLVL_DATA_ROOT}/az/gate_subset_300.jsonl" --init-head "${RLVL_DATA_ROOT}/az/value_probe_e4/probe_cvf.pt" \
  --gpu-mem "${GPU_MEM:-0.45}" --stop-at $(( END_TS - 25 * 60 )) --resume ${ARGS:-}
