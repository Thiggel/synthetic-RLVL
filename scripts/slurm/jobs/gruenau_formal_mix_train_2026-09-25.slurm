#!/usr/bin/env bash
#SBATCH --job-name=gru_fmix_train
#SBATCH --partition=gpu-staff
#SBATCH --gres=gpu:h100nvl:4
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Formal-CoT data-mixture sweep (2026-09-25): full-parameter SFT of a
# Qwen3.5 base model on one Dolci + rlvlgen mixture. One job trains one
# (model, percent) cell; scripts/submit_formal_mixture_sweep.sh submits the
# grid and chains an eval job (gruenau_formal_mix_eval_2026-09-25.slurm) on
# each with afterok.
#
# Recipe: Dolci-Instruct-SFT-No-Tools 100k (seed 3407) with X percent of the
# rows replaced by rlvlgen proofs, lr 5e-6 linear, warmup 3%, global batch
# 128, one epoch, max length 4096, ChatML, loss on assistant tokens, tool
# result contents masked. Trainer: scripts/train_formal_mixture_sft.py in
# .venv_rlvl_tf5 (transformers 5.17, torch 2.9, fla 0.5.2).
#
# Required env: MODEL_KEY (0.8b | 2b | 9b), MIX (mixture dir name, e.g.
# dolci_rlvlgen_p25 or smoke2k_p25). Optional: PER_DEVICE_BATCH, FSDP,
# GRAD_CKPT=1, MAX_STEPS, EVAL_ROWS, RUN_TAG, OUT_ROOT.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

ROOT=/vol/tmp2/laitenbf
VENV="${ROOT}/.venv_rlvl_tf5"
DATA="${ROOT}/rlvl_data"
export PATH="${VENV}/bin:${PATH}"
export VIRTUAL_ENV="${VENV}"
export HF_HOME="${ROOT}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
# DeepSpeedCPUAdam is JIT-built per node on first use. It is a CPU kernel, so
# skip DeepSpeed's torch-CUDA (12.8) vs CUDA_HOME (13.2, needed by fla) check.
export DS_SKIP_CUDA_CHECK=1
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${ROOT}/torch_extensions/$(hostname -s)}"
# fla 0.5.2 refuses the gated-delta backward on Hopper with Triton 3.5 (wrong
# results, fla #640) and routes it to tilelang, which JIT-compiles with nvcc.
# gruenau11 has no /usr/local/cuda symlink, and CUDA 12.x nvcc rejects the
# system gcc 15, so point at a CUDA 13 toolkit explicitly.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${DATA}/wandb"
hash -r 2>/dev/null || true

: "${MODEL_KEY:?set MODEL_KEY=0.8b|2b|9b}"
: "${MIX:?set MIX=<mixture dir name>}"
declare -A SNAP=(
  [0.8b]="${ROOT}/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
  [2b]="${ROOT}/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
  [9b]="${ROOT}/hub/models--Qwen--Qwen3.5-9B-Base/snapshots/68c46c4b3498877f3ef123c856ecfde50c39f404"
)
# 9B does not fit replicated (DDP) with Adam states on 94 GB; shard it.
# 9B: fp32 master weights + grads + AdamW state are ~150 GB, which does not fit
# FSDP-sharded on the 2 usable 80 GB H100s (jobs 6789/6804 OOM at the first
# optimizer step). DeepSpeed ZeRO-2 keeps bf16 weights/grads on the GPUs and
# the fp32 master weights + AdamW (DeepSpeedCPUAdam) in host RAM; same math.
declare -A DEF_FSDP=([0.8b]="" [2b]="" [9b]="")
declare -A DEF_DS=([0.8b]="" [2b]="" [9b]="configs/deepspeed/zero2_offload_optim.json")
# Per-device batch is bounded by the 248k-vocab logits at a 4096-token row
# (~23 GiB per row incl. fp32 CE): measured peaks on one H100 NVL with a full
# 4096-token microbatch are 0.8B x2 = 50.7 GiB, 2B x2 = 64.5 GiB; 0.8B x4 OOMs
# even with gradient checkpointing.
declare -A DEF_PDB=([0.8b]=2 [2b]=2 [9b]=2)
declare -A DEF_GC=([0.8b]=0 [2b]=0 [9b]=1)
# 9B: liger fused linear CE so the 4096 x 248k logits are never materialised.
declare -A DEF_LIGER=([0.8b]=0 [2b]=0 [9b]=1)
MODEL="${SNAP[$MODEL_KEY]}"
MIX_DIR="${DATA}/datasets/formal_mixture_20260925/mixtures/${MIX}"
SEED="${SEED:-3407}"
RUN_NAME="qwen35_${MODEL_KEY}_${MIX}_lr5em6_seed${SEED}${RUN_TAG:-}"
OUT_ROOT="${OUT_ROOT:-${DATA}/formal_mixture_sft_20260925}"
OUT_DIR="${OUT_ROOT}/${RUN_NAME}"

test -s "${MODEL}/config.json" || { echo "missing model: ${MODEL}" >&2; exit 2; }
test -s "${MIX_DIR}/mixture_manifest.json" || { echo "missing mixture: ${MIX_DIR}" >&2; exit 2; }
if [[ -s "${OUT_DIR}/final/config.json" ]]; then
  echo "complete run already exists: ${OUT_DIR}/final; skipping"; exit 0
fi

# gruenau11's two H100 PCIe cards are held outside Slurm by another user's
# long-running vLLM server (~72 GB each, seen 2026-09-25), and Slurm hands them
# out anyway. Train only on the allocated GPUs that are actually free.
MIN_FREE_MIB="${MIN_FREE_MIB:-60000}"
# If none is free, wait (the held cards are busy anyway) instead of failing:
# a quick failure would let the next job of a serial lane start and fail too.
# MIN_NGPU: 9B needs both cards.
declare -A DEF_MIN_NGPU=([0.8b]=1 [2b]=1 [9b]=2)
MIN_NGPU="${MIN_NGPU:-${DEF_MIN_NGPU[$MODEL_KEY]}}"
WAIT_MAX_S="${WAIT_MAX_S:-21600}"; waited=0
while :; do
  FREE_IDX=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', ' -v m="${MIN_FREE_MIB}" '$2 >= m {print $1}' | paste -sd, -)
  [[ -n "${FREE_IDX}" ]] && (( $(awk -F, '{print NF}' <<<"${FREE_IDX}") >= MIN_NGPU )) && break
  if (( waited >= WAIT_MAX_S )); then
    echo "no allocated GPU has ${MIN_FREE_MIB} MiB free after ${waited}s" >&2; nvidia-smi; exit 3
  fi
  echo "fewer than ${MIN_NGPU} allocated GPUs have ${MIN_FREE_MIB} MiB free; waiting (${waited}s)" >&2
  sleep 300; waited=$(( waited + 300 ))
done
export CUDA_VISIBLE_DEVICES="${FREE_IDX}"
NGPU=$(awk -F, '{print NF}' <<<"${FREE_IDX}")
echo "usable GPUs: ${FREE_IDX}"
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-${DEF_PDB[$MODEL_KEY]}}"
FSDP="${FSDP-${DEF_FSDP[$MODEL_KEY]}}"
DENOM=$(( NGPU * PER_DEVICE_BATCH ))
if (( GLOBAL_BATCH % DENOM != 0 )); then
  echo "global batch ${GLOBAL_BATCH} not divisible by ${NGPU} GPUs x ${PER_DEVICE_BATCH}" >&2; exit 2
fi
GRAD_ACCUM=$(( GLOBAL_BATCH / DENOM ))
echo "model=${MODEL_KEY} mix=${MIX} gpus=${NGPU} per_device=${PER_DEVICE_BATCH} accum=${GRAD_ACCUM} fsdp='${FSDP}' out=${OUT_DIR}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

EXTRA=()
[[ -n "${FSDP}" ]] && EXTRA+=(--fsdp "${FSDP}")
DS_CONFIG="${DS_CONFIG-${DEF_DS[$MODEL_KEY]}}"
[[ -n "${DS_CONFIG}" ]] && EXTRA+=(--deepspeed "${DS_CONFIG}")
[[ "${GRAD_CKPT:-${DEF_GC[$MODEL_KEY]}}" == 1 ]] && EXTRA+=(--gradient-checkpointing)
[[ -n "${EVAL_ROWS:-}" ]] && EXTRA+=(--eval-rows "${EVAL_ROWS}")
[[ "${LIGER_FLCE:-${DEF_LIGER[$MODEL_KEY]}}" == 1 ]] && EXTRA+=(--liger-flce)
echo "extra args: ${EXTRA[*]:-}"

mkdir -p "${OUT_DIR}"
python -m torch.distributed.run --standalone --nproc_per_node="${NGPU}" \
  scripts/train_formal_mixture_sft.py \
  --model "${MODEL}" \
  --prepared-dataset "${MIX_DIR}" \
  --output-dir "${OUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --seed "${SEED}" \
  --max-length "${MAX_LENGTH:-4096}" \
  --max-steps "${MAX_STEPS:--1}" \
  --lr "${LR:-5e-6}" \
  --per-device-batch-size "${PER_DEVICE_BATCH}" \
  --per-device-eval-batch-size "${PER_DEVICE_BATCH}" \
  --grad-accum "${GRAD_ACCUM}" \
  --logging-steps "${LOGGING_STEPS:-5}" \
  --eval-steps "${EVAL_STEPS:-250}" \
  --save-steps "${SAVE_STEPS:-250}" \
  --audit-output "${OUT_DIR}/tokenization_audit.json" \
  --max-truncated-frac "${MAX_TRUNCATED_FRAC:-0.01}" \
  --report-to "${REPORT_TO:-none}" \
  "${EXTRA[@]}"

test -s "${OUT_DIR}/final/config.json"
du -sh "${OUT_DIR}/final"
echo "done ${RUN_NAME}"
