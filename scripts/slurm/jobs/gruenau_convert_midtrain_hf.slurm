#!/usr/bin/env bash
#SBATCH --job-name=gru_convert_hf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Convert the midtrained nanotron checkpoints to Hugging Face format on
# gruenau, where the A100 nodes are idle, instead of waiting for alex to fall
# below its account GPU cap.
#
# nanotron imports flash-attention at module scope, and building the real
# library takes hours, so this environment carries a conversion-only stub whose
# every symbol raises when called. Weight conversion only remaps tensors and
# never runs attention, so the stub is never reached; if it ever were, the job
# would crash rather than emit wrong weights. The same conversion is queued on
# alex against the real library, and the two outputs get compared before
# anything downstream uses them.
#
# Array index: 0 control, 1 logic_band25, 2 nl_exact_band25.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

ROOT=/vol/home-vol2/ml/laitenbf
VENV="${ROOT}/.venv_rlvl_gruenau"
DATA="${ROOT}/rlvl_data"
export PATH="${VENV}/bin:${PATH}"
export VIRTUAL_ENV="${VENV}"
export PYTHONPATH="${ROOT}/nanotron_src:${PWD}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${DATA}/hf_home}"
hash -r 2>/dev/null || true

ARMS=(control logic_band25 nl_exact_band25)
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
SRC="${DATA}/nanotron_ckpt/${ARM}"
OUT="${DATA}/hf_midtrain/qwen25_7b_longwin_${ARM}_2p5b_hf"

# fail closed: the converter reads model_config.json and the model/ shards, so
# refuse to start on a transfer that has not finished
test -s "${SRC}/model_config.json" || { echo "missing model_config.json in ${SRC}" >&2; exit 2; }
test -d "${SRC}/model" || { echo "missing model/ in ${SRC}" >&2; exit 2; }
if [[ -s "${OUT}/config.json" ]]; then
  echo "already converted: ${OUT}; skipping"
  exit 0
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "converting ${ARM}: ${SRC} -> ${OUT}"

mkdir -p "$(dirname "${OUT}")"
rm -rf "${OUT}.tmp"
# nanotron reads WORLD_SIZE and friends at import, so the converter must be
# launched through torchrun even for a single process, as the accepted alex
# wrapper does; a plain python call dies with KeyError: 'WORLD_SIZE'.
export NANOTRON_ROOT="${ROOT}/nanotron_src"
torchrun --standalone --nproc_per_node=1 scripts/nanotron/convert_qwen2_nanotron_to_hf.py \
  --checkpoint-path "${SRC}" \
  --save-path "${OUT}.tmp" \
  --tokenizer-name "Qwen/Qwen2.5-7B"

test -s "${OUT}.tmp/config.json"
python scripts/nanotron/verify_qwen2_hf_checkpoint.py \
  --checkpoint "${OUT}.tmp" \
  --output "${OUT}.tmp/conversion_audit.json"

mv "${OUT}.tmp" "${OUT}"
du -sh "${OUT}"
echo "converted ${ARM}"
