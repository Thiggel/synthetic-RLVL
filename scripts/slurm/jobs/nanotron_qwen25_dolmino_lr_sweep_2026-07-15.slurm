#!/usr/bin/env bash
#SBATCH --job-name=q25_dolmino_lr
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh
source "${WORK}/nanotron/bin/activate"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export WANDB_MODE="${WANDB_MODE:-offline}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}"
NORMAL_DATASET_FOLDER="${NORMAL_DATASET_FOLDER:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1/dolmino}"
PRETRAINED_NANOTRON_CKPT="${PRETRAINED_NANOTRON_CKPT:-${HPCVAULT}/synthetic-RLVL/nanotron_checkpoints/qwen25_7b_tp1}"
TRAIN_STEPS="${TRAIN_STEPS:-256}"
WARMUP_STEPS="${WARMUP_STEPS:-32}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TP="${TP:-4}"
DP="${DP:-2}"
PP="${PP:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
BATCH_ACCUMULATION="${BATCH_ACCUMULATION:-16}"
TOPOLOGY_TAG="${TOPOLOGY_TAG:-}"
GLOBAL_BATCH_SIZE=$((MICRO_BATCH_SIZE * BATCH_ACCUMULATION * DP))
if (( NPROC_PER_NODE != TP * DP * PP )); then
  echo "NPROC_PER_NODE must equal TP*DP*PP" >&2
  exit 1
fi
if (( GLOBAL_BATCH_SIZE != 128 )); then
  echo "LR gate requires a global batch of 128 sequences, got ${GLOBAL_BATCH_SIZE}" >&2
  exit 1
fi
LRS=(6.0e-6 3.0e-6 1.0e-5)
LR="${LRS[$SLURM_ARRAY_TASK_ID]}"
LR_TAG="$(printf '%s' "${LR}" | tr '.-' 'pm')"
TOPOLOGY_SUFFIX="${TOPOLOGY_TAG:+_${TOPOLOGY_TAG}}"
RUN_TAG="qwen25_7b_dolmino_lr${LR_TAG}_${TRAIN_STEPS}step${TOPOLOGY_SUFFIX}"
RUN_BASE_ROOT="${RUN_BASE_ROOT:-${HPCVAULT}/synthetic-RLVL/nanotron_lr_sweep}"
RUN_ROOT="${RUN_BASE_ROOT}/${RUN_TAG}"
CONFIG="${RUN_ROOT}/config.yaml"

test -f "${PRETRAINED_NANOTRON_CKPT}/model_config.json"
compgen -G "${NORMAL_DATASET_FOLDER}/*.metadata" >/dev/null
find "${NORMAL_DATASET_FOLDER}" -maxdepth 1 -type f -name '*.ds' -size +0 -print -quit | grep -q .
if [[ -f "${RUN_ROOT}/complete.json" && "${FORCE_NANOTRON_TRAIN:-0}" != "1" ]]; then
  echo "Sweep row already complete: ${RUN_ROOT}"
  exit 0
fi
rm -rf "${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"

cat > "${CONFIG}" <<YAML
checkpoints:
  checkpoint_interval: 1000000
  checkpoints_path: ${RUN_ROOT}/checkpoints
  checkpoints_path_is_shared_file_system: false
  load_lr_scheduler: false
  load_optimizer: false
  resume_checkpoint_path: ${PRETRAINED_NANOTRON_CKPT}
  save_final_state: false
  save_initial_state: false
data_stages:
- data:
    dataset:
      dataset_folder:
      - ${NORMAL_DATASET_FOLDER}
      dataset_weights:
      - 1.0
      token_size_in_bytes: 4
      tokenizer_name: ${MODEL_ID}
      vocab_size: 152064
      shuffle_files: true
    num_loading_workers: 4
    seed: 42
  name: qwen25_dolmino_lr${LR_TAG}
  start_training_step: 1
general:
  benchmark_csv_path: ${RUN_ROOT}/benchmark.csv
  consumed_train_samples: null
  ignore_sanity_checks: true
  project: synthetic-rlvl-midtrain
  run: ${RUN_TAG}
  seed: 42
  step: null
lighteval: null
logging:
  iteration_step_info_interval: 1
  log_level: info
  log_level_replica: error
model:
  ddp_bucket_cap_mb: 25
  dtype: bfloat16
  init_method:
    std: 0.02
  make_vocab_size_divisible_by: 1
  model_config:
    bos_token_id: 151643
    eos_token_id: 151645
    hidden_act: silu
    hidden_size: 3584
    initializer_range: 0.02
    intermediate_size: 18944
    is_qwen2_config: true
    max_position_embeddings: 131072
    num_attention_heads: 28
    num_hidden_layers: 28
    num_key_value_heads: 4
    pad_token_id: null
    pretraining_tp: 1
    rms_norm_eps: 1.0e-06
    rope_scaling: null
    rope_theta: 1000000.0
    tie_word_embeddings: false
    use_cache: true
    vocab_size: 152064
    attention_bias: true
    _attn_implementation: flash_attention_2
optimizer:
  accumulate_grad_in_fp32: true
  clip_grad: 1.0
  learning_rate_scheduler:
    learning_rate: ${LR}
    lr_decay_starting_step: $((TRAIN_STEPS + 1))
    lr_decay_steps: 1
    lr_decay_style: cosine
    lr_warmup_steps: ${WARMUP_STEPS}
    lr_warmup_style: linear
    min_decay_lr: 1.0e-6
  optimizer_factory:
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-8
    name: adamW
    torch_adam_is_fused: true
  weight_decay: 0.01
  zero_stage: 0
parallelism:
  dp: ${DP}
  expert_parallel_size: 1
  pp: ${PP}
  pp_engine: 1f1b
  tp: ${TP}
  tp_linear_async_communication: true
  tp_mode: REDUCE_SCATTER
profiler: null
tokenizer:
  tokenizer_max_length: null
  tokenizer_name_or_path: ${MODEL_ID}
  tokenizer_revision: null
tokens:
  batch_accumulation_per_replica: ${BATCH_ACCUMULATION}
  limit_test_batches: 0
  limit_val_batches: 0
  micro_batch_size: ${MICRO_BATCH_SIZE}
  sequence_length: 4096
  train_steps: ${TRAIN_STEPS}
  val_check_interval: -1
YAML

if [[ "${NANOTRON_CONFIG_ONLY:-0}" == "1" ]]; then
  echo "Generated config only: ${CONFIG}"
  exit 0
fi

python - <<'PY'
import inspect
from nanotron.helpers import lr_scheduler_builder

source = inspect.getsource(lr_scheduler_builder)
if 'param_group.get("initial_lr", param_group["lr"])' not in source:
    raise SystemExit("Nanotron checkout lacks the LR-scheduler resume fix")
PY

cd ../nanotron
torchrun --nproc_per_node="${NPROC_PER_NODE}" run_train.py --config-file "${CONFIG}"
cd "${SLURM_SUBMIT_DIR}"

LR="${LR}" TRAIN_STEPS="${TRAIN_STEPS}" RUN_ROOT="${RUN_ROOT}" \
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" TP="${TP}" DP="${DP}" PP="${PP}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
payload = {
    "learning_rate": float(os.environ["LR"]),
    "train_steps": int(os.environ["TRAIN_STEPS"]),
    "nproc_per_node": int(os.environ["NPROC_PER_NODE"]),
    "tp": int(os.environ["TP"]),
    "dp": int(os.environ["DP"]),
    "pp": int(os.environ["PP"]),
    "global_batch_size": int(os.environ["GLOBAL_BATCH_SIZE"]),
    "scheduled_tokens": int(os.environ["TRAIN_STEPS"])
    * int(os.environ["GLOBAL_BATCH_SIZE"])
    * 4096,
    "config": str(root / "config.yaml"),
    "benchmark_csv": str(root / "benchmark.csv"),
}
(root / "complete.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
