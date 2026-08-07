#!/usr/bin/env bash
#SBATCH --job-name=nano_q25_midtrain
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --array=0-10%1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

source "${WORK}/nanotron/bin/activate"

if [[ "${NANOTRON_CONFIG_ONLY:-0}" != "1" ]]; then
python - <<'PY'
import inspect

from nanotron.helpers import lr_scheduler_builder

source = inspect.getsource(lr_scheduler_builder)
if 'param_group.get("initial_lr", param_group["lr"])' not in source:
    raise SystemExit("Nanotron checkout lacks the LR-scheduler resume fix")
PY
fi

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}"
SEQ_LEN="${SEQ_LEN:-4096}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
TRAIN_STEPS="${TRAIN_STEPS:-8192}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1024}"
LR="${LR:-1.0e-5}"
MIN_LR="${MIN_LR:-1.0e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-256}"
LR_DECAY_STARTING_STEP="${LR_DECAY_STARTING_STEP:-null}"
LR_DECAY_STEPS="${LR_DECAY_STEPS:-null}"

NORMAL_DATASET_FOLDER="${NORMAL_DATASET_FOLDER:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25/normal_continuation}"
LOGIC_DATASET_FOLDER="${LOGIC_DATASET_FOLDER:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25/logical_deduction_logic}"
NL_DATASET_FOLDER="${NL_DATASET_FOLDER:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25/logical_deduction_nl_exact}"
PRETRAINED_NANOTRON_CKPT="${PRETRAINED_NANOTRON_CKPT:-${HPCVAULT}/synthetic-RLVL/nanotron_checkpoints/qwen25_7b_tp1}"

MIX_KINDS=(control logic logic logic logic logic nl_exact nl_exact nl_exact nl_exact nl_exact)
MIX_PCTS=(0 5 10 15 20 25 5 10 15 20 25)

IDX="${SLURM_ARRAY_TASK_ID}"
MIX_KIND="${MIX_KINDS[$IDX]}"
MIX_PCT="${MIX_PCTS[$IDX]}"
RUN_TAG="${RUN_TAG_OVERRIDE:-qwen25_7b_midtrain_${MIX_KIND}_p${MIX_PCT}_4p3b}"
RUN_ROOT="${RUN_ROOT_OVERRIDE:-${HPCVAULT}/synthetic-RLVL/nanotron_midtrain/${RUN_TAG}}"
CONFIG="${RUN_ROOT}/config.yaml"

require_nanoset() {
  local path="$1"
  if ! compgen -G "${path}/*.metadata" >/dev/null; then
    echo "Missing pretokenized Nanoset metadata under ${path}" >&2
    echo "Set NORMAL_DATASET_FOLDER/LOGIC_DATASET_FOLDER/NL_DATASET_FOLDER before releasing this job." >&2
    exit 2
  fi
}

checkpoint_is_complete() {
  local checkpoint_dir="$1"
  local expected_optimizer_shards="${EXPECTED_OPTIMIZER_SHARDS:-4}"
  local min_optimizer_shard_bytes="${MIN_OPTIMIZER_SHARD_BYTES:-20000000000}"
  local optimizer_sizes=()
  local size

  [[ -s "${checkpoint_dir}/checkpoint_metadata.json" ]] || return 1
  [[ -s "${checkpoint_dir}/config.yaml" ]] || return 1
  [[ -s "${checkpoint_dir}/model_config.json" ]] || return 1
  if find "${checkpoint_dir}" -type f -size 0 -print -quit | grep -q .; then
    return 1
  fi

  mapfile -t optimizer_sizes < <(
    find "${checkpoint_dir}/optimizer" -maxdepth 1 -type f -name 'optimizer_*.pt' -printf '%s\n' | sort -n
  )
  [[ "${#optimizer_sizes[@]}" -eq "${expected_optimizer_shards}" ]] || return 1
  [[ "${optimizer_sizes[0]}" -ge "${min_optimizer_shard_bytes}" ]] || return 1
  for size in "${optimizer_sizes[@]}"; do
    [[ "${size}" -eq "${optimizer_sizes[0]}" ]] || return 1
  done
}

RESUME_CKPT="${PRETRAINED_NANOTRON_CKPT}"
LOAD_OPTIMIZER="${LOAD_OPTIMIZER:-false}"
LOAD_LR_SCHEDULER="${LOAD_LR_SCHEDULER:-false}"
if compgen -G "${RUN_ROOT}/checkpoints/[0-9]*" >/dev/null; then
  LATEST_STEP=""
  while IFS= read -r CANDIDATE_STEP; do
    if checkpoint_is_complete "${RUN_ROOT}/checkpoints/${CANDIDATE_STEP}"; then
      LATEST_STEP="${CANDIDATE_STEP}"
      break
    fi
    echo "Ignoring incomplete run checkpoint: ${RUN_ROOT}/checkpoints/${CANDIDATE_STEP}" >&2
  done < <(find "${RUN_ROOT}/checkpoints" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | rg '^[0-9]+$' | sort -nr)
  if [[ -n "${LATEST_STEP}" ]]; then
    RESUME_CKPT="${RUN_ROOT}/checkpoints/${LATEST_STEP}"
    LOAD_OPTIMIZER="${LOAD_OPTIMIZER_FOR_RUN_CKPT:-true}"
    LOAD_LR_SCHEDULER="${LOAD_LR_SCHEDULER_FOR_RUN_CKPT:-true}"
    if [[ "${PRUNE_SUPERSEDED_RUN_CHECKPOINTS:-0}" == "1" ]]; then
      while IFS= read -r CANDIDATE_STEP; do
        if [[ "${CANDIDATE_STEP}" -lt "${LATEST_STEP}" ]] &&
          checkpoint_is_complete "${RUN_ROOT}/checkpoints/${CANDIDATE_STEP}"; then
          echo "Removing superseded complete run checkpoint: ${RUN_ROOT}/checkpoints/${CANDIDATE_STEP}"
          find "${RUN_ROOT}/checkpoints/${CANDIDATE_STEP}" -depth -delete
        fi
      done < <(
        find "${RUN_ROOT}/checkpoints" -maxdepth 1 -mindepth 1 -type d \
          -printf '%f\n' | rg '^[0-9]+$' | sort -n
      )
    fi
  fi
fi

if [[ ! -f "${RESUME_CKPT}/model_config.json" ]]; then
  echo "Missing pretrained Nanotron checkpoint: ${PRETRAINED_NANOTRON_CKPT}" >&2
  echo "Resolved resume checkpoint was: ${RESUME_CKPT}" >&2
  echo "Run/fix the Qwen2.5 HF-to-Nanotron conversion smoke before releasing this grid." >&2
  exit 2
fi
require_nanoset "${NORMAL_DATASET_FOLDER}"
if [[ "${MIX_KIND}" == "logic" ]]; then
  PROOF_DATASET_FOLDER="${LOGIC_DATASET_FOLDER}"
  require_nanoset "${PROOF_DATASET_FOLDER}"
elif [[ "${MIX_KIND}" == "nl_exact" ]]; then
  PROOF_DATASET_FOLDER="${NL_DATASET_FOLDER}"
  require_nanoset "${PROOF_DATASET_FOLDER}"
else
  PROOF_DATASET_FOLDER=""
fi

if checkpoint_is_complete "${RUN_ROOT}/checkpoints/${TRAIN_STEPS}" && [[ "${FORCE_NANOTRON_TRAIN:-0}" != "1" ]]; then
  echo "Final checkpoint exists for ${RUN_TAG}; skipping"
  exit 0
fi

mkdir -p "${RUN_ROOT}"

if [[ "${MIX_KIND}" == "control" ]]; then
  DATASET_YAML="      dataset_folder:
      - ${NORMAL_DATASET_FOLDER}
      dataset_weights:
      - 1.0"
else
  if [[ -n "${PROOF_WEIGHT_OVERRIDE:-}" ]]; then
    # Document-preserving packed synthetic windows carry loss-masked padding;
    # the override is the padding-compensated weight from the mixture audit.
    PROOF_WEIGHT=$(python - <<PY
w = float("${PROOF_WEIGHT_OVERRIDE}")
assert 0.0 < w < 1.0
print(f"{w:.8f}")
PY
)
    NORMAL_WEIGHT=$(python - <<PY
w = float("${PROOF_WEIGHT_OVERRIDE}")
print(f"{1.0 - w:.8f}")
PY
)
  else
    NORMAL_WEIGHT=$(python - <<PY
p = int("${MIX_PCT}") / 100.0
print(f"{1.0 - p:.6f}")
PY
)
    PROOF_WEIGHT=$(python - <<PY
p = int("${MIX_PCT}") / 100.0
print(f"{p:.6f}")
PY
)
  fi
  DATASET_YAML="      dataset_folder:
      - ${NORMAL_DATASET_FOLDER}
      - ${PROOF_DATASET_FOLDER}
      dataset_weights:
      - ${NORMAL_WEIGHT}
      - ${PROOF_WEIGHT}"
fi
if [[ -n "${NANOSET_PADDING_LABEL_ID:-}" ]]; then
  # Loss-mask padding labels in document-preserving packed windows.
  DATASET_YAML="${DATASET_YAML}
      padding_label_id: ${NANOSET_PADDING_LABEL_ID}"
fi

cat > "${CONFIG}" <<YAML
checkpoints:
  checkpoint_interval: ${CHECKPOINT_INTERVAL}
  checkpoints_path: ${RUN_ROOT}/checkpoints
  checkpoints_path_is_shared_file_system: false
  load_lr_scheduler: ${LOAD_LR_SCHEDULER}
  load_optimizer: ${LOAD_OPTIMIZER}
  resume_checkpoint_path: ${RESUME_CKPT}
  save_final_state: true
  save_initial_state: false
data_stages:
- data:
    dataset:
${DATASET_YAML}
      token_size_in_bytes: 4
      tokenizer_name: ${MODEL_ID}
      vocab_size: 152064
      shuffle_files: true
    num_loading_workers: 4
    seed: 42
  name: qwen25_midtrain_${MIX_KIND}_p${MIX_PCT}
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
  iteration_step_info_interval: 10
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
    lr_decay_starting_step: ${LR_DECAY_STARTING_STEP}
    lr_decay_steps: ${LR_DECAY_STEPS}
    lr_decay_style: cosine
    lr_warmup_steps: ${WARMUP_STEPS}
    lr_warmup_style: linear
    min_decay_lr: ${MIN_LR}
  optimizer_factory:
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    name: adamW
    torch_adam_is_fused: true
  weight_decay: 0.01
  zero_stage: 0
parallelism:
  dp: 2
  expert_parallel_size: 1
  pp: 1
  pp_engine: 1f1b
  tp: 4
  tp_linear_async_communication: true
  tp_mode: REDUCE_SCATTER
profiler: null
tokenizer:
  tokenizer_max_length: null
  tokenizer_name_or_path: ${MODEL_ID}
  tokenizer_revision: null
tokens:
  batch_accumulation_per_replica: ${GRAD_ACCUM}
  limit_test_batches: 0
  limit_val_batches: 0
  micro_batch_size: ${MICRO_BATCH_SIZE}
  sequence_length: ${SEQ_LEN}
  train_steps: ${TRAIN_STEPS}
  val_check_interval: -1
YAML

if [[ "${NANOTRON_CONFIG_ONLY:-0}" == "1" ]]; then
  echo "Generated config only: ${CONFIG}"
  exit 0
fi

cd ../nanotron
MASTER_PORT_VALUE="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 40000))}"
torchrun --master_port="${MASTER_PORT_VALUE}" --nproc_per_node=8 \
  run_train.py --config-file "${CONFIG}"

if ! checkpoint_is_complete "${RUN_ROOT}/checkpoints/${TRAIN_STEPS}"; then
  echo "Training exited without a complete final checkpoint: ${RUN_ROOT}/checkpoints/${TRAIN_STEPS}" >&2
  exit 3
fi

if [[ -n "${FINAL_CHECKPOINT_ALIAS:-}" && -d "${RUN_ROOT}/checkpoints/${TRAIN_STEPS}" ]]; then
  ln -sfn "${TRAIN_STEPS}" "${RUN_ROOT}/checkpoints/${FINAL_CHECKPOINT_ALIAS}"
fi
