#!/usr/bin/env bash
#SBATCH --job-name=nano_q25_1gpu_smoke
#SBATCH --partition=a40,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export WANDB_MODE="${WANDB_MODE:-offline}"

source "${WORK}/nanotron/bin/activate"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}"
SEQ_LEN="${SEQ_LEN:-256}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-2}"
LR="${LR:-1.0e-6}"
MIX_KIND="${MIX_KIND:-logic}"
MIX_PCT="${MIX_PCT:-5}"

NORMAL_DATASET_FOLDER="${NORMAL_DATASET_FOLDER:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25/normal_continuation}"
LOGIC_DATASET_FOLDER="${LOGIC_DATASET_FOLDER:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25/logical_deduction_logic}"
NL_DATASET_FOLDER="${NL_DATASET_FOLDER:-${HPCVAULT}/synthetic-RLVL/nanosets/qwen25/logical_deduction_nl_exact}"
PRETRAINED_NANOTRON_CKPT="${PRETRAINED_NANOTRON_CKPT:-${HPCVAULT}/synthetic-RLVL/nanotron_checkpoints/qwen25_7b_tp1}"

RUN_TAG="qwen25_1gpu_${MIX_KIND}_p${MIX_PCT}_seq${SEQ_LEN}_steps${TRAIN_STEPS}_${SLURM_JOB_ID}"
RUN_ROOT="${HPCVAULT}/synthetic-RLVL/nanotron_smoke/${RUN_TAG}"
CONFIG="${RUN_ROOT}/config.yaml"

require_nanoset() {
  local path="$1"
  if ! compgen -G "${path}/*.metadata" >/dev/null; then
    echo "Missing pretokenized Nanoset metadata under ${path}" >&2
    exit 2
  fi
}

if [[ ! -f "${PRETRAINED_NANOTRON_CKPT}/model_config.json" ]]; then
  echo "Missing pretrained Nanotron checkpoint: ${PRETRAINED_NANOTRON_CKPT}" >&2
  exit 2
fi
require_nanoset "${NORMAL_DATASET_FOLDER}"
if [[ "${MIX_KIND}" == "logic" ]]; then
  PROOF_DATASET_FOLDER="${LOGIC_DATASET_FOLDER}"
  require_nanoset "${PROOF_DATASET_FOLDER}"
elif [[ "${MIX_KIND}" == "nl_exact" ]]; then
  PROOF_DATASET_FOLDER="${NL_DATASET_FOLDER}"
  require_nanoset "${PROOF_DATASET_FOLDER}"
elif [[ "${MIX_KIND}" == "control" ]]; then
  PROOF_DATASET_FOLDER=""
else
  echo "Unknown MIX_KIND=${MIX_KIND}; expected control, logic, or nl_exact" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}"

if [[ "${MIX_KIND}" == "control" ]]; then
  DATASET_YAML="      dataset_folder:
      - ${NORMAL_DATASET_FOLDER}
      dataset_weights:
      - 1.0"
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
  DATASET_YAML="      dataset_folder:
      - ${NORMAL_DATASET_FOLDER}
      - ${PROOF_DATASET_FOLDER}
      dataset_weights:
      - ${NORMAL_WEIGHT}
      - ${PROOF_WEIGHT}"
fi

cat > "${CONFIG}" <<YAML
checkpoints:
  checkpoint_interval: 999999
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
${DATASET_YAML}
      token_size_in_bytes: 4
      tokenizer_name: ${MODEL_ID}
      vocab_size: 152064
      shuffle_files: true
    num_loading_workers: 1
    seed: 42
  name: qwen25_single_gpu_smoke_${MIX_KIND}_p${MIX_PCT}
  start_training_step: 1
general:
  benchmark_csv_path: ${RUN_ROOT}/benchmark.csv
  consumed_train_samples: null
  ignore_sanity_checks: true
  project: synthetic-rlvl-midtrain-smoke
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
  accumulate_grad_in_fp32: false
  clip_grad: 1.0
  learning_rate_scheduler:
    learning_rate: ${LR}
    lr_decay_starting_step: null
    lr_decay_steps: ${TRAIN_STEPS}
    lr_decay_style: cosine
    lr_warmup_steps: 1
    lr_warmup_style: linear
    min_decay_lr: ${LR}
  optimizer_factory:
    name: sgd
  weight_decay: 0.0
  zero_stage: 0
parallelism:
  dp: 1
  expert_parallel_size: 1
  pp: 1
  pp_engine: 1f1b
  tp: 1
  tp_linear_async_communication: true
  tp_mode: REDUCE_SCATTER
profiler: null
tokenizer:
  tokenizer_max_length: null
  tokenizer_name_or_path: ${MODEL_ID}
  tokenizer_revision: null
tokens:
  batch_accumulation_per_replica: 1
  limit_test_batches: 0
  limit_val_batches: 0
  micro_batch_size: ${MICRO_BATCH_SIZE}
  sequence_length: ${SEQ_LEN}
  train_steps: ${TRAIN_STEPS}
  val_check_interval: -1
YAML

echo "### Single-GPU Nanotron smoke config"
echo "RUN_ROOT=${RUN_ROOT}"
echo "MIX_KIND=${MIX_KIND} MIX_PCT=${MIX_PCT} SEQ_LEN=${SEQ_LEN} TRAIN_STEPS=${TRAIN_STEPS}"
echo "PRETRAINED_NANOTRON_CKPT=${PRETRAINED_NANOTRON_CKPT}"
sed -n '1,260p' "${CONFIG}"

cd ../nanotron
torchrun --nproc_per_node=1 run_train.py --config-file "${CONFIG}"
