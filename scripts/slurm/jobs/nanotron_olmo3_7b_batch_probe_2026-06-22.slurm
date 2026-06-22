#!/usr/bin/env bash
#SBATCH --job-name=nanotron_olmo7b_probe
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --array=0-4%1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

MICRO_BATCHES=(1 2 4 8 16)
MICRO_BATCH_SIZE="${MICRO_BATCHES[${SLURM_ARRAY_TASK_ID}]}"
SEQ_LEN="${SEQ_LEN:-4096}"
TRAIN_STEPS="${TRAIN_STEPS:-2}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export WANDB_MODE="${WANDB_MODE:-offline}"

source "${WORK}/nanotron/bin/activate"

RUN_ROOT="${HPCVAULT}/synthetic-RLVL/nanotron_smoke/olmo3_7b_shape_mb${MICRO_BATCH_SIZE}_seq${SEQ_LEN}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${RUN_ROOT}"
CONFIG="${RUN_ROOT}/config.yaml"

cat > "${CONFIG}" <<YAML
checkpoints:
  checkpoint_interval: 1000
  checkpoints_path: ${RUN_ROOT}/checkpoints
  checkpoints_path_is_shared_file_system: false
  resume_checkpoint_path: null
  save_initial_state: false
data_stages:
- data:
    dataset: null
    num_loading_workers: 1
    seed: 42
  name: olmo3_7b_shape_probe
  start_training_step: 1
general:
  benchmark_csv_path: ${RUN_ROOT}/benchmark.csv
  consumed_train_samples: null
  ignore_sanity_checks: true
  project: nanotron_olmo_probe
  run: olmo3_7b_shape_mb${MICRO_BATCH_SIZE}_seq${SEQ_LEN}
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
    bos_token_id: 1
    eos_token_id: 2
    hidden_act: silu
    hidden_size: 4096
    initializer_range: 0.02
    intermediate_size: 11008
    is_llama_config: true
    max_position_embeddings: 65536
    num_attention_heads: 32
    num_hidden_layers: 32
    num_key_value_heads: 32
    pad_token_id: null
    pretraining_tp: 1
    rms_norm_eps: 1.0e-06
    rope_scaling:
      attention_factor: 1.2079441541679836
      beta_fast: 32.0
      beta_slow: 1.0
      factor: 8.0
      original_max_position_embeddings: 8192
      rope_type: yarn
    rope_theta: 500000.0
    tie_word_embeddings: false
    use_cache: true
    vocab_size: 100278
optimizer:
  accumulate_grad_in_fp32: true
  clip_grad: 1.0
  learning_rate_scheduler:
    learning_rate: 0.00003
    lr_decay_starting_step: null
    lr_decay_steps: ${TRAIN_STEPS}
    lr_decay_style: cosine
    lr_warmup_steps: 1
    lr_warmup_style: linear
    min_decay_lr: 1.0e-06
  optimizer_factory:
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    name: adamW
    torch_adam_is_fused: true
  weight_decay: 0.01
  zero_stage: 0
parallelism:
  dp: 1
  expert_parallel_size: 1
  pp: 1
  pp_engine: 1f1b
  tp: 8
  tp_linear_async_communication: true
  tp_mode: REDUCE_SCATTER
profiler: null
tokenizer:
  tokenizer_max_length: null
  tokenizer_name_or_path: allenai/Olmo-3-1025-7B
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

cd ../nanotron
torchrun --nproc_per_node=8 run_train.py --config-file "${CONFIG}"
