#!/usr/bin/env bash
#SBATCH --job-name=nanotron_tiny_smoke
#SBATCH --partition=a100,rtxpro6k
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
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

RUN_ROOT="${HPCVAULT}/synthetic-RLVL/nanotron_smoke/tiny_${SLURM_JOB_ID}"
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
  name: smoke
  start_training_step: 1
general:
  benchmark_csv_path: ${RUN_ROOT}/benchmark.csv
  consumed_train_samples: null
  ignore_sanity_checks: true
  project: nanotron_smoke
  run: tiny_${SLURM_JOB_ID}
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
    hidden_size: 512
    initializer_range: 0.02
    intermediate_size: 1376
    is_llama_config: true
    max_position_embeddings: 2048
    num_attention_heads: 8
    num_hidden_layers: 4
    num_key_value_heads: 8
    pad_token_id: null
    pretraining_tp: 1
    rms_norm_eps: 1.0e-06
    rope_scaling: null
    rope_theta: 10000.0
    tie_word_embeddings: false
    use_cache: true
    vocab_size: 8192
optimizer:
  accumulate_grad_in_fp32: true
  clip_grad: 1.0
  learning_rate_scheduler:
    learning_rate: 0.0003
    lr_decay_starting_step: null
    lr_decay_steps: 3
    lr_decay_style: cosine
    lr_warmup_steps: 1
    lr_warmup_style: linear
    min_decay_lr: 1.0e-05
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
  tp: 2
  tp_linear_async_communication: true
  tp_mode: REDUCE_SCATTER
profiler: null
tokenizer:
  tokenizer_max_length: null
  tokenizer_name_or_path: robot-test/dummy-tokenizer-wordlevel
  tokenizer_revision: null
tokens:
  batch_accumulation_per_replica: 1
  limit_test_batches: 0
  limit_val_batches: 0
  micro_batch_size: 2
  sequence_length: 512
  train_steps: 3
  val_check_interval: -1
YAML

cd ../nanotron
torchrun --nproc_per_node=4 run_train.py --config-file "${CONFIG}"
