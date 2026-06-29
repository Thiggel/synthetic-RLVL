#!/usr/bin/env bash
#SBATCH --job-name=nanotron_qwen25_realdata
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
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
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

source "${WORK}/nanotron/bin/activate"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}"
SEQ_LEN="${SEQ_LEN:-1024}"
TRAIN_STEPS="${TRAIN_STEPS:-2}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
RUN_ROOT="${HPCVAULT}/synthetic-RLVL/nanotron_smoke/qwen25_pretrained_realdata_${SLURM_JOB_ID}"
DATA_JSONL="${RUN_ROOT}/mixture.jsonl"
DATASET_SCRIPT="${RUN_ROOT}/smoke_dataset.py"
CONVERTED_CKPT="${HPCVAULT}/synthetic-RLVL/nanotron_checkpoints/qwen25_7b_tp1"
CONFIG="${RUN_ROOT}/config.yaml"
mkdir -p "${RUN_ROOT}" "$(dirname "${CONVERTED_CKPT}")"

echo "### Exporting tiny proof-chain/normal-text mixture"
NORMAL_JSONL="${RUN_ROOT}/normal.jsonl"
python - <<PY
import json
from pathlib import Path
out = Path("${NORMAL_JSONL}")
rows = [
    {"text": "This is a short continuation-training control sentence about careful reasoning and concise explanations.", "source": "normal"},
    {"text": "A model should preserve definitions, follow each implication, and state the final answer plainly.", "source": "normal"},
    {"text": "The training stream mixes ordinary prose with formal proof traces to test data plumbing.", "source": "normal"},
    {"text": "Each sequence is packed into fixed-length causal language modeling chunks for the smoke test.", "source": "normal"},
]
with out.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row) + "\\n")
PY

python scripts/data/export_midtraining_mixture.py \
  --component "name=normal,kind=jsonl,path=${NORMAL_JSONL},weight=8,limit=4" \
  --component "name=logic,kind=materialized,dataset_id=flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget-depth50,subset=train_fixedtarget_up_to_5_50k,template=logic,weight=1,limit=16" \
  --component "name=nl,kind=materialized,dataset_id=flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget-depth50,subset=train_fixedtarget_up_to_5_50k,template=nl_exact,weight=1,limit=16" \
  --output "${DATA_JSONL}" \
  --output-mode causal \
  --tokenizer "${MODEL_ID}" \
  --max-records 36

cat > "${DATASET_SCRIPT}" <<PY
import json
from pathlib import Path
import datasets

class SmokeDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({"text": datasets.Value("string"), "source": datasets.Value("string")})
        )

    def _split_generators(self, dl_manager):
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"path": Path(__file__).with_name("mixture.jsonl")})]

    def _generate_examples(self, path):
        with Path(path).open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                yield i, {"text": str(row["text"]), "source": str(row.get("source", ""))}
PY

echo "### Debug print raw and packed data"
python - <<PY
import json
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer

path = Path("${DATA_JSONL}")
tok = AutoTokenizer.from_pretrained("${MODEL_ID}")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
print("records", len(rows), "sources", Counter(row.get("source") for row in rows), flush=True)
for row in rows[:6]:
    print("\\n### RAW", row.get("source"), "tokens", len(tok.encode(row["text"], add_special_tokens=False)), flush=True)
    print(row["text"][:2000], flush=True)

ids = []
for row in rows:
    ids.extend(tok.encode(row["text"], add_special_tokens=False))
total = ((len(ids) - 1) // ${SEQ_LEN}) * ${SEQ_LEN} + 1 if len(ids) >= ${SEQ_LEN} + 1 else len(ids)
ids = ids[:total]
for i, start in enumerate(range(0, max(0, len(ids) - (${SEQ_LEN} + 1)), ${SEQ_LEN})):
    chunk = ids[start:start + ${SEQ_LEN} + 1]
    print(f"\\n### PACKED_CHUNK {i} len={len(chunk)}", flush=True)
    print(tok.decode(chunk[: min(len(chunk), 512)], clean_up_tokenization_spaces=False), flush=True)
    if i >= 2:
        break
PY

echo "### Converting ${MODEL_ID} to Nanotron checkpoint if needed"
if [ ! -f "${CONVERTED_CKPT}/model_config.json" ]; then
  cd ../nanotron
  python -m examples.llama.convert_hf_to_nanotron \
    --checkpoint_path "${MODEL_ID}" \
    --save_path "${CONVERTED_CKPT}"
  cd "${SLURM_SUBMIT_DIR:-$PWD}"
else
  echo "Using existing converted checkpoint: ${CONVERTED_CKPT}"
fi

cat > "${CONFIG}" <<YAML
checkpoints:
  checkpoint_interval: 1000
  checkpoints_path: ${RUN_ROOT}/checkpoints
  checkpoints_path_is_shared_file_system: false
  load_lr_scheduler: false
  load_optimizer: false
  resume_checkpoint_path: ${CONVERTED_CKPT}
  save_final_state: true
  save_initial_state: false
data_stages:
- data:
    dataset:
      hf_dataset_or_datasets: ${DATASET_SCRIPT}
      hf_dataset_splits: train
      dataset_processing_num_proc_per_process: 1
      dataset_overwrite_cache: true
      text_column_name: text
    num_loading_workers: 1
    seed: 42
  name: qwen25_realdata_smoke
  start_training_step: 1
general:
  benchmark_csv_path: ${RUN_ROOT}/benchmark.csv
  consumed_train_samples: null
  ignore_sanity_checks: true
  project: nanotron_qwen_probe
  run: qwen25_pretrained_realdata_${SLURM_JOB_ID}
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
    learning_rate: 0.000003
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
  dp: 8
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

echo "### Training from converted pretrained Qwen2.5 checkpoint on packed real data"
cd ../nanotron
torchrun --nproc_per_node=8 run_train.py --config-file "${CONFIG}"
