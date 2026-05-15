#!/usr/bin/env bash
# Merge hard-v5 SFT LoRA checkpoints and run a small generation sanity check.
# Submit:
#   sbatch --dependency=afterok:<sft_jobid> --array=0-2%3 scripts/slurm/jobs/merge_sft_hard_v5_full_2026-05-03.slurm

#SBATCH --job-name=merge_sft_hard_v5_full
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs tmp
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then
  VENV_PATH="${WORK}/.venv"
fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

SEEDS=(3407 3408 3409)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID}]}
RUN_NAME="sft_hard_v5_full_lr1e-4_seed${SEED}"
SFT_FINAL="${WORK}/synthetic-RLVL/runs/${RUN_NAME}/final"
MERGED="${WORK}/synthetic-RLVL/tmp/merged_sft_hard_v5_full_seed${SEED}"
SANITY_OUT="${WORK}/synthetic-RLVL/tmp/sanity_sft_hard_v5_full_seed${SEED}.json"

if [[ ! -f "${SFT_FINAL}/adapter_config.json" ]]; then
  echo "[merge-sft-v5] missing SFT LoRA checkpoint: ${SFT_FINAL}" >&2
  exit 2
fi

if [[ ! -f "${MERGED}/config.json" || "${FORCE_SFT_MERGE:-0}" == "1" ]]; then
  rm -rf "${MERGED}.tmp"
  python scripts/merge_lora_checkpoint.py \
    --checkpoint "${SFT_FINAL}" \
    --output "${MERGED}.tmp" \
    --dtype bf16 \
    --overwrite
  rm -rf "${MERGED}"
  mv "${MERGED}.tmp" "${MERGED}"
else
  echo "[merge-sft-v5] using existing merged checkpoint ${MERGED}"
fi

export SEED MERGED SANITY_OUT
python - <<'PY'
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from synthrlvl.task import task_sample_from_materialized_row
from synthrlvl.types import PrefillMode, StepRange, TaskConfig, TemplateName

merged = os.environ["MERGED"]
out_path = Path(os.environ["SANITY_OUT"])
seed = int(os.environ["SEED"])
row = load_dataset("flaitenberger/LogicalReasoning-hard-v5", "val_step_03_1k", split="train[:1]")[0]
task_cfg = TaskConfig(
    template=TemplateName.LOGIC,
    prefill=PrefillMode.NONE,
    train_steps=StepRange(1, 3),
    val_steps=StepRange(1, 20),
    distractor_ratio=0.5,
    difficulty="hard_v5",
    shortcut_rate=0.0,
    require_unique_solution=True,
    seed=seed,
)
sample = task_sample_from_materialized_row(row, cfg=task_cfg)
tokenizer = AutoTokenizer.from_pretrained(merged)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(merged, torch_dtype=torch.bfloat16, device_map="auto")
inputs = tokenizer(sample.prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_new_tokens=384,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
completion = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
payload = {
    "seed": seed,
    "checkpoint": merged,
    "prompt_chars": len(sample.prompt),
    "completion": completion,
    "starts_with_formal": completion.lstrip().startswith("<formal>"),
    "contains_formal": "<formal>" in completion and "</formal>" in completion,
}
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2), flush=True)
PY
