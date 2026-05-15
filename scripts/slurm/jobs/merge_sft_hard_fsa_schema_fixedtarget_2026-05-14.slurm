#!/usr/bin/env bash
#SBATCH --job-name=merge_sft_hfsa_fix
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=06:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs tmp
source ./scripts/env.sh
VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then VENV_PATH="${WORK}/.venv"; fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

SEED="${SEED:-3407}"
TAGS=(sft1to3 sft1to5)
TAG=${TAGS[${SLURM_ARRAY_TASK_ID}]}
RUN_NAME="sft_hfsa_fixedtarget_${TAG}_lr1e-4_seed${SEED}"
SFT_FINAL="${WORK}/synthetic-RLVL/runs/${RUN_NAME}/final"
MERGED="${WORK}/synthetic-RLVL/tmp/merged_sft_hfsa_fixedtarget_${TAG}_seed${SEED}"
SANITY_OUT="${WORK}/synthetic-RLVL/tmp/sanity_sft_hfsa_fixedtarget_${TAG}_seed${SEED}.json"

if [[ ! -f "${SFT_FINAL}/adapter_config.json" ]]; then
  echo "missing SFT LoRA checkpoint: ${SFT_FINAL}" >&2
  exit 2
fi

rm -rf "${MERGED}.tmp"
python scripts/merge_lora_checkpoint.py --checkpoint "${SFT_FINAL}" --output "${MERGED}.tmp" --dtype bf16 --overwrite
rm -rf "${MERGED}"
mv "${MERGED}.tmp" "${MERGED}"

export SEED MERGED SANITY_OUT TAG
python - <<'PY'
import json, os, torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from synthrlvl.task import task_sample_from_materialized_row
from synthrlvl.types import PrefillMode, StepRange, TaskConfig, TemplateName

row = load_dataset("flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget", "val_step_03_1k", split="train[:1]")[0]
cfg = TaskConfig(template=TemplateName.LOGIC, prefill=PrefillMode.NONE, train_steps=StepRange(1,3), val_steps=StepRange(1,25), distractor_ratio=0.5, difficulty="hard_fsa_schema", branching_factor=4, shortcut_rate=0.0, require_unique_solution=True, seed=int(os.environ["SEED"]))
sample = task_sample_from_materialized_row(row, cfg=cfg)
tok = AutoTokenizer.from_pretrained(os.environ["MERGED"])
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(os.environ["MERGED"], torch_dtype=torch.bfloat16, device_map="auto")
inputs = tok(sample.prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
completion = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
payload = {"seed": int(os.environ["SEED"]), "tag": os.environ["TAG"], "checkpoint": os.environ["MERGED"], "completion": completion, "starts_with_formal": completion.lstrip().startswith("<formal>"), "contains_formal": "<formal>" in completion and "</formal>" in completion}
Path(os.environ["SANITY_OUT"]).write_text(json.dumps(payload, indent=2)+"\n")
print(json.dumps(payload, indent=2), flush=True)
PY
