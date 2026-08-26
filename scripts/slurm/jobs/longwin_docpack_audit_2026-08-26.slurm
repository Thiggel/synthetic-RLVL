#!/usr/bin/env bash
#SBATCH --job-name=longwin_docpack_audit
#SBATCH --partition=a40,a100,a100mig
#SBATCH --gres=gpu:1
#SBATCH --exclude=a0531,a0532,a0533,a0934
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Long-window (8192) midtrain prep: run the four preregistered docpack audit
# gates for every treatment arm at the 2.5B/8192 blend horizon:
#   2385 steps x GBS 128 x seq 8192 = 2,500,853,760 tokens (identical budget
#   to the accepted 4096 rerun: 4770 x 128 x 4096), 10% replacement.
#
# Arms audited: longdoc_control, logic_band25, nl_exact_band25,
# condensed_logic_band25. The control arm is Dolmino-only and has no docpack.
# Each fail-closed train job reads its audit JSON (all_pass + blend size).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export WORK="${WORK:-/home/atuin/c107fa/c107fa12}"
export HPCVAULT="${HPCVAULT:-/home/vault/c107fa/c107fa12}"

TRAIN_STEPS="${LONGWIN_TRAIN_STEPS:-2385}"
GLOBAL_BATCH_SIZE="${LONGWIN_GBS:-128}"
SEQ_LEN="${LONGWIN_SEQ_LEN:-8192}"
TARGET_RATIO="${LONGWIN_TARGET_RATIO:-0.10}"
AUDIT_DIR="${AUDIT_DIR:-analysis/longwin_midtrain_prep_20260826}"

source ./scripts/env.sh
source "${WORK}/nanotron/bin/activate"
export HF_HOME="${HF_HOME:-${HPCVAULT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

PACK_ROOT="${PACK_ROOT:-${WORK}/synthetic-RLVL/nanosets_longwin_20260826}"
DOLMINO_NANOSET="${DOLMINO_NANOSET:-${WORK}/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino}"

mkdir -p "${AUDIT_DIR}"

declare -A TEMPLATES=(
  [longdoc_control]=longdoc
  [logic_band25]=sft_logic
  [nl_exact_band25]=sft_nl_exact
  [condensed_logic_band25]=condensed_logic
)

for ARM in longdoc_control logic_band25 nl_exact_band25 condensed_logic_band25; do
  FOLDER="${PACK_ROOT}/${ARM}"
  python scripts/nanotron/audit_docpack_training_path.py \
    --packed-folder "${FOLDER}" \
    --dolmino-folder "${DOLMINO_NANOSET}" \
    --tokenizer "Qwen/Qwen2.5-7B" \
    --template "${TEMPLATES[$ARM]}" \
    --seq-len "${SEQ_LEN}" \
    --train-steps "${TRAIN_STEPS}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --target-ratio "${TARGET_RATIO}" \
    --out-json "${AUDIT_DIR}/audit_docpack_${ARM}.json" \
    --out-md "${AUDIT_DIR}/audit_docpack_${ARM}.md" \
    --examples-out "${AUDIT_DIR}/decoded_windows_${ARM}.md"
done

echo "Long-window 2.5B-horizon audits complete: ${AUDIT_DIR}"
python - "${AUDIT_DIR}" <<"PY"
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
for name in sorted(d.glob("audit_docpack_*.json")):
    j = json.loads(name.read_text())
    g = j["gates"]["exact_mixture"]
    print(name.name, "all_pass=", j.get("all_pass"), "proof_weight=", g["proof_weight"],
          "realized=", g["realized_loss_token_ratio"], "epochs=", g["synthetic_epochs_consumed"])
PY
