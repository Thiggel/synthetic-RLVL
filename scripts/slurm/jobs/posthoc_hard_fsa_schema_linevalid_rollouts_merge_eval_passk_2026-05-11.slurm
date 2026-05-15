#!/usr/bin/env bash
# Merge and sampled pass@k eval for the hard-FSA-schema dense line-valid
# rollout-count ablation submitted as 3600829.
#SBATCH --job-name=hfsa_linevalid_rollouts_passk
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=18:00:00
#SBATCH --array=0-5%6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs passk_eval/hard_fsa_schema_short500_rollouts tmp
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then
  VENV_PATH="${WORK}/.venv"
fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

SEEDS=(3407 3408 3409)
ROLLOUTS=(16 32)
NUM_PROMPTS=8
SCHEMA="correct_times_citation_free_line_valid_plus_0p1_format"

IDX=${SLURM_ARRAY_TASK_ID}
ROLLOUT_IDX=$((IDX % ${#ROLLOUTS[@]}))
SEED=${SEEDS[$((IDX / ${#ROLLOUTS[@]}))]}
NUM_ROLLOUTS=${ROLLOUTS[$ROLLOUT_IDX]}

RUN_NAME="rl_hard_fsa_schema_short500_np${NUM_PROMPTS}_nr${NUM_ROLLOUTS}_${SCHEMA}_seed${SEED}_mrg"
GROUP="posttrain_hard_fsa_schema_short500_rollouts/${SCHEMA}_np${NUM_PROMPTS}_nr${NUM_ROLLOUTS}"

sleep $((SLURM_ARRAY_TASK_ID * ${PASSK_STAGGER_SECONDS:-120} + RANDOM % ${PASSK_JITTER_SECONDS:-30}))

RUN_ROOT="${WORK}/synthetic-RLVL/runs/${RUN_NAME}"
ACTOR=""
FINAL_STEP=""
for STEP in 501 500 251 250; do
  CANDIDATE="${RUN_ROOT}/global_step_${STEP}/actor"
  if [[ -f "${CANDIDATE}/lora_adapter/adapter_config.json" || -f "${CANDIDATE}/adapter_config.json" ]]; then
    ACTOR="${CANDIDATE}"
    FINAL_STEP="${STEP}"
    break
  fi
done

if [[ -z "${ACTOR}" ]]; then
  echo "missing final actor for ${RUN_NAME}; skipping"
  exit 0
fi

MERGED="${WORK}/synthetic-RLVL/tmp/merged_actor_${RUN_NAME}_final"
OUT="${WORK}/synthetic-RLVL/passk_eval/hard_fsa_schema_short500_rollouts/${RUN_NAME}_passk.json"
SAMPLES_OUT="${WORK}/synthetic-RLVL/passk_eval/hard_fsa_schema_short500_rollouts/${RUN_NAME}_samples.jsonl"

if [[ ! -f "${MERGED}/config.json" || "${FORCE_PASSK_MERGE:-0}" == "1" ]]; then
  rm -rf "${MERGED}.tmp"
  python scripts/merge_lora_checkpoint.py --checkpoint "${ACTOR}" --output "${MERGED}.tmp" --dtype bf16 --overwrite
  rm -rf "${MERGED}"
  mv "${MERGED}.tmp" "${MERGED}"
fi

if [[ -s "${OUT}" && "${FORCE_PASSK_EVAL:-0}" != "1" ]]; then
  echo "output already exists: ${OUT}; skipping eval"
  exit 0
fi

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export WANDB_GROUP="${GROUP}"
export WANDB_RUN_GROUP="${GROUP}"

python scripts/evaluate_checkpoint_passk.py \
  --checkpoint "${MERGED}" \
  --config conf/posttrain_grpo_hard_fsa_schema_500.yaml \
  --profile grpo \
  --backend vllm \
  --step-min 1 \
  --step-max 20 \
  --samples-per-step "${PASSK_SAMPLES_PER_STEP:-20}" \
  --num-generations "${PASSK_NUM_GENERATIONS:-64}" \
  --k-values 1,2,4,8,16,32,64 \
  --temperature 1.0 \
  --max-new-tokens 1024 \
  --batch-size 1024 \
  --gpu-memory-utilization 0.80 \
  --disable-external \
  --collect-samples 20 \
  --output "${OUT}" \
  --samples-output "${SAMPLES_OUT}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${GROUP}" \
  --wandb-run-name "hard_fsa_schema_short500_np${NUM_PROMPTS}_nr${NUM_ROLLOUTS}_${SCHEMA}_seed${SEED}_passk_step${FINAL_STEP}"
