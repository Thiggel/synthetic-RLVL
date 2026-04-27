#!/usr/bin/env bash
# Catch up only posttrain final-checkpoint gaps not already covered by live retry array 3557214.
# Submit example:
#   sbatch --array=0-11%4 scripts/slurm/jobs/posttrain_catchup_final_2026-04-24.slurm

#SBATCH --job-name=posttrain_catchup_final
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-11%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

WORK_ROOT="${WORK:?WORK must be set}"
VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then
  VENV_PATH="${WORK_ROOT}/.venv"
fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
hash -r 2>/dev/null || true

MANIFEST="docs/manifests/posttrain_catchup_2026-04-24.tsv"
ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
if [[ -z "${ROW}" ]]; then
  echo "No manifest row for array task ${SLURM_ARRAY_TASK_ID}; exiting."
  exit 0
fi
IFS=$'\t' read -r SWEEP KIND VAL SEED ACTION RUN_NAME <<< "${ROW}"

echo "[catchup] row=${SLURM_ARRAY_TASK_ID} sweep=${SWEEP} kind=${KIND} value=${VAL} seed=${SEED} action=${ACTION} run=${RUN_NAME}"

RUN_ROOT="/home/atuin/c107fa/c107fa12/synthetic-RLVL/runs/${RUN_NAME}"
for STEP in 1501 1500; do
  if [[ -f "${RUN_ROOT}/global_step_${STEP}/actor/lora_adapter/adapter_config.json" || -f "${RUN_ROOT}/global_step_${STEP}/actor/adapter_config.json" ]]; then
    echo "[catchup] final actor already exists for ${RUN_NAME} at global_step_${STEP}; skipping."
    exit 0
  fi
done

SFT_CKPT="/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/merged_sft_lr1e-4_seed${SEED}_remerge_20260421"
if [[ ! -f "${SFT_CKPT}/config.json" ]]; then
  SFT_CKPT="/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/merged_sft_lr1e-4_seed${SEED}"
fi
if [[ ! -f "${SFT_CKPT}/config.json" ]]; then
  echo "Missing merged SFT checkpoint for seed ${SEED}: ${SFT_CKPT}" >&2
  exit 1
fi

# Use auto for both continuation and rerun rows. If no checkpoint exists this
# starts fresh; if a prior chunk saved 500/1000 it continues. This is required
# for large num_prompts jobs that cannot reach 1500 in one 24h allocation.
RESUME_MODE="auto"

export WANDB_PROJECT="${WANDB_PROJECT:-synthetic-rlvl}"
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-120}"
JITTER_MAX_SECONDS="${RAY_STARTUP_JITTER_MAX_SECONDS:-180}"
JITTER_SECONDS=$((RANDOM % (JITTER_MAX_SECONDS + 1)))
echo "[ray-startup] jitter_sleep_s=${JITTER_SECONDS} max=${JITTER_MAX_SECONDS}"
sleep "${JITTER_SECONDS}"

COMMON=(
  seed="${SEED}"
  model.path="${SFT_CKPT}"
  model.merge_adapter_for_rollout=false
  task.template=logic
  task.prefill=none
  validation.enabled=true
  eval.sampled_enabled=false
  data.train_samples=120000
  grpo.max_num_batched_tokens=16384
  optim.micro_batch_size=4
  optim.logprob_micro_batch_size=4
  resume.mode="${RESUME_MODE}"
)

if [[ "${KIND}" == "prompts" ]]; then
  export WANDB_GROUP="posttrain/grpo_np${VAL}_nr8"
  export WANDB_RUN_GROUP="${WANDB_GROUP}"
  python posttrain_grpo_verl.py \
    run_name="${RUN_NAME}" \
    reward.schema=correct_plus_valid_plus_0p1_format \
    grpo.num_prompts="${VAL}" \
    grpo.num_rollouts=8 \
    "${COMMON[@]}"
elif [[ "${KIND}" == "rollouts" ]]; then
  export WANDB_GROUP="posttrain/grpo_np8_nr${VAL}"
  export WANDB_RUN_GROUP="${WANDB_GROUP}"
  python posttrain_grpo_verl.py \
    run_name="${RUN_NAME}" \
    reward.schema=correct_plus_valid_plus_0p1_format \
    grpo.num_prompts=8 \
    grpo.num_rollouts="${VAL}" \
    "${COMMON[@]}"
else
  echo "Unsupported catchup kind: ${KIND}" >&2
  exit 2
fi
