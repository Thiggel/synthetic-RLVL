#!/usr/bin/env bash
#SBATCH --job-name=experiment_oversight
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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

# `cs` is the user's Codex wrapper/alias with the desired permissions. Batch
# shells do not load interactive aliases unless we explicitly enable and source.
shopt -s expand_aliases
if [[ -f "${HOME}/.bash_profile" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.bash_profile"
elif [[ -f "${HOME}/.bashrc" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.bashrc"
fi
hash -r 2>/dev/null || true

type cs >/dev/null 2>&1 || {
  echo "cs wrapper/alias not found after sourcing shell profile" >&2
  exit 127
}

cs exec "Please monitor the Synthetic-RLVL experiments in this repository.

Read docs/current_system_state.md and docs/posttrain_recovery_passk_checklist_2026-04-24.md first. Then:
- Check Slurm state for active recovery arrays, merge/eval array, and oversight jobs.
- Check recent logs for failures, GPU-idle jobs, Ray startup failures, and timeout/cancelled tasks.
- If a recovery task failed and no equivalent live/pending retry exists, debug and resubmit only the missing rows/seeds.
- If long jobs are merely progressing slowly, do not cancel them; rely on checkpointed continuation waves.
- Keep docs/current_system_state.md and docs/posttrain_recovery_passk_checklist_2026-04-24.md updated with any actions and job IDs.
- If everything is healthy and no action is needed, write a concise status note in the docs and exit."
