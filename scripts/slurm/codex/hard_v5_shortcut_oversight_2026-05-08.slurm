#!/usr/bin/env bash
#SBATCH --job-name=hard_v5_short_oversight
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then
  VENV_PATH="${WORK}/.venv"
fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"

shopt -s expand_aliases
if [[ -f "${HOME}/.bash_profile" ]]; then
  source "${HOME}/.bash_profile"
elif [[ -f "${HOME}/.bashrc" ]]; then
  source "${HOME}/.bashrc"
fi
hash -r 2>/dev/null || true

type cs >/dev/null 2>&1 || {
  echo "cs wrapper/alias not found after sourcing shell profile" >&2
  exit 127
}

cs exec "Monitor the current Synthetic-RLVL hard-v5 shortcut rerun in /home/hpc/c107fa/c107fa12/synthetic-RLVL.

Focus on this chain only unless the docs show a direct dependency:
- dataset build/push: 3598347
- short SFT array: 3598348
- SFT merge/sanity array: 3598349
- GRPO reward ablation array: 3598350
- final merge + pass@k eval array: 3598351

Read AGENTS.md, docs/current_system_state.md, and docs/materialized_dataset.md first. Then:
- Check Slurm state and sacct for jobs 3598347-3598351.
- Check recent logs for failed HF upload, DependencyNeverSatisfied, SFT/merge failures, Ray startup failures, idle GPU, OOM, quota errors, or timeouts.
- If dataset upload failed due to transient Hugging Face 5xx, confirm retry logic exists, cancel dependency-dead children, and resubmit the same chain with the patched scripts.
- If a downstream task failed and no equivalent live/pending retry exists, debug and resubmit only the missing row/array with correct dependency and grouping.
- Do not touch unrelated tjepa_* or af_* jobs.
- Keep docs/current_system_state.md updated with status, actions, and new job IDs.
- If everything is healthy, write a short status note in docs/current_system_state.md and exit."
