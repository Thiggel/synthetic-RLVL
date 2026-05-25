#!/usr/bin/env bash
#SBATCH --job-name=hfsa_followup_oversight
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/hpc/c107fa/c107fa12/synthetic-RLVL}"
mkdir -p logs
source ./scripts/env.sh

VENV_PATH="${VENV_PATH:-${HPCVAULT:-}/.venv_rlvl_posttrain}"
if [[ -z "${HPCVAULT:-}" || ! -x "${VENV_PATH}/bin/python" ]]; then
  VENV_PATH="${WORK}/.venv"
fi
export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"

STATE_DIR="${WORK}/synthetic-RLVL/oversight/hfsa_followup_20260524"
mkdir -p "${STATE_DIR}"

OVERSIGHT_HOP="${OVERSIGHT_HOP:-0}"
OVERSIGHT_MAX_HOPS="${OVERSIGHT_MAX_HOPS:-18}"
OVERSIGHT_INTERVAL="${OVERSIGHT_INTERVAL:-4hours}"
NEXT_HOP=$((OVERSIGHT_HOP + 1))

if (( NEXT_HOP < OVERSIGHT_MAX_HOPS )); then
  sbatch \
    --begin="now+${OVERSIGHT_INTERVAL}" \
    --export=ALL,OVERSIGHT_HOP="${NEXT_HOP}",OVERSIGHT_MAX_HOPS="${OVERSIGHT_MAX_HOPS}",OVERSIGHT_INTERVAL="${OVERSIGHT_INTERVAL}" \
    scripts/slurm/codex/hfsa_followup_oversight_2026-05-24.slurm \
    > "${STATE_DIR}/scheduled_next_${SLURM_JOB_ID:-manual}.txt" || true
fi

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
  echo "cs wrapper/alias not found after sourcing ~/.bash_profile" >&2
  exit 127
}

cs exec "You are running scheduled autonomous Codex oversight for the Synthetic-RLVL HFSA follow-up wave in /home/hpc/c107fa/c107fa12/synthetic-RLVL.

This oversight job has already attempted to schedule the next oversight pass. Keep the handoff docs current in this same turn if jobs finish, fail, are submitted, canceled, dependency-edited, or yield new results.

First read:
- AGENTS.md
- docs/current_system_state.md
- docs/formal_logic_cot_research_plan_2026-05-19.md
- docs/hfsa_depth_scaling_plan_2026-05-19.md
- docs/materialized_dataset.md
- docs/paired_synthetic_benchmarks_2026-05-20.md

Use source ./scripts/env.sh for shell commands. Do not touch unrelated tjepa_* or seqedit_* jobs except to label them unrelated if visible in squeue.

Monitor these current chains:
- paired train-10: materialization 3656308, SFT 3656309, eval 3656310
- Qwen 7B ablation: SFT 3656217, eval 3656218
- Qwen-1.5B/Gemma ablation: SFT 3656323, retries 3656359 and 3656387, eval 3656389
- OLMo-32B pilot: SFT 3656335, eval 3656336
- tiny Llama scratch pretraining: pretrain 3656338, retries 3656360 and 3656388, eval 3656390

Tasks:
- Check squeue and sacct for those jobs, including array rows, states, elapsed time, exit codes, dependencies, and node failures.
- Inspect logs for Traceback, OOM, CUDA OOM, quota/no-space, DependencyNeverSatisfied, tokenizer/model-load errors, vLLM failures, idle GPU symptoms, and timeout/cancelled tasks.
- If a job failed due to a transient node failure and no equivalent live/pending retry exists, submit a replacement for only the missing row, excluding the bad node if one is implicated, and rewire dependent eval arrays if needed.
- If a job failed due to a code/config issue, inspect the traceback, implement the smallest local fix, run focused py_compile/bash -n/smoke checks, and resubmit only the affected row or dependent chain.
- If a job is healthy and progressing, leave it alone.
- When eval jobs finish, analyze JSON outputs with the existing analysis scripts or lightweight focused parsing, summarize results, and update the relevant docs.
- Do not launch broad new scientific experiments unless needed to recover the already submitted chains.
- Keep docs/current_system_state.md as the shortest live operational truth; move longer detail to the relevant experiment doc.
- End with a concise status report containing job IDs, states, output artifacts, failures or fixes, and the next safe action."
