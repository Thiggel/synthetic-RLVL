#!/usr/bin/env bash
#SBATCH --job-name=paired_full_oversight
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
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

STATE_DIR="${WORK}/synthetic-RLVL/oversight/paired_full_suite_20260528"
mkdir -p "${STATE_DIR}"

PAIRED_BUILD_JOB="${PAIRED_BUILD_JOB:-3672195}"
PAIRED_SFT_JOB="${PAIRED_SFT_JOB:-see-current-system-state}"
PAIRED_EVAL_JOB="${PAIRED_EVAL_JOB:-see-current-system-state}"

OVERSIGHT_HOP="${OVERSIGHT_HOP:-0}"
OVERSIGHT_MAX_HOPS="${OVERSIGHT_MAX_HOPS:-18}"
OVERSIGHT_INTERVAL="${OVERSIGHT_INTERVAL:-4hours}"
NEXT_HOP=$((OVERSIGHT_HOP + 1))

if (( NEXT_HOP < OVERSIGHT_MAX_HOPS )); then
  sbatch \
    --begin="now+${OVERSIGHT_INTERVAL}" \
    --export=ALL,OVERSIGHT_HOP="${NEXT_HOP}",OVERSIGHT_MAX_HOPS="${OVERSIGHT_MAX_HOPS}",OVERSIGHT_INTERVAL="${OVERSIGHT_INTERVAL}" \
    scripts/slurm/codex/paired_full_suite_oversight_2026-05-28.slurm \
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

cs exec "You are running scheduled autonomous Codex oversight for the Synthetic-RLVL paired full-suite wave in /home/hpc/c107fa/c107fa12/synthetic-RLVL.

This oversight job has already attempted to schedule the next oversight pass. Keep the handoff docs current in this same turn if jobs finish, fail, are submitted, canceled, dependency-edited, or yield new results.

First read:
- AGENTS.md
- docs/current_system_state.md
- docs/paired_synthetic_benchmarks_2026-05-20.md
- docs/materialized_dataset.md
- docs/formal_logic_cot_research_plan_2026-05-19.md
- docs/hfsa_depth_scaling_plan_2026-05-19.md

Use source ./scripts/env.sh for shell commands. Do not touch unrelated tjepa_* or seqedit_* jobs except to label them unrelated if visible in squeue.

Monitor this paired full-suite chain:
- build: ${PAIRED_BUILD_JOB}_[0-2%3]
- SFT: ${PAIRED_SFT_JOB}_[0-89%6]
- sparse pass@k eval: ${PAIRED_EVAL_JOB}_[0-89%4]

Scope:
- families: official_igsm, maze_navigation, attribute_constraints_hard
- train ranges: 1..5, 1..10, 1..15, 1..20, 1..25
- templates: logic, nl_exact
- seeds: 3407, 3408, 3409
- materialized roots: \${WORK}/synthetic-RLVL/datasets/materialized_paired_<family>_full_20260528
- eval outputs: \${WORK}/synthetic-RLVL/passk_eval/paired_full_suite_sparse_20260528/

Tasks:
- Check squeue and sacct for the paired full-suite jobs, including array rows, states, elapsed time, exit codes, dependencies, and node failures.
- Inspect relevant logs for Traceback, proof-validation failure, OOM, CUDA OOM, context-length failure, quota/no-space, DependencyNeverSatisfied, tokenizer/model-load errors, vLLM failures, idle GPU symptoms, and timeout/cancelled tasks.
- If a build row fails due to validation or generator logic, inspect the failing example, implement the smallest local fix, run focused tests/smokes, and resubmit only the affected family build plus dependent rows as needed.
- If an SFT/eval row fails due to transient node failure and no equivalent live/pending retry exists, submit a replacement only for the missing row, excluding the implicated bad node if known, and recreate dependent eval rows if needed.
- If an eval row fails due to context limits, resubmit only that row with larger vLLM context or smaller batch/max-new-token settings.
- If jobs are healthy and progressing, leave them alone.
- When eval rows finish, aggregate available JSON outputs, summarize substrate/dataset trends, and update docs/current_system_state.md plus docs/paired_synthetic_benchmarks_2026-05-20.md.
- Do not launch broad new scientific experiments unless needed to recover this already submitted chain.
- End with a concise status report containing job IDs, states, output artifacts, failures or fixes, and next safe action."
