#!/usr/bin/env bash
#SBATCH --job-name=hfsa_ablate_oversight
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=2:00:00
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

STATE_DIR="${WORK}/synthetic-RLVL/oversight/hfsa_ablation_20260529"
mkdir -p "${STATE_DIR}"

OVERSIGHT_HOP="${OVERSIGHT_HOP:-0}"
OVERSIGHT_MAX_HOPS="${OVERSIGHT_MAX_HOPS:-24}"
OVERSIGHT_INTERVAL="${OVERSIGHT_INTERVAL:-4hours}"
NEXT_HOP=$((OVERSIGHT_HOP + 1))

if (( NEXT_HOP < OVERSIGHT_MAX_HOPS )); then
  sbatch \
    --begin="now+${OVERSIGHT_INTERVAL}" \
    --export=ALL,OVERSIGHT_HOP="${NEXT_HOP}",OVERSIGHT_MAX_HOPS="${OVERSIGHT_MAX_HOPS}",OVERSIGHT_INTERVAL="${OVERSIGHT_INTERVAL}" \
    scripts/slurm/codex/hfsa_ablation_oversight_2026-05-29.slurm \
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

cs exec "You are running scheduled autonomous Codex oversight for the Synthetic-RLVL HFSA ablation wave submitted on 2026-05-29 in /home/hpc/c107fa/c107fa12/synthetic-RLVL.

This oversight job has already attempted to schedule the next oversight pass. Keep handoff docs current in this same turn if jobs finish, fail, are submitted, canceled, dependency-edited, or yield new results.

First read:
- AGENTS.md
- docs/current_system_state.md
- docs/running_experiments.md
- docs/experiment_backlog.md
- docs/project_log.md
- docs/formal_logic_cot_research_plan_2026-05-19.md
- docs/hfsa_depth_scaling_plan_2026-05-19.md
- docs/materialized_dataset.md
- analysis/logic_cot_report_2026-05-25/logic_cot_report_2026-05-25.tex

Use source ./scripts/env.sh for shell commands. Do not touch unrelated tjepa_*, seqedit_*, or puzzle_* jobs except to label them unrelated if visible in squeue.

Monitor these active chains:
- wordified length control: SFT 3674875, eval 3674876; duplicates 3674877/3674878 were intentionally canceled.
- conditioned dual 50k convergence: chunked SFT 3674879 -> 3674880 -> 3674881 -> 3674882 -> 3674883, final eval 3674884, checkpoint eval 3674885.
- shortcut-kind controls: build 3674886, SFT 3674887, eval 3674888.
- still-active predecessor ablations if present: trace-control 3661118/3661119, shortcut-rate 0.3 3671431/3671432, hybrid 3670782/3670783.

Tasks:
- Treat this as a plan-driven oversight pass, not just a queue check. Compare live state against docs/experiment_backlog.md and the active plan docs, decide whether any planned trigger is satisfied, and update the backlog when work is started, deferred, completed, or invalidated.
- Check squeue and sacct for those jobs, including array rows, states, elapsed time, exit codes, dependencies, and node failures.
- Inspect pending rows for compatible freer partitions. If widening is safe and useful, run scontrol update JobId=<jobid> Partition=<partition1,partition2> and record the edit in the handoff docs.
- Inspect logs for Traceback, OOM, CUDA OOM, quota/no-space, DependencyNeverSatisfied, tokenizer/model-load errors, vLLM failures, idle GPU symptoms, and timeout/cancelled tasks.
- Inspect output roots, manifests, eval JSONs, and sample-generation artifacts for every newly completed or suspicious family. Look across seeds, train depths, eval depths, templates, and success/failure cases when available.
- Before accepting any metric, question the evaluator and experiment assumptions: verify prompt format, answer extraction, trace translation, validity checks, token limits, stale outputs, report aggregation, and whether generations actually match the intended modality.
- If a job failed due to transient node failure and no equivalent live/pending retry exists, submit a replacement only for the missing row, excluding the bad node if implicated, and rewire dependent eval arrays if needed.
- If a job failed due to a code/config issue, inspect the traceback, implement the smallest local fix, run focused py_compile/bash -n/smoke checks, and resubmit only the affected row or dependent chain.
- If jobs are healthy and progressing, leave them alone.
- When eval jobs finish, analyze JSON outputs with the existing report builder and focused parsing. Create the most informative currently justified tables, figures, curves, and qualitative sample summaries; regenerate analysis/logic_cot_report_2026-05-25/ and mirror it to ../synthetic-RLVL-report.
- If the backlog/plan clearly says a next experiment should start and prerequisites are satisfied, submit the smallest appropriate job set, document the job IDs, and keep the plan/backlog synchronized. Be conservative with broad expensive launches unless the trigger is explicit.
- Keep docs/current_system_state.md as the shortest live operational truth; move longer detail to the relevant experiment doc.
- Update docs/current_system_state.md, docs/running_experiments.md, docs/experiment_backlog.md, docs/project_log.md, and relevant experiment docs for any state/result/insight change.
- Commit and push this repo and ../synthetic-RLVL-report when changes are made and network/authentication permit.
- End with a concise status report containing job IDs, states, output artifacts, sample-generation findings, figures/tables created, failures or fixes, submissions, and the next safe action."
