#!/usr/bin/env bash
#SBATCH --job-name=active_recovery_oversight
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
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
hash -r 2>/dev/null || true

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

cs exec "You are running scheduled autonomous Codex oversight for the active Synthetic-RLVL recovery/eval wave in /home/hpc/c107fa/c107fa12/synthetic-RLVL.

This is a one-off scheduled pass; do not schedule another oversight pass unless the user has explicitly requested it in the handoff docs.

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

Use source ./scripts/env.sh before shell commands. Do not touch unrelated puzzle_*, babylm-*, tjepa_*, or seqedit_* jobs except to label them unrelated if visible in squeue.

Primary active chains to monitor:
- typed maze rerun: build 3695237 complete, SFT 3695238/3705793 complete at 30/30 finals, eval 3705795_[0-29%3] under \$HPCVAULT/synthetic-RLVL/passk_eval/paired_maze_typed_sparse_20260603/
- HFSA batch-size ablation: missing bsz16 finals for rows 3/7/11; latest recovery 3715329_[3,7,11%3], eval 3715330_[0-15%4] under \$HPCVAULT/synthetic-RLVL/passk_eval/hfsa_batch_size_ablation_20260603/
- semantic iGSM rerun is complete and already report-ingested; only revisit it if report artifacts are accidentally stale.

Tasks:
- Treat this as plan-driven oversight, not just a queue check. Compare live state against the handoff docs and backlog; update docs if jobs started, completed, failed, were canceled, were resubmitted, or yielded new outputs.
- Check squeue and sacct for 3705795, 3715329, 3715330, and any dependency-related array rows. Include states, elapsed time, exit codes, dependencies, and node-failure/timeout reasons.
- Inspect whether pending rows can safely run on freer compatible partitions. For bsz16 recovery/eval, only use compatible A100-80GB resources; do not move training/eval to A40/RTX just to start it. For oversight itself, no GPU is needed.
- Inspect logs for Traceback, OOM/CUDA OOM, quota/no-space, DependencyNeverSatisfied, tokenizer/model-load errors, vLLM failures, idle GPU symptoms, and timeouts/cancellations.
- If bsz16 recovery 3715329 times out but advances checkpoints, record the new checkpoint depths. Resubmit only missing bsz16 rows if no equivalent live/pending retry exists; do not let a full eval run against missing final adapters.
- If 3715330 is stale because its dependency is satisfied without all finals present, cancel it and resubmit a replacement eval depending on the newest bsz16 recovery.
- If typed maze eval 3705795 produces JSONs, aggregate available results, inspect representative samples across logic/NL, train depths, eval depths, successes, and failures before making conclusions. Verify prompt format, answer extraction, token limits, and validity/translator assumptions.
- If batch-size eval 3715330 produces JSONs, aggregate available results and inspect representative samples before adding tables/figures to the report. Report bsz16 as effective batch size with gradient accumulation, not physical microbatch 16.
- When a result family has enough new outputs, regenerate analysis/logic_cot_report_2026-05-25/ with scripts/analysis/build_logic_cot_report.py, mirror to ../synthetic-RLVL-report, and include justified tables/figures/sample notes.
- Keep docs/current_system_state.md short and current; update docs/running_experiments.md, docs/experiment_backlog.md, and docs/project_log.md for every operational or scientific state change.
- Commit and push this repo and ../synthetic-RLVL-report when changes are made and network/authentication permit.
- End with a concise status report containing job IDs, states, output artifacts, sample-generation findings, figures/tables created, failures or fixes, submissions, and the next safe action."
