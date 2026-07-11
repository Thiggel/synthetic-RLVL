#!/usr/bin/env bash
#SBATCH --job-name=bp_nano_oversight
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100small:1
#SBATCH --cpus-per-task=4
#SBATCH --time=5:00:00
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

STATE_DIR="${WORK}/synthetic-RLVL/oversight/branchproof_nanotron_20260711"
mkdir -p "${STATE_DIR}"

OVERSIGHT_HOP="${OVERSIGHT_HOP:-0}"
OVERSIGHT_MAX_HOPS="${OVERSIGHT_MAX_HOPS:-120}"
OVERSIGHT_INTERVAL="${OVERSIGHT_INTERVAL:-6hours}"
NEXT_HOP=$((OVERSIGHT_HOP + 1))
NEXT_JOB_FILE="${STATE_DIR}/scheduled_next_${SLURM_JOB_ID:-manual}.txt"

# Schedule the successor before running Codex so a Codex failure does not break
# the six-hour continuation chain.
if (( NEXT_HOP < OVERSIGHT_MAX_HOPS )); then
  sbatch \
    --begin="now+${OVERSIGHT_INTERVAL}" \
    --export=ALL,OVERSIGHT_HOP="${NEXT_HOP}",OVERSIGHT_MAX_HOPS="${OVERSIGHT_MAX_HOPS}",OVERSIGHT_INTERVAL="${OVERSIGHT_INTERVAL}" \
    scripts/slurm/codex/branchproof_nanotron_oversight_2026-07-11.slurm \
    > "${NEXT_JOB_FILE}" || true
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
  echo "cs wrapper/alias not found after sourcing shell profile" >&2
  exit 127
}

cs exec "You are running scheduled autonomous Codex oversight for the active corrected BranchProof and Nanotron plan in /home/hpc/c107fa/c107fa12/synthetic-RLVL.

This watcher scheduled its successor before starting. The successor job ID is recorded in ${NEXT_JOB_FILE}. Keep the chain alive while any part of the plan remains incomplete. Cancel the recorded successor only after the entire plan below is complete and verified, not merely because current jobs are healthy or pending.

First read in full:
- AGENTS.md
- docs/current_system_state.md
- docs/running_experiments.md
- docs/experiment_backlog.md
- docs/project_log.md
- docs/formal_logic_cot_research_plan_2026-05-19.md
- docs/hfsa_depth_scaling_plan_2026-05-19.md
- docs/branchproof_uniqueness_audit_2026-07-10.md
- docs/nanotron_mixture_schedule_audit_2026-07-10.md
- analysis/logic_cot_report_2026-05-25/logic_cot_report_2026-05-25.tex
- ../synthetic-RLVL-report/main.tex

Use source ./scripts/env.sh before shell commands. Do not touch unrelated puzzle_*, babylm-*, tjepa_*, seqedit_*, or grid_goal_* jobs except to identify them as unrelated.

Active critical paths:
- Corrected BranchProof full SFT 3829072_[0-29%12], followed per row by eval 3834582_[0-29%6]. Eval must remain on partition a100 with feature a100_80 and retain 32 prompts/depth, 16 generations, all 14 depths, pass@1/2/4/8/16, context 16384, equal cap 7168, and two sampled generations per prompt for qualitative review.
- BranchProof row audits 3834706_[0-29%8] follow eval rows with aftercorr dependencies; replacement aggregation and cross-grid qualitative audit 3835779 run only after all 30 audits pass. Canceled aggregator 3834707 predates the qualitative gate. Recover any failed eval/audit rows before aggregation rather than bypassing either gate.
- Downstream-suite final smoke 3834836 must pass before production eval. FOLIO is unavailable, legacy LogiQA tasks use unsupported scripts, and installed FLD prompts for proofs while exact-matching labels, creating an extraction floor. The validated suite uses gsm8k, hendrycks_math500, arc_challenge, hellaswag, winogrande, piqa, agieval_logiqa_en, bbh, mmlu, and mmlu_pro.
- Nanotron normal control 3823434_0 -> failed W&B-only recovery 3828946_0 -> replacement recovery 3835438_0 -> upload 3831119, direct eval 3834906, instruction SFT 3831121, instruction eval 3834907.
- Corrected Nanotron logic p15 3830927_3 -> replacement recovery 3835442_3 -> upload 3831123, direct eval 3834908, instruction SFT 3831125, instruction eval 3834909. Superseded untouched recovery 3830928_3 was canceled.
- Corrected Nanotron NL p15 3831111_8 -> replacement recovery 3835443_8 -> upload 3831113, direct eval 3834904, instruction SFT 3831115, instruction eval 3834905. Superseded untouched recovery 3831112_8 was canceled.

Plan to execute end to end:
1. Complete corrected BranchProof SFT/eval. Check squeue, expanded sacct rows, logs, finals, eval JSONs, sample JSONLs, and quota. Inspect the first A100-80 eval runtime. If an eval approaches or hits 20--24 hours, preserve the protocol and make only unfinished rows depth-sharded/resumable. Recover only missing/failed rows.
2. Before accepting metrics, inspect representative raw generations across logic/NL, all seeds, train ranges, shallow/train/OOD/depth-50 cases, successes, failures, and cap hits. Verify prompts, answer extraction, citation-free validity, token limits, and unique constants. Aggregate mean/std for answer correctness and citation-free correct-and-valid performance. Compare greedy/pass@1 first, then pass@4/8/16.
3. Restore or reject the BranchProof narrative from the corrected three-seed evidence. Keep all old wrapped-constant BranchProof performance and ablations quarantined. Update the official preprint only after artifact and qualitative audits pass.
4. Finish the matched Nanotron p15 normal/logic/NL comparison. Verify every timeout/recovery checkpoint, optimizer/scheduler/RNG/data offsets, exact 15% blend, loss, throughput, and quota. Do not replay consumed tokens. After successful final checkpoints, verify HF conversion/upload and path-guarded local checkpoint cleanup.
5. Run and audit both downstream branches for all three conditions: direct reviewer evaluation, then native-Qwen-chat instruction SFT and the same post-instruction evaluation. Inspect benchmark samples and separate direct from post-instruction claims.
6. Conditional expansion: if corrected BranchProof shows a robust formal advantage at greedy/pass@1, submit only the highest-value corrected reruns first: one syntax control, one shortcut experiment, conditioned dual, and one independent architecture. If matched Nanotron p15 transfer is positive, submit corrected {logic,nl_exact} x {5,10,15,20,25} mixtures. If a result is negative, inspect generations and revise the claim before spending on the broader grid.
7. Report work: regenerate the in-repo report when justified, mirror only to ../synthetic-RLVL-report/informal_report, and update the official root preprint with corrected evidence, clear mean/std tables/figures, and concise claims. Preserve AttrCon as independent evidence; keep iGSM and Maze outside the main story.

Operational rules:
- Treat this as plan-driven work, not merely a queue check. Apply safe fixes, scheduler edits, targeted resubmissions, analyses, report updates, and explicitly triggered submissions in the same pass.
- Do not append no-op watcher entries to docs. Update handoff docs only for actual state changes, failures, submissions, results, insights, or decisions.
- Inspect raw generations before scientific conclusions. Question evaluator behavior before accepting surprising metrics.
- Keep BranchProof eval A100-80-only. Do not reduce the scientific protocol to fit runtime.
- Watch HPCVAULT usage because complete Nanotron checkpoints are about 199G each. Delete checkpoints only through the guarded post-upload path or when the docs prove they are stale and no recovery depends on them.
- Commit and push this repo and ../synthetic-RLVL-report whenever changes are made and authentication permits.
- End each pass with a concise status including exact job IDs, states, actions, artifacts, findings, and the next trigger."
