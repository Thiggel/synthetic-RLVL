#!/usr/bin/env bash
#SBATCH --job-name=bp_nano_oversight
#SBATCH --partition=a100mig
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
- Corrected BranchProof full SFT 3829072_[0-29%12], followed by validity-fixed eval 3838163_[0-29%6]. Eval must remain on partition a100 with feature a100_80 and retain 32 prompts/depth, 16 generations, all 14 depths, pass@1/2/4/8/16, context 16384, equal cap 7168, and two sampled generations per prompt for qualitative review. Old eval 3834582 was canceled after rows 1/2 had loaded the pre-fix scorer; completed row 0 is quarantined outside the aggregate input path.
- CPU-only BranchProof row audits 3847756_[0-29%8] follow validity-fixed eval rows with aftercorr dependencies; replacement aggregation and cross-grid qualitative audit 3847757 run only after all 30 audits pass. GPU-requesting audit/aggregate 3838164/3838165 and old pre-fix 3834706/3835779 were canceled. Recover any failed eval/audit rows before aggregation rather than bypassing either gate.
- The validity fix requires every generated premise to parse before strict or citation-free proof validity can be true. The row audit also rejects any retained citation-free-valid sample carrying a validity error, invalid proof line, or line-valid fraction below one. Never mix the quarantined pre-fix row-0 metrics with corrected rows.
- Corrected BranchProof SFT is complete at 30/30 and all final adapters passed the artifact gate. Eval rows 0..5 started on verified A100-80GB devices at 10:07--10:11 CEST on 2026-07-13; rows 6..29 are array-throttle pending. Monitor the six active rows and first completed artifact rather than editing their now-resolved dependencies.
- Downstream-suite final smoke 3834836 must pass before production eval. FOLIO is unavailable, legacy LogiQA tasks use unsupported scripts, and installed FLD prompts for proofs while exact-matching labels, creating an extraction floor. The validated suite uses gsm8k, hendrycks_math500, arc_challenge, hellaswag, winogrande, piqa, agieval_logiqa_en, bbh, mmlu, and mmlu_pro.
- Nanotron normal control 3823434_0 -> failed W&B-only recovery 3828946_0 -> completed replacement recovery 3835438_0. Upload 3831119 failed because the converter lacked torchrun; repaired upload 3847569 passed conversion, remote parity, and full CUDA forward, then guarded cleanup ran. Direct eval 3835927 failed on a transient hf_transfer shard-download 403; replacement 3847792 completed and passed the strict artifact gate. Instruction SFT 3831121 was canceled after the uploaded Transformers-5 tokenizer metadata proved incompatible with the downstream Transformers-4 stack; repaired replacement 3847661 completed. Instruction eval 3835928 produced a complete bundle but exited failed on the stale task-order/MATH gate. Corrected schema-v4 CPU gate 3849774 accepts that bundle; do not rerun its GPU evaluation. Old control evals 3834906/3834907 remain canceled.
- Corrected Nanotron logic p15 3830927_3 -> replacement recovery 3835442_3 -> repaired-payload upload 3847802, direct eval 3847804, instruction SFT 3847805, instruction eval 3847806. Dependency-held jobs 3831123/3834908/3831125/3834909 were canceled before start after their stored Slurm payloads proved to predate the converter, downloader, and resume fixes. Superseded untouched recovery 3830928_3 was canceled.
- Corrected Nanotron NL p15 3831111_8 -> completed replacement recovery 3835443_8. Upload 3831113 failed on the same missing torchrun environment; repaired upload 3847570 passed all gates and guarded cleanup. Direct eval 3834904 is accepted. Instruction SFT 3831115 failed on pre-repair tokenizer metadata; replacement 3847662 completed. Instruction eval 3834905 wrote a complete bundle but exited on the stale MATH gate; corrected CPU gate 3849775 accepts it. Superseded untouched recovery 3831112_8 remains canceled.
- CPU-only strict downstream aggregation 3849776 waits only on logic evals 3847804/3847806; completed schema-v4 CPU gates 3849774/3849775 substitute for the artifact-complete but failed-state instruction eval parents, while NL direct 3834904 and control direct 3847792 are already accepted. Earlier aggregates 3836159/3847793/3847807/3849683/3849736 were superseded. Do not analyze an incomplete subset as the matched pilot.
- Logic/NL first allocations checkpoint only at step 4096 and cannot reach step 8192 within 24 hours. As soon as each step-4096 tree has nonempty metadata/config/model config, no zero-byte files, and four equal optimizer shards above 20 GB, cancel only that corresponding parent allocation so its after-any recovery can resume at 4096. Do not let it spend the remaining approximately five hours on work that cannot be checkpointed before timeout, and do not cancel before the complete checkpoint is verified.

Plan to execute end to end:
1. Complete corrected BranchProof SFT/eval. Check squeue, expanded sacct rows, logs, finals, eval JSONs, sample JSONLs, and quota. Inspect the first A100-80 eval runtime. If an eval approaches or hits 20--24 hours, preserve the protocol and make only unfinished rows depth-sharded/resumable. Recover only missing/failed rows.
2. Before accepting metrics, inspect representative raw generations across logic/NL, all seeds, train ranges, shallow/train/OOD/depth-50 cases, successes, failures, and cap hits. Verify prompts, answer extraction, citation-free validity, token limits, and unique constants. Aggregate mean/std for answer correctness and citation-free correct-and-valid performance. Compare greedy/pass@1 first, then pass@4/8/16.
3. Restore or reject the BranchProof narrative from the corrected three-seed evidence. Keep all old wrapped-constant BranchProof performance and ablations quarantined. Update the official preprint only after artifact and qualitative audits pass.
4. Finish the matched Nanotron p15 normal/logic/NL comparison. Verify every timeout/recovery checkpoint, optimizer/scheduler/RNG/data offsets, exact 15% blend, loss, throughput, and quota. Apply the verified step-4096 parent-cancel transition above so recoveries do not replay post-checkpoint work. After successful final checkpoints, verify HF conversion/upload and path-guarded local checkpoint cleanup.
5. Run and audit both downstream branches for all three conditions: direct reviewer evaluation, then native-Qwen-chat instruction SFT and the same post-instruction evaluation. Inspect benchmark samples and separate direct from post-instruction claims.
6. Conditional expansion: if corrected BranchProof shows a robust formal advantage at greedy/pass@1, submit only the highest-value corrected reruns first: one syntax control, one shortcut experiment, conditioned dual, and one independent architecture. If matched Nanotron p15 transfer is positive, submit corrected {logic,nl_exact} x {5,10,15,20,25} mixtures. If a result is negative, inspect generations and revise the claim before spending on the broader grid.
7. Report work: regenerate the in-repo report when justified, mirror only to ../synthetic-RLVL-report/informal_report, and update the official root preprint with corrected evidence, clear mean/std tables/figures, and concise claims. Preserve AttrCon as independent evidence; keep iGSM and Maze outside the main story.

Operational rules:
- This oversight job is intentionally CPU-only on `a100mig`; do not add a GRES request. A no-GRES probe completed while the account GPU limit was full, so CPU-only scheduling keeps the six-hour chain independent of experiment GPU occupancy.
- Treat this as plan-driven work, not merely a queue check. Apply safe fixes, scheduler edits, targeted resubmissions, analyses, report updates, and explicitly triggered submissions in the same pass.
- Do not append no-op watcher entries to docs. Update handoff docs only for actual state changes, failures, submissions, results, insights, or decisions.
- Inspect raw generations before scientific conclusions. Question evaluator behavior before accepting surprising metrics.
- Keep BranchProof eval A100-80-only. Do not reduce the scientific protocol to fit runtime.
- Do not apply the generic fix_mistral_regex tokenizer suggestion to BranchProof: base-training and merged-eval default token IDs match on 704/704 audited task texts, while the suggested fix changes 640/704 and would create a train/eval mismatch.
- For Nanotron MATH-500, require the audit-generated schema-v4 `answer_prefix_math_verify,none` sidecar and keep stock `exact_match,none` only as a format diagnostic. The scorer uses the first answer line when explicit and otherwise ranks an answer-cued delimited expression above later plain-text suffixes; generated `Problem:` and `Question:` records are excluded. Do not use whole-response symbolic matching; it can credit later prompt leakage and shared scalars from wrong structured answers.
- For every completed Nanotron downstream bundle, inspect correct and incorrect raw generations and compare the aggregate generation diagnostics across conditions and branches. NL-direct currently has BBH/MMLU-Pro invalid-extraction rates 9.1%/20.5% and generated next-document marker rates 22.9%/3.7%, with zero prompt-marker incidence; determine whether logic changes these rates and whether instruction SFT removes them before interpreting modality transfer.
- Watch HPCVAULT usage because complete Nanotron checkpoints are about 199G each. Delete checkpoints only through the guarded post-upload path or when the docs prove they are stale and no recovery depends on them.
- Authenticated HF usedStorage is 63.610 GB after the control instruction-adapter upload and the verified cleanup. The pending logic upload projects 78.852 GB; one further full checkpoint fits at 94.095 GB, but two would reach 109.338 GB. Preserve the active control/NL/logic repos through all six evaluations and preserve the latest autoformalization adapter for each task. Before the broader mixture grid, implement guarded upload/evaluate/audit/delete checkpoint rotation; never delete a repo referenced by a pending/running job, and archive cross-project artifacts before removal.
- Commit and push this repo and ../synthetic-RLVL-report whenever changes are made and authentication permits.
- End each pass with a concise status including exact job IDs, states, actions, artifacts, findings, and the next trigger."
