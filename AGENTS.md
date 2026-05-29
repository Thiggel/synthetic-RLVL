# AGENTS.md

This is an ML research project.
Main Research Questions:
- Can we endow LLMs with a different substrate for Chain of Thought (Formal Logic)?
- Is logic cot better than natural language cot?
- is a hybrid of logic and natural language cot better than either alone?
- Does formal-logic CoT improve length extrapolation and downstream reasoning transfer under SFT/midtraining?
- Old/paused direction: does it improve reasoning to reward the validity of logic CoT during RL posttraining?

## Scope

- Synthetic logic dataset generation + proof validation.
- SFT training (`train_sft.py`).
- Formal-vs-natural-language CoT depth-scaling experiments.
- Real benchmark evaluation for synthetic SFT and midtraining transfer.
- GRPO/VERL posttrain (`posttrain_grpo_verl.py`).
- Slurm-based sweep orchestration with W&B tracking.

## Active Plan

- Current active research plan:
  - `docs/formal_logic_cot_research_plan_2026-05-19.md`
- Archived RL validity-reward state:
  - `docs/old_rl_validity_reward_direction_2026-05-19.md`
- Current depth-scaling implementation plan:
  - `docs/hfsa_depth_scaling_plan_2026-05-19.md`

## Environment

- Always bootstrap env first:
  - `source ./scripts/env.sh`
- Preferred venv:
  - `$HPCVAULT/.venv_rlvl_posttrain`
  - fallback: `$WORK/.venv`
- If `pytest` entrypoint is broken (stale shebang), run:
  - `${HPCVAULT}/.venv_rlvl_posttrain/bin/python -m pytest ...`

## Data + Checkpoints

- Materialized dataset docs:
  - `docs/materialized_dataset.md`
- Current posttrain policy:
  - use pre-merged per-seed SFT checkpoints, not raw LoRA adapter dirs.
- Common merged checkpoints:
  - `/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/merged_sft_lr1e-4_seed3407`
  - `/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/merged_sft_lr1e-4_seed3408`
  - `/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/merged_sft_lr1e-4_seed3409`
- For newer reruns, some jobs use:
  - `/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/merged_sft_lr1e-4_seed{seed}_remerge_20260421`

## Key Commands

- Quick tests:
  - `python -m pytest -q`
- SFT:
  - `python train_sft.py run_name=sft_logic_seed3407 seed=3407`
- Posttrain:
  - `python posttrain_grpo_verl.py run_name=rl_logic_seed3407 seed=3407`
- lm-eval wrapper:
  - `${HPCVAULT}/.venv_rlvl_posttrain/bin/python scripts/evaluate_lm_eval.py --checkpoint <model_or_path> --tasks gsm8k --output-path <out>`
- Paired synthetic materialization:
  - `${HPCVAULT}/.venv_rlvl_posttrain/bin/python scripts/data/build_paired_synthetic_dataset.py --kind graph_traversal --output-root <out>`

## Reward Schemas

Supported `reward.schema` values:

- `correct_plus_0p1_format`
- `indicator_correct_and_format`
- `correct_plus_valid_plus_0p1_format`
- `correct_plus_line_valid_plus_0p1_format`
- `correct_plus_0p75_valid_plus_0p1_format`
- `correct_plus_0p5_valid_plus_0p1_format`
- `correct_plus_0p25_valid_plus_0p1_format`
- `indicator_all`

`correct_plus_line_valid_plus_0p1_format` implements:
- `R = correctness + line_valid_fraction + 0.1 * format`
- where `line_valid_fraction = (# valid proof lines)/(# proof lines)` from `LogicEngine.analyze_proof`.

## Slurm Structure

- Single jobs: `scripts/slurm/jobs/`
- Sweeps: `scripts/slurm/sweeps/`
- Layout docs: `docs/slurm_layout.md`

## W&B Conventions

- Use both env vars in job scripts:
  - `WANDB_GROUP`
  - `WANDB_RUN_GROUP`
- Project default:
  - `synthetic-rlvl`

## Handoff Discipline

Update the handoff docs whenever operational or scientific state changes.

- Keep `docs/current_system_state.md` as the shortest current operational truth.
- Keep `docs/running_experiments.md` as the live Slurm dashboard.
- Keep `docs/experiment_backlog.md` as the planned-experiment backlog. Add new experiment ideas from the conversation, remove or mark completed items, and record the trigger for when each item should run.
- Keep `docs/project_log.md` as the short dated log for useful operational events, cleanup decisions, results updates, and handoff changes.
- Move historical detail to experiment-specific docs or archive docs instead of appending long stale snapshots.
- If a new Slurm job is submitted, canceled, resubmitted, dependency-edited, or partition-edited, update `docs/current_system_state.md`, `docs/running_experiments.md`, and the relevant experiment doc.
- If implementation is changed or extended, update the relevant docs with what changed, why, and any verification run.
- If new analysis, results, failure modes, or research insights are found, record the concise takeaway and affected artifacts in the handoff docs.

## Report Discipline

- The ongoing LaTeX report lives outside this repo in `../synthetic-RLVL-report`.
- Treat `../synthetic-RLVL-report` as the primary reader-facing report repo. Keep generated analysis in this repo as reproducible artifacts, but move polished narrative, selected tables, selected figures, current system state summaries, and short useful logs into the report repo.
- Whenever new results, plots, tables, sample generations, or important insights are produced, update the LaTeX report and push the report repo to GitHub.
- If local TeX tooling is unavailable, still update the `.tex` sources and note that compilation was not run.
- Keep raw or bulky artifacts out of the report repo unless they are explicitly needed by the LaTeX source.

## GitHub Push Discipline

- After making code, Slurm, docs, or report changes, commit and push the affected repo to GitHub yourself when network/authentication permits.
- For this repo, push changes to `git@github.com:Thiggel/synthetic-RLVL.git`.
- For the report repo, push changes to `git@github.com:Thiggel/synthetic-RLVL-report.git`.
- If direct push to the current branch is inappropriate, push a clearly named branch and record it in the final user update. If pushing fails because credentials, network, or tooling are unavailable, state the exact blocker.

## Slurm Housekeeping

- When checking pending jobs, inspect whether compatible partitions are free or likely to start sooner.
- If a pending job can safely run on more partitions, use:
  - `scontrol update JobId=<jobid> Partition=<partition1,partition2>`
- Only widen to partitions compatible with the job's GPU, memory, walltime, account, and software constraints.
- Record any partition edits in `docs/running_experiments.md` and `docs/current_system_state.md`.

## Current Live Runs

Do not embed long live-run snapshots in this file. Current operational truth lives in:

- `docs/current_system_state.md`
- `docs/running_experiments.md`
- `docs/experiment_backlog.md`

Check live status with:

```bash
squeue -u c107fa12 -o '%.18i %.9P %.34j %.2t %.11M %.6D %R'
sacct -j <jobids> --format=JobIDRaw,JobName%34,State,Elapsed,ExitCode -n -P
```

## Primary Docs

- `README.md`
- `docs/current_system_state.md`
- `docs/running_experiments.md`
- `docs/experiment_backlog.md`
- `docs/project_log.md`
- `docs/formal_logic_cot_research_plan_2026-05-19.md`
- `docs/hfsa_depth_scaling_plan_2026-05-19.md`
- `docs/old_rl_validity_reward_direction_2026-05-19.md`
- `docs/posttrain_status_2026-04-18.md`
- `docs/runtime_env.md`
- `../synthetic-RLVL-report/main.tex`

# Code Review

Claude Code will review all the code you write after you write it.

# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
