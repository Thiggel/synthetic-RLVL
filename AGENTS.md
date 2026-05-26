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

- If a new Slurm job is submitted, canceled, resubmitted, or dependency-edited, update `docs/current_system_state.md` and the relevant experiment doc.
- If implementation is changed or extended, update the relevant docs with what changed, why, and any verification run.
- If new analysis, results, failure modes, or research insights are found, record the concise takeaway and affected artifacts in the handoff docs.
- Keep `docs/current_system_state.md` as the shortest current operational truth; move historical detail to experiment-specific or archived docs instead of appending long stale snapshots.

## Current Live Runs

Current operational truth lives in `docs/current_system_state.md`; update it whenever jobs, code, or analysis state changes.

Snapshot as of 2026-05-25 18:07 CEST:

| Stage | Jobs | State | Note |
| --- | --- | --- | --- |
| HFSA 10k SFT | `3646736_[0-6]`, `3647379_[7-29%12]` | completed | all 30 main rows covered; row 0 skipped due existing final checkpoint; executed rows exit `0:0` |
| Old full-grid eval arrays | `3647708`, `3648279`, `3648280`, `3647711`, `3647712` | canceled | canceled 2026-05-22 10:47 CEST after sparse runtime patch |
| Sparse final eval | `3650951_[0-29%10]` | completed | 30/30 JSON files, all tasks exit `0:0` |
| Sparse intermediate eval | `3650952_[0,3,6,9,12,15,18,21,24,27%4]` | completed | seed-3407 checkpoint curves; 30/30 JSON files, all tasks exit `0:0` |
| Dense intermediate eval backfill | `3660813_[0,3,6,9,12,15,18,21,24,27%2]` | running | seed-3407 1k-grid checkpoint backfill with `CHECKPOINT_STEPS=1000,2000,...,10000`; existing outputs skip |
| Paired train-10 materialization | `3656210_1`, `3656308_0` | completed | `attribute_constraints` completed; `maze_navigation` completed after fixing depth-15 room vocabulary |
| Paired train-10 SFT/eval pilot | SFT `3656309`, retries `3657088`/`3657738`; eval `3656310`, `3657739`, replacement `3659556_[0-1%2]` | attribute complete, maze eval replacement running | maze SFT recovered; first maze eval hit 16k context cap and replacement uses 32k context |
| Hard attribute replacement | build `3659338`, SFT `3659339_[0-1%2]`, eval `3659340_[0-1%2]` | build/SFT completed, eval running | saturated `attribute_constraints` was hardened and resubmitted after local validation through depth 50 |
| Qwen HFSA model ablation | `3656217_[0-17%3]`, `3656218_[0-17%3]` | SFT rows 0-14 complete, 15-17 running; eval rows 0-12 complete, 13-14 running | `Qwen/Qwen2.5-7B`; train depths `1..10`, `1..20`, `1..25`; both templates; three seeds |
| Small-extra HFSA model ablation | `3656323_[0-35%4]`, retries `3656359_2` and `3656387_3`, eval `3656389_[0-35%4]` | rows 0,1,4-25 complete or recovered; 26-29 running; 30-35 pending | `Qwen/Qwen2.5-1.5B` and `google/gemma-3-4b-pt`; failed rows 2/3 recovered by retries |
| OLMo-32B pilot | `3656335_[0-1%1]`, replacement eval `3658461_[0-1%1]` | row 0 complete, row 1 running, eval pending | `allenai/OLMo-2-0325-32B`, train depth `1..20`, seed 3407, logic vs NL |
| Tiny Llama scratch pretraining | seed-3407 `3656338`, retries `3656360_1`/`3656388_0`; new seeds `3659626`, `3659630` | seed 3407 completed; seeds 3408/3409 running | random-init Llama3-tokenizer configs at `50M/100M/200M`; missing seeds were submitted 2026-05-25 with final/checkpoint/OOD eval dependencies |
| Tiny Llama checkpoint/final eval | seed-3407 final `3656390` and checkpoint replacement `3659415`; seed deps `3659627`, `3659628`, `3659631`, `3659632` | seed 3407 completed; new seed evals dependency-pending | final sparse pass@k plus checkpoint-10000/15000 pass@k; checkpoint script stages tokenizer metadata from `final/` |
| OOD lm-eval | pilots `3659344`/`3659348`, broad `3659356_[0-89%4]`, OLMo32 `3659357_[0-1%1]`, tiny `3659488`, EM rerun `3659634`, tiny seed deps `3659629`/`3659633` | pilots and tiny seed-3407 replacement/EM rerun completed; larger/new seed arrays pending | tag-aware GSM8K/HotpotQA/2Wiki/MuSiQue suite implemented; LongBench now reports F1 and exact match |
| Codex HFSA follow-up oversight | through `3659047`, next `3659552` | running / begin-time pending | `scripts/slurm/codex/hfsa_followup_oversight_2026-05-24.slurm`; self-schedules 4h follow-up passes up to `OVERSIGHT_MAX_HOPS=18` |
| Paired dataset audit | local materialization audits | mixed | `maze_navigation` and `attribute_constraints` pass; `official_igsm` blocked by subtraction proof validation |

Check live status:

```bash
squeue -u c107fa12 -o '%.18i %.9P %.34j %.2t %.11M %.6D %R'
sacct -j 3650951,3650952,3660813,3656210,3656308,3656309,3656310,3657088,3657738,3657739,3659556,3656217,3656218,3656323,3656359,3656387,3656389,3656335,3656336,3658461,3656338,3656360,3656388,3656390,3656509,3656510,3657079,3657734,3658457,3658813,3659047,3659552,3659338,3659339,3659340,3659344,3659348,3659356,3659357,3659392,3659405,3659415,3659488,3659626,3659627,3659628,3659629,3659630,3659631,3659632,3659633,3659634 --format=JobIDRaw,JobName%34,State,Elapsed,ExitCode -n -P
```

## Primary Docs

- `README.md`
- `docs/current_system_state.md`
- `docs/formal_logic_cot_research_plan_2026-05-19.md`
- `docs/hfsa_depth_scaling_plan_2026-05-19.md`
- `docs/old_rl_validity_reward_direction_2026-05-19.md`
- `docs/posttrain_status_2026-04-18.md`
- `docs/runtime_env.md`

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
