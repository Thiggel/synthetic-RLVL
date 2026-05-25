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

Snapshot as of 2026-05-25 10:39 CEST:

| Stage | Jobs | State | Note |
| --- | --- | --- | --- |
| HFSA 10k SFT | `3646736_[0-6]`, `3647379_[7-29%12]` | completed | all 30 main rows covered; row 0 skipped due existing final checkpoint; executed rows exit `0:0` |
| Old full-grid eval arrays | `3647708`, `3648279`, `3648280`, `3647711`, `3647712` | canceled | canceled 2026-05-22 10:47 CEST after sparse runtime patch |
| Sparse final eval | `3650951_[0-29%10]` | completed | 30/30 JSON files, all tasks exit `0:0` |
| Sparse intermediate eval | `3650952_[0,3,6,9,12,15,18,21,24,27%4]` | completed | seed-3407 checkpoint curves; 30/30 JSON files, all tasks exit `0:0` |
| Paired train-10 materialization | `3656210_1`, `3656308_0` | completed | `attribute_constraints` completed; `maze_navigation` completed after fixing depth-15 room vocabulary |
| Paired train-10 SFT/eval pilot | SFT `3656309`, retries `3657088`/`3657738`; eval `3656310`, `3657739` | attribute complete, maze eval running | maze SFT recovered; maze eval rows `3657739_0,1` are long-running and hitting generation caps |
| Hard attribute replacement | build `3659338`, SFT `3659339_[0-1%2]`, eval `3659340_[0-1%2]` | build completed, SFT running, eval dependency-pending | saturated `attribute_constraints` was hardened and resubmitted after local validation through depth 50 |
| Qwen HFSA model ablation | `3656217_[0-17%3]`, `3656218_[0-17%3]` | SFT rows 0-14 complete, 15-17 running; eval rows 0-12 complete, 13-14 running | `Qwen/Qwen2.5-7B`; train depths `1..10`, `1..20`, `1..25`; both templates; three seeds |
| Small-extra HFSA model ablation | `3656323_[0-35%4]`, retries `3656359_2` and `3656387_3`, eval `3656389_[0-35%4]` | rows 0,1,4-25 complete or recovered; 26-29 running; 30-35 pending | `Qwen/Qwen2.5-1.5B` and `google/gemma-3-4b-pt`; failed rows 2/3 recovered by retries |
| OLMo-32B pilot | `3656335_[0-1%1]`, replacement eval `3658461_[0-1%1]` | row 0 complete, row 1 running, eval pending | `allenai/OLMo-2-0325-32B`, train depth `1..20`, seed 3407, logic vs NL |
| Tiny Llama scratch pretraining | `3656338`, retries `3656360_1`/`3656388_0`, eval `3656390_[0-5%3]` | completed | random-init Llama3-tokenizer configs at `50M/100M/200M`; eval completed but joint/depth-50 metrics are zero |
| Tiny Llama checkpoint eval | original `3659405`, replacement `3659415_[0-11%3]` | completed | checkpoint-10000/15000 pass@k for tiny training curves; original failed due missing tokenizer files in Trainer checkpoints and script now stages tokenizer metadata from `final/` |
| OOD lm-eval | pilots `3659344`/`3659348`, broad `3659356_[0-89%4]`, OLMo32 `3659357_[0-1%1]`, tiny replacement `3659488_[0-5%3]` | pilots and tiny replacement completed; larger arrays dependency-pending | tag-aware GSM8K/HotpotQA/2Wiki/MuSiQue suite implemented; tiny replacement uses 8192 context after vLLM long-context assert |
| Codex HFSA follow-up oversight | through `3658813`, next `3659047` | completed / begin-time pending | `scripts/slurm/codex/hfsa_followup_oversight_2026-05-24.slurm`; self-schedules 4h follow-up passes up to `OVERSIGHT_MAX_HOPS=18` |
| Paired dataset audit | local materialization audits | mixed | `maze_navigation` and `attribute_constraints` pass; `official_igsm` blocked by subtraction proof validation |

Check live status:

```bash
squeue -u c107fa12 -o '%.18i %.9P %.34j %.2t %.11M %.6D %R'
sacct -j 3650951,3650952,3656210,3656308,3656309,3656310,3657088,3657738,3657739,3656217,3656218,3656323,3656359,3656387,3656389,3656335,3656336,3658461,3656338,3656360,3656388,3656390,3656509,3656510,3657079,3657734,3658457,3658813,3659047,3659338,3659339,3659340,3659344,3659348,3659356,3659357,3659392,3659405,3659415,3659488 --format=JobIDRaw,JobName%34,State,Elapsed,ExitCode -n -P
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
