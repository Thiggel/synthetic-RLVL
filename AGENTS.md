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

## Current Live Runs (as of 2026-05-20 09:25 CEST)

- HFSA depth-scaling dataset build: `3623863`
  - builds/pushes `flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget-depth50`.
  - completed.
- Cancelled/replaced old pending arrays:
  - `3623864`, `3623865`, `3624535`, `3624536`.
- HFSA depth-scaling SFT array: `3634790_[0-29%6]`
  - grid: templates `logic,nl_exact`; train depths `1..5,1..10,1..15,1..20,1..25`; seeds `3407,3408,3409`.
  - patched to retain up to 12 LoRA checkpoints.
  - current: `3634790_0` is running (`logic_train1to5_10k_seed3407`); remaining rows are pending by priority/resources.
- HFSA depth-scaling final merge/pass@k eval array: `3634791_[0-29%6]`
  - dependency: `aftercorr:3634790`.
  - eval depths `1..50`, 128 prompts/depth, 16 generations/prompt.
- HFSA depth-scaling intermediate checkpoint eval array: `3634792_[0-29%2]`
  - dependency: `aftercorr:3634790`.
  - default checkpoint steps: `1000,3000,10000`.
- 1k-sample sanity SFT array: `3634793_[0-9%5]`
  - seed `3407`; templates `logic,nl_exact`; train depths `1..5,1..10,1..15,1..20,1..25`.
  - status: 9/10 rows completed; `nl_exact_train1to25_1k_seed3407` failed with CUDA OOM.
  - retry: `3643001_[9]`, with gradient checkpointing enabled.
- 1k-sample sanity eval array: `3634794_[0-9%5]`
  - dependency: `aftercorr:3634793`.
  - status: rows `0..8` pending by priority; broken row `9` was cancelled.
  - retry eval: `3643002_[9]`, dependency `afterok:3643001`.

Checkpoint note:
- Jobs submitted before the 2026-05-19 script patch likely saved every 1000 steps but retained only `train.save_total_limit=2`.
- The patched HFSA depth-scaling SFT script now uses `train.save_total_limit=12` by default for resubmission.
- SFT now supports `train.gradient_checkpointing`; the fixed-target config defaults to `auto`, enabling checkpointing for long `nl_exact` depth-25 rows after the 1k OOM. `scripts/env.sh` also sets `PYTORCH_ALLOC_CONF=expandable_segments:True`.
- The affected arrays have been cancelled/resubmitted from patched scripts.
- Merge/eval scripts delete merged full-model checkpoints by default; do not accumulate merged OLMo-7B dirs in `${WORK}/synthetic-RLVL/tmp`.

Check live status:

```bash
squeue -u c107fa12 -o '%.18i %.9P %.32j %.2t %.10M %.6D %R'
sacct -j <jobid> --format=JobIDRaw,JobName%34,State,Elapsed,ExitCode -n -P
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
