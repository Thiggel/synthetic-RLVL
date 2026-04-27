# AGENTS.md

This is an ML research project.
Main Research Questions:
- Can we endow LLMs with a different substrate for Chain of Thought (Formal Logic)?
- Does it improve reasoning to reward the validity of logic CoT during RL posttraining?
- Is logic cot better than natural language cot?
- is a hybrid of logic and natural language cot better than either alone?

## Scope

- Synthetic logic dataset generation + proof validation.
- SFT training (`train_sft.py`).
- GRPO/VERL posttrain (`posttrain_grpo_verl.py`).
- Slurm-based sweep orchestration with W&B tracking.

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

## Current Live Runs (as of 2026-04-22)

- Retry array (older): `3556764`
  - active tasks still running: `_2,_3,_4,_5`
- New line-valid reward 3-seed run: `3557014`
  - running tasks: `_0,_1,_2`
- Jittered resubmit array: `3557214`
  - submitted for failed/pending remaining-seed indices with higher concurrency and startup jitter.

Check live status:

```bash
squeue -u c107fa12 -o '%.18i %.9P %.32j %.2t %.10M %.6D %R'
sacct -j <jobid> --format=JobIDRaw,JobName%34,State,Elapsed,ExitCode -n -P
```

## Primary Docs

- `README.md`
- `docs/current_system_state.md`
- `docs/posttrain_status_2026-04-18.md`
- `docs/runtime_env.md`

# Code Review

Claude Code will review all the code you write after you write it.
