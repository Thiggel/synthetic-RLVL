# Synthetic-RLVL: Current System State (2026-04-12)

## Summary

- Materialized-dataset SFT flow is stable and active.
- LR sweep (`3527879`) completed across all 21 runs.
- BSZ sweep (`3527865`) completed partially; high BSZ settings OOM and low-BSZ tasks had node failures.
- Three new template ablations at `lr=1e-4`, `bsz=1` were submitted:
  - `3532653` (`sft_nl_exact`)
  - `3532654` (`sft_formal_then_nl`)
  - `3532655` (`sft_nl_then_formal`)

## SFT Template Modes

Available `task.template` values:
- `logic`
- `natural`
- `logic_natural`
- `natural_logic`
- `nl_exact` (`<think>` block with NL premises/proof/conclusion)
- `formal_think` (`<formal>` then `<think>`, answer at end)
- `think_formal` (`<think>` then `<formal>`, answer at end)

## Sweep Policy (Current)

- LR sweep baseline remains: `bsz=1`, `train_samples=10000`, 3 seeds.
- Maintained BSZ sweep is now restricted to `{4,8,16}`.
- `bsz=1` is treated as redundant with LR sweep condition `lr=3e-5, bsz=1`.
- `bsz in {32,64}` is excluded due to repeated OOM on A100 80GB.

## Runtime Conventions

- Slurm scripts source `scripts/env.sh`.
- Preferred venv: `$HPCVAULT/.venv_rlvl_posttrain`; fallback: `$WORK/.venv`.
- W&B project for SFT sweeps: `synthetic-rlvr-sft`.
