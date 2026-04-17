# SFT Sweep Status (2026-04-12)

## Latest Arrays

- `3527879` (`sft_sweep_lr`, array `0-20`): `21 COMPLETED`
- `3527865` (`sft_sweep_bsz`, array `0-17`): `7 COMPLETED`, `6 FAILED`, `5 NODE_FAIL`

New template arrays submitted on 2026-04-12:
- `3532653` (`sft_nl_exact`, array `0-2`)
- `3532654` (`sft_formal_then_nl`, array `0-2`)
- `3532655` (`sft_nl_then_formal`, array `0-2`)

## Batch-Size Sweep Outcome (`3527865`)

Mapping: `task = 3 * hp_idx + seed_idx`, seeds `{3407,3408,3409}`, batch sizes `{1,4,8,16,32,64}`.

- `bsz=1` (tasks `0..2`): `NODE_FAIL` on node `a0533` (no model traceback in task logs).
- `bsz=4` (tasks `3..5`): two `NODE_FAIL` (`3,4`), one completed (`5`).
- `bsz=8` (tasks `6..8`): all completed.
- `bsz=16` (tasks `9..11`): all completed.
- `bsz=32` (tasks `12..14`): all failed with `torch.OutOfMemoryError`.
- `bsz=64` (tasks `15..17`): all failed with `torch.OutOfMemoryError`.

Practical decision:
- Drop `bsz=1` (duplicates the `lr=3e-5, bsz=1` condition from LR sweep).
- Drop `bsz in {32,64}` (persistent OOM on A100 80GB).
- Keep maintained BSZ sweep at `{4,8,16}`.

## LR Sweep Results (`3527879`)

Setup: one epoch (`train.num_epochs=1`, `train.max_steps=-1`), `bsz=1`, `train_samples=10000`, 3 seeds.

Mean final `eval_loss` by LR:

- `1e-5`: `0.505568`
- `3e-5`: `0.436907`
- `5e-5`: `0.414235`
- `7e-5`: `0.391132`
- `1e-4`: `0.384218` (best)
- `3e-4`: `0.464878`
- `5e-4`: `0.869841`

## BSZ Results On Successful Runs Only

- `bsz=4`: `mean eval_loss=0.440043` (`1/3` seeds completed)
- `bsz=8`: `mean eval_loss=0.441305` (`3/3`)
- `bsz=16`: `mean eval_loss=0.452232` (`3/3`)

## Notes

- Older arrays from 2026-04-09..2026-04-11 are superseded by `3527879`/`3527865` and archived only for provenance.
- Current sweep scripts now use one-epoch budgets instead of the earlier fixed-`10000`-step regime.
