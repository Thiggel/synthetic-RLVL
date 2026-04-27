# Posttraining Status (2026-04-22)

## Latest Update (2026-04-22)

- Retry array `3556764` (`posttrain_remain_2seed`, `array=0-5,7-26,29%4`) is active.
- Started retry tasks currently:
  - `RUNNING`: `3556764_2`, `3556764_3`, `3556764_4`, `3556764_5`
  - `FAILED`: `3556764_0`, `3556764_1` (Ray startup timeout: `The current node timed out during startup`)
- Legacy task still running from older submission:
  - `3555102_6` (`RUNNING`)
- New 3-seed reward run submitted:
  - `3557014` (`posttrain_rw_line_valid`, `array=0-2%3`, currently `PENDING`)
  - Schema: `correct_plus_line_valid_plus_0p1_format`
  - Reward: `R = correctness + line_valid_fraction + 0.1 * format`

## Note: Ray Startup Timeouts (Observed + Mitigations)

- Observed pattern: some array tasks fail early with Ray init timeout (`RPC error: Deadline Exceeded` / `The current node timed out during startup`) while later tasks with the same script can succeed, including on the same node.
- Interpretation: likely transient startup/load contention rather than a deterministic script bug.
- Mitigations for future submits:
  - Keep/strengthen array throttling (e.g. `%4` or `%2`).
  - Add randomized startup jitter (for example `sleep $((RANDOM % 30))`).
  - Increase Ray startup wait (`RAY_raylet_start_wait_time_s=120`).
  - Optional: run launch command via `srun --ntasks=1` for stronger step isolation on this cluster.

## Submitted Arrays (Current)

- `3548386` (`grpo_mrg_ablate_prompts`, `array=0-11`)
- `3548388` (`grpo_mrg_ablate_rollouts`, `array=0-11`)
- `3548387` (`posttrain_reward_ablation`, `array=0-20`)

These are the active 3-seed sweeps after migrating to pre-merged checkpoints.

## Checkpoint Policy

All current posttraining sweeps use pre-merged per-seed model directories:

- `/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/merged_sft_lr1e-4_seed3407`
- `/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/merged_sft_lr1e-4_seed3408`
- `/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/merged_sft_lr1e-4_seed3409`

Runtime config uses:
- `model.path=<merged_dir>`
- `model.merge_adapter_for_rollout=false`

No runtime LoRA merge step is required for these arrays.

## Array Semantics

### Prompts Ablation (`3548386`)

- Seeds: `{3407, 3408, 3409}`
- Prompt set: `{8, 16, 32, 64}`
- Rollouts fixed: `8`
- Mapping:
  - `seed_idx = task_id / 4`
  - `prompt_idx = task_id % 4`

### Rollouts Ablation (`3548388`)

- Seeds: `{3407, 3408, 3409}`
- Rollout set: `{8, 16, 32, 64}`
- Prompts fixed: `8`
- Mapping:
  - `seed_idx = task_id / 4`
  - `rollout_idx = task_id % 4`

### Reward Ablation (`3548387`)

- Seeds: `{3407, 3408, 3409}`
- Schemas:
  - `correct_plus_0p1_format`
  - `indicator_correct_and_format`
  - `correct_plus_valid_plus_0p1_format`
  - `correct_plus_line_valid_plus_0p1_format` (new run submitted separately as `3557014`)
  - `correct_plus_0p75_valid_plus_0p1_format`
  - `correct_plus_0p5_valid_plus_0p1_format`
  - `correct_plus_0p25_valid_plus_0p1_format`
  - `indicator_all`
- Mapping:
  - `seed_idx = task_id / 7`
  - `schema_idx = task_id % 7`

## Shared Runtime Settings

- `grpo.max_num_batched_tokens=16384`
- `optim.micro_batch_size=4`
- `optim.logprob_micro_batch_size=4`
- `task.template=logic`
- `task.prefill=none`

## Prior Arrays (Superseded / Canceled)

- `3546217` (single-seed rollout ablation) canceled after partial failures.
- `3546218` (single-seed prompts ablation) canceled after partial failures.
- `3548365` (earlier reward array run before final sweep alignment) canceled/replaced.

## Snapshot at Update Time

From `sacct -j 3548386,3548387,3548388` at update:
- `RUNNING`: 26 tasks
- `FAILED`: 19 tasks

For live state, run:

```bash
squeue -u c107fa12 -j 3548386,3548387,3548388
sacct -j 3548386,3548387,3548388 --format=JobID,JobName%30,State,ExitCode,Elapsed
```
