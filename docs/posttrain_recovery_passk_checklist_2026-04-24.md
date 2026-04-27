# Posttrain Recovery + Pass@k Checklist (2026-04-24)

## Current Principle

Use only final actor checkpoints at `global_step_1500` or `global_step_1501` for the main comparison plots. Earlier checkpoints (`500`, `1000`) are useful for debugging trajectories, but they should not be mixed into the final pass@k comparison.

Full sampled pass@k is run post hoc, not during training, because the default evaluation is expensive:
- validation set: 20 depths x 20 prompts/depth = 400 prompts
- samples: 64 completions per prompt
- total completions per checkpoint: 25,600
- metrics: `pass@k` for `k in {1,2,4,8,16,32,64}`, plus `valid_given_correct@k`

## Submitted Jobs

- Catch-up posttrain array: `3561187`
  - Script: `scripts/slurm/jobs/posttrain_catchup_final_2026-04-24.slurm`
  - Manifest: `docs/manifests/posttrain_catchup_2026-04-24.tsv`
  - Array: `0-11%4`
  - Purpose: fill prompt/rollout gaps not covered by currently running array `3557214`.
- Merge + post-hoc pass@k array: `3561188`
  - Script: `scripts/slurm/jobs/posthoc_merge_eval_passk_2026-04-24.slurm`
  - Manifest: `docs/manifests/passk_final_eval_2026-04-24.tsv`
  - Array: `0-47%2`
  - Dependency: `afterany:3557214:3561187`
  - Purpose: for every final actor that exists, merge LoRA into a standalone HF checkpoint and run full pass@k evaluation.
- Oversight jobs:
  - `3561189`: starts at `now+4hours`
  - `3561190`: starts at `now+8hours`
  - `3561191`: starts at `now+12hours`
  - `3561192`: starts at `now+16hours`
  - `3561193`: starts at `now+20hours`
  - `3561194`: starts at `now+24hours`
  - Script: `scripts/slurm/codex/experiment_oversight.slurm`

## Final Checkpoint Inventory

Expected final coverage:
- Prompt sweep: 12 final actor checkpoints (`3 seeds x 4 num_prompts`).
- Rollout sweep: 12 final actor checkpoints (`3 seeds x 4 num_rollouts`).
- Reward sweep: 24 final actor checkpoints (`3 seeds x 8 reward schemas`).

Snapshot before submitting `3561187`:
- Prompt sweep: 3/12 final actors present.
- Rollout sweep: 7/12 final actors present.
- Reward sweep: 21/24 final actors present.

## Rerun Checklist

These had no usable actor checkpoint at inventory time and are rerun from the merged SFT seed checkpoint:

- [ ] Prompt sweep, seed `3407`, `num_prompts=64`, run `rl_grpo_merge_ablate_prompts64_rollouts8_seed3407`, submitted in `3561187`.
- [ ] Prompt sweep, seed `3408`, `num_prompts=64`, run `rl_grpo_merge_ablate_prompts64_rollouts8_seed3408`, submitted in `3561187`.
- [ ] Prompt sweep, seed `3409`, `num_prompts=16`, run `rl_grpo_merge_ablate_prompts16_rollouts8_seed3409`, submitted in `3561187`.
- [ ] Prompt sweep, seed `3409`, `num_prompts=32`, run `rl_grpo_merge_ablate_prompts32_rollouts8_seed3409`, submitted in `3561187`.
- [ ] Prompt sweep, seed `3409`, `num_prompts=64`, run `rl_grpo_merge_ablate_prompts64_rollouts8_seed3409`, submitted in `3561187`.
- [ ] Rollout sweep, seed `3409`, `num_rollouts=32`, run `rl_grpo_merge_ablate_prompts8_rollouts32_seed3409`, submitted in `3561187`.

## Continue Checklist

These had partial checkpoints but no final actor at inventory time and are continued with `resume.mode=auto`:

- [ ] Prompt sweep, seed `3407`, `num_prompts=16`, run `rl_grpo_merge_ablate_prompts16_rollouts8_seed3407`, submitted in `3561187`.
- [ ] Prompt sweep, seed `3407`, `num_prompts=32`, run `rl_grpo_merge_ablate_prompts32_rollouts8_seed3407`, submitted in `3561187`.
- [ ] Prompt sweep, seed `3408`, `num_prompts=16`, run `rl_grpo_merge_ablate_prompts16_rollouts8_seed3408`, submitted in `3561187`.
- [ ] Prompt sweep, seed `3408`, `num_prompts=32`, run `rl_grpo_merge_ablate_prompts32_rollouts8_seed3408`, submitted in `3561187`.
- [ ] Rollout sweep, seed `3407`, `num_rollouts=64`, run `rl_grpo_merge_ablate_prompts8_rollouts64_seed3407`, submitted in `3561187`.
- [ ] Rollout sweep, seed `3409`, `num_rollouts=64`, run `rl_grpo_merge_ablate_prompts8_rollouts64_seed3409`, submitted in `3561187`.

## Wait-On-Existing Checklist

These gaps were already covered by live retry array `3557214` at submission time, so they were not duplicated:

- [ ] Rollout sweep, seed `3408`, `num_rollouts=16`, task `3557214_20`, run `rl_grpo_merge_ablate_prompts8_rollouts16_seed3408`.
- [ ] Rollout sweep, seed `3408`, `num_rollouts=64`, task `3557214_22`, run `rl_grpo_merge_ablate_prompts8_rollouts64_seed3408`.
- [ ] Reward sweep, seed `3408`, `correct_plus_0p1_format`, task `3557214_23`, run `rl_reward_correct_plus_0p1_format_seed3408_mrg`.
- [ ] Reward sweep, seed `3408`, `correct_plus_valid_plus_0p1_format`, task `3557214_25`, run `rl_reward_correct_plus_valid_plus_0p1_format_seed3408_mrg`.
- [ ] Reward sweep, seed `3408`, `correct_plus_0p75_valid_plus_0p1_format`, task `3557214_26`, run `rl_reward_correct_plus_0p75_valid_plus_0p1_format_seed3408_mrg`.

## Merge Checklist

Merge is handled by dependent array `3561188`. For each row in `docs/manifests/passk_final_eval_2026-04-24.tsv`:

- [ ] Resolve final actor from `runs/<run_name>/global_step_1501/actor` or `global_step_1500/actor`.
- [ ] Merge actor LoRA into `/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/merged_actor_<run_name>_final`.
- [ ] Skip rows without a final actor so oversight can inspect and resubmit failures without crashing the whole array.

## Eval Checklist

Post-hoc eval is handled by dependent array `3561188`. For each final merged actor:

- [ ] Run `scripts/evaluate_checkpoint_passk.py` with `--num-generations 64` and `--k-values 1,2,4,8,16,32,64`.
- [ ] Save JSON metrics to `/home/atuin/c107fa/c107fa12/synthetic-RLVL/passk_eval/<sweep>_<run_name>_passk.json`.
- [ ] Save sampled generations to `/home/atuin/c107fa/c107fa12/synthetic-RLVL/passk_eval/<sweep>_<run_name>_samples.jsonl`.
- [ ] Upload metrics to W&B project `synthetic-rlvl` under the original sweep group.

## Operational Notes

- Catch-up array uses startup jitter and `RAY_raylet_start_wait_time_s=120` to reduce Ray startup timeouts.
- Merge+eval uses merged actors, not LoRA-on-vLLM, because native vLLM `n=64` works reliably and gives higher throughput after merge.
- The pass@k eval array intentionally uses low concurrency (`%2`) because each task loads/merges a 24B model and then generates 25,600 completions.

## Status Update (2026-04-24 later)

- Catch-up tasks `3561187_3` and `3561187_4` failed with Ray node startup timeouts before training began.
- Resubmitted those two array indices as retry array `3561402` with `%1`, `RAY_raylet_start_wait_time_s=300`, and `RAY_STARTUP_JITTER_MAX_SECONDS=300`.
- Updated pass@k array `3561188` dependency to `afterany:3557214:3561187:3561402`, so merge/eval waits for the retry too.

## GPU Idle / Ray Packaging Update (2026-04-24)

- Found active idle-GPU job `3561402_4`: Slurm had allocated `gres/gpu:a100=1`, but no CUDA process was attached for more than 5 hours.
- The job hung immediately after Ray packaged/uploaded the repo as a runtime env (`467 MiB`) and before `SynthTaskRunner` or vLLM workers started.
- Patched `posttrain_grpo_verl.py` to exclude bulky local directories from Ray `runtime_env.working_dir` packaging: `logs`, `wandb`, `wandb_artifacts`, `passk_eval`, `runs`, `tmp`, `datasets`, caches, and pycache dirs.
- Cancelled idle task `3561402_4`.
- Resubmitted failed/hung catch-up indices `3,4,6,7,8,9,10,11` as retry array `3562056` with `%1`, `RAY_raylet_start_wait_time_s=300`, and startup jitter.
- Updated pass@k array `3561188` dependency to `afterany:3557214:3561187:3561402:3562056`.

## Train Sample Cap Clarification (2026-04-25)

- The materialized local train subsets are not exhausted: `train_up_to_10_1m/train.parquet` has 1,000,000 rows.
- The premature clean completions at approximately 1250/625/312 steps came from the GRPO config limit `data.train_samples=20000` combined with `total_epochs=1`, not from the underlying materialized dataset size.
- This exactly matches `20000 / num_prompts`: `16 -> 1250`, `32 -> 625`, `64 -> 312.5`.
- Catch-up jobs now override `data.train_samples=120000`, enough for 1500 steps at `num_prompts=64` (`1500 * 64 = 96000`) with margin.

## Recovery Resubmission Update (2026-04-25)

- The earlier retry array `3562056` was submitted before the `data.train_samples=120000` and `include_dashboard=false` fixes. Its running task `_9` is healthy and using GPU, so it was left running.
- Old pending tasks `3562056_10` and `3562056_11` were cancelled to avoid launching with stale settings.
- Submitted corrected idempotent recovery array `3562986` over catch-up rows `0-11%1`; the script skips rows that already have a final actor and resumes partial rows automatically.
- Updated pass@k job `3561188` to wait for `3562986` as well.

## Runtime Estimate + Chained Recovery Waves (2026-04-25)

- Runtime estimates from logs show `num_prompts=64` and `num_rollouts=64` are about 63-66 hours for 1500 steps, so they cannot finish in one 24h allocation.
- Observed median step times: prompts16 about 56s/it, prompts32 about 88-90s/it, prompts64 about 152-155s/it, rollouts64 about 158s/it.
- Cancelled single pending recovery array `3562986`; it would have been too brittle and too slow as one wave.
- Submitted chained idempotent recovery waves `3563089 -> 3563090 -> 3563091 -> 3563092`, each `0-11%3`.
- The recovery script skips rows with final actors and resumes partial rows automatically, so later waves only do useful continuation work.
- Updated pass@k array `3561188` to wait for all four recovery waves before merge/eval starts.

## Oversight Rescheduled (2026-04-25)

- Earlier oversight jobs `3561189`-`3561193` failed because the batch environment did not have the `cs` wrapper (`exit 127`). Pending stale oversight job `3561194` was cancelled.
- Patched `scripts/slurm/codex/experiment_oversight.slurm` to call `codex exec` directly and add the node/npm path explicitly.
- Scheduled replacement oversight jobs every 4 hours: `3563124`, `3563125`, `3563126`, `3563127`, `3563128`, `3563129`, `3563130`, `3563131`.

## Oversight Wrapper Correction (2026-04-25)

- The previous oversight replacement used `codex exec` directly, but the intended permission/context wrapper is the user's `cs` alias.
- `cs` is defined in `~/.bash_profile` as `codex --search --dangerously-bypass-approvals-and-sandbox`.
- Patched `scripts/slurm/codex/experiment_oversight.slurm` to enable alias expansion, source `~/.bash_profile`/`~/.bashrc`, verify `cs`, and call `cs` directly.
- Cancelled replacement oversight jobs `3563124`-`3563131`.
- Resubmitted oversight jobs every 4h with the `cs` wrapper: `3563169`, `3563170`, `3563171`, `3563172`, `3563173`, `3563174`, `3563175`, `3563176`.

## Oversight Pass (2026-04-26)

- Scheduler check:
  - Recovery wave `3563089` has rows `_0,_1,_2` running and rows `_3-11` pending by array limit.
  - Recovery waves `3563090`, `3563091`, and `3563092` are dependency-held.
  - Merge/eval array `3561188` is dependency-held.
  - Oversight job `3564067` is running; `3564068`-`3564072` are pending by begin time.
- Progress check:
  - `3563089_0` was around `1086/1500`.
  - `3563089_1` was around `555/1500`.
  - `3563089_2` was around `31/1500`; this is expected for the prompts64 row and should continue via chained waves rather than cancellation.
- Failure check:
  - Older failed catch-up tasks include Ray node startup timeouts.
  - Older oversight jobs `3563169`-`3563173` failed with `stdin is not a terminal`.
  - No current recovery log showed an idle-GPU hang; active rows are emitting training progress.
- Action taken:
  - `rl_grpo_merge_ablate_prompts8_rollouts64_seed3408` had no final actor and was no longer covered by a live or pending retry. Its original wait-on-existing task `3557214_22` timed out after progressing to about step `553/1500`.
  - Added a catch-up manifest row for `rollouts	rollouts	64	3408	continue	rl_grpo_merge_ablate_prompts8_rollouts64_seed3408`.
  - Submitted chained single-row continuation waves `3564074 -> 3564075 -> 3564076`; `3564074_12` is running and the later two are dependency-held.
  - Updated pass@k dependency for `3561188` to wait for `3563089`, `3563090`, `3563091`, `3563092`, `3564074`, `3564075`, and `3564076`.

## Reward-Only Pass@k + W&B Cleanup (2026-04-26 13:23 CEST)

- Reward-design final checkpoint inventory is complete: 24/24 reward runs have final actors at `global_step_1500` or `global_step_1501`.
- Submitted reward-only merge/eval array `3564081` over manifest rows `24-47` with concurrency `%2`.
  - Active at update time: `3564081_24`, `3564081_25`.
  - Pending by array limit: `3564081_[26-47]`.
- The original full merge/eval array `3561188` remains dependency-held on prompt/rollout recovery. It is still needed for prompt and rollout sweeps; reward rows should skip if their outputs already exist.
- W&B cleanup completed for training runs in `synthetic-rlvl`:
  - Deleted 15 stale duplicate/retry training runs.
  - Verified each manifest training group has exactly 3 training runs and 3 unique training run names.
  - Evaluation runs may appear as extra W&B runs in the same groups if they use `_passk_step...` run names; those should not be treated as duplicate training runs.
- Oversight update: fixed `scripts/slurm/codex/experiment_oversight.slurm` to invoke `cs exec` instead of interactive `cs`; interactive `cs` failed in batch with `stdin is not a terminal`.

## Oversight Pass (2026-04-26 17:10 CEST)

- Scheduler check:
  - Recovery wave `3563089` still has rows `_0,_1,_2` running and rows `_3-11` pending by array limit.
  - Recovery waves `3563090`, `3563091`, and `3563092` remain dependency-held.
  - Single-row rollout64 continuation `3564075_12` is running; `3564076_12` is dependency-held.
  - New extra row-12 continuation `3564344_12` is dependency-held after `3564076`.
  - Full merge/eval array `3561188` remains dependency-held on `3563089`, `3563090`, `3563091`, `3563092`, `3564075`, `3564076`, and `3564344`.
  - Reward-only merge/eval array `3564081` is active with rows `30` and `31` running, rows `24-29` completed, and rows `32-47` pending by array limit.
- Progress check:
  - `3563089_0` was around `1340/1500`.
  - `3563089_1` was around `715/1500`.
  - `3563089_2` was around `122/1500`; this remains consistent with the expected prompts64 runtime and should continue via chained waves.
- Failure / idle-GPU check:
  - `3564074_12` for `rl_grpo_merge_ablate_prompts8_rollouts64_seed3408` hung after Ray runtime-env package upload. It had nearly four hours elapsed, only about nine CPU-minutes in the batch step, and no `SynthTaskRunner` or training progress in the log.
  - Cancelled `3564074_12` rather than letting it burn the allocation idle.
- Action taken:
  - Existing dependent retry `3564075_12` started immediately after the cancellation and reached `SynthTaskRunner` startup after Ray packaging.
  - Submitted replacement continuation wave `3564344` with `--array=12-12%1`, dependent on `3564076`, and with `RAY_raylet_start_wait_time_s=300` plus `RAY_STARTUP_JITTER_MAX_SECONDS=240`.
  - Updated pass@k dependency for `3561188` to wait for the active recovery chain plus `3564344`.
  - No other current recovery or reward pass@k job required cancellation or resubmission.

## Oversight Pass (2026-04-26 21:10 CEST)

- Scheduler check:
  - Recovery wave `3563089` has `_0` completed, `_1`, `_2`, and `_3` running, and `_4-11` pending by array limit.
  - Recovery waves `3563090`, `3563091`, and `3563092` remain dependency-held.
  - Single-row rollout64 continuation `3564075_12` is running; `3564076_12` and `3564344_12` remain dependency-held.
  - Full merge/eval array `3561188` remains dependency-held on the recovery chain.
  - Reward-only pass@k array `3564081` has rows `24-35` completed, rows `36` and `37` running, and rows `38-47` pending by array limit.
  - Oversight jobs `3564067` and `3564068` completed successfully; `3564069` is running; `3564070`-`3564072` remain pending by begin time.
- Progress check:
  - `3563089_0` completed and wrote the final actor for `rl_grpo_merge_ablate_prompts16_rollouts8_seed3407` at `global_step_1500`. The end of the log has DataLoader/W&B teardown errors after final validation, but Slurm state is `COMPLETED` and the actor directory exists.
  - `3563089_1` was around `879/1500`.
  - `3563089_2` was around `214/1500`, matching the expected slow prompts64 runtime.
  - `3563089_3` was around `1071/1500`.
  - `3564075_12` was around `590/1500`, so the row-12 continuation is actively progressing after the previous idle task was cancelled.
  - Reward pass@k rows `36` and `37` reached sampled vLLM chunk `16/25`.
- Failure / idle-GPU check:
  - No active recovery task showed the idle-GPU/Ray-packaging hang pattern; active jobs have recent progress logs and nontrivial live `sstat` CPU/RSS.
  - No new uncovered failed recovery row was found.
  - Cancelled pass@k jobs `3564079` and `3564080` are older superseded attempts; active array `3564081` covers the reward rows.
- Action taken:
  - No cancellation or resubmission was needed in this pass.

## Oversight Pass (2026-04-27 01:10 CEST)

- Scheduler check:
  - Recovery wave `3563089` has `_1`, `_2`, and `_3` running, `_4-11` pending by array limit, and `_0` completed.
  - Recovery waves `3563090`, `3563091`, and `3563092` remain dependency-held.
  - Single-row rollout64 continuation `3564075_12` is running; `3564076_12` and `3564344_12` remain dependency-held.
  - Full merge/eval array `3561188` remains dependency-held on `3563089`, `3563090`, `3563091`, `3563092`, `3564075`, `3564076`, and `3564344`.
  - Reward-only pass@k array `3564081` has rows `24-43` completed, rows `44` and `45` running, and rows `46-47` pending by array limit.
  - Oversight jobs `3564067`, `3564068`, and `3564069` completed successfully; `3564070` is running; `3564071` and `3564072` remain pending by begin time.
- Progress check:
  - `3563089_1` was around `777/1500`.
  - `3563089_2` was around `277/1500`, matching expected prompts64 slow progress.
  - `3563089_3` was around `1279/1500`.
  - `3564075_12` was around `683/1500`, so the row-12 continuation is still progressing.
  - Reward pass@k rows `44` and `45` reached sampled vLLM chunks `6/25` and `7/25`; rows `46` and `47` are the only remaining pending reward eval rows.
- Failure / idle-GPU check:
  - `sacct` showed no new failed, cancelled, timed out, or OOM jobs in the monitored set since the previous pass.
  - Active recovery logs have recent `Training Progress` lines and nontrivial live `sstat` CPU/RSS usage; no current job showed the idle-GPU/Ray-packaging hang pattern.
  - Catch-up rows without final actors are still covered by active or dependency-held recovery waves, so no missing uncovered row was found.
- Action taken:
  - No cancellation or resubmission was needed in this pass.

## Oversight Pass (2026-04-27 05:10 CEST)

- Scheduler check:
  - Recovery wave `3563089` has `_1`, `_2`, and `_4` running, `_5-11` pending by array limit, and `_0`/`_3` completed.
  - Recovery waves `3563090`, `3563091`, and `3563092` remain dependency-held.
  - Single-row rollout64 continuation `3564075_12` is running; `3564076_12` and `3564344_12` remain dependency-held.
  - Full merge/eval array `3561188` remains dependency-held on `3563089`, `3563090`, `3563091`, `3563092`, `3564075`, `3564076`, and `3564344`.
  - Reward-only pass@k array `3564081` completed rows `24-47` successfully.
  - Oversight jobs: `3564070` completed, `3564071` is running, and `3564072` remains pending by begin time.
- Progress check:
  - `3563089_1` was around `1201/1500`.
  - `3563089_2` was around `397/1500`, still matching expected prompts64 slow progress.
  - `3563089_4` was around `544/1500`.
  - `3564075_12` was around `776/1500`.
  - Catch-up manifest rows `0` and `3` now have final actors at `global_step_1500`; rows still missing final actors are covered by active or dependency-held continuation waves.
- Failure / idle-GPU check:
  - No new failed, cancelled, timed out, or OOM jobs were found in the monitored set.
  - Active recovery logs have recent `Training Progress` lines and live `sstat` CPU/RSS usage, so there is no current idle-GPU/Ray-packaging hang.
  - Reward pass@k produced the expected 24 reward metrics files and 24 reward samples files.
- Action taken:
  - No cancellation or resubmission was needed in this pass.

## Oversight Pass (2026-04-27 09:10 CEST)

- Scheduler check:
  - Recovery wave `3563089` has `_1`, `_2`, and `_4` running, `_5-11` pending by array limit, and `_0`/`_3` completed with final actors.
  - Recovery waves `3563090`, `3563091`, and `3563092` remain dependency-held.
  - Single-row rollout64 continuation `3564075_12` is running; `3564076_12` and `3564344_12` remain dependency-held.
  - Full merge/eval array `3561188` remains dependency-held on `3563089`, `3563090`, `3563091`, `3563092`, `3564075`, `3564076`, and `3564344`.
  - Reward-only pass@k array `3564081` remains complete for rows `24-47`.
  - Oversight jobs `3564067`-`3564071` completed successfully; `3564072` is running.
- Progress check:
  - `3563089_1` was around `1365/1500`.
  - `3563089_2` was around `488/1500`, still matching expected prompts64 slow progress.
  - `3563089_4` was around `709/1500`.
  - `3564075_12` was around `872/1500`.
  - Catch-up manifest rows `0` and `3` have final actors at `global_step_1500`; remaining missing final actors are covered by active or dependency-held continuation waves.
- Failure / idle-GPU check:
  - No new failed, cancelled, timed out, or OOM jobs were found in the monitored set since `2026-04-27 05:00`.
  - Active recovery logs have recent `Training Progress` lines and live resource usage, so there is no current idle-GPU/Ray-packaging hang.
  - Reward pass@k still has the expected 24 reward metrics files and 24 reward samples files.
- Action taken:
  - No cancellation or resubmission was needed in this pass.
