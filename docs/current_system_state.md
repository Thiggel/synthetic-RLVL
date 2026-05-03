# Synthetic-RLVL: Current System State (2026-04-22)

## Summary

- Posttraining is now run from pre-merged per-seed checkpoints (not raw LoRA adapter dirs).
- Validation supports sampled pass@k metrics in addition to greedy @1 metrics.
  - Full sampled eval is disabled during training by default due runtime cost.
  - Post-hoc defaults: `sampled_num_generations=64`, `sampled_k_values=[1,2,4,8,16,32,64]`.
  - Main sampled metrics: `correct_pass@K`, `valid_pass@K`, `format_pass@K`, `joint_pass@K`, `valid_given_correct@K`.
  - OOD bands are logged as `synthetic_sampled/band_train`, `band_ood`, and `band_hard_tail`.
- Pre-merged checkpoints available:
  - `/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/merged_sft_lr1e-4_seed3407`
  - `/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/merged_sft_lr1e-4_seed3408`
  - `/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/merged_sft_lr1e-4_seed3409`
- Active 3-seed posttraining sweeps were submitted:
  - `3548386` (`grpo_mrg_ablate_prompts`, array `0-11`)
  - `3548388` (`grpo_mrg_ablate_rollouts`, array `0-11`)
  - `3548387` (`posttrain_reward_ablation`, array `0-20`)
- Snapshot at doc update time: `26 RUNNING`, `19 FAILED` across these three arrays (`sacct`).

## Active Jobs Snapshot (2026-04-22)

- Posttrain retries:
  - `3556764`: 4 running (`_2,_3,_4,_5`), 2 failed (`_0,_1`), remaining tasks pending by array limit.
  - `3555102_6`: still running.
- New reward experiment:
  - `3557014` (`posttrain_rw_line_valid`, array `0-2%3`) submitted and pending.
- Concurrent non-posttrain jobs under the same user account:
  - Running: `af_gsm8k` (`3556770`), `af_csqa` (`3556776`), `af_mmlu` (`3556782`), `af_folio` (`3556788`), `af_proofwriter` (`3556794`), `af_swebench` (`3556800`), `af_bbh` (`3556816`), plus one `interactive` allocation (`3556811`).
  - Additional `af_*` jobs are pending on dependencies.

## Ray Timeout Note

- Startup timeouts in the retry array appear transient/cluster-load related (some tasks fail, others succeed with identical script and config, including on the same node).
- Recommended operational mitigations:
  - lower array concurrency for Ray-heavy jobs,
  - add randomized pre-launch sleep,
  - set `RAY_raylet_start_wait_time_s` higher (for example `120`),
  - optionally use `srun --ntasks=1` for stricter Slurm step isolation.

## Reward Schemas (Posttrain)

Configured reward ablation set:
- `correct_plus_0p1_format`
- `indicator_correct_and_format`
- `correct_plus_valid_plus_0p1_format`
- `correct_plus_line_valid_plus_0p1_format` (new, submitted as `3557014`)
- `correct_plus_0p75_valid_plus_0p1_format`
- `correct_plus_0p5_valid_plus_0p1_format`
- `correct_plus_0p25_valid_plus_0p1_format`
- `indicator_all`

## GRPO Sweep Policy (Current)

- `grpo.max_num_batched_tokens=16384`
- `optim.micro_batch_size=4`
- `optim.logprob_micro_batch_size=4`
- `task.template=logic`, `task.prefill=none`
- Prompts ablation: `num_prompts in {8,16,32,64}`, `num_rollouts=8`
- Rollouts ablation: `num_prompts=8`, `num_rollouts in {8,16,32,64}`
- All above are now run across seeds `{3407,3408,3409}`

## Recovery + Pass@k Plan (2026-04-24)

- Current final-checkpoint recovery and post-hoc sampled pass@k checklist: `docs/posttrain_recovery_passk_checklist_2026-04-24.md`.
- Catch-up posttrain array submitted: `3561187` (`scripts/slurm/jobs/posttrain_catchup_final_2026-04-24.slurm`).
- Dependent merge + pass@k eval array submitted: `3561188` (`scripts/slurm/jobs/posthoc_merge_eval_passk_2026-04-24.slurm`), current dependency `afterany:3557214:3561187:3561402:3562056` after Ray-timeout/idle-GPU retries.
- Ray-timeout retry for catch-up indices `3,4`: `3561402` (`%1`, longer Ray wait/jitter); task `_4` later hung idle and was cancelled.
- Ray packaging/idle-GPU fix: `posttrain_grpo_verl.py` now excludes bulky local artifact directories from Ray runtime-env packaging; retry array `3562056` covers catch-up indices `3,4,6,7,8,9,10,11`.
- Oversight jobs submitted at 4-hour intervals: `3561189`, `3561190`, `3561191`, `3561192`, `3561193`, `3561194`.
- Main final analysis should compare only `global_step_1500`/`global_step_1501` actors. Partial `500`/`1000` checkpoints are debugging-only unless explicitly plotting learning curves.

## Runtime Conventions

- Slurm scripts source `scripts/env.sh`.
- Preferred venv: `$HPCVAULT/.venv_rlvl_posttrain`; fallback: `$WORK/.venv`.
- W&B project for posttraining sweeps: `synthetic-rlvl`.

## References

- Detailed live job mapping and array semantics: `docs/posttrain_status_2026-04-18.md`
- Constrained proof-line pass@k protocol: `docs/constrained_proof_line_eval.md`

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

## Recovery Oversight Update (2026-04-26)

- Active recovery wave `3563089` is healthy and using training progress logs:
  - `3563089_0` (`rl_grpo_merge_ablate_prompts16_rollouts8_seed3407`) was around `1086/1500`.
  - `3563089_1` (`rl_grpo_merge_ablate_prompts32_rollouts8_seed3407`) was around `555/1500`.
  - `3563089_2` (`rl_grpo_merge_ablate_prompts64_rollouts8_seed3407`) was around `31/1500`, consistent with the slower prompts64 runtime.
- Later catch-up waves `3563090`, `3563091`, and `3563092` remain dependency-held; `3561188` pass@k remains dependency-held.
- Found one missing final actor not covered by the 12-row catch-up waves: `rl_grpo_merge_ablate_prompts8_rollouts64_seed3408`. Its original wait-on-existing task `3557214_22` timed out at the 24h limit after progressing to about step `553/1500`, with GPU utilization recorded in Slurm job statistics.
- Added this row to `docs/manifests/posttrain_catchup_2026-04-24.tsv` and submitted chained continuation waves `3564074 -> 3564075 -> 3564076` using the idempotent catch-up script. `3564074_12` is running; `3564075_12` and `3564076_12` are dependency-held.
- Updated pass@k job `3561188` to wait for `3563089`, `3563090`, `3563091`, `3563092`, `3564074`, `3564075`, and `3564076`.
- No active recovery job showed the previous idle-GPU/Ray-packaging hang pattern in the checked logs. Older quick failures remain attributable to Ray startup timeouts; current long jobs are progressing slowly and were not cancelled.
- Oversight status: earlier jobs `3563169`-`3563173` failed with `stdin is not a terminal`; replacement oversight `3564067` is running, with `3564068`-`3564072` pending by begin time.

## Reward Eval + W&B Cleanup Update (2026-04-26 13:23 CEST)

- All 24 reward-design final actors are present at `global_step_1500` or `global_step_1501`.
- Submitted reward-design-only merge + sampled pass@k eval array `3564081` with `--array=24-47%2`, so reward eval does not wait for the still-running prompt/rollout recovery waves.
- `3564081_24` and `3564081_25` are currently running; `3564081_[26-47]` are pending by array limit.
- Full all-sweep merge/eval array `3561188` remains dependency-held for prompt/rollout completion; reward rows should be skipped later if outputs already exist.
- W&B cleanup completed for project `synthetic-rlvl`: deleted 15 stale duplicate training runs.
- Verified all manifest training groups now have exactly 3 training runs and 3 unique training run names. Pass@k eval runs may add additional runs to the same groups with `_passk_step...` names, which are evaluation runs rather than duplicate training runs.
- Oversight script correction: `scripts/slurm/codex/experiment_oversight.slurm` now uses `cs exec` for non-interactive batch execution. Earlier `cs` interactive invocation failed with `stdin is not a terminal`.
- Replacement oversight jobs: `3564067` running; `3564068`-`3564072` pending by begin time.

## Constrained Proof-Line Evaluation Option (2026-04-26)

- Added optional post-hoc constrained proof-line pass@k evaluation. It is disabled by default and does not change existing training or normal eval behavior.
- The constrained decoder samples the model's own formal prefix, then regenerates each `<proof>` line with a fixed `N = constrained_candidates_per_line` candidates.
- Candidate proof lines are ranked by the logic engine as random `<` syntactic `<` valid `<` valid-and-novel. This produces separate `synthetic_constrained_sampled/...` pass@k metrics for `syntactic`, `format`, `correct`, `valid`, `joint`, and `valid_given_correct`.
- Full protocol and scientific comparison plan: `docs/constrained_proof_line_eval.md`.
- CLI entry point: `scripts/evaluate_checkpoint_passk.py --constrained-enabled --constrained-num-generations <K> --constrained-candidates-per-line <N> --constrained-k-values 1,2,4,8`.
- End-to-end A100 smoke test passed on `/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/merged_actor_rl_reward_correct_plus_0p1_format_seed3407_mrg_final` with one prompt, one constrained sample, and two candidates per line. The smoke output wrote `tmp/constrained_smoke_passk.json` and `tmp/constrained_smoke_samples.jsonl`.

## Recovery Oversight Update (2026-04-26 17:10 CEST)

- Recovery wave `3563089` remains healthy:
  - `3563089_0` progressed to about `1340/1500`.
  - `3563089_1` progressed to about `715/1500`.
  - `3563089_2` progressed to about `122/1500`, consistent with the expected slow prompts64 row.
- Reward-only pass@k array `3564081` is progressing normally. Rows `24-29` completed successfully; rows `30` and `31` are running; rows `32-47` remain pending by array limit.
- Found row-12 continuation `3564074_12` (`rl_grpo_merge_ablate_prompts8_rollouts64_seed3408`) idle after Ray runtime-env upload: nearly four hours elapsed, only about nine CPU-minutes in the batch step, no `SynthTaskRunner` or training progress, and no CUDA process evidence in the job log after startup.
- Cancelled idle `3564074_12`. Its dependent continuation `3564075_12` started immediately and reached `SynthTaskRunner` startup after Ray packaging, with `3564076_12` still dependency-held.
- Submitted one extra replacement continuation wave for row 12 after `3564076`: `3564344` (`--array=12-12%1`, `RAY_raylet_start_wait_time_s=300`, `RAY_STARTUP_JITTER_MAX_SECONDS=240`).
- Updated full merge/eval job `3561188` dependency to wait for `3563089`, `3563090`, `3563091`, `3563092`, `3564075`, `3564076`, and `3564344`. It remains dependency-held.
- Oversight job `3564067` completed successfully. `3564068` is running; `3564069`-`3564072` are pending by begin time.

## Recovery Oversight Update (2026-04-26 21:10 CEST)

- Scheduler check:
  - Recovery wave `3563089`: `_0` completed, `_1`, `_2`, and `_3` are running, and `_4-11` are pending by array limit.
  - Recovery waves `3563090`, `3563091`, and `3563092` remain dependency-held.
  - Row-12 rollout64 continuation `3564075_12` is running; `3564076_12` and replacement `3564344_12` remain dependency-held.
  - Full merge/eval array `3561188` remains dependency-held on the recovery chain.
  - Reward-only pass@k array `3564081` is active with rows `36` and `37` running, rows `24-35` completed, and rows `38-47` pending by array limit.
  - Oversight jobs `3564067` and `3564068` completed successfully; `3564069` is running; `3564070`-`3564072` are pending by begin time.
- Progress check:
  - `3563089_0` reached `global_step_1500` and wrote the final actor under `/home/atuin/c107fa/c107fa12/synthetic-RLVL/runs/rl_grpo_merge_ablate_prompts16_rollouts8_seed3407/global_step_1500/actor`. Its later teardown logged DataLoader/W&B cleanup errors, but Slurm marked the task `COMPLETED`.
  - `3563089_1` progressed to about `879/1500`.
  - `3563089_2` progressed to about `214/1500`, consistent with prompts64 runtime.
  - `3563089_3` started and progressed to about `1071/1500`.
  - `3564075_12` progressed to about `590/1500`, so the row-12 retry is no longer idle.
  - Reward pass@k rows `36` and `37` reached sampled vLLM chunk `16/25`.
- Failure / idle-GPU check:
  - No current recovery task showed the previous idle-after-Ray-upload pattern; active recovery tasks have recent `Training Progress` logs and nontrivial `sstat` CPU/RSS usage.
  - No new uncovered failed recovery row was found. Older cancelled pass@k jobs `3564079` and `3564080` are superseded by the active `3564081` reward-only array.
- Action taken:
  - No cancellation or resubmission was needed in this pass.

## Recovery Oversight Update (2026-04-27 01:10 CEST)

- Scheduler check:
  - Recovery wave `3563089`: `_1`, `_2`, and `_3` are running; `_4-11` remain pending by array limit. `_0` previously completed with a final actor.
  - Recovery waves `3563090`, `3563091`, and `3563092` remain dependency-held.
  - Row-12 rollout64 continuation `3564075_12` is running; `3564076_12` and `3564344_12` remain dependency-held.
  - Full merge/eval array `3561188` remains dependency-held on the recovery chain.
  - Reward-only pass@k array `3564081` has rows `24-43` completed, rows `44` and `45` running, and rows `46-47` pending by array limit.
  - Oversight jobs `3564067`, `3564068`, and `3564069` completed successfully; `3564070` is running; `3564071` and `3564072` are pending by begin time.
- Progress check:
  - `3563089_1` is around `777/1500`.
  - `3563089_2` is around `277/1500`, consistent with the expected slow prompts64 runtime.
  - `3563089_3` is around `1279/1500`.
  - `3564075_12` is around `683/1500`, with recent training progress logs and nontrivial `sstat` CPU/RSS usage.
  - Reward pass@k rows `44` and `45` have merged actors and reached sampled vLLM chunks `6/25` and `7/25`, respectively.
- Failure / idle-GPU check:
  - No new `FAILED`, `CANCELLED`, `TIMEOUT`, or `OUT_OF_MEMORY` jobs were reported since the previous pass for the monitored recovery/eval/oversight set.
  - Active recovery logs show current training progress; no Ray startup timeout or idle-after-runtime-env-upload pattern was found.
  - Missing final actors in the catch-up manifest remain covered by active or dependency-held recovery waves.
- Action taken:
  - No cancellation or resubmission was needed in this pass.

## Recovery Oversight Update (2026-04-27 05:10 CEST)

- Scheduler check:
  - Recovery wave `3563089`: `_1`, `_2`, and `_4` are running; `_5-11` remain pending by array limit. `_0` and `_3` have completed with final actors.
  - Recovery waves `3563090`, `3563091`, and `3563092` remain dependency-held.
  - Single-row rollout64 continuation `3564075_12` is running; `3564076_12` and `3564344_12` remain dependency-held.
  - Full merge/eval array `3561188` remains dependency-held on the recovery chain.
  - Reward-only pass@k array `3564081` completed successfully for rows `24-47`.
  - Oversight jobs `3564070` completed, `3564071` is running, and `3564072` remains pending by begin time.
- Progress check:
  - `3563089_1` progressed to about `1201/1500`.
  - `3563089_2` progressed to about `397/1500`, consistent with the expected slow prompts64 runtime.
  - `3563089_4` started and progressed to about `544/1500`.
  - `3564075_12` progressed to about `776/1500`, with substantial live `sstat` CPU/RSS usage.
  - Catch-up final actor inventory now has rows `0` and `3` complete; the remaining missing final actors are still covered by active or dependency-held continuation waves.
- Failure / idle-GPU check:
  - `sacct` showed no new failed, cancelled, timed out, or OOM jobs in the monitored recovery/eval/oversight set.
  - Active recovery logs have recent `Training Progress` lines and nontrivial live `sstat` usage; no current job showed the idle-after-Ray-runtime-env-upload pattern.
  - Reward pass@k output inventory has 24 reward metrics JSON files and 24 reward samples JSONL files.
- Action taken:
  - No cancellation or resubmission was needed in this pass.

## Recovery Oversight Update (2026-04-27 09:10 CEST)

- Scheduler check:
  - Recovery wave `3563089` has `_1`, `_2`, and `_4` running, `_5-11` pending by array limit, and `_0`/`_3` completed with final actors.
  - Recovery waves `3563090`, `3563091`, and `3563092` remain dependency-held.
  - Single-row rollout64 continuation `3564075_12` is running; `3564076_12` and `3564344_12` remain dependency-held.
  - Full merge/eval array `3561188` remains dependency-held on the recovery chain.
  - Reward-only pass@k array `3564081` remains complete for rows `24-47`.
  - Oversight jobs `3564067`-`3564071` completed successfully; `3564072` is running.
- Progress check:
  - `3563089_1` progressed to about `1365/1500`.
  - `3563089_2` progressed to about `488/1500`, consistent with the expected slow prompts64 runtime.
  - `3563089_4` progressed to about `709/1500`.
  - `3564075_12` progressed to about `872/1500`, with live `sstat` CPU/RSS usage.
  - Catch-up manifest rows `0` and `3` have final actors at `global_step_1500`; remaining missing final actors are covered by active or dependency-held continuation waves.
- Failure / idle-GPU check:
  - `sacct` showed no new failed, cancelled, timed out, or OOM jobs in the monitored set since `2026-04-27 05:00`.
  - Active recovery logs have recent `Training Progress` lines; no current job showed the idle-after-Ray-runtime-env-upload pattern.
  - Reward pass@k output inventory remains complete with 24 reward metrics JSON files and 24 reward samples JSONL files.
- Action taken:
  - No cancellation or resubmission was needed in this pass.

## 2026-04-27 Hard-v1 Experiment Chain

- Generated and uploaded `flaitenberger/LogicalReasoning-hard-v1` with `train_up_to_5_1m`, `train_up_to_10_1m`, and `val_step_01_1k` through `val_step_20_1k`.
- Fresh HF reload checks passed for representative subsets; see `docs/hard_v1_experiment_status_2026-04-27.md`.
- Submitted clean hard-v1 SFT-before-RL chain:
  - SFT array `3566369`: `scripts/slurm/sweeps/sft/hard_v1_lr1e4.slurm`, seeds `3407/3408/3409`, W&B group `sft_hard_v1/lr1e-4`.
  - SFT merge/sanity array `3566370`, dependency `afterok:3566369`, outputs `$WORK/synthetic-RLVL/tmp/merged_sft_hard_v1_seed{seed}`.
  - GRPO reward-ablation array `3566371`, dependency `afterok:3566370`, schemas `correct_plus_0p1_format`, `correct_plus_valid_plus_0p1_format`, `correct_times_valid_plus_0p1_format`, `correct_plus_line_valid_plus_0p1_format`, `correct_times_line_valid_plus_0p1_format`.
  - Final actor merge + pass@k eval array `3566372`, dependency `afterany:3566371`, outputs under `$WORK/synthetic-RLVL/passk_eval/hard_v1/`.

## 2026-04-29 Hard-v1 Throttle Update

- Increased hard-v1 GRPO reward-ablation array `3566371` from `%2` to `%4` using `scontrol update JobId=3566371 ArrayTaskThrottle=4`.
- New tasks `3566371_4` and `3566371_5` started alongside already-running `3566371_2` and `3566371_3`.

## 2026-04-29 Hard-v1 Ray Startup Retry

- Hard-v1 GRPO tasks `4-10` in original array `3566371` failed quickly with Ray startup timeout after the temporary `%4` throttle increase.
- Reduced `3566371` back to `%2` and submitted retry array `3570395` for tasks `4-10` with `%1`, `STARTUP_JITTER_SECONDS=900`, and `RESUME_MODE=auto`.
- Updated pass@k eval array `3566372` to depend on both `3566371` and `3570395`.

## 2026-04-29 Hard-v1 Staggered Retry Replacement

- Added deterministic startup staggering support to `scripts/slurm/sweeps/posttrain_hard_v1_reward_ablation.slurm` via `STARTUP_STAGGER_SECONDS` and `STARTUP_STAGGER_BASE_ID`.
- Canceled `%1` retry `3570395` and submitted staggered retry `3570401_[4-10%7]` for failed hard-v1 tasks `4-10` with 10-minute spacing and small jitter.
- Updated pass@k eval `3566372` to depend on `3566371` and `3570401`.

## Hard-v3 Compact Adversarial Chain (2026-04-30)

- Added and uploaded `flaitenberger/LogicalReasoning-hard-v3`.
- Cancelled pending synthetic-RLVL hard-v1/catchup/posthoc jobs before submitting the new chain.
- Submitted hard-v3 chain: SFT `3571150`, SFT merge/sanity `3571151`, GRPO reward ablation `3571152`, final merge+pass@k eval `3571153`.
- Details and GRPO array mapping: `docs/hard_v3_experiment_status_2026-04-30.md`.

## Hard-v3 Shuffled Natural Theory Update (2026-04-30)

- Updated `hard_v3` generation to deterministically shuffle and renumber only the natural-language theory shown in prompts.
- Formal target premises remain complete and canonical; proof citations are unchanged.
- Regenerated and uploaded `flaitenberger/LogicalReasoning-hard-v3` after the shuffle change.
- Cancelled superseded pre-shuffle chain `3571150`/`3571151`/`3571152`/`3571153`.
- Submitted active shuffled-NL chain: SFT `3571290`, merge `3571291`, GRPO `3571292`, final merge+pass@k eval `3571293`.
- Details: `docs/hard_v3_experiment_status_2026-04-30.md`.

## Hard-v3 Runtime Update (2026-05-01)

- SFT `3571290` and merge/sanity `3571291` completed successfully.
- First GRPO wave `3571292_0` through `3571292_6` is healthy but will likely hit the 24h walltime before step 1500; step-1000 checkpoints exist.
- Queued continuation arrays `3572663` for tasks `0-6` and `3572664` for tasks `7-14` with `RESUME_MODE=auto`.
- Cancelled original eval `3571293`; replacement eval `3572665` waits for both continuation arrays.

## Hard-v3 Immediate Second-Wave Start (2026-05-01)

- Cancelled original pending `3571292_7-14`, delayed retry `3572664`, and eval `3572665`.
- Submitted immediate second-wave GRPO `3572693` for tasks `7-14` with 600s per-index startup stagger and jitter.
- Submitted replacement eval `3572694`, waiting for `3572663` and `3572693`.

## Hard-v3 Oversight Update (2026-05-01 21:14 CEST)

- Original hard-v3 GRPO rows `3571292_0`-`3571292_6` are still running at the 24h limit with recent progress/checkpoints; continuation array `3572663_[0-6%7]` remains dependency-held and covers these rows after timeout/completion.
- Immediate second-wave hard-v3 GRPO rows `3572693_7`-`3572693_14` are running and emitting recent `Training Progress` lines. Latest observed progress ranges from about step `31/1500` to `103/1500`, consistent with the configured startup stagger and roughly 57-60s/step runtime.
- Final hard-v3 merge/pass@k eval `3572694_[0-14%8]` is correctly dependency-held on `3572663` and `3572693`.
- Oversight job `3572763` is running; `3572764`-`3572767` are pending by begin time.
- No new monitored `FAILED`, `CANCELLED`, `TIMEOUT`, or `OUT_OF_MEMORY` job was found since midnight in the hard-v3/recovery/eval/oversight set. No Ray startup failure or idle-GPU hang was found in current hard-v3 logs, so no cancellation or resubmission was needed.

## Hard-v3 Oversight Update (2026-05-01 23:54 CEST)

- First-wave task jobs `3571439`-`3571445` timed out at the 24h limit after useful progress; continuation array `3572663_0`-`3572663_6` is now running and resuming rows `0-6`.
- Second-wave GRPO rows `3572693_7`-`3572693_14` remain healthy and are making steady progress, but their ETA is close to the 24h walltime.
- Submitted dependent continuation array `3573037_[7-14%8]` for only rows `7-14` with `RESUME_MODE=auto`, 600s per-index startup stagger, and 60s jitter.
- Updated final hard-v3 merge/pass@k eval `3572694_[0-14%8]` dependency to `afterany:3572663:3572693:3573037`, so eval cannot start before the second-wave continuation opportunity has completed.
- Latest observed progress: rows `0-6` around `1149`, `1140`, `1129`, `1118`, `1109`, `1097`, `1088`; rows `7-14` around `271`, `262`, `251`, `239`, `225`, `218`, `206`, `196`.
- Current hard-v3 logs show active `Training Progress` and live resource usage. No fatal Ray startup failure, traceback, OOM, or idle-GPU hang was found. No running jobs were cancelled.

## Hard-v3 Oversight Update (2026-05-02 03:55 CEST)

- Hard-v3 first-wave continuation `3572663_0`-`3572663_6` is running and has progressed rows `0-6` to roughly step `1268`-`1277` of `1500`.
- Hard-v3 second-wave `3572693_7`-`3572693_14` is running and has progressed rows `7-14` to roughly step `268`-`277` of `1500`.
- Continuation `3573037_[7-14%8]` remains dependency-held on `3572693`, and final merge/pass@k eval `3572694_[0-14%8]` remains dependency-held on `3572663`, `3572693`, and `3573037`.
- Oversight jobs `3572763` and `3572764` completed successfully; `3572765` is running; `3572766` and `3572767` remain pending by begin time.
- Recent hard-v3 logs and `sstat` show active training/resource usage. No fatal Ray startup failure, traceback, OOM, unexpected timeout/cancelled task, or idle-GPU/Ray-packaging hang was found.
- No recovery resubmission was needed; all non-final hard-v3 rows are covered by live jobs or dependency-held continuation/eval jobs.

## Hard-v3 Oversight Update (2026-05-02 11:03 CEST)

- Hard-v3 first-wave continuation `3572663` completed successfully. Rows `0-6` now have final actors at `global_step_1500`.
- Hard-v3 second-wave `3572693_7`-`3572693_14` is still running at about 15h40m elapsed. Latest stderr progress is about rows `7-14`: `969`, `959`, `952`, `930`, `907`, `902`, `895`, and `882` of `1500`.
- Continuation `3573037_[7-14%8]` remains dependency-held on `3572693`. Final hard-v3 merge/pass@k eval `3572694_[0-14%8]` remains dependency-held on `3572693` and `3573037`; `3572663` is fulfilled.
- Oversight jobs `3572763`-`3572765` completed successfully, `3572766` is running, and `3572767` is pending by begin time.
- Logs and `sstat` show active training/resource usage for rows `7-14`. No fatal Ray startup failure, OOM, unexpected cancellation, or idle-GPU/Ray-packaging hang was found.
- No recovery resubmission was needed. The remaining non-final hard-v3 rows are progressing slowly and are covered by live jobs plus checkpointed continuation.

## Hard-v3 Oversight Update (2026-05-02 11:58 CEST)

- Hard-v3 second-wave `3572693_7`-`3572693_14` is still running at about 16h36m elapsed. Latest stderr progress is about rows `7-14`: `1022`, `1013`, `1007`, `989`, `965`, `960`, `952`, and `940` of `1500`.
- Rows `0-6` have final actors at `global_step_1500`; rows `7-14` do not yet have final actors.
- Continuation `3573037_[7-14%8]` remains dependency-held on `3572693`. Final hard-v3 merge/pass@k eval `3572694_[0-14%8]` remains dependency-held on `3572693` and `3573037`.
- Oversight jobs `3572763`-`3572766` completed successfully; `3572767` is running.
- Logs and `sstat` show active training/resource usage for rows `7-14`. The current scan found only expected tokenizer/Ray/NCCL warnings; no fatal Ray startup failure, OOM, unexpected cancellation, or idle-GPU/Ray-packaging hang was found.
- No recovery resubmission was needed. The remaining non-final hard-v3 rows are progressing slowly and are covered by live jobs plus checkpointed continuation.
