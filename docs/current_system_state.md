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

## 2026-05-03 Hard-v5 Shortcut-Controlled Experiment

Added hard-v5 dataset/reward/eval support. Hard-v5 removes proof citations from SFT/RL targets and evaluates proof correctness with citation-free derivability, so citation bookkeeping is no longer the main validity confound.

Design doc: `docs/hard_v5_dataset_design.md`.
Status doc: `docs/hard_v5_experiment_status_2026-05-03.md`.

Planned Slurm chain:

- dataset build + HF push: `scripts/slurm/jobs/build_materialized_hard_v5.slurm`
- SFT: `scripts/slurm/sweeps/sft/hard_v5_lr1e4.slurm`
- SFT merge/sanity: `scripts/slurm/jobs/merge_sft_hard_v5_2026-05-03.slurm`
- GRPO reward ablation: `scripts/slurm/sweeps/posttrain_hard_v5_reward_ablation.slurm`
- final merge + pass@k eval: `scripts/slurm/jobs/posthoc_hard_v5_merge_eval_passk_2026-05-03.slurm`

Reward schemas: `correct_plus_0p1_format`, `correct_plus_citation_free_valid_plus_0p1_format`, `correct_times_citation_free_valid_plus_0p1_format`, `correct_plus_citation_free_line_valid_plus_0p1_format`, and `correct_times_citation_free_line_valid_plus_0p1_format`.


### Hard-v5 Submission, 2026-05-03

Submitted full hard-v5 chain:

- `3577927`: build and push `flaitenberger/LogicalReasoning-hard-v5`
- `3577928_[0-2%3]`: SFT seeds `3407`, `3408`, `3409`
- `3577929_[0-2%3]`: merge SFT checkpoints and sanity-check generations
- `3577930_[0-14%8]`: GRPO 3 seeds x 5 reward schemas
- `3577931_[0-14%4]`: final actor merge + sampled pass@k eval

Initial queue state: dataset build was running; dependent arrays were pending by dependency.


### Hard-v5 Runtime Status, 2026-05-04

Hard-v5 build/SFT/merge completed. GRPO `3577930` is running rows `0-7`; rows `8-14` are pending by throttle; final pass@k `3577931` remains dependency-held. The jobs are alive but slow: after about `18h`, running rows are only around `264-413/1500`, so the current `24h` limit is unlikely to produce final checkpoints for most rows. No `global_step_*` checkpoints were visible yet.

### Hard-v5 Replacement Chain, 2026-05-04

Cancelled slow hard-v5 GRPO/eval jobs `3577930` and `3577931`. The slowdown was caused by long clipped generations rather than missing vLLM batching: GRPO logs repeatedly hit the `2048` response cap with high generation time. The replacement keeps the same scientific ablation but uses a stronger SFT initialization and a lower rollout cap.

New chain:

- `3581290`: build/push `train_up_to_3_50k` into `flaitenberger/LogicalReasoning-hard-v5`; completed successfully in `00:00:47`, and HF load check confirmed `50,000` rows
- `3581291_[0-2%3]`: full SFT on depths `1..3`, `50k` rows, `5,000` optimizer steps
- `3581292_[0-2%3]`: merge the full-SFT LoRAs into local HF checkpoints
- `3581300_[0-14%15]`: GRPO reward ablation from merged full-SFT checkpoints with `max_response_length=1024`; replaces cancelled pending `3581293` and uses about 10 minutes of startup stagger per array index
- `3581301_[0-14%4]`: final actor merge + sampled pass@k eval; replaces cancelled pending `3581294`

New configs/scripts:

- `conf/sft_hard_v5_full.yaml`
- `conf/posttrain_grpo_hard_v5_fast.yaml`
- `scripts/slurm/jobs/build_materialized_hard_v5_sft50k.slurm`
- `scripts/slurm/sweeps/sft/hard_v5_full_lr1e4.slurm`
- `scripts/slurm/jobs/merge_sft_hard_v5_full_2026-05-04.slurm`
- `scripts/slurm/sweeps/posttrain_hard_v5_full_fast_reward_ablation.slurm`
- `scripts/slurm/jobs/posthoc_hard_v5_full_fast_merge_eval_passk_2026-05-04.slurm`

### Hard-v5 Full-Fast Retry, 2026-05-05

Full short-depth SFT and merge completed successfully, but GRPO array `3581300` failed for all rows while saving checkpoints. The failure was quota-related, not training-related: logs show normal progress until `Disk quota exceeded` at `global_step_100`, `global_step_200`, or `global_step_300`.

Changes made:

- Added VERL checkpoint-retention wiring in `posttrain_grpo_verl.py`.
- Set hard-v5-fast `validation.save_every=500`.
- Set hard-v5-fast `validation.max_actor_ckpt_to_keep=1`.
- Removed failed partial `rl_hard_v5_full_fast_*` run directories from `/home/atuin/c107fa/c107fa12/synthetic-RLVL/runs`.

Replacement jobs:

- `3584077_[0-14%15]`: GRPO retry. Initially submitted as `%5`, then raised to `%15` after deleting obsolete run directories and enabling checkpoint retention.
- `3584078_[0-14%4]`: dependent final actor merge + sampled pass@k eval.

Cleanup before raising concurrency:

- Deleted 90 obsolete run directories from `/home/atuin/c107fa/c107fa12/synthetic-RLVL/runs`.
- Remaining run artifacts are only the three full-SFT directories: `sft_hard_v5_full_lr1e-4_seed3407`, `sft_hard_v5_full_lr1e-4_seed3408`, and `sft_hard_v5_full_lr1e-4_seed3409`.
- Runs directory size dropped from about `5.4T` to `4.6G`.

### Hard-v5 Full-Fast Runtime Check, 2026-05-05

- Main GRPO retry `3584077` is progressing for 14/15 rows.
- Row `9` failed before training with a Ray startup timeout, not a model/checkpoint/quota error.
- Runs directory size is about `5.7G`, so the quota fix is holding.
- Progress snapshot: row `0` is around `354/1500`; other live rows are around `208-265/1500`.
- At current speed, a single 24h allocation will not finish; rows should checkpoint at step `500`, then continue.

Additional jobs:

- `3585899_[9%1]`: retry row 9 with `RESUME_MODE=auto`.
- `3585900_[0-14%15]`: first continuation wave after `3584077` and `3585899`.
- `3585901_[0-14%15]`: second continuation wave after `3585900`.
- `3584078_[0-14%4]`: final eval now waits for `afterany:3585901`.

### Hard-v5 Full-Fast Runtime Check, 2026-05-06

- Main wave `3584077` is still running close to the 24h walltime.
- 14 rows have successfully written `global_step_500` checkpoints.
- Row `9` retry `3585899_9` is running around `323/1500`; it has not checkpointed yet but is progressing.
- Progress snapshot: row `0` around `728/1500`; other main-wave rows around `544-626/1500`.
- Runs directory is about `395G`, consistent with one retained checkpoint for most rows.
- No new active-log failures were found beyond the already-known original row-9 Ray startup timeout.

### Hard-v5 Full-Fast Runtime Check, 2026-05-07

- First wave `3584077` and row-9 retry `3585899` timed out at the expected 24h limit.
- Continuation wave `3585900_[0-14%15]` is running for all rows.
- All rows resumed from `global_step_500`; logs show optimizer, RNG, and lr-scheduler restore from the step-500 actor checkpoints.
- Current progress range in `3585900` is about `803-995/1500`.
- No `global_step_1000` or final checkpoints exist yet.
- Runs directory is about `423G`.
- `3585901_[0-14%15]` is dependency-held as the next continuation wave.
- Final eval `3584078_[0-14%4]` remains dependency-held after `3585901`.

### Hard-v5 Full-Fast Runtime Check, 2026-05-08

- Continuation wave `3585900` finished overall; final wave `3585901` is now mostly complete.
- `10/15` runs have final `global_step_1500` or `global_step_1501` actors.
- The five remaining live rows are the only missing final actors:
  - task `6`: seed `3408`, `correct_plus_citation_free_valid_plus_0p1_format`, around `1431/1500`
  - task `7`: seed `3408`, `correct_times_citation_free_valid_plus_0p1_format`, around `1452/1500`
  - task `10`: seed `3409`, `correct_plus_0p1_format`, around `1450/1500`
  - task `11`: seed `3409`, `correct_plus_citation_free_valid_plus_0p1_format`, around `1450/1500`
  - task `12`: seed `3409`, `correct_times_citation_free_valid_plus_0p1_format`, around `1481/1500`
- Runs directory is about `1.1T`.
- Final eval `3584078_[0-14%4]` remains dependency-held and should start after the remaining `3585901` tasks finish.

### Hard-v5 Full-Fast Eval Check, 2026-05-08

- Training is complete: all `3585901` rows completed and all 15 runs have final actors at `global_step_1500` or `global_step_1501`.
- Post-hoc merge + pass@k eval `3584078_[0-14%4]` is running.
- `12/15` eval rows completed successfully and wrote pass@k JSON plus sample JSONL files.
- Remaining eval rows `12`, `13`, and `14` are running; logs show model loading/vLLM graph capture with no failure signatures.
- Current pass@k output coverage: `12` metrics JSON files and `12` sample JSONL files under `/home/atuin/c107fa/c107fa12/synthetic-RLVL/passk_eval/hard_v5_full_fast`.
- Raised active eval array throttle to `%15`; only rows `12`, `13`, and `14` were still running at that point, with no pending eval rows left.
- Updated the eval script default for future submissions to `--array=0-14%15` and added per-row startup jitter: `task_id * PASSK_STAGGER_SECONDS + random(0..PASSK_JITTER_SECONDS)`, defaulting to `120s` stagger and `30s` jitter.

## Hard-v5 Shortcut Rerun Plan (2026-05-08)

The active hard-v5 dataset generator was modified in place rather than adding a new `hard_v6` name.

Current design:
- Random per-example true path; answer is not determined by depth or global state order.
- Equal-length dormant shortcut path; shortcut final answer is explicitly different from the gold answer.
- Shortcut path is coherent but invalid because `dormant(a)` is never derivable.
- Gold proofs are citation-free; `analyze_proof_citation_free` validates them while strict citation validation intentionally does not.
- Training split uses `shortcut_rate=0.8`; validation uses `shortcut_rate=0.0`.

New fast experiment chain:
- Dataset build/push: `scripts/slurm/jobs/build_materialized_hard_v5_shortcut_2026-05-08.slurm`.
- SFT: `scripts/slurm/sweeps/sft/hard_v5_short_lr1e4_2026-05-08.slurm`, 3 seeds, steps `1..3`, `1000` optimizer steps.
- Merge: `scripts/slurm/jobs/merge_sft_hard_v5_short_2026-05-08.slurm`.
- GRPO: `scripts/slurm/sweeps/posttrain_hard_v5_short500_reward_ablation_2026-05-08.slurm`, 3 seeds x 5 reward schemas, `500` RL steps.
- Post-hoc pass@k: `scripts/slurm/jobs/posthoc_hard_v5_short500_merge_eval_passk_2026-05-08.slurm`.

Reward schemas:
- `correct_plus_0p1_format`
- `correct_plus_citation_free_valid_plus_0p1_format`
- `correct_times_citation_free_valid_plus_0p1_format`
- `correct_plus_citation_free_line_valid_plus_0p1_format`
- `correct_times_citation_free_line_valid_plus_0p1_format`

Faster diagnostics:
- GRPO validation runs every `100` steps and saves every `250` steps.
- Greedy validation logs `synthetic/step_N/correct`, `format`, `valid`, `citation_free_valid` for `N=1..20`.
- New shortcut diagnostic logs `synthetic/step_N/shortcut_answer` when dataset metadata includes `shortcut_branch_answer`.

Submitted chain (2026-05-08):
- Dataset build/push: `3598338`.
- Short SFT array: `3598339` (`0-2%3`), dependency `afterok:3598338`.
- SFT merge/sanity array: `3598340` (`0-2%3`), dependency `afterok:3598339`.
- GRPO reward ablation array: `3598341` (`0-14%15`), dependency `afterok:3598340`.
- Final merge + pass@k eval array: `3598342` (`0-14%15`), dependency `afterany:3598341`.

HF upload retry update (2026-05-08):
- Initial build `3598338` failed after local generation during HF upload with a transient Hugging Face `500 Internal Server Error`.
- Cancelled dependency-dead chain `3598339`-`3598342`.
- Patched `MaterializedDatasetBuilder.push_to_hub` to retry each subset upload up to 5 times with backoff.
- Resubmitted fresh chain:
  - Dataset build/push: `3598347`.
  - Short SFT array: `3598348`.
  - SFT merge/sanity array: `3598349`.
  - GRPO reward ablation array: `3598350`.
  - Final merge + pass@k eval array: `3598351`.

Oversight scheduled (2026-05-08):
- Added hard-v5-specific oversight wrapper: `scripts/slurm/codex/hard_v5_shortcut_oversight_2026-05-08.slurm`.
- Scheduled oversight jobs:
  - `3598360`, begin `2026-05-08T19:14:33 CEST`.
  - `3598361`, begin `2026-05-08T20:44:33 CEST`.

### Hard-v5 Shortcut Rerun Monitor (2026-05-08 19:16 CEST)

- Replacement dataset build/push `3598347` completed successfully in `00:05:24`.
- The build log confirms transient Hugging Face `500 Server Error` failures on several subset uploads, but the patched `MaterializedDatasetBuilder.push_to_hub` retry loop retried them and the job exited `0:0`.
- Current Slurm state for the monitored chain:
  - `3598348_0`, `3598348_1`, and `3598348_2` are running short SFT, about `31m` elapsed.
  - `3598349_[0-2%3]` is dependency-held on SFT.
  - `3598350_[0-14%15]` is dependency-held on merge/sanity.
  - `3598351_[0-14%15]` is dependency-held on GRPO.
- Recent SFT logs show normal loss/eval progress around steps `500+` of `1000`, with shortcut diagnostics logging. `sstat` shows nontrivial CPU/RSS usage for all three running tasks.
- Log scan found no unresolved HF upload failure, `DependencyNeverSatisfied`, SFT failure, merge failure, Ray startup failure, idle-GPU symptom, actual OOM, quota error, or timeout in the monitored chain. The only OOM-related matches were standard Accelerate memory-allocation warnings.
- Action: no cancellation or resubmission needed in this pass.

### Hard-v5 Shortcut Rerun Monitor (2026-05-08 20:46 CEST)

- Replacement dataset build/push `3598347`, short SFT array `3598348`, and SFT merge/sanity array `3598349` are all `COMPLETED` with exit code `0:0`.
- GRPO reward ablation array `3598350` has all rows `0-14` running, about `31m` elapsed. Rows `0-5` have reached `SynthTaskRunner`/W&B startup, and rows `0-2` show training progress around steps `2-6/500`.
- Rows `6-14` currently have zero stderr and near-zero batch CPU because the Slurm script intentionally staggers startup with `SLURM_ARRAY_TASK_ID * 300s + jitter`; this is expected for later array indices at this elapsed time, not an idle-GPU failure.
- Final merge + pass@k eval array `3598351` is dependency-held on `afterany:3598350_*`, as intended.
- Confirmed HF upload retry logic exists in `MaterializedDatasetBuilder.push_to_hub`: each subset upload retries up to 5 attempts with backoff. No dependency-dead child jobs remain in the monitored replacement chain.
- Log scan found no unresolved HF upload failure, `DependencyNeverSatisfied`, SFT/merge failure, Ray startup failure requiring action, OOM, quota error, or timeout. Early Ray worker-registration warnings appeared in rows that subsequently reached `SynthTaskRunner`.
- Action: no cancellation or resubmission needed in this pass.

Run-directory cleanup (2026-05-08):
- Deleted 18 obsolete run directories from `/home/atuin/c107fa/c107fa12/synthetic-RLVL/runs`:
  - all 15 previous `rl_hard_v5_full_fast_*` run directories,
  - all 3 previous `sft_hard_v5_full_lr1e-4_seed*` directories.
- Kept current `sft_hard_v5_short_lr1e-4_seed*` directories and all live `rl_hard_v5_short500_*` directories.
- Runs directory size dropped from about `1.3T` to `3.7G`.
- Deletion manifest: `tmp/delete_obsolete_runs_20260508.txt`.

## Hard-FSA Redesign + New Run Chain (2026-05-09)

Current repo-specific hard-v5-short GRPO/eval jobs were cancelled before launching the replacement experiment. Unrelated non-`synthetic-RLVL` jobs owned by the account were left untouched.

Implemented `difficulty=hard_fsa` as a finite-state automaton style dataset variant:

- Each example has `K=4` coherent branch trajectories.
- All branches start from the same visible initial state, so the first step is ambiguous.
- Only the gold branch has the derivable initial marker; wrong branches are locally coherent but globally invalid from the given facts.
- Branch order is shuffled at every step, so position and graph-connectivity heuristics should not reveal the gold path.
- Gold proofs are citation-free, so validity reward can target derivability rather than citation bookkeeping.
- Depth-20 prompt size remains bounded: `2 + 8 * depth = 162` premises and `2 * depth + 2 = 42` gold proof lines.

Code/config additions:

- Dataset generator: `synthetic_dataset.py`, `difficulty=hard_fsa`.
- Materializer support: `synthrlvl/datasets/materialize.py --difficulty hard_fsa`.
- SFT config: `conf/sft_hard_fsa_short.yaml`.
- GRPO config: `conf/posttrain_grpo_hard_fsa_500.yaml`.
- Dataset build job: `scripts/slurm/jobs/build_materialized_hard_fsa_2026-05-09.slurm`.
- SFT array: `scripts/slurm/sweeps/sft/hard_fsa_short_lr1e4_2026-05-09.slurm`.
- SFT merge/sanity array: `scripts/slurm/jobs/merge_sft_hard_fsa_short_2026-05-09.slurm`.
- GRPO reward ablation array: `scripts/slurm/sweeps/posttrain_hard_fsa_short500_reward_ablation_2026-05-09.slurm`.
- Merge + sampled pass@k eval array: `scripts/slurm/jobs/posthoc_hard_fsa_short500_merge_eval_passk_2026-05-09.slurm`.

Submitted chain:

- `3599283`: build and push `flaitenberger/LogicalReasoning-hard-fsa`.
- `3599284`: 3-seed short SFT, dependency `afterok:3599283`.
- `3599285`: merge/sanity for the three SFT checkpoints, dependency `afterok:3599284`.
- `3599286`: 15-job GRPO reward ablation, dependency `afterok:3599285`.
- `3599287`: 15-job final merge + sampled pass@k eval, dependency `afterok:3599286`.

GRPO ablations submitted for seeds `{3407,3408,3409}`:

- `correct_plus_0p1_format`
- `correct_plus_citation_free_valid_plus_0p1_format`
- `correct_times_citation_free_valid_plus_0p1_format`
- `correct_plus_citation_free_line_valid_plus_0p1_format`
- `correct_times_citation_free_line_valid_plus_0p1_format`

Also submitted post-hoc diagnostics for the previous hard-v5-short500 checkpoints:

- `3599282`: `classify_hard_v5_short500_failures_2026-05-09.slurm`, array `0-14%5`.
- Output target: `analysis/hard_v5_short500_failures/*.json`.
- Classifier categories include `correct_supported_by_valid_proof`, `correct_but_invalid_proof`, `shortcut_answer`, `shortcut_conclusion`, `gold_reached_then_broken`, `partial_valid_prefix`, `wrong_nonshortcut_answer`, and `format_or_empty_failure`.

Validation performed before submission:

- `python -m pytest tests/test_synthetic_dataset.py::test_hard_fsa_gold_trace_is_citation_free_valid_and_ambiguous -q` passed (`4 passed`).
- `python -m py_compile scripts/analysis/classify_hard_v5_failures.py synthetic_dataset.py synthrlvl/datasets/materialize.py` passed.
- Manual smoke check confirmed depths `3`, `10`, and `20` have citation-free valid gold proofs, distinct gold/shortcut final states, and shared branch initial states.

## Hard-FSA Strict Invariant Resubmission (2026-05-09)

After reviewing the initial `hard_fsa` generator, stricter invariants were added before training:

- Global `(state, marker)` pair uniqueness across all branch trajectories, preventing branch re-entry/merging at the automaton-state level.
- Shared visible initial state across branches, but unique initial markers; only the gold initial marker is given as a fact.
- Per-layer unique output states, preventing same-layer `state -> marker` ambiguity.
- Unique final answer states across all branches.
- No reused output atom `(state, constant)`, which prevents duplicate implication antecedents even when depth exceeds the `a-r` constant range and constants wrap.
- No duplicate implication antecedents in generated premises.

The first hard_fsa chain (`3599283`-`3599287`) was cancelled before SFT/GRPO. A fresh strict materialization root is now used: `${WORK}/synthetic-RLVL/datasets/materialized_logic_hard_fsa_strict_20260509`.

Resubmitted strict chain:

- `3599300`: build and push strict `flaitenberger/LogicalReasoning-hard-fsa`.
- `3599301`: 3-seed short SFT, dependency `afterok:3599300`.
- `3599302`: merge/sanity for the three SFT checkpoints, dependency `afterok:3599301`.
- `3599303`: 15-job GRPO reward ablation, dependency `afterok:3599302`.
- `3599304`: 15-job final merge + sampled pass@k eval, dependency `afterok:3599303`.

Validation before resubmission:

- `pytest tests/test_synthetic_dataset.py::test_hard_fsa_gold_trace_is_citation_free_valid_and_ambiguous -q` passed (`4 passed`).
- Manual invariant smoke checked 5 generated examples each at depths `3`, `10`, and `20`; all had citation-free valid gold proofs, unique `(state, marker)` pairs, shared initial state, unique initial markers, unique final states, and no duplicate implication antecedents.

## Hard-FSA Shortcut-Schema Experiment (2026-05-09)

Implemented a new `difficulty=hard_fsa_schema` dataset variant to directly test shortcut-rich training vs shortcut-neutral evaluation.

Design:

- Keeps the FSA proof structure: `(state_t, marker_t) -> state_{t+1}` and `state_{t+1} -> marker_{t+1}`.
- Uses natural state/marker words, mapped to one-letter predicates for the logic engine.
- Training examples use a configurable shortcut channel with `shortcut_rate=0.8`.
- Shortcut-enabled train examples have a shared family-level transition schema and marker redundancy, so a correctness-only model can learn a cheap shallow transition policy.
- Validation examples use `shortcut_rate=0.0`, falling back to strict exchangeable hard-fsa generation with no schema shortcut.
- Candidate answer metadata is balanced so gold candidate position is exactly uniform across generated indices.
- Citation-free gold proofs remain valid; citation bookkeeping is not part of the core signal.

Probe acceptance before submission:

- `scripts/analysis/probe_hard_fsa_schema.py --n-per-depth 100` passed.
- Train shortcut enabled rate: `0.83`.
- Train schema predictor accuracy: `0.83`.
- Eval schema predictor accuracy: `0.0`.
- Eval first/last candidate heuristics: exactly `0.25` each.
- Eval gold candidate position counts: exactly balanced across positions `0..3`.

New reward schema:

- `indicator_correct_and_citation_free_valid_plus_0p1_format`
- Implements `R = 1[correct and citation_free_valid] + 0.1 * format`.

Submitted chain:

- `3599557`: build/probe/push `flaitenberger/LogicalReasoning-hard-fsa-schema`.
- `3599558`: 3-seed shortcut-schema SFT, dependency `afterok:3599557`.
- `3599559`: SFT merge/sanity, dependency `afterok:3599558`.
- `3599560`: 6-job GRPO array, dependency `afterok:3599559`.
- `3599561`: final merge + sampled pass@k eval, dependency `afterok:3599560`.

GRPO conditions:

- Seeds: `3407`, `3408`, `3409`.
- Rewards: `correct_plus_0p1_format` and `indicator_correct_and_citation_free_valid_plus_0p1_format`.
- 500 GRPO steps, train depths `1..15`, validation depths `1..20`.

This chain runs in addition to the already-running strict `hard_fsa` chain; it does not cancel that earlier reasoning-only baseline.

## Hard-FSA Shortcut-Schema Runtime Update (2026-05-09)

Current active experiment is the shortcut-schema chain; older repo-specific hard-FSA jobs were cancelled when requested, while unrelated account jobs were left untouched.

Observed status:

- `3599557` build/probe/push completed successfully in `00:02:17`.
- Initial SFT array `3599558` produced one complete seed: `3409`.
- Initial SFT tasks for seeds `3407` and `3408` failed during online validation, not during training.
- Downstream initial dependency jobs `3599559`, `3599560`, and `3599561` were cancelled because the dependency chain was no longer satisfiable.

Root cause:

- Online validation rebuilt examples through `TaskBuilder` with `task.shortcut_rate=0.8`.
- For `difficulty=hard_fsa_schema`, train should be shortcut-rich but eval must be shortcut-neutral.
- The failed validation path attempted to generate shortcut-enabled schema examples outside the intended train distribution and hit a schema-state sampling failure.

Fix:

- `TaskBuilder` now keys cached generators by `(depth, train, shortcut_rate)`.
- For `hard_fsa_schema`, `train=False` forces `shortcut_rate=0.0`, so validation and post-hoc eval use the exchangeable shortcut-neutral split.
- Added regression coverage: `tests/test_training_stack.py::test_task_builder_keeps_hard_fsa_schema_eval_shortcut_neutral`.
- Targeted tests passed:
  - `python -m py_compile synthrlvl/task.py synthetic_dataset.py`
  - `pytest tests/test_training_stack.py::test_task_builder_keeps_hard_fsa_schema_eval_shortcut_neutral tests/test_synthetic_dataset.py::test_hard_fsa_schema_shortcut_mode_and_gold_validity -q`

Current resubmitted chain:

- `3599700`: retry SFT for failed seeds `3407` and `3408` only.
- `3599701`: merge/sanity for seeds `3407`, `3408`, `3409`, dependency `afterok:3599700`.
- `3599702`: 6-job GRPO array, dependency `afterok:3599701`.
- `3599703`: final merge + sampled pass@k eval, dependency `afterok:3599702`.

Partial failed SFT output directories for seeds `3407` and `3408` were moved aside with `_failed_eval_20260509_202629` suffixes before retry; seed `3409` final adapter output was kept.

### Hard-FSA Shortcut-Schema Runtime Check (2026-05-10)

- Retry SFT array `3599700` completed successfully for seeds `3407` and `3408`; the previously completed seed `3409` remains valid.
- Merge/sanity array `3599701` completed successfully for all three seeds.
- GRPO array `3599702_[0-5%6]` is running all six reward/seed rows.
- Current GRPO progress range is about `288/500` to `328/500` after about `11.6h`; all rows have written `global_step_250` actor checkpoints.
- Final merge + sampled pass@k eval array `3599703_[0-5%6]` remains dependency-held and should start after GRPO completion.
- No fatal log signatures were found in the active GRPO rows: no traceback, OOM, disk quota failure, dependency failure, or Ray startup failure requiring action. One Ray worker registration warning appeared in a row that subsequently reached normal training progress.
- Preliminary online eval at step `200` remains noisy due to only `10` samples per step, but shortcut-answer rates are low on shortcut-neutral eval, which is consistent with the intended eval intervention.

### Hard-FSA Shortcut-Schema 1000-Step Continuation (2026-05-10)

Submitted a continuation to test whether the shortcut-schema signal separates more clearly with longer GRPO:

- Cancelled pending step-500 pass@k array `3599703`.
- `3600041`: continuation GRPO array, dependency `afterany:3599702`, same six seed/reward rows, `RESUME_MODE=auto`, `TRAIN_STEPS=1000`.
- `3600042`: final merge + sampled pass@k eval, dependency `afterany:3600041`.

The pass@k script now searches for final actors in this order: `global_step_1001`, `global_step_1000`, `global_step_501`, `global_step_500`.

Interpretation note:

- `synthetic/step_N/shortcut_answer` measures whether the final answer equals the dataset metadata's coherent wrong branch answer.
- Low `shortcut_answer` in correctness-only does not imply good reasoning; current samples show many failures are arbitrary wrong-branch or malformed/partial-chain failures rather than the designated shortcut answer.

### Hard-FSA Shortcut-Schema Runtime Check (2026-05-10 Later)

- Initial GRPO wave `3599702_[0-5]` is still running; continuation `3600041` and pass@k `3600042` remain dependency-held.
- Current progress range is about `368/500` to `419/500`.
- Only `global_step_250` checkpoints exist so far; final `500/501` checkpoints have not been written yet.
- No fatal errors found in current logs. Warnings are tokenizer processor warnings and one old Ray worker-registration warning from a row that continued training normally.
- Latest online eval remains noisy (`10` samples per step). Current aggregate:
  - `correct_plus_0p1_format`: step10 correctness about `0.07`, step20 correctness about `0.03`, step20 citation-free validity about `0.20`.
  - `indicator_correct_and_citation_free_valid_plus_0p1_format`: step10 correctness about `0.13`, step20 correctness about `0.00`, step20 citation-free validity about `0.17`.
  - `shortcut_answer` remains low for both reward schemas.

### Hard-FSA Shortcut-Schema Runtime Check (2026-05-11)

- Initial 500-step GRPO wave `3599702_[0-5]` completed successfully.
- Continuation wave `3600041_[0-5]` is partially complete:
  - Validity-gated rows `1`, `3`, and `5` completed and wrote `global_step_1000`.
  - Correctness-only rows `0`, `2`, and `4` are still running around `933/1000` to `959/1000`; each has `global_step_750`.
- Final pass@k array `3600042_[0-5%6]` remains dependency-held until the remaining continuation rows finish.
- No unresolved fatal failures found. Some completed rows show shutdown-time Ray/vLLM/atexit warnings, but Slurm exit codes are `0:0` and final checkpoints exist.
- Latest online eval aggregate:
  - `correct_plus_0p1_format` at eval step `900`: step10 correctness about `0.13`, step15 about `0.13`, step20 about `0.00`; step20 citation-free validity about `0.27`.
  - `indicator_correct_and_citation_free_valid_plus_0p1_format` at eval step `1000`: step10 correctness about `0.10`, step15 about `0.00`, step20 about `0.00`; step20 citation-free validity about `0.30`.
  - The validity-gated model has higher long-step citation-free validity/format, but no clear correctness improvement in greedy online eval.

### Hard-FSA Schema Dense Line-Valid Rollout Ablation (2026-05-11)

Submitted a new 500-step GRPO rollout-count ablation for the dense gated citation-free line-valid reward:

- Slurm array: `3600829_[0-5%6]`
- Script: `scripts/slurm/sweeps/posttrain_hard_fsa_schema_short500_linevalid_rollouts_2026-05-11.slurm`
- Dataset/config: `posttrain_grpo_hard_fsa_schema_500`
- SFT initialization: `tmp/merged_sft_hard_fsa_schema_short_seed{3407,3408,3409}`
- Reward: `correct_times_citation_free_line_valid_plus_0p1_format`
- Reward formula: `R = correct * citation_free_line_valid_fraction + 0.1 * format`
- Rollout settings:
  - `num_prompts=8`, `num_rollouts=16`, seeds `3407,3408,3409`
  - `num_prompts=8`, `num_rollouts=32`, seeds `3407,3408,3409`
- W&B groups:
  - `posttrain_hard_fsa_schema_short500_rollouts/correct_times_citation_free_line_valid_plus_0p1_format_np8_nr16`
  - `posttrain_hard_fsa_schema_short500_rollouts/correct_times_citation_free_line_valid_plus_0p1_format_np8_nr32`

The array uses per-task startup staggering/jitter and is independent of the existing 1000-step continuation/pass@k chain (`3600041`, `3600042`).

A dependent merge + sampled pass@k eval array was also submitted:

- Slurm array: `3600839_[0-5%6]`
- Dependency: `afterany:3600829`
- Script: `scripts/slurm/jobs/posthoc_hard_fsa_schema_linevalid_rollouts_merge_eval_passk_2026-05-11.slurm`
- Output directory: `passk_eval/hard_fsa_schema_short500_rollouts/`
- It searches final actors in `global_step_501`, `500`, `251`, `250` order and skips rows with no available actor.

### Corrected Hard-FSA Schema Rollout / Line-Valid Submission (2026-05-11)

Correction to the immediately previous dense-line-valid rollout submission:

- Cancelled mistaken array `3600829` and dependent eval `3600839`.
- The mistaken array was cancelled during startup; only one local partial run directory was created, containing temporary parquet files, and it was removed.
- No matching local W&B run directory was found for the mistaken run names.

Submitted the intended 500-step GRPO array:

- Slurm array: `3600891_[0-8%9]`
- Script: `scripts/slurm/sweeps/posttrain_hard_fsa_schema_short500_rollout_and_linevalid_2026-05-11.slurm`
- SFT initialization: `tmp/merged_sft_hard_fsa_schema_short_seed{3407,3408,3409}`
- Rows:
  - `indicator_correct_and_citation_free_valid_plus_0p1_format`, `num_prompts=8`, `num_rollouts=16`, seeds `3407,3408,3409`
  - `indicator_correct_and_citation_free_valid_plus_0p1_format`, `num_prompts=8`, `num_rollouts=32`, seeds `3407,3408,3409`
  - `correct_times_citation_free_line_valid_plus_0p1_format`, `num_prompts=8`, `num_rollouts=8`, seeds `3407,3408,3409`
- W&B groups are under `posttrain_hard_fsa_schema_short500_corrected/<schema>_np<prompts>_nr<rollouts>`.

Submitted dependent merge + sampled pass@k eval:

- Slurm array: `3600900_[0-8%9]`
- Dependency: `afterany:3600891`
- Script: `scripts/slurm/jobs/posthoc_hard_fsa_schema_corrected_rollouts_merge_eval_passk_2026-05-11.slurm`
- Output directory: `passk_eval/hard_fsa_schema_short500_corrected/`

### Corrected Hard-FSA Schema Rollout Check (2026-05-11)

- Corrected training array `3600891_[0-8%9]` is running all nine rows.
- Dependent merge + pass@k eval `3600900_[0-8%9]` is pending on `afterany:3600891`.
- All corrected rows have reached Ray/vLLM startup and entered training; no fatal traceback/OOM/disk-quota signature found in current logs.
- Latest rough progress from Slurm logs:
  - `indicator_correct_and_citation_free_valid_plus_0p1_format`, `np8_nr16`: rows around `3/500` to `10/500` depending on startup delay/seed.
  - `indicator_correct_and_citation_free_valid_plus_0p1_format`, `np8_nr32`: rows around `1/500` to `6/500` depending on startup delay/seed.
  - `correct_times_citation_free_line_valid_plus_0p1_format`, `np8_nr8`: rows around `0/500` to `13/500` depending on startup delay/seed.
- Runtime caveat: `np8_nr16` and especially `np8_nr32` are substantially slower per GRPO step than `np8_nr8`; they will likely require continuation jobs to reach 500 steps.
- Older 1000-step continuation `3600041` is nearly finished; most rows are complete, with two correctness-only rows still near the final few steps. Its pass@k array `3600042` remains dependency-held.

### Runtime Check (2026-05-11 Later)

- Corrected rollout/dense-line-valid array `3600891_[0-8%9]` is running all nine rows after about `3.6h` wall time.
- Dependent corrected pass@k array `3600900_[0-8%9]` remains dependency-held.
- No fatal traceback, OOM, disk-quota, or Ray startup failure found in corrected-array logs. The repeated `Failed to create processor: Unsupported processor type: GPT2TokenizerFast` lines are warnings from VERL tokenization setup, not fatal failures.
- Approximate corrected-array progress:
  - `indicator_correct_and_citation_free_valid_plus_0p1_format`, `np8_nr16`: about `54-59/500`.
  - `indicator_correct_and_citation_free_valid_plus_0p1_format`, `np8_nr32`: about `30-35/500`.
  - `correct_times_citation_free_line_valid_plus_0p1_format`, `np8_nr8`: about `72-90/500`.
- Runtime expectation remains unchanged: `np8_nr16` and `np8_nr32` will need continuation or a multi-GPU resubmit to finish 500 steps quickly; `np8_nr8` is much closer to a one-allocation run.
- Older 1000-step continuation `3600041` completed successfully. Older pass@k array `3600042` is mostly complete; one row is still running.

### Multi-GPU GRPO Smoke Attempt (2026-05-11)

Implemented a configurable multi-GPU knob for VERL posttraining:

- `system.n_gpus_per_node` now drives `trainer.n_gpus_per_node` and rollout `n_gpus_per_node`.
- `system.ray_max_colocate_count` defaults to `3` to preserve existing one-GPU behavior.
- Multi-GPU smoke jobs set `system.ray_max_colocate_count=1` so FSDP ranks receive full, distinct GPU assignments rather than fractional colocated GPUs.

Smoke status:

- Initial 2-GPU smoke `3601634` failed during FSDP init with NCCL duplicate-GPU detection: rank 0 and rank 1 were both on the same CUDA device.
- A minimal Ray/Slurm diagnostic `3601663` completed successfully and showed ordinary Ray `num_gpus=1` actors get distinct GPUs under a 2-GPU Slurm allocation. Therefore Slurm GPU visibility is fine; the failure is specific to VERL fractional colocated workers.
- Patched the VERL resource-pool creation path in `posttrain_grpo_verl.py` to allow `system.ray_max_colocate_count=1` for multi-GPU smoke tests.
- Resubmitted 2-GPU smoke as `3601678`; it is pending on priority/resources at the time of this note.
- The 4-GPU `np8_nr32` smoke was cancelled before startup and should be retried only after the 2-GPU smoke reaches actual training progress.

No real experiment rows have been cancelled yet. Corrected one-GPU array `3600891` continues to run all nine rows.

### Multi-GPU Smoke Follow-Up (2026-05-11)

- Patched 2-GPU smoke `3601678` still failed with NCCL duplicate-GPU detection.
- Root cause: the resource-pool colocation patch was applied in the driver process, but VERL constructs the resource pool inside the remote `TaskRunner` process.
- Updated `SynthTaskRunner.run` to install `_install_resource_pool_colocation_patch(...)` inside the remote runner before `TaskRunner.run` initializes workers.
- Resubmitted 2-GPU smoke as `3601695`.
- Real one-GPU experiment array `3600891` remains untouched and running.

## 2026-05-11 Multi-GPU Rollout Resubmit

- The 2-GPU smoke for `np8_nr16` completed 5 GRPO steps successfully after enabling `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` and forwarding that env var through Ray runtime env.
- The 4-GPU smoke for `np8_nr32` reached `Training Progress` without the earlier NCCL duplicate-GPU failure, so startup was considered validated and the smoke was cancelled to free GPUs.
- Cancelled the slow 1-GPU high-rollout rows from `3600891`: tasks `0,1,3,4,6,7`.
- Kept the dense line-valid 1-GPU rows running: `3600891_2`, `3600891_5`, `3600891_8`.
- Removed local partial run directories for the cancelled high-rollout rows because they had no checkpoints yet.
- Resubmitted replacement arrays:
  - `3602083`: `indicator_correct_and_citation_free_valid_plus_0p1_format`, `num_prompts=8`, `num_rollouts=16`, seeds `3407,3408,3409`, 2 GPUs per row.
  - `3602084`: `indicator_correct_and_citation_free_valid_plus_0p1_format`, `num_prompts=8`, `num_rollouts=32`, seeds `3407,3408,3409`, 4 GPUs per row.
- Resubmitted dependent merge/pass@k eval as `3602089`, dependency `afterany:3600891:3602083:3602084`.
- New multi-GPU scripts:
  - `scripts/slurm/sweeps/posttrain_hard_fsa_schema_short500_nr16_2gpu_2026-05-11.slurm`
  - `scripts/slurm/sweeps/posttrain_hard_fsa_schema_short500_nr32_4gpu_2026-05-11.slurm`

## 2026-05-12 Gated Dense Validity Reward Ablation

Current 8x8 comparison from final in-process evals:

- `correct_plus_0p1_format`, 1000 steps: ID `1-10` correctness `0.453 +/- 0.025`; OOD `11-20` correctness `0.050 +/- 0.024`; step20 correctness `0.000`.
- `indicator_correct_and_citation_free_valid_plus_0p1_format`, 1000 steps: ID `1-10` correctness `0.447 +/- 0.025`; OOD `11-20` correctness `0.057 +/- 0.009`; step20 correctness `0.000`.
- `correct_times_citation_free_line_valid_plus_0p1_format`, 500 steps: ID `1-10` correctness `0.447 +/- 0.017`; OOD `11-20` correctness `0.067 +/- 0.038`; step20 correctness `0.000`.

Added reward schema:

- `citation_free_line_valid_plus_correct_if_full_valid_plus_0p1_format`
- Formula: `R = citation_free_line_valid_fraction + 1[citation_free_full_proof_valid] * correctness + 0.1 * format`
- Rationale: this gives dense local proof-validity shaping, but only pays correctness when the generated proof is a complete citation-free valid derivation. This is stricter than gating on `line_valid_fraction == 1`, because a short valid prefix can have line-valid fraction `1.0` while still not deriving the conclusion.

Submitted new 500-step 8x8 reward ablation:

- Training array: `3602693_[0-5%6]`
- Script: `scripts/slurm/sweeps/posttrain_hard_fsa_schema_short500_gated_validity_2026-05-12.slurm`
- Rows:
  - `citation_free_line_valid_plus_correct_if_full_valid_plus_0p1_format`, seeds `3407,3408,3409`
  - `correct_plus_citation_free_valid_plus_0p1_format`, seeds `3407,3408,3409`
- W&B groups under `posttrain_hard_fsa_schema_short500_gated_validity/<schema>_np8_nr8`.

Submitted dependent merge + pass@k eval:

- Eval array: `3602694_[0-5%6]`
- Dependency: `afterany:3602693`
- Script: `scripts/slurm/jobs/posthoc_hard_fsa_schema_gated_validity_merge_eval_passk_2026-05-12.slurm`
- Output directory: `passk_eval/hard_fsa_schema_short500_gated_validity/`

### 2026-05-12 Runtime Check

- `3602083` (`np8_nr16`, 2 GPUs): all 3 rows running at about `418-457/500`; checkpoint `400` exists for all rows. No fatal traceback/OOM/quota/NCCL duplicate-GPU signature found.
- `3602084` (`np8_nr32`, 4 GPUs): all 3 rows running at about `375-399/500`; checkpoints are at `300` or `400` depending on row. Slowest row is close to the 24h limit and may need continuation if it misses step 500.
- `3602693` (new 8x8 gated-dense/additive-validity ablation): all 6 rows running at about `68-78/500`; no checkpoint yet because save interval is 100.
- `3602089` and `3602694` pass@k arrays remain dependency-held.
- Patched `scripts/slurm/jobs/posthoc_hard_fsa_schema_corrected_rollouts_merge_eval_passk_2026-05-11.slurm` to search `501,500,401,400,301,300,251,250` so pending eval will not skip a row if a long rollout job times out after writing a 300/400 checkpoint.

### 2026-05-12 Runtime Check Later

- `3602083` (`np8_nr16`, 2 GPUs) completed all 3 seeds successfully at step `500`; final `global_step_500/actor` checkpoints exist for seeds `3407`, `3408`, and `3409`.
- The apparent `Traceback` lines in two `np8_nr16` logs are W&B teardown `BrokenPipeError` messages after successful W&B sync and Slurm `COMPLETED`; they are not training failures.
- `3602084` (`np8_nr32`, 4 GPUs) is still running at about `457-487/500` after about `22h`; all rows have at least `global_step_400/actor`. The slowest seed may be tight against the 24h walltime.
- `3602693` (new 8x8 gated-dense/additive-validity ablation) is running all 6 rows at about `163-184/500`; all rows have `global_step_100/actor`.
- Latest greedy eval summaries:
  - `np8_nr16` final step 500: ID correctness `0.460 +/- 0.022`, OOD correctness `0.037 +/- 0.009`, ID citation-free validity `0.673 +/- 0.026`, OOD validity `0.357 +/- 0.017`.
  - `np8_nr32` current step 400: ID correctness `0.450 +/- 0.022`, OOD correctness `0.070 +/- 0.028`, ID validity `0.657 +/- 0.021`, OOD validity `0.330 +/- 0.028`.
  - gated dense reward at step 100: ID correctness `0.450 +/- 0.024`, OOD correctness `0.067 +/- 0.024`, ID validity `0.653 +/- 0.038`, OOD validity `0.250 +/- 0.042`.
  - additive citation-free-valid reward at step 100: ID correctness `0.443 +/- 0.026`, OOD correctness `0.043 +/- 0.026`, ID validity `0.643 +/- 0.033`, OOD validity `0.237 +/- 0.049`.

### 2026-05-12 Runtime Check Late

- `3602084` (`np8_nr32`, 4 GPUs): seeds `3407` and `3409` completed to step `500`; seed `3408` is still running at about `491/500` after `23.5h`. It has `global_step_400/actor` and may need continuation only if it misses the 24h walltime.
- `3602693` (new 8x8 gated-dense/additive-validity ablation): all 6 rows are running at about `199-227/500`; all rows have `global_step_200/actor`.
- `3602089` and `3602694` pass@k arrays are still dependency-held; no pass@k outputs from these arrays yet.
- No fatal traceback, OOM, disk quota, or NCCL duplicate-GPU signature found in current active logs.
- Current greedy eval snapshot:
  - `np8_nr32`: two rows at step `500`, one at step `400`; ID correctness `0.453 +/- 0.021`, OOD correctness `0.057 +/- 0.025`, ID citation-free validity `0.680 +/- 0.022`, OOD validity `0.383 +/- 0.024`.
  - gated dense reward: step `200`; ID correctness `0.447 +/- 0.045`, OOD correctness `0.053 +/- 0.012`, ID validity `0.660 +/- 0.036`, OOD validity `0.277 +/- 0.074`.
  - additive citation-free-valid reward: mostly step `200` with one row still logged at step `100`; ID correctness `0.447 +/- 0.024`, OOD correctness `0.060 +/- 0.033`, ID validity `0.647 +/- 0.045`, OOD validity `0.227 +/- 0.037`.

### 2026-05-13 Completion Check

All current hard-FSA-schema corrected/gated jobs completed successfully.

Training completion:

- `3602083` (`np8_nr16`, 2 GPUs): all 3 seeds completed with `global_step_500/actor`.
- `3602084` (`np8_nr32`, 4 GPUs): all 3 seeds completed with `global_step_500/actor`.
- `3600891` dense line-valid `np8_nr8`: all 3 seeds completed with `global_step_500/actor`.
- `3602693` new gated-dense/additive-validity ablation: all 6 rows completed with `global_step_500/actor`.

Post-hoc pass@k completion:

- `3602089` corrected rollout pass@k array completed all 9 rows; outputs are in `${WORK}/synthetic-RLVL/passk_eval/hard_fsa_schema_short500_corrected/`.
- `3602694` gated/additive pass@k array completed all 6 rows; outputs are in `${WORK}/synthetic-RLVL/passk_eval/hard_fsa_schema_short500_gated_validity/`.
- Each completed pass@k row wrote a metrics JSON and sample JSONL, for 15 new pass@k metrics files across the corrected/gated submissions.
- Log scans found only non-fatal W&B teardown broken-pipe messages and NCCL/vLLM informational warnings after successful completion; no unresolved training/eval failure remains.

Pass@k snapshot, mean/std over 3 seeds:

- `np8_nr16`, `indicator_correct_and_citation_free_valid_plus_0p1_format`: OOD `correct_pass@1 = 0.042 +/- 0.006`, OOD `correct_pass@64 = 0.753 +/- 0.053`, OOD `citation_free_joint_pass@64 = 0.270 +/- 0.022`.
- `np8_nr32`, same reward: OOD `correct_pass@1 = 0.044 +/- 0.006`, OOD `correct_pass@64 = 0.767 +/- 0.041`, OOD `citation_free_joint_pass@64 = 0.270 +/- 0.024`.
- `np8_nr8`, `correct_times_citation_free_line_valid_plus_0p1_format`: OOD `correct_pass@1 = 0.044 +/- 0.006`, OOD `correct_pass@64 = 0.730 +/- 0.065`, OOD `citation_free_joint_pass@64 = 0.247 +/- 0.029`.
- `np8_nr8`, `citation_free_line_valid_plus_correct_if_full_valid_plus_0p1_format`: OOD `correct_pass@1 = 0.043 +/- 0.006`, OOD `correct_pass@64 = 0.743 +/- 0.060`, OOD `citation_free_joint_pass@64 = 0.260 +/- 0.008`.
- `np8_nr8`, `correct_plus_citation_free_valid_plus_0p1_format`: OOD `correct_pass@1 = 0.043 +/- 0.005`, OOD `correct_pass@64 = 0.720 +/- 0.107`, OOD `citation_free_joint_pass@64 = 0.270 +/- 0.024`.

No resubmission is needed for this batch.

## 2026-05-13 Hard-FSA-Schema Easy Curriculum

Implemented and submitted a smaller learnability-first curriculum because the previous K=4/depth-15 hard-FSA-schema setting did not reach high ID `correct@1`, making the validity-reward intervention hard to interpret.

Key design:

- Reuses `difficulty=hard_fsa_schema` with `branching_factor=2`.
- SFT trains on no-shortcut depths `1..5` for 3000 steps.
- GRPO trains on depths `1..5` and evaluates shortcut-neutral depths `1..20`.
- GRPO ablates train shortcut-rate `0.0` vs `0.5` and reward `correct_plus_0p1_format` vs `indicator_correct_and_citation_free_valid_plus_0p1_format`.
- The intended success criterion is ID `correct@1` saturation before using OOD behavior to judge shortcut-vs-validity effects.

New files:

- `conf/sft_hard_fsa_schema_easy.yaml`
- `conf/posttrain_grpo_hard_fsa_schema_easy_500.yaml`
- `scripts/slurm/jobs/build_materialized_hard_fsa_schema_easy_2026-05-13.slurm`
- `scripts/slurm/sweeps/sft/hard_fsa_schema_easy_lr1e4_2026-05-13.slurm`
- `scripts/slurm/jobs/merge_sft_hard_fsa_schema_easy_2026-05-13.slurm`
- `scripts/slurm/sweeps/posttrain_hard_fsa_schema_easy500_reward_ablation_2026-05-13.slurm`
- `scripts/slurm/jobs/posthoc_hard_fsa_schema_easy500_merge_eval_passk_2026-05-13.slurm`
- `docs/hard_fsa_schema_easy_curriculum_2026-05-13.md`

Dataset:

- HF: `flaitenberger/LogicalReasoning-hard-fsa-schema-easy`
- local root: `${WORK}/synthetic-RLVL/datasets/materialized_logic_hard_fsa_schema_easy_20260513`
- train subsets: `train_schema_easy0p0_up_to_5_50k`, `train_schema_easy0p5_up_to_5_50k`
- eval subsets: `val_step_01_1k` through `val_step_20_1k`

Submitted jobs:

- build + HF push: `3605313`
- SFT array: `3605314`
- SFT merge + sanity: `3605315`
- GRPO array: `3605316`
- final actor merge + pass@k eval: `3605317`

Dependency chain: `3605313 -> 3605314 -> 3605315 -> 3605316 -> 3605317`.

### Hard-FSA-Schema Easy Runtime Check (2026-05-13 14:52 CEST)

- Build/HF push job `3605313` completed successfully in `00:02:56`.
- The build log confirmed both probes were accepted.
- Example uploaded subset metadata was read back from HF for `train_schema_easy0p0_up_to_5_50k`, `train_schema_easy0p5_up_to_5_50k`, `val_step_05_1k`, and `val_step_20_1k`.
- SFT array `3605314_[0-2]` has started; downstream merge `3605315`, GRPO `3605316`, and pass@k `3605317` remain dependency-pending.

### Hard-FSA-Schema Easy Completion + First Results (2026-05-14)

All jobs in the easy curriculum chain completed successfully:

- Build/HF push `3605313`: completed.
- SFT `3605314_[0-2]`: completed, all final checkpoints merged by `3605315_[0-2]`.
- GRPO `3605316_[0-11]`: completed, all 12 rows at 500 steps.
- Merge + pass@k `3605317_[0-11]`: completed, all 12 rows produced metrics and samples.

Outputs:

- pass@k metrics and samples: `${WORK}/synthetic-RLVL/passk_eval/hard_fsa_schema_easy500/`
- merged SFT checkpoints: `${WORK}/synthetic-RLVL/tmp/merged_sft_hard_fsa_schema_easy_seed{3407,3408,3409}`

SFT sanity generations for all three seeds start with `<formal>` and contain complete formal blocks.

First pass@k result, mean/std over 3 seeds:

- Train band, depths 1..5: all four GRPO conditions reach about `correct_pass@1 = 0.998-0.999`, `citation_free_valid_pass@1 = 0.999`, and `citation_free_joint_pass@1 = 0.998`. This confirms the K=2/depth-5 curriculum is learnable.
- OOD band, depths 6..20: `correct_pass@1` is about `0.458-0.463`, `citation_free_valid_pass@1` about `0.314-0.326`, and `citation_free_joint_pass@1` about `0.269-0.282` across conditions.
- Hard tail, depths 15..20: `correct_pass@1` is about `0.187-0.196`, `citation_free_valid_pass@1` about `0.037-0.041`, and `citation_free_joint_pass@1` about `0.012-0.013` across conditions.
- Step 10 is strong: `correct_pass@1 ~= 0.70-0.73`, `citation_free_joint_pass@1 ~= 0.52-0.56`, and pass@16 is essentially saturated.
- Step 20 remains difficult: `correct_pass@1 ~= 0.06-0.07`; `citation_free_joint_pass@1 ~= 0.002-0.004`; `correct_pass@64` is high (`~0.83-0.97`) but joint valid+correct remains low (`~0.10-0.17`).

Interpretation:

- The easier curriculum fixed the earlier under-learning issue: ID is now near-perfect.
- OOD generalization exists up to roughly step 10, but long chains still fail mainly through citation-free validity/joint-validity collapse.
- The validity-gated reward is not yet producing a large separation from answer-only reward. In the shortcut-rate `0.5` condition it is slightly better on OOD citation-free-valid/joint metrics, but the effect is small relative to seed variance.
- Shortcut-rate `0.5` did not visibly harm train or OOD performance relative to shortcut-rate `0.0`; shortcut-answer rates in greedy eval remain low/moderate and do not dominate the failure mode.

## HFSA Easy Validity Diagnostic (2026-05-14)

A deep diagnostic report was added at `docs/hfsa_easy_validity_diagnostic_2026-05-14.md`.

Key conclusion: the current HFSA formal target concludes the final marker atom, while the task answer is the final state. Therefore citation-free validity can reward a proof that is internally valid but does not formally conclude the answer proposition. Training-depth validity is also saturated, and depth-20 evaluation is partly response-length capped because gold responses often exceed the current 1024-token generation cap.

Analysis artifacts are in `analysis/hfsa_easy_validity_2026-05-14/`, including pass@k figures, training reward-density plots, targeted probe outputs, and classified qualitative examples.

## Fixed-Target Hard-FSA-Schema Rerun (2026-05-14)

- Fixed `hard_fsa` and `hard_fsa_schema` gold traces so the final `<conclusion>` is the queried final state atom, matching `<answer>`, rather than the final marker atom.
- Added depth-25 support for no-shortcut K=4 HFSA eval by allowing state predicates to be reused across different constants while preserving unique output atoms and same-layer branch ambiguity.
- Focused dataset/task tests pass: `58 passed`.
- New docs: `docs/hard_fsa_schema_fixedtarget_2026-05-14.md`.
- Submitted one-seed fixed-target chain:
  - `3606767`: build/push `flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget`.
  - `3606768_[0-1]`: SFT `sft1to3` and `sft1to5`, seed `3407`.
  - `3606769_[0-1]`: merge SFT checkpoints and sanity-generate.
  - `3606770_[0-3]`: GRPO rows for `sft1to3_rl1to10` and `sft1to5_rl1to15`, rewards `correct_plus_0p1_format` vs `correct_plus_citation_free_valid_plus_0p1_format`.
  - `3606771_[0-3]`: post-hoc actor merge + pass@k eval over steps `1..25`.

## Fixed-Target GRPO Runtime Update (2026-05-15 09:47 CEST)

- Fixed-target prerequisites completed cleanly:
  - `3606767` dataset build/push completed in `00:03:32`.
  - `3606768` SFT array and `3606769` SFT merge array completed successfully.
  - SFT merge sanity generations for `sft1to3` and `sft1to5` start with `<formal>` and contain complete formal/answer blocks.
- Current GRPO array `3606770_[0-3]` is running and healthy, with recent training progress and nontrivial `sstat` CPU/RSS usage.
- Latest observed progress:
  - row 0 `sft1to3_rl1to10 / correct_plus_0p1_format`: about `199/500`, saved `global_step_200`.
  - row 1 `sft1to3_rl1to10 / correct_plus_citation_free_valid_plus_0p1_format`: about `199/500`, saved `global_step_200`.
  - row 2 `sft1to5_rl1to15 / correct_plus_0p1_format`: about `118/500`, saved `global_step_100`.
  - row 3 `sft1to5_rl1to15 / correct_plus_citation_free_valid_plus_0p1_format`: about `116/500`, saved `global_step_100`.
- These rows are too slow to finish 500 steps inside the first 24h allocation, especially the `rl1to15` rows.
- Cancelled old pass@k array `3606771`, which would have evaluated partial checkpoints after the first allocation ended.
- Submitted chained resume waves with `RESUME_MODE=auto`: `3608684 -> 3608685 -> 3608686`.
- Submitted replacement pass@k array `3608687` with dependency after `3608686`.

## Pure SFT Logic vs Natural-Language CoT First Wave (2026-05-15)

Added deterministic NL-proof-to-FOL validation and submitted a pure SFT/midtraining-style first wave because GRPO has not cleanly shown a validity-reward effect.

Implementation:

- New controlled NL trace translator/scorer: `synthrlvl/natural_logic.py`.
- `OutputEvaluator` now logs NL-to-logic diagnostics for natural templates: `nl_logic_parse`, `nl_logic_citation_free_valid`, `nl_logic_joint`, `nl_logic_line_valid_fraction`, and `nl_logic_valid_prefix_fraction`.
- pass@k now includes `nl_logic_*` metrics.
- `scripts/evaluate_checkpoint_passk.py` accepts OmegaConf dotlist overrides.
- Focused tests passed: `15 passed`.

Submitted jobs:

- SFT array `3612413_[0-3%4]`: `logic` vs `nl_exact`, train depths `1..10` and `1..15`, one seed `3407`, `10000` SFT steps.
- Dependent merge + posthoc eval array `3612414_[0-3%4]`.

Posthoc eval defaults:

- eval depths `1..25`.
- `128` prompts per proof length.
- sampled pass@k with `16` completions per prompt and `k=1,2,4,8,16`.
- constrained proof-line eval enabled for `logic` rows only.

Detailed plan/status: `docs/pure_sft_logic_vs_nl_2026-05-15.md`.

## Easy HFSA Completed Analysis Refresh (2026-05-15)

The completed hard-FSA-schema easy curriculum remains the cleanest GRPO result so far. Refreshed mean/std summary over all 12 pass@k files is in:

- `analysis/hfsa_easy_validity_2026-05-14/tables/easy500_passk_condition_summary_latest.csv`

Main result:

- Train depths `1..5` are saturated: correctness and citation-free joint validity are both about `0.998-0.999`.
- OOD depths `6..20` are not solved: `correct_pass@1 ~= 0.458-0.463`, `citation_free_joint_pass@1 ~= 0.269-0.282`.
- Hard tail `15..20` is mostly unsolved jointly: `correct_pass@1 ~= 0.187-0.196`, `citation_free_joint_pass@1 ~= 0.012-0.013`.
- Validity-gated reward is only slightly better in the shortcut-rate `0.5` condition and not enough to claim a robust algorithmic separation.
- Step 10 extrapolates reasonably; step 15 and especially step 20 collapse in valid+correct trace quality.

## Fixed-Target GRPO Runtime Update (2026-05-15 15:31 CEST)

Fixed-target GRPO `3606770_[0-3]` is still running:

- row 0 `sft1to3_rl1to10 / correct_plus_0p1_format`: about `292/500`.
- row 1 `sft1to3_rl1to10 / correct_plus_citation_free_valid_plus_0p1_format`: about `283/500`.
- row 2 `sft1to5_rl1to15 / correct_plus_0p1_format`: about `171/500`.
- row 3 `sft1to5_rl1to15 / correct_plus_citation_free_valid_plus_0p1_format`: about `170/500`.

Continuation waves `3608684 -> 3608685 -> 3608686` and replacement pass@k `3608687` remain queued. These rows are alive, but `rl1to15` is much slower and will need the continuation chain.
