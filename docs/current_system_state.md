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
