# Hard-v5 Experiment Status, 2026-05-03

## Implementation

- Added `difficulty=hard_v5` to `synthetic_dataset.py`.
- Added materialized subsets `train_up_to_3_1k` and `train_up_to_15_120k`.
- Added `shortcut_rate` to dataset and task configs.
- Added citation-free proof metrics and rewards without changing strict proof behavior.
- Added pass@k metrics for `citation_free_valid`, `citation_free_joint`, and citation-free conditional metrics.

## Tests

Command:

```bash
source ./scripts/env.sh
${HPCVAULT}/.venv_rlvl_posttrain/bin/python -m pytest tests/test_logic_engine.py tests/test_synthetic_dataset.py tests/test_training_stack.py tests/test_pass_at_k.py -q
```

Result: `58 passed`.

A tiny materialization smoke test also completed locally under `tmp/hard_v5_materialize_smoke`.

## Slurm Chain

Scripts added:

- Dataset build + HF push: `scripts/slurm/jobs/build_materialized_hard_v5.slurm`
- SFT array: `scripts/slurm/sweeps/sft/hard_v5_lr1e4.slurm`
- SFT merge + sanity: `scripts/slurm/jobs/merge_sft_hard_v5_2026-05-03.slurm`
- GRPO reward ablation: `scripts/slurm/sweeps/posttrain_hard_v5_reward_ablation.slurm`
- Final actor merge + pass@k eval: `scripts/slurm/jobs/posthoc_hard_v5_merge_eval_passk_2026-05-03.slurm`

## Planned Jobs

- Build and push `flaitenberger/LogicalReasoning-hard-v5`.
- SFT 3 seeds: `3407`, `3408`, `3409`, 1,000 optimizer steps, batch size 1, grad accumulation 1, depths `1..3`.
- Merge all three SFT LoRA checkpoints into local HF checkpoints.
- GRPO 15 runs: 3 seeds x 5 reward schemas, depths `1..15`, validation `1..20`.
- Merge final actor LoRAs and run full post-hoc pass@k eval for every run.

## Checklist

- [x] Implement hard-v5 generator.
- [x] Implement citation-free rewards and metrics.
- [x] Add tests and run targeted test suite.
- [x] Add Slurm scripts and configs.
- [x] Submit dataset build job: `3577927` (running at submission).
- [x] Submit dependent SFT array: `3577928_[0-2%3]`, dependency `afterok:3577927`.
- [x] Submit dependent SFT merge/sanity array: `3577929_[0-2%3]`, dependency `afterok:3577928`.
- [x] Submit dependent GRPO reward-ablation array: `3577930_[0-14%8]`, dependency `afterok:3577929`.
- [x] Submit dependent merge + pass@k eval array: `3577931_[0-14%4]`, dependency `afterany:3577930`.


## Submission Update, 2026-05-03

Submitted full chain:

- Dataset build/push: `3577927`
- SFT: `3577928_[0-2%3]`
- SFT merge/sanity: `3577929_[0-2%3]`
- GRPO reward ablation: `3577930_[0-14%8]`
- Final merge + pass@k eval: `3577931_[0-14%4]`

At submission, `3577927` was running on `a0531`; all dependent jobs were pending on dependency.


## Status Update, 2026-05-04

- Dataset build/push `3577927` completed successfully in `00:01:39`.
- SFT jobs completed successfully: array tasks materialized as jobs `3577963`, `3577964`, and `3577928`.
- SFT merge/sanity completed successfully: jobs `3578119`, `3578120`, and `3577929`. Sanity generations start with `<formal>` and contain a complete formal block; the short 384-token sanity cap truncates the final answer text, so it is only a structure sanity check.
- GRPO array `3577930_[0-14%8]`: rows `0-7` are running, rows `8-14` are pending by array throttle, and final pass@k eval `3577931_[0-14%4]` is dependency-held.
- Important runtime issue: rows `0-4` reached only about `264-275/1500` after roughly `18h`; rows `5-7` reached about `407-413/1500`. At this rate, most rows will hit the `24h` time limit before final checkpoints. No `global_step_*` checkpoint directories were present yet at this check.

Recommendation: for the next intervention, either reduce `grpo.train_steps`/evaluate intermediate checkpoints, lower generation cost (`max_response_length`, `num_rollouts`, or validation cadence), or resubmit continuations with more frequent `validation.save_every` so timeouts do not lose all progress before step 500.

## Replacement Chain, 2026-05-04

The first hard-v5 GRPO chain was cancelled before completion:

- Cancelled: `3577930` and dependent eval `3577931`.
- Reason: the bottleneck was rollout generation length, not Ray startup. Logs showed frequent `response_length/max=2048`, nontrivial `response_length/clip_ratio`, and generation phases taking about `100-300s`. vLLM/VERL continuous batching-style rollout execution was already enabled through the existing async vLLM rollout path, chunked prefill, prefix caching, and dynamic actor batching, so the practical fix is better SFT stopping behavior plus a shorter rollout cap.

Replacement changes:

- Add SFT subset `train_up_to_3_50k` to `flaitenberger/LogicalReasoning-hard-v5`.
- Run full short-depth SFT on depths `1..3` with `50k` rows and `5,000` optimizer steps.
- Merge the three full-SFT checkpoints and sanity-check generations.
- Run the same five GRPO reward schemas from the new merged SFT checkpoints with `max_response_length=1024` and validation every `100` steps.
- Run final merge + sampled pass@k eval from the final actors with the same `1024` generation cap.

New scripts:

- `scripts/slurm/jobs/build_materialized_hard_v5_sft50k.slurm`
- `scripts/slurm/sweeps/sft/hard_v5_full_lr1e4.slurm`
- `scripts/slurm/jobs/merge_sft_hard_v5_full_2026-05-04.slurm`
- `scripts/slurm/sweeps/posttrain_hard_v5_full_fast_reward_ablation.slurm`
- `scripts/slurm/jobs/posthoc_hard_v5_full_fast_merge_eval_passk_2026-05-04.slurm`

Submitted replacement chain:

- `3581290`: build/push `train_up_to_3_50k`
- `3581291_[0-2%3]`: full SFT seeds `3407`, `3408`, `3409`
- `3581292_[0-2%3]`: merge full-SFT checkpoints and sanity-check generations
- `3581293_[0-14%8]`: GRPO 3 seeds x 5 reward schemas, startup jitter enabled
- `3581294_[0-14%4]`: final actor merge + sampled pass@k eval

Current status after submission:

- `3581290` completed successfully in `00:00:47`.
- HF dataset load check succeeded for `train_up_to_3_50k` with `50,000` rows and existing `train_up_to_15_120k` with `120,000` rows.
- `3581291_[0-2%3]` is pending by scheduler priority; `3581292`, `3581293`, and `3581294` are dependency-held.

## GRPO Startup Stagger Update, 2026-05-04

Replaced the pending GRPO/eval arrays so all GRPO reward-ablation tasks can be scheduled concurrently while still avoiding simultaneous Ray/vLLM startup:

- Cancelled pending arrays: `3581293` and `3581294`.
- New GRPO array: `3581300_[0-14%15]`, dependency `afterok:3581292`.
- New eval array: `3581301_[0-14%4]`, dependency `afterany:3581300`.
- Startup behavior: each GRPO task sleeps `SLURM_ARRAY_TASK_ID * 600s + random(0..59s)` before launching, so all 15 tasks can allocate concurrently but Ray/vLLM startup is staggered by about 10 minutes per task.

## GRPO Failure + Retry, 2026-05-05

Status:

- Dataset build `3581290`, full SFT `3581291`, and SFT merge `3581292` completed successfully.
- GRPO array `3581300` failed for all 15 rows.
- Dependent eval array `3581301` completed quickly but did not run real pass@k evaluation because final actor checkpoints were missing.

Root cause:

- The GRPO jobs were training normally, but failed at checkpoint save time with `OSError: [Errno 122] Disk quota exceeded` under `/home/atuin/c107fa/c107fa12/synthetic-RLVL/runs/.../global_step_*`.
- The previous retry used `validation.save_every=100` and did not cap retained actor checkpoints. With 15 concurrent jobs and roughly 28 GB per actor checkpoint, partial checkpoints accumulated too quickly.

Fix:

- `posttrain_grpo_verl.py` now forwards optional `validation.max_actor_ckpt_to_keep` into VERL as both `trainer.max_actor_ckpt_to_keep` and `trainer.max_critic_ckpt_to_keep`.
- `conf/posttrain_grpo_hard_v5_fast.yaml` now uses `validation.save_every=500` and `validation.max_actor_ckpt_to_keep=1`.
- `scripts/slurm/sweeps/posttrain_hard_v5_full_fast_reward_ablation.slurm` now defaults `SAVE_EVERY=500`.
- Deleted only the failed partial `rl_hard_v5_full_fast_*` run directories to clear quota pressure.

Retry submitted:

- `3584077_[0-14%15]`: GRPO retry with checkpoint retention. Initially submitted as `%5`, then raised to `%15` after cleaning obsolete `$WORK/runs` artifacts.
- `3584078_[0-14%4]`: dependent final actor merge + sampled pass@k eval, dependency `afterany:3584077`.

Cleanup before raising concurrency:

- Deleted 90 obsolete run directories from `/home/atuin/c107fa/c107fa12/synthetic-RLVL/runs`.
- Kept only the three `sft_hard_v5_full_lr1e-4_seed*` directories needed by the current experiment.
- Runs directory size dropped from about `5.4T` to `4.6G`.

## Runtime Check, 2026-05-05

- `3584077`: 14/15 rows are running and progressing. Row `9` failed during Ray startup with `The current node timed out during startup`; it failed before training and before creating a useful checkpoint.
- Quota is healthy: runs directory is about `5.7G`.
- Progress snapshot: row `0` is around `354/1500`; the other live rows are around `208-265/1500`.
- Expected runtime is still too long for a single 24h allocation, so checkpointed continuation is required.

Actions:

- Submitted row-9 retry: `3585899_[9%1]` with `RESUME_MODE=auto`.
- Submitted continuation wave 1: `3585900_[0-14%15]`, dependency `afterany:3584077:3585899`, `RESUME_MODE=auto`.
- Submitted continuation wave 2: `3585901_[0-14%15]`, dependency `afterany:3585900`, `RESUME_MODE=auto`.
- Updated final pass@k eval `3584078` to wait for `afterany:3585901`.

## Runtime Check, 2026-05-06

- Main wave `3584077` is still running near the 24h walltime.
- 14 rows have passed step 500 and wrote `global_step_500` checkpoints.
- Row `9` retry `3585899_9` is running and is around `323/1500`; it has not checkpointed yet, but should reach `500` before walltime at the observed rate.
- Current progress snapshot: row `0` around `728/1500`; other main-wave rows around `544-626/1500`.
- Runs directory size is about `395G`, consistent with one retained actor checkpoint for most rows.
- No new quota/OOM/Ray failures were found in the active logs; the only known failure remains the original row-9 Ray startup timeout from `3584077_9`.

## Runtime Check, 2026-05-07

- First wave `3584077` and row-9 retry `3585899` timed out as expected at 24h.
- Continuation wave `3585900_[0-14%15]` is running for all 15 rows.
- All rows resumed correctly from `global_step_500`; logs show optimizer/RNG/lr-scheduler restore from the step-500 actor checkpoints.
- Current progress snapshot for `3585900`: rows are around `803-995/1500`.
- No `global_step_1000` or final `global_step_1500` checkpoints have been written yet at this check.
- Runs directory size is about `423G`.
- No active quota/OOM/Ray failures were found.
- Second continuation wave `3585901_[0-14%15]` remains dependency-held, and final eval `3584078_[0-14%4]` remains dependency-held after `3585901`.

## Runtime Check, 2026-05-08

- Continuation wave `3585900` finished overall: some rows completed, some timed out after writing `global_step_1000`.
- Final continuation wave `3585901` ran/finished most rows.
- `10/15` runs now have final actor checkpoints at `global_step_1500` or `global_step_1501`.
- The remaining live rows are exactly the missing final actors:
  - task `6`: `correct_plus_citation_free_valid_plus_0p1_format`, seed `3408`, around `1431/1500`
  - task `7`: `correct_times_citation_free_valid_plus_0p1_format`, seed `3408`, around `1452/1500`
  - task `10`: `correct_plus_0p1_format`, seed `3409`, around `1450/1500`
  - task `11`: `correct_plus_citation_free_valid_plus_0p1_format`, seed `3409`, around `1450/1500`
  - task `12`: `correct_times_citation_free_valid_plus_0p1_format`, seed `3409`, around `1481/1500`
- These five are close to completion and should finish within the current allocation if no new startup/teardown issue occurs.
- Runs directory is about `1.1T`; this is expected with final actors plus retained intermediate checkpoints.
- Eval `3584078_[0-14%4]` remains dependency-held and should start after the remaining `3585901` tasks finish.

## Eval Check, 2026-05-08

- Training is fully complete: all 15 runs have final actor checkpoints at `global_step_1500` or `global_step_1501`.
- Merge + post-hoc pass@k eval `3584078_[0-14%4]` is running.
- `12/15` eval rows completed successfully.
- Remaining eval rows `12`, `13`, and `14` are running and have reached model/vLLM initialization; no eval failure signatures were found.
- Current output coverage: `12` pass@k JSON files and `12` sample JSONL files in `passk_eval/hard_v5_full_fast`.
- Raised active eval array throttle to `%15`; only rows `12`, `13`, and `14` were still running at that point.
- Updated the eval script default to run all rows concurrently with jitter for future submissions: default `PASSK_STAGGER_SECONDS=120` and `PASSK_JITTER_SECONDS=30`.
