# Hard-v3 Experiment Status (2026-04-30)

HF dataset: `flaitenberger/LogicalReasoning-hard-v3`

Hard-v3 is a compact adversarial variant intended to make the answer difficult to recover without correct reasoning. It keeps gold formal proof length unchanged and adds a bounded adversarial premise budget:

- wrong-answer facts for other entities
- one full wrong-entity decoy chain
- query-entity dead-end branch rules
- missing-support conjunction traps
- entity-specific near-miss rules

Generation budget: `2 * depth + 12` adversarial premises. Step-20 rows have about 73 formal premises and 31 proof lines.

## Shuffled Natural Theory

The natural-language theory shown in the prompt is deterministically shuffled and renumbered for `hard_v3` (`metadata.nl_premises_shuffled = true`). This removes the ordering shortcut where the gold chain appears first.

The formal target premises are **not** shuffled. They still contain the full theory and remain in canonical order so proof citations are stable. The task should therefore not be solved by premise selection alone; the model sees all facts/rules formally and must construct a valid proof over them.

## Dataset Checks

Reloaded from HF after regenerating and checked representative rows from:

- `train_up_to_10_1m`
- `val_step_10_1k`
- `val_step_20_1k`

Checked rows had:

- `metadata.nl_premises_shuffled = true`
- valid and conclusion-supported formal proofs
- no duplicate natural-language premises
- shuffled natural premise order
- complete canonical formal premise order

## Cancelled Superseded Chain

Cancelled the pre-shuffle hard-v3 chain:

- SFT `3571150`
- SFT merge/sanity `3571151`
- GRPO `3571152`
- final merge+pass@k eval `3571153`

## Active Submitted Chain

Submitted after shuffled-NL dataset upload on 2026-04-30:

1. SFT array: `3571290`
   - Script: `scripts/slurm/sweeps/sft/hard_v3_lr1e4.slurm`
   - Array: `0-2%3`
   - Seeds: `3407`, `3408`, `3409`
   - Config: `conf/sft_hard_v3.yaml`
   - W&B group: `sft_hard_v3/lr1e-4`

2. SFT merge + sanity array: `3571291`
   - Dependency: `afterok:3571290`
   - Script: `scripts/slurm/jobs/merge_sft_hard_v3_2026-04-30.slurm`
   - Array: `0-2%3`
   - Outputs: `$WORK/synthetic-RLVL/tmp/merged_sft_hard_v3_seed{3407,3408,3409}`

3. GRPO reward ablation array: `3571292`
   - Dependency: `afterok:3571291`
   - Script: `scripts/slurm/sweeps/posttrain_hard_v3_reward_ablation.slurm`
   - Array: `0-14%7`
   - Startup stagger: 600 seconds per array index plus up to 60 seconds jitter
   - Config: `conf/posttrain_grpo_hard_v3.yaml`
   - W&B groups: `posttrain_hard_v3/{schema}`

4. Final actor merge + pass@k eval array: `3571293`
   - Dependency: `afterany:3571292`
   - Script: `scripts/slurm/jobs/posthoc_hard_v3_merge_eval_passk_2026-04-30.slurm`
   - Array: `0-14%8`
   - Outputs: `$WORK/synthetic-RLVL/passk_eval/hard_v3/`

## GRPO Array Mapping

- `0`: seed `3407`, `correct_plus_0p1_format`
- `1`: seed `3407`, `correct_plus_valid_plus_0p1_format`
- `2`: seed `3407`, `correct_times_valid_plus_0p1_format`
- `3`: seed `3407`, `correct_plus_line_valid_plus_0p1_format`
- `4`: seed `3407`, `correct_times_line_valid_plus_0p1_format`
- `5`: seed `3408`, `correct_plus_0p1_format`
- `6`: seed `3408`, `correct_plus_valid_plus_0p1_format`
- `7`: seed `3408`, `correct_times_valid_plus_0p1_format`
- `8`: seed `3408`, `correct_plus_line_valid_plus_0p1_format`
- `9`: seed `3408`, `correct_times_line_valid_plus_0p1_format`
- `10`: seed `3409`, `correct_plus_0p1_format`
- `11`: seed `3409`, `correct_plus_valid_plus_0p1_format`
- `12`: seed `3409`, `correct_times_valid_plus_0p1_format`
- `13`: seed `3409`, `correct_plus_line_valid_plus_0p1_format`
- `14`: seed `3409`, `correct_times_line_valid_plus_0p1_format`

## Runtime Update (2026-05-01)

SFT `3571290` and merge/sanity `3571291` completed successfully. Merged SFT checkpoints exist at:

- `$WORK/synthetic-RLVL/tmp/merged_sft_hard_v3_seed3407`
- `$WORK/synthetic-RLVL/tmp/merged_sft_hard_v3_seed3408`
- `$WORK/synthetic-RLVL/tmp/merged_sft_hard_v3_seed3409`

SFT sanity generations start with `<formal>` for all three seeds. The sanity generation cap is 256 new tokens, so `contains_formal=false` only means the sample did not reach the closing `</formal>` tag within that cap.

The first GRPO wave `3571292_0` through `3571292_6` is healthy but too slow for the 24h walltime: at about 22h runtime, tasks are near step 1150 with checkpoints at steps 500 and 1000. Slurm refused extending the running job walltime.

Queued continuation/recovery jobs:

- `3572663`: resumes GRPO tasks `0-6`, dependency `afterany` on running task job IDs `3571439` through `3571445`.
- `3572664`: resumes GRPO tasks `7-14`, dependency `afterany:3571292`.
- Cancelled original eval `3571293` because it depended only on the original GRPO array and could have started before continuations finished.
- Submitted replacement final merge+pass@k eval `3572665`, dependency `afterany:3572663:3572664`.

Current GRPO health from logs:

- no response clipping (`response_length/clip_ratio=0.0`)
- no aborted responses (`response/aborted_ratio=0.0`)
- format and correctness rewards are high in training batches
- validity/line-validity are lower and variable than on hard-v1, which is expected for the harder shuffled/adversarial setting

## Immediate Second-Wave Start (2026-05-01)

The original pending tasks `3571292_7` through `3571292_14` were cancelled so the second GRPO wave would not wait behind the first wave's array throttle. The delayed retry `3572664` and eval `3572665` were also cancelled.

Submitted immediate second-wave GRPO array:

- `3572693`: tasks `7-14`, array `7-14%8`, no dependency
- Uses `STARTUP_STAGGER_SECONDS=600`, `STARTUP_STAGGER_BASE_ID=7`, `STARTUP_JITTER_SECONDS=60`, `RESUME_MODE=auto`
- All tasks allocated and started; the script-level stagger makes task 7 begin first and later tasks sleep before launching Ray.

Submitted replacement eval:

- `3572694`: final merge+pass@k eval, dependency `afterany:3572663:3572693`

The first-wave continuation remains:

- `3572663`: resumes tasks `0-6` after the currently running first-wave task jobs finish/time out.

## Oversight Check (2026-05-01 21:14 CEST)

Scheduler state:

- `3571292_0`-`3571292_6`: still running at about 24h elapsed; these are expected to time out or finish partial work and are covered by continuation `3572663_[0-6%7]`.
- `3572663_[0-6%7]`: pending on dependency, ready to resume rows `0-6`.
- `3572693_7`-`3572693_14`: running.
- `3572694_[0-14%8]`: dependency-held for final merge + pass@k eval.
- Oversight: `3572763` running; `3572764`-`3572767` pending by begin time.

Log/progress state:

- First-wave rows have recent progress and checkpoint evidence; latest observed `training/global_step` values were `1184`, `1442`, `1148`, `1148`, `1148`, `1184`, and `1148` for rows `0`-`6`.
- Second-wave rows are past Ray/model startup and emitting `Training Progress` in stderr. Latest observed progress was approximately rows `7-14`: `103`, `94`, `83`, `72`, `59`, `52`, `41`, and `31` of `1500`.
- Current hard-v3 logs show no Ray node startup failure, fatal traceback, OOM, or idle-after-runtime-env-upload pattern. Live `sstat` for running hard-v3 batch steps shows nontrivial CPU/RSS usage.

Action:

- No cancellation or resubmission was needed. Slow rows should continue through checkpointed continuation waves rather than being cancelled.

## Oversight Check (2026-05-01 23:54 CEST)

Scheduler state:

- Original first-wave task jobs `3571439`-`3571445` timed out at the 24h walltime after making progress; this was expected and is covered by continuation array `3572663`.
- First-wave continuation `3572663_0`-`3572663_6` is running and has resumed rows `0-6`.
- Second-wave GRPO `3572693_7`-`3572693_14` is running and making steady progress.
- Submitted dependent second-wave continuation `3573037_[7-14%8]` with `RESUME_MODE=auto`, 600s per-index startup stagger, and 60s jitter so rows `7-14` are covered if `3572693` hits walltime.
- Updated final merge/pass@k eval `3572694_[0-14%8]` to wait on `3572663`, `3572693`, and `3573037`.
- Oversight `3572763` completed successfully; `3572764` is running; `3572765`-`3572767` remain pending by begin time.

Progress / health:

- Latest continuation progress for rows `0-6` is about `1149`, `1140`, `1129`, `1118`, `1109`, `1097`, and `1088` of `1500`.
- Latest second-wave progress for rows `7-14` is about `271`, `262`, `251`, `239`, `225`, `218`, `206`, and `196` of `1500`.
- Logs show active `Training Progress` and nontrivial live CPU/RSS usage. The only Ray messages in current logs are startup/package notices and transient worker-registration warnings; no fatal Ray startup failure, traceback, OOM, or idle-GPU pattern was found.

Action:

- No running job was cancelled. The new continuation `3573037` and updated eval dependency prevent premature eval if rows `7-14` time out before final actors are written.

## Oversight Check (2026-05-02 03:55 CEST)

Scheduler state:

- First-wave continuation `3572663_0`-`3572663_6` is running and covers hard-v3 rows `0-6`.
- Second-wave GRPO `3572693_7`-`3572693_14` is running and covers rows `7-14`.
- Second-wave continuation `3573037_[7-14%8]` remains dependency-held on `3572693`.
- Final merge/pass@k eval `3572694_[0-14%8]` is dependency-held on `3572663`, `3572693`, and `3573037`.
- Oversight jobs `3572763` and `3572764` completed successfully, `3572765` is running, and `3572766`/`3572767` remain pending by begin time.

Progress / health:

- Latest observed resumed progress for rows `0-6` is about `1277`, `1268`, `1268`, `1268`, `1268`, `1277`, and `1268` of `1500`.
- Latest observed second-wave progress for rows `7-14` is about `268`, `268`, `268`, `277`, `268`, `268`, `268`, and `268` of `1500`.
- Live `sstat` for the running hard-v3 batch steps shows nontrivial CPU/RSS usage. Logs contain expected Ray startup/package warnings and transient worker-registration warnings only; no fatal Ray startup failure, traceback, OOM, timeout/cancelled task beyond the expected original first-wave walltime exits, or idle-after-runtime-env-upload pattern was found.

Action:

- No cancellation or resubmission was needed. All rows without final actors remain covered by live GRPO jobs or dependency-held continuation/eval jobs.

## Oversight Check (2026-05-02 11:03 CEST)

Scheduler state:

- First-wave continuation `3572663` completed successfully; rows `0-6` now have `global_step_1500/actor` final checkpoints.
- Second-wave GRPO `3572693_7`-`3572693_14` is still running at about 15h40m elapsed.
- Second-wave continuation `3573037_[7-14%8]` remains dependency-held on `3572693`.
- Final merge/pass@k eval `3572694_[0-14%8]` remains dependency-held on `3572693` and `3573037`; the fulfilled `3572663` dependency has dropped from `scontrol show job`.
- Oversight jobs `3572763`, `3572764`, and `3572765` completed successfully; `3572766` is running and `3572767` is pending by begin time.

Progress / health:

- Latest stderr training progress for rows `7-14` is about `969`, `959`, `952`, `930`, `907`, `902`, `895`, and `882` of `1500`.
- Live `sstat` for the running hard-v3 batch steps shows substantial CPU time and RSS usage, so there is no idle-GPU/Ray-packaging hang pattern.
- Rows `0-6` logged DataLoader/W&B teardown errors after reaching step `1500`, but Slurm marked the continuation tasks `COMPLETED` and the final actor directories exist.
- No fatal Ray startup failure, OOM, traceback before final checkpoint, unexpected cancellation, or uncovered failed row was found.

Action:

- No cancellation or resubmission was needed. Rows `7-14` are progressing slowly and remain covered by the live second-wave job plus dependent continuation `3573037`.

## Oversight Check (2026-05-02 11:58 CEST)

Scheduler state:

- Second-wave GRPO `3572693_7`-`3572693_14` is still running at about 16h36m elapsed.
- Second-wave continuation `3573037_[7-14%8]` remains dependency-held on `3572693`.
- Final merge/pass@k eval `3572694_[0-14%8]` remains dependency-held on `3572693` and `3573037`.
- Oversight jobs `3572763`-`3572766` completed successfully; `3572767` is running.

Progress / health:

- Rows `0-6` still have final actors at `global_step_1500`; rows `7-14` do not yet have final actors.
- Latest stderr training progress for rows `7-14` is about `1022`, `1013`, `1007`, `989`, `965`, `960`, `952`, and `940` of `1500`.
- Live `sstat` for the running hard-v3 batch steps shows substantial CPU time and RSS usage, so there is no idle-GPU/Ray-packaging hang pattern.
- Current log scan found only expected tokenizer/Ray/NCCL warnings. No fatal Ray startup failure, OOM, traceback before checkpoint, unexpected cancellation, or uncovered failed row was found.

Action:

- No cancellation or resubmission was needed. Rows `7-14` are progressing slowly and remain covered by the live second-wave job plus dependent continuation `3573037`.
