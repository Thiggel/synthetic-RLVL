# Hard-v1 Experiment Status (2026-04-27)

## Dataset

HF dataset: `flaitenberger/LogicalReasoning-hard-v1`

Uploaded configs:

- `train_up_to_5_1m`: 1,000,000 rows
- `train_up_to_10_1m`: 1,000,000 rows
- `val_step_01_1k` through `val_step_20_1k`: 1,000 rows each

Fresh HF reload checks passed for representative train/validation subsets:

- `LogicEngine.analyze_proof` accepts sampled gold proofs.
- No duplicate premises in checked rows.
- No duplicate proof formulas in checked rows.
- No direct final-answer premise leakage in checked rows.
- Expected hard-v1 proof lengths hold, including depth 20 -> 31 proof lines.

## Submitted Chain

The clean hard-v1 experiment uses hard-v1 SFT before RL. This avoids confounding reward-design results with adaptation from an easier SFT trace distribution.

1. SFT array: `3566369`
   - Script: `scripts/slurm/sweeps/sft/hard_v1_lr1e4.slurm`
   - Array: `0-2%1`
   - Seeds: `3407`, `3408`, `3409`
   - W&B group: `sft_hard_v1/lr1e-4`
   - Run names: `sft_hard_v1_lr1e-4_seed{seed}`
   - Config: `conf/sft_hard_v1.yaml`
   - Overrides: `train.lr=1e-4`, `train.max_steps=1000`, `train.eval_steps=500`, `train.save_steps=500`, `data.train_samples=50000`

2. SFT merge + sanity array: `3566370`
   - Script: `scripts/slurm/jobs/merge_sft_hard_v1_2026-04-27.slurm`
   - Dependency: `afterok:3566369`
   - Array: `0-2%1`
   - Input LoRA dirs: `$WORK/synthetic-RLVL/runs/sft_hard_v1_lr1e-4_seed{seed}/final`
   - Merged outputs: `$WORK/synthetic-RLVL/tmp/merged_sft_hard_v1_seed{seed}`
   - Sanity outputs: `$WORK/synthetic-RLVL/tmp/sanity_sft_hard_v1_seed{seed}.json`

3. GRPO reward ablation array: `3566371`
   - Script: `scripts/slurm/sweeps/posttrain_hard_v1_reward_ablation.slurm`
   - Dependency: `afterok:3566370`
   - Array: `0-14%2`
   - Config: `conf/posttrain_grpo_hard_v1.yaml`
   - Init checkpoints: `$WORK/synthetic-RLVL/tmp/merged_sft_hard_v1_seed{seed}`
   - Train steps: `1500`
   - W&B groups: `posttrain_hard_v1/{schema}`

Reward schemas:

- `correct_plus_0p1_format`
- `correct_plus_valid_plus_0p1_format`
- `correct_times_valid_plus_0p1_format`
- `correct_plus_line_valid_plus_0p1_format`
- `correct_times_line_valid_plus_0p1_format`

4. Final actor merge + pass@k eval array: `3566372`
   - Script: `scripts/slurm/jobs/posthoc_hard_v1_merge_eval_passk_2026-04-27.slurm`
   - Dependency: `afterany:3566371`
   - Array: `0-14%2`
   - Actor inputs: `$WORK/synthetic-RLVL/runs/rl_hard_v1_{schema}_seed{seed}_mrg/global_step_{1500,1501}/actor`
   - Merged actor outputs: `$WORK/synthetic-RLVL/tmp/merged_actor_rl_hard_v1_{schema}_seed{seed}_mrg_final`
   - Pass@k outputs: `$WORK/synthetic-RLVL/passk_eval/hard_v1/reward_rl_hard_v1_{schema}_seed{seed}_mrg_passk.json`
   - Sample outputs: `$WORK/synthetic-RLVL/passk_eval/hard_v1/reward_rl_hard_v1_{schema}_seed{seed}_mrg_samples.jsonl`

Pass@k settings:

- Steps: `1-20`
- Samples per step: `20`
- Generations per prompt: `64`
- k-values: `1,2,4,8,16,32,64`
- Backend: `vLLM`
- External benchmarks disabled for this posthoc pass.

## Monitoring Commands

```bash
squeue -u c107fa12 -o '%.18i %.9P %.32j %.2t %.10M %.6D %R'
sacct -j 3566369 --format=JobIDRaw,JobName%34,State,Elapsed,ExitCode -n -P
sacct -j 3566370 --format=JobIDRaw,JobName%34,State,Elapsed,ExitCode -n -P
sacct -j 3566371 --format=JobIDRaw,JobName%34,State,Elapsed,ExitCode -n -P
sacct -j 3566372 --format=JobIDRaw,JobName%34,State,Elapsed,ExitCode -n -P
```

## 2026-04-29 Throttle Update

- Increased GRPO reward-ablation array `3566371` from `%2` to `%4` with `scontrol update JobId=3566371 ArrayTaskThrottle=4`.
- Reason: completed hard-v1 GRPO tasks take about 22.3h each, so `%2` would make the 15-task sweep unnecessarily slow.
- Current running tasks after the change: `3566371_2`, `3566371_3`, `3566371_4`, `3566371_5`.
- Pass@k eval array `3566372` remains dependency-held and throttled at `%2`; this is still appropriate because each eval loads/merges a large model and runs sampled pass@k.

## 2026-04-29 Ray Startup Retry

- After temporarily increasing `3566371` to `%4`, original array tasks `4-10` started too aggressively and failed quickly with Ray node startup timeouts before training progress.
- Reduced original array `3566371` back to `%2` with `scontrol update JobId=3566371 ArrayTaskThrottle=2`.
- Submitted conservative retry array `3570395` for failed task IDs `4-10` with `--array=4-10%1`, `STARTUP_JITTER_SECONDS=900`, and `RESUME_MODE=auto`.
- Retry dependency: `afterany:3566371`, so it waits until the original hard-v1 array finishes remaining running/pending tasks.
- Updated pass@k eval array `3566372` dependency to wait for both `3566371` and retry array `3570395`.

## 2026-04-29 Staggered Retry Replacement

- Replaced conservative retry array `3570395` with staggered retry array `3570401` for failed original task IDs `4-10`.
- Retry array: `3570401_[4-10%7]`.
- Startup controls: `STARTUP_STAGGER_SECONDS=600`, `STARTUP_STAGGER_BASE_ID=4`, `STARTUP_JITTER_SECONDS=60`, `RESUME_MODE=auto`.
- This queues all retry jobs immediately but staggers Ray startup by approximately 10 minutes per task ID: task 4 starts first, task 5 after ~10m, ..., task 10 after ~60m.
- Updated pass@k eval array `3566372` dependency to wait for original array `3566371` and retry array `3570401`.

## 2026-04-30 Maintenance / Tail Resubmit / Eval Throttle

- A100 partition still reports many nodes in `maint`; original tail tasks `3566371_13` and `3566371_14` remained pending with `ReqNodeNotAvail, Reserved for maintenance`.
- Canceled original pending tail tasks and submitted fresh tail array `3571114_[13-14%2]` for the two missing seed-3409 schemas.
- Fresh tail array is also pending with the same maintenance reason, so this is a scheduler/partition availability issue rather than stale job state.
- Updated pass@k eval array `3566372` to depend on `3571114`.
- Increased pass@k eval throttle from `%2` to `%4` with `scontrol update JobId=3566372 ArrayTaskThrottle=4`.
- Completed hard-v1 GRPO tasks `0-12` all have `global_step_1500` actor checkpoints. Remaining missing tasks are `13` and `14` only.
