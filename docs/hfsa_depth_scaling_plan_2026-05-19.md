# HFSA Depth-Scaling SFT Plan - 2026-05-19

## Motivation

The strongest current signal is from the pure-SFT fixed-target HFSA comparison: logic CoT appears to extrapolate better than controlled natural-language CoT when training depth is extended from `1..10` to `1..15`, especially under sampled pass@k and joint correct+valid metrics. That result is still only one seed and only evaluates to depth 25.

This wave tests whether the trend is robust across seeds and over longer reasoning ranges.

This experiment is now part of the broader active plan in:

```bash
docs/formal_logic_cot_research_plan_2026-05-19.md
```

The earlier RL-validity-reward direction is archived as old state in:

```bash
docs/old_rl_validity_reward_direction_2026-05-19.md
```

## Current Dataset

Dataset family: `hard_fsa_schema` in shortcut-neutral fixed-target mode.

HF dataset for this wave:

```bash
flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget-depth50
```

The task is a finite-state automaton proof problem:

- each example has depth `D`;
- the prompt contains natural-language facts and transition rules;
- each step has `K=4` locally plausible branch transitions;
- exactly one branch is reachable from the current state-marker pair;
- wrong branches remain coherent, so shallow branch selection is not trivially detectable;
- the gold proof derives alternating state and marker atoms and stops at the queried final state;
- proof length is `2D + 1` lines;
- proof validity can be checked citation-free by the logic engine;
- the final `<answer>` is the natural state label corresponding to the final proof conclusion.

The current depth-scaling dataset is intentionally shortcut-neutral: `shortcut_rate=0.0` for train and eval. Shortcut-rich SFT remains a planned follow-up, not part of the first depth-scaling grid.

## Materialized Subsets

The materializer now supports explicit train subsets up to depths 20 and 25 in addition to the previous 3/5/10/15 subsets.

Planned subsets:

| subset | depths | rows |
| --- | ---: | ---: |
| `train_fixedtarget_up_to_5_50k` | `1..5` | 50,000 |
| `train_fixedtarget_up_to_10_50k` | `1..10` | 50,000 |
| `train_fixedtarget_up_to_15_50k` | `1..15` | 50,000 |
| `train_fixedtarget_up_to_20_50k` | `1..20` | 50,000 |
| `train_fixedtarget_up_to_25_50k` | `1..25` | 50,000 |
| `val_step_01_1k` ... `val_step_50_1k` | fixed `D` | 1,000 each |

Build script:

```bash
scripts/slurm/jobs/build_materialized_hfsa_fixedtarget_depth50_2026-05-19.slurm
```

## Length Audit

A generated fixed-target sample with the local OLMo tokenizer has the following approximate lengths:

| template | depth | prompt tokens | target tokens | total SFT tokens |
| --- | ---: | ---: | ---: | ---: |
| `logic` | 25 | 2,927 | 1,915 | 4,843 |
| `logic` | 50 | 5,838 | 3,509 | 9,348 |
| `nl_exact` | 25 | 2,927 | 2,766 | 5,694 |
| `nl_exact` | 50 | 5,838 | 5,489 | 11,328 |

Training only uses depths up to 25, so the existing SFT `data.max_length=8192` is sufficient for the primary train sweep. Depth-50 post-hoc eval uses vLLM with `max_new_tokens=6144`. OLMo-3-1025-7B config reports `max_position_embeddings=65536`, so the depth-50 prompt+generation budget is within model context.

## Main 3-Seed Sweep

Script:

```bash
scripts/slurm/sweeps/sft/hfsa_depth_scaling_logic_vs_nl_2026-05-19.slurm
```

Grid:

| axis | values |
| --- | --- |
| trace template | `logic`, `nl_exact` |
| train depth range | `1..5`, `1..10`, `1..15`, `1..20`, `1..25` |
| seed | `3407`, `3408`, `3409` |
| SFT steps | 10,000 |
| effective batch | 1 sequence/update |
| model | `allenai/Olmo-3-1025-7B` + LoRA |
| checkpoint retention in patched script | save every 1000 steps, keep up to 12 checkpoints |

Total: `2 x 5 x 3 = 30` SFT runs.

W&B groups:

```bash
sft_hfsa_depth_scaling/<template>_train1to<max_depth>
```

## Post-Hoc Evaluation

Script:

```bash
scripts/slurm/jobs/posthoc_hfsa_depth_scaling_merge_eval_2026-05-19.slurm
```

Eval protocol:

| setting | value |
| --- | ---: |
| eval depths | `1..50` |
| prompts per depth | 128 |
| sampled generations per prompt | 16 |
| pass@k | `1,2,4,8,16` |
| max new tokens | 6144 |
| constrained line-level eval | disabled by default |

The eval job merges each LoRA checkpoint into a temporary standalone HF model, runs pass@k, writes JSON/JSONL outputs, logs to W&B, and deletes the merged model by default to avoid accumulating hundreds of GB in `tmp/`.

Outputs:

```bash
${WORK}/synthetic-RLVL/passk_eval/hfsa_depth_scaling/
```

Intermediate-checkpoint eval script:

```bash
scripts/slurm/jobs/posthoc_hfsa_depth_scaling_intermediate_eval_2026-05-19.slurm
```

Default checkpoint steps:

```bash
CHECKPOINT_STEPS=1000,3000,10000
```

The intermediate eval job merges one LoRA checkpoint at a time into a job-local tmp directory, evaluates it, writes JSON/JSONL outputs under:

```bash
${WORK}/synthetic-RLVL/passk_eval/hfsa_depth_scaling_intermediate/
```

and deletes the merged model immediately after each checkpoint.

Primary metrics:

- `correct_pass@k`
- `citation_free_valid_pass@k` for logic traces
- `nl_logic_citation_free_valid_pass@k` for NL traces translated back to FOL
- joint correct+valid pass@k
- `valid_given_correct@k`
- depth threshold curves: largest depth with metric above 80%, 50%, and 25%
- area under the depth curve over `1..50`

## Runtime Expectation

Prior 10k OLMo-7B LoRA SFT jobs on the fixed-target dataset took roughly 13-15 hours on one A100-80GB for train depths `1..10` and `1..15`. Depth `1..20` and `1..25` should still usually fit the 24h allocation because the max training sequence length remains below 8192 tokens, but they are expected to be slower.

Depth-50 pass@k eval is the higher-risk runtime component because it evaluates `50 x 128 x 16 = 102,400` generations per checkpoint with longer outputs. Eval jobs are therefore separate dependent jobs, and the scripts are idempotent: they skip existing JSON outputs unless `FORCE_PASSK_EVAL=1` is set.

## Follow-Up Plan

Do not immediately run the full factorial grid. The staged plan is:

1. Main 3-seed LoRA depth-scaling sweep above.
2. Batch-size ablation on representative train depths, not all depths.
3. Shortcut-rich SFT ablation with `shortcut_rate in {0.0, 0.5, 0.8}` after the shortcut-neutral trend is confirmed.
4. Full/non-LoRA or smaller-model pretraining experiments to check whether the LoRA result is an adapter-specific artifact.
5. Later dataset-family expansion: iGSM-style arithmetic graphs, explicit graph/maze traversal, and constraint-satisfaction tasks.

## Non-LoRA / Pretraining Follow-Up

The current sweep is LoRA SFT on OLMo-7B. If the formal-CoT scaling trend holds, the next stronger test is to train smaller models from scratch or with full-parameter continuation rather than adapters.

Recommended first non-LoRA wave:

| model size | training mode | purpose |
| ---: | --- | --- |
| 50-200M | from-scratch pretraining on HFSA traces | cheapest clean test of whether formal traces induce an algorithmic prior |
| 300M-1B | from-scratch or continued pretraining | check scaling of the same effect |
| 7B | full fine-tune / non-LoRA continuation if feasible | rule out LoRA bottlenecks on the same base model |

The smaller-model setting should use the same depth curriculum and eval protocol, but with many more tokens and explicit train/eval loss curves because full pretraining is a different regime from 10k-step adapter SFT.

## Submitted Jobs

Originally submitted on 2026-05-19, then cancelled after the checkpoint-retention patch:

| stage | Slurm job | dependency | notes |
| --- | ---: | --- | --- |
| build + push depth-50 dataset | `3623863` | none | builds local parquet and pushes to HF |
| 30-row SFT array | `3623864_[0-29%6]` | `afterok:3623863` | cancelled before start; old Slurm snapshot likely retained only 2 checkpoints |
| 30-row merge/pass@k eval array | `3623865_[0-29%6]` | `aftercorr:3623864` | cancelled and replaced |

Resubmitted from patched scripts on 2026-05-19 20:20 CEST:

| stage | Slurm job | dependency | notes |
| --- | ---: | --- | --- |
| 30-row SFT array | `3634790_[0-29%6]` | none; dataset already built | LoRA SFT, 10k steps, batch 1, `save_total_limit=12` |
| 30-row final merge/pass@k eval | `3634791_[0-29%6]` | `aftercorr:3634790` | per-row final eval; transient merged checkpoints |
| 30-row intermediate checkpoint eval | `3634792_[0-29%2]` | `aftercorr:3634790` | selected checkpoint pass@k, default `1000,3000,10000` |

Check status:

```bash
squeue -j 3634790,3634791,3634792,3634793,3634794,3643001,3643002 -o '%.18i %.9P %.32j %.8T %.10M %.6D %R'
sacct -j 3634790,3634791,3634792,3634793,3634794,3643001,3643002 --format=JobIDRaw,JobName%34,State,Elapsed,ExitCode -n -P
```

## Current Status - 2026-05-20 09:25 CEST

| stage | status | note |
| --- | --- | --- |
| main SFT `3634790_0` | running | `logic_train1to5_10k_seed3407`; reached the `3000/10000` online eval point in logs and continued |
| main SFT `3634790_[1-29%6]` | pending | pending by priority/resources |
| final eval `3634791_[0-29%6]` | dependency-pending | waits on corresponding main SFT rows |
| intermediate eval `3634792_[0-29%2]` | dependency-pending | waits on corresponding main SFT rows |
| 1k sanity SFT `3634793` | 9/10 completed, 1 failed | `nl_exact_train1to25_1k_seed3407` failed with CUDA OOM at step 20 |
| 1k sanity SFT retry `3643001_[9]` | pending | retries only the failed row with gradient checkpointing enabled |
| 1k sanity eval `3634794_0..8` | pending | existing eval rows for completed 1k SFT checkpoints |
| 1k sanity eval retry `3643002_[9]` | dependency-pending | waits on `3643001` |

OOM mitigation added after the 1k `nl_exact_train1to25` failure:

- `train_sft.py` supports `train.gradient_checkpointing`.
- `conf/sft_hard_fsa_schema_fixedtarget.yaml` defaults `train.gradient_checkpointing=auto`, which enables checkpointing for long `nl_exact` depth-25 rows.
- HFSA depth-scaling SFT Slurm scripts pass `train.gradient_checkpointing="${GRADIENT_CHECKPOINTING:-auto}"` for future submissions.
- `scripts/env.sh` sets `PYTORCH_ALLOC_CONF=expandable_segments:True`.

This should also protect pending long `nl_exact_train1to25` 10k SFT rows when they start, because they load the current config/code at runtime.

## Current Status - 2026-05-19 20:20 CEST

| stage | status | note |
| --- | --- | --- |
| build `3623863` | completed | dataset was built/pushed successfully |
| old main SFT/eval `3623864`, `3623865` | cancelled | replaced so patched checkpoint retention is used |
| main SFT `3634790_[0-29%6]` | pending | pending by priority/resources |
| final eval `3634791_[0-29%6]` | dependency-pending | waits on corresponding SFT rows |
| intermediate eval `3634792_[0-29%2]` | dependency-pending | waits on corresponding SFT rows |
| old 1k sanity SFT/eval `3624535`, `3624536` | cancelled | replaced so pending tasks/evals use patched scripts |
| 1k sanity SFT `3634793_[0-9%5]` | pending | idempotent; completed rows should skip if finals exist |
| 1k sanity eval `3634794_[0-9%5]` | dependency-pending | waits on corresponding 1k SFT rows |

No failure requiring resubmission was observed in these jobs. The bottleneck is cluster/node availability.

## Checkpoint And Convergence-Curve Caveat

The originally submitted main SFT array is aligned with the final depth-scaling comparison, but may not be fully aligned with the new convergence-curve requirement because Slurm snapshots job scripts at submission time.

Original submitted behavior:

- SFT runs save every `1000` steps through `train.save_steps=1000`.
- The submitted script likely used `train.save_total_limit=2`.
- Therefore a 10k run will likely retain only the last two checkpoints plus `final`.
- Online validation runs every `1000` steps, but only with `validation.samples_per_step=4` and sampled pass@k disabled.

Implication:

- Current jobs are enough for final correct@1/correct@16 depth-scaling curves.
- Original submitted jobs are not enough for robust correct@1/correct@16 over training progress.

Patched resubmission behavior:

```bash
train.save_steps=1000
train.save_total_limit=12
```

This keeps all ten 1000-step LoRA trainer checkpoints plus margin. Based on the observed LoRA checkpoint size of about 468 MB, retaining all 10 checkpoints for all 30 rows costs roughly 140 GB for checkpoint dirs. That is acceptable only because merged full checkpoints remain transient and are deleted after eval.

Post-hoc pass@k should still evaluate only selected checkpoints first, e.g. `checkpoint-1000`, `checkpoint-3000`, and `checkpoint-10000`, not every checkpoint for every row.

Observed artifact sizes:

| artifact/directory | approximate size |
| --- | ---: |
| LoRA SFT final adapter | 162 MB |
| LoRA SFT trainer checkpoint | 468 MB |
| merged OLMo-7B checkpoint | 14 GB |
| `${WORK}/synthetic-RLVL/tmp` | 2.0 TB |
| `${WORK}/synthetic-RLVL/runs` | 4.0 TB |

Eval jobs should keep merged checkpoints transient: merge, evaluate, log metrics, delete merged model.

Important operational note:

- Pending/running jobs submitted before this patch will not automatically use `save_total_limit=12`.
- To get convergence checkpoints, cancel/resubmit the SFT array and submit dependent final/intermediate eval arrays from the patched scripts.

## 1k-Sample Sanity Sweep

Motivation: Harada et al. (2025) report that 1k-sample SFT can be competitive with larger SFT sets in their controlled 7B-scale study, and that 20k samples had no consistent accuracy advantage over 1k in their sample-size comparison. Since our current SFT setup has effective batch size 1, a `1k samples` sanity check maps cleanly to `data.train_samples=1000` and `train.max_steps=1000`.

Reference: `https://arxiv.org/abs/2506.14681v2`.

Script:

```bash
scripts/slurm/sweeps/sft/hfsa_depth_scaling_1k_sanity_2026-05-19.slurm
```

Grid:

| axis | values |
| --- | --- |
| trace template | `logic`, `nl_exact` |
| train depth range | `1..5`, `1..10`, `1..15`, `1..20`, `1..25` |
| seed | `3407` only |
| SFT steps | 1,000 |
| train samples loaded | 1,000 |
| effective batch | 1 sequence/update |

Total: `2 x 5 x 1 = 10` SFT runs.

Post-hoc eval script:

```bash
scripts/slurm/jobs/posthoc_hfsa_depth_scaling_1k_merge_eval_2026-05-19.slurm
```

Eval is the same depth-50 pass@k protocol except `PASSK_SAMPLES_PER_STEP=64` by default for faster turnaround. This is intended as an early sanity readout, not the final 3-seed result.

Originally submitted 1k sanity jobs on 2026-05-19, then cancelled/replaced:

| stage | Slurm job | dependency |
| --- | ---: | --- |
| old 1k-sample SFT sanity array | `3624535_[0-9%5]` | cancelled |
| old 1k-sample merge/pass@k eval | `3624536_[0-9%5]` | cancelled |
| patched 1k-sample SFT sanity array | `3634793_[0-9%5]` | none |
| patched 1k-sample merge/pass@k eval | `3634794_[0-9%5]` | `aftercorr:3634793` |
