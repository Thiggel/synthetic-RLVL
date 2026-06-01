# Materialized Synthetic Dataset

This repo now supports using prebuilt parquet subsets instead of on-the-fly sample generation.

## Subsets

Training:
- `train_up_to_5_1m`: 1,000,000 rows mixed over depths `1..5`.
- `train_up_to_10_1m`: 1,000,000 rows mixed over depths `1..10`.

Validation:
- `val_step_01_1k` ... `val_step_20_1k`: each contains 1,000 rows at fixed depth.

## Build Locally

```bash
source ./scripts/env.sh
python -m synthrlvl.datasets.materialize \
  --output-root "$WORK/synthetic-RLVL/datasets/materialized_logic"
```

## Optional Push To Hugging Face

```bash
source ./scripts/env.sh
python -m synthrlvl.datasets.materialize \
  --output-root "$WORK/synthetic-RLVL/datasets/materialized_logic" \
  --push-to-hub \
  --hf-repo-id "<org-or-user>/<dataset-name>"
```

## Train With Materialized Data

SFT:
```bash
python train_sft.py \
  data.source=materialized \
  data.materialized.local_root="$WORK/synthetic-RLVL/datasets/materialized_logic"
```

GRPO:
```bash
python posttrain_grpo_verl.py \
  data.source=materialized \
  data.materialized.local_root="$WORK/synthetic-RLVL/datasets/materialized_logic"
```

You can also use HF-hosted subsets with:
- `data.materialized.dataset_id=<repo_id>`
- `data.materialized.local_root=null`

## Hard-v5 Shortcut Dataset Refresh (2026-05-08)

`hard_v5` is now the active shortcut-stress dataset variant. It keeps the same public difficulty name but changes the generator in place:

- Each example samples a randomized true state path, so depth no longer identifies the answer.
- Each example also contains an equally long dormant shortcut path with a coherent but wrong final answer.
- The shortcut path is invalid because the initial `dormant(a)` fact is never given; the true path starts from `active(a)`.
- Training uses `train_shortcut_rate=0.8`, so the true branch is often first and can create positional shortcut pressure.
- Validation uses `val_shortcut_rate=0.0`, so the shortcut branch is first and shortcut-following should fail.
- Gold traces remain citation-free formal proofs, so citation bookkeeping is not the validity bottleneck.

HF dataset target for the refreshed materialization remains:

```bash
flaitenberger/LogicalReasoning-hard-v5
```

Current intended subsets:

- `train_up_to_3_10k`: short-chain SFT on steps `1..3`.
- `train_up_to_15_50k`: fast GRPO on steps `1..15`.
- `val_step_01_1k` ... `val_step_20_1k`: fixed-depth validation, with `16..20` as OOD tail.

Build/push command used by Slurm:

```bash
PUSH_TO_HUB=1 HF_REPO_ID=flaitenberger/LogicalReasoning-hard-v5 \
  sbatch scripts/slurm/jobs/build_materialized_hard_v5_shortcut_2026-05-08.slurm
```

## Hard-FSA Materialization (2026-05-09)

`hard_fsa` is the replacement dataset variant for the next shortcut-vs-validity experiment.

Design summary:

- The prompt defines a compact finite-state automaton with four coherent candidate branches per step.
- All branches share the same initial visible state, but only the gold branch has the derivable initial marker.
- Wrong branches remain coherent if followed, so invalid shortcut reasoning produces plausible but wrong trajectories instead of obvious dead ends.
- Branch order is shuffled per step and validation contains no shortcut-position cue.
- Gold proofs are citation-free, enabling citation-free validity rewards and eval.

HF dataset target:

```bash
flaitenberger/LogicalReasoning-hard-fsa
```

Intended subsets:

- `train_up_to_3_10k`: short SFT on steps `1..3`.
- `train_up_to_15_50k`: GRPO on steps `1..15`.
- `val_step_01_1k` ... `val_step_20_1k`: fixed-depth validation with `16..20` as OOD tail.

Build/push Slurm command:

```bash
PUSH_TO_HUB=1 HF_REPO_ID=flaitenberger/LogicalReasoning-hard-fsa HF_PRIVATE=1 \
  sbatch scripts/slurm/jobs/build_materialized_hard_fsa_2026-05-09.slurm
```

### Strict Hard-FSA Invariants (2026-05-09 update)

The submitted hard-fsa materialization now enforces stricter invariants:

- no repeated `(state, marker)` automaton pair,
- no branch re-entry through the same automaton pair,
- no same-layer state collision,
- no duplicate implication antecedents,
- unique branch final states,
- no reused `(state, constant)` output atom when constants wrap at long depths.

The current strict Slurm build writes to:

```bash
${WORK}/synthetic-RLVL/datasets/materialized_logic_hard_fsa_strict_20260509
```

and pushes to the same HF target:

```bash
flaitenberger/LogicalReasoning-hard-fsa
```

## Hard-FSA Shortcut-Schema Dataset (2026-05-09)

`hard_fsa_schema` adds a train-only shortcut channel on top of the FSA task.

Training split:

- `shortcut_rate=0.8`.
- Shortcut-enabled examples use marker redundancy plus a shared family-level transition schema.
- This creates realistic shallow cues: the model can often ignore explicit markers or learn a reusable transition schema.

Validation split:

- `shortcut_rate=0.0`.
- Uses strict exchangeable FSA generation with no train schema signal.
- Candidate answer position is balanced and simple candidate heuristics are at chance.

HF target:

```bash
flaitenberger/LogicalReasoning-hard-fsa-schema
```

Subsets:

- `train_schema_0p8_up_to_3_10k`
- `train_schema_0p8_up_to_15_50k`
- `val_step_01_1k` ... `val_step_20_1k`

Build script:

```bash
PUSH_TO_HUB=1 HF_REPO_ID=flaitenberger/LogicalReasoning-hard-fsa-schema HF_PRIVATE=1 \
  sbatch scripts/slurm/jobs/build_materialized_hard_fsa_schema_2026-05-09.slurm
```

## Hard-FSA-Schema Easy Curriculum (2026-05-13)

HF dataset: `flaitenberger/LogicalReasoning-hard-fsa-schema-easy`

Purpose: easier learnability-first version of `hard_fsa_schema` before reintroducing harder branching/depth.

Subsets:

- `train_schema_easy0p0_up_to_5_50k`: K=2, shortcut-rate 0.0, train depths 1..5, 50k rows.
- `train_schema_easy0p5_up_to_5_50k`: K=2, shortcut-rate 0.5, train depths 1..5, 50k rows.
- `val_step_01_1k` ... `val_step_20_1k`: K=2, shortcut-rate 0.0, fixed-depth validation.

Local root:

- `${WORK}/synthetic-RLVL/datasets/materialized_logic_hard_fsa_schema_easy_20260513`

## Hard-FSA-Schema Fixed-Target Dataset (2026-05-14)

HF dataset: `flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget`

Purpose: rerun HFSA validity experiments after fixing the gold target. The proof conclusion now matches the queried final state atom, not the final marker atom.

Subsets:

- `train_fixedtarget_up_to_3_50k`: K=4, no shortcut, train depths `1..3`.
- `train_fixedtarget_up_to_5_50k`: K=4, no shortcut, train depths `1..5`.
- `train_fixedtarget_up_to_10_50k`: K=4, no shortcut, train depths `1..10`.
- `train_fixedtarget_up_to_15_50k`: K=4, no shortcut, train depths `1..15`.
- `val_step_01_1k` ... `val_step_25_1k`: K=4, no shortcut, fixed-depth validation.

Build script:

```bash
sbatch scripts/slurm/jobs/build_materialized_hard_fsa_schema_fixedtarget_2026-05-14.slurm
```

## Hard-FSA-Schema Fixed-Target Depth-50 Dataset (2026-05-19)

HF dataset: `flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget-depth50`

Purpose: extend the pure-SFT logic-vs-natural-language CoT scaling experiment to 3 seeds, train ranges through depth 25, and post-hoc pass@k evaluation through depth 50.

Subsets:

- `train_fixedtarget_up_to_5_50k`: K=4, no shortcut, train depths `1..5`.
- `train_fixedtarget_up_to_10_50k`: K=4, no shortcut, train depths `1..10`.
- `train_fixedtarget_up_to_15_50k`: K=4, no shortcut, train depths `1..15`.
- `train_fixedtarget_up_to_20_50k`: K=4, no shortcut, train depths `1..20`.
- `train_fixedtarget_up_to_25_50k`: K=4, no shortcut, train depths `1..25`.
- `val_step_01_1k` ... `val_step_50_1k`: K=4, no shortcut, fixed-depth validation/evaluation.

Build script:

```bash
sbatch scripts/slurm/jobs/build_materialized_hfsa_fixedtarget_depth50_2026-05-19.slurm
```

Experiment plan:

- `docs/hfsa_depth_scaling_plan_2026-05-19.md`

## Hard-FSA-Schema Shortcut-Kind Depth-50 Datasets (2026-05-29)

Local roots submitted for build:

- `${WORK}/synthetic-RLVL/datasets/materialized_hfsa_fixedtarget_depth50_shortcut_position_0p5_20260529`
- `${WORK}/synthetic-RLVL/datasets/materialized_hfsa_fixedtarget_depth50_shortcut_position_0p8_20260529`
- `${WORK}/synthetic-RLVL/datasets/materialized_hfsa_fixedtarget_depth50_shortcut_initial_marker_0p5_20260529`
- `${WORK}/synthetic-RLVL/datasets/materialized_hfsa_fixedtarget_depth50_shortcut_initial_marker_0p8_20260529`

Submitted chain/status:

- build `3674886_[0-3%2]`: completed exit `0:0` on 2026-05-29; all four local roots exist with the expected train plus depth-1..50 validation subsets.
- SFT `3674887_[0-23%3]`: original rows `0..21` and `23` complete, row `22` failed, and replacement SFT `3682458_22` completed.
- eval `3674888_[0-23%4]`: running as of 2026-06-01 06:35 CEST; rows `0..20` are complete and wrote `21/24` pass@k JSONs plus sample JSONLs, and rows `21..23` are running.

Purpose: repeat the shortcut robustness ablation with two concrete shortcut mechanisms beyond the existing schema shortcut. `position` makes the gold branch first on shortcut-enabled training examples. `initial_marker` fixes the gold path's initial marker to `north` on shortcut-enabled training examples. Evaluation remains shortcut-neutral with `val_shortcut_rate=0.0` for every row.

Generated subsets:

- `train_fixedtarget_up_to_25_50k`: K=4, train depths `1..25`, shortcut rate `0.5` or `0.8`.
- `val_step_01_1k` ... `val_step_50_1k`: K=4, shortcut-neutral fixed-depth validation/evaluation.

Implementation details:

- `synthrlvl.datasets.materialize` now accepts `--shortcut-kind {schema,position,initial_marker}`.
- `DatasetConfig.shortcut_kind` and `TaskConfig.shortcut_kind` propagate the setting into both materialization and on-the-fly eval.
- `scripts/analysis/probe_hard_fsa_schema.py` now probes the expected shortcut for each kind. Local probes and tiny materialization smokes passed for `position` and `initial_marker` before submission.

## Paired Synthetic Dataset Families (2026-05-20)

Initial paired natural-language / formal-logic dataset families are implemented for follow-up experiments:

- `official_igsm`: exact official `facebookresearch/iGSM` sampling with an intended 1:1 logic trace over the official modulo-23 arithmetic solution; subtraction-substitution validation was fixed locally on 2026-05-28, the seed-3407 train-10 materialization/SFT/eval chain completed, and the full paired suite is running.
- `maze_navigation`: keyed/constrained graph traversal with room reachability, held-key state, blocked decoy doors, and unreachable treasure decoys.
- `attribute_constraints`: multi-input slot-value constraint propagation; no candidate-assignment objects and no precomputed Mastermind feedback are provided.
- Backward-compatible aliases: `igsm_arithmetic`, `graph_traversal`, `mastermind_constraints`, `constraint_satisfaction`, `constraint_propagation`.

### Train-10 Follow-Up Materializations

Submitted 2026-05-24:

| family | job | status | local root |
| --- | ---: | --- | --- |
| `attribute_constraints` original | `3656210_1` | completed but saturated in seed-3407 eval | `${WORK}/synthetic-RLVL/datasets/materialized_paired_attribute_constraints_train10_20260524` |
| `maze_navigation` | `3656308_0` | completed after depth-15 room-vocabulary fix | `${WORK}/synthetic-RLVL/datasets/materialized_paired_maze_navigation_train10_20260524` |

Submitted 2026-05-25 after saturation:

| family | job | status | local root |
| --- | ---: | --- | --- |
| `attribute_constraints` hard | `3659338` | completed at 09:33 CEST; downstream SFT/eval jobs are `3659339`/`3659340` | `${WORK}/synthetic-RLVL/datasets/materialized_paired_attribute_constraints_hard_train10_20260525` |

The hard attribute generator maps requested depth to `floor(depth/2)+2` compact slots, uses compact `sN`/`vN` symbols, recent-window dependency DAGs, and adversarial decoys. Local smoke materialization with full validation through depth 50 passed before the Slurm build was submitted.

### Full Paired Follow-Up Materializations

Submitted 2026-05-28 as the broad repeat suite:

| family | build job | local root | train subsets |
| --- | ---: | --- | --- |
| `official_igsm` | `3672195_0` | `${WORK}/synthetic-RLVL/datasets/materialized_paired_official_igsm_full_20260528` | `train_official_igsm_up_to_{5,10,15,20,25}_50k` |
| `maze_navigation` | `3672195_1` | `${WORK}/synthetic-RLVL/datasets/materialized_paired_maze_navigation_full_20260528` | `train_maze_navigation_up_to_{5,10,15,20,25}_50k` |
| hard `attribute_constraints` | `3672195_2` | `${WORK}/synthetic-RLVL/datasets/materialized_paired_attribute_constraints_hard_full_20260528` | `train_attribute_constraints_hard_up_to_{5,10,15,20,25}_50k` |

Each build also writes `val_step_01_1k` through `val_step_50_1k` and validates every generated row with `--validate-examples -1`. Dependent SFT and sparse pass@k eval arrays were initially `3672212_[0-89%6]` and `3672213_[0-89%4]`, covering both `logic` and `nl_exact` with seeds `3407`, `3408`, and `3409`. Initial pending arrays `3672196`/`3672197` were canceled before start after fixing excessive startup staggering. As of 2026-06-01 06:49 CEST, all three build rows are complete; original SFT rows `0..54` and `58` are complete, row `56` was canceled after idle-GPU diagnosis and replaced by `3683070_56`, and interrupted rows `55/57/59-89` were covered by replacement SFT `3682411_[55,57,59-89%6]`. Replacement SFT and row-56 replacement are now complete, for `90/90` final SFT adapters. Stale eval `3672213` was canceled and replacement eval `3682449_[0-89%4]` has released; rows `0..29` are complete and wrote all `30/30` `official_igsm` pass@k JSONs plus sample JSONLs, rows `30/31/32/33` are running, rows `34..89` are throttle-pending, and `maze_navigation` plus hard `attribute_constraints` have no completed eval JSONs yet. A 2026-05-31 materialized/gold-target audit over sampled train-depth-25 and val-depth-50 rows re-confirmed matched logic/NL prompts, expected wrappers, final answer tags, strict and grounded logic proof validation, and correct/formatted NL targets; sampled paired NL gold traces still have zero NL-to-logic parse/translated validity, so paired NL validity metrics should not be used for claims until translator coverage is improved. A 2026-06-01 sample check over completed `official_igsm` eval JSONLs found the expected logic `<formal>` and NL `<think>` wrappers and answer extraction; shallow logic can be citation-free or internally valid, but grounded iGSM validity remains unreliable beyond trivial retrieval, and deeper logic/NL generations show answer or validity fragility. Full paired conclusions remain deferred until `3682449` completes.

The generator code lives in:

```bash
synthrlvl/datasets/paired_synthetic.py
```

Detailed construction rationale and exact generated example sequences:

```bash
docs/paired_synthetic_benchmarks_2026-05-20.md
```

The materializer writes the same `LogicExample`-compatible parquet schema as the existing synthetic datasets:

```bash
source ./scripts/env.sh
python scripts/data/build_paired_synthetic_dataset.py \
  --kind maze_navigation \
  --output-root "$WORK/synthetic-RLVL/datasets/materialized_maze_navigation" \
  --train-rows 50000 \
  --train-max-depth 10 \
  --val-rows-per-depth 128 \
  --val-max-depth 50
```

Each generated example is optionally validated with `LogicEngine` during materialization. Use:

```bash
--validate-examples -1
```

to validate every row, or:

```bash
--validate-examples 0
```

to disable validation for large production builds after a smoke test.

These paired families are implemented for substrate-transfer experiments alongside the active HFSA depth-scaling wave:

- `official_igsm`: samples from the local official `facebookresearch/iGSM` generator and converts the official arithmetic solution into a logic trace. Arithmetic is modulo 23, matching iGSM, via the `MOD23` proof rule. The 2026-05-23 subtraction-substitution blocker is fixed locally as of 2026-05-28; the seed-3407 train-10 chain `3671601`/`3671602`/`3671603` completed, and the full three-family suite has `90/90` SFT final adapters with replacement eval `3682449` running. As of 2026-06-01 06:49 CEST, `official_igsm` eval rows `0..29` are complete.
- `igsm_arithmetic`: compact backward-compatible register arithmetic used for smoke tests and older scripts.
- `maze_navigation`: a world-grounded keyed graph traversal task where the model proves which treasure room is reachable after exactly `N` moves while holding the required key for each traversed door.
- `graph_traversal`: backward-compatible alias for `maze_navigation`.
- `attribute_constraints`: a finite-domain constraint-propagation task where the model proves the value of each slot from two initial slot values and multi-input joint constraints over previous slots.
- `mastermind_constraints` / `constraint_satisfaction` / `constraint_propagation`: backward-compatible aliases for `attribute_constraints`.
- Logic evaluation reports both internal validity and grounded validity. Grounded validity validates generated proof lines against the gold canonical premises/conclusion instead of trusting generated premises.

For on-the-fly evaluation through `TaskBuilder` / `scripts/evaluate_checkpoint_passk.py`, the paired families can be selected with:

```bash
task.difficulty=official_igsm
task.difficulty=igsm_arithmetic
task.difficulty=maze_navigation
task.difficulty=graph_traversal
task.difficulty=attribute_constraints
task.difficulty=mastermind_constraints
task.difficulty=constraint_satisfaction
task.difficulty=constraint_propagation
```

For SFT from materialized parquet, set `data.materialized.local_root` and the explicit `data.materialized.train_subset` produced by the materializer.

### Paired Dataset Audit - 2026-05-23

The broader local materialization audit used `--validate-examples -1` on small depth-12 builds:

| family | status | artifact |
| --- | --- | --- |
| `maze_navigation` | fixed and passed; key vocabulary now extends with deterministic `key_XX` names when depth exceeds the color list | `analysis/paired_dataset_audit_2026-05-23/maze_navigation/` |
| `attribute_constraints` | passed | `analysis/paired_dataset_audit_2026-05-23/attribute_constraints/` |
| `official_igsm` | fixed locally 2026-05-28; depth-50 smoke validation passes after parser tokenization fix | old failing log: `analysis/paired_dataset_audit_2026-05-23/logs/official_igsm.log` |

### Paired Train-10 Pilot Materialization - 2026-05-24

Submitted Slurm job `3656210_[0-1%2]` to materialize the two audited paired families for the first transfer pilot. Row `1` (`attribute_constraints`) completed. Row `0` (`maze_navigation`) failed at depth 15 because the room-name word bank was finite; the generator was patched to extend room names deterministically and row `0` was resubmitted as `3656308_0`, which completed with every generated row validated.

| row | family | output root | train subset |
| ---: | --- | --- | --- |
| `0` | `maze_navigation` | `${WORK}/synthetic-RLVL/datasets/materialized_paired_maze_navigation_train10_20260524` | `train_maze_navigation_up_to_10_50k` |
| `1` | `attribute_constraints` | `${WORK}/synthetic-RLVL/datasets/materialized_paired_attribute_constraints_train10_20260524` | `train_attribute_constraints_up_to_10_50k` |

The materialization jobs write `val_step_01_1k` through `val_step_50_1k` and validate every generated row by default. The original dependent SFT/eval jobs were `3656211` and `3656213`.

After the maze failure, the original dependent jobs `3656211` and `3656213` were canceled. Replacement jobs are:

| stage | job | dependency |
| --- | ---: | --- |
| paired train-10 SFT pilot | `3656309_[0-3%2]` | `afterok:3656308` |
| paired train-10 sparse eval | `3656310_[0-3%2]` | `aftercorr:3656309` |

Oversight update 2026-05-24 15:06 CEST: paired maze SFT rows `3656309_0,1` failed with CUDA OOM at `data.max_length=8192` while gradient checkpointing was off. The dead eval rows `3656310_0,1` were canceled. Replacement maze SFT rows `3657088_0,1` are running with `GRADIENT_CHECKPOINTING=true` and had cleared the original OOM window by 15:05 CEST; replacement eval rows `3657089_0,1` depend on `aftercorr:3657088`. Attribute rows remain on the original `3656309`/`3656310` chain.

Oversight update 2026-05-25 02:44 CEST: attribute rows completed SFT and sparse eval on the original replacement chain. Maze rows completed SFT on retry `3657738_0,1` after disabling default online generation eval; sparse eval rows `3657739_0,1` are running.

Oversight update 2026-05-25 10:48 CEST: maze sparse eval row `3657739_0` failed because a depth-45 prompt had `16400` tokens under the old `vllm_max_model_len=16384`; row `3657739_1` was canceled before the same expected failure. The paired eval wrapper now uses a 32k vLLM context and batch `64` for `maze_navigation`; replacement `3659556_[0-1%2]` is running.
