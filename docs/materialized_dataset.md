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
