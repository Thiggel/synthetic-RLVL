# Hard-FSA-Schema Easy Curriculum (2026-05-13)

## Motivation

The previous hard-FSA-schema setting was too hard to interpret as a validity-reward intervention test. Train/ID `correct@1` stayed low, so OOD failure did not cleanly distinguish shortcut learning from basic task non-learnability.

This curriculum keeps the same formal finite-state proof substrate but makes the first learnability test easier:

- branch factor reduced from `K=4` to `K=2`;
- SFT trains on depths `1..5`, not only `1..3`;
- GRPO trains on depths `1..5`;
- validation remains shortcut-neutral and spans depths `1..20`;
- prompt/proof length is much shorter: depth 5 is about 22 premises, 12 proof lines, and roughly 640 OLMo tokens; depth 20 is roughly 82 premises, 42 proof lines, and roughly 2.2k tokens.

The immediate target is ID saturation. We should only interpret shortcut/OOD effects after ID `correct@1` is high, ideally above 80%.

## Dataset

HF dataset: `flaitenberger/LogicalReasoning-hard-fsa-schema-easy`

Generated subsets:

- `train_schema_easy0p0_up_to_5_50k`: no shortcut, depths `1..5`, 50k rows.
- `train_schema_easy0p5_up_to_5_50k`: shortcut-rate `0.5`, depths `1..5`, 50k rows.
- `val_step_01_1k` through `val_step_20_1k`: shortcut-neutral eval, fixed depth per split, 1k rows each.

Local materialization root:

- `${WORK}/synthetic-RLVL/datasets/materialized_logic_hard_fsa_schema_easy_20260513`

The generator still uses normal state/marker words mapped onto one-letter predicates and constants for the parser.

## Code Changes

- `synthetic_dataset.py`: `hard_fsa_schema` now supports `branching_factor` from 2 to 4. K=4 behavior is preserved.
- `synthrlvl/datasets/materialize.py`: CLI now exposes `--train-up-to-5-subset` and `--train-up-to-10-subset`.
- `scripts/analysis/probe_hard_fsa_schema.py`: chance-level diagnostics now use `1 / branching_factor`, not hard-coded 0.25.
- `tests/test_synthetic_dataset.py`: added K=2 schema curriculum regression test.

Validation run locally:

- `pytest` targeted schema tests: passed.
- K=2 schema probe with `shortcut_rate=0.5`: accepted.
- materialization smoke: passed.
- Slurm script syntax checks: passed.

## Training Plan

SFT:

- model: `allenai/Olmo-3-1025-7B`
- LoRA: r=16, alpha=32
- train subset: `train_schema_easy0p0_up_to_5_50k`
- train depths: `1..5`
- steps: `3000`
- seeds: `3407`, `3408`, `3409`

GRPO rows:

- train depths: `1..5`
- eval depths: `1..20`
- num prompts / rollouts: `8 x 8`
- steps: `500`
- seeds: `3407`, `3408`, `3409`

Ablation axes:

- train shortcut `0.0` vs `0.5`;
- reward `correct_plus_0p1_format` vs `indicator_correct_and_citation_free_valid_plus_0p1_format`.

Total GRPO rows: `2 shortcut settings x 2 rewards x 3 seeds = 12`.

## Submitted Jobs

Submitted on 2026-05-13:

- dataset build + HF push: `3605313`
- SFT array: `3605314`
- SFT merge + sanity array: `3605315`
- GRPO array: `3605316`
- final actor merge + pass@k eval array: `3605317`

Dependency chain:

```text
3605313 -> 3605314 -> 3605315 -> 3605316 -> 3605317
```

Pass@k outputs will be written under:

- `${WORK}/synthetic-RLVL/passk_eval/hard_fsa_schema_easy500/`

W&B grouping:

- SFT: `sft_hard_fsa_schema_easy/lr1e-4_k2_depth1to5_50k_shortcut0p0`
- GRPO: `posttrain_hard_fsa_schema_easy500/train0p0/<reward>`
- GRPO: `posttrain_hard_fsa_schema_easy500/train0p5/<reward>`
