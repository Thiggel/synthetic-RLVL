# Hard-FSA-Schema Fixed-Target Runs (2026-05-14)

## Why This Exists

The 2026-05-14 diagnostics found a target mismatch in `hard_fsa` / `hard_fsa_schema` gold traces:

- The question asks: `Which state applies to <constant>?`
- The `<answer>` is the final state word.
- The previous gold proof continued one extra line and concluded the final marker atom.

That meant citation-free validity could be awarded for proving a marker, not for proving the answer proposition. This made validity-reward comparisons ambiguous.

## Fix

`synthetic_dataset.py` now makes `hard_fsa` and shortcut-enabled `hard_fsa_schema` stop the gold proof at the final state atom:

- intermediate states still derive markers because later transitions need them;
- final step derives the queried state and stops;
- metadata now includes `final_conclusion_kind=state` and `expected_proof_lines=2 * depth + 1`.

Depth-25 support was also fixed for eval. The one-letter predicate parser allows only 26 predicate symbols, so the FSA generator now reuses state predicates across different constants while still enforcing:

- unique states within each layer;
- no repeated output atom `(state, constant)`;
- unique same-layer transition antecedents;
- coherent continuations for all branches.

Focused tests passed:

```bash
source ./scripts/env.sh
${HPCVAULT}/.venv_rlvl_posttrain/bin/python -m pytest \
  tests/test_synthetic_dataset.py tests/test_training_stack.py -q
# 58 passed
```

A concrete fixed-target SFT sanity example is written to:

```bash
tmp/hfsa_fixed_target_sanity_2026-05-14.txt
```

In that example, the final proof line is the answer state atom, e.g. `Ee ; ->E`, `<conclusion>Ee</conclusion>`, `<answer>hazel</answer>`.

## Dataset Build

New HF target:

```bash
flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget
```

Build job:

```bash
3606767  build_hfsa_fix
```

Build script:

```bash
scripts/slurm/jobs/build_materialized_hard_fsa_schema_fixedtarget_2026-05-14.slurm
```

Subsets:

- `train_fixedtarget_up_to_3_50k`: depths `1..3`, no shortcut, K=4.
- `train_fixedtarget_up_to_5_50k`: depths `1..5`, no shortcut, K=4.
- `train_fixedtarget_up_to_10_50k`: depths `1..10`, no shortcut, K=4.
- `train_fixedtarget_up_to_15_50k`: depths `1..15`, no shortcut, K=4.
- `val_step_01_1k` ... `val_step_25_1k`: fixed-depth validation, no shortcut, K=4.

The pre-build probe accepted the distribution: candidate position is exactly balanced at chance and simple first/last/alphabetic heuristics are near 25%.

## One-Seed Experiment Matrix

Seed: `3407`.

SFT jobs:

```bash
3606768_[0-1]  sft_hfsa_fix
```

SFT rows:

- `sft1to3`: SFT on depths `1..3`, dataset subset `train_fixedtarget_up_to_3_50k`.
- `sft1to5`: SFT on depths `1..5`, dataset subset `train_fixedtarget_up_to_5_50k`.

SFT merge + sanity generation:

```bash
3606769_[0-1]  merge_sft_hfsa_fix
```

Merged SFT checkpoints:

- `${WORK}/synthetic-RLVL/tmp/merged_sft_hfsa_fixedtarget_sft1to3_seed3407`
- `${WORK}/synthetic-RLVL/tmp/merged_sft_hfsa_fixedtarget_sft1to5_seed3407`

GRPO jobs:

```bash
3606770_[0-3]  posttrain_hfsa_fix
```

GRPO rows:

- `sft1to3_rl1to10 / correct_plus_0p1_format`
- `sft1to3_rl1to10 / correct_plus_citation_free_valid_plus_0p1_format`
- `sft1to5_rl1to15 / correct_plus_0p1_format`
- `sft1to5_rl1to15 / correct_plus_citation_free_valid_plus_0p1_format`

All GRPO rows use:

- `grpo.train_steps=500`
- `num_prompts=8`
- `num_rollouts=8`
- `task.val_max_step=25`
- `eval.max_new_tokens=2048`
- no train shortcut: `task.shortcut_rate=0.0`

Post-hoc merge + pass@k eval:

```bash
3606771_[0-3]  hfsa_fix_passk
```

Eval protocol:

- steps `1..25`
- `samples_per_step=20`
- `num_generations=64`
- `k_values=1,2,4,8,16,32,64`
- logs sample generations to `passk_eval/hard_fsa_schema_fixedtarget/`
- logs pass@k metrics to the same W&B groups as the corresponding GRPO rows.

## Scientific Readout

This one-seed run is not for final claims; it is a fast sanity check after removing the target mismatch.

Primary questions:

- Does citation-free validity reward now raise citation-free validity when the conclusion must be the answer state?
- Does training on `1..10` or `1..15` improve OOD depths up to `25`?
- Does validity reward improve `valid_given_correct`, not merely raw correctness?
- Does correctness-only learn invalid answer shortcuts even without an explicit shortcut channel?
