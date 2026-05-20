# Pure SFT Logic vs Natural-Language CoT - 2026-05-15

## Motivation

The recent GRPO runs are hard to interpret because reward variance is sparse/noisy, training-depth behavior can saturate, and long-depth generations often fail through length and step-skipping effects. This experiment isolates supervised learning of the reasoning substrate:

- logic CoT: formal proof trace in `<formal>` plus `<answer>`.
- natural-language CoT: controlled natural trace in `<think>` plus `<answer>`.

The core question is whether longer pure SFT/midtraining on proof traces produces better depth extrapolation, and whether formal logic traces are more robust than natural-language traces when evaluated on longer chains.

As of 2026-05-19, this is the active research direction. The broader plan is in:

- `docs/formal_logic_cot_research_plan_2026-05-19.md`

The previous GRPO validity-reward direction is archived in:

- `docs/old_rl_validity_reward_direction_2026-05-19.md`

## Implementation Added

- Added deterministic NL-proof-to-FOL scoring in `synthrlvl/natural_logic.py`.
- `OutputEvaluator` now reports NL trace diagnostics when the template contains a natural proof and the example has gold constants/predicates:
  - `nl_logic_parse`
  - `nl_logic_citation_free_valid`
  - `nl_logic_joint`
  - `nl_logic_line_valid_fraction`
  - `nl_logic_valid_prefix_fraction`
- pass@k now logs:
  - `nl_logic_parse_pass@k`
  - `nl_logic_citation_free_valid_pass@k`
  - `nl_logic_joint_pass@k`
  - `nl_logic_valid_given_correct@k`
- `scripts/evaluate_checkpoint_passk.py` now accepts OmegaConf dotlist overrides after normal CLI flags, e.g. `task.template=nl_exact`.
- Existing constrained proof-line rejection sampling is kept as a post-hoc intervention for `template=logic` only.

Focused validation:

```bash
${HPCVAULT}/.venv_rlvl_posttrain/bin/python -m pytest \
  tests/test_training_stack.py \
  tests/test_materialized_data.py \
  tests/test_constrained_generation.py -q
```

Result: `15 passed`.

## First-Wave Jobs

Submitted one-seed OLMo-7B LoRA pure-SFT first wave:

- SFT array: `3612413_[0-3%4]`
- dependent merge + posthoc eval array: `3612414_[0-3%4]`

Scripts:

- `scripts/slurm/sweeps/sft/hfsa_pure_long_logic_vs_nl_2026-05-15.slurm`
- `scripts/slurm/jobs/posthoc_hfsa_pure_long_merge_eval_2026-05-15.slurm`

Rows:

| row | template | train depths | train subset | SFT steps |
| --- | --- | --- | --- | --- |
| 0 | `logic` | `1..10` | `train_fixedtarget_up_to_10_50k` | `10000` |
| 1 | `nl_exact` | `1..10` | `train_fixedtarget_up_to_10_50k` | `10000` |
| 2 | `logic` | `1..15` | `train_fixedtarget_up_to_15_50k` | `10000` |
| 3 | `nl_exact` | `1..15` | `train_fixedtarget_up_to_15_50k` | `10000` |

Posthoc eval defaults:

- eval depths: `1..25`
- prompts per proof length: `128`
- sampled completions per prompt: `16`
- pass@k: `1,2,4,8,16`
- max new tokens: `3072`
- constrained logic eval: enabled for `logic` rows only
- constrained prompts per proof length: `16`
- constrained candidate lines per proof line: `8`
- constrained pass@k: `1,2,4,8`

Outputs will be written to:

- `${WORK}/synthetic-RLVL/passk_eval/hfsa_pure_long/`
- `${WORK}/synthetic-RLVL/tmp/merged_sft_hfsa_pure_long_*`

## Evaluation Interpretation

Primary comparison:

- `logic` vs `nl_exact` at matched train depths and SFT steps.
- ID: depths inside the SFT range.
- OOD: depths above the SFT range up to 25.

Logic-specific metrics:

- formal validity
- citation-free validity
- joint correct+valid
- constrained proof-line pass@k

Natural-language-specific metrics:

- deterministic NL-to-FOL parse rate
- translated citation-free proof validity
- translated joint correct+valid
- translated valid-given-correct

The NL validity diagnostic is intentionally grammar-bound. It is not a general semantic parser; it tests whether generated controlled NL proof lines correspond to a valid formal derivation under the same gold theory.

## Recent Completed GRPO Analysis

The hard-FSA-schema easy curriculum completed before this submission. Summary CSV:

- `analysis/hfsa_easy_validity_2026-05-14/tables/easy500_passk_condition_summary_latest.csv`

Mean/std over 3 seeds:

| train shortcut | reward | train correct@1 | train cf-joint@1 | OOD correct@1 | OOD cf-valid@1 | OOD cf-joint@1 | hard-tail correct@1 | hard-tail cf-joint@1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | `correct_plus_0p1_format` | `0.998 +/- 0.000` | `0.998 +/- 0.000` | `0.461 +/- 0.048` | `0.314 +/- 0.019` | `0.269 +/- 0.025` | `0.192 +/- 0.055` | `0.013 +/- 0.002` |
| 0.0 | `indicator_correct_and_citation_free_valid_plus_0p1_format` | `0.999 +/- 0.001` | `0.998 +/- 0.001` | `0.462 +/- 0.048` | `0.314 +/- 0.023` | `0.269 +/- 0.027` | `0.196 +/- 0.059` | `0.013 +/- 0.000` |
| 0.5 | `correct_plus_0p1_format` | `0.999 +/- 0.001` | `0.998 +/- 0.001` | `0.458 +/- 0.045` | `0.318 +/- 0.018` | `0.274 +/- 0.026` | `0.187 +/- 0.051` | `0.012 +/- 0.001` |
| 0.5 | `indicator_correct_and_citation_free_valid_plus_0p1_format` | `0.998 +/- 0.001` | `0.998 +/- 0.001` | `0.463 +/- 0.043` | `0.326 +/- 0.023` | `0.282 +/- 0.028` | `0.189 +/- 0.048` | `0.013 +/- 0.001` |

Depth-specific mean/std:

| train shortcut | reward | step10 correct@1 | step10 cf-joint@1 | step15 correct@1 | step15 cf-joint@1 | step20 correct@1 | step20 cf-joint@1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | `correct_plus_0p1_format` | `0.702 +/- 0.077` | `0.524 +/- 0.114` | `0.297 +/- 0.065` | `0.028 +/- 0.004` | `0.071 +/- 0.025` | `0.002 +/- 0.002` |
| 0.0 | `indicator_correct_and_citation_free_valid_plus_0p1_format` | `0.710 +/- 0.071` | `0.528 +/- 0.116` | `0.286 +/- 0.062` | `0.032 +/- 0.008` | `0.066 +/- 0.036` | `0.002 +/- 0.001` |
| 0.5 | `correct_plus_0p1_format` | `0.707 +/- 0.076` | `0.536 +/- 0.108` | `0.280 +/- 0.044` | `0.027 +/- 0.004` | `0.064 +/- 0.027` | `0.004 +/- 0.002` |
| 0.5 | `indicator_correct_and_citation_free_valid_plus_0p1_format` | `0.727 +/- 0.064` | `0.555 +/- 0.102` | `0.284 +/- 0.049` | `0.029 +/- 0.004` | `0.070 +/- 0.024` | `0.003 +/- 0.001` |

Interpretation:

- The easy curriculum is learnable: train depths `1..5` are saturated for correctness and citation-free joint validity.
- OOD degradation is severe and primarily a long-chain proof validity problem.
- Validity-gated reward is at best slightly better in the shortcut-rate `0.5` condition, but the aggregate effect is small relative to seed variance.
- Step 10 has meaningful extrapolation; steps 15 and 20 mostly collapse in joint correct+valid proof quality.
- pass@k shows that correct answers often exist in the sample set, but valid+correct traces remain much rarer, especially on the hard tail.

## Fixed-Target GRPO Status At Submission Time

The fixed-target GRPO rerun is still active:

- `3606770_0`: `sft1to3_rl1to10 / correct_plus_0p1_format`, about `292/500` after `19:49:29`.
- `3606770_1`: `sft1to3_rl1to10 / correct_plus_citation_free_valid_plus_0p1_format`, about `283/500` after `19:42:29`.
- `3606770_2`: `sft1to5_rl1to15 / correct_plus_0p1_format`, about `171/500` after `19:32:39`.
- `3606770_3`: `sft1to5_rl1to15 / correct_plus_citation_free_valid_plus_0p1_format`, about `170/500` after `19:28:37`.

Continuation chain remains queued:

- `3608684 -> 3608685 -> 3608686`
- replacement pass@k eval: `3608687`

The fixed-target rows are alive but too slow to finish in the first 24h allocation, especially `rl1to15`.

## Status Update - 2026-05-16 10:26 CEST

SFT array `3612413_[0-3%4]` completed all four first-wave jobs:

| row | template | train depths | status | train loss | eval loss |
| --- | --- | ---: | --- | ---: | ---: |
| `0` | `logic` | `1..10` | completed | `0.0228` | `0.0737` |
| `1` | `nl_exact` | `1..10` | completed | `0.0004` | `0.0085` |
| `2` | `logic` | `1..15` | completed | `0.0251` | `0.0500` |
| `3` | `nl_exact` | `1..15` | completed | `0.0003` | `0.0176` |

The dependent pass@k eval array `3612414_[0-3%4]` is running:

| row | template | train depths | eval state |
| --- | --- | ---: | --- |
| `0` | `logic` | `1..10` | running; sampled chunks progressing |
| `1` | `nl_exact` | `1..10` | running; vLLM initialized but no sampled chunk printed yet |
| `2` | `logic` | `1..15` | running; sampled chunks progressing |
| `3` | `nl_exact` | `1..15` | running; vLLM initialized but no sampled chunk printed yet |

Preliminary online SFT eval, averaged over depth bands:

| row | template | correct `1..10` | correct `11..15` | correct `16..25` | translated/formal joint `1..10` | translated/formal joint `11..15` | translated/formal joint `16..25` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `logic` | `0.319` | `0.338` | `0.138` | `0.119` | `0.075` | `0.038` |
| `1` | `nl_exact` | `0.481` | `0.563` | `0.144` | `0.350` | `0.475` | `0.131` |
| `2` | `logic` | `0.175` | `0.113` | `0.081` | `0.113` | `0.075` | `0.038` |
| `3` | `nl_exact` | `0.769` | `0.300` | `0.056` | `0.763` | `0.300` | `0.056` |

Interpretation:

- This online eval is small and should not be treated as the final result.
- The large pass@k eval is the primary readout because it uses 128 prompts per proof length and 16 generations per prompt.
- Even as a sanity check, the current signal says controlled NL traces are easier for OLMo-7B to fit than formal logic traces under this LoRA SFT setup.
- Both trace styles still show strong long-depth degradation, so the key question is whether pass@k reveals latent correct samples that greedy validation misses.

## Depth-Scaling Extension - 2026-05-19

The one-seed pure-SFT result is now being extended into a 3-seed depth-scaling sweep.

Plan doc: `docs/hfsa_depth_scaling_plan_2026-05-19.md`.

New dataset target:

```bash
flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget-depth50
```

New grid:

| template | train depths | seeds | SFT steps | eval depths |
| --- | --- | --- | ---: | --- |
| `logic` | `1..5`, `1..10`, `1..15`, `1..20`, `1..25` | `3407`, `3408`, `3409` | 10,000 | `1..50` |
| `nl_exact` | `1..5`, `1..10`, `1..15`, `1..20`, `1..25` | `3407`, `3408`, `3409` | 10,000 | `1..50` |

Scripts:

- `scripts/slurm/jobs/build_materialized_hfsa_fixedtarget_depth50_2026-05-19.slurm`
- `scripts/slurm/sweeps/sft/hfsa_depth_scaling_logic_vs_nl_2026-05-19.slurm`
- `scripts/slurm/jobs/posthoc_hfsa_depth_scaling_merge_eval_2026-05-19.slurm`

This remains a LoRA SFT experiment on OLMo-3-1025-7B. If the depth extrapolation trend holds across seeds, the next stronger test is a non-LoRA/full-parameter or smaller-model pretraining setup to rule out adapter-specific effects.
