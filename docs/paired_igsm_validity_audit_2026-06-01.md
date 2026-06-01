# Paired iGSM Validity Audit - 2026-06-01

## Inputs Checked

- Metrics root: `$WORK/synthetic-RLVL/passk_eval/paired_full_suite_sparse_20260528/`
- Completed family: `official_igsm` only, `30/30` pass@k JSONs and sample JSONLs.
- Focused sample files:
  - `sft_paired_full_official_igsm_logic_train1to25_10k_seed3409_samples.jsonl`
  - `sft_paired_full_official_igsm_nl_exact_train1to25_10k_seed3409_samples.jsonl`
- Reconstructed eval records with `TaskBuilder` using `official_igsm`, eval seed `3407`, sparse depths `1,2,5,10,12,15,18,20,25,30,35,40,45,50`, and `32` prompts per depth.

## Metric Readout

These are diagnostics-only until the rest of the paired family finishes and an iGSM-aware NL validity evaluator exists.

| Template | Train Max | OOD correct@16 | OOD internal/translated joint@16 | Depth-50 correct@16 | Depth-50 internal/translated joint@16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| logic | 5 | 0.312 | 0.255 | 0.240 | 0.177 |
| logic | 10 | 0.507 | 0.377 | 0.458 | 0.281 |
| logic | 15 | 0.546 | 0.392 | 0.542 | 0.292 |
| logic | 20 | 0.536 | 0.245 | 0.510 | 0.219 |
| logic | 25 | 0.487 | 0.106 | 0.490 | 0.031 |
| nl_exact | 5 | 0.366 | 0.000 | 0.333 | 0.000 |
| nl_exact | 10 | 0.589 | 0.000 | 0.490 | 0.000 |
| nl_exact | 15 | 0.618 | 0.000 | 0.521 | 0.000 |
| nl_exact | 20 | 0.576 | 0.000 | 0.573 | 0.000 |
| nl_exact | 25 | 0.585 | 0.000 | 0.521 | 0.000 |

Logic grounded joint was `0.000` in this aggregate readout, despite nonzero internal joint. That makes the iGSM logic result weaker than the internal-validity metric alone suggests.

## NL Translated Validity Diagnosis

The `nl_exact` translated validity of `0.000` is an evaluator coverage bug, not evidence that natural-language iGSM traces are intrinsically invalid.

Current `synthrlvl/natural_logic.py` supports controlled HFSA-style assertions such as `a is teal`. It does not parse iGSM proof grammar such as:

```text
From the official iGSM relation, v_k equals 20.
Substitute k = 20 into the current expression.
Evaluate the arithmetic modulo 23 to get v_v = 19.
```

A direct translator check returned only `INVALID ; R` lines with errors like `could not translate From the official iGSM relation...`. Gold iGSM `proof_nl` targets use the same unsupported grammar, so generated iGSM NL samples receive zero `nl_logic_parse` and zero translated validity even when they follow the expected iGSM prose surface.

## Sample Findings

- Logic step-10, answer-correct/internal-valid/ungrounded: question `How many Backpack does Dance Studio have?`, gold conclusion `v_a = 0`, generated conclusion `v_Y = 0`. The generated proof is internally coherent under its own invented premises (`v_m`, `v_y`, `v_S`, ...), but the variable chain does not match the gold chain (`v_S`, `v_U`, `v_y`, ..., `v_a`). This is answer-correct but not grounded.
- Logic step-25, wrong/internal-valid/ungrounded: question `How many Genetics Lab does Westland College have?`, gold conclusion `v_g = 1`, generated conclusion `v_m = 18`. Again the model produced a valid-looking formal chain, but with a different variable trajectory and the wrong answer.
- Logic step-50, answer-correct/invalid/ungrounded: question `How many Bone does Kruger National Park have?`, gold conclusion `v_r = 0`, generated conclusion `v_y = 0`. The numeric answer matches, but the proof is invalid and not the gold chain.
- NL step-15, answer-correct: question `How many Top-Loading Backpack does Genetics Lab have?`, gold conclusion `v_w = 7`, generated final line `Evaluate the arithmetic modulo 23 to get v_s = 7.` The prose format is the intended iGSM format, but it is not the gold variable chain.
- NL step-25, wrong: question `How many Genetics Lab does Westland College have?`, gold conclusion `v_g = 1`, generated final line `Evaluate the arithmetic modulo 23 to get v_o = 5.`
- NL step-50, answer-correct: question `How many Bone does Kruger National Park have?`, gold conclusion `v_r = 0`, generated final line `Evaluate the arithmetic modulo 23 to get v_N = 0.` This is answer-correct but not verified as a valid/gold iGSM derivation by the current evaluator.

## Interpretation

NL is ahead on answer correctness for completed iGSM rows, but its validity metrics are currently unavailable. Logic is worse on answer correctness because it often pays the full formalization burden, then drifts into a self-consistent but ungrounded formal problem. The internal proof checker can bless those invented formal chains, so internal joint is not enough for iGSM claims. For iGSM, answer correctness and internal validity should be reported separately from grounded/canonical validity until the evaluator is strengthened.

## Next Steps

1. Add an iGSM-specific NL proof parser/validator that recognizes official-relation, substitution, and modulo-23 lines.
2. Score generated iGSM NL traces against the actual prompt/gold equation chain, not only against generic HFSA natural-language translation.
3. Add regression tests using gold iGSM `proof_nl` lines so translated validity is nonzero for valid targets before rerunning or reaggregating paired NL validity.
4. For logic, emphasize grounded/citation-free-grounded validity in the report; internal validity alone overstates iGSM proof quality.
5. Recompute paired iGSM summaries after the parser fix, then analyze `maze_navigation` and hard `attribute_constraints` once their eval rows finish.

## Fix Status - 2026-06-01 15:29 CEST

Items 1 and 3 are implemented for iGSM:

- `synthrlvl/natural_logic.py` now translates official-relation, substitution, and modulo-23 iGSM NL proof lines into formal equality/MOD23 proof lines with citations.
- `synthrlvl/metrics.py` now lets translated NL validity use strict proof validation when citation-free recovery cannot justify equality substitution or MOD23.
- Regression coverage was added in `tests/test_training_stack.py`; `tests/test_training_stack.py` passed (`28 passed`) and `tests/test_paired_synthetic_datasets.py` passed (`9 passed`).
- Materialized gold official_iGSM `nl_exact` targets sampled at depths `1/10/25/50` now score format/correct/parse/valid as `1.0`.
- Minimal recomputation job `3689003_[3-5,9-11,15-17,21-23,27-29%4]` was submitted with `FORCE_PASSK_EVAL=1` to rerun only the 15 completed official_iGSM `nl_exact` pass@k rows.

## Partial Rerun Readout - 2026-06-01 18:41 CEST

Targeted rerun `3689003` has completed `8/15` rows so far: train-1-to-5 seeds `3407/3408/3409`, train-1-to-10 seeds `3407/3408/3409`, and train-1-to-15 seeds `3407/3408`.

Current rerun-visible `official_iGSM` `nl_exact` aggregate:

| Train Max | Rows rerun | OOD correct@16 | OOD parse@16 | OOD translated joint@16 | Depth-50 correct@16 | Depth-50 parse@16 | Depth-50 translated joint@16 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | `3407,3408,3409` | 0.359 | 1.000 | 0.000 | 0.260 | 1.000 | 0.000 |
| 10 | `3407,3408,3409` | 0.569 | 1.000 | 0.000 | 0.385 | 1.000 | 0.000 |
| 15 | `3407,3408` plus one stale row | 0.612 | 0.664 | 0.000 | 0.479 | 0.656 | 0.000 |
| 20 | none yet | 0.576 | 0.000 | 0.000 | 0.573 | 0.000 | 0.000 |
| 25 | none yet | 0.585 | 0.000 | 0.000 | 0.521 | 0.000 | 0.000 |

The parser-coverage part of the iGSM fix is working on completed rerun rows, but generated translated validity remains zero. Representative rerun samples parse as iGSM NL proof lines, then fail strict translated validation because generated variable names and chains often do not match the gold formal premises. This is now a generated-trace grounding/canonicality issue rather than the original parser-coverage bug.
