# Nanotron MATH-500 Scoring Audit

Date: 2026-07-13

## Problem

The production `hendrycks_math500` task uses `exact_match,none`. Its normalizer
does not isolate the answer before comparing the full continuation. This made
the metric primarily a format-compliance test for the packed-continuation
models. For example, mathematically correct continuations such as
`$-50$.\nSolution: ...` and `$6+12i-3i=6+9i$.` received zero credit.

Applying symbolic verification to the whole continuation is not safe. It can
credit a wrong leading answer when later prompt repetition contains the gold
number, and it can collapse wrong equations or tuples to a shared scalar.

## Accepted Scorer

`scripts/analysis/rescore_math500.py` implements
`answer_prefix_math_verify,none`:

1. Use only the first nonempty line after the benchmark's literal `Answer:`
   prompt.
2. Reject unbalanced math delimiters and incomplete trailing operators.
3. Preserve full equations, tuples, intervals, and comma-separated answers.
4. For a direct calculation chain whose target is not itself an equation,
   score the final right-hand side.
5. Parse the complete target and extracted candidate inside math delimiters and
   compare them with `math_verify` symbolic equivalence.

The raw lm-eval exact score remains in every aggregate row as a diagnostic.
The post-hoc score is the primary MATH-500 value used in macros. This rule is
deterministic, condition-blind, and applied to all control/logic/NL direct and
post-instruction bundles. It intentionally scores the declared first answer
even if later explanation contradicts it; the task asks for the answer at that
position.

## Verification

Focused tests cover correct answer plus explanation, equivalent calculation
chains, wrong leading answers followed by gold prompt text, wrong equations,
wrong tuples, incomplete repetition, and extra comma-separated answers. The
scorer, bundle audit, aggregate, and Slurm tests pass (`18 passed`). The shared
environment retains Hydra-compatible `antlr4-python3-runtime==4.9.3`; it adds
only `math-verify==0.9.0` and `latex2sympy2-extended==1.11.0`.

For completed NL-direct run `3834904_8`:

- rows / unique documents: `500 / 500`
- stock exact: `14/500 = 0.028`
- answer-prefix symbolic: `80/500 = 0.160` (`stderr = 0.0164`)
- newly credited: `66`
- stock positives lost: `0`

All 80 target/extracted-answer pairs were inspected. Representative rejected
cases include a wrong leading `10` followed by a repeated prompt containing
gold `4`, response `(2/3,-13/3)` for target `(3/2,-13)`, a wrong plane equation
sharing right-hand side zero, response `1, 2` for target `1`, and a truncated
repeated expression containing the target digit. Representative accepted cases
include symbolic forms, reordered root sets, units, calculation chains, and
answers followed by explanation or next-record text.

## Artifacts

- Production sidecar:
  `$HPCVAULT/synthetic-RLVL/lm_eval_results/qwen25_branchproof_unique_v2_pilot_20260710/qwen25_7b_midtrain_nl_exact_p15_bp_unique_v2_4p3b_step8192_direct/math500_answer_prefix_math_verify.json`
- Updated production audit:
  the adjacent `production_audit.json`
- Sample SHA-256:
  `f65d766ee336d2ff41aafef3bb0f59cccabde28563fd2ae85b09d3138c9aa266`

No GPU evaluation rerun is required. The live audit creates or refreshes the
sidecar from retained MATH samples, including for Slurm jobs submitted before
this scorer was added.
