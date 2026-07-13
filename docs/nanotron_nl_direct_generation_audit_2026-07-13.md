# Nanotron NL-Direct Generation Audit

Date: 2026-07-13

## Scope

This audit covers completed production run `3834904_8`, the direct reviewer
suite for the Qwen2.5-7B checkpoint continued for 4.3B tokens with a matched
15% BranchProof natural-language mixture. The production artifact gate accepts
all 10 requested task groups, 105 leaf sample files, and 50,693 scored rows.
The comparison remains provisional until matched control and formal-logic
runs complete.

## Primary Readout

| Task | Score |
| --- | ---: |
| GSM8K | 0.529 |
| MATH-500, answer-prefix symbolic | 0.160 |
| ARC-Challenge | 0.551 |
| HellaSwag | 0.757 |
| WinoGrande | 0.704 |
| PIQA | 0.794 |
| LogiQA | 0.378 |
| BBH | 0.525 |
| MMLU | 0.665 |
| MMLU-Pro | 0.301 |

Targeted scores are MMLU formal logic `0.444`, BBH formal fallacies `0.508`,
and BBH logical deduction with three/five/seven objects
`0.736/0.400/0.328`. MATH-500 stock exact is only `0.028`; the separately
audited answer-prefix symbolic scorer recovers correct answers followed by
explanation and yields `0.160` without losing a stock positive.

## Generation Diagnostics

| Family | Rows | Invalid extraction | Next-document marker | Response chars p50 / p95 / max |
| --- | ---: | ---: | ---: | ---: |
| GSM8K | 1,319 | 0.0% | 0.0% | 363 / 883 / 1,186 |
| MATH-500 | 500 | 0.0% | 0.0% | 474 / 1,071 / 1,359 |
| BBH | 6,511 | 9.1% | 22.9% | 706 / 2,208 / 5,885 |
| MMLU-Pro | 12,032 | 20.5% | 3.7% | 343 / 9,877 / 15,526 |

The next-document marker is the literal generated text `You are an AI
assistant that helps people find information.` after an answer. It occurs in
zero corresponding input prompts, so it is continuation behavior rather than
prompt copying. The direct model was trained as a packed causal LM rather than
an instruction model; the matched post-instruction branch will test whether
native-chat SFT removes this behavior. The six-run aggregate now emits the
same condition-blind diagnostics for every checkpoint and branch.

## Sample Findings

- A correct GSM8K sample computes each egg use and emits the expected
  `#### 18`. A representative failure repeats the problem statement instead
  of solving it; the extractor correctly rejects it.
- Correct BBH Boolean and ordering samples reproduce the intended
  step-by-step semantics and answer. Some otherwise correct rows append the
  next-document marker after the extracted answer. Incorrect rows exhibit
  genuine reasoning failures: one omits the red book from a three-object
  ordering, and another reverses a one-way implication in a formal-fallacy
  question.
- A correct MMLU-Pro program-tracing row derives `3y` and selects `A` before
  appending the next-document marker. A representative incorrect row chooses
  an arbitrary implementation behavior despite the specification being
  underspecified.
- The longest invalid MMLU-Pro generations are repetition failures, not
  extractor bugs. One biology row recursively repeats
  `enzyme-producing enzyme`; two law rows repeat the same proposition until
  the generation budget ends and never state an option.
- A MATH-500 response with an extra closing parenthesis is correctly rejected.
  This distinguishes the new scorer from permissive whole-response symbolic
  matching.

These examples support treating the current format pathologies as direct-LM
generation failures. They do not establish whether the NL mixture causes or
reduces them; that comparison requires matched control, formal logic, and
post-instruction artifacts.

## Artifacts

- Run root:
  `$HPCVAULT/synthetic-RLVL/lm_eval_results/qwen25_branchproof_unique_v2_pilot_20260710/qwen25_7b_midtrain_nl_exact_p15_bp_unique_v2_4p3b_step8192_direct`
- Structural audit: adjacent `production_audit.json`
- MATH sidecar: adjacent `math500_answer_prefix_math_verify.json`
- Matched aggregate implementation:
  `scripts/analysis/aggregate_nanotron_downstream_pilot.py`
