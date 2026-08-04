# ICLR 2027 Five-Week Execution Plan

Date: 2026-08-04

## Research Question

Does adding formal derivations to language-model training improve reasoning
more than adding natural-language renderings of the same derivations, even
when downstream responses use an ordinary instruction-following interface?

The causal transfer comparison is fixed to three conditions:

1. 100% Dolmino control.
2. 95% Dolmino plus 5% matched natural-language proof traces.
3. 95% Dolmino plus 5% matched formal proof traces.

Do not add a conditioned-dual Dolmino condition before this comparison is
complete. Dolmino already contains natural-language reasoning, and a second
mixed synthetic condition would be harder to interpret than the matched-NL
control.

## Primary Evidence

The corrected paired synthetic study establishes that the trace language can
change learned reasoning while prompts, answers, and latent proofs are fixed.
Use answer correctness as the primary outcome, joint correctness and validity
as a supporting outcome, and pass@16 as sampled support rather than a direct
claim about deployable top-1 performance.

The transfer experiment tests the practical claim. The primary readout is the
three-way comparison after identical instruction tuning. Direct base-model
evaluation remains a secondary diagnostic because raw midtraining checkpoints
show continuation and stopping artifacts.

## Preregistered Transfer Metrics

- Primary aggregate: unweighted macro over the frozen reviewer task suite
  after identical instruction tuning.
- Primary reasoning slice: unweighted QA-F1 macro over context-provided
  HotpotQA, 2WikiMultiHopQA, and MuSiQue.
- Required reporting: every task score, extraction/format failure rate,
  response length, and mean plus uncertainty over evaluation examples.
- Positive formal-transfer gate: the formal mixture must exceed both control
  and matched NL on the primary aggregate or on the preregistered multi-hop
  macro without a material regression on the other aggregate.
- Interpretation gate: inspect retained generations from every task and all
  three conditions before accepting an aggregate difference.

## Operational Sequence

### Week 1

- Finish and audit terminal matched-NL training `3875833_2`.
- Complete identical direct NL evaluation `3944017_2`.
- Use the fixed `5e-6` post-SFT recipe for all conditions; pending LR pilot
  `3944071_[0-1]` was canceled to avoid treatment-specific selection.
- Complete the benchmark-overlap audit on the already frozen instruction-data
  order before using the readout as publication evidence.
- Run identical control, matched-NL, and formal instruction tuning in
  `3950714`, followed by standard and multi-hop arrays `3950715/3950716`.
- Finish the six-row verifier-selection readout `3950178 -> 3950179`.

### Week 2

- Evaluate all three post-SFT checkpoints with the same standard, MATH, and
  multi-hop suite.
- Inspect generations and produce the first complete three-way table.
- Decide whether a 10B continuation is justified. Continue all three from
  their terminal checkpoints only if the 5B result is positive but uncertain
  or exhibits a consistent increasing trend.
- Add complete paired examples, full seed tables, and validator definitions to
  the ICLR appendix.

### Week 3

- If the 5B gate is positive, continue control, matched NL, and formal to 10B
  with the exact same data construction and schedule, then repeat the frozen
  post-SFT and evaluation protocol.
- If the gate is null, spend compute on one diagnosis supported by raw
  generations rather than launching a percentage grid.
- Freeze the main synthetic and hybrid figures.

### Week 4

- Run only targeted replication required by the selected headline result.
- Complete all tables, confidence reporting, ablation placement, and failure
  analysis.
- Conduct an internal claim-to-artifact audit of every sentence in the
  abstract and conclusion.

### Week 5

- Freeze experiments except for evaluator bugs that invalidate a reported
  result.
- Finish the paper, appendix, reproducibility statement, and artifact index.
- Perform independent code, sample, and LaTeX review.

## Stop Rules

- Do not launch the full 5/10/15/20/25 percentage grid before the clean 5%
  three-way post-SFT result is accepted.
- Do not launch a 32B midtraining wave within the five-week window unless the
  7B transfer result is clearly positive and the 32B token budget can complete
  with time left for post-SFT evaluation.
- Do not use iGSM or Maze as headline evidence. They remain negative task
  boundaries and are not part of the current paper claim.
- Do not use the original non-unique BranchProof results.
- Do not infer that pass@16 gains will become pass@1 gains under RL. Report
  verifier selection directly and describe RL as future work unless trained.
