# BranchProof Long-Depth Uniqueness Audit

Date: 2026-07-10

## Executive finding

The pre-2026-07-10 `hard_fsa`/`hard_fsa_schema` generator is not a
unique-answer benchmark at long depth. It selected from 18 one-letter constants
and wrapped the sequence after depth 17. Because state predicates are also
reused, different branches could later produce the same formal atom and make
multiple answer candidates derivable. The labeled answer remained derivable,
but it was often not the only valid answer.

All old BranchProof results above depth 17, including experiments derived from
that materialization, are quarantined. This is a dataset-construction defect,
not evidence that either trace substrate is better or worse.

## Closure audit

We parsed each generated premise set and computed Horn closure independently of
the gold proof. For each question, we counted how many of its four candidate
final-state atoms were derivable. The audit used 96 examples per depth (three
seeds, 32 indices each).

| Depth | Ambiguous examples | Mean derived candidates |
| ---: | ---: | ---: |
| 5 | 0/96 | 1.000 |
| 10 | 0/96 | 1.000 |
| 15 | 0/96 | 1.000 |
| 20 | 73/96 | 2.562 |
| 25 | 74/96 | 2.417 |
| 30 | 74/96 | 2.417 |
| 35 | 74/96 | 2.417 |
| 40 | 92/96 | 3.562 |
| 45 | 92/96 | 3.562 |
| 50 | 92/96 | 3.562 |

The labeled candidate was derivable in every audited example. The failure is
non-uniqueness: another candidate was also derivable in most long examples.

Stored generations expose the same issue without relying on the audit code. In
one depth-50 logic sample, the model answered a non-gold state, the answer metric
assigned `correct=0`, and the citation-free validator assigned `valid=1` to the
generated final atom. That output is valid under the old premises but counted
wrong under the single-label metric.

## Consequences for existing results

The old OLMo-7B train-1-to-25 aggregate also changes materially with the sample
budget:

| Metric | Logic | Natural language |
| --- | ---: | ---: |
| Long-depth pass@1 | 0.513 +/- 0.073 | 0.574 +/- 0.076 |
| Long-depth pass@4 | 0.753 | 0.699 |
| Long-depth pass@8 | 0.851 | 0.749 |
| Long-depth pass@16 | 0.921 | 0.794 |
| Depth-50 pass@1 | 0.241 +/- 0.012 | 0.147 +/- 0.011 |
| Depth-50 pass@16 | 0.833 | 0.510 |

Thus the old headline was not a robust statement that a random formal sample
is more likely to be correct. It was strongest at larger pass@k, where the
formal model may have explored more of several valid branches. Because the
benchmark itself was non-unique, neither that interpretation nor the old
architecture/syntax/shortcut/hybrid comparisons is publishable evidence.

Old eval also used unequal generation caps (formal `4096`, NL `6144`). A
corrected depth-50 token audit found gold prompt-plus-target totals below about
13.5k tokens, but target lengths can exceed both old caps. Corrected evaluation
therefore uses context length `16384` and the same `7168` new-token cap for both
modalities.

## Repair

The corrected generator:

1. Uses one fresh constant per layer, `c0..c_depth`, without wrapping.
2. Renders multi-character terms explicitly, for example `A(c18)`.
3. Computes Horn closure in the preflight probe.
4. Rejects a build unless every probed example has exactly one derivable
   candidate answer.
5. Preserves balanced gold-candidate positions and shortcut-neutral eval.

The closure audit after the fix found `0/96` ambiguous examples at every tested
depth from 5 through 50. The production gate over 1,000 train and 2,000 eval
examples reported unique-solution rate `1.0`, maximum derived candidates `1`,
zero failures, and balanced answer positions. Regression coverage includes
depths 25 and 50, strict/citation-free validation, translated NL validation,
and symbol-padded/wordified transforms over explicit atoms.

The corrected midtraining corpora have an independent packed-data audit at
`analysis/branchproof_unique_v2_corpus_audit_2026-07-10.json`. It records exact
source/packed token counts, all-record paired-prefix checks, and Qwen-tokenizer
round trips with decoded samples across depths and both modalities.

The SFT pilot now has a separate post-evaluation structural gate implemented
by `scripts/analysis/audit_branchproof_unique_v2_pilot_eval.py`. Before the
full grid can run, it requires all requested greedy and pass@k cells, finite
unit-range metrics, monotonic pass@k curves, all 14 requested depths, exactly
128 round-robin retained generations, and contiguous prompt constants
`c0..c_depth`. The gate uses 16 prompts/depth and 8 generations to avoid the
13--23-hour runtime of comparable full-size evals; the final 30-row evaluation
still uses 32 prompts/depth, 16 generations, and pass@16. Regression tests
explicitly reject the old depth-18 wrapped constant surface and missing primary
metrics.

At the planned 4,096-token Nanotron context, `44.3%` of formal documents and
`47.8%` of NL documents are longer than one context; none exceeds 8,192 tokens.
Nanotron therefore treats these as ordinary packed continuation documents, not
whole-example SFT records. The split exposure is closely matched across
modalities, but downstream claims should describe this as proof-corpus
midtraining rather than full-trace supervision.

The post-instruction downstream branch now uses Qwen's native chat template in
both UltraChat training and lm-eval, with loss restricted to assistant tokens.
Smoke `3831179` retained every sampled train/eval target and decoded to the
expected system/user/assistant surface. This replaces the earlier mismatched
custom-tag training plus untemplated evaluation path before any held
instruction job starts.

## Active recovery plan

| Stage | Jobs | Acceptance gate |
| --- | --- | --- |
| Materialized paired dataset | `3829067` complete | Probe accepted, all subsets present, private HF push succeeds |
| One-seed SFT pilot | `3829069_12` complete | Completed all 10,000 steps with final adapter and complete step-5000/10000 checkpoints; no truncation/data error |
| Pilot post-hoc eval | `3831135_12 -> 3831136`; eval running on A40 after live `a40,a100` widening | Automated audit verifies schema, all depths, corrected constants, and nonempty outputs; then inspect raw successes/failures at train, held-out, and depth 50 |
| Three-seed main grid | `3829072 -> 3829073` | Full SFT now depends on audit `3831136`, not process success alone; report greedy and pass@1 before pass@k |
| Corrected 1.2B-token corpora | builds and packed audit `3830855` complete | Full paired-prefix scan, metadata counts, and exact source-token/decode round trips passed |
| Midtraining mixtures | logic/NL smokes `3830924`/`3831110` complete; matched control/logic/NL p15 chains active or held | Compare direct and instruction-tuned downstream results, and launch the remaining percentages only after all three p15 conditions train, upload, and evaluate cleanly |

Selective ablation reruns are conditional on the corrected main result. If a
formal advantage survives at greedy/pass@1 across seeds, rerun one compact
syntax control, one shortcut mechanism, conditioned dual, and one independent
architecture before considering the broader old ablation matrix.
