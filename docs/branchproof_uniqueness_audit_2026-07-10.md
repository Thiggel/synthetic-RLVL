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
corrected exhaustive token audit rendered all 1,000 validation records at each
of 14 depths in both modalities with the OLMo tokenizer. Formal target/total
maxima are `6212/13123` tokens and NL maxima are `6674/13596`; zero of 28,000
gold traces exceed the shared generation or context budgets. Corrected
evaluation therefore uses context length `16384` and the same `7168`
new-token cap for both modalities, with at least 494 generation tokens and
2,788 context tokens of measured headroom. The accepted audit is
`analysis/branchproof_unique_v2_eval_token_budget_2026-07-11.json`.

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
`c0..c_depth`. It also requires every expected vLLM chunk-completion record and
stores greedy/sampled elapsed time, output-token throughput, maximum generated
length, and cap-hit counts. A cap hit is a model-behavior diagnostic and does
not by itself invalidate the evaluation. The gate uses 16 prompts/depth and 8 generations to avoid the
13--23-hour runtime of comparable full-size evals; the final 30-row evaluation
still uses 32 prompts/depth, 16 generations, and pass@16. Regression tests
explicitly reject the old depth-18 wrapped constant surface and missing primary
metrics.

The main gate's retained-sample budget is filled by greedy generations before
sampled scoring, so it cannot establish qualitative pass@k behavior. The
evaluator now accepts an independent sampled-sample budget and retains outputs
round-robin across prompts and generation indices. Dependency chain
`3833178_12 -> 3833179` reruns only a compact qualitative slice at depths
`1/25/30/50`, four prompts per depth and eight generations per prompt, and
requires all 128 sampled outputs with exact per-depth prompt/index coverage.
The full SFT array was held until both automated audits and manual
greedy/sampled generation review passed. The user approved the corrected
runtime protocol and the hold was released on 2026-07-11 at 10:15 CEST.

## One-seed corrected pilot outcome

Gate `3832945_12` completed in `07:10:38`; structural/runtime audit `3831136`
and sampled qualitative probe/audit `3833178_12 -> 3833179` accepted the
artifacts. The corrected prompts have fresh contiguous constants, complete
depth coverage, and expected answer extraction. The trained logic format uses
rule labels without numbered premise citations, so strict `valid=0` is expected;
the relevant self-contained proof metric is citation-free validity.

Citation-free correct-and-valid is perfect through train depth 25. At depths
30, 40, and 50, sampled pass@1 is `0.883`, `0.625`, and `0.344`; pass@8 is
`1.000`, `1.000`, and `0.938`. The separately retained depth-50 slice has
`15/32` correct answers, `11/32` citation-free valid-and-correct proofs, and
`24/32` complete-format generations. Manual inspection found correct complete
traces, late wrong-branch transitions, and repetitive/nonterminating outputs.
All 32 depth-50 rows preserved fresh constants. These failures are model length
extrapolation behavior, not evidence of renewed dataset ambiguity.

The gate measured `3826.3s` of greedy and `20677.8s` of sampled generation on
A40. The separate qualitative slice also ran on an A40 (`a0121`), despite an
earlier handoff interpretation that treated it as an A100 comparison. The
above-25-hour full-protocol projection is therefore A40-specific. Old full
A100-80 rows took roughly 3.5--8 hours, including `03:56:41` for matched old
row 12. Corrected retained outputs are substantially longer (depth-25 mean
`3236` vs `1923` tokens; depth-50 `6475` vs `2678`), use cap `7168` instead of
`4096`, batch `64` instead of `128`, and add greedy generation. The user
approved running the unchanged full protocol on A100-80GB. Eval array
`3834582` is hard-constrained accordingly; its first completed row is the
production timing gate, and only unfinished rows should be depth-sharded if a
row approaches the 24-hour limit. It replaces canceled pending array
`3829073` and retains two sampled generations for every prompt so qualitative
review spans all depths and rows.

Operational update 2026-07-11 15:30 CEST: eval rows `3834582_0/1/2` started
on A100-80GB without changing the protocol. Row 0 completed greedy generation
and reached sampled chunk `48/112` after about `2:37`. Its completed deeper
chunks take roughly three to five minutes each, projecting a whole-row runtime
comfortably below 24 hours; no depth sharding is currently justified. Many
deep train-1-to-5 formal chunks reach the shared `7168` cap. This is retained
as a model-behavior diagnostic until the row audit and raw samples are
available, not treated as a dataset or evaluator failure.

## Full-grid aggregation gate

`scripts/analysis/aggregate_hfsa_depth_scaling.py` now recognizes both the old
HFSA and corrected BranchProof-v2 run names. Corrected analysis must use
`--skip-intermediate` so old checkpoint curves cannot be mixed into the new
result, and `--strict-final-grid` so no tables or figures are written unless
all 30 unique logic/NL, train-range, and seed rows are present. Strict mode
also requires 448 prompts, 16 sampled generations, greedy correctness and
substrate-validity cells, and pass@`1/2/4/8/16` correctness, validity, and joint
cells at every depth, including monotonic pass@k checks.

```bash
source ./scripts/env.sh
${HPCVAULT}/.venv_rlvl_posttrain/bin/python \
  scripts/analysis/aggregate_hfsa_depth_scaling.py \
  --final-dir "${HPCVAULT}/synthetic-RLVL/passk_eval/branchproof_unique_v2_20260710" \
  --skip-intermediate \
  --strict-final-grid \
  --out-dir analysis/branchproof_unique_v2_20260711
```

The gate smoke correctly rejected the current one-row pilot and wrote a
machine-readable incomplete manifest. Focused tests and old-grid compatibility
checks pass; the existing old directory still resolves exactly 30 rows with no
completeness problems.

The aggregation outputs now enforce the planned analysis order. They preserve
the historical pass@16 columns while adding greedy correctness/validity and
pass@`1/2/4/8/16` correctness, modality-appropriate validity, and joint values
to per-run, grouped mean/std, depth, and paired-delta CSVs. A dedicated primary
figure separates greedy from sampled pass@1 OOD correctness; a second figure
shows train-1-to-25 correctness and joint scaling with sample budget. Focused
pytest passes (`4 passed`), and a 30-row compatibility smoke generated every
new table and figure. The compatibility grid predates greedy evaluation and
therefore renders that panel empty, while corrected production is strict-gated
on complete greedy cells.

Row audit array `3834706_[0-29%8]` is dependency-linked to eval `3834582`.
The row audit now treats translated NL evidence as a first-class required
artifact: NL rows must contain greedy and sampled translation parse,
citation-free validity, and correct-and-valid joint metrics, while every
retained row must contain finite unit-valued formal and translated validity
fields. This catches evaluator coverage loss per row rather than waiting for
the final 30-row aggregator. Seventeen focused tests and a rerun against the
real corrected pilot pass.
For each row it requires all greedy and sampled metric cells, exactly 1,024
retained generations, 896 sampled rows, 64 sampled rows and 32 unique prompts
at each depth, sample indices `0/1`, complete greedy/sampled chunk logs, and cap
diagnostics. Formal rows additionally require fresh `c0..c_depth` constants;
NL prompts are exempt from this surface-only check because they intentionally
render natural names. Replacement aggregation `3835779` depends on all 30
audits succeeding; canceled pending job `3834707` predates the final
qualitative gate. The replacement also runs
`audit_branchproof_unique_v2_qualitative_grid.py`, which requires all 30 sample
artifacts and exact retained counts before indexing shallow, train-edge,
first-OOD, and depth-50 correct/incorrect/valid/invalid cases across every
modality, train range, and seed. It additionally records examples from chunks
whose maximum observed generation reached the `7168` cap and writes JSON plus
Markdown review supplements. The 13-test focused pytest suite passed before
the final field-validation tightening, and all three qualitative test
functions pass afterward under direct system-Python invocation. A later
vault-venv pytest restart hit a filesystem-I/O timeout before collection.

The pilot gate initially used row 12's production filenames even though it had
only 224 prompts and eight sampled generations per prompt. The metrics and
samples are preserved with the row-12 run stem and a `_pilot_gate` suffix;
the unsuffixed names were cleared before any production eval started. Future
wrapper runs skip an existing output only after verifying 448 prompts, 16
sampled generations per prompt, 1,024 retained rows, and 896 sampled rows.

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
| Pilot post-hoc eval | corrected `3832945_12 -> 3831136` and sampled qualitative `3833178_12 -> 3833179` complete | Both audits accepted; manual review confirms intended prompt/extraction behavior and ordinary long-trace failures |
| Three-seed main grid | released `3829072 -> 3834582 -> 3834706 -> 3835779` (`3834707` canceled) | SFT hold released after user runtime approval. Eval is A100-80-only with unchanged full protocol and two retained sampled generations per prompt; row audits, cross-grid qualitative coverage, and strict aggregation are dependency-gated. Inspect the first row's runtime/raw outputs, then report greedy/pass@1 before pass@k |
| Corrected 1.2B-token corpora | builds and packed audit `3830855` complete | Full paired-prefix scan, metadata counts, and exact source-token/decode round trips passed |
| Midtraining mixtures | logic/NL smokes `3830924`/`3831110` complete; matched control/logic/NL p15 chains active or held | Compare direct and instruction-tuned downstream results, and launch the remaining percentages only after all three p15 conditions train, upload, and evaluate cleanly |

Selective ablation reruns are conditional on the corrected main result. If a
formal advantage survives at greedy/pass@1 across seeds, rerun one compact
syntax control, one shortcut mechanism, conditioned dual, and one independent
architecture before considering the broader old ablation matrix.

The official preprint now enforces this quarantine structurally. Its rendered
results contain only AttrCon, the uniqueness audit, and the corrected protocol;
all old BranchProof performance and derived ablation sections are behind a
disabled provenance switch. The informal report remains the historical record.
