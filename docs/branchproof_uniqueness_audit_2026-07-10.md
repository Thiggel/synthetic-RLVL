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

Operational update 2026-07-12 03:19 CEST: corrected NL SFT rows
`3829072_24/25` completed `0:0` and wrote the exact train-1-to-20
seed-3407/3408 final adapters. After verifying nonempty adapter configs and no
zero-byte files, only stale eval dependencies `3838163_24/25` were cleared.
The corrected grid now has `26/30` final adapters and 26 scheduler-eligible
A100-80 eval rows. Rows `26..29` remain running and dependency-gated near
steps `9209/6493/6469/6046`; no corrected production JSON/sample bundle exists
yet, so the audit and aggregation gates remain closed.

Operational update 2026-07-12 09:21 CEST: corrected NL rows
`3829072_26..29` completed `0:0`, bringing the full corrected SFT grid to
`30/30`. Their exact train-1-to-20 seed-3409 and train-1-to-25 three-seed final
adapters have nonempty configs, no zero-byte files, and no fatal/OOM/quota log
signature. Slurm resolved the four remaining row-wise dependencies without a
manual edit. All 30 tasks in validity-fixed eval `3838163` are therefore
dependency-free, remain constrained to `a100` plus `a100_80`, and wait only on
account GRES. The production output root still contains only the four suffixed
pilot/qualitative files, so no corrected metric or raw-generation conclusion is
available and both downstream audit gates remain closed.

Operational update 2026-07-12 18:50 CEST: all 30 validity-fixed eval tasks
`3838163_[0-29%6]` remain dependency-free and retain partition `a100`, feature
`a100_80`, and the unchanged full protocol. They are pending only on the
account GPU ceiling (`AssocGrpGRES`), even though an A100-80 node is idle. No
corrected production JSON or sample bundle exists, so row audits `3838164` and
aggregate/qualitative gate `3838165` remain closed and no scientific result is
available. No dependency, partition, throttle, or protocol edit was made.

Operational update 2026-07-13 00:55 CEST: all 30 validity-fixed eval tasks
remain dependency-free, A100-80-only, and blocked by the account GPU ceiling.
A temporary scheduler estimate for row 0 disappeared on the next cycle, so no
row has launched and the production root still contains only the four
explicitly suffixed pilot/qualitative files. The only idle A100 nodes were
`a0903/a0905`, both `a100_40`, and are incompatible with the required feature;
no partition, feature, throttle, dependency, or protocol edit was made.
Successor oversight `3845763` is scheduled for 06:46 CEST with the corrected
artifact and qualitative gates preserved.

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
to per-run, grouped mean/std, depth, paired-seed, and paired-delta-summary CSVs.
The paired summary reports mean/std across matched seeds for greedy and every
sample budget; missing evidence remains `NaN` instead of becoming a fabricated
zero delta. A dedicated primary
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
| Three-seed main grid | SFT `3829072` complete; answer-fixed replacement `3853284 -> 3853285 -> 3853286`; older eval/audit/aggregate chains canceled and quarantined | A later audit found permissive answer-token containment, so even the validity-fixed `3838163` outputs are diagnostics only. Replacement eval is A100-80-only with unchanged full protocol and two retained sampled generations per prompt; CPU row audits, cross-grid qualitative coverage, and strict aggregation are dependency-gated. Inspect the first row's runtime/raw outputs, then report greedy/pass@1 before pass@k. |
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

## 2026-07-14 Answer-Matcher Supersession

The uniqueness and premise-validity corrections remain valid, but their first
full evaluation chain is additionally quarantined. The common answer matcher
credited a gold token embedded among alternatives; two such retained false
positives were found in five complete bundles, and full pass@k cannot be
post-hoc reconstructed from the retained subset. Chain
`3838163 -> 3847756 -> 3847757` was canceled, its artifacts were moved under
`quarantine/pre_answer_match_fix_20260714/`, and exact-or-single-assertion
matching plus an independent audit guard now pass the full suite. Clean
A100-80/CPU replacement chain `3853284 -> 3853285 -> 3853286` preserves every
scientific setting. No prior corrected correctness number is report evidence.

## 2026-07-14 Tiny/32B Recovery and Baseline Runtime Audit

The no-repeat tiny final eval `3850492` completed all 18
size/template/seed rows. A structural audit accepted every metrics JSON,
sample JSONL, and log; representative raw review covered logic and NL at
depths 1, 10, and 50, all three seeds and sizes, correct/incorrect cases, and
generation-cap hits. Shallow logic can be citation-free valid, but many
depth-10 answers have invalid traces and depth-50 output frequently
degenerates or truncates. Joint correct-and-valid pass@1/4/8 is zero in every
reported size/template aggregate. These are provisional mechanism-smoke
findings, not report evidence, until the checkpoint curve completes. The
machine-readable audit bundle is
`$HPCVAULT/synthetic-RLVL/analysis/branchproof_unique_v2_tiny_100k_final_audits_20260714`.

Checkpoint eval `3850493` had no usable outputs: its first 39 tasks failed
because an intermediate checkpoint contains model/config files but no
tokenizer assets, causing tokenizer resolution to fall back to an incompatible
class. The evaluator and wrapper now support an explicit tokenizer path, and
the Slurm job uses the matching run's verified `final/` tokenizer while
loading weights from each intermediate checkpoint. Preflight checks accepted
all 90 checkpoint directories and all 18 final tokenizers. The exact
90-task/three-concurrent replacement is `3854813_[0-89%3]`; the failed array
was canceled before the remaining 49 tasks consumed GPUs.

Corrected 32B SFT tasks 0 and 1 failed while downloading the public base due
to transient expired Hugging Face Xet signed URLs. Task 2 downloaded the same
base later and is training normally, so no code or protocol change is
justified. Targeted recovery `3854837_[0-1%1]` preserves the original array
indices and resources. Eval `3850123` was dependency-edited to require the
recovery and original successful-required tasks `3850115_2..14`; it cannot
release on a partial SFT grid.

At 07:14 CEST clean baseline eval `3853284_0..5` had run for
`4:58--5:46` on verified A100-80GB devices. Sampled-chunk progress projected
roughly `6.5--11.8` hours total, below the `20--24` hour sharding threshold.
Rows `6..29`, audits `3853285`, and aggregate `3853286` remain pending by
throttle/dependency; the full protocol is unchanged. The complete repository
suite after the evaluator/audit changes passes `216 passed, 3 skipped`.
Report and preprint regeneration is deferred until the baseline, report-wide
families, and tiny checkpoint curve pass their artifact and qualitative gates.

## 2026-07-14 First Clean Full-Protocol Rows

Clean exact-answer eval rows `3853284_0/3/5` completed in
`11:05:17/10:09:23/10:34:35`, and row audits `3853285_0/3/5` accepted all
three. Each metrics bundle has 448 prompts, 16 sampled generations per prompt,
the full 2,665-metric schema, and a matching 1,024-row sample JSONL with 896
sampled rows. Fresh-constant failure count is zero throughout.

Representative logic samples across seeds 3407/3409 and train maxima 5/10
confirm the intended corrected prompt (`c0..c_depth`) and exact answer
extraction. Train-band outputs are typically correct and citation-free valid.
At depth 25, answer-correct traces can already be invalid because later proof
lines are unsupported or malformed. At depth 50, many retained generations
repeat labels/rules until the 7,168-token cap and omit a usable answer. The
three rows' sampled pass@1 OOD correctness is provisional and incomplete; NL
rows have not completed, so no modality claim is permitted.

Rows `3853284_1/2/4` remained healthy at sampled chunks `83/95/108` of 112,
and rows 6/7 backfilled as slots opened. Completed runtimes and current
progress remain below the 20--24-hour sharding trigger, so the A100-80-only
protocol was not changed. The full suite now passes `223 passed, 3 skipped`.

The tiny checkpoint replacement separately accumulated seven Slurm-startup
cancellations without output artifacts. Exact recovery
`3856145_[24,26,28-32%3]` preserves those indices and is safely widened to
generic one-GPU `a40,a100`; it does not alter the clean 7B baseline policy.

## 2026-07-14 Tiny Checkpoint-Curve Acceptance

Replacement/recovery `3854813 + 3856145` completed the intended `90/90`
checkpoint rows: three sizes, two templates, three seeds, and five checkpoints
at 20k-example intervals. Each metrics bundle has all 1,899 expected metrics;
each sample JSONL has 288 rows, including 192 sampled rows with 16 unique
prompts at every requested depth; each exact original/recovery log has complete
greedy and sampled chunk accounting under the 7,168-token cap. Fresh-constant
failures are zero.

The retained-sample audit was strengthened to apply the diagnostic-consistency
gate whenever `citation_free_valid=1`, independent of strict validity. All 90
rows re-audited accepted. Across the complete retained set, every credited
citation-free-valid record has an empty validity error, no invalid proof lines,
and line-valid fraction one.

Three-seed checkpoint means show limited answer-only learning but no valid
length extrapolation. The largest OOD correct pass@1 mean is about `0.052` and
the largest OOD correct pass@8 mean is about `0.221`; modality-appropriate OOD
and depth-50 joint pass@1/4/8 are `0.000` for every size/template/exposure
aggregate. Raw review across all sizes/templates/seeds, early/middle/final
checkpoints, depths 1/10/50, correct/incorrect cases, and cap hits confirms the
mechanism: shallow outputs can be correct and valid, depth-10 outputs are often
answer-correct with non-derivable traces, and depth-50 outputs truncate,
repeat, or lose structure. The accepted conclusion is negative and limited to
these 50M--200M one-pass scratch models.

Audit artifacts are under
`$HPCVAULT/synthetic-RLVL/analysis/branchproof_unique_v2_tiny_100k_checkpoint_audits_20260714`.
After acceptance, a guard verified all 18 finals, 90 metrics, 90 samples, 90
audits, terminal eval parents, and no live dependency. It then removed only the
90 intermediate checkpoints (`102G`); finals and all evaluation artifacts were
preserved.
