# BranchProof Validity Evaluator Audit (2026-07-12)

## Finding

The first full corrected BranchProof production eval exposed an evaluator bug
that was independent of the corrected dataset generator. `ProofAnalyzer`
recorded malformed generated premises and a `premise parse failed` error, but
its final `ok` value only required valid proof lines plus a supported
conclusion. It did not require every premise to parse. Consequently, a trace
could receive `citation_free_valid=1` while the retained diagnostics for the
same trace reported a citation-free validity error.

This defect is a validity/joint-metric bug. Its fix did not alter answer
extraction. A later, independent answer-matching audit on 2026-07-14 found a
separate correctness defect, documented below; that later finding supersedes
the original acceptance of answer correctness from this chain.

## Evidence

- Pre-fix eval row `3834582_0` completed on an A100-80GB GPU in `11:18:40`
  with the intended 448 prompts, 16 generations per prompt, and 1,024 retained
  samples.
- Re-running the strengthened artifact audit rejected 14 of the 896 retained
  sampled rows with `citation_free_valid=1` plus a premise-parse error.
- The contradictions occurred at depths 10 (1), 12 (3), 15 (8), 18 (1), and
  20 (1). Examples included malformed premises such as `H(c6)[+++]`, an empty
  formula, `M(c10)_Print`, invalid conjunctions, and invalid term syntax.
- Direct sample inspection also found duplicate generated predicate aliases,
  answer-correct but malformed self-contained theories, wrong yet internally
  derivable conclusions, and depth-50 repetition that exhausted the generation
  cap without an answer. These are model failures; they must not count as valid
  proofs.

The pre-fix metrics and retained samples are quarantined under:

- `$HPCVAULT/synthetic-RLVL/passk_eval/branchproof_unique_v2_20260710/quarantine_pre_validity_fix_20260712/`
- `analysis/branchproof_unique_v2_quarantine_pre_validity_fix_20260712/`

They are outside the strict aggregator's input directory and are not paper
evidence.

## Fix And Verification

- Strict and citation-free proof reports now require every parsed premise to be
  syntactically valid before `ok` can be true.
- The row audit rejects a retained `citation_free_valid=1` sample if it also
  has a validity error, nonempty invalid-line diagnostics, or a line-valid
  fraction below one.
- The full-grid audit wrapper now passes each row's actual training maximum.
- A regression test constructs a malformed premise plus an otherwise supported
  conclusion and requires both strict and citation-free validation to fail.
- All 26 focused test functions across the logic engine, BranchProof artifact
  audit, and Nanotron checkpoint verifier pass through the preferred project
  environment (`26 passed in 0.34s`). Bytecode, shell syntax, and
  `git diff --check` also passed. This closes the earlier verification gap from
  transient vault metadata I/O during concurrent 199 GB checkpoint writes.

## Recovery

Old eval `3834582`, audit `3834706`, and aggregate `3835779` were canceled.
Rows 1 and 2 were stopped after about `09:57` rather than spending another
5--8 hours finishing with the pre-fix in-memory scorer. The clean replacement
chain is:

- A100-80GB eval: `3838163_[0-29%6]`
- CPU-only row audit: `3847756_[0-29%8]`
- CPU-only strict aggregate and qualitative audit: `3847757`

The original replacement audit/aggregate `3838164/3838165` were canceled
before start because they unnecessarily requested MIG GPUs. The new jobs keep
the same `aftercorr:3838163` and all-rows `afterok` gates, request no GRES, and
default to the corrected eval array ID. The aggregate now reports three-seed
standard deviations in the primary Markdown table and as bands on every depth
curve.

Only eval rows 0--2 repeat generation work. Rows 3--29 had not started in the
old array. No corrected BranchProof validity or joint result is accepted until
all replacement rows pass the strengthened gate. Correctness is still the
primary claim metric, but the rerun is necessary to keep the secondary
correct-and-valid evidence defensible.

## 2026-07-14 Answer-Match Correction

An independent retained-generation scan found that the common
`_is_answer_match` implementation accepted the gold token anywhere in the
normalized `<answer>` body. Five complete `3838163` bundles contained one
false-positive retained greedy row among 640 and one false-positive retained
sampled row among 4,480. One malformed answer listed 20 labels including the
gold; the other listed several labels including the gold while still passing
the format check. These are scorer errors, not model successes.

Only two of 16 sampled generations per prompt were retained, so the full
pass@k cells cannot be exactly rescored. Eval `3838163`, audits `3847756`, and
aggregate `3847757` were canceled. Five complete metric/sample pairs plus
partial diagnostic outputs are outside the production input path under
`$HPCVAULT/synthetic-RLVL/passk_eval/branchproof_unique_v2_20260710/quarantine/pre_answer_match_fix_20260714/`.
Previously accepted tracked audits are preserved under
`analysis/branchproof_unique_v2_full_grid_audits_pre_answer_match_fix_20260714/`
and must not be treated as evidence.

The matcher now accepts either an exact normalized answer or a single-line
natural assertion such as `Yara is sparse`; it rejects alternative lists and
multi-line bodies. The row audit independently recomputes that strict
answer-shape match from each retained generation and gold answer. Targeted and
full verification pass (`213 passed, 3 skipped`), and `git diff --check`
passes. The clean replacement chain is A100-80 eval
`3853284_[0-29%6]`, CPU row audit `3853285_[0-29%8]`, and CPU strict
aggregate/qualitative gate `3853286`. It preserves the full scientific
protocol and repeats only evaluation, not the already accepted 30 SFT rows.

Operationally, the replacement array was submitted after 22 SFT tasks had
already completed, and Slurm left their wildcard `aftercorr` dependencies
unfulfilled. At 00:41 CEST, those 22 parent rows were verified `COMPLETED 0:0`
and their final adapters were checked before clearing only the corresponding
eval-task dependencies. Rows `13/14/24..29` retain `aftercorr`; all eval tasks
remain constrained to A100-80GB GPUs.

Rows `13/14` completed successfully and wrote final adapters at 01:04/01:05
CEST. After the same parent-state and artifact checks, only their corresponding
dependencies were cleared. The eligible corrected eval set is now `24/30`;
active NL rows `24..29` remain gated.

## Replacement Launch

By 2026-07-13 08:16 CEST, full corrected SFT was complete at `30/30`, all
final adapters passed the nonempty/no-zero-byte gate, and every eval task was
dependency-free. Replacement eval rows `3838163_0..5` started at
10:07--10:11 CEST on six verified A100-SXM4-80GB allocations. Their startup
logs show the intended corrected adapters, isolated merge roots, a 16,384-token
vLLM context, and no fatal/OOM/quota signature. Row 0 entered the unchanged
greedy generation protocol with 64 prompts per chunk and
`max_new_tokens=7168`; its first chunk completed successfully. Rows `6..29`
now wait only on the six-task array throttle. No replacement production JSON
or sample bundle is complete yet, so this establishes launch health only; the
strengthened row audits and cross-grid qualitative gate remain mandatory.

## Tokenizer Consistency

The replacement logs emit a generic Transformers warning suggesting
`fix_mistral_regex=True`. This warning is not evidence of a train/eval
tokenization mismatch for the OLMo checkpoint used here. A corpus-level audit
compared the training tokenizer loaded from `allenai/Olmo-3-1025-7B` with the
default tokenizer copied into a current merged evaluation checkpoint on 704
actual BranchProof texts: prompts and targets for 16 rows per modality at
depths 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, and 50. Their token IDs matched
on all `704/704` texts. The core tokenizer files are also byte-identical.

Conversely, enabling `fix_mistral_regex=True` on the merged tokenizer changes
token IDs for `640/704` audited texts. Applying the suggested fix during this
evaluation would therefore create a train/eval mismatch. The only serialized
file difference is `special_tokens_map.json`: the base uses strings for eos
and pad, while the merged checkpoint uses semantically equivalent AddedToken
dictionaries with the same content and IDs. The running protocol correctly
keeps the default tokenizer. Exact hashes and counts are in
`analysis/branchproof_tokenizer_consistency_2026-07-13.json`.
