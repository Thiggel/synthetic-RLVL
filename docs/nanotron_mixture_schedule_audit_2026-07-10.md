# Nanotron Mixture-Schedule Audit (2026-07-10)

## Conclusion

The corrected Qwen2.5-7B midtraining pilot implements the intended matched
token mixture. Nanotron blends fixed 4,096-token packed chunks. For each 15%
proof condition, the full 8,192-step schedule contains 157,287 proof chunks
and 891,289 normal-text chunks, corresponding to 644,247,552 proof tokens out
of 4,294,967,296 total tokens (`15.000057%`). Logic and NL therefore receive
the same proof-token exposure to within the identity of the source corpus.

No schedule, capacity, or checkpoint-resume bug was found. The three-step
integration smokes realize one proof chunk out of three (`33.3%`) because they
are intentionally too small to represent 15% finely; this smoke-only
granularity does not apply to production.

## Downstream acceptance gate

The installed lm-eval task registry does not contain `folio`; the original
default would therefore fail all downstream jobs during task resolution.
Legacy `logiqa` and `logiqa2` also depend on dataset scripts rejected by the
installed `datasets`; `agieval_logiqa_en` is the maintained replacement.
Diagnostic raw samples additionally showed that installed FLD prompts request
a proof while exact-match scoring expects only a class label. Because that
creates an extraction-confounded floor, both FLD surfaces are excluded from
the production transfer claim rather than patched post hoc.

The final suite is `gsm8k`, `hendrycks_math500`, `arc_challenge`,
`hellaswag`, `winogrande`, `piqa`, `agieval_logiqa_en`, `bbh`, `mmlu`, and
`mmlu_pro`. MMLU formal logic and BBH's logic subtasks provide targeted logic
readouts; the full groups and remaining tasks measure broader transfer.

Smokes `3834728/3834737` found the LogiQA incompatibilities. Diagnostic smoke
`3834738` completed direct and native-chat execution and exposed the FLD
metric problem through retained generations. Production evals are gated on
final ten-task smoke `3834836`.
The eval wrapper preflights actual task/dataset construction and archives incomplete output directories;
the previous directory-exists guard could incorrectly suppress a retry after
lm-eval wrote only `command.json` and then failed.

Production output acceptance is also explicit. The downstream wrapper runs
`scripts/analysis/audit_nanotron_downstream_eval.py` before accepting either a
new or pre-existing result. The audit requires the exact ten-task command, all
ten top-level task/group results, all 105 expanded leaf-task sample files,
full unique-document coverage from lm-eval's `n-samples` metadata, finite
primary metrics, an un-limited production run, and direct versus Qwen-chat
prompt rendering consistent with the evaluation branch. Five focused tests
and both final-smoke branches pass; the smoke has 106 JSONL rows because GSM8K
stores separate strict and flexible-extraction filter rows for the same
document, which the audit correctly counts once by `doc_id`.

The matched comparison is also fixed before results. Control evals
`3835927/3835928`, logic `3834908/3834909`, and NL `3834904/3834905` all write
to the unified corrected root and feed strict aggregate `3836159`. Canceled
pending control evals `3834906/3834907` used the legacy default root. The
aggregate requires all six production audits and reports each primary task,
task stderr, deltas from control, and instruction-minus-direct deltas. Its
predeclared unweighted macros are all-primary, reasoning-core, general
multiple-choice, and targeted logic. Targeted logic contains LogiQA, MMLU
formal logic, BBH formal fallacies, and BBH logical deduction at three, five,
and seven objects. Representative correct/incorrect samples are indexed with
the exact primary filter, avoiding GSM8K strict-versus-flexible mixing. Because
there is one training run per condition, macro results do not estimate
training-seed variance.

## Production Configuration

The production job uses:

- sequence length: `4096`
- tensor parallelism: `4`
- data parallelism: `2`
- microbatch size per replica: `4`
- gradient accumulation: `16`
- optimizer steps: `8192`
- global chunks per optimizer step: `4 * 2 * 16 = 128`
- global tokens per optimizer step: `128 * 4096 = 524,288`
- total scheduled chunks: `8192 * 128 = 1,048,576`
- total scheduled tokens: `1,048,576 * 4096 = 4,294,967,296`

The p15 job configuration lists normal text first and corrected BranchProof-v2
logic or NL second, with normalized weights `[0.85, 0.15]`.

## Exact Realized Counts

Counts were recomputed with Nanotron's compiled
`helpers.build_blending_indices` implementation, the same helper called by
`BlendableDataset`:

| Prefix | Normal chunks | Proof chunks | Proof fraction | Proof tokens |
| ---: | ---: | ---: | ---: | ---: |
| First global batch (128 chunks) | 108 | 20 | 15.625000% | 81,920 |
| Checkpoint 4096 (524,288 chunks) | 445,644 | 78,644 | 15.000153% | 322,125,824 |
| Final step 8192 (1,048,576 chunks) | 891,289 | 157,287 | 15.000057% | 644,247,552 |

The final difference from an exact real-valued 15% allocation is 0.6 of one
4,096-token chunk. Logic and NL use the same weights, seed, and total schedule,
so their realized source counts are identical.

## Corpus Capacity

Pretokenized metadata reports:

| Corpus | Available packed tokens | Maximum production use | Wraparound |
| --- | ---: | ---: | --- |
| Normal continuation | 4,804,719,208 | 4,294,967,296 in control; 3,650,719,744 in p15 | No |
| Corrected logic | 1,200,313,814 | 644,247,552 | No |
| Corrected NL | 1,200,312,018 | 644,247,552 | No |

Thus neither the control nor either p15 arm repeats its packed stream during
the planned run.

## Resume Semantics

`BlendableDataset` builds a deterministic full-run index from weights and
seed. Its dataloader resumes at `consumed_train_samples`, while checkpoint
metadata stores `consumed_tokens_per_dataset_folder`. Consumption accounting
uses the absolute optimizer-step interval in that same full index. The normal
control checkpoint at step 4096 independently records exactly 524,288 chunks
and 2,147,483,648 normal tokens.

Consequently, the pre-submitted recovery jobs continue from the next absolute
chunk and preserve both source proportions and per-source offsets. They do not
restart the blend or replay the first half.

Operational update 2026-07-11 15:30 CEST: normal-control recovery
`3828946_0` resolved the complete step-4096 checkpoint and logged
`start_iteration_step: 4096`, `consumed_samples: 524288`, and exactly
`2147483648` consumed normal tokens. It then failed before its first resumed
optimizer step because W&B's local service did not publish its port file on
node `a0831`; no checkpoint or sampler state changed. Replacement control
recovery `3835438_0` disables W&B and excludes `a0831`. Untouched logic/NL
recoveries were proactively replaced by W&B-disabled `3835442_3/3835443_8`
with the same after-any parents, run roots, corpus overrides, and step-4096
checkpoint interval. Upload jobs `3831119/3831123/3831113` now depend on those
replacement recoveries.

## Scientific Interpretation

This is packed continuation midtraining, not whole-example SFT. A source proof
can cross a 4,096-token boundary, and a packed chunk can contain the end of one
record and the start of another. The experimental intervention is therefore
best described as adding 15% formal-proof or matched NL-proof *tokens* to a
normal continuation stream. The logic/NL comparison remains exposure-matched,
but claims should not imply that every optimizer example contains one intact
proof problem.

## Evidence

- Production wrapper:
  `scripts/slurm/jobs/nanotron_qwen25_midtrain_grid_2026-06-24.slurm`
- Nanotron dataset construction:
  `../nanotron/src/nanotron/data/tokenized_bytes.py`
- Exact blend helper and resume accounting:
  `../nanotron/src/nanotron/data/nemo_dataset/blendable_dataset.py` and
  `../nanotron/src/nanotron/data/nemo_dataset/helpers.cpp`
- Verified control metadata:
  `$HPCVAULT/synthetic-RLVL/nanotron_midtrain/qwen25_7b_midtrain_control_p0_4p3b/checkpoints/4096/checkpoint_metadata.json`
- Corrected corpus audit:
  `analysis/branchproof_unique_v2_corpus_audit_2026-07-10.json`
