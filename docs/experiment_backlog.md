# Experiment Backlog

Last updated: 2026-05-31 10:31 CEST.

This file is for planned work that is not yet running. Running jobs live in `docs/running_experiments.md`.

## Do After Current Active Jobs Finish

| Priority | Item | Why | Trigger |
| --- | --- | --- | --- |
| P0 | Analyze and report full paired-family suite | This is the real repeat of the HFSA logic-vs-NL comparison on `official_igsm`, `maze_navigation`, and hardened `attribute_constraints`. Deferred at the 2026-05-31 10:31 paired oversight pass because original SFT rows `54/58`, replacement SFT rows `55/57/59/81/82/83`, row-56 replacement `3683070`, and replacement eval `3682449` are still active/dependency-pending, and there are zero eval JSONs. | `3682449` completes. |
| P0 | Add completed trace-control, hybrid, shortcut-kind, and conditioned-50k results to the LaTeX report | These are reviewer-facing ablations. Shortcut-rate `0.3`, wordified length-control, trace controls `17/18` (`terse_nl`, repaired `rule_annotated_nl`, `shuffled_logic`, `invalid_logic`, `shuffled_nl`, partial `pseudocode`), shortcut-kind controls `4/24`, hybrid `think_formal` through train-1-to-20 plus two-seed train-1-to-25, and active-artifact status are included. Remaining report triggers are trace row `3682460_8`, remaining hybrid replacements, remaining shortcut-kind eval rows, and conditioned-50k eval JSONs. | Corresponding eval JSONs appear. |
| P0 | Inspect sample generations for each completed ablation | Tables alone are not enough to understand invalid/wrong failure modes. | After each eval family completes. |
| P1 | Improve paired-family NL validity translation | Current paired pilots often have meaningful correctness but `nl_exact` joint validity is `0.000` because translator coverage is incomplete. A 2026-05-31 gold-target audit over sampled full-suite paired train/val rows also found paired NL targets answer and format correctly while `nl_logic_parse` and translated validity remain `0.0`. | Before making NL-vs-logic validity claims on paired families. |
| P1 | Build support-facts/context-QA reasoning eval | Current HotpotQA/2Wiki/MuSiQue are context-provided answer-only probes; they do not force or verify explicit multi-hop reasoning traces. | After current OOD tables are stable. |
| P1 | Decide whether conditioned dual is undertrained | Current 10k conditioned dual is weaker than single-modality logic. | After 50k checkpoint curves `3674885` complete. |

## Candidate Follow-Up Experiments

| Experiment | Current status | Notes |
| --- | --- | --- |
| Total-token-matched NL budget | Not running | Earlier "same-token" run matched target-token exposure only. Total prompt-plus-target matching would be about `8600` NL steps for train-1-to-25, but the wordified logic control is now the cleaner length-control test. |
| More shortcut mechanisms | Partly running | The two highest-value mechanisms, `position` and `initial_marker`, are already running. Additional shortcuts should wait until those results are interpreted. |
| Proof-only targets | Not running | Could separate premise-copy/translation overhead from proof construction, but lower priority than current ablations. |
| Grounded validity verifier | Not running | Needed only if we want grounded/canonical validity claims for HFSA; current citation-free validity is the stable metric. |
| Full finetune / non-LoRA SFT | Not running | Useful if reviewers argue LoRA-specific effects; lower priority while tiny scratch pretraining and architecture ablations already provide some substrate checks. |
| Larger pretraining / midtraining | Not running | The repo's tiny scratch trainer is a mechanism smoke path, not serious large-scale pretraining infrastructure. Use Nanotron or equivalent before any 50B-scale run. |
| More architecture repeats | Not running | Qwen-2.5-7B, Qwen-2.5-1.5B, Gemma-3-4B, and OLMo-2-32B short-context are already complete. Further architectures should wait for current ablation analysis. |

## Report/Artifact Work

- Plan-driven oversight is active through refreshed jobs: ablation `3683966` is running with next pass `3684370` begin-time pending after `3683563` completed, and paired `3683967` is running with next pass `3684369` begin-time pending after `3683562` completed. Each pass should read/update this backlog, inspect sample generations and evaluator assumptions, analyze newly finished outputs, create justified plots/tables, regenerate/mirror the report when results change, and submit only the smallest safe triggered or recovery jobs.
- 2026-05-30 18:40-22:39 ablation oversight recovered interrupted rows by submitting `3682457_[3,6-14%4]` plus later `3682492_[5%1]` for conditioned-dual 30k, `3682458_[22%1]` for shortcut-kind SFT, `3682459_[12,14-17%3]` for original trace-control eval, `3682460_[5-8%3]` for fixed-translator trace repair, and `3682461_[13,15-29%4]` for hybrid-order eval. `3682492_5` completed cleanly, so `3674882` now waits only on `afterok:3682457`; `3674888` now waits only on replacement row `3682458`.
- 2026-05-30 18:32 paired oversight recovered interrupted SFT rows by submitting targeted replacement `3682411_[55,57,59-89%6]`, canceling stale eval `3672213`, and submitting replacement eval `3682449_[0-89%4]`. No paired-family analysis trigger is satisfied yet because there are still zero paired full-suite eval JSONs.
- 2026-05-30 22:31 paired oversight canceled stuck original row `3672212_56`, submitted targeted replacement `3683070_[56%1]` with `--exclude=a0831`, and rewired eval `3682449` to depend on `afterok:3681398:3683070:3681586:3682411`. No paired-family analysis trigger is satisfied yet because there are still zero paired full-suite eval JSONs.
- 2026-05-31 02:29 paired oversight found replacement SFT rows `3682411_66..71` completed cleanly and paired final adapters at `66/90`; active paired rows are `3672212_54/58`, `3683070_56`, and `3682411_55/57/59/72/73/74`, with `3682411_75..89` pending by throttle. No new failure, partition edit, resubmission, eval output, aggregation, or report trigger is satisfied yet.
- 2026-05-31 02:38 paired oversight `3683024` completed cleanly after recording the 02:29 paired state and scheduling next pass `3683562`. It found no paired eval outputs, no new failures, and made no additional scheduler/report changes; the paired analysis trigger remains `3682449` completion.
- 2026-05-31 06:35 paired oversight found replacement SFT rows `3682411_72..77` completed cleanly and paired final adapters at `72/90`; active paired rows are `3672212_54/58`, `3683070_56`, and `3682411_55/57/59/78/79/80`, with `3682411_81..89` pending by throttle. Full-suite manifests remain complete, but there is still no paired eval output directory or JSON. Gold-target sample inspection re-confirmed matched logic/NL prompts and strict proof validation, while sampled paired NL targets still have zero NL-to-logic parse/translated validity, so the paired analysis trigger remains `3682449` completion plus translator caution for validity claims.
- 2026-05-31 10:31 paired oversight found replacement SFT rows `3682411_78..80` completed cleanly and paired final adapters at `75/90`; active paired rows are `3672212_54/58`, `3683070_56`, and `3682411_55/57/59/81/82/83`, with `3682411_84..89` pending by throttle. Full-suite manifests remain complete, but there is still no paired eval output directory or JSON. Gold-target sample inspection again found matched logic/NL prompts and strict proof validation, while sampled paired NL targets still have zero NL-to-logic parse/translated validity, so the paired analysis trigger remains `3682449` completion plus translator caution for validity claims.
- 2026-05-31 02:35 report regeneration included new trace-control outputs `invalid_logic` seed `3407`, repaired `rule_annotated_nl` seed `3409`, `pseudocode` seeds `3407/3408`, and `shuffled_nl` seed `3407`, bringing trace-control artifacts to `15/18`; it mirrored `64` PDFs plus `53` CSVs to `../synthetic-RLVL-report`. Sample inspection covered repaired rule annotations, pseudocode wrappers, shuffled-NL order failures, and invalid-logic grounding failures. Remaining broad triggers are still deferred because trace rows `3682459_16/17` and `3682460_8`, hybrid replacements, shortcut-kind eval `3674888`, and conditioned-dual 50k evals `3674884/3674885` are still running or dependency-pending.
- 2026-05-31 06:35 ablation/report refresh found trace rows `3682459_16/17` complete, bringing trace-control artifacts to `17/18`; only `3682460_8` remains. Shortcut-kind eval rows `3674888_0..3` completed and are report-ingested as the first `4/24` JSONs. Paired replacements `3682411_72..77` also completed, bringing paired SFT adapters to `72/90`, but paired eval output is still absent. Regenerated/mirrored the report with `64` PDFs and `55` CSVs, including new shortcut-kind tables; patched report-builder status prose and verified `py_compile`. No partition edit, dependency edit, cancellation, resubmission, or broad new science launch was made. Remaining triggers are still deferred to `3682460_8`, remaining hybrid/shortcut-kind eval rows, conditioned 50k evals, and paired eval `3682449`.
- Regenerate `analysis/logic_cot_report_2026-05-25/` after every newly completed eval family.
- Add convergence curves from `hfsa_conditioned_dual_50k_intermediate_20260529` once available.
- Shortcut-rate `0.3/0.5/0.8` matched logic/NL rows are complete and included in the report; revisit only if adding new rates or mechanisms.
- Add shortcut-kind comparison tables/plots once `hfsa_shortcut_kind_ablation_20260529` is complete.
- Wordified-vs-compact-vs-NL length-control tables/plots are now included; revisit only if adding another length-control strategy.
- Keep full raw generation examples in a supplemental Markdown file; keep only selected examples in the main LaTeX report.

## Not Currently Worth Starting

- Broad new paired-family variants before the current full paired suite finishes.
- More OOD reruns with the same answer-only LongBench prompting.
- More shortcut rates beyond `0.3/0.5/0.8` until the two shortcut-kind mechanisms are analyzed.
