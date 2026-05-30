# Experiment Backlog

Last updated: 2026-05-30 18:32 CEST.

This file is for planned work that is not yet running. Running jobs live in `docs/running_experiments.md`.

## Do After Current Active Jobs Finish

| Priority | Item | Why | Trigger |
| --- | --- | --- | --- |
| P0 | Analyze and report full paired-family suite | This is the real repeat of the HFSA logic-vs-NL comparison on `official_igsm`, `maze_navigation`, and hardened `attribute_constraints`. Deferred at the 2026-05-30 18:32 oversight pass because replacement SFT `3682411` is running, replacement eval `3682449` is dependency-pending, and there are zero eval JSONs. | `3682449` completes. |
| P0 | Add completed trace-control, hybrid, shortcut-kind, and conditioned-50k results to the LaTeX report | These are reviewer-facing ablations. Shortcut-rate `0.3`, wordified length-control, completed `shuffled_logic` trace-control, partial trace-control, partial hybrid-order, and active-artifact status are included. The 2026-05-30 14:36 regeneration added `shuffled_logic`; pending repair/pseudocode, hybrid, shortcut-kind, and conditioned-50k rows should be ingested when their JSONs appear. | Corresponding eval JSONs appear. |
| P0 | Inspect sample generations for each completed ablation | Tables alone are not enough to understand invalid/wrong failure modes. | After each eval family completes. |
| P1 | Improve paired-family NL validity translation | Current paired pilots often have meaningful correctness but `nl_exact` joint validity is `0.000` because translator coverage is incomplete. | Before making NL-vs-logic validity claims on paired families. |
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

- Plan-driven oversight is active through refreshed jobs: paired `3680777` is running with next pass `3682410` begin-time pending, and ablation oversight continues separately. Each pass should read/update this backlog, inspect sample generations and evaluator assumptions, analyze newly finished outputs, create justified plots/tables, regenerate/mirror the report when results change, and submit only the smallest safe triggered or recovery jobs.
- 2026-05-30 18:32 paired oversight recovered interrupted SFT rows by submitting targeted replacement `3682411_[55,57,59-89%6]`, canceling stale eval `3672213`, and submitting replacement eval `3682449_[0-89%4]`. No paired-family analysis trigger is satisfied yet because there are still zero paired full-suite eval JSONs.
- 2026-05-30 14:36 ablation oversight found one newly completed ablation slice: `shuffled_logic` now has 3 JSONs and was report-ingested. Remaining broad triggers are still deferred because trace repair `3680004`, original trace rows `3661119_12..17`, hybrid eval `3670783`, shortcut-kind eval `3674888`, and conditioned-dual 50k evals `3674884/3674885` are still running or dependency-pending.
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
