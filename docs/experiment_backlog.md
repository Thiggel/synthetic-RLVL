# Experiment Backlog

Last updated: 2026-05-29 09:09 CEST.

This file is for planned work that is not yet running. Running jobs live in `docs/running_experiments.md`.

## Do After Current Active Jobs Finish

| Priority | Item | Why | Trigger |
| --- | --- | --- | --- |
| P0 | Analyze and report full paired-family suite | This is the real repeat of the HFSA logic-vs-NL comparison on `official_igsm`, `maze_navigation`, and hardened `attribute_constraints`. | `3672213` completes. |
| P0 | Add completed trace-control, hybrid, shortcut-rate `0.3`, shortcut-kind, wordified, and conditioned-50k results to the LaTeX report | These are reviewer-facing ablations. | Corresponding eval JSONs appear. |
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

- Regenerate `analysis/logic_cot_report_2026-05-25/` after every newly completed eval family.
- Add convergence curves from `hfsa_conditioned_dual_50k_intermediate_20260529` once available.
- Add shortcut-kind comparison tables/plots once `hfsa_shortcut_kind_ablation_20260529` is complete.
- Add wordified-vs-compact-vs-NL length-control plots once `hfsa_logic_wordified_20260529` is complete.
- Keep full raw generation examples in a supplemental Markdown file; keep only selected examples in the main LaTeX report.

## Not Currently Worth Starting

- Broad new paired-family variants before the current full paired suite finishes.
- More OOD reruns with the same answer-only LongBench prompting.
- More shortcut rates beyond `0.3/0.5/0.8` until the two shortcut-kind mechanisms are analyzed.
