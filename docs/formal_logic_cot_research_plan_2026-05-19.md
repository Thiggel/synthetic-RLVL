# Formal Logic CoT Research Plan - 2026-05-19

## Active Direction

We are pausing the RL-validity-reward direction and focusing on supervised / midtraining experiments.

Main question:

> Does reasoning in a formal-logic chain-of-thought substrate improve a language model's reasoning ability and length extrapolation compared with semantically matched natural-language CoT?

Correction 2026-07-14: the BranchProof/HFSA performance table and claims below
are historical hypotheses, not current evidence. A 2026-07-10 closure audit
found wrapped constants and multiple derivable answers above depth 17; a later
answer-matcher audit found embedded-alternative false positives. Every old
BranchProof performance and ablation result is quarantined. Clean exact-answer
replacement is `3853284 -> 3853285 -> 3853286`, and the corrected report-wide
matrix is tracked in `docs/branchproof_report_rerun_matrix_2026-07-13.md`.
Until those gates finish, AttrCon remains independent evidence and no old
BranchProof number below should be used in a scientific claim.

Update 2026-07-22: the corrected exact-answer baseline is complete and
accepted at `30/30`; aggregate/qualitative gate `3857769` passed. NL is
stronger at greedy/pass@1 for train maxima 5--20, while train-25 logic has a
large three-seed advantage: OOD logic-minus-NL greedy `+0.3333 +/- 0.2680`,
answer pass@1 `+0.5712 +/- 0.1399`, joint pass@1 `+0.5100 +/- 0.1443`, and
joint pass@16 `+0.5792 +/- 0.0457`. This replaces the historical table below
with a depth-dependent formal-advantage claim, not uniform superiority. The
accepted bundle is `analysis/branchproof_unique_v2_20260711/`; selected
corrected controls are `3881774..3881781`. The old table and interpretations
below remain historical only.

The current strongest signal is from the completed 3-seed sparse-protocol HFSA depth-scaling wave:

| train depths | logic OOD joint@16 | `nl_exact` OOD joint@16 | logic - NL |
| --- | ---: | ---: | ---: |
| `1..5` | 0.052 | 0.318 | -0.266 |
| `1..10` | 0.362 | 0.299 | +0.064 |
| `1..15` | 0.293 | 0.137 | +0.156 |
| `1..20` | 0.441 | 0.219 | +0.222 |
| `1..25` | 0.710 | 0.748 | -0.038 |

Interpretation: logic CoT is not automatically better at shallow train depth, but it is more depth/sample efficient in the intermediate ranges `1..10`, `1..15`, and `1..20`. At `1..25`, matched deterministic natural-language CoT catches up and slightly exceeds logic on sparse OOD joint@16/AUC, so the defensible claim is efficiency and scaling-shape, not strict asymptotic dominance.

## Claims We Can Aim To Support

Primary claim:

- Formal-logic CoT can be a more sample/depth-efficient supervised reasoning substrate than matched deterministic natural-language CoT for compositional length extrapolation.

Secondary claim:

- Formal-logic traces are more sample/compute efficient: they reach stronger OOD reasoning performance at equal or lower optimizer-step and token budgets.

Transfer claim, if real-benchmark results support it:

- Training or midtraining on formal-logic traces can improve downstream reasoning benchmarks more than matched natural-language traces or other structured-data controls.

Claims to avoid unless directly supported:

- Formal logic universally improves real-world reasoning.
- Validity-reward RL is the mechanism of improvement.
- The effect is independent of trace length, syntax compactness, or LoRA without controls.

## Current Dataset: HFSA Fixed-Target

The active controlled task is the fixed-target hard-FSA-schema dataset:

```bash
flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget-depth50
```

Design:

- Prompt is natural language in both conditions.
- The model must follow a finite-state automaton path.
- Each step has `K=4` locally plausible branches.
- Exactly one branch is reachable from the current state-marker pair.
- Wrong branches are coherent, so local plausibility is not enough.
- Gold proof length is `2D + 1` lines.
- The final proof conclusion is the queried final state atom.
- Eval is shortcut-neutral.

Trace comparison:

- `logic`: target contains `<formal>` with constants, predicates, formal premises, formal proof, conclusion, answer.
- `nl_exact`: target contains `<think>` with the same natural premises, a deterministic 1:1 natural-language proof, conclusion, answer.

Important length fact from a local OLMo tokenizer audit:

| depth | logic target tokens | nl_exact target tokens | logic proof-only tokens | nl proof-only tokens |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 458 | 598 | 58 | 45 |
| 10 | 820 | 1130 | 110 | 89 |
| 15 | 1197 | 1693 | 170 | 131 |
| 20 | 1578 | 2212 | 229 | 168 |
| 25 | 1915 | 2766 | 285 | 213 |
| 50 | 3509 | 5489 | 569 | 424 |

Full logic targets are shorter than full NL targets because symbolic premises are compact. Proof-only logic is slightly longer than proof-only NL. Both facts must be controlled for.

Update 2026-05-27: a training-range audit over 2048 examples per OLMo-7B train range confirms the same length direction under the actual SFT mixture. Mean target lengths for train `1..5/10/15/20/25` are logic `322/500/681/863/1049` tokens vs `nl_exact` `382/653/925/1196/1469` tokens, with zero sampled truncation at max length `8192`. The CSV artifact is `analysis/logic_cot_report_2026-05-25/tables/main_olmo7b_sft_token_lengths.csv`.

## Core Synthetic Experiment

Current active sweep:

| axis | values |
| --- | --- |
| model | `allenai/Olmo-3-1025-7B` |
| training | LoRA SFT |
| trace templates | `logic`, `nl_exact` |
| train depths | `1..5`, `1..10`, `1..15`, `1..20`, `1..25` |
| seeds | `3407`, `3408`, `3409` |
| SFT steps | 10,000 |
| eval depths | sparse final grid `{1,2,5,10,12,15,18,20,25,30,35,40,45,50}` |
| posthoc samples | 32 prompts/depth, 16 generations/prompt; separate greedy pass skipped |
| primary metrics | correct@1, correct@16, valid/joint metrics, OOD AUC |

Primary plots:

- Correct@1 vs eval depth.
- Correct@16 vs eval depth.
- Joint correct+valid vs eval depth.
- OOD AUC as a function of max train depth.
- Largest solved depth at thresholds 25%, 50%, 80%.
- Correct@1/correct@16 over training progress for selected checkpoints.

## Other Synthetic Families

The reviewer-facing broad repeat on non-HFSA paired datasets is now submitted. After fixing iGSM subtraction substitution and hardening saturated `attribute_constraints`, the full paired suite was launched on 2026-05-28:

| family | materialization | SFT/eval scope |
| --- | --- | --- |
| `official_igsm` | `3672195_0` | train ranges `1..5/10/15/20/25`, `logic` and `nl_exact`, seeds `3407/3408/3409`, sparse eval to depth 50 |
| `maze_navigation` | `3672195_1` | same full suite |
| hard `attribute_constraints` | `3672195_2` | same full suite, using the hardened generator under `attribute_constraints_hard` materialized roots |

Dependent arrays started as SFT `3672212_[0-89%6]` and sparse pass@k eval `3672213_[0-89%4]`, with repeated paired-suite oversight passes. As of 2026-06-02 06:37 CEST, original SFT rows `0..54` and `58` are complete, row `56` was canceled after idle-GPU/stale-log diagnosis and replaced by `3683070_[56%1]`, interrupted rows `55/57/59-89` were covered by replacement SFT `3682411_[55,57,59-89%6]`, and all replacements are complete, giving `90/90` final adapters (`official_igsm` `30/30`, `maze_navigation` `30/30`, hard `attribute_constraints` `30/30`). Stale eval `3672213` has been canceled, and replacement eval `3682449_[0-89%4]` has rows `0..29/33/34` complete, rows `30/31/32` timed out at 24h without final JSONs, rows `35..38` running, and rows `39..89` pending by array throttle. Targeted recovery eval `3691024_[30-32%3]` is running only the missing `maze_navigation` `logic` train-1-to-5 rows with `PASSK_MAX_NEW_TOKENS=4096`. The eval output root has `32/90` pass@k JSONs and sample JSONLs: all `30/30` `official_igsm` rows plus two `maze_navigation` `nl_exact` train-1-to-5 rows. Hard `attribute_constraints` still has no completed eval outputs, so full paired-family analysis remains deferred until the suite and recovery finish. Targeted iGSM NL rerun `3689003` is complete at `15/15`: generated iGSM NL parser coverage is now near-complete on OOD/depth-50 slices, but translated validity remains `0.000` because generated variable chains often do not match gold formal premises. The two completed maze NL rows have mean OOD correct@16 `0.148`, depth-50 correct@16 `0.000`, and maze NL parse/validity `0.000`; maze NL validity is unsupported by the current HFSA/iGSM translator. The first pending SFT/eval/oversight submissions `3672196`/`3672197`/`3672208` were canceled before start after fixing excessive startup staggering. Detailed operational status is in `docs/current_system_state.md`, and construction details are in `docs/paired_synthetic_benchmarks_2026-05-20.md`.

## Required Controls

### Length And Compute

Run these before making a strong paper claim:

| control | implementation | why |
| --- | --- | --- |
| same examples, same optimizer steps | current setup | base comparison |
| same target-token budget | choose per-template max steps from token audit | controls shorter full logic targets |
| padded-symbol logic at same sequence length | lengthen formal predicate/constant symbols while keeping proof semantics and optimizer steps fixed | controls whether logic advantage is just shorter symbolic traces |
| same train loss / same ID accuracy | evaluate intermediate checkpoints | tests whether NL simply converges slower |
| overtrained NL | train NL beyond 10k or to matched ID score | tests catch-up hypothesis |
| proof-only targets | prompt contains premises; target only proof/conclusion/answer | removes premise-copy/translation overhead |

Update 2026-05-29: the earlier completed `same_target_token_budget` run matched target-token exposure by reducing NL to 7140 optimizer steps. A more direct length-control was then added: `logic_symbol_padded` rewrites compact formal atoms like `Ba` into explicit longer symbols like `PB(ca)`, preserving the proof graph while matching `nl_exact` sequence lengths. On a 512-row train-1-to-25 OLMo audit, padded logic has target/total means `1443/2965` versus `1454/2976` for `nl_exact` and `1038/2560` for compact logic, with zero truncation at max length 8192. Three-seed SFT/eval jobs `3672286 -> 3672287` completed, but eval was worse than compact logic, likely because atom tokenization changed too much. The cleaner `logic_wordified` follow-up keeps constants compact and uses natural predicate names such as `Teal(a)` while preserving formal proof rules; its 512-row audit gives target/total means `1470/2991` versus `1454/2975` for `nl_exact`. SFT/eval `3674875 -> 3674876` completed on 2026-05-29. Wordified logic also underperforms compact logic and `nl_exact`: train-1-to-25 mean OOD correct/joint@16 is `0.508/0.323`, and depth-50 correct/joint@16 is `0.344/0.094`.

If NL catches up under overtraining or matched ID accuracy, the result is still meaningful but the claim becomes sample/compute efficiency rather than strictly better asymptotic extrapolation.

### Syntax And Representation

Run a small targeted control set, not a full factorial grid:

| control trace | purpose |
| --- | --- |
| `terse_nl` | deterministic compact NL, tests verbosity confound |
| `rule_annotated_nl` | NL proof lines plus explicit inference labels/citations, tests whether formal rule-application supervision rather than formal syntax drives the effect |
| `pseudocode` | structured non-FOL symbolic trace, tests rigid-notation advantage |
| invalid/shuffled logic negative control | tests whether valid derivation structure matters beyond formal-looking tokens |
| hybrid logic+NL | tests whether combining formal structure and natural semantics helps |

Existing code already supports hybrid templates (`formal_think`, `think_formal`, `logic_natural`, `natural_logic`). On 2026-05-25, targeted trace controls were implemented for `terse_nl`, `rule_annotated_nl`, `pseudocode`, `shuffled_logic`, `invalid_logic`, and `shuffled_nl`, with sparse pass@k jobs submitted as `3661118 -> 3661119`. Proof-only remains unsubmitted because the current priority is to answer the reviewer-facing verbosity/rule-label/notation/valid-structure confounds first. Current `nl_exact` traces are line/proposition aligned with formal proofs, but they verbalize the operation instead of preserving formal labels such as `->E`; `rule_annotated_nl` directly tests that gap.

## Real Benchmark Evaluation

Real benchmarks should be included, but not as the only evidence. The synthetic depth-scaling result should remain the causal core because it has paired trace supervision and controllable extrapolation.

Recommended benchmark suite:

| category | benchmarks | first-pass mode |
| --- | --- | --- |
| multi-hop QA | HotpotQA, 2WikiMultiHopQA, MuSiQue | context-provided/gold-context where possible, answer EM/F1 |
| math | GSM8K | answer normalization, pass@k/self-consistency |
| logical reasoning | FOLIO, ProofWriter, LogiQA/LogiQA2 | answer accuracy; proof validity where applicable |
| optional stress | AIME, GPQA-Diamond | exploratory only, not central |

Evaluation modes:

| mode | question answered |
| --- | --- |
| answer-only | did training improve representations independent of output format? |
| format-matched CoT | does the learned substrate help when the model reasons in that substrate? |
| pass@k / self-consistency | does formal training put correct real-benchmark answers in the sample distribution more often? |
| valid-trace diagnostics | for logic outputs, are correct answers supported by valid formal traces? |

Important design choice: for HotpotQA/2Wiki/MuSiQue, use context-provided evaluation first to isolate reasoning from retrieval/world-knowledge confounds. Closed-book evaluation can be an appendix.

Implementation update 2026-05-25: the first-pass OOD suite is implemented through lm-eval local tasks in `lm_eval_tasks/synthrlvl_ood/` and the repo wrapper `scripts/evaluate_lm_eval.py --suite synthrlvl_ood`. The tasks are:

| task | source/eval shape | scoring note |
| --- | --- | --- |
| `synthrlvl_gsm8k_tagged` | GSM8K | extracts explicit `<answer>...</answer>` or answer-marker text, then numeric-normalizes; no raw-trace number fallback |
| `synthrlvl_longbench_hotpotqa_tagged` | LongBench HotpotQA context-provided QA | strict answer extraction before QA F1 |
| `synthrlvl_longbench_2wikimqa_tagged` | LongBench 2WikiMultiHopQA context-provided QA | strict answer extraction before QA F1 |
| `synthrlvl_longbench_musique_tagged` | LongBench MuSiQue context-provided QA | strict answer extraction before QA F1 |

Strict extraction is intentional for the context-provided QA tasks: if a model copies passage text in `<think>`/`<premises>` without a final `<answer>` tag or explicit answer marker, the prediction is scored as empty. This avoids false F1 credit from answer strings appearing in the supplied context.

Pilot jobs `3659344` and strict rerun `3659348` completed on two Qwen-1.5B train-1-to-10 checkpoints. The extractor correctly pulled learned `<answer>` tags in sample generations, and the strict rerun removed accidental context-copy credit. Broad OOD eval `3659356_[0-89%4]` for non-OLMo-32B 1-GPU models completed all 90 result JSONs by 2026-05-27. A 2026-05-27 GSM8K scorer fix removed fallback to arbitrary raw-trace numbers; strict recomputed GSM8K metrics live in `analysis/logic_cot_report_2026-05-25/tables/ood_gsm8k_strict_recompute_from_samples.csv`. Main OLMo-7B logic train ranges `1..5/10/15/20/25` have strict GSM8K EM `0.051/0.072/0.070/0.079/0.049` and mean LongBench F1 `0.395/0.391/0.383/0.408/0.404`; matched `nl_exact` has strict GSM8K EM `0.491/0.478/0.322/0.409/0.256` and mean LongBench F1 `0.171/0.179/0.214/0.185/0.145`. Paired sample inspection suggests the GSM8K gap is task/manifold mismatch rather than only extraction: logic often tries to formalize arithmetic word problems without a useful arithmetic schema, while NL keeps a natural-language reasoning scaffold. OLMo-32B full OOD array `3659357_[0-1%1]` failed because LongBench contexts exceed OLMo-2 32B's 4096-position limit; replacement `3660240_[0-1%1]` completed GSM8K-only short-context, but its stored JSON metrics predate the strict GSM8K fallback fix and should be recomputed before citation. Tiny scratch checkpoints are complete for all three seeds; GSM8K EM is near-zero and strict LongBench F1/EM is `0.000` across size/template groups. OLMo-7B UltraChat instruction-tuning control SFT `3666639_0` and OOD eval `3666640_0` completed: GSM8K EM `0.755`, Hotpot EM/F1 `0.050/0.343`, 2Wiki `0.005/0.207`, MuSiQue `0.010/0.195`. This confirms the instruction-control is much better than synthetic logic/NL on GSM8K but not competitive with logic on strict context-QA F1.

Clarification 2026-05-27: the current LongBench HotpotQA/2Wiki/MuSiQue tasks are context-provided, so there is no external retrieval component, but they are still long-context answer-only prompts over full passages and do not ask either model to produce a logic or NL reasoning chain. Sample generations are often degenerate or answer-only. These numbers should not be framed as evidence of explicit downstream multi-hop reasoning. The next OOD eval should add gold-supporting-facts/facts-only variants and format-matched CoT prompts, then run short pilots with sample inspection before broad submission.

Correction 2026-07-13: the Qwen2.5 Nanotron transfer pilot now evaluates both
the stock LongBench short-answer protocol and a strict tagged protocol. The
first smoke is rejected because its 8192-token model window left-truncated all
six tested prompts and its converted HF config was read with an incorrect RoPE
base by the Transformers-4 downstream stack. Across the complete 200-example
HotpotQA/2Wiki/MuSiQue splits, Qwen-tokenized prompt maxima are
`17684/17079/17927`; corrected evaluation therefore fixes the model window at
32768 and audits that value. Converted configs must expose Qwen2.5's
`rope_theta=1000000` to both Transformers 5 and 4.57. No full transfer result
is accepted until corrected control smokes produce coherent raw generations;
then control, logic-p15, and NL-p15 are run in direct and native-chat modes.

Correction 2026-07-14: the 32,768-window smokes exposed a second independent
protocol defect. `Xnhyacinth/LongBench` embeds the stock instruction prefix and
suffix in `context`, and its `question` field already starts with `Question:`;
the local tagged and standard templates wrapped both again. The shared document
renderers now strip exactly those known wrappers, retain the complete passage
body, and emit one normalized question. Tagged decoding is capped at 64 tokens,
while stock decoding remains at 32. Prompt-retaining audits reject duplicate
wrappers, duplicate question prefixes, and stale caps. Prompt-fixed control
smokes `3855269/3855270` gate full direct/instruction control, logic-p15, and
NL-p15 arrays `3855271/3855272`; aggregate `3855273` produces the matched table.

Smoke-gate update 2026-07-14 13:13 CEST: direct smoke `3855269` passed.
Instruction smoke `3855270` completed the intended run but exposed an audit
false negative because Qwen chat tokens precede the stock user prompt. The
audit now validates the extracted single user turn; stored artifacts and
CPU-only gate `3856131` pass, and `3855272` depends on that gate. All retained
prompts have one wrapper/question and the intended caps. Raw instruction
generations remain weak but bounded, so the full comparison must report tag
adherence, explanations, and repetition rather than hiding those failures.

Corrected p15 update 2026-07-14 19:26 CEST: replacement logic instruction eval
`3854824` and strict six-bundle aggregate `3854847` completed and passed their
artifact gates. The result is null/mixed: direct logic changes all-primary and
reasoning macros by only `+0.0033/+0.0071` versus control and reduces targeted
logic by `-0.0116`; NL and post-instruction deltas are similarly small. Raw
review found increased direct next-document continuation for both proof
mixtures, long post-instruction repetition, and a BBH instruction extraction
floor. The broader Nanotron mixture grid is therefore not triggered. This is a
one-run negative/mixed pilot with evaluator caveats, not evidence for transfer.

Corrected multi-hop update 2026-07-15 07:06 CEST: prompt-fixed direct
`3855271`, instruction `3855272`, and aggregate `3855273` completed with all
six 1,200-row bundles accepted under `rope_theta=1000000`, a 32,768-token
window, and retained prompt/cap coverage. Direct stock control/logic/NL QA-F1
is `0.189/0.250/0.238`, but an answer-head sensitivity rescore is
`0.349/0.361/0.367`, so most of the apparent mixture gain is shorter or less
contaminating continuation. The direct tagged prompt triggers `<formal>` in
`98.5--99.0%` of logic rows and `<think>` in `97.0--99.0%` of NL rows, usually
reaching the 64-token cap before an answer. Instruction SFT removes those
substrate openings, while stock QA-F1 remains near `0.09--0.10` and almost all
32-token responses cap. This bounded result diagnoses response control; it is
not evidence of multi-hop reasoning transfer and does not reopen the broader
mixture grid.

Format-matched OOD pilot update 2026-05-27 13:15 CEST: added `synthrlvl_ood_cot_bare` and `synthrlvl_ood_cot_prompted` suites. The bare suite removes answer-only instructions and leaves all task content inside `<question>...</question>`, relying on the model's learned output manifold; the prompted suite adds a minimal request to reason in the learned format before `<answer>`. LongBench context cleaning strips the embedded "only give me the answer" prefix. Short pilot `3667055_[0-3%2]` compares bare vs prompted on the matched OLMo-7B `logic_train1to25_seed3407` and `nl_exact_train1to25_seed3407` checkpoints with `LM_EVAL_LIMIT=8` before any broad rerun.

Pilot readout 2026-05-27 13:36 CEST: all four rows completed exit `0:0`. Prompted format improves LongBench answer-tag adherence for NL and improves several tiny-sample LongBench EM cells, but LongBench samples still often look like direct entity extraction or long unclosed NL traces rather than explicit chain reasoning. Treat this as evidence that prompt format matters, not yet as a valid downstream multi-hop reasoning result; the next useful OOD step is a gold-supporting-facts/facts-only controlled suite.

Full bare-format rerun 2026-05-27 14:04 CEST: after inspecting samples, the user preferred the bare setting, so full `synthrlvl_ood_cot_bare` reruns were submitted. `3667168_[0-90%3]` covers the previous 90 non-tiny OOD rows plus the UltraChat instruction-control row; `3667167_[0-17%3]` and `3667169_[0-17%3]` cover tiny 20k and 100k checkpoints; `3667166_[0-1%1]` covers OLMo-32B GSM8K only. Full LongBench is intentionally not submitted for OLMo-2 32B because its real 4096-position limit already invalidated full LongBench runs.

Bare-format readout 2026-05-28 10:59 CEST: the main 30-row OLMo-7B slice is complete. Logic train ranges `1..5/10/15/20/25` get GSM8K EM `0.046/0.043/0.064/0.066/0.025` and mean LongBench F1 about `0.404/0.412/0.411/0.416/0.407`; matched `nl_exact` gets GSM8K EM `0.369/0.341/0.287/0.277/0.242` and mean LongBench F1 about `0.254/0.261/0.235/0.263/0.114`. Tiny 20k and tiny 100k bare reruns completed with strict EM/F1 `0.000` across GSM8K and LongBench; tag adherence varies but does not translate into correctness. OLMo-32B GSM8K-only bare completed for both templates: logic EM `0.2335`, NL EM `0.6755`. Prompt sanity check: the LongBench bare prompt includes passages and question inside `<question>` tags; it does not insert `Gold:` or a gold-answer label, and `doc_to_target: "{{answers}}"` is used only for scoring. The answer string can appear naturally inside the context passage. The report now includes bare-format OOD tables and direct sample generations in `analysis/logic_cot_report_2026-05-25/logic_cot_report_2026-05-25.tex`; untruncated OLMo-7B/OLMo-32B raw sequence supplements are in `analysis/logic_cot_report_2026-05-25/full_generation_sequences_olmo7b_olmo32b_2026-05-28.md`.

## Midtraining / Data-Mixture Experiments

This is the second story layer. It should be framed differently from the SFT substrate comparison.

Hypothesis:

- Seeing formal reasoning traces during continued pretraining/midtraining induces representations that transfer to downstream reasoning tasks better than seeing matched NL traces or generic structured data.

Mixtures:

| mixture | meaning | control purpose |
| --- | --- | --- |
| plain text only | continue on generic text for the same token budget | controls extra training |
| text + NL traces | generic text plus matched NL reasoning traces | controls reasoning-data exposure |
| text + logic traces | generic text plus matched formal traces | main intervention |
| text + hybrid traces | generic text plus both formal and NL traces | tests complementarity |
| text + code | generic text plus code data at matched token budget | tests whether logic is better than generic structured tokens |
| synthetic-only curriculum | only synthetic reasoning traces | clean algorithmic prior test, weaker real-transfer claim |

For real-world trace data such as OpenThoughts, keep this as a follow-up layer unless we can generate high-quality verified formal traces. A clean synthetic paper is easier to defend than a large, noisy distilled-trace paper.

## Model-Regime Controls

The current sweep is LoRA SFT. That is efficient but a reviewer will ask whether the effect is LoRA-specific.

Recommended staged controls:

| regime | scope | purpose |
| --- | --- | --- |
| OLMo-7B LoRA | full current sweep | main result |
| OLMo-7B full finetune | small subset only | rule out adapter bottleneck |
| Qwen 7B-class LoRA | representative subset | rule out OLMo-specific behavior without repeating the full 30-row grid |
| OLMo-3-32B LoRA/full where feasible | smallest useful subset only | architecture/scale ablation after a resource script is validated |
| 50M-200M pretraining | from scratch or continued pretraining | mechanism/scaling evidence |

Do not make small-model pretraining the initial main result; use it as mechanism/scaling evidence after the 7B LoRA trend is confirmed. For model-family ablations, start with a representative subset such as train depths `1..10`, `1..20`, and `1..25`, both templates, and seeds `3407..3409`; repeating the full factorial grid on every architecture is not the first move.

Submitted follow-up wave on 2026-05-24:

| regime | jobs | note |
| --- | --- | --- |
| Qwen 7B-class LoRA | `3656217` SFT, `3656218` eval | `Qwen/Qwen2.5-7B`, representative train-depth/template/seed grid |
| Qwen/Gemma smaller LoRA | `3656323` SFT, retries `3656359` and `3656387`, eval `3656389`, Gemma eval replacement `3665578` | `Qwen/Qwen2.5-1.5B` because exact `Qwen/Qwen2.5-1B` was not available; `google/gemma-3-4b-pt` for the requested Gemma-4B-style ablation. Qwen-1.5B and Gemma pass@k are complete; Gemma eval required the processor-metadata merge fix |
| OLMo-32B pilot | `3656335` SFT, original `3656336` eval, failed replacement `3658461`, short-context replacement `3660238` eval | `allenai/OLMo-2-0325-32B`; exact OLMo-3 base 32B was not available, while the checked OLMo-3.1 32B model is a Think checkpoint. OLMo-2 32B enforces a 4096-position limit, so this is now a short-context pilot only |
| 50M-200M scratch pretraining | `3656338` pretrain, retries `3656360` and `3656388`, eval `3656390` | random-init Llama configs with a Llama3 tokenizer; this does not cover a true 50B run |

Rows that hit `NODE_FAIL` were retried rather than treated as scientific failures. The second retries exclude node `a0934`, where the repeated immediate node failures occurred. The current live operational state and dependency rewiring are in `docs/current_system_state.md`.

Ablation submission update 2026-05-25 19:01 CEST:

- Trace controls: SFT array `3661118_[0-17%3]` and eval array `3661119_[0-17%3]` cover `terse_nl`, `rule_annotated_nl`, `pseudocode`, `shuffled_logic`, `invalid_logic`, and `shuffled_nl`, each with seeds `3407..3409`, train depths `1..25`, and the sparse depth-50 eval protocol.
- Same target-token budget: SFT array `3661120_[0-5%3]` and eval array `3661121_[0-5%3]` cover `logic` at 10k steps and `nl_exact` at 7140 steps. The step ratio comes from a 512-row OLMo tokenizer audit on train-1..25 targets: mean logic target length `1038` tokens and mean `nl_exact` target length `1454` tokens.
- Shortcut robustness: initial build `3661122_[0-1%1]` failed in the probe because the old shortcut schema exhausted high-depth state-word/predicate capacity; dependents `3661123`/`3661124` were canceled. After expanding the schema word banks and enabling extended predicate rendering, local probes for shortcut rates `0.5` and `0.8` passed and replacement build `3661135_[0-1%1]` completed. Original SFT rows `3661136_0..2` OOMed under `train.gradient_checkpointing=auto`, so the wrapper now defaults shortcut SFT to checkpointing on; replacement SFT `3662743_[0-2,6-8%3]` covers failed rows and at-risk logic shortcut-0.8 rows, and eval `3661137` was replaced by `3662744`. The existing main grid is the shortcut-rate `0.0` baseline. On 2026-05-28, rate-list support was added to the shortcut wrappers and shortcut-rate `0.3` dose-response jobs were submitted as build `3671430`, SFT `3671431_[0-5%3]`, and eval `3671432_[0-5%3]`; all `0.3/0.5/0.8` logic and NL eval rows are complete and report-ingested as of 2026-05-30.
- Shortcut-kind robustness: submitted on 2026-05-29 as build `3674886_[0-3%2]`, SFT `3674887_[0-23%3]`, and eval `3674888_[0-23%4]`. The two new mechanisms are `position` (gold branch first on shortcut-enabled training examples) and `initial_marker` (gold path initial marker fixed to `north`), both at rates `0.5` and `0.8`, with shortcut-neutral eval. Build and SFT are complete after replacement SFT `3682458_22`; as of 2026-06-01 14:42 CEST, eval rows `3674888_0..23` are complete and all `24/24` JSONs are report-ingested. Final three-seed means: `position` rate `0.5` logic OOD correct/joint@16 `0.900/0.619`, depth-50 `0.844/0.312`; `position` rate `0.8` logic OOD `0.879/0.650`, depth-50 `0.760/0.323`; matched `nl_exact` rate `0.5` OOD `0.540/0.431`, depth-50 `0.396/0.260`; matched `nl_exact` rate `0.8` OOD `0.512/0.487`, depth-50 `0.396/0.354`; `initial_marker` logic rates `0.5/0.8` OOD `0.883/0.625` and `0.885/0.610`, depth-50 `0.854/0.344` and `0.865/0.344`; `initial_marker` `nl_exact` rates `0.5/0.8` OOD correct/translated-joint@16 `0.469/0.421` and `0.771/0.702`, depth-50 `0.115/0.094` and `0.667/0.500`.

Ablation readout 2026-05-28 10:01 CEST:

- Same target-token budget is complete. Logic train-1-to-25 at 10k steps has OOD correct/joint@16 `0.898/0.335` and depth-50 `0.792/0.125`; token-matched NL at 7140 steps has OOD `0.554/0.473` and depth-50 `0.344/0.219`. Logic keeps the answer-correctness advantage at matched target-token budget, while NL has higher joint validity.
- Shortcut rate 0.5 eval is complete and shortcut rate 0.8 eval is running. On shortcut-neutral eval, shortcut-0.5 logic gets OOD correct/joint@16 `0.906/0.677` and depth-50 `0.833/0.375`; shortcut-0.5 NL gets OOD `0.642/0.585` and depth-50 `0.385/0.312`. Shortcut-0.8 currently has two logic seeds complete with OOD `0.953/0.788` and depth-50 `0.875/0.422`; do not interpret until the third logic seed and all three NL seeds complete.
- Conditioned dual-modality eval is partial (22/30 JSONs). Early rows are report-tabulated but not yet interpreted.
- Trace-control SFT is still running/pending; eval remains dependency-pending.
- Hybrid-order SFT exposed a second memory issue: train-1-to-20 rows `9..11` and `24..26` OOMed because auto checkpointing only enabled at train-1-to-25. The hybrid wrapper now defaults to gradient checkpointing and expandable CUDA segments; stale eval `3666425` was canceled, replacement SFT `3670782_[9-11,24-26%3]` and eval `3670783_[0-29%4]` were submitted.

Ablation oversight readout 2026-05-30 09:41 CEST:

- Shortcut-rate `0.3` is fully matched. Logic mean OOD correct/joint@16 is `0.892/0.598` and depth-50 correct/joint@16 is `0.844/0.375`; matched NL mean OOD correct/translated-joint@16 is `0.588/0.571` and depth-50 correct/translated-joint@16 is `0.458/0.438`. Across rates `0.3/0.5/0.8`, NL depth-50 joint falls `0.438 -> 0.312 -> 0.146`, while logic depth-50 joint is `0.375 -> 0.375 -> 0.417`.
- Trace-control `terse_nl` is three-seed complete with mean OOD correct/translated-joint@16 `0.348/0.277` and depth-50 correct/translated-joint@16 `0.094/0.010`. `rule_annotated_nl` is now three-seed complete with OOD correct/translated-joint@16 `0.579/0.000` and depth-50 correct/translated-joint@16 `0.365/0.000`, so it gets answers but fails the translated-validity check. `pseudocode` eval rows `6..8` are running.
- Hybrid `think_formal` train-1-to-15 is three-seed complete with mean OOD correct/formal-joint/translated-joint@16 `0.353/0.111/0.111` and depth-50 correct/formal-joint@16 `0.312/0.000`. Train-1-to-20 has seeds `3407/3408` complete with OOD correct/formal-joint/translated-joint@16 `0.419/0.016/0.078` and depth-50 correct/formal-joint@16 `0.594/0.000`; treat train-1-to-20 and train-1-to-25 as partial until the running rows finish.
- Oversight correction 2026-05-30 10:33 CEST: sample inspection confirmed `think_formal` is the NL-then-formal template, matching the Slurm README and submission note. The report builder was patched to label `think_formal` as NL then formal and to parse pending `formal_think` JSONs instead of the nonexistent `think_natural` key. Verification: `py_compile` and `scripts/analysis/build_logic_cot_report.py` passed, and the report bundle was mirrored with 64 PDF references and zero missing figure references.
- Ablation update 2026-05-30 14:36 CEST: `shuffled_logic` trace-control eval rows `3661119_9..11` are three-seed complete and report-ingested. Mean OOD correct/formal-joint@16 is `0.690/0.002`, and depth-50 correct/formal-joint@16 is `0.510/0.000`. Sample generations keep the formal surface and answer tags, but higher-depth proofs are often invalid or unparsable, so this supports the negative-control interpretation: answer correctness can persist without valid formal derivations.
- Ablation update 2026-05-30 18:47 CEST: later trace/hybrid/conditioned/shortcut-kind rows were interrupted without a code/config signature and have targeted replacement arrays, including conditioned-dual row-5 replacement `3682492_[5%1]` after the final dependency check. The report now filters the stale pre-fix `rule_annotated_nl` seed-3409 artifact until repair overwrite; repaired seeds 3407/3408 show translated joint is nonzero, so do not use the old all-zero rule-annotated row. Hybrid `think_formal` train-1-to-20 is three-seed complete with mean OOD correct/formal-joint/translated-joint@16 `0.434/0.028/0.148` and depth-50 correct/formal-joint@16 `0.469/0.000`; sample inspection confirms the intended NL-then-formal surface but weak depth-50 validity.
- Ablation update 2026-05-30 22:54 CEST: conditioned-dual row-5 replacement `3682492_5` completed, so the 40k chunk now waits only on `3682457`. Trace-control `invalid_logic` seeds `3408/3409` and hybrid `think_formal` train-1-to-25 seed `3409` are report-ingested. `invalid_logic` is a two-seed partial with OOD correct/formal-joint@16 `0.906/0.544` and depth-50 `0.734/0.188`, but seed-3409 samples show the distinction between citation-free validity and grounded validity: shallow traces are often internally citation-free-valid while grounded validity is zero, and depth-50 sampled rows have zero grounded validity. Treat this as evaluator-sensitive negative-control evidence until the remaining trace rows finish. Hybrid `think_formal` train-1-to-25 remains two-seed partial with OOD correct/formal-joint/translated-joint@16 `0.584/0.188/0.459` and depth-50 `0.344/0.000/0.172`; samples preserve the intended NL-then-formal surface but formal validity remains weak.
- Ablation update 2026-05-31 02:35 CEST: trace-control artifacts are now `15/18` after completing `invalid_logic` seed `3407`, repaired `rule_annotated_nl` seed `3409`, `pseudocode` seeds `3407/3408`, and `shuffled_nl` seed `3407`. Repaired `rule_annotated_nl` is three-seed with OOD correct/translated-joint@16 `0.575/0.485` and depth-50 `0.344/0.146`; `[rule: ...]` traces now translate. `invalid_logic` is three-seed with high answer accuracy but zero grounded validity in inspected samples, so it remains a negative-control/evaluator-sanity result rather than evidence for invalid proof reasoning. `shuffled_nl` parses but has translated joint `0.000`, consistent with the intended proof-order control. Remaining trace rows are `pseudocode` seed `3409` and `shuffled_nl` seeds `3408/3409`.
- Ablation update 2026-05-31 06:35 CEST: trace-control artifacts are now `17/18` after completing `shuffled_nl` seeds `3408/3409`; the only remaining trace row is `pseudocode` seed `3409` (`3682460_8`). `shuffled_nl` is three-seed with OOD correct/translated-joint@16 `0.490/0.000` and depth-50 `0.344/0.000`; samples keep parseable NL proof surfaces while failing translated validity because order is wrong. Shortcut-kind has its first `4/24` eval JSONs as described above. The report builder now reports dynamic trace/shortcut artifact counts, and the report was regenerated/mirrored with `64` PDFs and `55` CSVs; local TeX compilation remains unavailable.
- Ablation update 2026-05-31 10:45 CEST: trace-control artifacts are now complete at `18/18`; `pseudocode` is three-seed with OOD correct/translated-joint@16 `0.544/0.479` and depth-50 `0.208/0.104`. Shortcut-kind has `7/24` eval JSONs as described above. Hybrid `think_formal` is now three-seed complete through train-1-to-25 with OOD correct/formal-joint/translated-joint@16 `0.573/0.204/0.419` and depth-50 `0.344/0.000/0.135`; pending `formal_think` rows are still running or throttle-pending. Conditioned-dual 40k chunk `3674882` has rows `0/1/2` complete, `3/4/5/6` running, and `7..14` throttle-pending. The report was regenerated/mirrored with `64` PDFs and `55` CSVs; local TeX compilation remains unavailable.
- Ablation update 2026-05-31 14:38 CEST: shortcut-kind has `9/24` eval JSONs as described above after rows `5` and `8` completed. Trace-control remains complete at `18/18`, hybrid order remains `15/30`, and conditioned-dual 40k chunk `3674882` has rows `0..5` and `7` complete, `6/8/9/10` running, and `11..14` throttle-pending. New shortcut-kind sample inspection found shortcut-neutral prompts, intended wrappers, normal `<answer>` extraction, and expected depth-50 validity/grounding fragility. The report was regenerated/mirrored with `64` PDFs and `55` CSVs; local TeX compilation remains unavailable.
- Ablation update 2026-05-31 18:35 CEST: shortcut-kind has `13/24` eval JSONs as described above after rows `9..12` completed. Trace-control remains complete at `18/18`, hybrid order advanced to `18/30` after `formal_think` train-1-to-5 completed, and conditioned-dual 40k chunk `3674882` has rows `0..8` complete with `9/10/11/12` running. The first completed `formal_think` slice has OOD correct/formal-joint/translated-joint@16 `0.538/0.120/0.347` and depth-50 `0.323/0.010/0.000`. New sample inspection found shortcut-neutral eval prompts, intended wrappers, normal `<answer>` extraction, valid shallow samples, and expected depth-50 truncation/validity fragility. The report was regenerated/mirrored with `64` PDFs and `55` CSVs; local TeX compilation remains unavailable.
- Ablation update 2026-05-31 22:35 CEST: shortcut-kind has `15/24` eval JSONs after rows `13/14` completed. `initial_marker` logic rate `0.5` is now three-seed with OOD correct/joint@16 `0.883/0.625` and depth-50 `0.854/0.344`; sample inspection found shortcut-neutral prompts, intended `<formal>` wrappers, normal answer extraction except one deeper truncated failure, and expected validity/grounding fragility. Conditioned-dual 40k rows `0..11` are complete, rows `12/13/14` are running with high GPU utilization, and final/checkpoint eval JSONs still do not exist. Hybrid remains `18/30`, with rows `18..21` running. The report was regenerated and mirrored with `64` PDFs and `55` CSVs; local TeX compilation remains unavailable.
- Ablation update 2026-06-01 02:45 CEST: shortcut-kind has `17/24` eval JSONs after rows `15/16` completed. `initial_marker` `nl_exact` rate `0.5` is now two-seed with OOD correct/translated-joint@16 `0.509/0.481`, depth-50 `0.125/0.109`, and OOD parse@16 `0.991`; inspected samples keep shortcut-neutral prompts, intended `<think>`/`<answer>` formatting, and normal answer extraction, but depth-50 remains fragile. Conditioned-dual 40k is complete and 50k chunk `3674883` has rows `0..3` running with high GPU utilization; final/checkpoint eval JSONs still do not exist. Hybrid remains `18/30`, with rows `18..21` running. The report was regenerated and mirrored with `65` PDFs, `57` CSVs, and `5` Markdown supplements; local TeX compilation remains unavailable.
- Ablation update 2026-06-01 10:40 CEST: shortcut-kind has `23/24` eval JSONs after rows `21/22` completed. `initial_marker` `nl_exact` rate `0.8` is two-seed with OOD correct/translated-joint@16 `0.675/0.572`, depth-50 `0.594/0.344`, and inspected samples preserve shortcut-neutral prompts, intended `<think>/<answer>` wrappers, and working answer extraction, with depth-50 truncation/validity fragility. Hybrid-order has `21/30` eval JSONs; `formal_think` train-1-to-10 is now three-seed with OOD correct/formal-joint/translated-joint@16 `0.626/0.279/0.381` and depth-50 `0.396/0.000/0.000`, with samples confirming the intended formal-then-NL surface. Conditioned-dual 50k has rows `0..8` complete, rows `9..12` running, and final/checkpoint eval JSONs still absent. The report was regenerated and mirrored with `65` PDFs, `57` CSVs, and `5` Markdown supplements; local TeX compilation remains unavailable. No scheduler edit, partition edit, cancellation, resubmission, broad launch, or fix was made.
- Ablation update 2026-06-01 14:42 CEST: shortcut-kind is complete at `24/24`; final `initial_marker` `nl_exact` rate `0.8` three-seed means are OOD correct/translated-joint@16 `0.771/0.702` and depth-50 `0.667/0.500`. Hybrid-order has `22/30` eval JSONs after `3682461_21`; `formal_think` train-1-to-15 seed `3407` has OOD correct/formal-joint/translated-joint@16 `0.656/0.301/0.250` and depth-50 `0.688/0.125/0.000`, with samples confirming the intended formal-then-NL surface and expected deep validity fragility. Conditioned-dual 50k has rows `0..11` complete and rows `12..14` running; final/checkpoint eval JSONs remain absent. The report was regenerated and mirrored with `65` PDFs, `57` CSVs, and `5` Markdown supplements; local TeX compilation remains unavailable. No scheduler edit, partition edit, cancellation, resubmission, broad launch, or fix was made.

- Ablation/paired update 2026-06-02 06:37 CEST: conditioned-dual 50k SFT `3674883` is complete, final eval has `4/30` JSONs, and checkpoint eval has `21/30` provisional train-1-to-25 JSONs. Paired eval has `32/90` JSONs after the second `maze_navigation` `nl_exact` row completed; `3682449_30/31/32` timed out at 24h without final JSONs under the 8192-token cap, and targeted recovery `3691024_[30-32%3]` is running with `PASSK_MAX_NEW_TOKENS=4096`. Targeted iGSM NL rerun `3689003` completed all `15/15` rows and now has near-complete OOD/depth-50 parser coverage, but generated translated validity remains zero due variable-chain mismatch. The two completed maze NL rows average OOD correct@16 `0.148`, NL parse@16 `0.000`, and depth-50 correct@16 `0.000`; maze NL validity remains unsupported by the current translator. The report was regenerated and mirrored with `66` PDFs, `61` CSVs, and `5` Markdown supplements; local TeX compilation remains unavailable. No partition edit, dependency edit, cancellation, broad launch, generator fix, or evaluator fix was made.

- Ablation update 2026-06-02 02:48 CEST: hybrid-order has `24/30` eval JSONs after `formal_think` train-1-to-15 became three-seed complete. That slice averages OOD correct/formal-joint/translated-joint@16 `0.568/0.258/0.264` and depth-50 `0.479/0.083/0.000`; inspected depth-50 failures are mainly truncation/repetition and validity fragility. Conditioned-dual 50k final eval has `2/30` JSONs and checkpoint eval has `15/30` JSONs, completing the `conditioned_logic` train-1-to-25 curve through 50k. The logic checkpoint curve peaks at 20k on OOD joint (`0.451`) while depth-50 joint remains low (`0.125` at 40k/50k); `conditioned_nl` checkpoint rows are still running, so the undertraining decision remains deferred. The report was regenerated/mirrored with `66` PDFs, `61` CSVs, and `5` Markdown supplements after a caption/status-text fix; TeX compilation remains unavailable. No failure recovery or scheduler/partition edit was needed.

Ablation submission update 2026-05-25 19:25 CEST:

- Hybrid order full suite: original SFT `3661162_[0-29%4]` and eval `3661164_[0-29%4]` cover `think_formal` (NL then logic) and `formal_think` (logic then NL), with train depths `1..5/10/15/20/25`, seeds `3407..3409`, and one final `<answer>` tag after both traces. Rows `3661162_0,1` timed out after reaching 10k steps but before final save; originals were canceled and replaced by SFT `3666424_[0-29%4]` with `RESUME_FROM_CHECKPOINT=auto`. Rows `3666424_9..11` and `24..26` then OOMed at train-1-to-20, so targeted replacement `3670782_[9-11,24-26%3]` and new dependent eval `3670783_[0-29%4]` were submitted after forcing gradient checkpointing in the hybrid wrapper. Replacement eval `3682461` is recovering the interrupted rows; current live status is in `docs/running_experiments.md`.
- Conditioned dual-modality full suite: SFT `3661165_[0-14%4]` trains one checkpoint per train-depth/seed on both modalities as separate examples, using `<reasoning_mode>formal_logic</reasoning_mode>` or `<reasoning_mode>natural_language</reasoning_mode>` in the prompt. Eval `3661166_[0-29%4]` evaluates each checkpoint twice, once as `conditioned_logic` and once as `conditioned_nl`.
- Conditioned dual 50k convergence extension: submitted 2026-05-29 as a five-stage resume chain because `a100` max runtime is one day. Arrays `3674879 -> 3674880 -> 3674881 -> 3674882 -> 3674883` train to `10000/20000/30000/40000/50000` optimizer steps; eval arrays are `3674884` for final checkpoints and `3674885` for train-1-to-25 checkpoint curves. The SFT chain is complete. As of 2026-06-02 02:48 CEST, final eval has `2/30` JSONs, and checkpoint eval has `15/30` JSONs covering all `conditioned_logic` train-1-to-25 seeds at checkpoints 10k/20k/30k/40k/50k; `conditioned_nl` checkpoint rows are running.

Oversight update 2026-05-24 15:06 CEST: Qwen 7B `logic_train1to10` eval completed all three seeds. Seeds `3407`, `3408`, and `3409` report OOD correct@16/joint@16 `0.591/0.309`, `0.781/0.400`, and `0.481/0.250`; the three-seed mean is `0.618/0.320`. Matched `nl_exact` Qwen rows are still pending, so this is not yet a logic-vs-NL model-family readout. Tiny Llama scratch eval `3656390` completed all six rows; OOD correct@8 is nonzero but OOD joint@8 and depth-50 correct/joint are `0.0` for every row. Paired maze SFT rows `3656309_0,1` exposed an OLMo-7B memory configuration issue at 8192 tokens; they were resubmitted as `3657088_0,1` with gradient checkpointing enabled, and both replacements cleared the original OOM window by 15:05 CEST.

Oversight update 2026-05-24 18:45 CEST: Qwen 7B `logic_train1to20_seed3407` eval completed with OOD correct@16/joint@16 `0.703/0.182` and depth-50 correct@16/joint@16 `0.500/0.000`; matched Qwen `nl_exact` rows are still pending. Paired `attribute_constraints` train-10 eval completed for seed 3407: both templates have OOD and depth-50 correct@16 `1.000`; logic grounded joint@16 is also `1.000`, while `nl_exact` NL-to-FOL validity remains `0.000`, likely a translator coverage issue for this paired family. Paired maze retry `3657088_0,1` fixed the training-step OOM but failed during online generation eval at step 2000; the paired SFT script now defaults `train.eval_steps` to `max_steps + 1`, dead eval `3657089` was canceled, and maze replacements `3657738_[0-1]` plus dependent eval `3657739_[0-1]` were submitted.

Oversight update 2026-05-24 22:44 CEST: Qwen 7B logic eval now covers all three seeds for train range `1..20`. The `logic_train1to20` mean is OOD correct@16/joint@16 `0.753/0.165` and depth-50 correct@16/joint@16 `0.656/0.021`; this has higher correctness but lower joint validity than the completed `logic_train1to10` mean `0.618/0.320`, so defer architecture-ablation conclusions until `1..25` and matched `nl_exact` rows finish. OLMo-32B SFT row `3656335_0` completed, but original eval row `3656336_0` failed before generation because vLLM rejected `max_model_len=16384` against the OLMo-2 config `max_position_embeddings=4096`. The OLMo eval wrapper now exports `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, `bash -n` passed, stale pending row `3656336_1` was canceled, and replacement eval array `3658461_[0-1%1]` was submitted with `aftercorr:3656335`.

Oversight update 2026-05-25 02:44 CEST: Qwen 7B logic eval now covers all three train ranges and all three seeds. Three-seed means for `logic_train1to10`, `logic_train1to20`, and `logic_train1to25` are OOD correct@16/joint@16 `0.618/0.320`, `0.753/0.165`, and `0.906/0.431`; depth-50 correct@16/joint@16 `0.292/0.031`, `0.656/0.021`, and `0.854/0.156`. The `1..25` logic slice recovers joint validity relative to `1..20`, but it is still below the main OLMo-7B `logic_train1to25` joint result. Matched Qwen `nl_exact` rows have started but are incomplete, so this remains a logic-only architecture signal.

Oversight update 2026-05-25 06:45 CEST: Qwen 7B matched `nl_exact_train1to10` eval now covers all three seeds. The three-seed mean is OOD correct@16/joint@16 `0.461/0.279` and depth-50 correct@16/joint@16 `0.427/0.000`, compared with matched Qwen `logic_train1to10` `0.618/0.320` OOD and `0.292/0.031` depth-50. This is the first completed Qwen logic-vs-NL slice and weakly favors logic on OOD joint validity, but Qwen `nl_exact_train1to20/25` rows are still incomplete. Paired maze sparse eval `3657739_0,1` is still running and is producing long capped generations in higher-depth chunks; no new Traceback/OOM/no-space failure was found in the live target logs.

Oversight update 2026-05-25 10:48 CEST: Qwen 7B `nl_exact_train1to20` now has two eval seeds. The partial mean is OOD correct@16/joint@16 `0.576/0.503` and depth-50 correct@16/joint@16 `0.406/0.203`, so the `1..20` Qwen architecture readout may not favor logic once matched NL finishes. Defer conclusions until `nl_exact_train1to20_seed3409` and the `1..25` NL rows complete. Paired maze sparse eval hit a context-config issue: `3657739_0` failed with prompt length `16400` above `vllm_max_model_len=16384`, `3657739_1` was canceled before the same expected failure, the paired eval wrapper now defaults maze eval to `32768` context and batch `64`, and replacement `3659556_[0-1%2]` is running.

Oversight update 2026-05-25 14:50 CEST: Qwen 7B `nl_exact_train1to20` now has all three eval seeds. Its mean is OOD correct@16/joint@16 `0.438/0.339` and depth-50 correct@16/joint@16 `0.333/0.135`, compared with Qwen `logic_train1to20` `0.753/0.165` OOD and `0.656/0.021` depth-50. The new seed `3409` was much weaker on joint validity, so the earlier two-seed NL advantage did not hold at three seeds. Qwen `nl_exact_train1to25_seed3407` is also complete with OOD/depth-50 joint@16 `0.688/0.562`; seeds `3408`/`3409` are still running, so the deepest Qwen comparison remains incomplete. OLMo-32B SFT is complete, but full depth-50 eval is invalid for `allenai/OLMo-2-0325-32B`: forcing 16k context in replacement `3658461` caused CUDA position-index asserts because the model has a real 4096-position table. The OLMo eval wrapper now writes a separate short-context slice through depth 15 as replacement `3660238`, and OLMo OOD is narrowed to GSM8K-only replacement `3660240`.

Oversight update 2026-05-26 02:51 CEST: small-extra SFT `3656323` and retries are complete, so eval `3656389` and broad OOD `3659356` are priority-pending. Paired maze replacement eval `3659556_[0-1]` completed: logic train/OOD/depth-50 correct@16 is `0.750/0.003/0.000` with joint `0.750/0.000/0.000`; `nl_exact` train/OOD/depth-50 correct@16 is `1.000/0.250/0.000`, but NL-to-FOL parse/joint remains `0.000`. Shortcut original rows `3661136_0..2` failed CUDA OOM; rows `6..8` were canceled before the same likely issue, replacement SFT `3662743_[0-2,6-8%3]` was submitted with checkpointing forced on, and eval `3661137` was replaced by `3662744`.

Oversight update 2026-05-26 09:22 CEST: no new unrecovered severe failures were found in the monitored HFSA follow-up logs. OLMo-32B short-context eval `3660238_[0-1]` completed with both templates saturated through depth 15. Dense backfill `3661090_3` completed the `logic_train1to10_seed3407` 10-point checkpoint grid, while `3661090_0` continues the `logic_train1to5_seed3407` grid. Tiny 100k pretraining has rows `0..2` complete for seeds 3407/3408 and seed-3409 row 2 running; trace/token-budget/shortcut/hybrid rows are active or pending without new code/config failures.

Oversight update 2026-05-26 13:29 CEST: no new unrecovered severe failures were found. Dense backfill `3661090_0` completed the `logic_train1to5_seed3407` 10-point checkpoint grid; rows `6` and `9` are now running the `logic_train1to15/20_seed3407` grids. Broad OOD `3659356` has completed rows `0..22`: all main OLMo-7B logic rows, all `nl_exact_train1to5/10` rows, and two `nl_exact_train1to15` seeds. Three-seed logic means across train depths `1..5/10/15/20/25` are GSM8K EM `0.056/0.079/0.074/0.086/0.056` and mean strict LongBench F1 `0.395/0.391/0.383/0.408/0.404`; completed `nl_exact` means for `1..5/10` are GSM8K EM `0.493/0.479` and mean LongBench F1 `0.171/0.179`, while two completed `nl_exact_train1to15` seeds average GSM8K EM `0.253` and LongBench F1 `0.267`. Treat this as an early downstream tradeoff, not a final modality result. Same-target-token rows `3661120_0..2`, trace-control rows `3661118_0,2`, and conditioned-dual rows `3661165_0..4` completed; active logs for the remaining running rows show progress without Traceback/OOM/quota/no-space/tokenizer/vLLM failures. The next oversight pass is `3664182`, scheduled for 2026-05-26 17:16 CEST.

Oversight update 2026-05-26 17:27 CEST: no new unrecovered severe failures were found. Broad OOD `3659356` has 40 completed result JSONs; main OLMo `nl_exact` is now complete, with GSM8K EM by train depth `1..5/10/15/20/25` of `0.493/0.479/0.326/0.411/0.257` and mean strict LongBench F1 `0.171/0.179/0.214/0.185/0.145`. Paired OOD rows continue the downstream tradeoff: maze `logic/nl_exact` GSM8K EM is `0.114/0.597` and LongBench F1 is `0.403/0.179`; hard attribute `logic/nl_exact` GSM8K EM is `0.139/0.197` and LongBench F1 is `0.273/0.044`. Dense checkpoint output count is now 50 JSONs, with partial `logic_train1to15/20/25_seed3407` additions and active rows `3661090_6,9` plus targeted `3664473_0,1`. Trace-control rows `0..2`, same-token-budget rows `0..2`, and conditioned-dual rows `0..5` are complete; active monitored logs show progress without Traceback/OOM/quota/no-space/tokenizer/vLLM failures. Oversight `3663541` completed, `3664182` is running, and the next pass is `3664671`.

Oversight update 2026-05-26 21:22 CEST: no new unrecovered severe failures were found. Broad OOD `3659356` has 57 completed result JSONs: all main OLMo rows, all paired rows, all Qwen-7B rows, and Qwen-1.5B `logic_train1to10` across three seeds. Qwen-7B OOD shows GSM8K EM strongly favoring `nl_exact` (`0.559/0.532/0.785` for train `1..10/20/25`) over logic (`0.044/0.105/0.084`), while logic has higher mean LongBench F1 at all three train ranges (`0.343/0.360/0.342` vs `0.100/0.240/0.269`). Qwen-1.5B sparse pass@k now has seven logic JSONs: three-seed train-1-to-10 and train-1-to-20, plus train-1-to-25 seed 3407. Dense checkpoint output count is 59 JSONs; `3664473` has added logic train-25 checkpoints `2000/4000/5000` and NL train-25 checkpoints `2000/4000`. Conditioned-dual SFT rows `0..10` are complete, tiny-100k rows `0..3` are complete for all seeds, and the next oversight pass is already queued as `3665088`.

Oversight update 2026-05-27 01:28 CEST: original small-extra Gemma eval row `3656389_18` failed during vLLM startup because the merged Gemma3 checkpoint lacked processor metadata (`preprocessor_config.json`). The merge script now saves optional `AutoProcessor` metadata, the small-extra eval wrapper now guards zero jitter, focused checks passed, stale original Gemma rows `19..35` were canceled, and replacement `3665578_[18-35%4]` is running/pending with `FORCE_SFT_MERGE=1`. Qwen-1.5B pass@k now has 16 JSONs: logic three-seed OOD correct/joint@16 is `0.691/0.231`, `0.694/0.208`, `0.771/0.425` for train `1..10/20/25`; matched NL is complete for train `1..10/20` with `0.561/0.278` and `0.354/0.075`, while train `1..25` has seed `3407` only (`0.525/0.419`). Broad OOD now has all 18 Qwen-1.5B rows; Qwen-1.5B logic train `1..10/20/25` gives GSM8K EM `0.143/0.063/0.053` and mean LongBench F1 `0.146/0.119/0.184`, while matched NL gives GSM8K EM `0.163/0.258/0.326` and mean LongBench F1 `0.063/0.100/0.080`. Dense checkpoint output count is 68 JSONs, with full dense logic grids through train `1..20` and targeted train-25 rows still running.

Oversight update 2026-05-27 05:25 CEST: no new unrecovered severe failures were found in the monitored logs; fresh severe matches were the already-recovered shortcut OOM rows plus benign allocator warnings. Qwen-1.5B pass@k is now complete at 18 JSONs: matched `nl_exact_train1to25` three-seed OOD/depth-50 correct-joint@16 is `0.542/0.292` and `0.438/0.010`, below Qwen-1.5B logic `0.771/0.425` and `0.708/0.260`. Gemma replacement `3665578` has 8 logic JSONs so far; logic train ranges `1..10/20/25` have OOD correct/joint@16 `0.691/0.522`, `0.769/0.288`, and partial `0.681/0.163`. Dense checkpoint output count is 75 JSONs, with full seed-3407 logic grids for all train ranges and `nl_exact_train1to25` missing only checkpoint 9000. Tiny 100k pretraining plus final/checkpoint/OOD eval arrays completed cleanly, but strict extrapolation remains weak: OOD joint@8 is `0.000` except 200M logic at `0.008`, depth-50 joint@8 is `0.000` for every size/template, and tiny-100k strict LongBench F1/EM is `0.000` across groups.

Oversight update 2026-05-27 05:36 CEST: targeted dense train-25 eval `3664473_1` completed while the handoff was being updated. Dense checkpoint output count is now 76 JSONs, and the matched `logic/nl_exact_train1to25_seed3407` dense pair both have full `1000..10000` checkpoint grids.

Oversight update 2026-05-27 05:39 CEST: broad OOD `3659356` advanced to 87 completed result JSONs. Gemma `nl_exact_train1to20` is now three-seed complete with GSM8K EM `0.069`, mean LongBench F1 `0.077`, and mean LongBench EM `0.058`; Gemma `nl_exact_train1to25` rows `87..89` are still running.

Oversight update 2026-05-27 09:29 CEST: broad OOD `3659356` completed all 90 result JSONs. Gemma OOD is fully aggregated and remains weak: logic train ranges `1..10/20/25` have GSM8K EM `0.010/0.013/0.010` and mean LongBench F1 `0.016/0.057/0.009`; matched `nl_exact` has GSM8K EM `0.114/0.069/0.179` and mean LongBench F1 `0.034/0.077/0.027`. Gemma pass@k replacement `3665578` has 15/18 JSONs, with rows `33..35` running for final `nl_exact_train1to25`. Dense checkpoint output count is 80 JSONs. Hybrid original SFT `3661162` was canceled after rows `0,1` timed out post-10k-step/pre-final-save; `train_sft.py` now supports `train.resume_from_checkpoint`, the hybrid wrapper defaults online eval past `max_steps`, replacement SFT `3666424` is running with auto-resume, and replacement eval `3666425` is dependency-pending.

Update 2026-05-27 11:30 CEST: Gemma pass@k replacement `3665578` completed all 18 JSONs; final `nl_exact_train1to25` OOD/depth-50 correct-joint@16 is `0.394/0.394` and `0.302/0.302`. Dense checkpoint output count is now 82 JSONs after `nl_exact_train1to5/10` gained checkpoint `7000`. OLMo-7B UltraChat instruction-control SFT `3666639_0` is running, with OOD eval `3666640_0` dependency-pending.

Update 2026-05-25 09:25 CEST: the first `attribute_constraints` pilot is saturated, so the generator was hardened and replacement jobs were submitted as `3659338 -> 3659339 -> 3659340`. The small scratch-pretraining wave should be treated as a smoke/mechanism result: train-band answer pass@8 is nontrivial, especially 200M logic (`0.859` train and `0.273` OOD correct@8), but OOD joint@8 and depth-50 correct/joint are `0.0` for every tiny row.

## Current Live Jobs And Alignment

Current operational state has moved to `docs/current_system_state.md`. The table below is retained as the 2026-05-19 submission snapshot, not as live truth.

As of 2026-05-19 20:20 CEST:

| job | state | alignment with plan |
| --- | --- | --- |
| `3623863` build | completed | builds the correct depth-50 HFSA dataset |
| `3623864`, `3623865`, `3624535`, `3624536` | cancelled/replaced | old pending arrays used pre-patch Slurm snapshots |
| `3634790_[0-29%6]` main SFT | pending by priority/resources | directly aligned with core synthetic depth-scaling plan; retains up to 12 LoRA checkpoints |
| `3634791_[0-29%6]` final posthoc eval | dependency-pending | aligned with final pass@k depth-50 evaluation |
| `3634792_[0-29%2]` intermediate posthoc eval | dependency-pending | evaluates selected convergence checkpoints |
| `3634793_[0-9%5]` 1k sanity SFT | pending | useful sample-size sanity, one seed only; idempotent skip for completed rows |
| `3634794_[0-9%5]` 1k eval | dependency-pending | aligned as quick sanity eval, not final evidence |

Current SFT jobs log online greedy validation every 1000 steps, but with only `validation.samples_per_step=4` and sampled pass@k disabled. These online metrics are useful as health checks, not final scientific curves.

Checkpoint status:

- SFT saves every 1000 steps.
- The originally submitted main 10k jobs likely used `train.save_total_limit=2`, because Slurm snapshots scripts at submission time.
- Those old jobs have been cancelled.
- The patched resubmission `3634790` overrides `train.save_total_limit=12`, so it can retain the 1000-step LoRA checkpoints needed for selected convergence plots.

If we want correct@1/correct@16 over training progress, we should not blindly retain many checkpoints for the full 30-row array. Storage is already constrained, especially on `${WORK}`. The better design is:

- resubmit the full 30-row sweep with LoRA-only checkpoint retention if we need convergence plots for every row;
- evaluate only selected intermediate LoRA checkpoints first;
- delete merged full-model artifacts immediately after eval.

Patched resubmission setting:

```bash
train.save_steps=1000
train.save_total_limit=12
```

Then run posthoc pass@k on selected checkpoints, e.g. `checkpoint-1000`, `checkpoint-3000`, `checkpoint-10000`, not every checkpoint for every row.

Storage audit:

| artifact/directory | observed approximate size |
| --- | ---: |
| LoRA SFT final adapter | 162 MB |
| LoRA SFT trainer checkpoint | 468 MB |
| merged OLMo-7B checkpoint | 14 GB |
| `${WORK}/synthetic-RLVL/tmp` | 2.0 TB |
| `${WORK}/synthetic-RLVL/runs` | 4.0 TB |
| `${WORK}/synthetic-RLVL/datasets` | 3.5 GB |
| `${WORK}/synthetic-RLVL/passk_eval` | 20 MB |

Policy:

- Merged checkpoints must be transient.
- Eval jobs should merge into `tmp`, run eval, write JSON/JSONL metrics, log W&B, then delete the merged directory.
- Do not keep 10+ merged checkpoints per run.
- If we need many intermediate checkpoints, push small LoRA adapters to HF or evaluate-and-delete locally.
- Old RL/merged artifacts are the dominant current space cost; do not create more of that pattern.

## Implementation Roadmap

Needed for the current core paper:

1. Intermediate-checkpoint evaluation.
   - Implemented: patched HFSA depth-scaling SFT Slurm keeps up to 12 LoRA checkpoints on resubmission.
   - Implemented: `scripts/slurm/jobs/posthoc_hfsa_depth_scaling_intermediate_eval_2026-05-19.slurm` merges/evaluates selected `checkpoint-*` LoRA dirs as transient full checkpoints.
   - Implemented first checkpoint-step plots for seed 3407 under `analysis/hfsa_depth_scaling_2026-05-23/figures/intermediate_seed3407_curves.png`.

2. Plotting and aggregation.
   - Implemented first sparse-grid aggregator: `scripts/analysis/aggregate_hfsa_depth_scaling.py`.
   - Generated per-run, group-summary, depth-curve, paired-delta, and intermediate tables under `analysis/hfsa_depth_scaling_2026-05-23/tables/`.
   - Generated first paper-facing figures under `analysis/hfsa_depth_scaling_2026-05-23/figures/`.

3. Target-template controls.
   - Add proof-only logic and proof-only NL templates.
   - Add terse-NL and pseudocode templates.
   - Add negative-control invalid/shuffled-logic target generation.
   - Extend scoring so generated control traces can be mapped back to formal states where appropriate.

4. Real benchmark eval.
   - Prefer `lm-evaluation-harness` for standard benchmark loaders/metrics rather than maintaining bespoke loaders.
   - Keep custom code as a thin wrapper for model loading, prompt modes, pass@k/self-consistency, W&B grouping, and any missing tasks.
   - Implemented first wrapper: `scripts/evaluate_lm_eval.py`.
   - Implemented first Slurm hook: `scripts/slurm/jobs/lm_eval_hfsa_depth_scaling_2026-05-19.slurm`.
   - Existing `synthrlvl/external_eval.py` only covers a small greedy set: ProofWriter, FOLIO, ProntoQA, ProverQA, LogiQA2, BBH boolean expressions, MMLU philosophy/formal logic, StrategyQA.
   - Need route HotpotQA, 2WikiMultiHopQA, MuSiQue, GSM8K, and optional AIME/GPQA through lm-eval where possible.
   - Need answer normalization/EM/F1 where lm-eval does not already provide the exact mode we need.
   - Need pass@k/self-consistency, not only greedy exact match.
   - Need prompt modes: answer-only, forced-NL-CoT, forced-logic-CoT.

5. Midtraining pipeline.
   - For serious pre/midtraining, use Nanotron or a dedicated external training repo rather than growing this repo into a distributed pretraining stack.
   - Keep this repo responsible for data generation, trace conversion, validation, manifests, and evaluation.
   - Put Nanotron configs/scripts in a separate folder/repo and consume HF datasets produced here.
   - Implemented first token-budgeted mixture exporter: `scripts/data/export_midtraining_mixture.py`.
   - Add plain-text, NL-trace, logic-trace, hybrid-trace, and code-mixture configs.
   - Decide whether each experiment is SFT-style response training or causal continued-pretraining over packed sequences.
   - Add matched-token accounting and W&B logging.

6. Non-LoRA controls.
   - For 7B full finetune, add an FSDP/DeepSpeed/Accelerate path; current `train_sft.py` is LoRA-first and single-process HF Trainer.
   - For 50M-200M models, add a small-model pretraining config/script and model initialization path.
   - For Qwen/OLMo architecture ablations, add a separate small sweep script before submitting jobs; do not hand-overload the OLMo-specific full-grid Slurm scripts.

7. Additional synthetic dataset families.
   - Implemented paired generators: `synthrlvl/datasets/paired_synthetic.py`.
   - Implemented materializer: `scripts/data/build_paired_synthetic_dataset.py`.
   - Detailed construction note and exact generated examples: `docs/paired_synthetic_benchmarks_2026-05-20.md`.
   - Current canonical families:
     - `official_igsm`: fixed locally on 2026-05-28 after changing term tokenization so `-` is parsed as arithmetic rather than part of an identifier. Local depth-50 validation smoke passes, and train-10 build/SFT/eval jobs `3671601`/`3671602`/`3671603` are submitted.
     - `maze_navigation`: keyed/constrained graph traversal with room reachability, held-key state, blocked decoy doors, and unreachable treasure decoys; tests graph/state reachability rather than scalar arithmetic. A 2026-05-23 depth-12 materialization audit passes after extending the key vocabulary.
     - `attribute_constraints`: multi-input slot-value constraint propagation; tests whether the model can carry exact assignments across slots and apply joint constraints without candidate-assignment objects or precomputed feedback facts. A 2026-05-23 depth-12 materialization audit passes.
   - Backward-compatible aliases remain available: `igsm_arithmetic`, `graph_traversal`, `mastermind_constraints`, `constraint_satisfaction`, `constraint_propagation`; the old Mastermind name now routes to `attribute_constraints`.
   - Implemented `TaskBuilder` routing via `task.difficulty=<family>` so the existing post-hoc pass@k evaluator can generate paired-family eval prompts.
   - Each dataset must expose the same latent derivation in paired forms: logic CoT, deterministic NL CoT, and optionally hybrid/pseudocode.
   - The logic trace must validate under our logic engine or under a deliberate arithmetic/relational extension of it. The implemented iGSM path adds `MOD23` for official iGSM arithmetic.
   - Evaluation now reports grounded validity for generated logic traces: proof validity against gold canonical premises/conclusions, separate from internal validity against generated premises.

## Near-Term Decision

As of 2026-05-23 21:15 CEST, the 30-row 10k SFT depth-scaling sweep and sparse post-hoc evals are complete. The comparable final eval array is `3650951_[0-29%10]`; the seed-3407 intermediate eval array is `3650952_[0,3,6,9,12,15,18,21,24,27%4]`. Both completed all expected JSON outputs with exit `0:0`.

The main claim should now be narrowed: logic CoT is more sample/depth efficient than matched `nl_exact` at intermediate train-depth ranges, especially `1..20`, but not uniformly better. `nl_exact` is better for `1..5`, and at `1..25` it catches up/slightly exceeds logic on sparse OOD joint@16 and AUC. Treat current HFSA grounded-validity metrics as an evaluator artifact until semantic/canonicalized grounding is implemented; use internal citation-free logic validity and translated NL-to-FOL validity for the current readout.

Immediate next steps:

1. Use the generated analysis tables/figures under `analysis/hfsa_depth_scaling_2026-05-23/` for the first write-up and sanity-review the seed-level variance before making publication claims.
2. Run follow-up synthetic transfer on audited paired families. `maze_navigation` and hardened `attribute_constraints` have completed train-10 pilots; `official_igsm` is now fixed locally and submitted as train-10 jobs `3671601`/`3671602`/`3671603`.
3. Add a small model-ablation Slurm script before submitting Qwen/OLMo-3-32B jobs. Start with representative train depths `1..10`, `1..20`, `1..25`, both templates, and seeds `3407..3409`; OLMo-3-32B should be a smallest-useful resource-validated subset, not a full-grid repeat.
4. Keep 50M-200M pretraining as a planned mechanism/scaling experiment, but implement a dedicated pretraining stack/config first. The current repo is ready for data generation/eval, not serious distributed pretraining.

Update 2026-05-24 09:18 CEST: the first follow-up wave has been submitted.

- Paired-family transfer pilot:
  - materialization `3656210_[0-1%2]` for `maze_navigation` and `attribute_constraints`, train depths `1..10`, 50k train rows, depth-50 validation, all rows validated;
  - SFT `3656211_[0-3%2]`, dependency `afterok:3656210`, seed `3407`, `logic`/`nl_exact`;
  - sparse eval `3656213_[0-3%2]`, dependency `aftercorr:3656211`.
- Architecture ablation:
  - Qwen SFT `3656217_[0-17%3]` using `Qwen/Qwen2.5-7B`, train depths `1..10`, `1..20`, `1..25`, both templates, seeds `3407..3409`;
  - sparse eval `3656218_[0-17%3]`, dependency `aftercorr:3656217`.
- OLMo-3-32B remains unsubmitted until a 32B resource-validated script/config exists.
