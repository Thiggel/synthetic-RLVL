# Formal Logic CoT Research Plan - 2026-05-19

## Active Direction

We are pausing the RL-validity-reward direction and focusing on supervised / midtraining experiments.

Main question:

> Does reasoning in a formal-logic chain-of-thought substrate improve a language model's reasoning ability and length extrapolation compared with semantically matched natural-language CoT?

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

## Required Controls

### Length And Compute

Run these before making a strong paper claim:

| control | implementation | why |
| --- | --- | --- |
| same examples, same optimizer steps | current setup | base comparison |
| same target-token budget | choose per-template max steps from token audit | controls shorter full logic targets |
| same train loss / same ID accuracy | evaluate intermediate checkpoints | tests whether NL simply converges slower |
| overtrained NL | train NL beyond 10k or to matched ID score | tests catch-up hypothesis |
| proof-only targets | prompt contains premises; target only proof/conclusion/answer | removes premise-copy/translation overhead |

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
| `synthrlvl_gsm8k_tagged` | GSM8K | extracts `<answer>...</answer>` first, then numeric-normalizes |
| `synthrlvl_longbench_hotpotqa_tagged` | LongBench HotpotQA context-provided QA | strict answer extraction before QA F1 |
| `synthrlvl_longbench_2wikimqa_tagged` | LongBench 2WikiMultiHopQA context-provided QA | strict answer extraction before QA F1 |
| `synthrlvl_longbench_musique_tagged` | LongBench MuSiQue context-provided QA | strict answer extraction before QA F1 |

Strict extraction is intentional for the context-provided QA tasks: if a model copies passage text in `<think>`/`<premises>` without a final `<answer>` tag or explicit answer marker, the prediction is scored as empty. This avoids false F1 credit from answer strings appearing in the supplied context.

Pilot jobs `3659344` and strict rerun `3659348` completed on two Qwen-1.5B train-1-to-10 checkpoints. The extractor correctly pulled learned `<answer>` tags in sample generations, and the strict rerun removed accidental context-copy credit. Broad OOD eval is running/pending as `3659356_[0-89%4]` for non-OLMo-32B 1-GPU models; rows `0..22` are complete, rows `23..26` are running, and rows `27..89` are pending by array throttle. The OLMo-32B full OOD array `3659357_[0-1%1]` failed because LongBench contexts exceed OLMo-2 32B's 4096-position limit; replacement `3660240_[0-1%1]` completed GSM8K-only short-context with EM `0.197` for logic and `0.683` for `nl_exact`. Tiny scratch checkpoints are complete for all three seeds; GSM8K EM is near-zero and strict LongBench F1/EM is `0.000` across size/template groups.

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
| Qwen/Gemma smaller LoRA | `3656323` SFT, retries `3656359` and `3656387`, eval `3656389` | `Qwen/Qwen2.5-1.5B` because exact `Qwen/Qwen2.5-1B` was not available; `google/gemma-3-4b-pt` for the requested Gemma-4B-style ablation |
| OLMo-32B pilot | `3656335` SFT, original `3656336` eval, failed replacement `3658461`, short-context replacement `3660238` eval | `allenai/OLMo-2-0325-32B`; exact OLMo-3 base 32B was not available, while the checked OLMo-3.1 32B model is a Think checkpoint. OLMo-2 32B enforces a 4096-position limit, so this is now a short-context pilot only |
| 50M-200M scratch pretraining | `3656338` pretrain, retries `3656360` and `3656388`, eval `3656390` | random-init Llama configs with a Llama3 tokenizer; this does not cover a true 50B run |

Rows that hit `NODE_FAIL` were retried rather than treated as scientific failures. The second retries exclude node `a0934`, where the repeated immediate node failures occurred. The current live operational state and dependency rewiring are in `docs/current_system_state.md`.

Ablation submission update 2026-05-25 19:01 CEST:

- Trace controls: SFT array `3661118_[0-17%3]` and eval array `3661119_[0-17%3]` cover `terse_nl`, `rule_annotated_nl`, `pseudocode`, `shuffled_logic`, `invalid_logic`, and `shuffled_nl`, each with seeds `3407..3409`, train depths `1..25`, and the sparse depth-50 eval protocol.
- Same target-token budget: SFT array `3661120_[0-5%3]` and eval array `3661121_[0-5%3]` cover `logic` at 10k steps and `nl_exact` at 7140 steps. The step ratio comes from a 512-row OLMo tokenizer audit on train-1..25 targets: mean logic target length `1038` tokens and mean `nl_exact` target length `1454` tokens.
- Shortcut robustness: initial build `3661122_[0-1%1]` failed in the probe because the old shortcut schema exhausted high-depth state-word/predicate capacity; dependents `3661123`/`3661124` were canceled. After expanding the schema word banks and enabling extended predicate rendering, local probes for shortcut rates `0.5` and `0.8` passed and replacement build `3661135_[0-1%1]` completed. Original SFT rows `3661136_0..2` OOMed under `train.gradient_checkpointing=auto`, so the wrapper now defaults shortcut SFT to checkpointing on; replacement SFT `3662743_[0-2,6-8%3]` covers failed rows and at-risk logic shortcut-0.8 rows, and eval `3661137` was replaced by `3662744`. The existing main grid is the shortcut-rate `0.0` baseline.

Ablation submission update 2026-05-25 19:25 CEST:

- Hybrid order full suite: SFT `3661162_[0-29%4]` and eval `3661164_[0-29%4]` cover `think_formal` (NL then logic) and `formal_think` (logic then NL), with train depths `1..5/10/15/20/25`, seeds `3407..3409`, and one final `<answer>` tag after both traces.
- Conditioned dual-modality full suite: SFT `3661165_[0-14%4]` trains one checkpoint per train-depth/seed on both modalities as separate examples, using `<reasoning_mode>formal_logic</reasoning_mode>` or `<reasoning_mode>natural_language</reasoning_mode>` in the prompt. Eval `3661166_[0-29%4]` evaluates each checkpoint twice, once as `conditioned_logic` and once as `conditioned_nl`.

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
     - `official_igsm`: blocked after a broader 2026-05-23 audit; subtraction-substitution proof lines can fail validation, so do not train on it until proof generation/verifier support is fixed.
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
2. Run follow-up synthetic transfer only on audited paired families for now: `maze_navigation` and `attribute_constraints`. Do not launch `official_igsm` until the subtraction proof-generation/verifier failure is fixed.
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
