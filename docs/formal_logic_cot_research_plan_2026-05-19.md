# Formal Logic CoT Research Plan - 2026-05-19

## Active Direction

We are pausing the RL-validity-reward direction and focusing on supervised / midtraining experiments.

Main question:

> Does reasoning in a formal-logic chain-of-thought substrate improve a language model's reasoning ability and length extrapolation compared with semantically matched natural-language CoT?

The current strongest signal is from the pure-SFT HFSA first wave:

| train depths | trace | correct@1 on 16..25 | correct@16 on 16..25 |
| --- | --- | ---: | ---: |
| 1..10 | logic CoT | 0.287 | 0.842 |
| 1..10 | natural-language CoT | 0.336 | 0.385 |
| 1..15 | logic CoT | 0.511 | 0.959 |
| 1..15 | natural-language CoT | 0.363 | 0.428 |

Interpretation: logic CoT appears to extrapolate much better under sampling. The gap is especially large for correct@16, which suggests the logic-trained model places correct long-chain solutions in its sample distribution much more often.

## Claims We Can Aim To Support

Primary claim:

- Formal-logic CoT is a better supervised reasoning substrate than matched deterministic natural-language CoT for compositional length extrapolation.

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
| eval depths | `1..50` |
| posthoc samples | 128 prompts/depth, 16 generations/prompt |
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
| `pseudocode` | structured non-FOL symbolic trace, tests rigid-notation advantage |
| invalid/shuffled logic negative control | tests whether valid derivation structure matters beyond formal-looking tokens |
| hybrid logic+NL | tests whether combining formal structure and natural semantics helps |

Existing code already supports some hybrid templates (`formal_think`, `think_formal`, `logic_natural`, `natural_logic`), but proof-only, terse-NL, pseudocode, and invalid-logic controls still need implementation.

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
| second 7B base model | representative subset | rule out OLMo-specific behavior |
| 50M-200M pretraining | from scratch or continued pretraining | mechanism/scaling evidence |

Do not make small-model pretraining the initial main result; use it as mechanism/scaling evidence after the 7B LoRA trend is confirmed.

## Current Live Jobs And Alignment

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
   - Add plotting code for metric vs optimizer step and metric vs target tokens seen.

2. Plotting and aggregation.
   - Aggregate pass@k JSON across seeds.
   - Plot mean/std by trace, train depth, eval depth, and checkpoint step.
   - Compute OOD AUC and threshold-depth summaries.

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

7. Additional synthetic dataset families.
   - Implemented paired generators: `synthrlvl/datasets/paired_synthetic.py`.
   - Implemented materializer: `scripts/data/build_paired_synthetic_dataset.py`.
   - Detailed construction note and exact generated examples: `docs/paired_synthetic_benchmarks_2026-05-20.md`.
   - Current canonical families:
     - `official_igsm`: official `facebookresearch/iGSM` sampling plus a 1:1 validated logic trace over the official modulo-23 arithmetic solution.
     - `maze_navigation`: keyed/constrained graph traversal with room reachability, held-key state, blocked decoy doors, and unreachable treasure decoys; tests graph/state reachability rather than scalar arithmetic.
     - `attribute_constraints`: multi-input slot-value constraint propagation; tests whether the model can carry exact assignments across slots and apply joint constraints without candidate-assignment objects or precomputed feedback facts.
   - Backward-compatible aliases remain available: `igsm_arithmetic`, `graph_traversal`, `mastermind_constraints`, `constraint_satisfaction`, `constraint_propagation`; the old Mastermind name now routes to `attribute_constraints`.
   - Implemented `TaskBuilder` routing via `task.difficulty=<family>` so the existing post-hoc pass@k evaluator can generate paired-family eval prompts.
   - Each dataset must expose the same latent derivation in paired forms: logic CoT, deterministic NL CoT, and optionally hybrid/pseudocode.
   - The logic trace must validate under our logic engine or under a deliberate arithmetic/relational extension of it. The implemented iGSM path adds `MOD23` for official iGSM arithmetic.
   - Evaluation now reports grounded validity for generated logic traces: proof validity against gold canonical premises/conclusions, separate from internal validity against generated premises.

## Near-Term Decision

Because the main 30-row depth-scaling array was submitted before the checkpoint-retention patch, we should cancel/resubmit it if convergence pass@k curves are required for the same wave. If we keep it as-is, it remains valuable for the final depth-scaling result but will not support complete training-progress pass@k curves.
