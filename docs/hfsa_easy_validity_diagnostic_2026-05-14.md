# HFSA Easy Validity Reward Diagnostic - 2026-05-14

## Question

We wanted to understand why adding a citation-free proof-validity reward did not clearly improve validation validity or correctness, especially on OOD depths. The core concerns were:

- Is the validity reward too sparse?
- Do correctness-only and correctness-plus-validity models follow different routes?
- Are both models skipping proof steps or taking shortcut branches?
- Is the model doing multiple reasoning steps internally and then writing a post-hoc/sloppy proof?
- Is there a code/dataset/reward-design issue that makes validity reward fail to target the intended behavior?

## Artifacts

Generated analysis artifacts:

- Figures: `analysis/hfsa_easy_validity_2026-05-14/figures/`
- Tables: `analysis/hfsa_easy_validity_2026-05-14/tables/`
- Targeted generations: `analysis/hfsa_easy_validity_2026-05-14/targeted/`
- Qualitative examples: `analysis/hfsa_easy_validity_2026-05-14/examples/qualitative_examples.md`

Analysis scripts added:

- `scripts/analysis/hfsa_validity_analysis_lib.py`
- `scripts/analysis/hfsa_easy_targeted_probe.py`
- `scripts/analysis/hfsa_easy_validity_diagnostics.py`

Key figures:

- `analysis/hfsa_easy_validity_2026-05-14/figures/pass1_step_curves.pdf`
- `analysis/hfsa_easy_validity_2026-05-14/figures/passk_depth_slices.pdf`
- `analysis/hfsa_easy_validity_2026-05-14/figures/train_rollout_reward_density.pdf`
- `analysis/hfsa_easy_validity_2026-05-14/figures/targeted_targeted_normal_rates.pdf`
- `analysis/hfsa_easy_validity_2026-05-14/figures/targeted_failure_stack_step10.pdf`
- `analysis/hfsa_easy_validity_2026-05-14/figures/targeted_failure_stack_step15.pdf`
- `analysis/hfsa_easy_validity_2026-05-14/figures/targeted_failure_stack_step20.pdf`
- `analysis/hfsa_easy_validity_2026-05-14/figures/gold_response_token_lengths.pdf`

## Data Analyzed

I analyzed:

- All 12 completed `hard_fsa_schema_easy500` pass@k JSONs.
- All 12 existing greedy sample JSONLs.
- All GRPO training logs for the 12 runs.
- A new targeted vLLM probe on one seed per condition:
  - `train0p0_correct_only`
  - `train0p0_validity_gated`
  - `train0p5_correct_only`
  - `train0p5_validity_gated`
- Targeted probe size:
  - depths `5, 10, 15, 20`
  - 12 prompts per depth
  - normal prompt and initial-marker-swap prompt
  - 16 generations per prompt
  - total 6,144 generated outputs

The targeted probe classifies each generation by:

- correctness
- citation-free validity
- whether the proof contains the gold answer atom on the queried constant
- whether the proof contains the shortcut answer atom
- whether the output is shorter than the expected proof length
- whether the output is malformed/truncated
- whether the answer equals the shortcut answer
- failure category

## Main Finding

There is a major reward/target alignment problem in the current HFSA dataset target.

The question asks:

```text
Which state applies to f?
```

The `<answer>` is the final state, e.g. `poppy`.

But the formal target currently ends with the final marker as `<conclusion>`, not the final state. Example from a fully valid depth-5 generation:

```text
<proof>
...
Df ; ->E
Hf ; ->E
</proof>
<conclusion>
Hf
</conclusion>
...
<answer>
poppy
</answer>
```

Here `D` maps to `poppy`, but `H` maps to a marker such as `south`. So the valid proof concludes the marker, while the answer is the previous state atom.

This means the citation-free validity reward optimizes internal proof validity for the generated conclusion, but the generated conclusion is not the answer proposition. Correctness is checked separately from `<answer>`. This allows three bad behaviors:

- The model can produce a formally valid proof to a marker and separately output a correct answer string.
- The model can produce a formally valid proof to the wrong branch/shortcut marker and separately output the correct answer string.
- The validity reward does not directly require the final answer state to be the proof conclusion.

This is not just a metric interpretation issue. It is a dataset/reward alignment issue. The formal conclusion should be the queried final state atom, or the reward should explicitly require that the generated answer atom is derived in the proof.

## Is The Validity Reward Sparse?

On the training distribution, no. It is mostly saturated.

Training rollout means across all 500 steps:

| condition | correct | citation-free valid | main reward event | zero-reward-variance batches |
|---|---:|---:|---:|---:|
| train0p0 correct-only | 0.996 | 0.992 | 0.996 | 0.765 |
| train0p0 validity-gated | 0.996 | 0.992 | 0.990 | 0.568 |
| train0p5 correct-only | 0.982 | 0.928 | 0.982 | 0.398 |
| train0p5 validity-gated | 0.984 | 0.938 | 0.932 | 0.130 |

Late training steps 450-500:

| condition | correct | citation-free valid | main reward event | zero-reward-variance batches |
|---|---:|---:|---:|---:|
| train0p0 correct-only | 0.996 | 0.993 | 0.996 | 0.765 |
| train0p0 validity-gated | 0.996 | 0.993 | 0.991 | 0.562 |
| train0p5 correct-only | 0.984 | 0.930 | 0.984 | 0.444 |
| train0p5 validity-gated | 0.990 | 0.963 | 0.959 | 0.170 |

Interpretation:

- For shortcut rate 0.0, SFT already makes training-depth generations almost perfectly correct and citation-free valid. Validity reward has no room to improve training behavior.
- For shortcut rate 0.5, the gated validity reward does increase training citation-free validity from about `0.930` to `0.963` late in training.
- So the validity reward is not absent. It gives some training-distribution signal in the harder shortcut split.
- But this signal does not transfer strongly to OOD depths.
- Many batches still have zero reward variance, especially in the non-shortcut setting, so GRPO often has no useful advantage signal.

## Aggregate pass@1 Results

Mean over three seeds.

| condition | train 1-5 correct | train 1-5 cf-valid | OOD 6-20 correct | OOD 6-20 cf-valid | OOD 6-20 cf-joint | tail 15-20 correct | tail 15-20 cf-valid | tail 15-20 cf-joint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train0p0 correct-only | 0.998 | 0.999 | 0.461 | 0.314 | 0.269 | 0.192 | 0.041 | 0.013 |
| train0p0 validity-gated | 0.999 | 0.999 | 0.462 | 0.314 | 0.269 | 0.196 | 0.039 | 0.013 |
| train0p5 correct-only | 0.999 | 0.999 | 0.458 | 0.318 | 0.274 | 0.187 | 0.037 | 0.012 |
| train0p5 validity-gated | 0.998 | 0.999 | 0.463 | 0.326 | 0.282 | 0.189 | 0.039 | 0.013 |

At depth 10:

| condition | correct@1 | cf-valid@1 | cf-joint@1 | cf-valid given correct@1 |
|---|---:|---:|---:|---:|
| train0p0 correct-only | 0.702 | 0.605 | 0.524 | 0.737 |
| train0p0 validity-gated | 0.710 | 0.606 | 0.528 | 0.734 |
| train0p5 correct-only | 0.707 | 0.616 | 0.536 | 0.750 |
| train0p5 validity-gated | 0.727 | 0.633 | 0.555 | 0.757 |

At depth 20:

| condition | correct@1 | cf-valid@1 | cf-joint@1 |
|---|---:|---:|---:|
| train0p0 correct-only | 0.071 | 0.023 | 0.002 |
| train0p0 validity-gated | 0.066 | 0.027 | 0.002 |
| train0p5 correct-only | 0.064 | 0.019 | 0.004 |
| train0p5 validity-gated | 0.070 | 0.026 | 0.003 |

Interpretation:

- Validity-gated reward has a small advantage at depth 10 in the shortcut split, but it is not a large separation.
- At depths 15-20, validity reward does not materially improve either correctness or citation-free validity.
- Most depth-20 correctness is invalid; cf-joint is near zero.

## Targeted Probe Results

Normal prompts, seed 3407, 16 generations per prompt.

| condition | step | correct | cf-valid | correct + cf-valid + gold answer atom derived | shortcut answer | proof shorter than expected | cf line-valid fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| train0p0 correct-only | 5 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| train0p0 validity-gated | 5 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| train0p5 correct-only | 5 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| train0p5 validity-gated | 5 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| train0p0 correct-only | 10 | 0.568 | 0.516 | 0.375 | 0.323 | 0.234 | 0.807 |
| train0p0 validity-gated | 10 | 0.578 | 0.500 | 0.354 | 0.349 | 0.229 | 0.792 |
| train0p5 correct-only | 10 | 0.542 | 0.562 | 0.365 | 0.370 | 0.203 | 0.831 |
| train0p5 validity-gated | 10 | 0.583 | 0.562 | 0.417 | 0.354 | 0.141 | 0.840 |
| train0p0 correct-only | 15 | 0.161 | 0.094 | 0.016 | 0.271 | 0.615 | 0.190 |
| train0p0 validity-gated | 15 | 0.203 | 0.099 | 0.010 | 0.245 | 0.589 | 0.218 |
| train0p5 correct-only | 15 | 0.172 | 0.094 | 0.026 | 0.208 | 0.646 | 0.188 |
| train0p5 validity-gated | 15 | 0.198 | 0.109 | 0.010 | 0.281 | 0.625 | 0.232 |
| train0p0 correct-only | 20 | 0.047 | 0.021 | 0.000 | 0.031 | 0.859 | 0.069 |
| train0p0 validity-gated | 20 | 0.047 | 0.010 | 0.000 | 0.057 | 0.854 | 0.083 |
| train0p5 correct-only | 20 | 0.052 | 0.026 | 0.000 | 0.047 | 0.865 | 0.065 |
| train0p5 validity-gated | 20 | 0.042 | 0.042 | 0.005 | 0.057 | 0.818 | 0.093 |

Paired differences, validity-gated minus correctness-only, same prompts:

| shortcut split | step | correct | cf-valid | correct+cf-valid+gold atom | shortcut answer | proof shorter |
|---|---:|---:|---:|---:|---:|---:|
| train0p0 | 10 | +0.010 | -0.016 | -0.021 | +0.026 | -0.005 |
| train0p0 | 15 | +0.042 | +0.005 | -0.005 | -0.026 | -0.026 |
| train0p0 | 20 | +0.000 | -0.010 | +0.000 | +0.026 | -0.005 |
| train0p5 | 10 | +0.042 | +0.000 | +0.052 | -0.016 | -0.062 |
| train0p5 | 15 | +0.026 | +0.016 | -0.016 | +0.073 | -0.021 |
| train0p5 | 20 | -0.010 | +0.016 | +0.005 | +0.010 | -0.047 |

Interpretation:

- There is a real but small route difference at depth 10 for the shortcut split: validity-gated has fewer shortened proofs and higher correct+valid+gold-atom rate.
- This disappears or reverses at depth 15.
- At depth 20 all conditions are dominated by truncation/malformed/short outputs.
- The models are not clearly learning different algorithms; they are mostly failing in similar ways.

## Failure Modes

### 1. Formal conclusion is not the answer state

This is the most important issue. The gold target and model generations usually conclude the final marker, not the final state. Validity therefore proves the wrong formal object relative to the natural question.

Current target shape:

```text
state_D(query_const) ; ->E
marker_D(query_const) ; ->E
<conclusion>
marker_D(query_const)
</conclusion>
<answer>
state_D_name
</answer>
```

Better target shape:

```text
state_D(query_const) ; ->E
<conclusion>
state_D(query_const)
</conclusion>
<answer>
state_D_name
</answer>
```

For intermediate steps we still need marker derivations because the next transition depends on the marker. For the final step, the marker derivation is unnecessary and actively misaligns the validity objective.

### 2. Validity is internal consistency, not prompt-grounded validity

The model generates its own `<constants>`, `<predicates>`, and `<premises>` from the natural-language prompt. The evaluator checks whether the generated proof is valid relative to those generated premises and generated conclusion.

That means citation-free validity currently measures:

```text
generated proof is internally valid under generated formalization
```

It does not measure:

```text
generated proof faithfully formalizes and solves the prompt
```

This is not necessarily wrong if we intentionally want the model to learn formalization too. But it weakens the validity reward as a training signal for the desired path, because a self-consistent but wrong or shortcut formalization can still receive validity credit.

Example failure seen in targeted generations:

- answer is correct: `ivory`
- proof is citation-free valid
- proof derives the shortcut atom `willow`, not the gold `ivory` atom
- `<answer>` says `ivory` anyway

This receives correctness from `<answer>` and validity from a self-consistent proof, but not because the proof established the answer.

### 3. Step skipping / post-hoc proof failure grows sharply with depth

The targeted probe shows that proof shortening becomes dominant at long depth.

Normal prompts:

| condition | step 10 shorter | step 15 shorter | step 20 shorter |
|---|---:|---:|---:|
| train0p0 correct-only | 0.234 | 0.615 | 0.859 |
| train0p0 validity-gated | 0.229 | 0.589 | 0.854 |
| train0p5 correct-only | 0.203 | 0.646 | 0.865 |
| train0p5 validity-gated | 0.141 | 0.625 | 0.818 |

This is consistent with the model often knowing or guessing an answer/branch in latent space but failing to write the full line-by-line proof.

Among correct normal samples:

| condition | step | cf-valid among correct | gold atom derived among correct | shorter among correct |
|---|---:|---:|---:|---:|
| train0p0 correct-only | 10 | 0.706 | 0.862 | 0.183 |
| train0p0 validity-gated | 10 | 0.658 | 0.865 | 0.171 |
| train0p5 correct-only | 10 | 0.712 | 0.933 | 0.163 |
| train0p5 validity-gated | 10 | 0.732 | 0.920 | 0.098 |
| train0p0 correct-only | 15 | 0.129 | 0.516 | 0.323 |
| train0p0 validity-gated | 15 | 0.128 | 0.385 | 0.462 |
| train0p5 correct-only | 15 | 0.152 | 0.485 | 0.455 |
| train0p5 validity-gated | 15 | 0.158 | 0.526 | 0.263 |
| train0p0 correct-only | 20 | 0.000 | 0.444 | 0.556 |
| train0p0 validity-gated | 20 | 0.000 | 0.222 | 0.444 |
| train0p5 correct-only | 20 | 0.100 | 0.300 | 0.400 |
| train0p5 validity-gated | 20 | 0.125 | 0.250 | 0.375 |

At depth 15-20, correct answers often do not come with a valid proof. This is exactly the post-hoc/sloppy proof behavior we were worried about.

### 4. Shortcut-branch behavior is present in both models

Normal prompt shortcut-answer rates:

| condition | step 10 | step 15 | step 20 |
|---|---:|---:|---:|
| train0p0 correct-only | 0.323 | 0.271 | 0.031 |
| train0p0 validity-gated | 0.349 | 0.245 | 0.057 |
| train0p5 correct-only | 0.370 | 0.208 | 0.047 |
| train0p5 validity-gated | 0.354 | 0.281 | 0.057 |

The validity-gated model does not reliably suppress shortcut answers. In the shortcut split at depth 10 it suppresses shortcut answers only slightly (`0.370 -> 0.354`). At depth 15 it is actually higher in this targeted sample (`0.208 -> 0.281`).

This again suggests the validity reward is not cleanly forcing the model onto the gold reasoning route.

### 5. Depth-20 evaluation is partly confounded by response length

Gold response token lengths grow with depth:

| depth | prompt mean | prompt max | gold response mean | gold response max |
|---:|---:|---:|---:|---:|
| 5 | 315.8 | 330 | 325.2 | 337 |
| 10 | 605.0 | 623 | 570.2 | 588 |
| 15 | 899.0 | 917 | 816.0 | 836 |
| 20 | 1190.2 | 1198 | 1040.3 | 1061 |

Current `max_new_tokens` / `max_response_length` is `1024`.

So depth-20 gold outputs often exceed the generation cap. This makes depth-20 validity/format artificially bad and means the depth-20 tail should not be interpreted without rerunning post-hoc eval at `max_new_tokens >= 1536`, preferably 2048.

This does not explain depth-10 and depth-15 failures, because those fit in the cap. But it does explain why depth-20 has so many malformed/short outputs.

## Counterfactual Marker-Swap Probe

I also generated prompt variants where the initial marker is swapped to the other branch, so the correct answer changes while the global prompt structure stays similar.

At depth 10:

| condition | normal correct | swap-marker correct | normal cf-valid | swap-marker cf-valid | swap-marker shortcut answer |
|---|---:|---:|---:|---:|---:|
| train0p0 correct-only | 0.568 | 0.604 | 0.516 | 0.427 | 0.312 |
| train0p0 validity-gated | 0.578 | 0.599 | 0.500 | 0.464 | 0.323 |
| train0p5 correct-only | 0.542 | 0.599 | 0.562 | 0.495 | 0.323 |
| train0p5 validity-gated | 0.583 | 0.656 | 0.562 | 0.547 | 0.266 |

Interpretation:

- The models are not completely ignoring the initial marker; they can often switch to the alternate branch.
- The shortcut-trained validity-gated model is best at depth-10 swap-marker correctness and has the lowest swap-marker shortcut-answer rate.
- But the effect is still modest and does not persist strongly at deeper depths.

## Why Validity Reward Is Not Separating Strongly

The evidence points to multiple simultaneous reasons:

1. Training-depth performance is saturated.
   - On depths 1-5, both correctness and citation-free validity are already near 1.0.
   - The validity reward cannot add much gradient when the SFT model already produces valid train-depth outputs.

2. GRPO often has low or zero reward variance.
   - In the non-shortcut setting, 56-77% of training batches have zero reward variance depending on reward.
   - If all rollouts get the same reward, GRPO cannot learn much from that batch.

3. The formal conclusion is misaligned with the answer.
   - The proof validity reward targets final marker validity, while correctness targets the final state answer string.
   - This permits valid proofs that are not proofs of the answer.

4. The validity check is self-consistency over generated formalization.
   - The model can alter/generated premises and still receive citation-free validity.
   - This weakens the connection between validity reward and solving the given prompt.

5. The reward is all-or-nothing for long proof validity.
   - At OOD depths, once the proof skips a necessary step, full validity drops to zero.
   - The reward does not tell the model how far along the valid route it got.

6. The model has a strong step-skipping tendency.
   - At depth 15 and 20, many outputs are shorter than a complete proof.
   - Correct answers often occur with invalid or incomplete proof traces.

7. Depth-20 is currently response-length capped.
   - Gold responses often exceed 1024 tokens.
   - This makes many depth-20 generations impossible to complete under the current cap.

## Are Correctness-only And Validity-gated Models Behaving Differently?

Only slightly.

Evidence for a small difference:

- On shortcut-rate 0.5 training rollouts, gated validity increases citation-free validity late in training from `0.930` to `0.963`.
- On targeted normal depth 10, shortcut-rate 0.5 gated reward improves `correct + cf-valid + gold atom derived` from `0.365` to `0.417`.
- On targeted normal depth 10, shortcut-rate 0.5 gated reward reduces shortened proofs from `0.203` to `0.141`.

Evidence against a strong algorithmic difference:

- Aggregate OOD pass@1 differences are tiny.
- Depth-15 and depth-20 targeted rates are mostly indistinguishable.
- Both models take shortcut answers at similar rates.
- Both models frequently output correct answers with incomplete/invalid proofs.
- Both models are badly affected by response length/truncation at depth 20.

So the current experiments do not yet show that validity reward induces a robust different reasoning algorithm. It has a weak local effect but is dominated by target misalignment, train-depth saturation, and long-depth generation failure.

## Recommended Fixes Before The Next Main Run

### Fix 1: Align formal conclusion with the answer

Change HFSA generation so the final formal conclusion is the queried final state atom, not the final marker atom.

Current depth-D target:

```text
state_D(x_D)
marker_D(x_D)
conclusion = marker_D(x_D)
answer = state_D_name
```

Preferred target:

```text
state_D(x_D)
conclusion = state_D(x_D)
answer = state_D_name
```

For intermediate depths, keep marker derivations because they are needed to choose the next transition. For the last transition, stop at the final state.

This is the cleanest fix because it makes validity naturally validate the answer proposition.

### Fix 2: Add an answer-derived validity reward variant

Without checking the gold proof, we can still require:

```text
generated answer word maps to a generated predicate P
proof derives P(query_constant)
```

A useful reward would be:

```text
R = 1[correct and citation_free_valid and generated_answer_atom_derived] + 0.1 * format
```

Or, if we want dense shaping:

```text
R = correct * answer_atom_valid_prefix_fraction + 0.1 * format
```

This avoids full gold-proof checking while preventing the model from receiving validity credit for a proof that does not establish its own answer.

### Fix 3: Increase post-hoc eval max_new_tokens

Before interpreting depths 15-20, rerun post-hoc eval with:

```text
max_new_tokens = 1536 or 2048
```

Depth 20 gold responses exceed 1024 tokens, so the current depth-20 validity/format numbers are partly capped.

### Fix 4: Use valid-prefix or max-valid-depth reward

For long-chain RL, full binary validity is too brittle. Dense reward should be based on valid progress along the chain, not just line-valid fraction.

Preferred dense signal:

```text
max_valid_prefix_depth / target_depth
```

This would distinguish:

- proof valid through 14 of 15 steps
- proof invalid after 2 steps

The current binary validity reward treats both as zero.

### Fix 5: Consider taking generated premises out of the model output

If we want validity to mean validity relative to the prompt, the cleanest design is to provide/formalize premises outside the model and ask the model to emit only:

```text
<proof>
...
</proof>
<conclusion>
...
</conclusion>
<answer>
...
</answer>
```

This removes the self-consistent-but-wrong formalization loophole. If we still want the model to learn formalization, keep the current setup for a separate experiment, but do not interpret internal validity as prompt-grounded validity.

## Bottom Line

The validity reward is not failing because the model never produces valid samples. On the training depths it produces valid samples almost all the time. The deeper problem is that the current reward is saturated on train depths and misaligned with the scientific target:

- the formal conclusion is a marker, not the answer state;
- validity is checked against generated premises/conclusion, not necessarily the prompt;
- long-depth generation often skips steps;
- depth-20 is response-length capped.

The immediate next step should be to fix the HFSA target so the formal conclusion is the final state atom, then rerun a small sanity experiment and post-hoc eval with a larger generation cap. After that, test an answer-derived validity reward and a valid-prefix-depth reward.
