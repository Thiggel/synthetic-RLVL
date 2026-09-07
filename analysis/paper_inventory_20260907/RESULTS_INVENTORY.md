# Verified results inventory for the ICLR paper (compiled 2026-09-07)

Every number below was read from an artifact on disk during the 2026-09-06/07
sessions. Provenance is given for each block. Nothing here is from memory.

## 1. Synthetic paired-trace grid (OLMo-3-7B)

Source: `analysis/branchproof_unique_v2_20260711/tables/final_group_summary.csv`,
BranchProof-unique-v2, 3 seeds per cell, OOD split.

| train_max | logic greedy | logic p@1 | logic p@16 | nl greedy | nl p@1 | nl p@16 |
|---|---|---|---|---|---|---|
| 5  | 0.108 | 0.098 | 0.500 | 0.563 | 0.524 | 0.735 |
| 10 | 0.331 | 0.308 | 0.678 | 0.600 | 0.580 | 0.711 |
| 15 | 0.281 | 0.261 | 0.699 | 0.574 | 0.495 | 0.569 |
| 20 | 0.220 | 0.232 | 0.804 | 0.684 | 0.312 | 0.453 |
| 25 | 0.821 | 0.750 | 0.996 | 0.487 | 0.178 | 0.425 |

TWO thresholds on TRAINING depth, not test depth:
- sampled (pass@16) crossover at train_max 15
- greedy / pass@1 crossover at train_max 25
Formal buys coverage roughly ten depth levels before it buys greedy accuracy.

## 2. Long-window midtrain, Qwen2.5-7B (the real-model evidence)

Chain: Qwen2.5-7B -> nanotron midtrain (Dolmino, seq 8192, 2385 steps,
2,500,853,760 tokens, 10% replacement slice) -> HF convert -> Dolci SFT
(100k, lr 5e-6, seed 3407, full-parameter FSDP) -> eval.
All five arms reached step 2385. Post-SFT `final/` verified for all five.

### 2.1 Corpus lengths (Qwen2.5-7B tokenizer, n=72,000 per corpus)
Source: `analysis/longwin_midtrain_prep_20260826/PREP_REPORT.md` sections 2, 3, 9.

| corpus | p50 | mean | p95 | p99 | max |
|---|---|---|---|---|---|
| logic | 3,749 | 3,856 | 7,357 | 7,673 | 7,727 |
| longdoc | 3,712 | 3,832 | - | - | 7,750 |
| nl_exact | 3,819 | 3,907 | 7,417 | 7,744 | 7,851 |
| condensed | 1,450 | 1,500 | 2,886 | 3,007 | 3,014 |

longdoc was length-matched to the band-25 logic histogram. p50 within 1.0%,
mean within 0.6%, max within 0.3%, all 30 populated 250-token bins within ~3%.
Zero documents exceeded the 8192 window in any arm, zero split documents.
Epochs over the proof set: logic ~0.90, nl_exact ~0.89, condensed 2.31.

### 2.2 Greedy graded-deduction readout
Source: `$HPCVAULT/synthetic-RLVL/lm_eval_results/qwen25_longwin_graded_deduction_20260906/`,
metric `exact_match,none`. ProofWriter n=500, BranchProof n=200.

| task | control | longdoc | logic25 | nl_ex25 | cond25 |
|---|---|---|---|---|---|
| pw_d0 | 0.442 | 0.458 | 0.480 | 0.488 | 0.460 |
| pw_d1 | 0.328 | 0.322 | 0.358 | 0.388 | 0.346 |
| pw_d2 | 0.440 | 0.444 | 0.494 | 0.562 | 0.460 |
| pw_d3 | 0.444 | 0.454 | 0.500 | 0.570 | 0.480 |
| pw_d5 | 0.460 | 0.440 | 0.502 | 0.564 | 0.474 |
| bp_cot_d5 | 0.050 | 0.075 | 0.315 | 0.525 | 0.345 |
| bp_cot_d10 | 0.035 | 0.110 | 0.175 | 0.345 | 0.285 |
| bp_cot_d15 | 0.110 | 0.120 | 0.275 | 0.305 | 0.185 |
| bp_cot_d20 | 0.090 | 0.150 | 0.215 | 0.245 | 0.205 |
| bp_cot_d25 | 0.065 | 0.120 | 0.195 | 0.295 | 0.240 |

On ProofWriter, longdoc is flat against control (0.458/0.322/0.444/0.454/0.440
vs 0.442/0.328/0.440/0.444/0.460). On BranchProof-CoT longdoc is above control.
Under greedy, nl_exact beats logic on 13 of 15 tasks.

### 2.3 Sampled readout (pass@k), BranchProof-CoT only
Source: `$HPCVAULT/synthetic-RLVL/lm_eval_results/qwen25_longwin_passk_20260907/`.
n=16 samples, T=0.8, top_p=0.95, seed 20260806, all five audits accepted=True,
n=200/task, zero empty and zero truncated generations.
Scorer imports the same `process_deduction_bp_cot` used by the greedy eval and
was verified to reproduce all 15 published greedy values exactly.

pass@16:
| depth | control | longdoc | logic25 | nl_ex25 | cond25 |
|---|---|---|---|---|---|
| d5  | 0.480 | 0.715 | 0.880 | 0.845 | 0.825 |
| d10 | 0.515 | 0.630 | 0.790 | 0.720 | 0.850 |
| d15 | 0.465 | 0.625 | 0.755 | 0.615 | 0.790 |
| d20 | 0.535 | 0.640 | 0.620 | 0.580 | 0.695 |
| d25 | 0.485 | 0.560 | 0.565 | 0.600 | 0.700 |

Termination (tag_rate), greedy: control 0.645-0.785, longdoc 0.795-0.925,
logic 0.940-0.995, nl_exact 1.000, condensed 0.955-0.985.

logic minus nl_exact, by decoding:
| depth | greedy | pass@16 | maj@16 |
|---|---|---|---|
| d5  | -0.175 | +0.035 | -0.140 |
| d10 | -0.165 | +0.070 | -0.030 |
| d15 | -0.055 | +0.140 | -0.005 |
| d20 | -0.040 | +0.040 | +0.025 |
| d25 | -0.105 | -0.035 | -0.070 |

### 2.4 Paired bootstrap over documents (4000 reps, pooled d5-d25, pass@16)
| contrast | diff | 95% CI | P(diff<=0) |
|---|---|---|---|
| logic - nl_exact | +0.050 | [0.013, 0.087] | 0.003 |
| condensed - nl_exact | +0.100 | [0.064, 0.136] | <0.001 |
| condensed - logic | +0.050 | [0.015, 0.085] | 0.003 |
| logic - control | +0.226 | [0.186, 0.265] | <0.001 |
| nl_exact - control | +0.176 | [0.139, 0.214] | <0.001 |
| condensed - control | +0.276 | [0.238, 0.313] | <0.001 |
| longdoc - control | +0.138 | [0.104, 0.171] | <0.001 |
| logic - longdoc | +0.088 | [0.050, 0.125] | <0.001 |
| condensed - longdoc | +0.138 | [0.104, 0.172] | <0.001 |

Deepest bands only (d20, d25): logic - nl_exact +0.003 [-0.057, 0.065], p=0.477.
condensed - nl_exact +0.107 [0.050, 0.165].
pass@1 (mean sampled), pooled: logic - nl_exact +0.025 [0.015, 0.036], p<0.001.

### 2.5 Training-run variance (the limit on the above)
Source: `lm_eval_results/qwen25_mixdepth_graded_deduction_20260826/`, the same
eval suite, 5 conditions x 2 SFT seeds (3407, 3408).
Seed-to-seed |difference| of the SAME condition: mean 0.0229, median 0.0110,
max 0.1350, n=50 task-condition pairs. Implied per-run sd ~0.019, so the sd of
a difference between two independently seeded arms is ~0.027.

Consequence. The document bootstrap in 2.4 estimates evaluation sampling error
only. Combining it with the training term gives sd ~0.033 for logic - nl_exact,
so the honest interval is about [-0.015, +0.115] and includes zero. The
contrasts against control (0.176 to 0.276) are six to ten times the training sd
and are not at risk. The logic - nl_exact contrast at n=1 seed is.

## 3. Known-false claims in the current draft, to be removed
- The abstract attributes the transfer null to mid-proof document splitting.
  The docpack rerun (2026-08-19) answered that negatively: MACRO10 0.5815 /
  0.5835 / 0.5847 for control / logic / nl, flat, matching the 5B result. The
  document-preserving objective did not rescue transfer.
- The paper calls the transfer study preregistered. The execution plan of
  2026-08-04 does not name depth as a transfer moderator. The depth threshold
  was established 2026-07-22, before the 2026-08-06 transfer readout, so
  "established before" is accurate and "preregistered" is not.
- The claim that trace conditions roughly double answer-format compliance
  weakened in the rerun and is partly template echo.

## 4. Status of work still running (2026-09-07)
- 4195939: post-SFT seed 3408 for all five arms, to give n=2 on 2.4.
- 4195998: fresh RL corpus, BranchProof band 25, seed 20260907, disjoint from
  the midtrain corpus (seed 20260830).
- 4196011: 20-step GRPO smoke, chained on the corpus build.
- Planned: 6 GRPO runs, logic and nl_exact x three reward definitions
  (correct_only, correct_times_valid, correct_plus_valid), LoRA r16, 300 steps.
