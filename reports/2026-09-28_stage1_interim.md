# Stage 1 interim report: formal-mixture sweep (2026-09-28)

Plan: `docs/research_plan.md`, Stage 1.

**Setup:**
- **Models:** Qwen3.5-Base 0.8B and 2B.
- **Data:** SFT on Dolci-100k, with X% replaced by rlvlgen formal proofs (tag `<formal>`).
- **9B:** p0 is trained; p50, p25 and p10 are training on gruenau11. They will be added later.

## 1. In-domain: unseen generator problems (n=2000 per cell)

![in-domain curves](figures/stage1_indomain_curves.png)

| model | X | faithful | grammatical | valid | answer acc | Dolci loss |
|---|---|---|---|---|---|---|
| 0.8B | 0 | .000 | .000 | .000 | .000 | 1.016 |
| 0.8B | 10 | .232 | .483 | .071 | .501 | 1.018 |
| 0.8B | 25 | .422 | .708 | .213 | .629 | 1.055* |
| 0.8B | 50 | .628 | .861 | .446 | .753 | 1.028 |
| 2B | 0 | .000 | .000 | .000 | .005 | .868 |
| 2B | 10 | .414 | .662 | .252 | .718 | .870 |
| 2B | 25 | .666 | .845 | .470 | .808 | .873 |
| 2B | 50 | .827 | .931 | .675 | .874 | .879 |
| 9B | 0 | .000 | .000 | .000 | .001 | – |
| 9B | 50 | .999 | .999 | .983 | .992 | – |

\*The 0.8B p25 run trained on 1 GPU with grad-accum 128 (a busy-card fallback), so its loss is not directly comparable.

**Findings:**
- All four skills rise monotonically with X and have not saturated at 50%. The 2B model learns about 1.5–2× more per percent than the 0.8B.
- Faithful translation (prompt → premises) is the bottleneck: 2B p50 is 83% faithful but 93% grammatical.
- The cost on held-out Dolci loss is small (+0.011 for 2B at 50%).
- *Update 22:55:* **9B p50 almost solves the in-domain task.** It is 99.9% faithful, 98.3% valid and 99.2% correct on the answer; 2B p50 is at 83% / 68% / 87%. The faithfulness bottleneck is a capacity limit that 9B removes. The figure now draws the full 0.8B and 2B curves (X = 0, 5, ..., 50) and the 9B points; 9B p25 and p10 are still training.

## 2. Transfer without the tag (untagged lm-eval suite, "Lead into Gold" protocol)

![trade-off](figures/stage1_tradeoff.png)

![delta heatmap](figures/stage1_bench_delta_heatmap.png)

**2B:**
- The mixture is essentially free on general benchmarks: the mean Δ stays within ±0.4 points up to 50%.
- On the reasoning set the gain grows with X, reaching +1.1 points at 50%. The main driver is ProofWriter with CoT: +17 (d3) and +14 (d5) at 50%.

**Costs for 2B:**
- BBH logical_deduction_five_objects falls by 16 points at 50% and by 10 at 25%.
- Direct-answer ProofWriter falls by 2–4 points.
- The likely cause is that ordering and constraint puzzles are not a family in the generator, while the model's reasoning style shifts toward formal chains. This is a generator coverage gap to fix.

**0.8B:** the small model pays a capacity cost.
- GSM8K drops by up to 11 points at 50%; MMLU by 1–4 points.
- The general mean Δ is −1.9 at 50%, and there is no PW CoT gain.

**BBH web_of_lies** is 100% for every model and X. The chain-of-thought answers are genuinely correct and the targets are balanced, so this is not a scoring bug; the task is saturated or contaminated for Qwen3.5 and carries no signal.

**Tentative best X (plan criterion: max in-system rate with ≤ 2 points general loss):**
- **2B:** 50%. General Δ is −0.4, so the criterion does not bind yet. Test >50% only if the tagged eval confirms the trend.
- **0.8B:** 25–35%, where the general Δ is about −0.6 to −1.0. At 40–50% the GSM8K loss is too large.

## 3. Tagged eval on real benchmarks (the Stage-2 gate)

This eval asks each benchmark item with `<formal>` and checks the proof. Smoke test 6899 (0.8B p50, 5 items per benchmark, n=210):

| metric | all | system-answerable |
|---|---|---|
| has proof | .91 | 1.00 |
| grammatical | .20 | .38 |
| valid | .03 | .06 |
| in-system correct | .02 | .06 |

ProofWriter: 68% grammatical, 20% valid, 16% in-system.

**Observations:**
- **HotpotQA and MuSiQue:** the model skips the proof and answers directly on the long LongBench contexts. The generator has no long-context retrieval problems.
- **Unfaithful premise translation:** for example, "The tiger chases the dog" was formalised as `sees(tiger, dog)`. The checker's quote grounding only tests substring presence, not whether the formula matches the quote.

**Implication for Stage 2:** at 0.8B the in-system rate on real problems is far below the in-domain rate. The full runs (jobs 6900–6925, all X, 0.8B/2B/9B) will show whether 2B and 9B at high X clear the gate. If they don't, the planned remedy is generator families closer to the RL data (math word problems, ordering puzzles, long-context retrieval) before GRPO.

## Pending
- Full tagged eval for all checkpoints.
- 9B results.
- 2B untagged benches and tagged evals for the last cells (p20 6824/6919 are running). The in-domain eval of every 2B cell is done.

## Update 01:35 (2026-09-29): 2B tagged benchmarks complete (p0–p50)

With 2B p20, p40 and p45 in (7032–7034), all 11 fractions of the 2B tagged sweep are done. Tables are in `analysis/formal_mixture_sweep_20260925/tagged/tagged_2b.md` and curves in `tagged/tagged_curves.png` (`scripts/analysis/formal_mix_tagged_table.py`). All numbers are % of items with the `<formal>` tag, as grammatical / valid / correct / in-system.

| subset | p0 | p5 | p10 | p20 | p30 | p40 | p50 |
|---|---|---|---|---|---|---|---|
| deduction, all (2703) | 0/0/27/0 | 29/6/29/5 | 44/9/34/7 | 53/11/40/9 | 59/13/40/10 | 61/15/41/11 | 63/15/42/11 |
| answerable overall (3187) | 0/0/34/0 | 22/4/38/4 | 35/7/43/6 | 43/9/48/9 | 48/11/49/9 | 48/11/51/10 | 52/12/52/10 |
| GSM8K (500) | 0/0/42/0 | 12/1/38/1 | 26/3/34/1 | 34/5/34/4 | 40/5/35/3 | 44/4/38/3 | 51/6/37/4 |

On deduction, grammaticality, validity and correctness all rise with X and flatten beyond about 30%. Tagged GSM8K accuracy drops 4–8 points at every X > 0, and only 4–6% of its proofs are valid. The tagged math proofs mostly fail the checker, and forcing the format costs math accuracy.

## Update 2026-09-29 ~19:15 — 9B p10 (eval 6844, tagged 6923 on gruenau11)

In-domain (2000 unseen generator problems): 9B p10 faithful .976 / grammatical .997 / valid .944 / answer .979; p25 .963 valid, p50 .983 — at 9B, 10k generator rows (10% of 100k) already give 94% valid proofs, and Dolci eval loss is flat across X (.650 / .651 / .653 / .658). Tagged benchmarks (grammatical / valid / correct / in-system, %): overall 37 / 12 / 0 / 9; ProofWriter 77 / 34 / 55 / 32; FOLIO 66 / 21 / 54 / 15; BBH 25 / 3 / 52 / 1; GSM8K 82 / 12 / 77 / 10; GPQA 1 / 0 / 43 / 0. p25/p50 tagged and all 9B untagged benches are running (gruenau11 lanes). Figures regenerated: `stage1_indomain_curves.png`, `stage1_tradeoff.png`, `stage1_bench_delta_heatmap.png`, `sft_scale_curve.png`, `analysis/formal_mixture_sweep_20260925/tagged/tagged_curves.png`.

## Update 2026-09-29 ~20:20 — 9B tagged p10 / p25 / p50

Tagged overall (grammatical / valid / correct-on-answerable / in-system, %): p10 37 / 12 / – / 9, p25 33 / 12 / – / 11, p50 33 / 12 / – / 11; ProofWriter valid 34 → 40 → 40, FOLIO 21 → 18 → 19, BBH 3 → 2 → 2. At 9B, transfer of valid proofs to benchmarks saturates at the 10–25% mixture; more formal data beyond 25% buys nothing tagged. Table: `analysis/formal_mixture_sweep_20260925/tagged/tagged_9b.md`; figures regenerated.

## Update 2026-09-29 ~21:10 — 9B untagged benchmarks complete (p0 / p10 / p25 / p50)

9B is the only size trained with fp32 master weights throughout (ZeRO-2), so these are the clean Stage-1 numbers. Accuracy (%), delta vs the pure-Dolci p0 in brackets (full table: `analysis/formal_mixture_sweep_20260925/bench/`, figures `stage1_bench_delta_heatmap.png`, `stage1_tradeoff.png`):

| benchmark | p0 | p10 | p25 | p50 |
|---|---:|---:|---:|---:|
| ProofWriter CoT d0–d5 (mean) | 54.4 | 63.6 (+9.2) | 65.7 (+11.3) | 60.7 (+6.3) |
| ProofWriter direct d0–d5 (mean) | 50.3 | 49.2 (-1.1) | 51.6 (+1.4) | 46.5 (-3.8) |
| FOLIO | 65.0 | 67.0 | 66.5 | 64.0 |
| BBH (all) | 84.4 | 84.4 | 83.8 | 84.2 |
| BBH formal_fallacies | 72.8 | 66.0 (-6.8) | 68.4 (-4.4) | 61.6 (-11.2) |
| GPQA-Diamond (n=198, SE ~3.5) | 45.5 | 50.0 | 50.5 | 50.0 |
| MuSiQue | 46.0 | 44.9 | 43.8 | 41.0 (-5.0) |
| GSM8K / MMLU / ARC-C | 87.0 / 78.0 / 60.9 | 86.2 / 77.8 / 60.4 | 86.3 / 77.8 / 61.8 | 86.4 / 77.8 / 61.1 |
| HumanEval / MBPP | 65.2 / 63.4 | 67.1 / 64.2 | 69.5 / 64.4 | 67.7 / 64.2 |

- Untagged general capability is preserved at every X (GSM8K, MMLU, ARC, HellaSwag, PIQA, code all within ±2).
- Multi-step deductive reasoning in natural-language CoT improves (ProofWriter CoT +9–11 at p10/p25, largest at depth 2–3), peaking at p25 and falling back at p50.
- Costs grow with X: BBH formal_fallacies (-7 to -11) and MuSiQue (-5 at p50).
- Stage-1 recommendation at 9B: X = 25% (best CoT transfer, in-domain valid .963, tagged validity already at its plateau, no general-benchmark cost). Single seed; GPQA/FOLIO deltas are within noise.

## 2026-09-29 23:25: p50 x3, checkpoint-781 (first point of the longer SFT, ~50k generator rows)

Run: `qwen35_2b_dolci_rlvlgen_p50_x3` (p50 recipe at 3x the rows, 2342 steps; ZeRO-2 with fp32 master weights, so it does not have the bf16 precision bug). Checkpoint-781 has seen ~50k generator rows, as many as the original p50 final. Its LR is not decayed yet (cosine over 2342 steps). Figure: `figures/sft_scale_curve.png`.

In-domain (2000 unseen generator problems):

| model | generator rows | faithful | grammatical | valid | answer acc |
|---|---|---|---|---|---|
| 2B p50 (bf16 DDP, crippled) | 50k | — | — | .675 | — |
| **2B p50 x3 @781 (fp32 master)** | ~50k | .986 | .996 | **.942** | .976 |
| 9B p10 | 10k | — | — | .944 | — |
| 9B p50 | 50k | — | — | .983 | — |

Dolci gate (`rl_gate_dolci`, answers must also be proven):

| model | has_proof | grammatical | format_ok | valid_eval | valid | in-system | correct |
|---|---|---|---|---|---|---|---|
| SFT p50 | .399 | .082 | .328 | .008 | .004 | .002 | .222 |
| SFT p50 + lemma catalog (fp32) | .598 | .189 | .484 | .011 | .007 | .003 | .224 |
| **SFT p50 x3 @781** | .299 | .134 | .275 | .023 | .007 | .004 | .214 |

**Reading.** With correct precision, the 2B model reaches 9B-level in-domain validity after the same 50k generator rows: .942 vs .675. So most of the earlier 2B/9B gap was the precision bug, not model size. Transfer to Dolci does not follow. valid_eval doubles (.008 → .023, the best SFT so far), but end-to-end valid stays at .007 and the model writes fewer proofs (has_proof .30). In-domain competence is now saturated at 2B. The remaining bottleneck is transfer to natural prompts. That is the Stage-2 GRPO / EI question, not a question of longer SFT. Checkpoints 1562 and final follow (the watcher submits their eval and gate automatically), and the corrected 2B p00/p10/p25/p50 reruns (7146/7215/7217/7147) give the matched mixture curve.

## 2026-09-30 03:45: corrected 2B sweep, first points (p10, p25 with fp32 master weights)

These rerun the 2B mixture sweep without the bf16 rounding bug (2dea2ee). Same data, lr and steps. In-domain rates are on 2000 unseen generator problems:

| run | faithful | grammatical | valid | answer acc |
|---|---|---|---|---|
| p10 bf16-rounded | .414 | .662 | .252 | .718 |
| **p10 fp32m** | **.923** | **.980** | **.845** | **.934** |
| p25 bf16-rounded | .666 | .845 | .470 | .808 |
| **p25 fp32m** | **.970** | **.994** | **.913** | **.964** |
| p50 x3 @781 (fp32, same 50k gen rows as p50) | .986 | .996 | .942 | .976 |
| 9B p10 / p25 | .976 / .992 | .997 / .997 | .944 / .963 | .979 / .989 |

- The steep 2B "more formal data helps" slope in the original sweep was mostly an optimizer artefact.
- With correct precision, 10k generator rows already give valid .85, and 25k give .91. 2B now sits just below 9B.

![scale curve](figures/sft_scale_curve.png)

Dolci gate (950 natural prompts, greedy; `scripts/analysis/grpo_gate_ckpts.py`):

| model | has_proof | grammatical | format_ok | valid_eval | valid | correct |
|---|---|---|---|---|---|---|
| SFT p0 | .000 | .000 | .000 | .000 | .000 | .203 |
| SFT p50 bf16-rounded | .399 | .082 | .328 | .008 | .004 | .222 |
| SFT p10 fp32m | .597 | .173 | .498 | .026 | .005 | .206 |
| SFT p25 fp32m | .402 | .171 | .360 | .021 | .008 | .206 |
| SFT p50 x3 @781 | .299 | .134 | .275 | .023 | .007 | .214 |

- Correct precision roughly doubles grammatical proofs on natural prompts, from .08 to .17.
- Strict validity on natural prompts stays below 1% at every mixture ratio.
- Answer correctness is unchanged versus pure instruction tuning (p0 .203). The formal mix costs no accuracy on the gate.
- Benchmark and tagged results follow once 7222/7225 and 7223/7226 finish.

## 2026-09-30 04:50: corrected 2B p10 / p25, benchmarks

Untagged downstream benchmarks, run exactly as in the sweep (`analysis/formal_mixture_sweep_20260925/bench/bench_2b-fp32m.md`, `bench_curves.png`). The comparison is against the bf16-rounded runs at the same mix. The fp32m pure-instruction baseline (p00 fp32m, job 7146) is still training, so deltas against the proper baseline come later. The bf16-rounded p0 is a weakened baseline too.

| benchmark | p0 bf16 | p10 bf16 | **p10 fp32m** | p25 bf16 | **p25 fp32m** |
|---|---|---|---|---|---|
| ProofWriter d0 / d3 / d5 | 39.6 / 48.8 / 50.6 | 40.0 / 46.2 / 47.8 | **69.6 / 52.2 / 50.4** | 36.6 / 46.0 / 48.4 | **66.0 / 58.6 / 53.2** |
| FOLIO | 45.3 | 43.3 | **50.2** | 43.8 | **50.7** |
| BBH all / chain-8 | 65.2 / 66.4 | 65.5 / 65.7 | 66.1 / 66.2 | 65.2 / 64.8 | 64.6 / 63.7 |
| HotpotQA / 2Wiki / MuSiQue | 47.9 / 35.7 / 26.7 | 46.8 / 35.8 / 27.0 | **49.4 / 37.9 / 28.3** | 45.8 / 34.6 / 26.5 | **49.4 / 38.0 / 29.7** |
| GPQA-Diamond / quant | 36.9 / 37.0 | 35.4 / 32.9 | 38.4 / 49.3 | 36.9 / 32.9 | 37.9 / 39.7 |
| gsm8k | 66.3 | 67.9 | 62.7 | 67.6 | 63.6 |
| mmlu / arc_c / hellaswag | 59.1 / 48.5 / 61.5 | 59.3 / 48.5 / 61.5 | 60.5 / 49.7 / 62.0 | 59.2 / 48.0 / 61.1 | 59.8 / 49.6 / 62.1 |
| humaneval / mbpp | 34.8 / 34.6 | 34.8 / 35.6 | 37.8 / 34.4 | 34.8 / 35.6 | 39.6 / 34.8 |

Format-tagged benchmarks (`tagged/tagged_2b-fp32m.md`). Cells are grammatical / valid / correct / in-system, in %:

| subset | p10 bf16 | **p10 fp32m** | p25 bf16 | **p25 fp32m** |
|---|---|---|---|---|
| overall | 15 / 3 / 0 / 2 | **29 / 7 / 0 / 5** | 19 / 4 / 0 / 3 | **30 / 8 / 0 / 6** |
| deduction | 44 / 9 / 34 / 7 | **78 / 22 / 45 / 18** | 55 / 11 / 40 / 9 | **78 / 24 / 47 / 20** |
| FOLIO | 29 / 2 / 32 / 1 | **62 / 7 / 45 / 5** | 43 / 2 / 42 / 1 | **71 / 7 / 47 / 5** |
| BBH | 4 / 0 / 27 / 0 | 13 / 1 / 30 / 0 | 6 / 0 / 28 / 0 | 15 / 1 / 32 / 1 |
| gsm8k | 26 / 3 / 34 / 1 | **66 / 6 / 41 / 4** | 39 / 5 / 37 / 4 | **74 / 9 / 44 / 5** |

- The precision fix makes the formal mix pay off downstream:
  - ProofWriter d0 +30 points.
  - FOLIO +5–7.
  - Multi-hop QA +1.5–3.
- The tagged valid-proof rate on deduction more than doubles, from 9–11% to 22–24%.
- gsm8k (untagged) drops by about 4 points. This needs the fp32m p00 baseline before reading it as a cost.
- Whether the ProofWriter / FOLIO gains come from the formal data, or partly from correct training in general, is also only settled against fp32m p00.

## 2026-09-30 10:00: 2B fp32m sweep now has its matched 0% baseline

The fp32m p00 run (pure Dolci, same fp32-master recipe) finished bench and tagged evals. This gives the fp32m 10%/25% runs a matched baseline; before this we compared against the bf16 p00. Full tables: `analysis/formal_mixture_sweep_20260925/bench/bench_2b-fp32m.md`, `tagged/tagged_2b-fp32m.md`; figures `bench/bench_curves.png`, `tagged/tagged_curves.png`.

Untagged benchmarks, 2B fp32m (Δ vs fp32m 0%):

| benchmark | 0% | 10% | 25% |
|---|---:|---:|---:|
| PW CoT d3 | 33.8 | 43.8 (+10.0) | 47.8 (+14.0) |
| PW CoT d5 | 23.4 | 31.8 (+8.4) | 35.2 (+11.8) |
| FOLIO | 44.3 | 50.2 (+5.9) | 50.7 (+6.4) |
| GPQA-Diamond | 32.3 | 38.4 (+6.1) | 37.9 (+5.6) |
| BBH (all) | 64.9 | 66.1 (+1.2) | 64.6 (-0.3) |
| gsm8k | 64.8 | 62.7 (-2.1) | 63.6 (-1.2) |
| mmlu | 60.2 | 60.5 (+0.3) | 59.8 (-0.4) |
| humaneval | 36.0 | 37.8 (+1.8) | 39.6 (+3.7) |

- **The gsm8k question is answered:** the earlier −4 came from comparing against the bf16 p00 (66.3). Against the matched fp32m p00 (64.8), the cost is −2.1 (10%) and −1.2 (25%), within about 1–2 SE on 1319 items (SE ≈ 1.3). It is small and not clearly real.
- **The multi-step deduction gains are large and grow with depth:** PW CoT d3/d5 gain +10 to +14 points and FOLIO +6. Standard benchmarks (mmlu, arc, hellaswag, piqa, winogrande) move by at most 0.5.
- **The bf16 2B sweep understated the mixture:** its FOLIO was −1 to −3 and GPQA flat. The fixed fp32m runs show FOLIO +6 and GPQA +6. The precision bug (2dea2ee) mostly erased what the formal data taught.
- **Tagged:** the 0% model never produces `<formal>` (0 grammatical), as expected. At 10%/25%, deduction tagged-correct is 45/47 vs 28 for 0%, but tagged gsm8k-correct drops (69 → 41/44) and bbh multistep_arithmetic too. The formal mode is worse than free text for arithmetic word problems at 2B, so it should stay opt-in (tag) and not be a default.

## 0.8B fp32m: matched 0% vs 25% (2026-09-30)

The 0.8B pair was rerun with fp32 master weights (2dea2ee), so it has a matched 0% baseline like 2B fp32m.
Tables: `analysis/formal_mixture_sweep_20260925/bench/bench_0.8b-fp32m.md`, `tagged/tagged_0.8b-fp32m.md`;
figures: `bench/bench_curves.png`, `tagged/tagged_curves.png`.

Untagged benchmarks, 25% minus 0% (accuracy points):

| bench | Δ |
|---|---:|
| ProofWriter CoT d0–d5 | +11 to +15 |
| ProofWriter non-CoT d1–d5 | −6.6 to −11 |
| FOLIO | +2.5 |
| GPQA-D | +4.0 |
| gsm8k | −1.2 |
| mmlu | −3.4 |
| HotpotQA | −2.7 |

- The CoT deduction gain has the same size as at 2B (+8 to +14). The costs are larger at 0.8B: at 2B, standard benchmarks moved by at most 0.5, while at 0.8B mmlu drops 3.4 and non-CoT ProofWriter drops 6 to 11. A plausible reading is that the smaller model has less spare capacity, so the formal format competes with direct answering.
- Tagged (`<formal>` mode): ProofWriter correct goes from 24 to 44, and grammatical is 82% with 19% valid. gsm8k correct falls from 53 to 25 in formal mode. That is the same pattern as 2B, where formal mode helps deduction and hurts arithmetic word problems.
- Dolci RL gate on the 25% final (950 items): has_proof .884, grammatical .174, valid .024, correct .151 (0% baseline: correct .139). The 2B models have the same bottleneck: the model almost always writes a proof but it rarely checks on out-of-distribution prompts.
- In-domain (generator test split), the proofs are valid .834 at 25k synthetic rows (scale curve).

## 2026-09-30 12:00: p50 x3, checkpoint-1562 (~100k generator rows)

Same run as above, 2/3 through its cosine schedule. Figure: `figures/sft_scale_curve.png` (regenerated).

| checkpoint | gen rows | in-domain faithful | valid | answer acc | gate has_proof | gate grammatical | gate valid | gate correct | gate valid∧correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| x3 @781 | ~50k | .986 | .942 | .976 | .299 | .134 | .023 | .214 | .006 |
| x3 @1562 | ~100k | .996 | .963 | .984 | .401 | .148 | .029 | .224 | .008 |

- In-domain, the model is close to saturated (valid .942 → .963).
- On Dolci, doubling the generator data raises has_proof by 10 points, but valid rises only 0.6 points. So more SFT on the same generator distribution does not close the Dolci validity gap. That gap needs RL, or generator coverage of Dolci-style domains (the open question to the user on extending the lib).

## 2026-09-30 14:20: Dolci gate sampled pass@k (does more SFT data help GRPO?)

The user asked whether more SFT data makes the model know the lemmas better, and so helps RL, even when greedy validity barely moves. The greedy gate cannot answer that. GRPO samples 8 rollouts at T=1.0, so what matters is how many prompts produce at least one valid proof, and how many 8-rollout groups mix passes and fails. That mix is what gives a non-zero advantage.

Setup: 950 Dolci gate items, 16 samples each at T=1.0 with max 2048 tokens, which matches the GRPO rollouts. The metrics are unbiased pass@k and mixed@8 = pass@8(pass) + pass@8(fail) − 1. Tool: `scripts/eval_formal_bench_vllm.py --n-samples 16 --temperature 1.0`. Table: `analysis/passk_ladder.md` (from `scripts/analysis/passk_ladder.py`).

![pass@k ladder](figures/passk_ladder.png)

| model (2B) | has_proof @1/@8/@16/mixed@8 | grammatical | valid | correct | valid∧correct |
|---|---|---|---|---|---|
| SFT x3 @781 (~50k gen rows) | .276/.391/.442/.206 | .090/.225/.261/.216 | .010/.044/.063/.044 | .176/.427/.511/.404 | .003/.013/.019/.013 |
| SFT x3 @1562 (~100k) | .359/.493/.537/.225 | .118/.287/.333/.276 | .010/.047/.066/.047 | .183/.425/.501/.395 | .003/.015/.023/.015 |
| SFT p50 + lemma catalog fp32m | .531/.725/.780/.370 | .107/.307/.371/.303 | .004/.026/.043/.026 | .186/.448/.527/.430 | .002/.013/.023/.013 |
| GRPO G8 final (lc base, 200 steps) | .874/.957/.971/.217 | .141/.341/.405/.317 | .032/.096/.121/.094 | .220/.442/.524/.372 | .020/.055/.065/.053 |

**valid@8 per bench:**

| model | dapo | knowledge | math | wordprob | yesno |
|---|---|---|---|---|---|
| x3 @781 | 0 | .182 | .008 | .065 | .042 |
| x3 @1562 | 0 | .199 | .003 | .056 | .061 |
| lemma catalog | 0 | .061 | .005 | .029 | .055 |
| G8 | 0 | .194 | .005 | .173 | .179 |

**Findings**
- **More SFT data raises has_proof and grammatical, but not valid.** From 50k to 100k rows, has_proof@8 goes .39 → .49 and grammatical@8 goes .23 → .29. valid@8 only goes .044 → .047, and the valid∧correct signal rate goes .013 → .015. On lemmas, the model cites real lib lemmas far more often (67 → 168 cites; see the x3@1562 section). Those extra cites do not become checkable proofs on Dolci. More data mainly makes the model try `<formal>` more often; it does not make it prove more.
- **The lemma catalog raises attempts and lowers validity.** has_proof@8 is .73, but valid@8 drops to .026 (knowledge .18 → .06).
- **GRPO is the only intervention that moves valid@k.** G8 doubles valid@k and roughly quadruples the valid∧correct signal (mixed@8 .013 → .053). Its has_proof is saturated at .96, but its has_proof mixed@8 is lower because nearly every group passes.
- **The reward signal is sparse.** With the cvf reward, about 95% of Dolci prompts give all-zero groups at 8 rollouts, even after GRPO. dapo never produces a valid proof in 16 samples, and math almost never does (≤ .008). The lib has no lemmas for these domains (gcd/mod, geometry, motion).
- **Where proofs are missing:** at x3@781, 98% of proof-less samples end with finish=stop, not length. On math, the model ignores `<formal>` and writes NL CoT.

**Implications for Stage 2**
- More SFT data is not the bottleneck. It may still help RL a little through has_proof, but validity is the binding constraint.
- Candidate next steps:
  - pass@k-filtered prompt sampling: drop prompts with valid@16 = 0 and upweight knowledge/wordprob/yesno.
  - a denser curriculum reward (grammatical → valid → correct).
  - extending the lib to the missing math domains (awaiting a decision from the user).
- Pending: rows for p25 fp32m (~25k, job 7913) and x3 final (~150k, job 7915). They will complete the data-scale ladder.
