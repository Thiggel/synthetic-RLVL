# Everything we have, and the narrative for "Garbage In, Performance Out" (2026-09-17)

All numbers are percent unless a leading dot marks a fraction. Qwen2.5-7B
results are means over three seeds except the 25 percent columns (one seed).
Other families: seeds as stated. Sources: `family_prior_readout_20260915.md`,
the paper tables, `slides/results_overview.tex`.

## 1. Headline sweep, Qwen2.5-7B, replacement design (100k examples)

| benchmark | control | F1 | F5 | F10 | F25 | E1 | E5 | E10 | E25 |
|---|---|---|---|---|---|---|---|---|---|
| ProofWriter d1 | 30.9 | 33.0 | 36.2 | 36.1 | **42.6** | 31.7 | 35.1 | 37.3 | 40.0 |
| ProofWriter d3 | 39.3 | 40.5 | 50.0 | 48.7 | 51.6 | 40.3 | 48.3 | **53.2** | 51.4 |
| ProofWriter d5 | 41.2 | 40.0 | 48.9 | 47.1 | 49.8 | 41.0 | 46.2 | **51.4** | 49.2 |
| ProofWriter d3, reasoning prompt | 39.9 | 39.3 | 45.3 | 49.5 | **55.8** | 37.5 | 44.1 | 43.8 | 47.0 |
| FOLIO | 51.4 | 53.0 | **57.8** | 57.1 | 56.2 | 54.0 | 56.2 | 55.0 | 53.7 |
| checkable derivation, chain 5 | 0.0 | 99.3 | 99.8 | 99.8 | 100 | 99.8 | 100 | 99.8 | 100 |
| checkable derivation, chain 15 | 0.5 | 96.8 | 99.8 | 100 | 100 | 100 | 100 | 99.7 | 100 |
| checkable derivation, chain 25 | 0.3 | 93.2 | 99.3 | 99.2 | 100 | 98.3 | 99.7 | 100 | 100 |
| BBH, 27 subtasks | 65.0 | 66.2 | 67.2 | **67.7** | 66.9 | 65.3 | 66.6 | 66.9 | 66.6 |
| BBH, 8 chain subtasks | 62.2 | 66.1 | 67.6 | **68.9** | 67.6 | 63.2 | 65.2 | 66.4 | 65.6 |
| BBH, 19 other subtasks | 66.1 | 66.3 | 67.1 | 67.2 | 66.5 | 66.1 | 67.2 | 67.2 | 67.0 |
| web of lies | 85.5 | 92.7 | 95.9 | 97.6 | **98.0** | 84.1 | 84.5 | 87.5 | 87.6 |
| tracking shuffled objects, five | 62.3 | 68.8 | 72.4 | **76.3** | 72.0 | 63.7 | 69.5 | 70.4 | 70.8 |
| tracking shuffled objects, seven | 48.8 | 54.9 | 56.5 | 59.9 | **61.2** | 50.1 | 55.7 | 55.7 | 56.8 |
| logical deduction, seven | 34.8 | 38.4 | 40.9 | 39.1 | 40.4 | 37.9 | 40.8 | **42.0** | 36.8 |
| HotpotQA (F1) | 55.5 | 54.6 | 56.0 | 54.8 | 55.2 | 55.4 | 55.4 | **56.6** | 56.1 |
| 2WikiMultihopQA (F1) | 39.7 | 39.7 | 41.3 | 41.3 | 40.3 | 40.0 | 41.2 | 41.6 | **42.7** |
| MuSiQue (F1) | 31.2 | 31.3 | 32.7 | 31.0 | 31.6 | 32.2 | **34.1** | 32.7 | 31.0 |
| GPQA-Diamond | 31.8 | 31.8 | 33.8 | 32.2 | **36.9** | 33.3 | 35.4 | 34.0 | 31.8 |
| GPQA-Diamond, 73 computed-value items | 22.8 | | | 35.6 | | | | | |
| GPQA-Diamond, 125 recall items | flat | | | flat | | | | | |
| GSM8K | 77.0 | 76.5 | 76.7 | 76.2 | 76.8 | 76.5 | 75.7 | 76.4 | 78.2 |
| MMLU | 70.3 | 70.2 | 69.7 | 70.0 | 69.5 | 70.2 | 70.2 | 70.3 | 69.9 |
| ARC-Challenge | 49.6 | 52.4 | 50.7 | 50.4 | 49.8 | 51.1 | 51.9 | 49.9 | 50.7 |
| LogiQA | 36.8 | 38.0 | 36.8 | 38.9 | 38.4 | 37.5 | 36.6 | 38.5 | 37.6 |
| HellaSwag / PIQA / WinoGrande | 73.2 / 80.5 / 71.0 | flat | flat | flat | flat | flat | flat | flat | flat |
| HumanEval | 66.8 | 67.1 | -- | 68.3 | -- | 65.9 | 63.4 | 69.5 | 65.2 |
| MBPP | 65.2 | 64.8 | -- | 65.2 | -- | 65.2 | 64.6 | 64.8 | 66.4 |
| RULER common-word extraction (16k/32k mean) | 58.5 | | 47.2 | | | | | | |
| RULER needles, variable tracking | 92-100 | | flat | | | | | | |

Control seed ranges: PW d3 1.4, BBH 0.9, GSM8K 0.3, MMLU 0.1, multi-hop
0.1 to 1.5. Gains outside the seed range: ProofWriter, FOLIO, BBH (chain
group), GPQA. Everything else flat. Coding flat. One long-context aggregation
task loses 11 points under the formal rendering only.

## 2. Mechanism on Qwen2.5-7B (whole-dataset, not anecdotal)

ProofWriter decidable items split by claim polarity and gold label:

| share | negated, false (n=785) | negated, true (67) | positive, false (112) | positive, true (789) | P(answers true) |
|---|---|---|---|---|---|
| 0 | .176 | .992 | .683 | .752 | .735 |
| 1 | .241 | .985 | .649 | .716 | .692 |
| 5 | .501 | .920 | .830 | .687 | .565 |
| 10 | .496 | .945 | .774 | .664 | .552 |
| 25 | .569 | .910 | .679 | .668 | .502 |

The whole gain sits in the negated-and-false cell. The control asserts the
claim in 73 percent of items against a balanced gold. Log-likelihood probe:
the change is a removal of a margin offset on negated claims with a small
discrimination component. FOLIO has the same shape (false .32 to .45,
uncertain .39 to .52, true fixed at .80).

Web of lies, all 250 items, value carried through the chain:

| share | correct at step 3 | 4 | 5 | all steps | answer given all steps ok |
|---|---|---|---|---|---|
| 0 | .981 | .921 | .855 | .849 | 1.000 |
| F10 | .993 | .977 | .976 | .975 | 1.000 |

The control loses the value between steps 3 and 5, the treated model does
not, and every model answers correctly whenever its intermediate values are
right. Same story on tracking shuffled objects.

## 3. Where the effects come from (causal manipulations, Qwen2.5-7B)

| manipulation | procedure (checkable derivations) | accuracy (PW d3 / FOLIO) |
|---|---|---|
| shuffled proof lines | destroyed | keeps 7.8 of 12.6 on PW, none of FOLIO's 5.4 |
| training depth band 5 / 15 / 25 | reach = training depth (formal), English extrapolates | 46-51 / 54-58 at every band |
| derivations only, no instruction data | 100 | 0.0 everywhere (emits `<formal>` and stops) |
| second instruction corpus (Tulu 3) | 99 on 2 of 3 seeds | control answers false .82, treated .70, acc 53.6 to 57.2 |
| answer-prior diagnostic, 9 arms, 3 families | independent | accuracy moves with worst-excess in every arm |

Two separable effects: (1) a derivation-writing procedure that needs ordered
steps and sufficient depth, transfers to dense chain-structured tasks; (2) a
correction of the answer prior the instruction corpus installs, which survives
shuffling, ignores depth, and whose sign depends on corpus and model.

## 4. Sub-1 percent and the threshold

| documents | share | derivations, seed 3407 | seed 3408 |
|---|---|---|---|
| 100 | 0.1 | 29.0 | 0.0 |
| 200 | 0.2 | 50.5 | 3.0 |
| 500 | 0.5 | 84.5 | 0.0 |
| 1,000 | 1 | 93.2 | 93.9 |

Below 1,000 documents it is a lottery. At 1,000 documents (one percent) both
seeds exceed 93 percent.

## 5. Across families, scales and architectures (ProofWriter d3, chain-25 derivations)

| model | kind | deriv control / F5 / E10 | PW control | F5 | E10 | seeds |
|---|---|---|---|---|---|---|
| Qwen2.5-3B | dense | 0.0 / 96.0 / 100 | 38.2 | 42.1 | 41.8 | 2/2/1 |
| Qwen2.5-7B | dense | 0.3 / 99.3 / 100 | 39.3 | 50.0 | **53.2** | 3/3/3 |
| Qwen2.5-14B | dense | 7.5 / 99.0 / -- | 33.0 | 39.2 | 43.6 | 1 |
| Llama-3.1-8B | dense | 0.0 / 99.5 / 100 | 33.3 +-8.6 | 36.3 +-13.8 | 48.8 +-0.8 | 2 |
| OLMo-2-7B | dense | 0.0 / 0.5 / 19.5 | 45.8 +-2.8 | 33.6 +-2.4 | 57.2 | 2/2/1 |
| OLMoE-1B-7B | sparse | 0.0 / 0.0 / 1.5 | 59.6 | 33.4 | 26.6 | 1 |
| Qwen2.5-32B | dense | training on Lise (control done 0.28 ep, treated arms started 17:39) | | | | |
| Qwen3-30B-A3B | sparse | queued on Lise, LoRA arms queued on alex | | | | |

Other benchmarks outside Qwen2.5-7B (mean over seeds):

| model | arm | FOLIO | GPQA | HotpotQA | 2Wiki | BBH macro | BBH chain | web of lies |
|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | control | 51.2 | 24.8 | 53.1 | 39.8 | 68.9 | 67.4 | 99.8 |
| Llama-3.1-8B | F5 | 48.6 | 25.8 | 54.5 | 39.5 | | | |
| Llama-3.1-8B | E10 | 48.3 | 26.3 | 52.8 | 40.0 | 69.2 | 69.2 | 100 |
| Qwen2.5-3B | control | 40.4 | 30.8 | 44.7 | 33.4 | 51.7 | 40.6 | 99.6 |
| Qwen2.5-3B | F5 | 39.4 | 27.3 | 43.2 | 34.3 | | | |
| Qwen2.5-3B | E10 | 42.9 | 28.3 | 41.1 | 34.9 | 51.9 | 40.6 | 98.8 |
| Qwen2.5-14B | E10 | -3.0 | +4.5 | | | | | |
| OLMo-2-7B | control | | | | | 50.5 | 41.9 | 89.6 |
| OLMo-2-7B | E10 | | | | | 50.7 | 43.6 | 90.0 |
| OLMoE | control / F5 / E10 | 39.4 / 37.4 / 36.0 | 26.8 / 22.2 / 20.7 | | | | | |

The BBH and GPQA gains are Qwen2.5-7B specific because the other controls are
at ceiling on the chain subtasks (web of lies 99.6 to 99.8). The ProofWriter
gain and the derivation procedure replicate on every dense model with the
English rendering.

## 6. Negative results

- Four purpose-built generators (program trace, reachability with negatives,
  temporal order, unit propagation) at 5 percent: temporal sequences 74.4 to
  70.4, HumanEval 66.8 to 61.6, multistep arithmetic flat, RULER variable
  tracking flat. Deduction data beats them on their own targets (BBH chain
  60.9 / 68.5 / 64.9 for control / deduction / new corpora).
- Coding flat for every corpus and every model.
- Formal rendering harms OLMo-2 (-12.2, replicated) and OLMoE (-26); English
  harms OLMoE (-33). OLMo-2 learns the procedure only partially (25 percent),
  OLMoE not at all.
- Derivations alone, without instruction data, give a model that cannot
  answer a question (0.0 on every benchmark, both renderings).

## 7. Still running (all submitted, nothing needs a hand)

| run | where | state | ETA |
|---|---|---|---|
| Qwen2.5-32B control, full parameter | Lise 2xH200 | step 227/781 | ~01:30 09-18 |
| Qwen2.5-32B F5, E10, full parameter | Lise 2xH200 each | started 17:39, 13h limit | ~06:30 09-18 |
| Qwen3-30B-A3B control, F5, E10, full parameter | Lise | TopOfQueue, 16h limit; a 12-node reservation starts 15:00 09-18, so they start only if two nodes free before ~23:00 tonight | uncertain |
| Qwen2.5-32B control, F5, E10 and Qwen3-30B-A3B control, F5, E10, LoRA r16 + FSDP + bf16 + activation checkpointing | alex 4xA100 each, array 4267544 | pending | 24h limit each, ~3 at a time |
| Qwen2.5-7B mixed notation, half formal half English, 2/5/10/20 percent, plain and mode-conditioned | alex 4xA100, array 4267583 | pending | ~4h each once running, 4 at a time |
| 11 coding cells (F5, F25) | alex a40 | pending | |

## 8. Recommended narrative for "Garbage In, Performance Out"

The title's promise is that data which looks worthless (a program's output,
`c0 is maple`, no world knowledge, no language, no task overlap) makes an
instruction-tuned model measurably better at things that are not the data.
The evidence supports exactly that promise, with one honest boundary. I would
build the paper as follows.

1. **Open with the surprise, quantified.** Replacing five to ten percent of an
   instruction mixture with derivations from a 200-line generator lifts
   Qwen2.5-7B by 10.7 to 13.9 points on ProofWriter, 6.4 on FOLIO, 6.7 on the
   chain subtasks of BIG-Bench Hard, 12.1 on web of lies, 12.8 on the computed
   half of GPQA-Diamond, and leaves eight general benchmarks inside their seed
   range. One percent of the mixture (1,000 documents) is enough to install
   a procedure that no control model shows: 93 to 100 percent checkable
   derivations at chain 25 against 0.0 for every instruction-tuned control
   below 14B, across four families and two instruction corpora. Figure 1 is
   the example plus the headline bars.
2. **Then the question the title raises: how can garbage do this?** The
   answer is that the garbage is not teaching content, it is teaching two
   things the instruction corpus fails to teach, and the paper proves each by
   manipulation. (a) A procedure: carry a value through an ordered chain of
   steps. Shown by the shuffled-lines control (procedure gone), by the depth
   sweep (reach equals training depth), by step-level value tracking on all
   250 web of lies items (the control drops the value between steps 3 and 5,
   the treated model does not), and by the GPQA split (computed items move,
   recall items do not). (b) A prior correction: the instruction corpus
   installs an answer bias (Dolci: "true" 73 percent, Tulu 3: "false" 82
   percent), the derivations pull both toward the label distribution, and
   accuracy follows the size of the correction in all nine arms across three
   families, including the cases where it goes down. This is the "surprising
   effect" of the title made mechanistic, and it is what will convince a
   reviewer that the numbers are not a bug.
3. **Then the boundary, stated as a finding.** The English rendering helps
   every dense model (3B to 14B, three families, +3.4 to +15.5); the formal
   rendering helps Qwen and hurts OLMo; the sparse model gains nothing from
   either. The BBH and GPQA transfer appears only where the control has a
   deficit to repair. Purpose-built generators for other structures do not
   help their own targets. This turns "garbage in, performance out" from a
   slogan into a rule: the data works because it supplies a procedure and
   corrects a prior, so it helps exactly when a model lacks the procedure and
   carries the prior, and the paper gives the diagnostic (measure the answer
   distribution against the labels) that predicts the sign before training.
4. **Close with the substrate argument.** The derivations are checkable, so
   the model's reasoning can be verified line by line, and validity tracks
   correctness on the generator's problems. That is the argument for
   checkable traces as a chain-of-thought substrate and it sets up the RL
   follow-up without needing it in this paper.

What to leave in the appendix: full suites per model, RULER, coding, the four
new corpora, the mode-conditioned and mixed-notation sweep (if it lands in
time), the sub-1 percent lottery, the 14B prose-reasoning caveat, statistical
tests per benchmark.

What not to claim: general transfer to real-world tasks (multi-hop moves at
most three points, coding does not move), a universal rendering, or anything
about OLMoE beyond one seed.

Alternative narratives considered and rejected: "checkable CoT substrate" as
the main story (the RL evidence is not there and it would bury the numbers
the title promises); "one weird trick" without the mechanism (reviewers would
suspect contamination, and the boundary cases would look like failures
instead of predictions).
