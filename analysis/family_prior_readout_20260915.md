# Answer calibration predicts the sign of the effect (2026-09-15)

ProofWriter depth 3, 500 items, one seed per cell, gold labels are 42.8 percent
false, 39.8 true and 17.4 unknown.

| model | condition | worst excess over gold | accuracy |
|---|---|---|---|
| Qwen2.5-7B | control | true +23.5 | 39.3 |
| Qwen2.5-7B | formal 5% | true +0.2 | 50.0 |
| Qwen2.5-3B | control | true +52.2 | 38.4 |
| Qwen2.5-3B | formal 5% | true +52.0 | 41.8 |
| Llama-3.1-8B | control | unknown +26.8 | 37.6 |
| Llama-3.1-8B | formal 5% | unknown +48.0 | 29.4 |
| Llama-3.1-8B | English 10% | true +5.4 | 49.2 |

Accuracy moves with calibration in every cell measured so far. Where the
replacement pulls the answer distribution onto the label distribution accuracy
rises, where it pushes the distribution further out accuracy falls, and where
it leaves the distribution alone accuracy barely moves.

Two consequences for the paper.

The derivation-writing ability is independent of this. Every treated model
writes checkable derivations, 99.3 percent for Qwen2.5-7B, 96.0 for Qwen2.5-3B
and 99.0 for the Llama arm whose accuracy fell, against essentially zero for
every control. Installing the format and the procedure is robust. Converting it
into answers is not.

The notation is not a fixed property. On Llama the English rendering corrects
the prior and the formal rendering worsens it, which is the reverse of the
ordering on the chain subtasks of BIG-Bench Hard for Qwen. The rendering has to
be chosen against the target model's measured answer distribution.

Open: one seed per cell, a second Llama seed is training, and the OLMo arms
will test the prediction that a well-calibrated control gains little, since its
control carries an excess of only 8.6 points.

## Update, 2026-09-16: the rule covers seven arms and three families

ProofWriter depth 3, 500 items, one seed per cell. Worst excess is the largest
gap between the model's answer distribution and the gold distribution, which is
42.8 percent false, 39.8 true, 17.4 unknown.

| model | arm | worst excess | accuracy |
|---|---|---|---|
| Qwen2.5-7B | control | true +23.5 | 39.3 |
| Qwen2.5-7B | formal 5% | true +0.2 | 50.0 |
| Qwen2.5-3B | control | true +52.2 | 38.4 |
| Qwen2.5-3B | formal 5% | true +52.0 | 41.8 |
| Llama-3.1-8B | control | unknown +26.8 | 37.6 |
| Llama-3.1-8B | formal 5% | unknown +48.0 | 29.4 |
| Llama-3.1-8B | English 10% | true +5.4 | 49.2 |
| OLMo-2-7B | control | true +8.6 | 44.4 |
| OLMo-2-7B | formal 5% | unknown +46.2 | 34.8 |

Accuracy falls as miscalibration rises, with no exception across three families
and two scales. OLMo is the case that refutes the simpler reading. We predicted
a small gain because its control is the best calibrated of the four, and it
lost 9.6 points, because a well-calibrated model has nothing to gain and a
great deal to lose when the injected data carries a distribution of its own.

The derivation-writing ability is unaffected by any of this. Every treated arm
writes proofs a checker accepts, 96.0 to 100 percent at chain length 25,
against essentially zero for every control, including the arms whose accuracy
fell.

What the paper should claim. A tiny quantity of program-generated data reliably
installs checkable derivation writing in every family and scale we measured.
Benchmark accuracy is a separate matter that follows the answer distribution,
so it has to be diagnosed per model and per benchmark before the data is
chosen, and the formal rendering helped only Qwen.

## The instruction corpus installs the prior, shown by manipulation

Same base model, same recipe, same hyperparameters, only the instruction corpus
differs. ProofWriter depth 3, 500 items, one seed.

| corpus | condition | answer distribution | worst excess | accuracy |
|---|---|---|---|---|
| Dolci | control | true .73 | true +23.5 | 39.3 |
| Dolci | plus 5% formal | true .50 | true +0.2 | 50.0 |
| Tulu 3 | control, 2 seeds | false .82 | false +39.6 | 53.6 |
| Tulu 3 | plus 5% formal, seed 3407 | false .71 | false +28.6 | 55.6 |
| Tulu 3 | plus 5% formal, seed 3409 | false .70 | false +27.0 | 57.2 |
| Tulu 3 | plus 5% formal, seed 3408 | as control | -- | 51.0 |

The two corpora install opposite priors on the same base model, which is the
paper's causal claim shown by intervention instead of inferred from one corpus.
The same five percent of derivations moves both toward the label distribution,
by 23 points of excess on one corpus and 11 on the other, and accuracy rises in
both, by 10.7 and 1.8 points.

Neither control writes a derivation a checker accepts, 0.3 percent and 0.0, in
line with every control measured across three model families.

Three treated seeds: two reach 99 percent on derivations and pull the negative
prior back by 11 to 13 points with accuracy rising, and the third learns
nothing and behaves like a control. Five percent is therefore at the
reliability threshold on this corpus where it is comfortably above it on ours,
which is itself worth reporting, since the required share depends on the
instruction corpus.

Caveat: the Tulu-trained models are verbose and do not reliably stop after a
short answer, so their accuracy is scored on partly degenerate text. The
first-word distribution is unambiguous and the within-corpus comparison is
sound, but the cross-corpus accuracy comparison is not like for like.

## Below one percent the behaviour is unreliable, not merely weaker

Derivations a checker accepts at chain length 25, two seeds.

| documents | share | seed 3407 | seed 3408 |
|---|---|---|---|
| 100 | 0.1% | 29.0 | 0.0 |
| 200 | 0.2% | 50.5 | 3.0 |
| 500 | 0.5% | 84.5 | 0.0 |
| 1,000 | 1% | 93.2 | 93.9 |

The two seeds disagree by 29 and 47 points below 500 documents and agree within
a point at 1,000. ProofWriter accuracy in the same arms differs by about ten
points between seeds, against a control seed spread of 1.4 in the main sweep.

So the correct claim is that the ability is unreliable at every share below one
percent and dependable at one percent, where both seeds exceed 93 percent. At
500 documents one seed reaches 84.5 percent and the other writes nothing, so
there is no threshold to report below 1,000, only a lottery. An earlier reading
of seed 3407 alone suggested a smooth ramp from 100 documents, which the second
seed does not support.

## The broad suite on other families, 2026-09-16

Llama-3.1-8B, mean of two seeds, and Qwen2.5-3B, one seed.

| model | arm | derivations | PW d3 | FOLIO | GPQA | HotpotQA | 2Wiki |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | control | 0.0 | 33.3 | 51.2 | 24.8 | 53.1 | 39.8 |
| Llama-3.1-8B | formal 5% | 99.5 | 36.3 | 48.6 | 25.8 | 54.5 | 39.5 |
| Llama-3.1-8B | English 10% | 100.0 | 48.8 | 48.3 | 26.3 | 52.8 | 40.0 |
| Qwen2.5-3B | control | 0.0 | 38.4 | 40.4 | 30.8 | 44.7 | 33.4 |
| Qwen2.5-3B | formal 5% | 96.0 | 41.8 | 39.4 | 27.3 | 43.2 | 34.3 |
| Qwen2.5-3B | English 10% | 100.0 | 41.8 | 42.9 | 28.3 | 41.1 | 34.9 |

Outside Qwen2.5-7B only two effects replicate, the derivation-writing
capability and the ProofWriter gain. FOLIO loses about three points on Llama in
both treated arms, GPQA moves inside its noise, and the multi-hop datasets do
not move. The broader transfer measured on the 7B model, which includes FOLIO,
BIG-Bench Hard chain subtasks and GPQA quantitative items, is so far specific
to that model.

Llama's control varies by 8.6 points across seeds on ProofWriter against 1.4
for the Qwen control in the main sweep, so single-seed comparisons on that
model are not usable, and an earlier single-seed reading of a loss under the
formal rendering did not survive the second seed.

## BIG-Bench Hard does not transfer outside Qwen2.5-7B, 2026-09-16

Mean over available seeds, one for OLMo and Qwen2.5-3B and two for Llama.

| model | arm | macro | chain (8) | other (19) | web of lies |
|---|---|---|---|---|---|
| Qwen2.5-7B | control | 65.0 | 62.2 | 66.1 | 85.5 |
| Qwen2.5-7B | formal 10% | 67.7 | 68.9 | 67.2 | 97.6 |
| Llama-3.1-8B | control | 68.9 | 67.4 | 69.5 | 99.8 |
| Llama-3.1-8B | English 10% | 69.2 | 69.2 | 69.2 | 100.0 |
| OLMo-2-7B | control | 50.5 | 41.9 | 54.1 | 89.6 |
| OLMo-2-7B | English 10% | 50.7 | 43.6 | 53.7 | 90.0 |
| Qwen2.5-3B | control | 51.7 | 40.6 | 56.4 | 99.6 |
| Qwen2.5-3B | English 10% | 51.9 | 40.6 | 56.7 | 98.8 |

Web of lies is already saturated in three of the four controls, at 99.8, 99.6
and 89.6, against 85.5 for Qwen2.5-7B. The twelve-point gain on that subtask
exists because that model was unusually weak there, and no other model has room
to move. Outside the 7B the chain group changes by at most 1.8 points and the
macro by at most 0.9.

The subtask localisation on Qwen2.5-7B remains correct as an account of what
changed in that model. It is not evidence that the intervention transfers to
general benchmarks in general, and the paper should present it as a repaired
deficit rather than as transfer.

## Derivation writing by chain length, 2026-09-16

Fraction of the generator's own problems where a checker accepts the written
proof and the answer is correct.

| model | arm | chain 5 | chain 15 | chain 25 |
|---|---|---|---|---|
| Llama-3.1-8B | formal 5% | 100.0 | 100.0 | 99.0 |
| Llama-3.1-8B | English 10% | 100.0 | 100.0 | 100.0 |
| Qwen2.5-3B | formal 5% | 99.5 | 99.5 | 96.0 |
| Qwen2.5-3B | English 10% | 100.0 | 100.0 | 100.0 |
| OLMo-2-7B | formal 5% | 24.5 | 27.5 | 0.5 |
| OLMo-2-7B | English 10% | 20.0 | 25.5 | 19.5 |
| every control, every family | | 0.0 | 0.0 | 0.0 |

OLMo is weak at chain 5 as well, where the prompt fits its 4,096-token window
comfortably, so its context limit does not explain the gap. It learns the
behaviour partially where the other three families learn it almost perfectly.

The capability claim is therefore near-universal rather than universal. Three
of four families reach essentially 100 percent at every depth from five or ten
percent of the mixture, the fourth reaches a quarter, and no control in any
family or corpus writes a single accepted derivation.

## Chain length: the procedure needs depth, the calibration does not

Training on band 5 means every training proof has five hops. Band 25 is the
main sweep. Formal rendering, five percent share, one seed so far.

| training band | deriv chain 5 | deriv chain 15 | deriv chain 25 | PW d3 | FOLIO |
|---|---|---|---|---|---|
| 5 | 100.0 | 13.0 | 10.5 | 46.4 | 58.1 |
| 25 | 99.8 | 100.0 | 99.3 | 50.0 | 57.8 |

A shallow curriculum does not generalise upward. Trained at five hops the model
writes accepted derivations on every five-hop problem and almost none at
fifteen or twenty-five, where the band 25 arm holds above 99 percent at every
depth and extends to chain 45, about twice its training depth.

Benchmark accuracy does not need the depth. The band 5 arms reach 46.4 and 49.0
on ProofWriter depth 3 and 58.1 on FOLIO, comparable to the band 25 arms, which
separates the two effects once more. The inference procedure requires
derivations at least as deep as the target, and the answer-prior correction
does not.

Practical consequence: generate at the depth the model has to reach, since
depth is cheap for a program and cannot be recovered afterwards.

## The recipe does not transfer to arbitrary chain-structured tasks

Four new generators, program tracing, reachability with balanced negatives,
temporal ordering and unit propagation, mixed at five percent in the formal
rendering, one seed. Each targets a task the deduction corpus leaves flat.

| benchmark | control | deduction 5% | new corpora 5% |
|---|---|---|---|
| temporal sequences | 74.4 | 72.0 | 70.4 |
| date understanding | 70.8 | 72.0 | 68.8 |
| multistep arithmetic | 88.0 | 90.4 | 88.4 |
| HumanEval | 66.8 | 67.1 | 61.6 |
| MBPP | 65.2 | 64.8 | 65.0 |
| BBH chain group | 60.9 | 68.5 | 64.9 |
| FOLIO | 51.4 | 57.8 | 53.2 |
| GPQA | 31.8 | 33.8 | 31.3 |

Every targeted prediction fails. Temporal ordering was the sharpest one, since
the generator matches the subtask structure and the subtask is flat under
deduction data, and it falls 4 points. Program tracing was aimed at coding and
HumanEval falls 5.2. Unit propagation was aimed at multistep arithmetic, which
does not move.

The new corpora do lift the chain group, 60.9 to 64.9, but by less than the
deduction corpus does at 68.5, on subtasks the deduction corpus was not built
for either.

So matching the latent structure of a generator to a target task is not
sufficient to improve that task. The natural extrapolation from the earlier
results, that a capability can be engineered by writing the right generator, is
not supported. Open: one seed, one share, one rendering, and RULER variable
tracking is still pending.

## English extrapolates beyond its training depth, formal does not

Fraction of the generator's own problems where a checker accepts the proof,
seed 3407, five percent share for formal and ten for English.

| training band | notation | chain 5 | chain 15 | chain 25 |
|---|---|---|---|---|
| 5 | formal | 100.0, 99.5 | 13.0, 13.5 | 10.5, 13.0 |
| 5 | English | 100.0, 100.0 | 86.0, 85.0 | 51.5, 58.5 |
| 15 | formal | 100.0, 100.0 | 100.0, 100.0 | 50.5, 39.5 |
| 15 | English | 100.0, 100.0 | 99.5, 100.0 | 99.0, 100.0 |
| 25 | formal | 99.8 | 100.0 | 99.3 |

Every cell has two seeds, listed in order. They
agree to within half a point at the training depth and within three points
beyond it, so the collapse past the training band belongs to the rendering
rather than to a seed. ProofWriter depth 3 sits at 46 to 51 and FOLIO at 54 to 58 across every band,
notation and seed, against a control of 39.3 and 51.4. The accuracy gains are
therefore indifferent to training depth and to rendering, where the procedure
depends on both.

At both shallow bands the English rendering carries the procedure far past the
depth it was trained on and the formal rendering does not. Trained at fifteen
hops English holds 99.0 at twenty-five where formal drops to 50.5, and trained
at five hops English holds 86.0 at fifteen where formal collapses to 13.0.

Practical reading: if only shallow proofs can be generated, use the English
rendering. If the generator can reach the target depth, either works.

This is a different question from the controlled study on the generator alone,
which compared accuracy at a fixed evaluation depth and found formal
supervision ahead at the deepest band. Performance at the training depth and
extrapolation beyond it have to be reported separately.

## Both new-corpus renderings agree, and deduction data wins on their own targets

| metric | control | deduction F5 | deduction E10 | new formal | new English |
|---|---|---|---|---|---|
| BBH macro (27) | 64.5 | 67.4 | 66.9 | 66.3 | 65.9 |
| BBH chain (8) | 60.9 | 68.5 | 66.2 | 64.9 | 64.3 |
| temporal sequences | 74.4 | 72.0 | 75.2 | 70.4 | 69.2 |
| multistep arithmetic | 88.0 | 90.4 | 90.0 | 88.4 | 89.2 |
| object counting | 83.2 | 80.4 | 80.4 | 85.2 | 84.8 |
| HumanEval | 66.8 | 67.1 | 69.5 | 61.6 | pending |

Temporal ordering had a generator built for it and falls 4 to 5 points in both
new arms, while the deduction English arm raises it to 75.2. Unit propagation
had a generator built for it and multistep arithmetic does not respond. Program
tracing had a generator built for it and HumanEval falls 5.2.

Deduction data therefore outperforms purpose-built data on the tasks the
purpose-built data targets. The two renderings of the new corpora agree with
each other, so this is not a one-off.

The exception is aggregation, where the new corpora gain 2 points on object
counting and the deduction corpus loses 2.8, consistent with the 11-point loss
the deduction corpus causes on long-context word counting.

## Without instruction data the model cannot answer a question

| trained on | deriv chain 25 | PW d3 | PW d5 | FOLIO |
|---|---|---|---|---|
| instruction data only | 0.3 | 39.3 | 41.2 | 51.4 |
| 5 percent derivations in instruction data | 99.3 | 50.0 | 48.9 | 57.8 |
| derivations only, formal, no instruction data | 100.0 | 0.0 | 0.0 | 0.0 |
| derivations only, English, no instruction data | 100.0 | 0.0 | 0.0 | 0.0 |

The zero is a format collapse, not a scoring artefact. Asked a ProofWriter
question the model emits the single token `<formal>` and stops, because the
only document shape it has produced begins that way.

Both renderings collapse identically, so this is not a quirk of symbolic
markup. The English rendering is ordinary sentences and still leaves the model
unable to answer a plain question.

So the two ingredients are each useless alone for this purpose. The
derivations supply the procedure and the instruction mixture supplies the
ability to respond to an arbitrary prompt. The replacement design is not a
convenience, it is the only configuration in which the capability is usable.

## Two-seed cross-family summary, ProofWriter depth 3

| model | control | formal 5% | English 10% |
|---|---|---|---|
| Qwen2.5-7B | 39.3 (3 seeds, spread 1.4) | 50.0 (3) | 53.2 (3) |
| Qwen2.5-3B | 38.2 +- 0.4 (2) | 42.1 +- 0.6 (2) | 41.8 (1) |
| Llama-3.1-8B | 33.3 +- 8.6 (2) | 36.3 +- 13.8 (2) | 48.8 +- 0.8 (2) |
| OLMo-2-7B | 45.8 +- 2.8 (2) | 33.6 +- 2.4 (2) | 57.2 (1) |

The OLMo loss under the formal rendering replicates, 12.2 points below its
control with seed spreads of 2.4 and 2.8, so it is not the single-seed artefact
that the apparent Llama loss turned out to be.

The English rendering helps every model where it has been measured, by 3.4 to
15.5 points, and is the most stable arm in the study, with Llama's two seeds
agreeing to 0.8 points. The formal rendering helps Qwen at both scales, does
nothing measurable on Llama given its variance, and harms OLMo.

Recommendation for the paper: the English rendering is the default, and the
formal rendering is conditional on the model and should be checked with the
answer-prior diagnostic before use.

## The zero-control claim needs a scale qualifier

Qwen2.5-14B, control, derivations a checker accepts.

| chain | accepted |
|---|---|
| 5 | 0.0 |
| 15 | 2.5 |
| 25 | 7.5 |

Unlike every control at 3B, 7B and 8B, which score exactly zero at every depth
across four families and two instruction corpora, the 14B control produces a
few accepted items, and the rate rises with chain length rather than falling.
Reading the samples explains it. The model answers in ordinary prose with a
numbered step-by-step explanation, solves the chain correctly and states the
right answer, and the checker accepts that. Longer problems make it more likely
to lay the reasoning out explicitly, hence the rise with depth.

So the claim should read that no instruction-tuned control below 14B writes a
derivation a checker accepts, and that at 14B a control occasionally reaches an
accepted answer by reasoning aloud instead. The trained format is still absent
from every control.

## Long context: the aggregation damage belongs to the deduction corpus

RULER at 16,384 and 32,768 tokens, mean of the two lengths.

| task | control | deduction 5% | new corpora 5% |
|---|---|---|---|
| single needle | 100.0 | 100.0 | 100.0 |
| multi-query needle | 99.4 | 99.2 | 99.2 |
| multi-value needle | 92.0 | 91.9 | 87.9 |
| variable tracking | 97.0 | 97.3 | 97.1 |
| common word extraction | 58.5 | 47.2 | 56.4 |

Common word extraction is an aggregation task and the deduction corpus costs it
11.3 points where the new corpora cost 2.1, which matches their advantage on
object counting in BIG-Bench Hard. Different synthetic corpora damage different
things.

Variable tracking does not move under either corpus, including the one with a
program-tracing generator built for that structure, so the last targeted
prediction fails alongside temporal ordering, arithmetic and coding.

## The scale axis within Qwen2.5, ProofWriter depth 3

| size | control | formal 5% | English 10% |
|---|---|---|---|
| 3B | 38.2 +- 0.4 (2) | 42.1 +- 0.6 (2) | 41.8 (1) |
| 7B | 39.3 (3) | 50.0 (3) | 53.2 (3) |
| 14B | 33.0 (1) | 39.2 (1) | 43.6 (1) |

The English rendering is ahead of the formal one at every size, and ahead on
every family measured, by 3.4 to 15.5 points over the respective control. There
is no monotone trend in the size of the gain, which peaks at 7B.

At 14B the English arm also gains 4.5 points on GPQA-Diamond where the formal
arm gains 0.5, and loses 3.0 on FOLIO where the formal arm is flat.

## A mixture of experts behaves like the dense models

OLMoE-1B-7B, 64 experts, one billion active parameters, one seed.

| arm | predicted false | true | unknown | worst excess | PW d3 | FOLIO | GPQA |
|---|---|---|---|---|---|---|---|
| control | .308 | .686 | .006 | true +28.8 | 59.6 | 39.4 | 26.8 |
| formal 5% | .000 | .852 | .148 | true +45.4 | 33.4 | 37.4 | 22.2 |

The control produces a genuine three-way distribution and its 59.6 is the
highest control score in the study, from a model with one billion active
parameters. Its derivation cell is 0.0 at every depth, like every other
control.

The treated arm stops producing "false" altogether, its excess rises from 28.8
to 45.4 points and accuracy falls 26.2, the largest single degradation we have
measured. Sparsity therefore does not change the mechanism, only how far the
injected distribution pulls the model.

Across the study the formal rendering at five percent moves the answer
distribution toward the labels on two models, Qwen2.5-7B and Qwen2.5-3B, and
away from them on three, Llama-3.1-8B, OLMo-2-7B and OLMoE.
