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
| Tulu 3 | control | false .82 | false +39.6 | 53.8 |
| Tulu 3 | plus 5% formal | false .71 | false +28.6 | 55.6 |

The two corpora install opposite priors on the same base model, which is the
paper's causal claim shown by intervention instead of inferred from one corpus.
The same five percent of derivations moves both toward the label distribution,
by 23 points of excess on one corpus and 11 on the other, and accuracy rises in
both, by 10.7 and 1.8 points.

Neither control writes a derivation a checker accepts, 0.3 percent and 0.0, in
line with every control measured across three model families.

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
