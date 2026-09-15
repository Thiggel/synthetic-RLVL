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
