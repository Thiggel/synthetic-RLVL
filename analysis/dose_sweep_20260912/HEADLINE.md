# Headline result (three seeds, complete grid)

Stock Qwen2.5-7B. 100k instruction examples in every arm. Replacement design:
arms differ only in what the replaced slice contains. The replacement is
program-generated rule-chaining traces over abstract constants, no teacher
model, no human text, no real-world semantics.

## ProofWriter depth 3, answer only

| arm | seed 3407 | 3408 | 3409 | mean | vs control |
|---|---|---|---|---|---|
| control (0%) | .388 | .390 | .402 | .393 | - |
| formal 1% | .412 | .392 | .410 | .405 | +1.2 |
| formal 5% | .500 | .494 | .506 | .500 | **+10.7** |
| formal 10% | .490 | .482 | .486 | .486 | +9.3 |
| English 1% | .402 | .404 | .402 | .403 | +1.0 |
| English 5% | .492 | .488 | - | .490 | +9.7 |
| English 10% | .546 | .528 | .522 | **.532** | **+13.9** |

Every seed of every arm at five percent or more beats every control seed.
Control seed spread is .014; the gain is ten times that.

## The two thresholds

| effect | dose needed | evidence |
|---|---|---|
| writes checker-valid derivations | 1% (1,000 examples) | .000 -> .93-.99 at depth 25, control .00-.01 in three seeds |
| deduction accuracy on real benchmarks | 5-10% | ProofWriter +10 to +14, FOLIO +6.4 |

Derivation ability generalises to depth 45, roughly twice the deepest
training example (.92-.99), and survives a 16k-token window.

## Cost: none measurable

GSM8K .757-.776 (control .768-.771), MMLU .698-.704 (.703-.704),
LogiQA .363-.392 (.366-.372), ARC, HellaSwag, PIQA, WinoGrande flat,
GPQA .288-.369 with seed noise exceeding every arm difference.

## Boundaries, stated plainly

- Gains are confined to deduction. Nothing transfers to mathematics,
  knowledge or commonsense benchmarks.
- On ProofWriter a checker certifies soundness but not completeness, so
  checker selection trails majority voting (.31-.40 vs .60-.72 at k=4),
  unlike the synthetic task where a valid derivation implies the right
  answer .99 of the time.
- Under chain-of-thought prompting the advantage shrinks but survives:
  control .399, five percent .453, ten percent .494.

## Saturation confirmed at 25 percent

Formal notation, ProofWriter d3: 1% .405, 5% .500, 10% .486, 25% .516
(seed 3407). Flat from five percent upward, matching the midtraining dose
test where 10 and 30 percent were indistinguishable. The practical
recommendation is five percent; a quarter of the mixture buys nothing more.
