# Dose sweep: how little synthetic logic data is enough? (2026-09-12)

Stock Qwen2.5-7B, no midtraining. 100k instruction examples in every arm,
REPLACEMENT design, so arms differ only in what the replaced slice contains.
Band-25 BranchProof traces, formal or English notation, at 1, 5, 10 and 25
percent, against a 0 percent control, three seeds. Training
`scripts/slurm/jobs/alex_dose_sweep_sft_2026-09-12.slurm` (array 4219223),
mixtures `datasets/dose_sweep_20260912`, evaluation
`scripts/slurm/jobs/alex_dose_eval_2026-09-12.slurm`.

## Formal notation, seed 3407 (first complete curve)

| share | traces | PW d0 | PW d2 | PW d3 | PW d5 | FOLIO | GPQA | native d5 / d15 / d25 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | .504 | .388 | .388 | .406 | .517 | .323 | .000 / .000 / .000 |
| 1 | 1,000 | .556 | .412 | .412 | .412 | .537 | .308 | .995 / .955 / .940 |
| 5 | 5,000 | .546 | .484 | .500 | .480 | .567 | .369 | .995 / 1.000 / 1.000 |
| 10 | 10,000 | .574 | .488 | .490 | .472 | .586 | .288 | .995 / 1.000 / 1.000 |

Two separable effects:

1. **Derivation ability is essentially free.** One percent takes the model
   from writing nothing a checker accepts to correct, machine-checkable
   51-line derivations at depth 25. The control never writes one.
2. **Benchmark transfer needs a slightly larger dose and then saturates.**
   ProofWriter d3 goes .388 -> .412 -> .500 -> .490 and FOLIO .517 -> .537 ->
   .567 -> .586, i.e. about +11 points on ProofWriter and +7 on FOLIO by five
   to ten percent, with the curve flat afterwards. This matches the midtrain
   dose test, where 10 and 30 percent were indistinguishable.

Costs: none measurable. LogiQA .372 control vs .382 / .363 at 1 and 5 percent,
MMLU .704 / .701 / .698, ARC .497 / .518 / .499, HellaSwag, PIQA and
WinoGrande flat. Note this is the LogiQA cost that derivation MIDTRAINING
reliably shows (-2 to -3 points, both seeds); at instruction-tuning time it
does not appear.

Still running: the 25 percent formal arm, every English arm, and seeds 3408
and 3409. No number above should be quoted without its seed replicate.
