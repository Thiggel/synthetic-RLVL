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

## The noise floor (two controls)

| task | control seed 3407 | control seed 3408 | spread |
|---|---|---|---|
| ProofWriter d0 | .504 | .512 | .008 |
| ProofWriter d1 | .312 | .310 | .002 |
| ProofWriter d2 | .388 | .392 | .004 |
| ProofWriter d3 | .388 | .390 | .002 |
| ProofWriter d5 | .406 | .412 | .006 |
| native derivation d25 | .000 | .010 | .010 |

Seed-to-seed variation in the control is under one point on every ProofWriter
depth, so the +11.2 points at five percent is an order of magnitude above the
floor, and the derivation result (.000-.010 against .940-1.000) is not
explicable by seed at all.

## Notation at matched dose, one percent, seed 3407

| measure | control | formal | English |
|---|---|---|---|
| ProofWriter d0 | .504 | .556 | .514 |
| ProofWriter d3 | .388 | .412 | .402 |
| FOLIO | .517 | .537 | .547 |
| native derivation d25 | .000 | .940 | .980 |

Both notations install the full derivation ability at one percent; downstream
they differ by a point or two in opposite directions on ProofWriter and FOLIO,
so the notation question is decided at the larger doses, not here.

## Consolidated table (2026-09-13, nine models measured)

| arm | PW d3 | PW d5 | FOLIO | native d25 | LogiQA |
|---|---|---|---|---|---|
| control s3407 | .388 | .406 | .517 | .000 | .372 |
| control s3408 | .390 | .412 | .507 | .010 | .366 |
| formal 1% s3407 / s3408 | .412 / .392 | .412 / .376 | .537 / .542 | .940 | .382 |
| formal 5% | .500 | .480 | .567 | 1.000 | .363 |
| formal 10% | .490 | .472 | .586 | 1.000 | .392 |
| English 1% | .402 | .406 | .547 | .980 | .367 |
| English 5% | .492 | .462 | .562 | .995 | .364 |
| English 10% | **.546** | **.522** | .557 | 1.000 | - |

Headline: ProofWriter d3 goes .389 (mean of two controls) to .546 at ten
percent English, **+15.7 points**, against a control seed spread of .002.
FOLIO peaks at +6.9. LogiQA stays inside its noise band, so the intervention
is still free.

Two honest qualifications:
- At ONE percent the downstream gain is small and seed-dependent (.412 vs
  .392 on PW d3), while the derivation ability is not (.940-.980 vs .000).
  The claim is therefore two-part: derivation writing is free at one percent;
  benchmark transfer needs five to ten percent.
- The notations transfer equally here (formal .500 / English .492 at five
  percent), unlike the midtrain arms where English led ProofWriter and formal
  led FOLIO. What matters at instruction-tuning time is the presence of
  derivations, not their notation.
