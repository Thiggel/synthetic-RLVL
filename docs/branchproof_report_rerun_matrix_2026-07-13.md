# Corrected BranchProof Report Rerun Matrix

Last updated: 2026-07-14 08:12 CEST.

## Scope

The 2026-07-10 closure audit invalidates every old BranchProof result that uses
depth above 17. This includes all report-facing surface, shortcut, hybrid,
conditioned-dual, architecture, batch-size, 32B, and tiny-scratch families.
The corrected source is private Hub dataset `flaitenberger/BranchProof-unique-v2`.

## Submitted matrix

| Family | Train | Eval | Rows |
| --- | --- | --- | ---: |
| Surface/syntax | `3850105` | `3850116` | 27 |
| Shortcut schema/position/marker | `3850212 -> 3850213 + 3854948_[3-4]` | `3850214` | 42 |
| Hybrid order | `3850107` | `3850118` | 30 |
| Conditioned dual 10k | `3850108` | `3850119` | 15 train / 30 eval |
| Conditioned dual 50k | `3850109..3850112` | `3850120` | 15 train / 30 eval |
| Architecture | `3850113` | `3850121` | 54 |
| Batch size | `3850114` | `3850122` | 36 train / 48 eval |
| OLMo-3/Qwen3 32B | `3850115` | `3850123` | 15 train / 18 eval |
| Tiny scratch, 100k unique examples | `3850488 -> 3850490` | `3850492/3850493` | 18 final / 90 checkpoints |

Every report comparison has three seeds. Exact duplicate compact-logic rows
reuse the corrected baseline only when model, train range, seed, steps, and
data surface match exactly.

At 08:12 CEST on July 14, shortcut tasks `3850213_3/4` were found canceled
before Python startup. Exact recovery `3854948_[3-4%2]` started on A40s, and
eval `3850214` was changed from the now-unsatisfiable original `afterok` gate
to `afterany:3850213,afterok:3854948`. No successful row is being rerun.

At 17:58 CEST, surface, hybrid, conditioned-dual, and architecture rows were
actively training. All seven shortcut corpus builds had passed; their SFT
array was pending on scheduler capacity. Tiny build `3850394` rejected six
sequence collisions; deduplicating CPU-only replacement `3850488` completed
with exact local and remote 100k-row gates. Train `3850490` is released. No
other active row showed a fatal error.

## Independent truncation findings

- Hybrid targets average about 10k OLMo tokens at depth 25; old 8192-token SFT
  truncated the second modality and often the answer. Corrected hybrid SFT and
  eval context use 16384.
- Tiny depth-10 logic/NL sequences average about 2.7k tokens, so old 2048 SFT
  truncated targets. A second audit found that the initial corrected launch
  would have run 100k optimizer steps at effective batch 16 over only 50k
  rows, or about 32 corpus passes. Jobs `3850072..3850078` were canceled before
  one pass. The replacement uses 100k distinct paired examples exactly once:
  6,250 steps at effective batch 16, 4096-token SFT, 16384 context, checkpoints
  every 1,250 steps (20k examples), and a runtime no-reuse guard. Its builder
  oversamples then rejects any collision on either the complete formal or NL
  training-sequence fields while preserving exactly 10k rows per depth.
- Other audited depth-25 surfaces fit the 8192 SFT cap. Rule-annotated and
  pseudocode are longest at roughly 7.1k and 7.3k tokens respectively.

## Acceptance

No family is reportable until all expected rows pass structural/metric gates
and representative shallow, boundary, OOD, and depth-50 generations have been
inspected across seeds and success/failure cases. Old and corrected numbers
must never be pooled.
