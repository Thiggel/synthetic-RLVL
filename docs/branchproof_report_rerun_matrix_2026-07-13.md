# Corrected BranchProof Report Rerun Matrix

Last updated: 2026-07-15 13:06 CEST.

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
| Tiny scratch, 100k unique examples | `3850488 -> 3850490` | final `3850492`; checkpoint replacement `3854813` | 18 final / 90 checkpoints |

Every report comparison has three seeds. Exact duplicate compact-logic rows
reuse the corrected baseline only when model, train range, seed, steps, and
data surface match exactly.

Batch-size rows `3850114_3/4/5` reached approximately
`9472/9416/9472` of 10,000 steps and then hit the 24-hour ceiling. Their
launches predated the 1,000-step checkpoint change and retained no resumable
state. Exact recovery `3859299_[3-5%3]` is running with the current checkpoint
policy. Eval `3850122` now waits on terminal original array `3850114` plus
successful recovery `3859299`; no successful row is rerun.

At 08:12 CEST on July 14, shortcut tasks `3850213_3/4` were found canceled
before Python startup. Exact recovery `3854948_[3-4%2]` started on A40s, and
eval `3850214` was changed from the now-unsatisfiable original `afterok` gate
to `afterany:3850213,afterok:3854948`. No successful row is being rerun.
At 08:14, missing-tokenizer replacement `3854813` was widened from
A100-80-only to compatible `a40,a100`; rows `0..2` started immediately on
A40s under the unchanged three-row throttle.

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

Tiny scratch is the first accepted family. Final eval is `18/18`; checkpoint
eval/recovery `3854813 + 3856145` is `90/90`. Every checkpoint passed exact
metric/sample/chunk-log/fresh-constant coverage and the strengthened
citation-free-valid diagnostic invariant. Representative raw review spans all
sizes, templates, and seeds plus 20k/60k/100k exposure. Answer-only OOD signal
is nonzero, but modality-appropriate OOD and depth-50 joint pass@1/4/8 is zero
throughout the three-seed aggregates. This is a negative under-capacity
mechanism diagnostic, not evidence for the main 7B narrative. After the gate,
all 90 intermediate checkpoints were deleted under a final/output/job guard;
18 finals and all curve artifacts remain.
