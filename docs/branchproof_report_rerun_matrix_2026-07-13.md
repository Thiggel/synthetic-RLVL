# Corrected BranchProof Report Rerun Matrix

Last updated: 2026-07-21 19:30 CEST.

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

By 19:30 CEST on July 21, surface rows `3850105_24..26` also completed and
passed the final adapter plus terminal-step-7,140 gate, bringing surface SFT
to `27/27`. The first corrected report artifacts are structurally accepted:
hybrid train-1-to-10 rows `3850118_3/4/5` across all three seeds and
conditioned-10k train-1-to-15 seed-3407 rows `3850119_12/13` for logic/NL.
Each bundle has 448 prompts, 16 generations per prompt, all 14 depths,
complete `7/112` greedy/sampled chunks, 576 retained records, fresh constants,
and no citation-free-valid diagnostic contradiction. Family completion is
hybrid `3/30` and conditioned-10k `14/30`; neither family is reportable yet.

Representative review covers shallow, train-boundary, OOD, depth-50,
success/failure, malformed, and cap-hit cases. Prompts, wrappers, extraction,
and fresh constants match the corrected protocol. Hybrid train-1-to-10
degrades OOD: formal proofs often fail before translated NL traces, while
depth-50 outputs truncate or collapse. Its three-seed greedy OOD answer mean
is `0.3458 +/- 0.0331`; pass@1 OOD answer/formal-joint/translated-joint is
`0.2173 +/- 0.0250`, `0.1294 +/- 0.0422`, and `0.1926 +/- 0.0079`.
Conditioned seed 3407 is clean through depth 25 in both modalities but also
collapses at depth 50; this single-seed partial cannot support a comparison.

The artifact audit had inferred NL-only evaluation solely from `_nl_exact_`
in the checkpoint path, so `conditioned_nl` was falsely required to have
positive formal syntax. It now recognizes every NL-only report surface from
checkpoint and artifact names and continues to formal-audit hybrid surfaces.
All 18 focused tests pass; real row `3850119_13` passes the full structural and
translated-metric gate without `--allow-all-zero-train-metrics`.

By 01:20 CEST on July 21, the resumed wave had added 14 artifact-clean finals:
surface rows 15--20, shortcut rows 19--24, and architecture rows 49/50. Both
targeted replacement arrays `3872657_[15-17]` and `3872658_[19-21]` completed
successfully. Every new final has a nonempty adapter/config, a terminal
step-10,000 trainer state, and no empty file. Family completion is now surface
`21/27`, shortcut `25/42`, and architecture `51/54`. Eleven A40 rows remain
active, while all rebuilt evals are A100 account-GRES or dependency pending.
No corrected eval artifact exists yet, so this remains SFT artifact acceptance
only and does not open a scientific result gate.

Batch-size rows `3850114_3/4/5` reached approximately
`9472/9416/9472` of 10,000 steps and then hit the 24-hour ceiling. Their
launches predated the 1,000-step checkpoint change and retained no resumable
state. Exact recovery `3859299_[3-5%3]` reached roughly step 3.4k and timed
out after writing complete optimizer/scheduler/RNG `checkpoint-3000` states.
Second exact resume `3863546_[3-5%3]` ran from checkpoint 3000 until the
capacity pause described below. No successful row is rerun.

## Four-day capacity pause and restart plan

At 16:30 CEST on July 16, the surface, shortcut, and batch families were
paused to make room for an urgent project. Remaining tasks in training arrays
`3850105`, `3850114`, and `3850213` are user-held. Active surface rows
15--17, shortcut rows 19--21, batch-size-8 rows 6--8, and bsz4 recovery rows
`3863546_[3-5]` were canceled. This released three A100s and nine A40s. Stale
eval arrays `3850116`, `3850122`, and `3850214` were canceled rather than
left with unsatisfiable dependencies.

All other pending BranchProof arrays were user-held to prevent released GPUs
from backfilling before the urgent project could start. This includes
`3850115`, `3854837`, `3857767`, `3850119`, `3850118`, `3863525`,
`3850123`, `3850121`, `3850120`, `3850113`, `3850109`, and
`3850110..3850112`, in addition to the three family arrays above. Hybrid eval
rows `3850118_0/1/2` raced into the released A100 slots and were canceled after
five minutes. Established rows already running in other BranchProof families
continue; only pending work is paused.

Baseline row 18 subsequently ended `NODE_FAIL` after sampled chunk `107/112`
without a final artifact. Held exact replacement `3865321_[18%1]` and CPU
audit `3865322_18` join the existing held row-13/14 recovery chain. Aggregate
`3857769` requires both replacement audits. Release `3865321` with the other
baseline eval work only after the July-20 capacity check; do not rerun rows
19/20, which completed and passed their audits.

Conditioned-50k rows `3850109_10/11/12`, which were already running when the
pause began, subsequently reached the 24-hour limit at steps
`26924/26751/18891`. Their complete latest restart states are
`checkpoint-25000`, `checkpoint-25000`, and `checkpoint-15000`, each with
nonempty optimizer, scheduler, RNG, trainer, and adapter files and zero empty
files. No final exists and no fatal signature preceded timeout. Do not submit a
new recovery: the already-held staged `afterany` chain `3850110..3850112` is
the intended exact continuation after the July-20 capacity decision.

Restart after 16:30 CEST on July 20, only if capacity is available and the
user-wide Vault is below both soft quotas with margin. At the July-19 audit it
was `1001G/1000G` soft and `200k/200k` files, while all repo-owned restart
states remained intact.

1. Confirm no replacement project still needs the GPUs, confirm Vault space
   and file-count margin, and re-audit final and checkpoint roots before
   submitting anything.
2. Release the held pending tasks in `3850105`, `3850114`, and `3850213`.
   Release the additional pending BranchProof arrays listed above only after
   checking the urgent project's remaining demand.
3. Resubmit surface rows 15--17 and shortcut rows 19--21 from scratch; neither
   family had an intermediate checkpoint.
4. Resubmit batch rows 3--5 with automatic resume from their complete
   checkpoint-3000 states. Restart rows 6--8 from scratch because their old
   launches predated resumable checkpointing and wrote no state.
5. For every batch row, save every 250 steps and retain two checkpoints. The
   10k-update protocol is unchanged; rows that exceed 24 hours are resumed in
   another allocation, with at most 249 optimizer steps replayed.
6. Recreate eval arrays for surface, batch, and shortcut with dependencies on
   all original and replacement training jobs. Existing complete output files
   remain skip-gated, so only missing rows evaluate.
7. Update all three live handoff documents with new job IDs and accept no
   result until structural audits and representative generation review pass.

## July 20 resumption

The quota gate was restored before release: final-backed checkpoint cleanup
and removal of superseded FineWeb data reduced user-wide Vault usage from
`1070G` to `861G`, while preserving all incomplete-run restart states. Every
held original array was released; no `JobHeldUser` task remains.

Canceled work was reconstructed as follows:

| Family | Replacement SFT/eval | Dependency policy |
| --- | --- | --- |
| Surface | SFT `3872657_[15-17]`; eval `3872661_[0-26]` | eval waits `afterany:3850105` and `afterok:3872657` |
| Shortcut | SFT `3872658_[19-21]`; eval `3872662_[0-41]` | eval waits `afterany:3850213` and `afterok:3872658`; prior row-3--6 replacement finals were reverified |
| Batch | SFT `3872659_[3-8]`; eval `3872663_[0-47]` | eval waits `afterany:3850114` and `afterok:3872659`; rows 3--5 resume checkpoint 3000 and 6--8 restart |
| Hybrid | eval `3872660_[0-2]` | training finals reverified; original eval rows 3--29 released |

All eval wrappers retain their complete-output skip gate, so rebuilding full
arrays cannot overwrite or recompute already accepted outputs.

The cleanup removed terminal Trainer checkpoints from completed
conditioned-50k rows. Before staged successor `3850110` could start, the SFT
wrapper's stale special case was removed: any verified nonempty final adapter
now skips. Completed conditioned rows 0--2 and 4--5 will not retrain;
unfinished rows still have no final and resume from the preserved checkpoint
states listed above.

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
