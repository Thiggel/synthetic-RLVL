# Running Experiments

Last updated: 2026-07-20 13:15 CEST.

This file is the live Slurm dashboard. Historical details live in `docs/operational_history_2026-05-29.md`; planned-but-not-running work lives in `docs/experiment_backlog.md`.

## Live Delta At 13:15 CEST On July 20

- Seventeen corrected BranchProof SFT rows are active on A40s. Replacement
  surface/shortcut `3872657_[15-17]` and `3872658_[19-21]`, resumed surface
  `3850105_[18-20]`, and shortcut `3850213_[22-24]` are around `32--34%`;
  architecture `3850113_48/49/50` are around `55/7/6%`; conditioned-50k
  `3850109_13/14` are around `6%`. Their logs show normal training with no
  fatal, OOM, quota, or no-space signature.
- A100-80 work remains account-GRES pending: baseline
  `3857767_[21-29]`, recoveries `3863525_[13-14]` and `3865321_18`, hybrid
  eval `3872660`, batch recovery `3872659`, and Dolmino p5 `3872664`.
  Dolmino's current projection is 17:34 CEST. Dependent CPU audits/evals and
  aggregate remain correctly gated; no new eval or Nanotron result artifact
  appeared after 08:52.
- Vault is `863G/1000G` user-wide and `300,344,897 KiB` repo-owned with `26`
  repo Trainer checkpoints. The shared file count is again marginally over
  soft quota at `201k/200k` (`400k` hard), so recheck before any additional
  storage-producing launch. Current watcher `3872507` is CPU-only; CPU-only
  successor `3873713` is BeginTime-pending for 19:07 CEST and remains needed.

## Live Delta At 08:52 CEST On July 20

- All BranchProof user holds are released. Sixteen A40 SFT rows started
  immediately across the resumed and replacement surface, shortcut,
  conditioned-50k, and architecture families. A100 work is normal
  account-GRES/dependency pending.
- New targeted jobs: surface SFT `3872657_[15-17]`, shortcut SFT
  `3872658_[19-21]`, batch SFT `3872659_[3-8]`, hybrid eval
  `3872660_[0-2]`, and rebuilt dependent evals surface `3872661`, shortcut
  `3872662`, batch `3872663`. Bsz4 rows resume checkpoint 3000; bsz8 restarts
  with 250-step checkpoints and two-state retention.
- Dolmino formal/NL p5 confirmation `3872664_[0-1%2]` is submitted at shared
  LR `1e-5`; Slurm estimates 10:24 CEST. Each row uses four A100-80GB GPUs,
  256 steps, global batch 128, and a 95/5 Dolmino/intervention blend.
- Guarded cleanup removed `130.04 GiB` of final-backed Trainer intermediates
  and `78.96 GiB` of superseded FineWeb data while preserving all 24 active
  restart checkpoints. Vault is now `861G/1000G`; its file count is no longer
  marked over quota. No fresh startup error is present.
- Corrected the conditioned-50k skip gate before staged successors start:
  verified final adapters now skip independently of deleted terminal Trainer
  checkpoints; unfinished rows continue from the 24 preserved restart states.

## Live Delta At 19:09 CEST On July 19

- The corrected baseline remains `18/30` accepted. Held A100-80 eval rows
  `3857767_[21-29]`, local-snapshot recoveries `3863525_[13-14]` and
  `3865321_18`, their audits, and aggregate `3857769` are unchanged. No new
  in-scope metric, sample, log, or eval artifact exists after 13:09, so no
  analysis or report gate opened.
- User-wide Vault usage is now `1070G/1000G` soft and `202k/200k` files. The
  repo-owned tree remains `517,225,409 KiB` with `8,711` files and `170`
  intact Trainer checkpoint directories. Release no held BranchProof or
  Dolmino work until both soft quotas have margin and the July-20 capacity
  pause has ended. No unrelated job or artifact was touched.
- CPU-only watcher `3870759` is running on `a100mig`. Recorded successor
  `3871736` requests only `cpu=4,mem=30000M`, has no GRES, and is
  BeginTime-pending for 01:07 CEST on July 20. It remains queued because the
  plan is incomplete.

## Live Delta At 13:09 CEST On July 19

- The corrected baseline remains `18/30` accepted. Held A100-80 eval rows
  `3857767_[21-29]`, local-snapshot recoveries `3863525_[13-14]` and
  `3865321_18`, their audits, and aggregate `3857769` are unchanged. No new
  in-scope metric, sample, log, or eval artifact exists after the 07:15
  handoff, so no analysis or report gate opened.
- User-wide Vault usage is now `1068G/1000G` soft and `201k/200k` files.
  The repo-owned tree remains `517,225,409 KiB` (about `493 GiB`) with `8,711`
  files and `170` intact Trainer checkpoint directories. Release no held
  BranchProof or Dolmino work until both soft quotas have margin and the
  July-20 capacity pause has ended. No unrelated job or artifact was touched.
- CPU-only watcher `3870293` is running on `a100mig`. Successor `3870759`
  requests only `cpu=4,mem=30000M`, has no GRES, and is BeginTime-pending for
  19:06 CEST. It remains queued because the plan is incomplete.

## Live Delta At 07:15 CEST On July 19

- The corrected baseline remains `18/30` accepted. Held A100-80 eval rows
  `3857767_[21-29]`, local-snapshot recoveries `3863525_[13-14]` and
  `3865321_18`, their audits, and aggregate `3857769` are unchanged. No new
  in-scope metric, sample, log, or eval artifact exists after the July-18
  19:15 handoff, so no analysis or report gate opened.
- User-wide Vault usage reached `1001G/1000G` soft and `200k/200k` files.
  The repo-owned tree remains `517,225,409 KiB` (about `493 GiB`) with `8,711`
  files. Its `170` Trainer checkpoint directories are intact, including the
  conditioned-50k and batch restart states. This is now a hard prerequisite
  check: release no held BranchProof work until both soft quotas have margin
  and the July-20 capacity pause has ended.
- CPU-only watcher `3869995` is running on `a100mig`. Successor `3870293`
  requests only `cpu=4,mem=30000M`, has no GRES, and is BeginTime-pending for
  13:06 CEST. It remains queued because the plan is incomplete.

## Live Delta At 19:15 CEST On July 18

- The corrected baseline remains `18/30` accepted. Held A100-80 eval rows
  `3857767_[21-29]`, local-snapshot recoveries `3863525_[13-14]` and
  `3865321_18`, their audits, and aggregate `3857769` are unchanged. No new
  in-scope BranchProof or Nanotron metric/sample/log artifact exists after the
  July-17 07:06 handoff. The urgent out-of-scope GPU wave remains active, so
  the July-20 capacity pause is still in force.
- User-wide Vault use is now `960G/1000G` soft with `198k/200k` files. The
  repo-owned tree is unchanged at about `493 GiB` and only `8,711` files; the
  file-count pressure is dominated by shared environment/cache and unrelated
  roots, which were inspected read-only and left untouched. Recheck this
  two-thousand-file margin before releasing the held BranchProof arrays.
- CPU-only watcher `3867919` is running on `a100mig`. Recorded successor
  `3869120` requests only `cpu=4,mem=30000M`, has no GRES, and is scheduled
  for 01:06 CEST on July 19. It remains queued because the plan is incomplete.

## Live Delta At 19:12 CEST

- The corrected baseline remains `18/30` accepted. Held A100-80 eval work
  `3857767_[21-29]`, `3863525_[13-14]`, and `3865321_18`, its CPU audits, and
  aggregate `3857769` are unchanged under the July-20 capacity pause. No new
  in-scope eval JSON, sample JSONL, or log appeared after the 07:06 handoff.
- Fixed the oversight Slurm wrapper's prompt quoting: Markdown backticks are
  now protected by a quoted heredoc instead of being executed as command
  substitutions. Bash syntax and rendered-prompt checks pass; shellcheck is
  unavailable in the current environment.
- CPU-only watcher `3865931` is running on `a100mig`; recorded successor
  `3866771` is CPU-only with no GRES and is BeginTime-pending for 01:03 CEST on
  July 18. Repo-owned Vault use is unchanged at about `493 GiB`. No experiment
  release, resubmission, dependency edit, partition edit, cleanup, analysis,
  or report trigger fired.

## Prior Live Delta At 07:06 CEST

- Conditioned-50k SFT rows `3850109_10/11/12` timed out at the expected
  24-hour limit after steps `26924/26751/18891`. Complete restart states exist
  at `checkpoint-25000`, `checkpoint-25000`, and `checkpoint-15000`; all have
  nonempty optimizer/scheduler/RNG/trainer/adapter files and zero empty files.
  Finals are absent, as expected. Preserve the held staged resume chain
  `3850110..3850112`; no duplicate recovery or dependency edit is needed.
- The corrected baseline remains `18/30` accepted. Original eval rows
  `3857767_[21-29]`, local-snapshot recoveries `3863525_[13-14]`, and
  node-failure recovery `3865321_18` remain user-held under the July-20 pause;
  their CPU audits and aggregate remain dependency-gated.
- CPU-only watcher `3865320` is running on `a100mig`; recorded CPU-only
  successor `3865444` is BeginTime-pending for 13:03 CEST. Repo-owned Vault
  use is about `493 GiB`; no cleanup, resubmission, report, or scheduler trigger
  is active before the capacity decision.

## Prior Live Delta At 01:08 CEST

- Baseline eval/audit rows 19/20 completed and passed, bringing the accepted
  declaration-fixed gate to `18/30`. Raw NL train-1-to-10 review at depths
  `1/10/12/25/50` found clean train-band extraction/translation and ordinary
  depth-50 truncation/answer failures; no family claim is released.
- Baseline eval row `3857767_18` ended `NODE_FAIL` on `a0632` after sampled
  chunk `107/112` and produced no final artifact. Exact local-snapshot
  recovery `3865321_[18%1]` is submitted A100-80-only and user-held under the
  July-20 pause; CPU audit `3865322_18` follows it. Original impossible audit
  task `3857768_18` was canceled, and aggregate `3857769` now requires both
  replacement audit families `3863527` and `3865322` after the original audit
  array terminates.
- Removed only the orphaned `28G` merge root from failed internal job
  `3863485`; current repo-owned Vault use is about `494G`. Conditioned-50k
  rows 10/11/12 remain running, while all pending BranchProof work and
  recoveries 13/14/18 remain held. CPU-only successor `3865320` is scheduled
  for 07:03 CEST.

## Prior Live Delta At 19:35 CEST

- Baseline NL eval rows 15--17 completed, while original audits
  `3857768_15/16/17` failed only because the audit incorrectly demanded a
  positive formal-syntax score from natural-language outputs. The fixed audit
  uses `nl_logic_parse` for NL train-band structure; replacement CPU array
  `3864893_[15-17%3]` completed `3/3` and accepted all rows. Baseline is now
  `16/30` accepted. Aggregate `3857769` waits for the original audit array to
  terminate and for both replacement audit arrays `3863527` and `3864893` to
  pass. Raw review across three NL seeds and depths `1/5/10/18/25/50` found
  clean train-band extraction/translation and ordinary wrong, malformed, and
  capped long-depth failures; no family claim is released.
- Dolmino LR array `3859711_[0-2]` is complete. The `1e-5` row finished in
  `02:27:54` at 15.3K tokens/s with 256 finite steps and 134,217,728 scheduled
  tokens. Matched post-warmup mean loss favors `1e-5` (`0.95675`) over `6e-6`
  (`0.96040`) and `3e-6` (`0.97130`), so `1e-5` is nominated for the formal/NL
  p5 confirmation. Do not submit those GPU jobs during the four-day capacity
  pause; reassess after 16:30 CEST on July 20. The terminal manifests point to
  benchmark CSVs that were not written; retain the complete stepwise logs.
- Baseline rows 18/19/20 are healthy at sampled chunks `102/101/78` of 112.
  All existing holds remain intentional. Watcher successor `3864892` remains
  BeginTime pending and must not be canceled. Repo-owned Vault use is about
  `627G`; no cleanup trigger is active.
- Original audit tasks `3857768_13/14` were canceled from permanent
  `DependencyNeverSatisfied` state. Exact held replacements
  `3863525_[13-14] -> 3863527_[13-14]` preserve their required eval/audit
  gates; remaining original audit rows 18--29 still follow their evals.

## Prior Live Delta At 16:37 CEST

- Four-day capacity pause: held pending surface `3850105`, batch `3850114`,
  and shortcut `3850213` rows. Canceled active surface 15--17, batch 6--8,
  shortcut 19--21, and batch recovery `3863546_[3-5]`, releasing `3` A100s
  and `9` A40s. Canceled stale dependent eval arrays
  `3850116/3850122/3850214`; replacements must depend on the resumed training
  jobs. Planned restart is after 16:30 CEST on July 20.
- Batch rows now checkpoint every 250 steps and retain two checkpoints.
  Bsz4 rows 3--5 resume from checkpoint 3000; bsz8 rows 6--8 restart because
  their old launches wrote no checkpoint. Surface/shortcut canceled rows also
  restart from scratch.
- Dolmino `1e-5` is at step 234/256 and remains healthy; expected completion
  is around 16:50 CEST, releasing four additional A100s.
- Baseline logic train-25 seed 3407 passed its strict audit, bringing accepted
  baseline rows to `13/30`.
- All other pending BranchProof arrays are also user-held to prevent
  backfilling. Hybrid eval rows `3850118_0/1/2` briefly backfilled the released
  A100 slots and were canceled after five minutes; non-BranchProof
  `3863071_0/1/2` then started in those slots. Already-running BranchProof
  rows continue to completion.

## Prior Live Delta At 14:25 CEST

- Dolmino LR rows `3859711_0/1` (`6e-6/3e-6`) completed cleanly in about
  2h28m each. Across 224 identical post-warmup batches, `6e-6` has lower loss
  on 210 with mean paired advantage `0.0109`; batch-loss spikes are aligned
  across runs. Do not select before `1e-5` row 2, which started on four A100s
  at 14:21 CEST. Redundant 8-GPU fallback `3859297` was then canceled.
- Baseline eval/audit rows `0..11` are accepted (`12/30`), completing all
  logic seeds for train maxima `5/10/15/20`. Logic train-25 row 12 runs;
  transient-Hub rows 13/14 await local-snapshot recovery, and NL rows 15..19
  are running. Batch checkpoint resumes `3863546_[3-5]` were healthy before
  the capacity pause.

## Prior Live Delta At 13:13 CEST

- Baseline eval rows 13/14 failed before evaluation on transient Hub
  `504`/timeouts while merging the unchanged OLMo-3 base. Exact local-snapshot
  recovery `3863525_[13-14%2]` is A100-80-only; replacement CPU audits
  `3863527_[13-14%2]` follow it. Aggregate `3857769` waits original audit
  termination plus successful replacement audits. The qualitative gate now
  uses each accepted row audit's exact log provenance and requires complete
  sampled-chunk logs (`7 passed`).
- Baseline eval/audit rows 0..10 are accepted (`11/30`). Rows 11/12 and
  15..18 run on A100-80GB; row 11 has generated all 112/112 sampled chunks
  and is finalizing, row 12 is at 105/112, and NL rows 15/16/17 are at
  68/53/46. Representative logic raw review across seeds/train ranges and
  depths `1/15/20/25/50` confirms clean train-band extraction/validity and
  correct rejection of parse, unsupported-line, answer, format, and long-cap
  failures. No matched modality claim is released.
- Batch jobs `3859299_[3-5]` timed out at 24 hours with complete
  `checkpoint-3000` state. Exact resumes `3863546_[3-5%3]` are running on
  A40s, and eval `3850122` now requires successful `3863546` after both old
  arrays terminate.
- Dolmino `6e-6` row `3859711_0` completed 256/256 steps in `02:27:56` at
  about 15.4K tokens/s with finite diagnostics and `complete.json`.
  `3859711_1` (`3e-6`) is running at matched throughput; row 2 and sequential
  fallback `3859297` remain pending. Preserve the fallback until row 2 starts.
  Successor watcher `3863505` remains begin-time pending because the plan is
  incomplete. Vault usage is 827G and 187k files.

## Prior Live Delta At 09:32 CEST

- Four-GPU Dolmino LR row `3859711_0` (`6e-6`) started at 09:25 CEST on
  A100-80GB node `a0832`. At step 8/256 it sustains about 15.4K tokens/s with
  finite loss/gradients and 62.4GB peak allocation; ETA is about 11:53 CEST.
  Rows 1/2 remain account-GRES pending with 10:48 estimates. Sequential
  8-GPU fallback `3859297` now estimates 2026-07-17 02:30; preserve it until
  rows 1/2 start, then cancel redundant unstarted work.
- Baseline eval/audit row 7 completed and is accepted, bringing the corrected
  row-scoped gate to `8/30`. Representative samples at depths
  `1/15/18/25/50` match intended behavior; long-depth duplicate declarations
  and malformed traces are rejected. No modality claim is released.

## Prior Live Delta At 07:20 CEST

- Baseline eval/audit rows `3857767_0..6 -> 3857768_0..6` are complete and
  accepted at row scope. The seven audits each pass the exact prompt,
  generation, retained-row, metric, chunk-log, cap, fresh-constant,
  declaration, answer, and validity gates. This completes three logic seeds
  for train maxima 5 and 10, plus one train-15 seed, but still does not provide
  a matched modality result.
- New row-6 raw review covers sampled depths `1/15/18/20/25/50`, successes,
  failures, and cap hits plus a greedy duplicate-declaration failure. Intended
  shallow proofs and extraction are clean; unsupported lines, wrong answers,
  repetition, and duplicate declarations are rejected by the corrected
  validity path. Active baseline rows `7..12` have completed sampled chunks
  `102/88/89/88/71/52` of 112 on verified A100-80GB nodes. Rows `13..29` are
  throttle-pending; no fatal signature or runtime intervention trigger exists.
- Batch recovery `3859299_[3-5]` is at approximately
  `2,512/2,509/2,527` of 10,000 after 18 hours. Its periodic checkpoints are
  present; wait for an actual timeout before submitting a targeted resume.
  Conditioned-50k originals `3850109_10/11/12` are at approximately
  `6,293/6,233/972` of 50,000 and remain covered by their staged resume chain.
- Dolmino LR gates `3859297` and `3859711_[0-2]` remain account-GRES pending;
  current scheduler projections are about 21:15 and 08:45 CEST. CPU-only
  watcher `3862186` is running and successor `3862431` is begin-time pending
  for about 13:01 CEST. Vault use is about 717 GiB, with 151 protected Trainer
  checkpoints and nine active merge roots.

## Prior Live Delta At 01:05 CEST

- Declaration-fixed baseline rows `3857767_3/4` and CPU audits
  `3857768_3/4` completed `0:0`. Both audits accept 448 prompts, 16
  generations, 1,024 retained rows, 2,665 metrics, complete `7/112` chunk
  logs, fresh constants, strict answers, and the strengthened declaration and
  validity invariants. Raw depths `1/5/10/12/25/50` show clean train-band
  proofs and ordinary wrong-branch, malformed, repetitive, and cap-hit deep
  failures. They are a partial logic-only slice and do not release a claim.
- Active baseline rows `1/2/5/6/7` are at sampled chunks
  `109/101/109/92/38`; row 8 is in startup on A100-80GB. Row 1 is nearly
  complete after 18 hours, so no depth sharding or protocol change is
  justified. No monitored baseline log has a fatal signature.
- Conditioned-50k rows `7/8` are near the 24-hour limit at steps
  `42,816/43,078`; row 9 is at `21,837`. Preserve staged after-any resume jobs
  `3850110..3850112`. Batch recovery `3859299_[3-5]` is at approximately
  `1,658/1,656/1,667` after 12 hours with new 1,000-step checkpoints present;
  submit no duplicate recovery before a terminal timeout.
- LR gates `3859297` and `3859711_[0-2]` remain account-GRES pending. Current
  Slurm estimates favor the 4-GPU rows at about 03:17 CEST; the sequential
  8-GPU fallback estimates 18:03 CEST. CPU-only successor `3862186` is
  begin-time pending for 07:01 CEST and remains required. Vault use is about
  `706 GiB`, with 141 protected Trainer checkpoints and nine active merge
  roots.

## Prior Live Delta At 19:10 CEST

- Declaration-fixed baseline row `3857767_0` completed in `11:07:55`; CPU
  audit `3857768_0` passed the full shape, log, cap, constant, answer, duplicate
  declaration, and validity gates. Raw depths `1/5/25/50` are sample-clean for
  the intended evaluator behavior: shallow proofs are correct and
  citation-free valid, while malformed/duplicate long traces receive zero
  validity. Rows `1..5` are at sampled chunks `84/67/92/85/75`; row 6 is in
  greedy generation. Row 1 is the runtime watch at 12.1 hours, still below the
  20-hour depth-sharding trigger.
- Conditioned-10k row 0 completed both modality outputs and bounded raw review;
  row 3 also completed its NL bundle, and rows 2/4 continue. This remains
  partial family evidence.
  No active/recent monitored log has a new fatal, OOM, quota, or dependency
  signature.
- Batch recovery `3859299_[3-5]` is only at steps `815/815/821` after about
  six hours on A40. These rows restarted because the timed-out parents had no
  checkpoints; the new 1,000-step checkpoint gate has not yet fired. Preserve
  them and recover again only after an actual timeout. Vault use is about
  `691 GiB`, with 128 protected Trainer checkpoints and nine active merge
  roots.
- LR gates `3859297` and `3859711_[0-2]` remain account-GRES pending. Current
  CPU-only watcher `3859290` scheduled CPU-only successor `3860702`; preserve
  it while the plan is incomplete.

## Prior Live Delta At 15:03 CEST

- Dolmino row `3858584_0` stopped at `4.128B/4.8B` tokens on a transient Hub
  Xet/CAS HTTP 500. Exact recovery `3859296_0` resumed without replay and
  finished the raw export at `4,800,000,272` tokens in `10,705,908` records
  across 120 sources. Manifest sums and five byte-spaced raw records pass.
  Nanoset preprocessing completed at `4,810,706,180` packed tokens in one
  nonempty 19.24GB shard; the delta confirms one EOS per source record.
  Replacement sequential LR gate `3859297` supersedes canceled dependency-dead
  `3858902` and is `AssocGrpGRES`-pending with a 2026-07-16 10:48 CEST Slurm
  estimate. Independent matched 4-GPU rows `3859711_[0-2%3]` are also
  account-GRES pending, each with a 2026-07-16 03:49 CEST estimate. They use
  TP4/DP1, microbatch 4, and
  accumulation 32, preserving the 8-GPU gate's global batch and token budget.
- Batch rows `3850114_3/4/5` timed out near `94--95%`. Exact recovery
  `3859299_[3-5%3]` is running with 1,000-step checkpoint retention, and eval
  `3850122` now requires the original array plus this recovery.
- Declaration-fixed baseline `3857767_0/1` is at sampled chunks `82/55`; rows
  `2/3/4` are in greedy generation on A100-80GB. Conditioned-10k eval row 1
  completed and passed bounded raw review; it remains partial evidence.
- Watcher successor is now CPU-only `3859290`; keep it queued.

- Dolmino build `3858584_[0-2]` started immediately on RTX Pro 6000 nodes.
  It builds 4.8B Qwen tokens of released Dolmino and matched 550M-token neutral
  formal/NL sources. The released default HF config was rejected after a local
  smoke exposed incompatible heterogeneous Arrow schemas; the production row
  instead reads shuffled JSONL.zst shards directly and records source-token
  counts for mixture auditing.
- Formal/NL build rows completed in about 12 minutes. Dolmino row 0 is healthy
  at approximately 2.13B/4.8B tokens after 56 minutes. Separately queued LR
  jobs `3858587/3858588` were canceled before start and consolidated into
  dependency-held `3858902`: one 12-hour 8xA100-80GB allocation runs `6e-6`,
  `3e-6`, and `1e-5` sequentially, each for 256 steps/134.2M tokens with no
  checkpoint. Full control/formal/NL midtraining remains deliberately
  unsubmitted until one shared LR is selected and briefly confirmed on both
  intervention modalities.

- Nanotron postmortem found randomized rather than strict batch stratification:
  seed-42 global updates have 6--35 proof chunks out of 128, mean `19.2001`
  and near-binomial standard deviation `4.0848`, with no proof-empty update.
  Nanoset randomizes packed sample indices despite each source having one
  nonempty pretokenized shard. It did find a matched scheduler-resume rescaling
  bug and an overlong cosine span. Future code is patched; completed
  control/logic/NL runs share the same defect. Full-document objective and
  response-surface learning are the stronger scientific concerns.

- Declaration-fixed baseline rows `3857767_0/1` started on verified A100-80GB
  devices at `06:59/07:01` CEST. Row 0 completed greedy chunks 1--3 and row 1
  completed chunks 1--2; deeper chunks show expected 7,168-token cap hits but
  no fatal signature. Their 30 CPU audits and aggregate remain dependency-gated;
  all `3853284` evidence stays quarantined.
- Prompt-fixed multi-hop direct `3855271`, instruction `3855272`, and aggregate
  `3855273` completed and passed full artifact gates. Raw review shows that the
  direct stock F1 advantage largely collapses under answer-head rescoring and
  that tagged prompts trigger the learned formal/NL substrate into the cap.
  Treat this as response-control evidence, not positive transfer.
- Conditioned-10k eval rows `3850119_0/1/2` are healthy at chunks
  `64/87/39` of 112. Conditioned-50k row `3850109_6` timed out at step 43,105
  and is covered by staged resumes `3850110..3850112`. Batch rows
  `3850114_3/4/5` are walltime-risk at about 72%; pending batch rows now save
  every 1,000 steps so any genuine timeout can resume. Shortcut recoveries
  `3854948_3/4` and `3856142_5/6` completed cleanly.
- Repo-owned Vault use is `439.6 GiB` with 110 protected Trainer checkpoints.
  CPU-only watcher `3857722` is running and successor `3858016` remains queued.

## Critical Correction and Active Chains

All pre-2026-07-10 BranchProof results above depth 17 are quarantined after a
closure audit found multiple derivable answers caused by wrapped constants.
This includes old architecture, syntax, shortcut, hybrid, conditioned-dual,
batch-size, and proof-mixture results. See
`docs/branchproof_uniqueness_audit_2026-07-10.md`.

| Experiment | Jobs | Live state at 2026-07-15 15:03 CEST | Outputs / next gate |
| --- | --- | --- | --- |
| Corrected report-wide BranchProof reruns | surface `3850105 -> 3850116`; hybrid `3850107 -> 3850118`; conditioned 10k `3850108 -> 3850119`; conditioned 50k `3850109..3850112 -> 3850120`; architecture `3850113 -> 3850121`; batch `3850114 -> 3850122`; 32B `3850115 + 3854837 -> 3850123`; shortcuts `3850212 -> 3850213 + 3854948 + 3856142 -> 3850214`; no-repeat tiny `3850488 -> 3850490 -> 3850492` plus checkpoint replacement/recovery `3854813 + 3856145` | Tiny final `18/18` and checkpoint curve `90/90` are terminal and accepted. Shortcut recoveries `3854948_[3-4]` and `3856142_[5-6]` are complete; original shortcut rows 10/11/12 continue. Conditioned-10k eval rows 0/1/2 are running cleanly. Conditioned-50k row 6 timed out at 43,105/50,000 and remains covered by the staged resume chain. Batch rows 3/4/5 may hit the hard 24-hour ceiling; pending batch rows now save every 1,000 steps, and only actually failed rows should be recovered. The 32B originals/recovery and other families continue under their current throttles/dependencies. | Tiny audit bundles: `$HPCVAULT/synthetic-RLVL/analysis/branchproof_unique_v2_tiny_100k_final_audits_20260714` and `$HPCVAULT/synthetic-RLVL/analysis/branchproof_unique_v2_tiny_100k_checkpoint_audits_20260714`. Recover only failed rows. Accept each family only after strict audit and representative raw review; never blend with old report numbers. |
| Dolmino midtraining prerequisite and shared-LR gate | original build `3858584_[0-2]`; row-0 recovery `3859296`; independent 4-GPU rows `3859711_[0-2%3]`; canceled fallbacks `3858902/3859297` and obsolete jobs `3858587/3858588` | All data prerequisites pass. Four-GPU `6e-6/3e-6` rows completed; paired loss slightly favors `6e-6`. The `1e-5` row started at 14:21 CEST. The redundant 8-GPU fallback was canceled. | Select one shared LR only after all three rows and run formal/NL p5 confirmations before staged full training. |
| Nanotron HF RoPE compatibility and multi-hop QA recovery | old diagnostic smokes `3850353/3850354`; prompt-fixed smokes `3855269/3855270`; CPU re-audit gate `3856131`; full direct/instruction arrays `3855271/3855272`; aggregate `3855273`; canceled flawed full jobs `3850099/3850100` and aggregates `3850207/3850217` | Complete. All six production bundles contain 1,200 rows, use corrected RoPE and a 32,768 window, and passed prompt/cap/coverage audits; aggregate `3855273` completed `0:0`. Direct stock control/logic/NL QA-F1 is `0.189/0.250/0.238`, but answer-head sensitivity is `0.349/0.361/0.367`. Direct tagged logic/NL usually launch their learned trace substrate and hit the 64-token cap; instruction stock rows are also cap-limited with QA-F1 near `0.09--0.10`. | Accepted artifacts: `analysis/nanotron_branchproof_unique_v2_multihop_promptfix_20260714/`. Treat the result as a response-format/continuation diagnostic, not clean reasoning transfer. Old RoPE/truncation/prompt-duplication bundles remain diagnostic only. |
| Corrected BranchProof-v2 materialization | build/push `3829067` | Completed cleanly in `00:06:54`. Production gate passed on 3,000 examples with unique-solution rate `1.0`, max one derived answer, no failures, and balanced gold positions; HF smoke loads also passed. | `$HPCVAULT/synthetic-RLVL/datasets/branchproof_unique_v2_20260710`; private HF repo `flaitenberger/BranchProof-unique-v2`. |
| Corrected BranchProof-v2 SFT/eval | full SFT `3829072_[0-29%12]`; declaration-fixed replacement eval `3857767_[0-29%6]`; CPU audits `3857768_[0-29%8]`; aggregate/qualitative gate `3857769`; newly quarantined `3853284/3853285/3853286`; older quarantined chains `3834582/3834706/3835779/3838163/3847756/3847757/3838164/3838165` | SFT is complete at `30/30`. The old chain and all its metrics remain quarantined. Replacement row 0 and audit row 0 are complete/accepted; rows 1..6 are active on A100-80GB, with row 1 the runtime watch below 20 hours. Later rows are throttle pending. | Preserve 32 prompts/depth, 16 generations, all 14 depths, pass@`1/2/4/8/16`, 16,384 context, 7,168 cap, and A100-80-only placement. Accept no family metric until all 30 replacement audits and representative logic/NL review pass. |
| Corrected BranchProof-v2 Nanotron corpora and matched p15 pilot | completed builds `3829068_1` and raw row `3829071`; packed audit `3830855`; logic/NL smokes `3830924/3831110`; native-chat smoke `3831179`; logic train/recovery/upload `3830927 -> 3835442 -> 3847802`; NL train/recovery/upload `3831111 -> 3835443 -> 3847570`; downstream branches below | All three p15 checkpoints, repaired HF consumers, six reviewer bundles, and strict aggregate are complete. Replacement logic instruction eval `3854824_3` and aggregate `3854847` completed `0:0`; the manifest accepts all six bundles. Direct logic is `+0.0033` all-primary and `+0.0071` reasoning versus control but `-0.0116` on targeted logic; NL and post-instruction deltas are similarly small/mixed. Raw review finds more direct next-document continuation for both proof mixtures and severe instruction repetition/BBH extraction floor. | The broader mixture grid is rejected because p15 is neither positive nor sample-clean. Preserve the accepted artifact bundle `analysis/nanotron_branchproof_unique_v2_p15_20260711/`; treat instruction BBH and instruction-minus-direct macro collapse as evaluator/generation diagnostics, not modality evidence. |
| Qwen2.5 normal-continuation control | completed recovery/upload `3835438 -> 3847569`; corrected adapter `3850351`; corrected direct/instruction eval `3850385/3850387`; invalid old bundles `3847792/3835928` | Corrected direct and instruction bundles completed, pass consumer RoPE and production artifact audits, and were schema-v4 MATH-rescored with no lost stock-positive rows. Direct BBH/MMLU-Pro invalid extraction is `4.38/4.56%` and next-document marker incidence `35.60/12.07%`; instruction rates are `3.65/1.22%` and `0/0%`. | Retain through `3854847`; interpret only in the matched six-condition aggregate and raw comparison. |
| Qwen2.5 matched NL p15 | completed recovery/upload `3835443 -> 3847570`; corrected adapter `3850352`; corrected direct/instruction eval `3850386/3850388`; invalid old bundles `3834904/3834905` | Corrected direct and instruction bundles completed and pass the same gates. Direct BBH/MMLU-Pro invalid extraction is `4.39/4.74%` and marker incidence `58.01/49.51%`; instruction rates are `3.70/1.26%` and `0/0%`. | Retain through `3854847`; do not reuse the older RoPE-invalid `9.1/20.5%` diagnostics as corrected evidence. |
| Downstream reviewer-suite production and comparison | corrected control/NL `3850351/3850352/3850385..3850388`; logic upload/direct/instruction SFT `3847802/3847804/3847805`; replacement logic instruction eval `3854824`; strict aggregate `3854847`; canceled stale dependent/aggregate `3847806/3850389` | Complete at six accepted corrected bundles. Schema-v4 MATH remains the production score. Aggregate macros are null/mixed, with one run per condition and no seed variance. Correct/incorrect raw review shows direct proof/new-document continuations; instruction removes literal markers but introduces long repetition. Correct leading BBH choices followed by suffixes score zero under `get-answer`, so the instruction BBH cells are an extraction floor. | Do not launch broader mixture training. If this pilot later becomes report evidence, add an independently audited BBH answer-prefix sidecar or exclude those cells; never interpret the current zero-BBH instruction macro drop as transfer. |
| Six-hour autonomous oversight | current watcher `3859290`; recorded successor `3860702`; self-scheduled successors via `scripts/slurm/codex/branchproof_nanotron_oversight_2026-07-11.slurm` | CPU-only watcher `3859290` is running without GRES. Successor `3860702` is begin-time pending. | Preserve `3860702` until every baseline, report-wide, Dolmino, audit, and report gate is complete. |

All old proof-mixture training rows and dependents were canceled, and their
stale/incomplete proof checkpoints were removed. Do not resubmit them against
the old Nanosets. The unrelated `grid_goal_struct_eval` jobs visible in the
user-wide queue belong to `/home/hpc/c107fa/c107fa12/sequence-editing` and are
not managed here.

Vault cleanup at 11:19 CEST reclaimed `93.9 GiB` without touching active job
inputs. The accepted tiny checkpoint curve released a second guarded cleanup
at 19:26: all 90 tiny intermediate checkpoints (`102G`) were removed while 18
finals, 90 metrics, 90 samples, and 90 audits were retained. The six active
`3853284` merge roots were later removed after that chain was canceled and
quarantined. The converted Qwen base, corrected Nanosets/raw corpora, current
`3857767` merge roots, and corrected report-wide checkpoints remain protected.
Delete other intermediate checkpoints only after their corresponding
final/eval/audit gate is accepted; verify replacement merge roots self-delete
as each `3857767` row becomes terminal.

## Tracked Slurm Chains

| Experiment | Jobs | State | Expected outputs | Notes |
| --- | --- | --- | --- | --- |
| Full paired-family suite | original SFT `3672212_[0-89%6]`, replacement SFT `3682411_[55,57,59-89%6]`, row-56 replacement SFT `3683070_[56%1]`, replacement eval `3682449_[0-89%4]`, completed recovery `3691024_[30-32%3]`, failed recoveries `3694618_[36-38%3]` and `3694619_[40-89%4]`, completed official_iGSM NL validity rerun `3689003_[3-5,9-11,15-17,21-23,27-29%4]` | SFT/build are complete at `90/90` final adapters. Eval outputs are `37/90`: `official_igsm` `30/30`, `maze_navigation` `7/30`, `attribute_constraints_hard` `0/30`. Old recovery `3694619_[40-89%4]` failed quickly because merge output still targeted `/home/atuin/.../tmp` and hit `$WORK` quota (`OSError: [Errno 122] Disk quota exceeded`). | `$WORK/synthetic-RLVL/passk_eval/paired_full_suite_sparse_20260528/` | This old full-suite recovery is stale for iGSM semantic grounding and maze typed-symbol questions. Do not resubmit unchanged. Current conclusions should use fresh semantic iGSM, typed maze, and hard-attribute reruns. |
| Hard attribute/constraint full eval | fresh eval `3716216_[0-29%3]`; completed targeted row-1 recovery `3739163_[1%1]`; partial recovery `3743048_[21-29%3]`; completed final missing-row recovery `3748682_[27-29%3]`; forced NL-validity re-eval `3758371_[3-5,9-11,15-17,21-23,27-29%3]` via `scripts/slurm/jobs/posthoc_paired_attribute_constraints_hard_full_eval_2026-06-10.slurm` | Complete. Original logic/NL full eval is `30/30`; forced NL-validity re-eval `3758371` completed all `15/15` NL rows by 2026-06-20. | `$HPCVAULT/synthetic-RLVL/passk_eval/paired_attribute_constraints_hard_full_20260610/` | Logic held-out hard-tail joint@16 improves with train max (`0.108/0.455/0.500/0.737/0.736` for train max `5/10/15/20/25`), but depth-50 joint remains `0.000`. The patched hard-attribute NL translator now works: report-ingested mean NL hard-tail `correct/parse/joint@16` by train max `5/10/15/20/25` is `0.420/0.833/0.006`, `0.791/0.800/0.660`, `0.762/0.766/0.661`, `0.862/0.824/0.769`, `0.808/0.815/0.795`; depth-50 translated joint remains weak (`0.000/0.031/0.000/0.010/0.073`). Report refreshed from these corrected NL rows on 2026-06-22. |
| Trace-control ablations | SFT `3661118_[0-17%3]`, original eval `3661119_[0-17%3]`, original repair `3680004_[3-8%3]`, replacement evals `3682459_[12,14-17%3]` and `3682460_[5-8%3]` | SFT complete. Original eval rows `0..5`, `9..11`, and `13` complete; rows `12/14/15..17` failed/killed. Original repair rows `3..4` complete and `5..8` failed/killed. Replacement `3682459` rows `12/14/15/16/17` and replacement repair `3682460` rows `5/6/7/8` are complete. | `passk_eval/hfsa_ablation_trace_controls_20260525/` | `18/18` eval JSONs plus sample JSONLs are present. Important metric correction at 13:28 CEST: `invalid_logic` can score as citation-free valid because citation-free reconstruction ignores deliberately broken citations, so the report builder now uses strict grounded joint for formal trace-control rows. Gold-target tests confirm `terse_nl`, `rule_annotated_nl`, and `pseudocode` translate to valid logic; `shuffled_nl` parses but fails translated validity as intended; symbol-padded/wordified logic validate under formal proof scoring. The report was regenerated/mirrored after the fix; refreshed tables show `invalid_logic` and `shuffled_logic` formal joint validity as `0.000`. |
| Shortcut-rate `0.3` | SFT `3671431_[0-5%3]`, eval `3671432_[0-5%3]` | complete; all 18 shortcut-rate JSONs exist across `0.3/0.5/0.8` and `logic/nl_exact` | `$WORK/synthetic-RLVL/passk_eval/hfsa_shortcut_rate_ablation_20260525/` | The `0.3` row is now fully matched: logic OOD correct/joint@16 `0.892/0.598`, depth-50 correct/joint@16 `0.844/0.375`; NL OOD correct/translated-joint@16 `0.588/0.571`, depth-50 correct/translated-joint@16 `0.458/0.438`. Across rates `0.3/0.5/0.8`, NL depth-50 joint falls `0.438 -> 0.312 -> 0.146`, while logic depth-50 joint is `0.375 -> 0.375 -> 0.417`. |
| Hybrid order | targeted SFT `3670782`, original eval `3670783_[0-29%4]`, replacement eval `3682461_[13,15-29%4]` | SFT complete; eval complete at `30/30` JSONs. | `$WORK/synthetic-RLVL/passk_eval/hfsa_hybrid_order_full_20260525/` | Completed `think_formal` is three-seed complete through train-1-to-25, and `formal_think` is now also three-seed complete through train-1-to-25. The 2026-06-10 20:12 report refresh includes the full-grid tables/figures and qualitative sample interpretation; current conclusion is that hybrids preserve intended surfaces but do not close the long-depth validity gap. |
| Wordified length-control logic | SFT `3674875_[0-2%3]`, eval `3674876_[0-2%3]` | complete; 3 eval JSONs written | `$WORK/synthetic-RLVL/passk_eval/hfsa_logic_wordified_20260529/` | Cleaner equal-length control: predicates become word names such as `Teal(a)`, constants stay compact. Duplicate `3674877/3674878` was canceled. Mean OOD correct/joint@16 `0.508/0.323`; depth-50 correct/joint@16 `0.344/0.094`. |
| Conditioned dual 50k | SFT chunks `3674879 -> 3674880 -> 3674881/3682457/3682492 -> 3674882 -> 3674883`, original final eval `3674884`, checkpoint eval `3674885`, targeted final recovery `3716219_[17,18,20-29%3]` | Complete. 10k, 20k, repaired 30k, 40k, and 50k chunks are done; checkpoint eval is `30/30`, and final eval is now `30/30` JSONs/sample JSONLs after recovery `3716219` completed cleanly. | `$WORK/synthetic-RLVL/passk_eval/hfsa_conditioned_dual_50k_20260529/`, `$WORK/synthetic-RLVL/passk_eval/hfsa_conditioned_dual_50k_intermediate_20260529/` | Final train-1-to-25 means: `conditioned_logic` OOD/depth-50 correct@16 `0.833/0.677`, joint@16 `0.348/0.146`; `conditioned_nl` OOD/depth-50 correct@16 `0.675/0.531`, translated-joint@16 `0.531/0.250`. Sample checks preserve mode-conditioned prompts and normal answer extraction. The result is not a prompt mismatch; the batch-size ablation remains the direct test for modality-mixing/effective-batch effects. |
| OLMo-3-32B and Qwen3-32B normal proof-chain baselines | original SFT `3758372_[0-11%1]`; completed row `3758372_5` as raw job `3770814`; recovery SFT `3771012_[6-11%1]`; completed targeted row-8 recovery `3775860_[8%1]`; completed replacement eval `3775864_[0-11%1]` | Complete: SFT finals and eval outputs are `12/12`. Eval rows completed by 2026-06-28 and wrote 12 pass@k JSONs plus 12 sample JSONLs. | SFT finals under `$HPCVAULT/synthetic-RLVL/runs/sft_hfsa_modelablate_{olmo3_1125_32b,qwen3_32b}_{logic,nl_exact}_train1to25_10k_seed{3407,3408,3409}/final`; eval JSONs under `$HPCVAULT/synthetic-RLVL/passk_eval/hfsa_model_ablation_32b_train25_20260619/`; summary CSV `analysis/logic_cot_report_2026-05-25/tables/hfsa_model_ablation_32b_train25_summary.csv` | Initial readout: OLMo-3-32B logic has high OOD/depth-50 correct@16 `0.954/0.917` but citation-free joint only `0.477/0.115` and strict grounded joint `0.000/0.000`; OLMo-3-32B NL translated joint is `0.546/0.208`. Qwen3-32B NL is much stronger than Qwen3-32B logic (`0.838/0.792` translated joint vs `0.065/0.010` citation-free logic joint). Samples show OLMo logic citation fragility and Qwen3 logic lowercase-predicate syntax drift. |
| OLMo-3-32B conditioned-dual capacity follow-up | Canceled stale OLMo-2 SFT/eval `3756255`/`3756256`; failed SFT `3758374_[0-2%1]`; canceled dependency-stuck eval `3758375_[0-5%1]`; completed row `3768259_0` as raw job `3768361`; failed/timed-out recovery SFT `3771013_[1-2%1]`; completed/timed-out targeted recovery `3775861_[1-2%1]`; completed final row-2 recovery `3795088`; completed replacement eval `3795089_[0-5%1]` | Complete: SFT finals and eval outputs are `6/6`. Minimal resume `3795088` produced the seed-3409 `final/` adapter; replacement eval rows completed cleanly by 2026-06-30. After eval completion, seed-3409 intermediate checkpoints and the now-inactive OLMo-3-1125-32B HF cache/offload were removed; finals and eval outputs remain. | SFT finals under `$HPCVAULT/synthetic-RLVL/runs/sft_hfsa_modelablate_olmo3_1125_32b_conditioned_dual_train1to25_10k_seed{3407,3408,3409}/final`; eval JSONs under `$HPCVAULT/synthetic-RLVL/passk_eval/hfsa_conditioned_dual_olmo3_1125_32b_20260619/` | 3-seed summary: conditioned logic OOD/hard-tail correct@16 `0.963/0.979`, citation-free joint@16 `0.487/0.715`, strict/grounded joint `0.000`; conditioned NL OOD/hard-tail correct@16 `0.608/0.782`, translated joint@16 `0.537/0.743`, parse@16 `0.938/0.965`. Corrected apples-to-apples comparison with single-modality OLMo-3-32B on the same OOD/hard-tail bands: conditioned logic is slightly higher in correctness (`0.963/0.979` vs `0.954/0.975`) and citation-free joint (`0.487/0.715` vs `0.477/0.709`); conditioned NL remains lower/near-tied (`0.608/0.782` correct and `0.537/0.743` translated joint vs `0.685/0.825` and `0.546/0.748`). Samples preserve intended mode-conditioned prompts. |
| Hard-attribute NL validity re-eval | Forced NL-only eval `3758371_[3-5,9-11,15-17,21-23,27-29%3]` via `scripts/slurm/jobs/posthoc_paired_attribute_constraints_hard_full_eval_2026-06-10.slurm`; accidental full-array submit `3758370` canceled before start | Complete. `3758371` completed all `15/15` NL rows by 2026-06-20 and the report was refreshed on 2026-06-22. | Overwrote NL rows in `$HPCVAULT/synthetic-RLVL/passk_eval/paired_attribute_constraints_hard_full_20260610/` | `synthrlvl/natural_logic.py` now supports the controlled hard-attribute NL proof grammar. Report-ingested mean hard-tail NL `correct/parse/joint@16` by train max `5/10/15/20/25` is `0.420/0.833/0.006`, `0.791/0.800/0.660`, `0.762/0.766/0.661`, `0.862/0.824/0.769`, `0.808/0.815/0.795`; depth-50 translated joint remains weak (`0.000/0.031/0.000/0.010/0.073`). |
| Nanotron FlashAttention install | one-off installer `3758362` via `scripts/slurm/jobs/install_flash_attn_nanotron_2026-06-19.slurm`; source-rebuild repairs `3758382`, `3760040`, and `3760067` via `scripts/slurm/jobs/repair_flash_attn_nanotron_source_2026-06-19.slurm` | Complete. Initial job installed `flash-attn==2.8.3.post1` but failed the import check; first source repair `3758382` OOMed after `00:27:01`; resolver-risk retry `3760040` was canceled; corrected no-deps retry `3760067` completed after `03:29:56`. | Logs: `logs/install_flash_nanotron_3758362.*`, `logs/repair_flash_nanotron_{3758382,3760040,3760067}.*`; target env: `$WORK/nanotron` | Verified on 2026-06-22: `$WORK/nanotron` imports `torch 2.6.0+cu124`, CUDA `12.4`, and `flash_attn 2.7.4.post1`. Do not upgrade this env to FlashAttention 2.8.x unless Torch is upgraded/tested too. |
| Nanotron launch and Qwen2.5 batch probe | failed tiny smoke `3768303`; failed/canceled OLMo-shaped batch probe `3768304_[0-3%1]`; grouped-GEMM installers `3768308` and `3768311`; successful tiny smoke `3768319`; canceled OLMo-shaped proxy `3768322`; Qwen2.5 probe `3771016_[0-4%1]` | Dense Nanotron launch is verified. Probe rows for micro-batches `1/2/4` completed; rows for micro-batches `8/16` failed with CUDA OOM on A100-80GB at seq len 4096. | Logs: `logs/nanotron_qwen25_7b_probe_3771016_*.{out,err}`. | Current memory envelope for `Qwen/Qwen2.5-7B`, TP=4, DP=2, seq len `4096`: microbatch `4` fits; `8` and `16` do not fit without changing recompute/sequence length/parallelism or using gradient accumulation from a smaller microbatch. |
| Nanotron pretrained Qwen2.5 real-data smoke | integrated smoke `3771017` via `scripts/slurm/jobs/nanotron_qwen25_pretrained_realdata_smoke_2026-06-22.slurm` | Failed after data export/debug printing. The tiny weighted mixture and decoded packed chunk look correct; failure was `ImportError: attempted relative import with no known parent package` from invoking `examples/llama/convert_hf_to_nanotron.py` as a script. The wrapper is now patched to launch conversion through `torchrun --standalone --nproc_per_node=1 -m examples.llama.convert_hf_to_nanotron`, but the smoke has not been resubmitted because the production prereq builder supersedes it. | Logs: `logs/nanotron_qwen25_realdata_3771017.{out,err}`. Converted checkpoint target is now present at `$HPCVAULT/synthetic-RLVL/nanotron_checkpoints/qwen25_7b_tp1`. | Tiny printed data showed 4 normal, 16 logic, and 16 NL records, packed into fixed causal-LM chunks. Use completed prereq `3797409` as the current conversion/tokenization path. |
| Qwen2.5 proof-mixture midtraining and downstream eval | repaired control/NL weights; logic `3830927 -> 3835442 -> 3847802 -> 3847804/3847805/3847806`; corrected control/NL adapters `3850351/3850352`; corrected evals `3850385..3850388`; strict aggregate `3850389`; final suite smoke `3834836` | All three training conditions have accepted step-8192 weights. Logic upload is capacity-pending; all prior control/NL downstream outputs remain quarantined after the RoPE metadata audit and are being rerun. | Corrected output root remains `$HPCVAULT/synthetic-RLVL/lm_eval_results/qwen25_branchproof_unique_v2_pilot_20260710`; invalid bundles carry `.rope10000_invalid_20260713` suffixes. Every GPU production eval is A100-80GB-only and artifact-audited. | Inspect corrected strict direct versus post-instruction tables from `3850389` before launching the broader grid. |
| Shortcut-kind controls | build `3674886_[0-3%2]`, original SFT `3674887_[0-23%3]`, replacement SFT `3682458_[22%1]`, eval `3674888_[0-23%4]` | build and SFT complete after replacement `3682458_22`; eval rows `3674888_0..23` complete. | `$WORK/synthetic-RLVL/passk_eval/hfsa_shortcut_kind_ablation_20260529/` | Tests `position` and `initial_marker` shortcuts at rates `0.5` and `0.8`, both templates, three seeds. Eval is shortcut-neutral. All `24/24` JSONs are report-ingested: `position` rate `0.5` logic OOD correct/joint@16 `0.900/0.619`, depth-50 `0.844/0.312`; `position` rate `0.8` logic OOD `0.879/0.650`, depth-50 `0.760/0.323`; matched `nl_exact` rate `0.5` OOD `0.540/0.431`, depth-50 `0.396/0.260`; matched `nl_exact` rate `0.8` OOD `0.512/0.487`, depth-50 `0.396/0.354`; `initial_marker` logic rates `0.5/0.8` OOD `0.883/0.625` and `0.885/0.610`, depth-50 `0.854/0.344` and `0.865/0.344`; `initial_marker` `nl_exact` rates `0.5/0.8` OOD `0.469/0.421` and `0.771/0.702`, depth-50 `0.115/0.094` and `0.667/0.500`. |
| HFSA batch-size ablation | canceled first SFT `3695143`, canceled first eval `3695147`; replacement SFT `3695197_[0-11%3]`; completed recoveries `3698877_[2%1]`, `3702079_[6%1]`, `3705794_[10%1]`, and `3722466_[7,11%2]`; replacement eval `3722467_[0-15%4]` | Complete: SFT `12/12`, eval `16/16`. | SFT finals under `$HPCVAULT/synthetic-RLVL/runs/sft_hfsa_batch_bsz{2,4,8,16}_{logic,nl_exact,conditioned_dual}_train1to20_10k_seed3407/final`; eval JSONs under `$HPCVAULT/synthetic-RLVL/passk_eval/hfsa_batch_size_ablation_20260603/` | Interpret bsz16 as effective batch size, not physical microbatch size; true physical bsz16 OOMed earlier on A100-80GB. Report now includes `hfsa_batch_size_ablation_diagnostics.csv` and `hfsa_batch_size_conditioned_delta.{pdf,png}`. The result does not support a simple "larger stratified batches fix conditioned dual" story: conditioned-logic OOD joint is best at bsz2 (`0.618`) and near-tied at bsz16 (`0.587`), while conditioned-NL OOD/depth-50 joint is best at bsz2 (`0.781/0.344`) and often worsens at larger batches. |
| Typed maze rerun | build `3695237`, original SFT `3695238_[0-29%3]`, completed recovery SFT `3705793_[12,22%2]`, canceled high-cap eval `3705795_[0-29%3]`, replacement eval `3722471_[0-29%3]`, partial recovery `3743047_[9-29%3]`, final missing-row recovery `3748683_[15-29%15]` | Complete: build/SFT/eval are `30/30`. High-cap eval `3705795` produced `0` JSONs and was canceled. The wrapper now uses formal `PASSK_MAX_NEW_TOKENS=4096` by default while leaving NL default `6144`. Final recovery rows `27..29` completed cleanly on 2026-06-18. | Dataset root `$HPCVAULT/synthetic-RLVL/datasets/materialized_paired_maze_navigation_typed_20260603/`; SFT finals under `$HPCVAULT/synthetic-RLVL/runs/sft_paired_maze_typed_{logic,nl_exact}_train1to{5,10,15,20,25}_10k_seed{3407,3408,3409}/final`; eval JSONs under `$HPCVAULT/synthetic-RLVL/passk_eval/paired_maze_typed_sparse_20260603/` | Maze generator now uses typed formal symbols (`r_<room>` for rooms, `k_<key>` for keys). Typed maze remains a clear negative result: logic OOD/depth-50 joint is `0.000` through train-1-to-25; NL train-1-to-25 OOD/depth-50 correct@16 is `0.111/0.000`. Sample checks show long depth-25/50 logic and NL generations copy premises/partial traces and omit `<answer>`. |
| Semantic iGSM rerun | failed build `3695464`, canceled dependents `3695465`/`3695466`; replacement build `3695525`, SFT `3695526_[0-29%3]`, eval `3695527_[0-29%3]`, quota recovery eval `3702073_[24-28%3]`, clean NL-only re-eval `3705807_[3-5,9-11,15-17,21-23,27-29%4]` | Complete: build/SFT/eval are `30/30`, and the clean NL-only forced re-eval completed all `15/15` NL rows. Corrected metrics: NL parser coverage is near-complete, but OOD/depth-50 translated joint remains `0.000`; NL train-1-to-25 OOD/depth-50 correct@16 is `0.873/0.677`, much higher than logic `0.612/0.281`. Logic has nonzero internal validity but strict grounded joint remains `0.000`. | Dataset root `$HPCVAULT/synthetic-RLVL/datasets/materialized_paired_official_igsm_semantic_20260603/`; SFT finals under `$HPCVAULT/synthetic-RLVL/runs/sft_paired_igsm_semantic_{logic,nl_exact}_train1to{5,10,15,20,25}_10k_seed{3407,3408,3409}/final`; eval JSONs under `$HPCVAULT/synthetic-RLVL/passk_eval/paired_igsm_semantic_sparse_20260603/` | Parser coverage is no longer the blocker. Remaining validity failure is generated long-trace drift/truncation and strict grounding/equivalence, so report conclusions should separate answer accuracy from validated reasoning. |
| Active recovery oversight | one-off oversight jobs `3715439`, `3715440`, `3715441`, `3715442`, `3715443` using `scripts/slurm/codex/active_recovery_oversight_2026-06-10.slurm` | Complete. `3715439`, `3715441`, `3715442`, and `3715443` completed cleanly; `3715440` completed its local audit/commit attempt but was canceled after a stuck fallback push. | Handoff docs, report artifacts, and targeted recovery submissions if needed. | No further one-off oversight is scheduled. |
| New ablation oversight | completed oversight `3687984`, completed oversight `3688814`, completed oversight `3689677`, completed oversight `3690212`, completed oversight `3690645`, next oversight `3691029` | `3687984`, `3688814`, `3689677`, `3690212`, and `3690645` completed cleanly; next plan-driven pass `3691029` is begin-time pending | handoff updates, result analysis, report updates, targeted recovery jobs, and triggered backlog submissions if appropriate | The latest ablation pass found no unrecovered failures or scheduler edits to make, regenerated/mirrored the report after new hybrid and conditioned-dual JSONs, and deferred broad next launches until remaining hybrid, conditioned-NL checkpoint, conditioned final, and paired non-iGSM rows complete. |

## Partition Audit

Checked at 2026-07-15 07:18 CEST. Active repo jobs are the corrected
report-wide BranchProof training/eval matrix and declaration-fixed baseline
rows `3857767_0/1`; corrected Nanotron downstream and multi-hop jobs are
complete. Do not
widen full Nanotron training to
`rtxpro6k`: current torch does not support Blackwell `sm_120`.

Declaration-fixed BranchProof rows `3857767_0/1` are active on A100-80GB;
remaining rows require the same resources and are capacity/throttle
pending. No compatible partition or feature widening is available.

Visible `grid_goal_struct_eval` dependency-failed jobs belong to
`/home/hpc/c107fa/c107fa12/sequence-editing` and are not managed here. No
visible `tjepa_*` job belongs to this repo.

## Watch Rules

- Preserve the accepted Nanotron checkpoint and adapter repositories through
  any remaining report work; the upload, downstream, and multi-hop chains are
  complete and must not be revived from older RoPE-invalid bundles.
- Watch declaration-fixed BranchProof eval `3857767`; it is A100-80-only.
  Inspect the first row's device,
  runtime, cap hits, metrics, and raw generations before broad scientific use.
- If a Nanotron row fails again, inspect the matching `logs/nano_q25_midtrain_<array>_<row>.{out,err}` before touching downstream dependencies.
- If new science jobs are launched later, restore normal watch rules: inspect `squeue`/`sacct`, logs, output roots, and compatible partitions; resubmit only failed rows; update handoff docs after every scheduler/report change.

## Commands

```bash
source ./scripts/env.sh
squeue -u c107fa12 -o '%.18i %.9P %.34j %.2t %.11M %.6D %.24E %R'
sacct -j 3823434,3828946,3829067,3829068,3829069,3829072,3829073,3830855,3830924,3830927,3830928,3831110,3831111,3831112,3831113,3831115,3831119,3831121,3831135,3831136,3831179,3832945,3833178,3833179,3834582,3834706,3834707,3834728,3834737,3834738,3834836,3834904,3834905,3834906,3834907,3834911,3835433,3835438,3835442,3835443,3835779,3835927,3835928,3838163,3847756,3847757,3847792,3847802,3847804,3847805,3847806,3847808,3849774,3849775,3849776 --format=JobID,JobName%34,State,Elapsed,ExitCode -n -P
```

Useful log tails:

```bash
for f in \
  logs/sft_pair_full_3672212_*.out logs/sft_pair_full_3682411_*.out logs/sft_pair_full_3683070_*.out logs/pair_full_eval_3682449_*.out logs/pair_full_eval_3691024_*.out logs/pair_full_eval_3689003_*.out \
  logs/sft_hfsa_trace_ctl_3661118_*.out logs/hfsa_trace_ctl_eval_3661119_*.out \
  logs/hfsa_trace_ctl_eval_3682459_*.out logs/hfsa_trace_ctl_eval_3682460_*.out \
  logs/sft_hfsa_shortcut_3671431_*.out logs/hfsa_shortcut_eval_3671432_*.out \
  logs/hfsa_hybrid_eval_3670783_*.out logs/hfsa_hybrid_eval_3682461_*.out \
  logs/sft_hfsa_word_3674875_*.out logs/sft_hfsa_word_3674875_*.err logs/hfsa_word_eval_3674876_*.out \
  logs/sft_hfsa_cond50k_3674879_*.out logs/sft_hfsa_cond50k_3674879_*.err logs/sft_hfsa_cond50k_3682457_*.out logs/sft_hfsa_cond50k_3682492_*.out logs/hfsa_cond50k_eval_3674884_*.out logs/hfsa_cond50k_ckpt_3674885_*.out \
  logs/build_hfsa_shkind_3674886_*.out logs/sft_hfsa_shortkind_3674887_*.out logs/sft_hfsa_shortkind_3674887_*.err logs/sft_hfsa_shortkind_3682458_*.out logs/hfsa_shortkind_eval_3674888_*.out \
  logs/sft_hfsa_bsz_3695143_*.out logs/sft_hfsa_bsz_3695143_*.err logs/hfsa_bsz_eval_3695147_*.out logs/hfsa_bsz_eval_3695147_*.err logs/sft_hfsa_bsz_3695197_*.out logs/sft_hfsa_bsz_3695197_*.err logs/hfsa_bsz_eval_3695199_*.out logs/hfsa_bsz_eval_3695199_*.err \
  logs/build_maze_typed_3695237.* logs/sft_maze_typed_3695238_*.out logs/sft_maze_typed_3695238_*.err logs/maze_typed_eval_3695239_*.out logs/maze_typed_eval_3695239_*.err \
  logs/build_igsm_sem_3695464.* logs/build_igsm_sem_3695525.* logs/sft_igsm_sem_3695526_*.out logs/sft_igsm_sem_3695526_*.err logs/igsm_sem_eval_3695527_*.out logs/igsm_sem_eval_3695527_*.err \
	  logs/hfsa_ablate_oversight_3678051.* logs/hfsa_ablate_oversight_3679095.* logs/hfsa_ablate_oversight_3680036.* logs/hfsa_ablate_oversight_3680038.* logs/hfsa_ablate_oversight_3682409.* logs/hfsa_ablate_oversight_3683563.* logs/hfsa_ablate_oversight_3683966.* logs/hfsa_ablate_oversight_3684370.* logs/hfsa_ablate_oversight_3687984.* logs/hfsa_ablate_oversight_3690212.* logs/hfsa_ablate_oversight_3690645.* \
	  logs/paired_full_oversight_3680037.* logs/paired_full_oversight_3680039.* logs/paired_full_oversight_3682410.* logs/paired_full_oversight_3683024.* logs/paired_full_oversight_3683562.* logs/paired_full_oversight_3683967.* logs/paired_full_oversight_3684369.* logs/paired_full_oversight_3685027.* logs/paired_full_oversight_3687983.* logs/paired_full_oversight_3688815.* logs/paired_full_oversight_3689676.* logs/paired_full_oversight_3690207.* logs/paired_full_oversight_3690641.*; do
  [ -f "$f" ] && echo "### $f" && tail -n 20 "$f"
done
```
