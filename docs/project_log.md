# Project Log

Short dated notes for useful operational events, cleanup decisions, results updates, and handoff changes. Keep this concise; move bulky history to experiment-specific docs or archives.

## 2026-07-16

- 13:13 CEST baseline eval rows 13/14 failed before evaluation on repeated Hub
  `504`/timeouts during OLMo-3 merge. Added a local base-model override,
  submitted exact A100-80 recovery `3863525_[13-14%2]` and CPU audits
  `3863527_[13-14%2]`, and rewired aggregate `3857769` to require both. The
  qualitative gate now follows each accepted row audit's exact log provenance
  and rejects incomplete sampled logs; focused tests pass (`7 passed`). Rows/audits 0..10
  are accepted at row scope. Raw logic review across train-15/train-20 seeds
  and depths `1/15/20/25/50` confirms clean train-band behavior and correctly
  rejected parse, proof, answer, format, and cap failures; no modality result
  was accepted.
- 13:13 CEST batch recovery `3859299_[3-5]` timed out at 24 hours with valid
  optimizer/scheduler/RNG `checkpoint-3000` states. Submitted exact resume
  `3863546_[3-5%3]` and changed eval `3850122` to require it after both old
  arrays terminate. Dolmino `6e-6` row `3859711_0` completed 256 steps in
  `02:27:56` at 15.4K tokens/s with finite diagnostics and terminal manifest;
  `3e-6` row 1 is running. Successor `3863505` is preserved. Vault is 827G
  and 187k files.
- 13:17 CEST committed the recovery, audit, code, and handoff changes as
  `49d168f`. A bounded 90-second SSH push produced no output and timed out;
  after this publication-state record, local `main` is two commits ahead of
  `origin/main`. The report repo was
  unchanged because the matched BranchProof gate is incomplete.

- 09:32 CEST the 4-GPU Dolmino `6e-6` LR row `3859711_0` started on A100-80GB
  node `a0832` and reached step 8/256 at 15.4K tokens/s, loss about `0.52`,
  finite gradients, and 62.4GB peak allocation. Rows 1/2 estimate 10:48 CEST;
  the 8-GPU fallback now estimates 2026-07-17 02:30 and remains only until the
  remaining lower-resource rows start. Declaration-fixed baseline row/audit 7
  completed and passed, bringing accepted row-scoped audits to `8/30`; bounded
  depths `1/15/18/25/50` show intended shallow validity and rejected deep
  declaration/format collapse, without releasing a modality claim.

- 07:20 CEST accepted declaration-fixed baseline rows `3857767_0..6` at row
  scope after audits `3857768_0..6` passed the complete artifact and
  strengthened declaration/validity gates. This completes three logic seeds
  at train maxima 5 and 10 plus one train-15 seed. New row-6 raw inspection at
  depths `1/15/18/20/25/50` found clean shallow/train-band proofs and correctly
  rejected unsupported, wrong-answer, repetitive, capped, and duplicate-
  declaration deep traces. No matched modality or report claim was accepted.
- 07:20 CEST baseline rows `7..12` were running on verified A100-80GB nodes at
  sampled chunks `102/88/89/88/71/52` of 112, with rows `13..29` throttle-
  pending and no fatal signature. Batch recoveries `3859299_[3-5]` reached
  about `2.51k/10k` after 18 hours; preserve their checkpoints and recover
  only after actual timeout. Conditioned-50k original rows `10/11/12` were at
  approximately `6,293/6,233/972` of 50,000 under the existing staged chain.
- 07:20 CEST Dolmino gates `3859297` and `3859711_[0-2]` remained
  account-GRES pending with projections near 21:15 and 08:45 CEST. Running
  CPU-only watcher `3862186` had already scheduled successor `3862431` for
  about 13:01 CEST; it was preserved because the corrected baseline, report
  matrix, Dolmino path, and report replacement remain incomplete. Vault use
  was about 717 GiB with 151 protected Trainer checkpoints.
- 07:23 CEST committed the seven accepted row audits and handoff update as
  `e1dc824`. Publishing remains externally blocked: the configured SSH push
  made no progress before its 90-second timeout. Local `main` is four commits
  ahead of `origin/main`; the queued successor should retry without discarding
  them. The report repository was unchanged because no matched result gate is
  complete.

- 01:05 CEST accepted declaration-fixed baseline rows `3857767_3/4` only at
  row scope after CPU audits `3857768_3/4` passed exact shape, metric, chunk,
  cap, fresh-constant, answer, declaration, and validity gates. Raw review at
  depths `1/5/10/12/25/50` shows clean train-band proofs and expected
  wrong-branch, malformed, repetitive, and cap-hit deep failures. These are
  two logic seeds, not matched evidence, so no report metric was accepted.
- 01:05 CEST baseline rows `1/2/5/6/7` were at sampled chunks
  `109/101/109/92/38`, with row 8 starting on A100-80GB. Row 1 is close to
  completion at 18 hours and does not justify sharding. Conditioned-50k rows
  `7/8` were near the 24-hour limit at `42,816/43,078` steps and remain
  correctly covered by staged after-any resumes `3850110..3850112`. Batch
  recovery `3859299_[3-5]` reached about 1.66k steps with 1,000-step
  checkpoints; no duplicate recovery was submitted.
- 01:05 CEST Dolmino gates `3859297` and `3859711_[0-2]` remained
  account-GRES pending. The running CPU-only watcher scheduled CPU-only
  successor `3862186` for 07:01 CEST; it is preserved because the baseline,
  report matrix, Dolmino gate, and report replacement remain incomplete.
  Vault use is about 706 GiB.

## 2026-07-15

- 19:10 CEST accepted the first declaration-fixed baseline artifact only at
  row scope. Eval `3857767_0` completed in `11:07:55`, audit `3857768_0`
  passed all structural/log/cap/constant/answer/declaration/validity gates,
  and raw depths `1/5/25/50` show shallow valid proofs plus correctly rejected
  malformed and duplicate-declaration long traces. Sampled correct/joint
  pass@1 is `1.000/1.000` at depths 1 and 5, `0.066/0.000` at 25, and
  `0.037/0.000` at 50. No family or modality claim was accepted.
- 19:10 CEST live audit found baseline rows 1..5 at chunks
  `84/67/92/85/75`, row 6 in greedy generation, and no fatal signature. Row 1
  remains below the 20-hour sharding trigger. Conditioned-10k row 0 completed
  both modalities with clean shallow/degraded deep raw behavior; the family is
  partial. Batch recovery `3859299_[3-5]` is slow at steps `815/815/821` after
  six hours and has not yet reached its first new checkpoint; do not create a
  second recovery until a row actually times out. Vault use is about 691 GiB.
- 19:10 CEST shared-LR jobs `3859297` and `3859711_[0-2]` remain account-GRES
  pending. CPU-only watcher `3859290` scheduled CPU-only successor `3860702`
  before starting, and the successor was preserved because the plan remains
  incomplete.
- 19:15 CEST committed the row audit and handoff update as `07a9840`.
  Publishing remains externally blocked: the configured SSH push made no
  remote progress before its 90-second timeout, and the
  `ssh.github.com:443` fallback timed out during connection. Local `main` is
  one commit ahead of `origin/main`; the successor should retry without
  discarding the commit. The report repository was unchanged.

- 15:24 CEST designed the conditional 20B Dolmino production experiment.
  Use one continuous 38,147-step schedule with immutable weight milestones at
  `9537/19073/28610/38147` (`5/10/15/20B` tokens), evaluate all conditions at
  5B, and continue all three only under a predeclared downstream-signal gate.
  The scheduler has the 20B horizon from step 1; instruction tuning branches
  from snapshots and never alters resumable bases. Eventual data must come from
  the 100B Dolmino release: at least 21B normal plus 1.1B formal and 1.1B NL,
  built in bounded shards with a shared replacement-slot schedule. Rotate full
  optimizer states while retaining verified BF16 milestone weights; current HF
  quota cannot be assumed to hold all 12 snapshots.

- 15:03 CEST Slurm assigned 2026-07-16 03:49 CEST estimates to all three
  matched 4-GPU LR rows `3859711_[0-2]`, versus 10:48 CEST for sequential
  8-GPU fallback `3859297`. All remain account-GRES pending; estimates are
  projections, not guaranteed reservations.

- 15:01 CEST added a matched lower-resource race for the Dolmino LR gate.
  Parameterized the training wrapper by topology with hard checks that world
  size and global batch remain exact. Submitted independent 4xA100-80GB rows
  `3859711_[0-2%3]` using TP4/DP1, microbatch 4, and accumulation 32, matching
  the 8-GPU gate's global batch 128 and 134.2M tokens per LR. The existing
  sequential 8-GPU gate `3859297` remains queued and now has a 2026-07-16
  10:48 CEST estimate; 4-GPU rows have no firm estimate. Config-only checks
  passed for both topologies, along with `bash -n` and `git diff --check`.

- 14:37 CEST accepted the recovered Dolmino prerequisite. Nanoset processing
  completed with one nonempty 19.24GB shard and `4,810,706,180` packed tokens;
  the exact `10,705,908`-token increase over source tokens confirms one EOS per
  record. Metadata, source composition, raw samples, capacity, and terminal
  state pass. Sequential LR gate `3859297` is normal account-GRES pending;
  Slurm currently estimates 2026-07-16 12:49 CEST.

- 13:25 CEST Dolmino recovery `3859296_0` reached `4,800,000,272` tokens in
  `10,705,908` records across 120 source groups without replay. Manifest sums
  and five byte-spaced raw records pass. Nanoset preprocessing is running;
  replacement LR gate `3859297` remains dependency-held until metadata,
  nonempty dataset, EOS, and terminal checks pass. Full tests pass at
  `232 passed, 3 skipped`.
- 13:06 CEST recovered two terminal paths. Dolmino row `3858584_0` failed at
  `4.128B/4.8B` tokens on a transient Hub HTTP 500; resume state is intact.
  Added bounded download retries (`2 passed`), submitted recovery `3859296`,
  canceled dependency-dead unstarted LR gate `3858902`, and submitted
  replacement `3859297`. Batch rows `3850114_3/4/5` timed out near `94--95%`;
  submitted exact recovery `3859299_[3-5%3]` under the new checkpoint policy
  and rewired eval `3850122` to require it.
- 13:06 CEST baseline `3857767_0/1` reached sampled chunks `82/55`; rows
  `2/3/4` entered greedy generation on A100-80GB. Conditioned-10k eval row 1
  completed; raw NL checks across depths `1/5/10/25/50` found clean shallow
  translated reasoning and deeper failure. No family conclusion or report
  refresh was made. Watcher `3858016` preserved successor `3859290`.

- 11:49 CEST consolidated the short Dolmino LR gate into one full-node
  allocation. Formal/NL prerequisite rows completed cleanly; Dolmino row 0 is
  healthy at about 2.13B/4.8B tokens. Canceled dependency-pending jobs
  `3858587/3858588` before start and submitted `3858902` after the build. It
  requests 12 hours on 8xA100-80GB and runs the three 256-step LRs
  (`6e-6`, `3e-6`, `1e-5`) sequentially. Slurm's test-only estimate was the
  same for 6-, 12-, and 24-hour requests; consolidation removes requeue gaps,
  while a 24-hour limit itself would not accelerate scheduling.

- 10:52 CEST implemented and submitted the gated Dolmino midtraining path.
  A local `datasets` streaming smoke rejected the release's `default` config
  because heterogeneous metadata cannot be unified into one Arrow schema; no
  production artifact was written. Added a resumable direct JSONL.zst Hub
  reader that shuffles the 1,113 released shards deterministically, preserves
  native text, and records source-token counts. Added/tested neutral paired
  export with shared `Solution/Context/Derivation/Conclusion/Final answer`
  headings and no modality tags. Build `3858584_[0-2]` started on RTX Pro 6000.
  Submitted shared-LR control gate `3858587_0` (`6e-6`) after the build and
  `3858588_[1-2%1]` (`3e-6`, `1e-5`) after the first row; each is 256 steps
  and 134.2M tokens with no large checkpoint. Full control/formal/NL runs are
  intentionally deferred until LR selection and short p5 modality checks.

- 10:18 CEST corrected the Nanotron experiment terminology and next-corpus
  plan. The completed Qwen2.5/FineWeb-Edu comparison is continual pretraining,
  not midtraining. Inspected the official Dolma 3 Dolmino 10B/100B releases
  and representative source records: Dolmino is a heterogeneous plain-text
  stage-2 mix whose shared contract is text plus EOS, not a universal chat
  template. Selected the 10B micro-anneal release for a future 4.3B-token
  pilot. Dolmino records will remain native; paired formal/NL injections will
  share `problem\n\nSolution:\n{trace}\n\nFinal answer: {answer}` with no
  modality tags. Proposed replacement-token fractions `{0,2,5,10}%`, matched
  source order and token budgets, and no job submission.

- 10:09 CEST established the next Nanotron format/readout gate. FineWeb input
  is raw document text plus EOS, while proof interventions use `<question>`,
  modality-specific `<formal>`/`<think>`, nested proof tags, and `<answer>`.
  Future formal, NL, and ordinary-reasoning interventions will use the same
  neutral `Problem/Reasoning/Conclusion/Final answer` envelope. The completed
  UltraChat readout is generic assistant-only rank-16 LoRA at effective batch
  one and exhibited repetition/extraction failures; future pilots must retain
  direct eval and apply identical answer-only calibration plus identical
  modality-neutral reasoning SFT to every checkpoint. No jobs were submitted.

- 09:54 CEST completed the Nanotron mixture/schedule postmortem. Exact
  production-path replay shows randomized, not strict, batches: 6--35 proof
  chunks per 128-chunk update, mean `19.2001`, near-binomial standard deviation
  `4.0848`, and no proof-empty global update. Nanoset randomizes packed sample
  indices despite one nonempty pretokenized shard per source. Reproduced a
  scheduler resume bug (`5.94e-6 -> 6.25e-6`) caused by
  normalizing a reconstructed LambdaLR with checkpoint-current rather than
  original LR. Patched the local Nanotron checkout to use `initial_lr`, added a
  regression test, and changed the repo job template to derive the post-warmup
  decay span. Also recorded exact global stratification as an optional ablation
  and the stronger full-document/response-manifold mismatch diagnosis. No new
  training jobs were submitted.

- 07:18 CEST first declaration-fixed runtime check: `3857767_0/1` remain on
  verified A100-80GB devices. Row 0 completed greedy chunks 1--3 and row 1
  completed chunks 1--2; deeper chunks hit the intended 7,168-token cap, with
  no fatal signature. Conditioned-10k eval rows reached sampled chunks
  `64/87/39` of 112. Batch SFT rows `3850114_3/4/5` reached
  `7282/7335/7279` of 10,000 after about 18.5 hours and remain likely to hit
  the hard 24-hour limit; recover only an actually failed row. Successor
  `3858016` remains begin-time pending and was preserved.
- 07:18 CEST created local audit commit `2cf8a08` and official-report commit
  `afb2d34`. Direct `git push origin main` attempts for both repositories timed
  out after 90 seconds with no remote output. This repo and the report repo
  remain ahead of `origin/main`; the successor should retry publication.
- 07:06 CEST accepted prompt-fixed multi-hop direct `3855271`, instruction
  `3855272`, and aggregate `3855273`. All six 1,200-row bundles pass corrected
  RoPE, 32,768-window, prompt, cap, and coverage gates. Direct stock
  control/logic/NL QA-F1 `0.189/0.250/0.238` becomes
  `0.349/0.361/0.367` under answer-head sensitivity; direct tagged logic/NL
  usually launch their learned trace substrate into the cap. The result is a
  response-control diagnostic, not positive transfer. Persisted analysis:
  `analysis/nanotron_branchproof_unique_v2_multihop_promptfix_20260714/`.
- 07:06 CEST declaration-fixed baseline rows `3857767_0/1` started on verified
  A100-80GB devices. Conditioned-10k eval `3850119_0/1/2` is healthy at chunks
  `63/85/37` of 112. Conditioned-50k row `3850109_6` timed out at step
  43,105/50,000 and remains covered by staged resumes `3850110..3850112`.
- 07:06 CEST batch SFT rows `3850114_3/4/5` are walltime-risk at about 72%
  after 18.3--18.5 hours, with a hard 24-hour partition ceiling. Patched the
  pending batch rows to retain only the latest 1,000-step checkpoint; submit
  targeted recovery only for rows 3/4/5 if they actually time out. Repo-owned
  Vault use is `439.6 GiB` with 110 protected Trainer checkpoints. Watcher
  `3857722` preserved successor `3858016`.
- 07:06 CEST updated the official root preprint with the accepted one-run p15
  null/mixed table and the multi-hop response-control audit. Historical
  BranchProof quantitative sections remain disabled, and the informal report
  was not regenerated from its quarantined roots. Static structure/diff checks
  pass; no local TeX engine is available for compilation.

- 01:17 CEST conditioned-50k row `3850109_5` completed its active 50k chunk
  in `19:45:22` and row 7 started. Direct multi-hop rows `3855271_0/1` are
  actively generating at `388/1,200` and `268/1,200` requests with no fatal
  signature; declaration-fixed baseline `3857767` remains account-GRES
  pending behind active report/multi-hop work.
- 01:08 CEST raw BranchProof review superseded the first eight exact-answer
  logic bundles. Generated formal wrappers can redeclare a predicate symbol
  for different state words while the old evaluator ignores declarations and
  credits internal citation-free validity. Across `7,168` retained sampled
  rows, `1,616` contain duplicates, `94` were credited citation-free valid,
  and `53` were credited correct-and-valid (`46` at depth 20). No prior
  `3853284` metric remains evidence.
- 01:08 CEST patched `OutputEvaluator` so duplicate constant/predicate
  declarations force format, syntax, and every formal validity metric to zero;
  strengthened the row audit to reject credited duplicates. Official iGSM
  empty predicate blocks and case-distinct arithmetic symbols remain valid.
  Focused checks and the complete suite pass: `226 passed, 3 skipped`.
- 01:08 CEST canceled old `3853284/3853285/3853286`, quarantined 16 eval
  files, 8 audit JSONs, and 44 logs under the repo/Vault
  `pre_declaration_fix_20260715` quarantine, and submitted unchanged-protocol
  A100-80 replacement `3857767_[0-29%6] -> 3857768_[0-29%8] -> 3857769`.
  The replacement is account-GRES pending. Cancel cleanup removed all obsolete
  merge roots; repo-owned vault use is `326,520,653 KiB` with 95 report
  checkpoints.
- 01:08 CEST direct multi-hop `3855271_0/1` started on verified A100-80GB
  nodes and passed consumer `rope_theta=1000000` plus 32,768-window preflight;
  row 2 is throttle-pending, instruction `3855272` is GRES-pending, and
  aggregate `3855273` remains held. Watcher `3857212` preserved recorded
  successor `3857722` for 06:49 CEST. No report/preprint regeneration ran.
- 01:14 CEST committed the declaration-validity correction as `381b388`.
  `git push origin main` timed out after 60 seconds without a remote response;
  the correction plus this handoff update leave local `main` two commits ahead
  of `origin/main`. The report repo was not changed.

## 2026-07-14

- 21:52 CEST live refresh found clean BranchProof baseline rows/audits `0..5`
  accepted. These complete the three-seed logic train-1-to-5 and train-1-to-10
  blocks: OOD correct pass@1 increases `0.099 +/- 0.057 -> 0.313 +/- 0.020`,
  citation-free joint pass@1 increases `0.035 +/- 0.027 -> 0.245 +/- 0.012`,
  and hard-tail correct/joint pass@1 increases `0.060/0.011 -> 0.239/0.166`;
  depth-50 correct and joint remain near zero. This is within-logic scaling,
  not a logic-vs-NL result. Rows `6/7/8` are near completion at chunks
  `104/110/93`; rows `9/10` are at `27/17`, and row 11 is in startup.
- 21:52 CEST report/multi-hop operational audit found no new unrecovered fatal
  signature. Active corrected SFT rows are making progress; 32B row 2 is about
  `94%`, shortcut recovery `3856142_5/6` is about `67%/63%`, and conditioned
  50k rows `4/5/6` are about `82%/82%/58%`. The slower conditioned row is
  checkpointed and already covered by staged resume `3850110..3850112`.
  Multi-hop arrays `3855271/3855272` and 32B recovery `3854837` are cleanly
  account-GRES pending on required A100-80GB resources. Vault use is about
  `462G` as active report jobs create checkpoints; six live merge roots and 87
  Trainer checkpoints remain protected until their acceptance gates pass.

- 19:47 CEST reconciled authenticated per-repository Hugging Face storage
  after the completed logic checkpoint and adapter uploads: `79.281G` total
  (`50.846G` models, `28.435G` datasets, 66 repositories), leaving `20.719G`
  against the nominal quota. No repository was deleted. Preserve all three
  p15 checkpoints and their current adapters through pending multi-hop eval;
  guarded rotation remains a prerequisite if the rejected broader grid is
  ever reconsidered. Updated `analysis/hf_storage_cleanup_2026-07-13.json`.
- 19:39 CEST committed the accepted tiny/Nanotron analysis and audit hardening.
  Publishing is still blocked: configured GitHub SSH timed out after 55
  seconds, and `ssh.github.com:443` timed out during connection. `gh` is absent
  and HTTPS has no noninteractive credentials. Local `main` is two commits
  ahead of `origin/main` for the successor to retry.
- 19:26 CEST accepted the corrected tiny checkpoint curve. Replacement and
  recovery parents `3854813/3856145` are terminal; all 90 metric JSONs, 90
  sample JSONLs, and exact original/recovery chunk logs passed the structural,
  cap, fresh-constant, and retained-validity audit. Strengthened the audit so
  every citation-free-valid sample must have empty validity errors, no invalid
  lines, and line-valid fraction one even when strict validity is also true;
  the regression passes and all 90 real rows re-audited accepted. Raw review
  covered logic/NL, 50M/100M/200M, all seeds, 20k/60k/100k exposure,
  depths 1/10/50, correct/incorrect outputs, and cap degeneration. OOD and
  depth-50 modality-appropriate joint pass@1/4/8 remain zero throughout, so
  this is an accepted negative tiny mechanism result, not report evidence.
- 19:26 CEST ran the documented post-acceptance cleanup guard: verified 18
  nonempty finals, 90 metrics, 90 samples, 90 accepted audits, terminal parent
  jobs, and no live dependency, then deleted only the 90 tiny intermediate
  checkpoints (`102G`). Finals and curve artifacts remain; repo-owned
  `$HPCVAULT/synthetic-RLVL` now measures about `393G`.
- 19:26 CEST accepted corrected Nanotron p15 eval/aggregate
  `3854824_3 -> 3854847`. The six-bundle manifest is under
  `analysis/nanotron_branchproof_unique_v2_p15_20260711/`. Direct logic gives
  only `+0.0033` all-primary and `+0.0071` reasoning versus control while
  targeted logic falls `-0.0116`; NL and post-instruction deltas are similarly
  small/mixed. Correct/incorrect raw review found increased direct
  next-document continuation for both proof mixtures, long instruction
  repetition, and a BBH instruction extraction floor despite correct leading
  choices. The broader mixture grid was therefore rejected as neither positive
  nor sample-clean. Prompt-fixed multi-hop arrays remain pending as evaluation
  of the existing checkpoints.
- 19:26 CEST clean BranchProof rows `3853284_0/2/3/4/5` and their CPU audits
  are accepted. Row 1 reached sampled chunk 110/112 after about 18 hours,
  still below the depth-sharding trigger; rows 6/7/8 are active. Current
  CPU-only watcher `3856057` preserved CPU-only successor `3857212` for 00:49
  CEST because the baseline, report matrix, multi-hop, and report gates remain
  incomplete. Full repository verification passes `223 passed, 3 skipped`.
- 13:22 CEST tracked oversight changes were committed locally, but pushes via
  the configured SSH remote and SSH port 443 timed out; HTTPS had no
  non-interactive credentials. The branch remains one commit ahead of
  `origin/main` for the successor watcher to retry.
- 13:13 CEST clean baseline and recovery pass: BranchProof eval rows
  `3853284_0/3/5` completed in `10:09--11:05` and audits
  `3853285_0/3/5` passed with full coverage and zero fresh-constant failures.
  Raw samples show clean train-band proofs, correct-but-invalid depth-25
  traces, and depth-50 cap-hit repetition; no cross-modality conclusion was
  accepted. Rows `1/2/4` remain healthy and rows `6/7` backfilled. Shortcut rows
  `3850213_5/6` failed only at W&B service startup, so exact recovery
  `3856142_[5-6%2]` was submitted and added to eval `3850214`'s gate. Seven
  tiny checkpoint rows were canceled before producing artifacts; exact
  recovery `3856145_[24,26,28-32%3]` was submitted and safely widened to
  generic one-GPU `a40,a100`.
- 13:13 CEST multi-hop instruction-gate repair: prompt-fixed direct smoke
  `3855269` passed. Instruction smoke `3855270` completed its 12 generations,
  RoPE preflight, and 32,768-window run, but the audit assumed an untemplated
  prompt began at byte zero. The audit now unwraps exactly one Qwen user turn;
  a regression and the full suite pass (`223 passed, 3 skipped`). The stored
  smoke re-audited cleanly, CPU-only gate `3856131` completed `0:0`, and full
  instruction array `3855272` was rewired to it. Raw smoke generations remain
  bounded but show explanations/repetition and incomplete tag adherence, which
  must remain visible in the production analysis. Logic reviewer eval
  `3854824_3` is running cleanly; no report regeneration trigger is satisfied.

- 11:19 CEST reclaimed `93.9 GiB` from repo-owned `$HPCVAULT`, reducing
  `$HPCVAULT/synthetic-RLVL` from `622.9 GiB` to `529.0 GiB`. Deleted 60
  Trainer checkpoints from 30 completed corrected baseline SFT runs and nine
  from three completed Nanotron instruction-SFT runs only after verifying all
  33 final adapters; also removed superseded ambiguous pre-BranchProof
  logic/NL raw corpora and Nanosets plus known-invalid/incomplete eval, smoke,
  and quarantine outputs. Post-delete checks retained all finals, corrected
  BranchProof and normal-continuation data, converted Qwen checkpoint, six
  active `3853284` merge roots, and checkpoints needed by `3854813` and the
  corrected report arrays. Added backlog cleanup gates for terminal clean eval,
  accepted tiny curves, and accepted report-family eval/audit waves.

- 10:00 CEST repaired and submitted the Nanotron multi-hop downstream suite.
  Raw smokes showed that `Xnhyacinth/LongBench` stores its stock instruction
  wrapper inside `context` and stores a `Question:` prefix inside `question`;
  the old tagged and standard YAMLs duplicated both. Added shared document
  renderers that remove exactly the known wrapper, preserve passages, and
  normalize the question; reduced tagged short-answer generation from 512 to
  64 tokens; and made the audit reject nested prompts or stale caps. Verified
  real HotpotQA/2Wiki/MuSiQue records, all six task registrations, Slurm static
  validation, focused tests, and the full suite (`220 passed, 3 skipped`).
  Re-auditing both old smoke bundles now rejects every tagged task for the
  embedded wrapper, duplicated question, and 512-token cap, and every standard
  task for wrapper/question duplication.
  Submitted prompt-fixed smokes `3855269/3855270`, full baseline/NL/logic
  direct/instruction arrays `3855271/3855272`, and aggregate `3855273` through
  branch-specific `afterok` dependencies. The aggregate writes per-benchmark
  and compact baseline/NL/logic tables under
  `analysis/nanotron_branchproof_unique_v2_multihop_promptfix_20260714/`.
- 10:00 CEST clarified the corrected tiny result. These jobs cover only the
  50M/100M/200M one-pass scratch diagnostic; report-wide surface, hybrid,
  conditioned, architecture, batch, 32B, and shortcut corrections are separate
  arrays. Logic/NL use their intended modality-specific validity metrics. Raw
  depth-10 samples frequently reach the right label through a non-derivable
  branch, explaining zero joint without an evaluator mix-up. Old depth-50
  behavior is not comparable because the old generator was ambiguous above
  depth 17 and the old 100k-step setup recycled its corpus; corrected training
  is exactly one pass over 100k unique sequences. `3854813` remains the
  checkpoint-curve gate.
- 10:00 CEST freed about `20.66 GiB` from repo-owned `$WORK`: `18.86 GiB`
  from superseded ambiguous-generator tiny runs/evals, obsolete OLMo-2 32B
  adapters, and invalid long-depth BranchProof materializations, plus `1.80
  GiB` from 121 W&B local caches whose Slurm jobs are terminal. Retained all
  active corrected artifacts, current downstream outputs, old large-model
  report outputs, and paused paired-task datasets. Repo usage fell from about
  `88.4 GiB` to `67.7 GiB`.

- 08:14 CEST safely widened tiny checkpoint replacement `3854813_[0-89%3]`
  from A100-80-only to `a40,a100` with generic one-GPU GRES. The 50M--200M
  evaluator fits the A40 memory/runtime envelope; rows `0..2` started
  immediately on A40s with the existing three-row throttle.
- 08:12 CEST post-oversight recovery: shortcut SFT tasks `3850213_3/4`
  were canceled by Slurm before Python startup, leaving no stderr or artifact
  and making eval `3850214`'s `afterok` dependency unsatisfiable. Submitted
  only the missing rows as `3854948_[3-4%2]`; both started immediately on
  A40s. Rewired `3850214` to `afterany:3850213,afterok:3854948`, so all
  successful original rows are retained and evaluation cannot release early.
  Conditioned-50k task `3850109_3` had the same launcher-level cancellation,
  but its staged `afterany` recovery jobs `3850110..3850112` already recover
  that row and require no extra submission.
- 07:14 CEST oversight recovered three blocked paths. Canceled checkpoint eval
  `3850493` after 39 rows deterministically failed because intermediate tiny
  checkpoints lack tokenizer files; added an explicit tokenizer override,
  verified all 90 checkpoints plus 18 final tokenizers, and submitted exact
  replacement `3854813_[0-89%3]`. All `18/18` final tiny eval rows in
  `3850492` passed structural and representative raw review; they show
  shallow valid proofs but depth-10 answer-correct/invalid traces and severe
  depth-50 truncation, with zero joint@`1/4/8` in every size/template
  aggregate. The result remains provisional until the checkpoint curve passes.
  Corrected 32B rows `3850115_0/1` failed on expired Xet download URLs; row 2
  is training normally, targeted recovery `3854837_[0-1%1]` was submitted,
  and held eval `3850123` was rewired to require the recovery plus original
  tasks 2..14.
- 07:14 CEST logic Nanotron upload `3847802_3` passed conversion,
  Transformers-4 RoPE `1000000`, finite-logit, and local/remote parity gates;
  guarded cleanup removed the local full checkpoint, leaving vault quota at
  `781G/1000G` soft and `153k/200k` files. Direct eval `3847804_3`
  completed. Instruction SFT `3847805_3` completed 10,000 steps but failed
  only on a transient Hub-upload 401; retried the complete local adapter
  without training and verified commit
  `3d1e4a751150fffbb26e23e6f759c402bf203b4d`. Canceled stale dependent
  `3847806` and aggregate `3850389`, then submitted replacement instruction
  eval `3854824_[3%1]` and strict six-bundle aggregate `3854847`.
- 07:14 CEST fixed schema-v4 MATH-500 handling of escaped currency `\$`, then
  forced rescore/production audit across the five complete corrected
  downstream bundles; all passed with zero lost stock positives. Raw review
  found direct control/logic/NL BBH/MMLU-Pro next-document marker rates of
  `35.6/12.1%`, `60.0/45.5%`, and `58.0/49.5%`; instruction SFT removes those
  markers but often leaves long repetition. Corrected instruction multi-hop
  smoke `3850354_0` passed its structural gate but failed the sample-clean
  gate on 2Wiki/MuSiQue, so no full grid was submitted. Baseline eval
  `3853284_0..5` remains healthy with `6.5--11.8` hour projections. Full tests
  pass `216 passed, 3 skipped`; report/preprint regeneration remains held for
  complete corrected aggregates. Successor watcher `3854785` was preserved.
  The answer-matcher and current oversight commits remain local: pushes through GitHub SSH port 22 and
  `ssh.github.com:443` each timed out without server output. The report repo
  was unchanged.
- 01:16 CEST BranchProof answer-matcher correction: an independent scan of the
  five complete validity-fixed bundles found that the common scorer credited a
  gold token anywhere inside `<answer>`. Two retained false positives were
  confirmed (`1/640` greedy and `1/4480` sampled), including multi-label
  answers. Because unretained generations prevent exact pass@k rescoring,
  canceled eval/audit/aggregate `3838163/3847756/3847757` and quarantined all
  their outputs under
  `$HPCVAULT/.../quarantine/pre_answer_match_fix_20260714/`; four tracked audit
  JSONs moved to
  `analysis/branchproof_unique_v2_full_grid_audits_pre_answer_match_fix_20260714/`.
  Patched matching to accept only exact answers or single-line natural
  assertions, added an independent retained-answer audit guard, and verified
  `213 passed, 3 skipped`. Submitted unchanged-protocol replacement eval
  `3853284_[0-29%6]`, CPU audits `3853285_[0-29%8]`, and CPU aggregate
  `3853286`; they are capacity/dependency pending. Logic Nanotron recovery
  `3835442_3` completed step 8192 in `19:26:16`; independent audit accepted all
  645 files, exact state/offsets, zero empty files, and proof share
  `15.000057%`. Vault quota is `901G/1000G` soft and `150k/200k` files.
  Upload `3847802_3` is released and capacity-pending. Active
  CPU-only watcher `3850618` preserved scheduled successor `3853210` for 06:47
  CEST because the overall plan remains incomplete.
- 01:22 CEST corrected control/NL consumers started on A100-80GB: native-chat
  adapter retraining `3850351_0/3850352_8` and direct reviewer eval
  `3850385_0/3850386_8`. All four startup preflights resolved
  `rope_theta=1000000`, and both direct jobs validated the intended production
  task suite without a fatal signature. Post-instruction evals remain held on
  the new adapters; no score is accepted before artifact and raw-generation
  audits.
- Repository changes were committed locally. Push via normal GitHub SSH and
  the `ssh.github.com:443` fallback timed out; HTTPS could read the public
  remote but no noninteractive credentials were available for writing. The
  successor should retry the push. The report repo was not changed.
- 01:31 CEST clean answer-fixed BranchProof eval rows `3853284_0/1` started at
  01:28 on A100-SXM4-80GB nodes `a0633/a0832`. Both selected the intended
  seed-3407/3408 train-1-to-5 adapters, used isolated merge roots, and
  initialized vLLM with `max_model_len=16384`; no fatal/OOM signature is
  present. Their first 64-prompt greedy chunks completed in `7.6/7.7s` with
  the intended 7,168-token cap and `</answer>` stop, reaching only 367 tokens.
  The next runtime trigger is the first sampled-chunk projection against the
  20-hour intervention threshold.
- 01:36 CEST corrected direct multi-hop smoke `3850353_0` completed `0:0` in
  `00:04:10`. Its audit accepted RoPE `1000000`, `max_model_len=32768`, all six
  stock/tagged task cells, six sample files, and all 12 rows. Manual inspection
  covered every generation. Tagged prompts found/extracted answers in `6/6`
  without continuing into another question; stock responses emitted a next
  question or assistant preamble in `4/6`. Both stock Hotpot rows began with
  the correct answer but leaked suffixes; tagged exact match was `1/6`, while
  the two 2Wiki and two MuSiQue examples were wrong under both protocols.
  These are protocol diagnostics at `n=2`, not performance evidence.
  Instruction smoke `3850354` remains dependency-held; no full multi-hop grid
  was submitted.

## 2026-07-13

- 20:55 CEST live audit: no new failed, OOM, timed-out, or dependency-stale
  rows in the corrected report matrix. No-repeat tiny rows `3850490_0..2`
  completed exactly 6,250 steps/one epoch over 100k unique examples in
  73--77 minutes, with complete target checkpoints and 78,266,808-byte final
  weights; rows `3..5` backfilled. Validity-fixed baseline eval rows
  `3838163_3/5` completed in `10:14:41/10:33:40`, and row audits
  `3847756_3/5` accepted both 448-prompt, 16-generation, 1,024-retained-row
  bundles without errors. Manual samples show complete in-range proofs and
  expected OOD wrong-branch, repetition, format, and cap failures; fresh
  constants remain intact and no ambiguity/parser regression was found.
  Preliminary train-1-to-10 logic depth-25 correct/citation-free-joint@16 is
  `0.938/0.031` and `0.875/0.219` across the two complete seeds; depth-50 is
  `0.406/0.000` and `0.219/0.031`. These are not report-ready until seed 3408
  and matched NL finish. Logic Nanotron reached step `7271/8192` at `30.9K`
  tokens/s with finite loss. Corrected downstream jobs remain capacity-pending,
  not failed. Added the two accepted row-audit JSONs under
  `analysis/branchproof_unique_v2_full_grid_audits/`.
- 18:57 CEST autonomous oversight refresh: corrected architecture SFT rows
  `3850113_0..2` completed `0:0`; each has its exact step-10000 trainer state,
  a nonempty `73,911,112`-byte final adapter, and no zero-byte/fatal artifact.
  Rows `3..5` backfilled on A40s; corrected shortcut SFT `3850213_0..2`
  started at 18:56--18:59, and no-repeat tiny training `3850490_0..2`
  started at 18:59--19:01. Validity-fixed BranchProof eval `3838163_0..5` remained on
  verified A100-80GB nodes at sampled chunks `98/67/81/106/96/104` of 112.
  The slowest recent-ten-chunk projection is about `16.2` hours total, below
  the 20-hour sharding trigger; no corrected production bundle exists yet.
  Logic Nanotron recovery `3835442_3` reached `6851/8192` at `30.9K` tokens/s
  with finite loss `1.81` and retained the exact step-4096 optimizer,
  scheduler, RNG, sample/token-offset, and 85/15 blend state. No scheduler
  edit, recovery, report regeneration, or scientific claim was triggered.
  Current CPU-only watcher `3850497` preserved its scheduled CPU-only
  successor `3850618` for 00:47 CEST on July 14. The handoff commit was
  created locally; pushes through normal GitHub SSH and the port-443 fallback
  both timed out without output, leaving this repo one commit ahead of
  `origin/main` for the successor to retry.
- 18:04 CEST first no-repeat tiny build `3850394` failed its intended gate:
  record keys were all unique, but full content hashing found only `99,994`
  distinct sequences in 100k rows. Canceled its unsatisfied dependents
  `3850395..3850397`. The builder now materializes a 1% surplus, deduplicates
  independently on complete formal and NL sequence fields, preserves exactly
  10k rows per depth, publishes atomically, and uses no GPU. Submitted build
  `3850488`, train `3850490`, final eval `3850492`, and checkpoint eval
  `3850493`.
- 18:08 CEST build `3850488` completed `0:0` in `00:02:11`: exact depth
  balance, `100,000` distinct formal fingerprints, `100,000` distinct NL
  fingerprints, and remote subset reload at `100,000` rows all passed. It
  released training `3850490`; Slurm currently estimates a 19:54 CEST start.
  Remote regression loads also preserve the existing 50k train and depth-50
  validation configs at their expected sizes. The reconciled full repository
  suite passes `205 passed, 3 skipped`.
- 18:06 CEST replaced still-pending oversight `3850404` with `3850497` at the
  same 18:46 CEST begin time so the stored watcher payload references the
  deduplicated no-repeat tiny chain rather than the canceled first attempt.
- 17:58 CEST stopped tiny correction jobs `3850072..3850078` before their
  first epoch boundary. A sample-budget audit found the nominal 100k run was
  100k optimizer steps at effective batch 16 over 50k rows, which would reuse
  the corpus about 32 times. Added a trainer-side no-reuse guard and a dedicated
  100k-row paired subset build with exact sequence-hash uniqueness checks.
  Submitted replacement build `3850394`, training `3850395` for exactly 6,250
  steps/100k examples, final eval `3850396`, and five-checkpoint eval `3850397`.
  Focused generator/tiny tests pass (`51 passed`); the full suite passed
  `200 passed, 3 skipped` before adding the two guard tests.
- 17:58 CEST replaced pending oversight `3848334` with `3850404` at the same
  18:46 CEST begin time. The old frozen prompt incorrectly called the
  RoPE-invalid bundles accepted; the replacement records the quarantine,
  corrected report-wide arrays, corrected six-way aggregate, and 32k-context
  multi-hop smoke gate. The watcher remains CPU-only.
- 17:56 CEST canceled stale aggregate `3849776`, preserved all four invalid
  control/NL reviewer bundles under `.rope10000_invalid_20260713` suffixes,
  and submitted corrected direct jobs `3850385/3850386`, dependency-held
  instruction jobs `3850387/3850388`, and six-way corrected aggregate
  `3850389`. Slurm estimates both direct jobs can start around 19:53 CEST.
- 17:53 CEST full-suite verification exposed a legacy `hard_v2/hard_v3`
  generator bug for extended predicate names: direct concatenation produced
  malformed atoms such as `P37b`. Replaced those rule renderings with the
  existing predicate-atom renderer, which preserves one-letter syntax and
  emits `P37(b)` when needed. This path is separate from BranchProof-v2 and
  does not invalidate active jobs. Targeted tests pass `48/48`; the complete
  suite passes `200 passed, 3 skipped`.
- 17:53 CEST queue refresh: corrected surface, hybrid, conditioned,
  architecture, and tiny rows are actively training without fatal signatures;
  all seven corrected shortcut corpora completed their build gates. Corrected
  instruction retrains `3850351/3850352` and direct 32k-context multi-hop smoke
  `3850353` remain account-GRES pending; instruction smoke `3850354` remains
  correctly dependency gated.
- 17:41 CEST Nanotron downstream correction: the Transformers-5 converter
  serialized Qwen2.5 RoPE base `1000000` only as `rope_parameters`; the
  Transformers-4.57 downstream environment ignored it and silently resolved
  `rope_theta=10000`. Nanotron training logs/configs remain correct. Patched
  conversion to emit both fields and patched verification to require both plus
  a downstream-env load resolving `1000000` (`9` focused tests pass). Repaired
  the control and corrected-NL Hub `config.json` files in place and verified
  them with Transformers 4.57. Prior control/NL direct/instruction results,
  including MATH symbolic scores, are quarantined pending rerun.
- 17:41 CEST removed only the two local/Hub UltraChat adapters trained with the
  wrong resolved RoPE metadata (about `6.4G` local) and submitted clean
  retraining `3850351` control and `3850352` NL. No model weights or result
  bundles were deleted.
- 17:41 CEST multi-hop smoke audit: jobs `3850097/3850098` completed but all
  six raw generations per mode were degenerate. Logs prove lm-eval left-
  truncated 11.8k--17.3k-token prompts into the 8192 window. Full Qwen
  tokenization of all 200 examples per task found maximum prompt lengths
  `17684` HotpotQA, `17079` 2Wiki, and `17927` MuSiQue; none exceed a 32768
  window with generation allowance. Added stock LongBench short-answer tasks,
  a strict 32768-window audit gate, and corrected smokes `3850353 -> 3850354`.
  Canceled unstarted flawed production/aggregate jobs
  `3850099/3850100/3850207/3850217`.
- 17:43 CEST added a consumer-side RoPE preflight to instruction SFT and both
  downstream evaluators, then canceled/replaced still-pending jobs
  `3850339..3850342` so their stored Slurm payloads include it. Replacement
  jobs are `3850351..3850354`.
- 17:20 CEST submitted corrected three-seed BranchProof report coverage after
  auditing every official/informal-report family. Active arrays cover surface,
  shortcuts, hybrid order, conditioned dual 10k/50k, architecture, batch size,
  32B, and tiny scratch (`3850072..3850123`, `3850212..3850214`). Hybrid
  corrected SFT uses 16384 tokens because targets average about 10k; tiny uses
  4096-token SFT and 16384 context because old 2048 training truncated even
  depth-10 targets. The first submission encountered global home-filesystem
  inode exhaustion; failed Hydra outputs were removed, Hydra output was
  redirected to the vault, and only affected arrays were resubmitted. Active
  rows now progress without fatal signatures.

- 16:00 CEST control direct eval `3847792_0` completed `0:0` in `01:06:47`
  and passed the ten-task/105-file/50,693-row production gate. Control
  instruction eval `3835928_0` completed the full GPU workload but exited
  `1:0` on a post-hoc task-order comparison and one correct MATH response that
  reasoned before stating `x = 11`. Made task validation order-insensitive and
  added a schema-v4 final-explicit-answer fallback only when the first line has
  no answer token, excluding generated next-`Problem:`/`Question:` records.
  All 100 control fallback rows were enumerated; schema v4 accepts 36
  symbolically equivalent explicit final answers. Focused tests pass
  (`24 passed`).
- 16:05 CEST NL post-instruction eval `3834905_8` likewise completed its full
  GPU workload and 50,693-row bundle before the stale in-job gate exited
  `1:0`. Its only schema-v3 rejection was a correct `$x = 11$` followed by a
  malformed repeated decimal suffix. Schema v4 ranks answer-cued delimited
  math above later plain text and adds next-prompt and malformed-suffix
  regressions. All 100 NL fallback rows were enumerated and 35 end in
  symbolically equivalent answers; no A100 rerun is warranted.
- 16:10 CEST final schema-v4 CPU gates `3849774/3849775` completed `0:0`
  and accepted the existing control/NL instruction bundles, so no A100 rerun
  is needed. MATH symbolic is control direct/instruction `79/65` of 500 and
  NL direct/instruction `80/61` of 500, with no stock positive lost. Final
  CPU aggregate `3849776` now waits only on logic evals `3847804/3847806`;
  dependency-dead intermediate aggregates were canceled.
- 12:55 CEST replacement NL instruction SFT `3847662_8` completed `0:0` in
  `01:04:38`, uploaded verified adapter commit
  `cddf739f4b4332e1d9f3d71b825e52c836476679`, and released post-instruction
  eval `3834905_8`. Control post-instruction eval `3835928_0` started at 12:48
  on A100-80GB and initialized the intended merged 8192-context checkpoint;
  control direct `3847792_0` remains healthy. Logic recovery reached
  `5571/8192` with finite loss `1.69`. Corrected BranchProof rows `0..5`
  reached sampled chunks `49/35/39/64/61/64`; a row-local worst-chunk bound
  keeps even the slowest projected total below 14 hours, with no production
  artifact or fatal signature yet. Verified CPU-only successor `3848334` is
  queued for 18:46 CEST with no GRES.
- 12:54 CEST authenticated per-repository Hub storage, after the NL
  instruction adapter upload, is `63.782G`: `35.430G` models and `28.352G`
  datasets. Pending logic projects to `79.025G`; logic plus one more full
  checkpoint projects to `94.268G`, but two project to `109.511G`. No repo was
  deleted; guarded broader-grid rotation remains mandatory. Updated
  `analysis/hf_storage_cleanup_2026-07-13.json`.
- 12:31 CEST completed raw-generation audit of NL-direct downstream run
  `3834904_8`. Correct samples show coherent GSM8K/BBH/MMLU-Pro reasoning;
  failures include omitted ordering constraints, false implication reversals,
  and repetition to the cap. BBH/MMLU-Pro invalid extraction is `9.1%/20.5%`
  and generated next-document preamble incidence is `22.9%/3.7%`; the marker
  appears in none of the prompts. Added condition-blind generation diagnostics
  to the six-run aggregate and documented the provisional result in
  `docs/nanotron_nl_direct_generation_audit_2026-07-13.md`. The qualitative
  index now preserves raw response heads/tails and prompt tails instead of
  filtered answers alone. Focused tests pass (`20 passed`).
- 12:20 CEST reconciled Hugging Face storage through authenticated
  per-repository `usedStorage` after the control instruction adapter upload.
  The live account uses `63.610G`: `35.257G` across 14 model repos and
  `28.352G` across 49 dataset repos. Pending logic projects to `78.852G`; one
  more full checkpoint fits at `94.095G`, whereas two reach `109.338G`.
  Preserved the latest cross-project adapters because all together they free
  only `4.119G` and deleting them would not avoid guarded checkpoint rotation.
  No further model was deleted.
- 12:14 CEST replaced format-confounded MATH-500 aggregation with a tested
  answer-prefix symbolic sidecar while retaining stock exact as diagnostic.
  Whole-response symbolic matching was rejected because it credits later
  prompt leakage and shared scalars from wrong equations/tuples. Focused tests
  pass (`18 passed`), the venv dependency check is clean, and the completed NL
  direct bundle is accepted at post-hoc `80/500 = 0.160` versus stock
  `14/500 = 0.028`, with no stock positives lost. The live audit generates the
  sidecar for old stored Slurm payloads, so no GPU rerun/resubmission is needed.
  Detailed audit: `docs/nanotron_math500_scoring_audit_2026-07-13.md`.
- 12:14 CEST live refresh: control instruction SFT `3847661_0` completed and
  released `3835928`; replacement control direct eval `3847792_0` started on
  A100-80GB. NL instruction SFT reached about `4176/10000`, logic midtraining
  reached `5431/8192`, and corrected BranchProof rows `0..5` reached sampled
  chunks `39/30/33/55/53/55` of 112. No fatal signature or completed
  BranchProof production bundle appeared.
- 11:58 CEST live-path refresh: corrected BranchProof eval rows `0..5`
  reached sampled chunks `34/27/30/51/49/51` of 112. Logic midtraining reached
  `5381/8192` at `30.8K` tokens/s with finite loss `1.71`; control/NL native
  instruction SFTs reached about `8660/1632` of 10000 with finite losses.
  NL direct eval `3834904_8` completed `0:0` in `01:06:23` and its structural
  production audit accepts all ten task groups and 50,693 unique scored docs.
  Its strict MATH-500 metric remains provisional because sample inspection
  found mathematically equivalent answers rejected when explanation follows
  the answer; use a tested post-hoc final-answer scorer before aggregation.
- 11:57 CEST archived nine superseded `autoformalization-*` adapters before
  deleting their Hub repositories. Exact commit snapshots, 117-file
  size/SHA-256 manifests, and a restoration index are under
  `$HPCVAULT/hf_model_archives/2026-07-13_autoformalization_superseded/`.
  Preserved the latest adapter for each task, the strict MMLU variant, all
  datasets, all active Qwen models, and all reconstructing LoRAs. API checks
  verified every deletion. Retained Hub LFS fell `69.497G -> 63.487G`,
  reclaiming `6.010G` including history; the pending logic upload now projects
  `78.730G`, leaving `21.270G`. The broader grid still requires checkpoint
  rotation.
- 11:41 CEST active-path audit: corrected BranchProof rows `0..5` all
  finished greedy generation and reached sampled chunks
  `28/18/27/43/42/44` of `112`; no production bundle is complete. Logic
  midtraining reached `5311/8192` at `30.9K` tokens/s with finite loss `1.69`.
  Control instruction SFT reached `5227/10000`, while NL direct evaluation
  reached `67407/109837` requests in its current task stage. Fatal-log scans
  were clean and all dependencies remain correct. Fixed the stale dashboard
  reference from superseded aggregate `3847793` to strict gate `3847807`.
- 11:40 CEST closed the BranchProof tokenizer-warning audit. On 704 actual
  prompt/target texts across logic/NL and depths 1--50, the base training
  tokenizer and merged-eval default tokenizer matched exactly (`704/704`).
  Setting `fix_mistral_regex=True` would change `640/704` texts and create a
  train/eval mismatch, so the running corrected protocol remains unchanged.
  Recorded hashes and counts in
  `analysis/branchproof_tokenizer_consistency_2026-07-13.json` and the
  evaluator audit.
- 11:34 CEST re-audited all `71` Hugging Face repositories through the LFS
  API. Retained usage is still `69.497G` (`41.094G` models and `28.403G`
  datasets), projecting to `84.740G` after the pending logic checkpoint.
  Preserved `17` private `autoformalization-*` adapters (`10.129G`) because
  that other project intentionally removed its local checkpoints after upload;
  deleting all of them would still not fit another `15.243G` full model.
  Chose guarded upload/evaluate/audit/delete rotation for the broader mixture
  grid and recorded the exact category totals in
  `analysis/hf_storage_cleanup_2026-07-13.json`.
- 11:22 CEST reconciled Hugging Face head sizes with retained LFS history.
  Retained storage was `86.191G`, so the pending 15.243 GB logic checkpoint
  would have exceeded 100 GB despite current heads totaling only `62.021G`.
  Permanently removed 34 historical objects (`16.694G`) from
  `imagenet-100-LT-balanced` only after proving they were unreachable from all
  live branches, tags, and conversion refs. Verified the live `main` and
  `parquet` trees path/size/SHA-identical afterward. No additional repo or
  model was deleted. Retained LFS is now `69.497G`, projected `84.740G` after
  logic; the broader mixture grid needs checkpoint rotation or external
  archival. Updated `analysis/hf_storage_cleanup_2026-07-13.json`.
- 11:12 CEST audited stored Slurm scripts before logic completion and found
  unstarted jobs `3831123/3834908/3831125/3834909` predated the converter,
  downloader, and resume fixes. Submitted and verified repaired chain
  `3847802 -> 3847804/3847805 -> 3847806`, then canceled only the stale jobs.
  CPU-only six-eval aggregate is now `3847807`. Stored-payload-verified watcher
  `3847808` replaces `3847795` at the same 12:46 start and self-schedules every
  six hours before invoking Codex.
- 11:06 CEST control direct eval `3835927_0` failed before model load on a
  transient `hf_transfer` 403. Forced standard resumable HTTP in both remote
  model wrappers and submitted A100-80 replacement `3847792`. Replaced
  dependency-unsatisfiable/GPU-requesting aggregate `3836159` with CPU-only
  `3847793`; focused tests pass (`17 passed`). Control instruction SFT
  `3847661_0` is running normally. Stored-payload-verified watcher `3847795`
  replaces `3847769` at the same 12:46 start.
- 10:59 CEST confirmed corrected BranchProof eval `3838163` is explicitly
  constrained to `a100_80` and every active row is on an 80 GB A100. Replaced
  still-pending GPU-requesting audits `3838164 -> 3838165` with CPU-only
  `3847756 -> 3847757`, preserving row-wise and all-row dependencies. Updated
  aggregation to show three-seed std in the primary table and depth bands;
  focused tests pass (`8 passed`).
- 10:59 CEST logic midtraining was healthy at `5171/8192`, while NL direct
  reviewer eval reached `3571/20362` prompts and control direct eval started;
  all are on A100-80GB. Replaced queued watcher `3847703` with CPU-only
  `3847769` at the same 12:46 start so its stored payload tracks the new audit
  IDs.
- 10:53 CEST made pending native-chat instruction SFT restart-safe. The live
  wrapper now auto-resumes the latest Trainer checkpoint; all three output
  roots are currently absent, so clean first launches are unchanged. Focused
  tests and an exact repaired-checkpoint dry run pass (`7 passed`).
- 10:50 CEST removed unnecessary GPU allocation from the six-hour oversight
  wrapper after CPU-only `a100mig` probe `3847702` completed `0:0` during full
  account GPU utilization. Verified replacement watcher `3847703` is queued
  for 12:46 CEST with four CPUs, 30 GB RAM, and no GRES; only then canceled
  GPU-requesting predecessor `3847667`.
- 10:46 CEST exact UltraChat dry runs against the repaired control/NL remote
  checkpoints retained all sampled train/eval rows, rendered native Qwen chat,
  and supervised assistant tokens only. Focused downstream gates pass
  (`12 passed`). NL direct eval completed full-model and KV-cache startup and
  began full-suite context construction.
- 10:41 CEST NL direct eval `3834904_8` started on A100-80GB and passed the
  repaired-model startup boundary: all ten tasks selected, Qwen architecture
  resolved, and vLLM initialized without the prior tokenizer exception.
- 10:40 CEST replaced queued watcher `3846896` with refreshed watcher
  `3847667` at the same 12:46 CEST start time. The stored payload was verified
  to contain the repaired uploads, tokenizer fix, replacement instruction
  jobs, HF cleanup, current BranchProof rows, and unchanged end-to-end gates
  before the stale watcher was canceled.
- 10:36 CEST repaired the Nanotron-to-HF path after uploads `3831119/3831113`
  failed on missing `WORLD_SIZE`. The wrapper now launches conversion with
  single-rank `torchrun` and verifies with the downstream environment;
  focused checks pass (`5 passed`). Replacements `3847569/3847570` completed
  with four shards, finite CUDA logits `[1,152064]`, zero remote omissions,
  and guarded cleanup. Direct evals `3835927/3834904` are released.
- 10:36 CEST normalized the two uploaded Qwen tokenizer configs from the
  Transformers-5 `extra_special_tokens` field to the Transformers-4-compatible
  `additional_special_tokens` field. Fresh 4.57.3 and 5.12.1 loads preserve
  all 13 special-token IDs and native-chat rendering. Failed/canceled
  instruction parents `3831115/3831121` were replaced by
  `3847662/3847661`; existing instruction evals and aggregate were rewired.
- 10:36 CEST HF storage cleanup deleted only three superseded merged OLMo SFT
  seed repos (`43.818G`). The retained LoRA repos plus public base can
  reconstruct them; all datasets and unrelated/new models remain. Model
  storage fell `83.942G -> 40.125G`; total model+dataset storage is now about
  `62.021G`, projected `77.264G` after logic upload. Audit:
  `analysis/hf_storage_cleanup_2026-07-13.json`.
- 10:14 CEST corrected BranchProof eval rows `3838163_0..5` started on six
  verified A100-SXM4-80GB allocations; rows `6..29` are array-throttle
  pending. All active rows selected the intended corrected adapters and
  reached merge/model startup without a fatal/OOM/quota signature. Row 0
  initialized the 16,384-token vLLM engine and completed its first greedy
  64-prompt chunk at the unchanged 7,168-token cap. No production artifact is
  complete, so runtime, row-audit, and scientific gates remain open.
- 10:10 CEST NL recovery `3835443_8` completed `0:0` at step 8192.
  Independent verifier artifact
  `analysis/nanotron_checkpoint_audits/nl_exact_step8192.json` accepts 645
  files, no empty files, exact `8192/1048576/4294967296` offsets, exact
  `3650719744 + 644247552` normal/NL token accounting, complete
  optimizer/scheduler/RNG groups, and four equal 22.85 GB optimizer shards.
  Upload `3831113` is released but account-GRES pending; no cleanup has run.
- 10:10 CEST logic recovery `3835442_3` remained healthy at `4991/8192`,
  `30.9K` tokens/s, finite loss `1.76`, and a projected completion near 01:12
  CEST on July 14. Control/NL uploads and all corrected BranchProof eval rows
  are A100-80-only and account-GRES pending; no production eval artifacts are
  available. Watcher `3845763` completed `0:0`, and successor `3846896`
  remains queued for 12:46 CEST. Total vault usage is `1268G` of the
  `2097.2G` hard quota.
- 06:58 CEST committed the checkpoint/handoff transition locally.
  Pushes through both `github.com:22` and `ssh.github.com:443` timed out
  without transferring; `gh` is not installed. The main repository remains
  two commits ahead of `origin/main`, while the report repository is clean and
  synchronized.
- 06:53 CEST control recovery `3835438_0` completed `0:0` after saving step
  8192. Independent verifier artifact
  `analysis/nanotron_checkpoint_audits/control_step8192.json` accepts 645
  files, no empty files, exact `8192/1048576/4294967296` offsets, complete
  optimizer/scheduler/RNG groups, and four equal 22.85 GB optimizer shards.
  Upload `3831119` is released but account-GRES pending; no guarded cleanup
  has run.
- 06:53 CEST logic recovery `3835442_3` passed its first-resume audit after
  starting at 05:48: it restored optimizer/LR-scheduler state, offsets
  `4096/524288/2147483648`, and the exact `1825357824 + 322125824`
  normal/logic token split, then logged finite iteration-4101 loss `1.71` and
  advanced to `4301/8192` at `30.9K` tokens/s. NL `3835443_8` remained healthy
  at `7971/8192` with about one hour left. Total vault use is `1211.8G` of a
  `2097.2G` hard quota; the three matched roots occupy `795G`.
- 06:53 CEST corrected BranchProof eval `3838163`, control upload `3831119`,
  and their downstream gates remained pending without new production
  artifacts; no scientific analysis or report refresh was triggered. Watcher
  `3845763` scheduled successor `3846896` for 12:46 CEST before starting. The
  full plan remains incomplete, so the successor was preserved.
- 00:59 CEST committed the scoped oversight handoff locally. Pushes through
  both `github.com:22` and `ssh.github.com:443` timed out without transferring,
  so the main repository remains one commit ahead of `origin/main`. The report
  repository is clean and synchronized.
- 00:55 CEST watcher `3843796` verified control/NL recoveries healthy at
  iterations `7161/6711` of `8192`, about `30.9K` tokens/s, finite losses
  `2.01/1.63`, and no fatal/OOM/quota signature. Only accepted step-4096
  checkpoint trees exist and project usage remains `871G`; logic recovery
  `3835442_3` and corrected BranchProof eval `3838163` remain dependency-free,
  A100-80-only, and account-GRES pending without production outputs. A
  temporary BranchProof row-0 start estimate disappeared on the next scheduler
  cycle and was not treated as a launch. Idle `a0903/a0905` are A100-40 nodes,
  so no compatible widening exists. The watcher scheduled successor `3845763`
  for 06:46 CEST before starting and its stored payload passed the corrected
  critical-path/protocol/stop-condition check; the plan remains incomplete.

## 2026-07-12

- 21:12 CEST control/NL recoveries remained healthy at iterations
  `6371/5921` of `8192`, both near `30.9K` tokens/s with finite losses and no
  fatal/OOM/quota signature. Logic remains account-GRES pending with a 07:00
  estimate; BranchProof eval remains pending without production outputs.
  Watcher `3842454` completed `0:0`; successor `3843796` remains queued for
  00:45. Pushed scoped handoff commit `cf5162e` successfully from the login
  node, superseding the transient watcher push failure.
- 18:54 CEST committed the scoped oversight update locally. Pushes to
  both `github.com:22` and `ssh.github.com:443` timed out without transferring,
  so the main repository remains one commit ahead of `origin/main`. The report
  repository is clean and synchronized.
- 18:50 CEST corrected NL recovery `3835443_8` passed the no-replay gate on
  A100-80 node `a0532`: optimizer/LR-scheduler loading was enabled, sampler
  offsets restored at `4096/524288/2147483648`, and metadata retained the
  exact `1825357824 + 322125824` normal/NL token split. It advanced to
  `5421/8192` at `30.9K` tokens/s with finite loss `1.74`; control
  `3835438_0` reached `5871/8192` with finite loss `2.11`. Logic recovery
  `3835442_3` and corrected BranchProof eval `3838163` remain dependency-free,
  A100-80-only, and blocked solely by `AssocGrpGRES`; no production eval or
  step-8192 artifact exists, so no scheduler/protocol edit or analysis was
  triggered. Project usage remains `871G` by `du`.
- 18:46 CEST watcher `3841073` had failed at 12:45 after 13 seconds because
  Codex reported a usage limit, but its wrapper had already scheduled current
  pass `3842454`. The current pass scheduled successor `3843796` for 00:45
  CEST on 2026-07-13. The full plan remains incomplete, so the successor stays
  queued.
- 10:45 CEST watcher `3840018` completed `0:0` in `00:09:18`; successor
  `3841073` remains queued for 12:45 with the corrected payload. Retried the
  scoped handoff from the login node and pushed `19df2c4` successfully to
  `origin/main`, superseding the transient SSH-timeout note below. The report
  repo remains clean.
- 10:39 CEST watcher `3840018` confirmed control recovery `3835438_0`
  advanced cleanly to iteration `4141/8192` at `30.8K` tokens/s with finite
  loss `1.98`; logic/NL recoveries remain account-GRES pending near 12:27 and
  all corrected BranchProof eval rows remain dependency-free, A100-80-only,
  and account-GRES pending with no production outputs. The watcher scheduled
  successor `3841073` before starting. Advanced only that successor from
  16:35 to 12:45 CEST so the next pass can verify the projected logic/NL
  first-resume transition; the end-to-end plan remains incomplete. Committed
  the scoped handoff locally; pushes to `github.com:22` and
  `ssh.github.com:443` both timed out, so the main repository remains one
  commit ahead of `origin/main`.
- 10:29 CEST control Nanotron recovery `3835438_0` started on full A100-80
  node `a0531` and passed the resume gate. It loaded run checkpoint 4096 with
  optimizer/LR scheduler enabled, restored `524288` samples and
  `2147483648` consumed tokens, and logged iteration `4101/8192` at `30.9K`
  tokens/s with finite loss `2.07`. Logic/NL recoveries remain account-GRES
  pending near 12:27; BranchProof eval remains account-GRES pending.
- 09:33 CEST watcher `3839693` completed `0:0` in `00:14:19`; successor
  `3840018` remains scheduled for 10:35 with a verified corrected payload.
  Retried the two pending main-repo commits from the login node and pushed
  `cb70aa6` plus `18f79e5` successfully to `origin/main`, superseding the
  transient SSH-timeout note below. The report repo remains clean.
- 09:29 CEST committed the scoped SFT/scheduler handoff locally. Direct GitHub
  SSH produced no response within 30 seconds and the port-443 fallback timed
  out connecting to `ssh.github.com`; this repository remains two commits
  ahead of `origin/main`. The report repository has no changes and is already
  synchronized.
- 09:29 CEST advanced queued successor watcher `3840018` from 15:16 to 10:35
  CEST, shortly after Slurm's common 10:23 estimate for Nanotron recoveries
  `3835438_0/3835442_3/3835443_8`. This preserves the recorded successor and
  closes the otherwise five-hour first-resume verification gap; its stored
  payload was already verified to name the corrected critical paths.
- 09:21 CEST corrected BranchProof SFT `3829072` completed at `30/30` after
  NL rows `26..29` finished `0:0` at 03:59/07:40/07:43/08:16. All 30 exact
  final adapters have nonempty configs and no zero-byte files; the four newest
  logs have no fatal/OOM/quota signature. Slurm resolved their remaining
  row-wise dependencies without a manual edit, leaving every validity-fixed
  eval task `3838163_[0-29%6]` dependency-free, A100-80-only, and pending on
  account GRES. The corrected output root still contains only four suffixed
  pilot/qualitative files, so audits `3838164` and aggregate `3838165` remain
  closed. Project usage is `871G` by `du`; the latest quota epilogue reports
  `996.7G`. Watcher `3839693` scheduled successor `3840018` for 15:16; its
  stored payload names the corrected critical paths, and the successor remains
  queued because the end-to-end plan is incomplete.
- 03:19 CEST NL SFT rows `3829072_24/25` completed `0:0` at 01:55/03:15.
  Their exact train-1-to-20 seed-3407/3408 final adapters were present with
  nonempty configs and no zero-byte files, so only stale child dependencies
  `3838163_24/25` were cleared. The corrected BranchProof grid is now `26/30`
  finals and 26 scheduler-eligible A100-80 eval rows. Rows `26..29` remain
  running and gated near steps `9209/6493/6469/6046`; the corrected output root
  still has no production JSON/sample bundle. Project usage is `866G`.
  Watcher `3839191` scheduled successor `3839693`, which remains queued because
  the end-to-end plan is incomplete. A scoped handoff commit was created
  locally; pushes to `github.com:22` and `ssh.github.com:443` both
  timed out, so the repository remains one commit ahead of `origin/main`.
- 01:06 CEST logic SFT rows `3829072_13/14` completed `0:0`, each wrote the
  expected final adapter, and corresponding eval tasks `3838163_13/14` had
  only their stale dependencies cleared after verification. Corrected SFT
  finals and scheduler-eligible A100-80 evals are now `24/30`. Active NL rows
  `24..29` remain healthy around steps `9076/7523/6701/4695/4685/4272` and
  retain their one-to-one dependencies.
- 00:48 CEST inspected the queued Slurm batch payload for oversight job
  `3837467` and found it still named canceled pre-fix BranchProof jobs because
  Slurm had snapshotted the earlier script. Submitted replacement watcher
  `3839191` for the identical `03:15:38` begin time with hop `2/120`, verified
  its stored payload names eval `3838163` and the final-adapter-gated stale
  dependency rule, then canceled `3837467`. Six-hour continuity is preserved.
- 00:46 CEST refreshed Nanotron peak-storage accounting before the three
  resumes. The project tree is `858G` and each accepted step-4096 checkpoint is
  `199G`. Using the larger recent Slurm-epilogue usage (`1072.3G`) plus three
  final checkpoints projects about `1.67T`, roughly `428G` below the
  `2097.2G` hard limit. The soft warning threshold will be crossed, but hard
  space and file counts cover final checkpoint and HF staging needs. Preserve
  step 4096 until step 8192 verifies; cleanup remains post-upload and guarded.
- 00:41 CEST corrected a late-submission `aftercorr` scheduling issue without
  weakening the SFT gate. Parent rows `0..12/15..23` were `COMPLETED 0:0` and
  all 22 corresponding final adapters existed, but every replacement eval task
  still showed the wildcard dependency unfulfilled. Cleared dependencies only
  for those 22 eval tasks; they are now scheduler-eligible and remain pinned to
  `a100_80`. Active SFT rows `13/14/24..29` retain their dependencies. The
  watcher now applies the same parent-state plus final-adapter gate before
  releasing any remaining stale child. The three Nanotron recoveries keep the
  same approximate 05:14 start estimate.
- 00:35 CEST reran the complete focused BranchProof validity and Nanotron
  checkpoint-verifier selection through the preferred project environment;
  all 26 tests passed in `0.34s`, closing the earlier transient vault-I/O
  verification gap. Active BranchProof SFT rows `13/14/24..29` were all
  advancing with fresh logs and no fatal/OOM/quota signature; the corrected
  eval chain remains A100-80-only and dependency pending.
- 00:14 CEST first full corrected BranchProof row `3834582_0` completed on
  A100-80GB in `11:18:40`, then failed qualitative validity review: 14/896
  retained sampled traces had `citation_free_valid=1` alongside premise-parse
  errors. `ProofAnalyzer.ok` ignored malformed premises. Strict and
  citation-free checks now require all premises to parse; the row audit rejects
  validity/error, invalid-line, and line-fraction contradictions. Twenty-six
  focused tests pass. Canceled old eval/audit/aggregate
  `3834582/3834706/3835779`, quarantined pre-fix row 0, and submitted clean
  A100-80GB replacement chain `3838163 -> 3838164 -> 3838165`. Rows 1/2 were
  stopped near `09:57`; rows 3--29 had not started.
- 00:01 CEST accepted both corrected Nanotron step-4096 checkpoints before
  canceling parents `3830927_3/3831111_8`. Each has 625 model files, four equal
  `22,848,937,060`-byte optimizer shards, four scheduler files, eight RNG
  files, no empty files, offsets `4096/524288/2147483648`, and exact
  `1,825,357,824 + 322,125,824` normal/proof token accounting. Recoveries
  `3835442_3/3835443_8` and control `3835438_0` are account-GRES pending;
  project usage is about `912G`.

## 2026-07-11

- 23:24 CEST live audit found 22/30 corrected BranchProof final adapters:
  logic rows `0..12` and NL rows `15..23`; active rows are logic `13/14` and
  NL `24..29`. Eval rows `3834582_0/1/2` reached sampled chunks `110/70/83`
  of `112` after about `10.7/9.1/9.1` hours without a fatal signature. Slurm
  reports `Features=a100_80` on the full eval array and all six downstream
  Nanotron evals, and the active eval nodes expose the same feature. Corrected
  logic/NL Nanotron runs remain matched near steps `3961/3981` and
  `2.08B/2.09B` tokens. Their step-4096 checkpoints are imminent; validate the
  complete checkpoint trees before ending the parents and releasing their
  `afterany` recoveries, avoiding roughly five hours of uncheckpointed replay.
- 21:20 CEST plan-driven oversight found no recovery trigger but recorded
  material progress. All 30 corrected BranchProof SFT rows have started and 19
  final adapters are present; active A100-80 eval rows `3834582_0/1/2` reached
  sampled chunks `96/59/69` of `112` after about `8.6/7.0/7.0` hours. Deep
  chunks are slower and often hit the shared `7168` cap, but no row approaches
  the 20-hour sharding trigger and no production JSON/sample has finalized, so
  the protocol remains unchanged. Corrected logic/NL Nanotron jobs
  `3830927_3/3831111_8` are matched near steps `3521/3531`, `1.85B` tokens,
  and `30.9--31.2K` tokens/s with no step-4096 checkpoint yet. Control recovery
  `3835438_0` remains account-GRES pending with a 05:14 estimate; project usage
  is `536G`, leaving soft-quota headroom for both expected proof checkpoints.
  Watcher `3835433` scheduled successor `3837467` before this pass. No failure,
  scheduler/partition edit, resubmission, cancellation, report regeneration,
  or conditional science launch was justified.
- 17:22 CEST made Nanotron conversion/upload cleanup fail-closed. Previously,
  `upload_folder` success immediately allowed deletion of the approximately
  `199G` source checkpoint without an independent converted-model load test.
  The converter now rejects unmapped HF parameters; the upload wrapper now
  requires a complete safetensors shard manifest, full CUDA reload, finite
  forward logits, and remote-file manifest parity before cleanup. Verification
  writes `hf_verify_step8192.json` in each run root, and any failure preserves
  both staging and Nanotron checkpoints. Ten focused conversion/downstream
  tests, `py_compile`, shell syntax, and `git diff --check` pass. Pending uploads
  `3831119/3831123/3831113` use current scripts, so no resubmission was needed.
- 17:16 CEST completed the paired-statistics export required for the corrected
  claim decision. The aggregator previously wrote per-seed deltas and plotted
  mean/std only for OOD joint@16; it now writes
  `paired_delta_summary.csv` for all greedy and pass@`1/2/4/8/16`
  correctness/joint deltas and `paired_delta_primary.md` for the primary
  correctness readout. Removed silent `None -> 0` delta fallback, so absent
  greedy evidence in historical compatibility data remains `NaN`/`N/A`.
  Eighteen focused tests pass; a 30-row smoke emits five train-range groups,
  each with three paired seeds.
- 17:12 CEST closed a row-audit coverage gap before any production eval
  finalized. The strict final aggregator already required translated NL
  validity/joint metrics, but `audit_branchproof_unique_v2_pilot_eval.py` did
  not require them per row and did not reject retained samples missing
  modality-relevant validity fields. The row gate now requires NL translation
  parse/citation-free-valid/joint metrics and finite unit-valued formal/NL
  sample fields. Seventeen focused tests pass, including missing-NL-metric and
  missing-sample-field regressions; the real corrected pilot is re-accepted
  with 2,155 metrics and 128 samples. Pending audit array `3834706` uses the
  current script at runtime, so no job resubmission was needed.
- 17:07 CEST audited the `AssocGrpGRES` tradeoff rather than trusting the
  projected control start time. This repo currently uses 31 GPUs: 12 corrected
  BranchProof SFT A40s, 16 A100s for logic/NL Nanotron, and three A100-80GB
  BranchProof evals. Pending corrected eval rows have slightly higher scheduler
  priority than control recovery `3835438`. Making room for its eight GPUs now
  would require pausing or canceling several active P0 BranchProof jobs, so the
  plan-order-preserving action is no scheduler edit; control remains queued.
- 17:03 CEST expanded the active audit beyond array summaries. The 13
  BranchProof final adapters exactly match the 13 completed SFT rows; no
  production eval JSON exists yet, and all three A100-80GB eval logs continue
  advancing without fatal signatures. Logic/NL Nanotron reached steps
  `2601/2611` at `30.9/31.2K` tokens/s with no checkpoint yet, as expected
  before step 4096. Project vault usage is `522G`; adding both approximately
  `199G` step-4096 checkpoints remains below the documented soft quota.
  Control recovery `3835438` now shows `AssocGrpGRES`, not a broken dependency:
  the account's GPU allocation is saturated. Its 17:59 scheduler estimate is
  provisional and no safe resource edit is warranted.
- 16:58 CEST verified every corrected BranchProof and Nanotron production eval
  job still requests `Partition=a100` and `Features=a100_80`. BranchProof rows
  `0/1/2` had completed sampled chunks `64/33/38` of `112`, respectively,
  without a fatal signature. The control recovery scheduler estimate moved
  forward from July 12 to 17:59 CEST today; no dependency edit was needed.
- 16:52 CEST closed the Nanotron downstream comparison gap before eval release.
  Pending control evals `3834906/3834907` omitted the corrected unified output
  root; canceled them untouched and submitted otherwise identical A100-80GB
  replacements `3835927/3835928` behind upload/instruction parents
  `3831119/3831121`. Added strict analyzer and dependency job `3836159` behind
  all six control/logic/NL direct/instruction evals. It re-runs production
  audits, exports per-task stderr and control/instruction deltas, computes four
  predeclared macros, and indexes fixed-filter correct/incorrect samples.
  Eight focused tests, `py_compile`, shell syntax, Slurm test submission, and a
  real smoke-schema exercise pass (`96` task rows, `24` macro rows, `30`
  qualitative selections). The pilot has one training run per condition, so
  macro values are not presented as training-seed means.
- 16:43 CEST live BranchProof transition: SFT rows `16/17` completed cleanly,
  rows `23/24` started, and the active/pending split is now completed
  `0..8,12,15..17`, running `9/10/11/13/14/18..24`, and throttle-pending
  `25..29`. Eval `0/1/2` remains active on A100-80GB; row `3` is the next
  priority-eligible eval with an estimate near 21:35. No fatal signature or
  failed row appeared. Vault usage is `522G`, still below the soft quota.
- 16:39 CEST aligned corrected BranchProof aggregation with the claim-decision
  protocol. The aggregator previously validated greedy and all pass@k cells
  but exported mostly pass@16 summaries. It now writes per-run, grouped
  mean/std, depth, and paired-delta fields for greedy correctness/validity and
  pass@`1/2/4/8/16` correctness/validity/joint, plus separate primary
  greedy/pass@1 and train-1-to-25 sampling-efficiency figures. Existing
  pass@16 tables/plots remain. Focused pytest passes (`4 passed`), strict
  checks now include band-level metrics, and a full 30-row compatibility smoke
  generated all CSV/Markdown/PDF/PNG outputs. Visual inspection passed; an
  apparent black rectangle in one PNG was the viewer's RGBA handling and the
  same pixels/flattened RGB and PDF are normal. No live eval job changed.
- 16:18 CEST closed the full-grid qualitative-coverage gap before results are
  available. Added `audit_branchproof_unique_v2_qualitative_grid.py`, which
  requires the complete 30-row sample grid and exact per-depth retention, then
  indexes shallow/train-edge/first-OOD/depth-50 correct, incorrect, valid,
  invalid, and cap-chunk examples across modalities, train ranges, and seeds.
  It emits JSON/Markdown review supplements after the strict metric gate.
  The 13-test focused pytest suite passed before the final field-validation
  tightening; all three qualitative test functions pass afterward under a
  direct system-Python invocation. `py_compile`, Slurm syntax, and
  `git diff --check` also pass. A repeated vault-venv pytest startup later hit
  a 60-second filesystem-I/O timeout before collection. Canceled untouched old
  aggregation `3834707`, whose Slurm-spooled
  script predated this gate, and submitted replacement `3835779` with the same
  `afterok:3834706` dependency. No training, eval, or row-audit job changed.
- 16:05 CEST first production BranchProof timing is safely below the sharding
  trigger. A100-80 row `3834582_0` completed all seven greedy chunks and
  sampled `55/112` after `3:19`. Completed sampled generation totals `2.05h`;
  charging all 57 unfinished chunks the worst observed `350.2s` gives `7.60h`
  sampled total and roughly nine hours including greedy/setup/scoring. Keep
  the full protocol unchanged. Logic/NL Nanotron remain healthy near steps
  `2391/2411`. Control recovery `3835438_0` currently estimates 2026-07-12
  12:00; every compatible 8xA100-80GB node is allocated or mixed, so no safe
  widening is available. Successor watcher `3835433` remains pending for
  21:15.
- 15:30 CEST control recovery `3828946_0` failed before its first resumed
  optimizer step because W&B's local service did not publish a port file on
  node `a0831`. Logs verified the intended step-4096 checkpoint, optimizer/LR
  scheduler load, `524288` consumed samples, and `2147483648` consumed normal
  tokens; the only checkpoint remains complete and unchanged at about `199G`.
  Submitted W&B-disabled replacement `3835438_0` excluding `a0831` and rewired
  upload `3831119`. Proactively replaced untouched logic/NL recoveries
  `3830928_3/3831112_8` with W&B-disabled `3835442_3/3835443_8`, preserving
  parents, isolated run roots, corrected corpus paths, and checkpoint interval;
  rewired uploads `3831123/3831113` and canceled only the superseded recovery
  rows. Logic/NL live training remained healthy near steps `2200/2210` and
  `31K` tokens/s. BranchProof SFT completed rows `0..5`, `12`, and `15`; eval
  `3834582_0/1/2` started on A100-80GB. Row 0 reached sampled chunk `48/112`
  after about `2:37`, projecting below walltime, so the full protocol remains
  unchanged. Recorded oversight successor `3835433` remains pending.
- 11:35 CEST added a strict downstream production artifact gate. It verifies
  the exact ten-task command, all required task/group results, all 105 leaf
  sample files, full unique-document coverage, finite primary metrics, no
  production limit, and direct versus Qwen-chat rendering. Existing results
  are skippable only after passing the same audit. Five focused tests pass;
  real final-smoke direct/chat bundles both pass with 105 leaf tasks/files and
  106 filter-expanded rows. Replaced untouched evals with audited A100-80GB
  rows: NL `3834904/3834905`, control `3834906/3834907`, and logic
  `3834908/3834909`. Slurm rejected reattaching completed smoke `3834836` as a
  new dependency, so the replacements wait on their live model parents while
  embedding the already-accepted suite and audit. Replaced queued watcher
  `3834848` with current-ID watcher `3834911` at the same 15:15 CEST start.
- 11:25 CEST final reviewer smoke `3834836` completed in `00:08:50`. Direct
  and native-chat branches each wrote one aggregate result containing all ten
  required task/group keys plus 105 nonempty sample files. The command and raw
  prompt arguments verify no chat wrapper in direct mode and Qwen
  system/user/assistant rendering in chat mode. The downstream suite gate is
  accepted; production evals now wait only on their model branches.
- 11:20 CEST replaced six untouched production eval jobs that still carried a
  stale task environment. New NL `3834842/3834843`, control
  `3834844/3834845`, and logic `3834846/3834847` evals wait on both final
  smoke `3834836` and their model/instruction parent. Slurm confirms every row
  requests partition `a100`, feature `a100_80`, one A100 GPU, and 240 GB host
  RAM. Replaced queued watcher `3834736` with current-ID watcher `3834848` at
  the same 15:15 CEST start.
- 11:15 CEST diagnostic smoke `3834738` completed direct and native-chat
  branches. Raw FLD generations exposed a scientific prompt/metric mismatch:
  the prompt asks for a proof, while exact match scores only the class label.
  Removed both FLD tasks rather than accepting an extraction-confounded floor.
  `hendrycks_math500` constructed cleanly and replaces that coverage in the
  final ten-task suite. Submitted final direct/native-chat smoke `3834836`;
  all ten datasets constructed successfully before model loading.
- 11:04 CEST submitted third reviewer smoke `3834738` with
  `agieval_logiqa_en`, after directly constructing its 651-example test split.
  Added `scripts/validate_lm_eval_tasks.py` so smoke and production wrappers
  instantiate every task dataset before loading a model. Rewired all six held
  evals from failed `3834737` to `afterok:3834738` while preserving their
  model/instruction dependencies and A100-80GB constraints.
- 11:02 CEST replacement smoke `3834737` showed `logiqa2` has the same hidden
  legacy-script incompatibility as `logiqa`. Direct task construction of
  `agieval_logiqa_en` succeeded on 651 examples, so it replaces both. Added a
  reusable preflight that constructs every task dataset before model startup;
  a third smoke and production dependency rewiring follow.
- 10:58 CEST submitted corrected downstream-suite smoke `3834737` with
  `logiqa2` and rewired production NL `3834729/3834730`, control
  `3834731/3834732`, and logic `3834733/3834734` evals away from failed smoke
  `3834728`. Every eval now waits on both its model/instruction artifact and
  `afterok:3834737`; no production eval can release on the failed gate.
- 10:57 CEST reviewer smoke `3834728` failed as intended before production
  release: task registry and model initialization passed, but actual dataset
  construction showed legacy `logiqa` depends on `logiqa.py`, rejected by the
  installed `datasets` version. Switched the suite and named shortcuts to the
  supported `logiqa2`; replacement smoke and dependency rewiring follow.
- 10:53 CEST downstream preflight found `folio` absent from the installed
  lm-eval registry, which would have failed all six held evals before model
  loading. Replaced it with paired `fld_default` and
  `fld_logical_formula_default`, added `mmlu_pro`, and validated all 11
  task/group names. Added task preflight and incomplete-output archival to the
  eval wrapper. Submitted direct/native-chat limit-one suite smoke `3834728`,
  now running on A40, and made every production eval depend on it. Canceled
  untouched old evals `3831114/3831116/3831120/3831122/3831124/3831126` and
  submitted NL `3834729/3834730`, control `3834731/3834732`, and logic
  `3834733/3834734`; all remain A100-80GB-only. Replaced queued watcher
  `3834722` with current-prompt watcher `3834736` at the same 15:15 start.
- 10:46 CEST replaced the queued first watcher so its Slurm-spooled prompt
  contains the current recovery graph. Canceled untouched begin-time job
  `3834564` and submitted `3834722` for the same 15:15 CEST start. The updated
  prompt tracks eval `3834582`, row audits `3834706`, and aggregation `3834707`;
  it retains the six-hour self-scheduling chain and full plan scope.
- 10:43 CEST prevented a stale pilot artifact from suppressing production row
  12. The pilot gate had written an unsuffixed 224-prompt/8-generation JSON,
  which the full wrapper would otherwise treat as complete. Preserved its
  metrics/samples under `_pilot_gate`, cleared the production names before any
  eval started, and hardened the local wrapper to skip only exact
  448-prompt/16-generation artifacts with 1,024 retained rows and 896 sampled
  rows. Shell syntax and diff checks pass.
- 10:40 CEST installed the corrected full-grid artifact acceptance chain.
  Extended the pilot auditor to filter mixed greedy/sampled JSONL rows while
  checking total retention and unique prompt coverage. Added row audit array
  `3834706_[0-29%8]` after eval `3834582` and strict aggregation `3834707`
  after all audits pass. Each row requires complete metrics, 1,024 retained
  rows, all 448 prompts, sample indices `0/1`, complete chunk logs/cap
  diagnostics, and fresh formal constants. Tests and shell/Slurm checks pass
  (`12 passed`). Live SFT rows were at `239..1333/10000`; Nanotron logic/NL
  were matched at step `1241`, `651M` tokens, and `30.9--31.3K` tokens/s.
- 10:34 CEST repaired full-grid qualitative retention before any eval started.
  The pending wrapper computed all 16 sampled generations but retained none.
  It now retains two sampled generations for each of the 448 prompts (896 rows)
  without changing generation compute or metrics. Tests and shell/Slurm checks
  pass (`4 passed`). Canceled untouched eval array `3829073` and submitted
  replacement `3834582_[0-29%6]` with the same row-wise dependency on
  `3829072`, A100-80GB constraint, throttle 6, 24-hour limit, and full protocol.
- 10:30 CEST prepared the corrected full-grid analysis gate. Extended
  `aggregate_hfsa_depth_scaling.py` to recognize `branchproof_unique_v2` run
  names, explicitly skip old intermediate checkpoints, and reject aggregation
  unless all 30 unique rows have exact prompt/generation metadata plus complete
  greedy and pass@`1/2/4/8/16` correctness/validity/joint cells at every depth.
  It also checks pass@k monotonicity. Focused tests pass (`9 passed`),
  `py_compile` passes, old-grid compatibility still finds `30/30`, and a strict
  smoke rejects the current one-row pilot with an incomplete manifest.
- 10:27 CEST live continuation: all released BranchProof SFT rows `0..11`
  entered optimizer training and ranged from steps `23..522/10000`, with
  sampled GPUs at 100% utilization and no fatal/OOM/quota signature. The
  control recovery estimate moved to 14:43 CEST, so oversight job `3834564`
  was safely moved from 16:21 to 15:15 CEST to verify its actual resume path;
  future successors remain six hours apart.
- 10:23 CEST installed persistent six-hour critical-path oversight. Added
  `scripts/slurm/codex/branchproof_nanotron_oversight_2026-07-11.slurm`,
  validated it with `bash -n`, a batch-shell `cs` lookup, Slurm `--test-only`,
  and `git diff --check`, and submitted initial begin-time job `3834564` for
  about 16:21 CEST on one A100-MIG slice. Each pass schedules its successor
  before Codex, executes the full BranchProof/Nanotron/downstream/conditional/
  report plan, avoids no-op doc churn, and cancels the chain only after the
  complete plan is verified. Maximum default coverage is 120 six-hour passes.
- 10:21 CEST Nanotron checkpoint/quota verification: corrected logic/NL jobs
  are healthy near steps `1181/1191`, `619M/624M` tokens, and `31K` tokens/s.
  Their live exports override checkpoint interval to `4096`; the earlier
  handoff wording about 1024-step checkpoints was wrong. Neither corrected run
  has written a checkpoint yet, while the unaffected control retains one
  complete `199G` step-4096 checkpoint. This keeps the expected first recovery
  boundary below the vault hard quota.
- 10:15 CEST user accepted the expected corrected eval runtime and required
  A100-80GB evaluation. Verified submitted eval array `3829073_[0-29%6]` is
  partition `a100` with feature `a100_80`, 24-hour limit, and per-row
  `aftercorr:3829072_*` dependencies. Released manual hold on full corrected
  SFT `3829072_[0-29%12]`; it is now eligible on `a40,a100`. Preserve the full
  32-prompt, 16-generation, all-depth, pass@16 eval protocol and use the first
  completed A100-80 row as the production runtime check.
- 10:15 CEST released SFT rows `3829072_0..11` started immediately on A40
  nodes; rows `12..29` are pending only on the array throttle. Representative
  startup logs show no fatal/OOM/quota signature. Eval remains dependency-held
  and A100-80-only.
- 10:07 CEST corrected the runtime interpretation after direct node inspection:
  qualitative probe node `a0121` is also an A40, not an A100. Thus the
  above-25-hour full-row estimate is an A40 extrapolation, not a measured
  A100-80 result. Old full A100-80 rows took roughly 3.5--8 hours; matched old
  row 12 took `03:56:41`. The corrected surface is nevertheless materially
  heavier: retained depth-25 output averages `3236` vs `1923` old tokens and
  depth-50 `6475` vs `2678`, with cap `7168` vs `4096`, batch `64` vs `128`,
  and an additional greedy pass. Keep full SFT held pending a corrected
  A100-80 timing probe, then choose whole-row or depth-sharded eval.
- 09:55 CEST corrected BranchProof gate and qualitative review completed.
  Gate `3832945_12` finished in `07:10:38`; structural/runtime audit `3831136`,
  sampled probe `3833178_12`, and sampled audit `3833179` all completed and
  accepted. Citation-free joint is perfect through train depth 25 and reaches
  depth-30/40/50 pass@1 `0.883/0.625/0.344` and pass@8
  `1.000/1.000/0.938`. Strict `valid=0` is expected because the trained proof
  syntax omits numbered citations; it is not the intended self-contained
  validity metric. The retained depth-50 slice contains `15/32` correct,
  `11/32` citation-free joint, and `24/32` complete-format outputs. Failures
  are late branch transitions or repetitive cap hits; all prompts retain fresh
  constants and the corrected unique-answer construction.
- 09:55 CEST runtime gate: measured A40 generation was `3826.3s` greedy plus
  `20677.8s` sampled, with one greedy and seven sampled cap-hit chunks. Scaling
  this A40 measurement to 32 prompts and 16 generations projects above 25
  hours before scoring. No scientific protocol reduction is approved.
- 09:49 CEST corrected Nanotron p15 logic `3830927_3` and NL `3831111_8` are
  running on full A100-80GB nodes `a0534/a0535`. Both are near step
  `1071/8192`, `562M/4.295B` tokens, `31.0--31.3K` tokens/s, and loss about
  `1.73`, with no fatal/OOM/quota signature. Their recovery rows will resume
  after the expected 24-hour allocation boundary from the complete step-4096
  checkpoint. Normal-control recovery `3828946_0` remains priority-pending
  with a 17:45 CEST start estimate.
- 02:10 CEST sampled qualitative coverage repair: discovered that the gate's
  `collect_samples=128` budget fills entirely during greedy scoring, leaving no
  raw sampled pass@k generations. Added independent
  `--collect-sampled-samples` support and changed sampled retention to cover
  prompts before later generation indices while recording `sample_index`.
  Focused/regression tests, `py_compile`, shell syntax, Slurm `--test-only`,
  and `git diff --check` pass (`10 passed`). Submitted sampled-only probe
  `3833178_12` after gate `3832945_12`: depths `1/25/30/50`, four prompts/depth,
  eight generations, pass@`1/2/4/8`, all 128 outputs retained under a separate
  suffix. Audit `3833179` then requires source `synthetic_sampled`, 32 rows,
  four unique prompts, all indices `0..7`, fresh constants, and complete
  sampled metrics at every depth. Full SFT remains held for both audits and
  manual review.
- 02:03 CEST corrected gate entered sampling. All four greedy chunks completed;
  chunk 4 produced `193844` tokens in `1463.2s` with max exactly `7168`, so at
  least one depth-45/50 model generation hit the cap. Every corrected formal
  gold target is at most `6212` tokens, making this a model length/nontermination
  failure rather than inadequate intended-target budget. Sampled chunks
  `1..8/28` completed with expected shallow lengths and no sampled cap hit by
  02:10.
- 01:44 CEST full-eval recoverability check: both `a40` and `a100` have
  `MaxTime=24:00:00`, and corrected full rows would contain 112 sampled chunks
  while the evaluator writes outputs only at the end. Added a release trigger:
  after gate `3832945` provides complete sampled throughput, project the
  32-prompt, 16-generation A100 runtime. If a conservative row estimate
  approaches 20 hours, make downstream eval `3829073` depth-sharded/resumable
  before it starts. Preserve the full depth, prompt, generation, and pass@16
  protocol rather than reducing evidence to fit walltime.
- 01:42 CEST provisional BranchProof length diagnostic: completed greedy gate
  chunks 1--3 differ from the corresponding per-depth median-gold target-token
  totals by only `-0.01%`, `-0.06%`, and `+0.19%`. Together with maximum
  generated length `5212 < 7168`, this rules out gross under-generation,
  truncation, or runaway repetition through depth 40. It does not establish
  answer correctness or proof validity; those remain blocked on scored
  artifacts and raw-generation review.
- 01:35 CEST strengthened pilot runtime evidence while gate `3832945_12` is
  running. `audit_branchproof_unique_v2_pilot_eval.py` now parses vLLM
  completion records, requires all four greedy and 28 sampled chunks exactly
  once, and records elapsed time, output tokens, throughput, maximum generated
  length, and cap-hit count. Cap hits remain diagnostics, not automatic
  rejection. Audit job `3831136` will read the current gate log. Focused tests,
  `py_compile`, `bash -n`, Slurm `--test-only`, and `git diff --check` pass
  (`7 passed`). Live greedy chunk 3 completed in `1682.2s` with `264625`
  output tokens and max `5212`; no completed greedy chunk hit cap `7168`, and
  final greedy chunk 4 is active at 99% GPU utilization.
- 01:32 CEST enforced the manual BranchProof release gate: full corrected SFT
  array `3829072` still depends on `afterok:3831136` and is now additionally
  `JobHeldUser`. This prevents the 30 expensive runs from becoming eligible
  immediately after a structural-only audit. After the audit passes, inspect
  representative shallow, held-out, depth-50, correct, incorrect, and any
  cap-length generations; release the same array with
  `scontrol release 3829072` only if those samples confirm the intended task
  and evaluator behavior. No resubmission or dependency loss occurred.
- 01:22 CEST queue refresh: corrected Nanotron p15 NL, logic, and unaffected
  control recovery remain priority-pending. Slurm's current, non-guaranteed
  start estimates are 11:13, 12:27, and 15:04 CEST, respectively. No safe
  partition widening is available for these full-node A100 jobs.
- 01:20 CEST corrected-eval token-budget audit: added a reproducible audit over
  every corrected validation row at all 14 depths under the actual OLMo
  tokenizer and task renderer. All 14,000 formal and 14,000 NL gold traces fit
  the shared 7,168-new-token cap and 16,384-token context. Formal target/total
  maxima are `6212/13123`; NL maxima are `6674/13596`, leaving minimum
  headroom `494/2788`. The accepted machine-readable artifact is
  `analysis/branchproof_unique_v2_eval_token_budget_2026-07-11.json`; focused
  script/audit regression tests pass (`5 passed`). Live gate `3832945_12`
  remains healthy on A40 and had completed two greedy chunks by 01:17.
- 01:00 CEST official-preprint integrity repair: a source audit found that the
  abstract and status box quarantined old BranchProof evidence while the title,
  introduction, contributions, main tables/figures, discussion, and conclusion
  still asserted the invalidated effect. Pulled the report repo, changed the
  title to the neutral `Formal Logic as a Substrate for Symbolic
  Chain-of-Thought`, and made the rendered evidence contain only the independent
  AttrCon signal, the closure audit, and the corrected evaluation protocol.
  Old BranchProof performance, architecture, syntax, shortcut, hybrid,
  conditioned-dual, and mixture sections remain behind a disabled provenance
  switch and in the informal report. Static checks found one active evidence
  figure, no duplicate active labels, no active references to hidden result
  labels, and no whitespace errors. TeX compilation remains unavailable.
  Updated `AGENTS.md` so future generated-report mirroring targets
  `informal_report/main.tex` and cannot overwrite the official root preprint.
  Pushed the report repair as `eba30a1`.
- 00:52 CEST corrected a silent pilot-eval launch bug before significant GPU
  time was spent. Slurm interpreted `PASSK_K_VALUES=1,2,4,8` as an export list,
  and process inspection showed `3831135_12` had launched with only
  `--k-values 1`. Canceled it after four minutes, removed its 14 GB merge temp,
  added colon-delimited `PASSK_K_VALUES_COLON` resolution to the eval wrapper,
  and verified the shell expansion plus `bash -n`. Corrected gate
  `3832945_12` started immediately on A40 `a1721`; dependency-edited audit
  `3831136` to `afterok:3832945_12`. Live process inspection confirms the
  corrected command contains `--k-values 1,2,4,8`; focused audit tests pass
  (`3 passed`). The vLLM tokenizer warning was also checked directly: base and
  merged OLMo GPT-2 tokenizers produced identical IDs and lengths on formal,
  prose, whitespace, and newline probes. Full SFT remains behind the audit.
- 00:49 CEST corrected BranchProof gate release: one-seed pilot SFT
  `3829069_12` completed all 10,000 steps in `12:39:51`, with final train loss
  `0.0171`, a final adapter, and complete step-5000/10000 checkpoints. Pending
  gate eval `3831135_12` was safely widened from A100-only to `a40,a100`
  without changing its job ID or dependencies and started immediately on A40
  `a0226`. The 30-row SFT remains gated on structural audit `3831136`.
- 00:49 CEST Nanotron control transition: unaffected control `3823434_0`
  reached its exact 24-hour wall limit after logged iteration `5051`, as expected.
  Direct post-timeout inspection reconfirmed step 4096 as a complete `199G`
  recovery checkpoint with model, four equal optimizer shards, scheduler, RNG,
  and metadata recording exactly `2,147,483,648` consumed tokens. Guarded
  recovery `3828946_0` is priority-pending; current control/logic/NL start
  estimates are 15:04/11:13/11:46 CEST.

## 2026-07-10

- 17:14 CEST Nanotron mixture-schedule audit: verified that production uses
  `TokenizedBytes` plus `BlendableDataset` over 4,096-token packed chunks.
  Nanotron's exact compiled helper realizes 157,287 proof chunks out of
  1,048,576 (`644,247,552/4,294,967,296` tokens, `15.000057%`) for both logic
  and NL p15. Normal and corrected proof corpora have enough capacity to avoid
  wraparound. Step-4096 control metadata records the exact consumed token
  count, and absolute sampler/consumption offsets preserve the blend after
  recovery. The three-step smoke's one-of-three proof share is only small-smoke
  granularity. Added `docs/nanotron_mixture_schedule_audit_2026-07-10.md`.
  Current Slurm estimates improved to 2026-07-11 06:45 for corrected logic and
  10:21 for corrected NL.

- 16:50 CEST instruction-transfer format repair: audited the held downstream
  branch and found a train-eval mismatch. UltraChat SFT used custom
  `<question>/<answer>` wrappers, while lm-eval used neither those wrappers nor
  a chat template. `train_instruction_sft.py` now uses the Qwen tokenizer's
  native system/user/assistant template, masks all prompt tokens, supervises
  only the assistant response, and filters examples truncated before any target
  token. Instruction-branch lm-eval now applies the same chat template; direct
  base-model eval remains untemplated. Four focused tests, a real cached
  Qwen-tokenizer prefix/masking check, and the full suite pass (`129 passed, 3
  skipped`). A100-MIG data smoke `3831179`
  completed in `00:02:15`: `32/32` train and `16/16` eval examples retained
  targets, sampled lengths were `71/537/1202` min/mean/max, and decoded input/
  supervised spans matched the intended native chat rendering. Dependency-
  edited all three instruction jobs to require this smoke. Vault epilogue
  accounting is `511.9G` used, `1048.6G` soft quota, `2097.2G` hard quota.

- 16:38 CEST matched corrected midtraining pilot and quota guard: corrected NL
  Nanoset smoke `3831110` completed in `00:00:34`, loaded the exact corrected
  NL path at `0.85/0.15` normal/proof weights, and finished three optimizer
  steps. Submitted NL p15 train/recovery `3831111 -> 3831112`, upload/direct
  eval/instruction SFT/instruction eval `3831113..3831116`. Replaced the still
  pending control downstream jobs by `3831119..3831122` and corrected-logic
  upload/eval jobs by `3831123..3831126`; all dependencies now use `afterok`
  after upload, and both logic/NL have direct plus instruction-tuned branches.
  A complete Nanotron checkpoint occupies about `199G` on the vault (`29G`
  model, `171G` optimizer), so the upload wrapper now supports a path-guarded
  opt-in that deletes all run checkpoints only after successful HF upload; it
  is enabled for all three p15 conditions. Logic/NL full-node starts are
  currently estimated for 2026-07-11 10:21/11:07 CEST.
- 16:38 CEST faster corrected SFT gate: comparable old eval rows took 13--23
  hours, making the original full-size pilot eval a 24-hour timeout risk.
  Canceled pending `3829070 -> 3831023` and submitted replacement gate eval
  `3831135_12` with 16 prompts/depth, 8 generations, greedy and
  pass@`1/2/4/8`, followed by structural audit `3831136`. Full SFT `3829072`
  now waits on `afterok:3831136`. The eventual 30-row eval `3829073` is
  unchanged at 32 prompts/depth, 16 generations, and pass@16.

- 16:23 CEST corrected SFT artifact gate: added
  `audit_branchproof_unique_v2_pilot_eval.py` plus three regression tests. The
  gate checks all 14 depths, greedy and pass@`1/2/4/8/16` metric cells,
  monotonic/finite pass@k values, exactly 128 retained raw generations, and
  contiguous prompt constants `c0..c_depth`; the stale depth-18 wrapped form
  and missing metrics are rejected. Tests pass (`3 passed`). Submitted audit
  `3831023` after pilot eval `3829070` and dependency-edited full SFT
  `3829072` to `afterok:3831023`. Pilot SFT is healthy around step `3300`; the
  normal Nanotron control is healthy at iteration `4711/8192` (`2.47B`
  tokens, `30.8K` tokens/s).

- 16:06 CEST normal-control checkpoint gate: while `3823434_0` continued at
  iteration `4641/8192`, verified checkpoint `4096` has complete metadata, 625
  model files, four equal `22,848,937,060`-byte optimizer shards, and no
  zero-byte files. Correction at 16:38: `91.4G` was the logical optimizer total,
  not the full checkpoint footprint; filesystem-accounted model plus optimizer
  is about `199G`. Guarded recovery `3828946_0` has a safe resume point when the
  current allocation ends.
- 16:02 CEST corrected corpus context audit: packed document lengths are
  closely matched but not always contained in one 4,096-token window. Logic
  median/p95/max is `3753/7358/7725` tokens with `44.3%` above 4096; NL is
  `3826/7418/7841` with `47.8%` above 4096; neither modality exceeds 8192.
  Record this as packed continuation midtraining, not whole-example SFT.
- 15:58 CEST corrected Nanotron release gate: packed audit `3830855`
  completed cleanly in `00:01:46`. Metadata, binary file sizes, source token
  counts, and one EOS per record all agree; 15 packed records spanning depths
  3--24 exactly match Qwen source tokenization and decode round trips for both
  logic and NL. Corrected 15% logic Nanoset smoke `3830924` then loaded the
  intended 85/15 normal/proof blend and completed three optimizer steps.
- 15:58 CEST corrected midtraining pilot: patched train/push/eval wrappers to
  accept isolated run-tag/root overrides, avoiding stale `logic_p15` paths.
  Submitted full-node 15% logic pilot `3830927_3`, automatic `afterany`
  resume/skip `3830928_3`, HF upload `3830939_3`, and direct reviewer eval
  `3830940_3`. The pilot uses checkpoint interval `4096` and a fresh
  `qwen25_7b_midtrain_logic_p15_bp_unique_v2_4p3b` run root; broad corrected
  mixture rows remain gated on its train/eval inspection.
- 15:49 CEST corrected artifact audit: materialized build/push `3829067` and
  both 1.2B-token proof corpus/Nanoset builds completed cleanly. Logic contains
  `311,301` records and `1,200,002,513` source tokens; NL contains `307,329`
  records and `1,200,004,689` source tokens. Packed totals equal source tokens
  plus one EOS per record. A full scan of all `307,329` paired records found
  zero prompt, answer, wrapper, or fresh-constant-contiguity mismatches.
  Added exact packed-token/decode audit utility and submitted A100-MIG job
  `3830855`; corrected proof-mixture training remains gated on it.
- 15:49 CEST corrected SFT orchestration: pilot `3829069_12` is healthy around
  step `2725/10000`; eval `3829070_12` is dependency-pending. Edited full SFT
  `3829072` from `afterok:3829069` to `afterok:3829070`, preventing the 30-row
  grid from releasing before pilot evaluation succeeds. Full eval `3829073`
  remains behind the full SFT array.
- 15:49 CEST normal-control status: unaffected Nanotron control `3823434_0`
  reached iteration `4561/8192` (`2.39B` tokens, about `30.8K` tokens/s). Its
  ETA exceeds the current allocation, so guarded recovery `3828946_0` is
  expected to resume from the latest complete checkpoint; downstream
  `3828947..3828950` remain dependency-gated.

- 11:49 CEST BranchProof validity audit: forward-chaining the old
  `hard_fsa_schema` prompts found no ambiguity through depth 15, but multiple
  derivable candidate answers in `73/96` examples at depth 20, `74/96` at
  depths 25/30/35, and `92/96` at depths 40/45/50. The generator reused its
  18 one-letter constants at long depth, allowing branches to re-enter the
  same formal atoms. A stored depth-50 logic generation was independently
  wrong by the labeled answer but citation-free valid for another candidate.
  Quarantined all old long-depth BranchProof results and derivative ablations.
- 11:49 CEST generator/evaluator repair: changed `hard_fsa` and
  `hard_fsa_schema` to use unique constants `c0..c_depth` and explicit atoms
  such as `A(c18)`; extended the natural-logic renderer and the symbol-padded
  and wordified syntax transforms; added closure/unique-answer checks to the
  production probe and regression tests. Corrected closure audits found zero
  ambiguous examples at every tested depth `5..50`; production probe passed
  1,000 train plus 2,000 eval examples at unique-solution rate `1.0` and max
  derived candidates `1`. Focused tests pass (`84 passed`); the full suite also
  passes (`122 passed, 3 skipped`).
- 11:49 CEST corrected reruns: submitted materialized build/push `3829067`,
  corrected 1.2B-token Nanotron corpus builds `3829068_[0-1%2]`, one-row SFT
  pilot/eval `3829069_12 -> 3829070_12`, and pilot-gated full SFT/eval
  `3829072 -> 3829073`. Corrected eval uses equal `7168` generation caps,
  context `16384`, greedy accuracy, and pass@`1/2/4/8/16`.
- 11:49 CEST Nanotron containment and storage cleanup: checkpoint inspection
  found silent zero-byte/truncated files under vault pressure, so train/push
  wrappers now reject incomplete checkpoints and resume only from complete
  optimizer/model snapshots. Canceled all old proof-mixture rows/dependents and
  removed stale proof checkpoints/corpora, reducing
  `$HPCVAULT/synthetic-RLVL/nanotron_midtrain` from about `1.7T` to essentially
  empty before the normal control's next save. Kept unaffected normal-corpus
  control `3823434_0` and its guarded recovery/downstream chain
  `3828946..3828950`.

## 2026-07-08

- 10:15 CEST Nanotron resume fix: Qwen2.5 row `3819135_3` (`logic_p15`) is running cleanly on `a0932` and passed local iteration `2531/8192`; it has complete checkpoints through local step `2048`. Recovery row `3819040_0` (`control_p0`) also started on `a0931` and reached local step `2521`, but log/checkpoint inspection showed the old wrapper loaded the previous `4096` weights while resetting the local step counter. Continuing that row to local `8192` would overtrain the control condition, so it was canceled after verifying a complete local-step-2048 checkpoint. Patched `scripts/slurm/jobs/nanotron_qwen25_midtrain_grid_2026-06-24.slurm` so run-checkpoint resumes load optimizer and LR scheduler state; pretrained Qwen loads remain weight-only. Added an optional `FINAL_CHECKPOINT_ALIAS` hook, but do not use it for the control baseline unless a shortened recovery is intentionally accepted.
- 10:20 CEST Nanotron clean control restart and quota cleanup: removed superseded active-run checkpoints control `p0` `1024`, old pre-recovery `4096`, compromised reset-state `2048`, and logic `p15` `1024`; kept logic `p5` `1024`, logic `p10` `4096`, and logic `p15` local `2048`. `/home/vault` is about `856G/1000G` and `$HPCVAULT/synthetic-RLVL/nanotron_midtrain` is `596G`. Canceled superseded recovery chains `3819040..3819044` and `3823409..3823413`; submitted clean control restart `3823434_[0%1]`, automatic `afterany` resume/skip `3823435_[0%1]`, and dependents `3823436..3823439`. Corrected rows `1..2` recovery remains `3823414_[1-2%1]` with dependents `3823415..3823418`. Dependency spot checks show the new afterany/aftercorr chains are correctly attached.

## 2026-07-07

- 11:17 CEST Nanotron quota cleanup and recovery: checked Qwen2.5 proof-mixture production chain. Rows `3808429_0` and `3808429_2` timed out after 24h with usable step-4096 checkpoints; row `3808429_1` failed at step 2048 while saving due `/home/vault` quota (`Disk quota exceeded`) and left a partial checkpoint. Removed only stale/intermediate Nanotron checkpoints and the partial failed checkpoint: control `p0` steps `1024/2048/3072`, logic `p10` steps `1024/2048/3072`, and logic `p5` partial step `2048`. Kept latest usable checkpoints: control `p0` step `4096`, logic `p5` step `1024`, logic `p10` step `4096`, plus base checkpoint and Nanosets. `/home/vault` quota dropped from about `1427G/1000G` to `822G/1000G`; `$HPCVAULT/synthetic-RLVL/nanotron_midtrain` is now about `596G`.
- 11:17 CEST Nanotron dependency repair: canceled poisoned downstream arrays `3808430/3808431/3808432/3808433`, which were stuck after the failed row marked the original chain `DependencyNeverSatisfied`. Submitted recovery training `3819040_[0-2%1]`; dependent recovery push/direct-eval/instruction-SFT/instruction-eval arrays are `3819041/3819042/3819043/3819044`. Kept original queued training `3808429_[3-10%1]` for untouched rows and submitted split dependents `3819053/3819054/3819055/3819056` for rows `3-10`. Slurm currently gives no start estimate; `3819040_[0-2]` is priority-pending, and `3808429_[3-10]` is pending on unavailable node `a0803`.
- 11:22 CEST Nanotron rows-3-to-10 resubmission: at user request, canceled stuck original training `3808429_[3-10]` plus its dependents `3819053/3819054/3819055/3819056` and resubmitted only rows `3..10`. New train array is `3819135_[3-10%1]`, with `ExcNodeList=a0803`; dependents are push `3819136`, direct eval `3819137`, instruction SFT `3819138`, and instruction eval `3819139`. Recovery rows `0..2` and dependents `3819040..3819044` were left untouched.

## 2026-07-03

- 10:36 CEST Nanotron midtraining recovery: Qwen2.5 proof-mixture training array `3801554_[0-10%1]` started on 2026-07-02 and all rows failed before optimizer progress with `AssertionError: Tokenizer vocab size (151665) does not match model config vocab size (152064)`. This is Qwen2.5's padded model vocabulary, not a bad Nanoset: token IDs fit inside the embedding table, but local Nanotron required exact tokenizer/model vocab equality. Patched local `../nanotron/run_train.py` so Nanoset loading accepts `len(tokenizer) <= model_config.vocab_size` and logs a rank-0 warning for padded vocabularies. Canceled poisoned downstream arrays `3801555/3801556/3801557/3801558` and submitted replacement dependency chain: training `3808220_[0-10%1]`, HF push `3808241_[0-10%2]`, direct eval `3808252_[0-10%3]`, instruction SFT `3808253_[0-10%2]`, and instruction eval `3808274_[0-10%3]`. The chain is priority/dependency-pending on `a100`; no partition widening is safe because training still needs a full 8xA100-80GB node.
- 11:25 CEST Nanotron single-GPU debug and resubmission: canceled unstarted chain `3808220/3808241/3808252/3808253/3808274` per user request and debugged Nanotron through fast single-GPU smokes before full-node resubmission. RTX Pro smoke `3808324` failed because current torch `2.6.0+cu124` does not support Blackwell `sm_120`, so Nanotron must avoid `rtxpro6k` until the env is rebuilt. Added tiny and full single-GPU smoke wrappers. Fixed installed-Datatrove constructor compatibility in local `../nanotron`, installed `pybind11`, patched the C++ helper build to use the active venv Python, and removed the S3-only assertion from local consumption stats. Tiny Qwen/Nanoset smoke `3808410` completed 3 steps on A100-MIG; full Qwen2.5-7B smoke `3808424` completed 2 steps on one A40 after loading all 199 converted checkpoint shards. Resubmitted production chain as training `3808429_[0-10%1]`, HF push `3808430_[0-10%2]`, direct eval `3808431_[0-10%3]`, instruction SFT `3808432_[0-10%2]`, and instruction eval `3808433_[0-10%3]`; training is priority-pending on A100-80 with no start estimate yet.
- 14:50 CEST Nanotron queue check: production chain is still pending unchanged. `3808429_[0-10%1]` is eligible but priority-pending with no start estimate and no logs; `3808430/3808431/3808432/3808433` remain dependency-pending. Partition audit found no safe workaround: `a100` nodes are draining for `wrong kernel version` / `Reboot ASAP`, `a40` nodes are draining or rebooting, `a100mig` is draining, and `rtxpro6k` is incompatible with current torch for Nanotron. Wait for cluster reboot/kernel recovery, then inspect `3808429_0` as soon as it starts.

## 2026-07-02

- 09:36 CEST experiment status refresh: live `squeue`, `sacct`, `scontrol`, partition availability, prerequisite paths, and Nanotron output roots were checked. The only active `synthetic-RLVL` science chain is the Qwen2.5 proof-mixture Nanotron chain: training `3801554_[0-10%1]` is still priority-pending with Slurm estimate 2026-07-02 18:08 CEST on `a0535`; downstream `3801555/3801556/3801557/3801558` are dependency-pending as intended. All three Qwen Nanosets and the converted checkpoint exist, and no new repo job failures were found after the fixed resubmission. No partition widening was made because the job requires a full 8xA100-80GB node and visible alternatives would require a real memory/config change. Visible `babylm-*` jobs are unrelated to this repo.

## 2026-07-01

- 13:34 CEST OLMo-3-32B conditioned-dual sample audit: inspected seed-level pass@16 metrics and matched stored samples for single vs conditioned logic/NL. Exact prompt matching found `240/240` matches for both modes, with no visible mode leakage (`conditioned_logic` all `<formal>`, `conditioned_nl` all `<think>`). The correct interpretation is same-step mixed exposure, not additive data. Conditioned logic slightly exceeds single logic on correctness in Table 7, but the effect is small; conditioned NL is high-variance, with seed `3409` better than single NL and seeds `3407/3408` worse. Representative samples fail by wrong branch/state continuation, not parser or prompt-format collapse. Added audit note `analysis/logic_cot_report_2026-05-25/conditioned_dual_32b_sample_audit_2026-07-01.md` and tightened the official preprint wording in `../synthetic-RLVL-report/main.tex`.
- 11:32 CEST conditioned-dual framing correction: updated `../synthetic-RLVL-report/main.tex` and the handoff docs so the OLMo-3-32B conditioned-dual follow-up is compared apples-to-apples on the same OOD/hard-tail bands. Correct readout: 7B conditioned-dual remains weaker than single modality, but at 32B the formal conditioned mode slightly exceeds single-modality logic in correctness (`0.963/0.979` vs `0.954/0.975`) and citation-free joint (`0.487/0.715` vs `0.477/0.709`). Conditioned NL remains below/near single-modality NL (`0.608/0.782` correct and `0.537/0.743` translated joint vs `0.685/0.825` and `0.546/0.748`). This supports the capacity-dependent framing for dual-modality logic without making a broad dual-modality or grounding claim.
- 11:06 CEST Nanotron midtraining recovery: first Qwen2.5 proof-mixture training array `3776105_[0-10%1]` started but all 11 rows failed before optimizer progress with `dacite.exceptions.UnionMatchError` on `data_stages.data.dataset`. Root cause is the generated Nanoset YAML: it passed `tokenizer_name` but omitted explicit `token_size_in_bytes`/`vocab_size`, causing Nanotron's strict union parser to swallow the metadata assertion as a generic union mismatch. Patched `scripts/slurm/jobs/nanotron_qwen25_midtrain_grid_2026-06-24.slurm` to emit `token_size_in_bytes: 4` and `vocab_size: 152064`; validation passed with `bash -n`, `sbatch --test-only`, and `git diff --check`. Canceled stale dependents `3776106`/`3776107`/`3776108`/`3776109` and submitted fresh chain: training `3801554`, HF push `3801555`, direct eval `3801556`, instruction SFT `3801557`, instruction eval `3801558`. `3801554` is priority-pending with Slurm estimate 2026-07-01 21:22 CEST on `a0633`.

## 2026-06-30

- 10:18 CEST conditioned-dual 32B result, corrected 2026-07-01 11:32 CEST: replacement eval `3795089` completed all `6/6` OLMo-3-32B conditioned-dual rows. Summary: conditioned logic OOD/hard-tail correct@16 `0.963/0.979`, citation-free joint@16 `0.487/0.715`, strict/grounded joint `0.000`; conditioned NL OOD/hard-tail correct@16 `0.608/0.782`, translated joint@16 `0.537/0.743`, parse@16 `0.938/0.965`. Apples-to-apples with single-modality OLMo-3-32B on the same OOD/hard-tail bands, conditioned logic slightly exceeds single-modality logic in correctness (`0.963/0.979` vs `0.954/0.975`) and citation-free joint (`0.487/0.715` vs `0.477/0.709`), while conditioned NL remains lower/near-tied (`0.608/0.782` correct and `0.537/0.743` translated joint vs `0.685/0.825` and `0.546/0.748`). Sample checks found intended mode prompts and surfaces.
- 10:18 CEST Nanotron prereq recovery: prereq `3795206` failed after finishing raw JSONLs and Nanoset tokenization because the HF-to-Nanotron converter assumed Llama-style config fields absent from the installed Qwen2.5 config (`pretraining_tp`, attention-bias/RoPE field shape). Patched local `../nanotron/examples/llama/convert_hf_to_nanotron.py` to default missing Qwen2 config fields and patched `convert_weights.py` to map Q/K/V biases. Validation: `py_compile`, `git diff --check`, `bash -n`, and `sbatch --test-only` passed; login-node import still cannot run because Triton needs a GPU driver. Submitted prereq rerun `3797384` and updated `3776105` dependency to `afterok:3797384`.
- 10:24 CEST Nanotron prereq recovery follow-up: rerun `3797384` failed quickly during conversion because Nanotron expected distributed env vars such as `WORLD_SIZE`; patched the production prereq and stale real-data smoke Slurm wrappers to invoke the converter with `torchrun --standalone --nproc_per_node=1 -m examples.llama.convert_hf_to_nanotron`. Validation passed (`bash -n`, `sbatch --test-only`, `torchrun --help`, `git diff --check`). Submitted prereq rerun `3797409` and rewired `3776105` to `afterok:3797409`; `3776105` is now ordinary dependency-pending again.
- 10:27 CEST Nanotron prereq complete: `3797409` completed cleanly after `00:01:22`, verified the three Nanoset roots, and wrote the converted Qwen2.5 Nanotron checkpoint at `$HPCVAULT/synthetic-RLVL/nanotron_checkpoints/qwen25_7b_tp1` (29G). Training array `3776105_[0-10%1]` is now released with no dependency and priority-pending; Slurm currently estimates first start at 2026-06-30 21:22 CEST on `a0534`.
- 10:18 CEST post-eval cleanup: removed the now-inactive OLMo-3-1125-32B HF cache/offload and conditioned-dual seed-3409 intermediate checkpoints after eval completion. Kept final adapters and eval outputs. `$HPCVAULT` quota is about `543G/1000G`, and `cache_offload` is essentially empty.

## 2026-06-29

- 14:15 CEST conditioned-dual 32B recovery completion: resume job `3795088` completed cleanly after `02:38:56` and produced the missing seed-3409 `final/` adapter. Replacement eval `3795089` started; row `0` is running as raw job `3795535`, and rows `1..5` are pending behind the array throttle. Keep the seed-3409 recovery checkpoints until eval completes, then remove them if no rerun/debugging is needed.
- 14:15 CEST storage cleanup: reduced `$HPCVAULT` quota pressure from about `595G` to about `451G` immediately after deletion; final check was about `474G` because the active Nanotron prereq job kept writing data. Deleted the inactive re-downloadable Qwen3-32B HF cache and matching `$WORK` symlink, ten completed-run intermediate 32B `checkpoint-*` dirs whose parents have `final/` artifacts, repo-local W&B logs/caches, and tiny smoke/temp dirs under `$HPCVAULT/synthetic-RLVL`. Under `$WORK/synthetic-RLVL/runs`, deleted `144` old `checkpoint-*` dirs whose parents have `final/` artifacts plus local W&B dirs; that run tree dropped from about `141G` to `59G` and now has zero `checkpoint-*` dirs. Left active OLMo-3-1125-32B cache/offload, active conditioned-dual seed-3409 checkpoints, active Nanotron Qwen2.5 prereq outputs, final adapters, datasets, eval outputs, envs, and other-project directories untouched.
- 12:38 CEST Nanotron unblock: added resumable token-budget exporters `scripts/data/export_hf_text_jsonl_token_budget.py` and `scripts/data/export_generated_proof_jsonl.py`, plus prereq builder `scripts/slurm/jobs/nanotron_qwen25_build_prereqs_2026-06-29.slurm`. Patched local `../nanotron/tools/preprocess_data.py` for the installed `datatrove` `DocumentTokenizer(shuffle_documents=...)` signature and patched `scripts/slurm/jobs/nanotron_qwen25_pretrained_realdata_smoke_2026-06-22.slurm` to invoke the HF-to-Nanotron converter as `python -m examples.llama.convert_hf_to_nanotron`; the local Nanotron compatibility patch is committed in `../nanotron` as `fad45309`. Validation passed: exporter `py_compile`, Slurm `bash -n`, `sbatch --test-only`, `git diff --check`, tiny logic/NL/FineWeb export, and tiny Nanotron JSONL preprocessing to `.ds/.index/.metadata`. Submitted prereq job `3795206` on flexible single-GPU partitions; it started immediately on A40 node `a0323`. Released midtraining array `3776105_[0-10%1]` from `JobHeldUser` and set `Dependency=afterok:3795206`, so it is now normal dependency-pending.
- 12:38 CEST Nanotron data sizing decision: the existing HFSA Hub materialized train subsets are only `50k` rows, which is too small for a no-repeat 25% proof mixture in the planned `4.29B` token Nanotron runs. The prereq builder therefore generates fresh HFSA proof corpora to about `1.2B` Qwen tokens each for compact logic and exact NL, and uses `HuggingFaceFW/fineweb-edu` `sample-10BT` for about `4.8B` Qwen tokens of normal continuation text. `allenai/dolma` was not used because the current `datasets` version rejects its dataset script path; FineWeb-Edu streaming loaded successfully.
- 11:33 CEST live recovery/results update: normal OLMo-3/Qwen3 32B proof-chain eval `3775864_[0-11%1]` completed all `12/12` pass@k JSONs and sample JSONLs by 2026-06-28. Added summary artifact `analysis/logic_cot_report_2026-05-25/tables/hfsa_model_ablation_32b_train25_summary.csv`. Initial means: OLMo-3-32B logic OOD/depth-50 correct@16 `0.954/0.917`, citation-free joint `0.477/0.115`, strict grounded joint `0.000/0.000`; OLMo-3-32B NL correct `0.685/0.427`, translated joint `0.546/0.208`; Qwen3-32B logic correct `0.871/0.594`, citation-free joint `0.065/0.010`, strict grounded joint `0.000/0.000`; Qwen3-32B NL correct `0.846/0.802`, translated joint `0.838/0.792`. Sample inspection found intended wrappers and answer extraction, but OLMo logic often omits strict citations and Qwen3 logic often emits invalid lowercase predicate handles such as `aa`/`ab`, so this is a diagnostic result rather than clean formal-validity support.
- 11:33 CEST conditioned-dual 32B recovery: recovery row `3775861_2` timed out after 24h at about step `9861/10000` with checkpoints through `checkpoint-9000`, leaving old eval `3775868` in `DependencyNeverSatisfied`. Canceled `3775868`, submitted single-row resume `3795088_[2%1]` with 8h walltime, `SAVE_STEPS=500`, and no startup stagger; submitted replacement eval `3795089_[0-5%1]` with `afterok:3795088`. `3795088_2` started immediately on `a0532`.
- 11:33 CEST wrapper hygiene: patched `scripts/slurm/sweeps/sft/hfsa_conditioned_dual_olmo32_2026-06-19.slurm` so `STARTUP_JITTER_SECONDS=0` no longer triggers a bash modulo-by-zero warning in future targeted recoveries. The already-running `3795088_2` emitted the warning before the patch but continued into model loading.
- 11:33 CEST Nanotron prerequisite check: production Qwen2.5 midtraining chain remains held/dependency-pending. Missing prerequisites are still `$HPCVAULT/synthetic-RLVL/nanosets/qwen25/{normal_continuation,logical_deduction_logic,logical_deduction_nl_exact}` and `$HPCVAULT/synthetic-RLVL/nanotron_checkpoints/qwen25_7b_tp1`; do not release `3776105` until these are built and task registry checks are done.

## 2026-06-25

- 09:34 CEST live job update: normal 32B baseline recovery is progressing. Original row `3771012_9` completed cleanly, targeted row-8 recovery `3775860_8` completed cleanly, original row `3771012_10` is running, and original row `3771012_11` remains array-pending. Conditioned-dual recovery row `3775861_1` completed cleanly and row `3775861_2` is running. Replacement evals `3775864` and `3775868` remain dependency-pending. Qwen2.5 midtraining array `3776105` remains held; prerequisite check still reports missing `$HPCVAULT/synthetic-RLVL/nanosets/qwen25/{normal_continuation,logical_deduction_logic,logical_deduction_nl_exact}` and `$HPCVAULT/synthetic-RLVL/nanotron_checkpoints/qwen25_7b_tp1`.

## 2026-06-24

- 12:11 CEST Qwen2.5 midtraining orchestration: added Qwen2 Nanotron-to-HF conversion/upload tooling and Slurm wrappers for final-checkpoint proof-mixture midtraining plus downstream evaluation. Submitted training array `3776105_[0-10%1]` in held state, with dependent HF push `3776106`, direct downstream eval `3776107`, UltraChat instruction SFT `3776108`, and instruction downstream eval `3776109` behind `aftercorr` dependencies. The grid is `0%` control plus `{logic,nl_exact} x {5,10,15,20,25}` at 8192 steps, seq4096, TP=4/DP=2, microbatch `4`, grad accumulation `16`, about `4.29B` tokens/run. Validation passed for `bash -n`, converter `py_compile`, and Slurm `--test-only`; HF token is present. Release blockers recorded: missing Qwen-tokenized normal/logic/NL Nanoset roots and missing `$HPCVAULT/synthetic-RLVL/nanotron_checkpoints/qwen25_7b_tp1`; keep `3776105` held until these exist and reviewer lm-eval tasks are validated.
- 11:39 CEST figure intentionality pass: refactored `../synthetic-RLVL-report/official_preprint/scripts/build_preprint_figures.py` so supporting figures use claim-level titles, a more colorblind-friendly palette, distinct marker shapes, direct line labels where useful, and seed dots instead of shaded bands for raw three-seed views. The architecture figure now uses point intervals instead of ceiling-clipped bars, and the hybrid/conditioned-dual panels are simplified to endpoint summaries. Tightened remaining dense abstract/contribution wording in `../synthetic-RLVL-report/main.tex` and renamed the architecture subsection to `The Pattern Repeats Across Model Families`. Static checks pass: 9 figure refs, 0 missing figures, 0 non-PDF result figure refs, 0 PNG result files, 10 refs, 20 citations, 7 semicolons only in formal examples, 0 `--` ranges, no targeted stale phrases in the canonical report/script, figure-script `py_compile`, and `git diff --check`. Local visual rendering and TeX compilation remain unavailable because no PDF renderer or TeX engine is installed.
- 11:30 CEST quota cleanup: removed only clearly disposable repo artifacts to alleviate `$WORK`/`$HPCVAULT` pressure. Deleted Nanotron smoke/probe checkpoint payloads, stale merged eval dir `merged_sft_hfsa_conditioned_dual_train1to5_50k_seed3409_conditioned_nl`, canceled OLMo-3-1025-7B cache, pip cache, and old completed experiment-specific HF caches for hard attribute, batch-size, semantic iGSM, and typed maze. `$HPCVAULT` quota usage dropped from about `1975G` to about `1204G`; `$WORK/synthetic-RLVL` is about `168G`, `$HPCVAULT/synthetic-RLVL` about `41G`, `$HPCVAULT/cache_offload` about `243G`, and `$WORK/nanotron` about `7.0G`. Left active final adapters, eval outputs, datasets, current 32B Qwen3/OLMo cache-offload symlinks, and active envs in place.
- 11:23 CEST report voice polish: revised `../synthetic-RLVL-report/main.tex` and the official preprint figure builder to remove remaining stock manuscript rhythm. The title is now `Formal Traces Improve Length Generalization in Symbolic Chain-of-Thought`; the abstract now ends with the claim rather than meta phrasing; result/analysis subsection titles are shorter; key paragraphs now open with claims instead of "Table/Figure shows" transitions; Figure 1 panel titles now state insights; prose semicolons and LaTeX `--` ranges were removed from the main report. Static checks pass: no targeted stock phrases in the canonical report/script, main report semicolons are down to 7 formal grammar/proof-example uses, 0 `--` ranges, 0 PNG result files, 9 figure refs with 0 missing/non-PDF result refs, 10 refs, 20 citations, figure-script `py_compile`, and `git diff --check`. Local TeX compilation remains unavailable because no TeX engine is installed.
- 10:39 CEST report polish: revised `../synthetic-RLVL-report/main.tex` and `official_preprint/scripts/build_preprint_figures.py` after the latest review. The first-page overview now uses provenance-clean excerpts from `sample_generation_snippets.csv`, all result figures are PDF-only and the figure builder no longer writes PNG sidecars, high-variance AttrCon language is softened, the mechanism and hybrid claims no longer assert untested causes, and the appendix now includes formal grammar, generator/evaluator mechanics, and a minimal paired example while keeping the preprint scoped to `BranchProof`/`AttrCon`. Static checks pass: 9 figure refs, 0 missing figures, 0 non-PDF result figure refs, 0 PNG result files, 10 refs, 20 citations, no targeted stale phrases in the canonical report/script, `py_compile` for the figure script, and `git diff --check`. Local TeX compilation remains unavailable because no TeX engine is installed.
- 10:25 CEST live recovery: normal 32B row `3771012_8` and conditioned-dual row `3771013_2` failed before optimizer progress because Hugging Face dataset-lock creation under `$WORK/.cache/hf/datasets` hit quota. Conditioned-dual row `3771013_1` timed out after 24h at about step `9972/10000`, with only `checkpoint-5000` present. Cleaned quota pressure by deleting safe completed-run intermediate checkpoints, removing reproducible local caches, and moving large HF hub model cache directories (`Qwen3-32B`, `Olmo-3-1125-32B`, `Olmo-3-1025-7B`) to `/home/vault/c107fa/c107fa12/cache_offload/hf_hub` with symlinks restored under `$WORK/.cache/hf/hub`. `$WORK/.cache/hf/datasets` write probe now passes.
- 10:25 CEST targeted resubmissions: canceled stale evals `3771014` and `3771015`. Submitted normal row-8 recovery `3775860_[8%1]`, conditioned-dual row-1/2 recovery `3775861_[1-2%1]` with `SAVE_STEPS=1000,SAVE_TOTAL_LIMIT=5`, replacement normal eval `3775864_[0-11%1]` with dependency `afterany:3771012,afterok:3775860`, and replacement conditioned-dual eval `3775868_[0-5%1]` with dependency `afterok:3775861`. Current running SFT is normal row `3771012_9`; rows `3771012_[10-11%1]` remain array-pending. Watch `3775864`: because Slurm rejected fine-grained array-task dependencies, it must be canceled/reissued if original rows `3771012_9..11` fail.

## 2026-06-23

- 17:37 CEST report hardening: revised `../synthetic-RLVL-report/main.tex` using the NeurIPS-style critique as an objection map. Added `official_preprint/figures/overview_claim.pdf` as the first claim figure, moved AttrCon/syntax/architecture/hybrid figures into the body, expanded Related Work, added SFT/eval/sample-count details, added uncertainty to shortcut/syntax/appendix tables, caveated high-variance AttrCon and architecture checks, removed the weak terse-NL control from main evidence, and expanded the appendix. Replaced stale `official_preprint/main.tex` with an archival placeholder. Static checks pass for figure paths, refs, citations, report terminology, figure-script `py_compile`, and `git diff --check`; local TeX compilation remains unavailable because no TeX engine is installed.
- 17:07 CEST report rewrite: rewrote `../synthetic-RLVL-report/main.tex` into a conventional NeurIPS-style paper with Introduction, Related Work, Method, Experimental Setup, Results, Analysis, Discussion, Limitations, Conclusion, and appendix sections. The main narrative now centers only `BranchProof` and `AttrCon`, replaces ambiguous depth terminology with `training range`, `long-depth band`, `hard-depth band`, and `depth-50 endpoint`, and uses claim-specific tables plus three focused figures. Regenerated the official preprint figure labels. Static checks pass for banned report terms, figure paths, references, citations, and `git diff --check`; local TeX compilation remains unavailable because no TeX engine is installed on this node.
- 15:28 CEST live job update: older carried-over SFT rows `3770814` (normal 32B proof-chain baseline) and `3768361` (OLMo-3-32B conditioned-dual) completed cleanly. Current active SFT children are normal 32B `3771012_7` running since 10:25 CEST and conditioned-dual `3771013_1` running since 2026-06-22 21:56 CEST; remaining rows `3771012_[8-11%1]` and `3771013_[2%1]` are pending by array task limit. Evals `3771014` and `3771015` remain dependency-pending.
- 15:28 CEST Nanotron update: Qwen2.5 random-init batch probe `3771016_[0-4%1]` completed rows for microbatches `1/2/4`; microbatches `8/16` OOMed on A100-80GB at seq len `4096` under TP=4/DP=2, so the current safe envelope is microbatch `<=4` unless recompute/sequence length/parallelism changes. Pretrained real-data smoke `3771017` exported and debug-printed the intended packed dataset mixture (4 normal, 16 logic, 16 NL records); the printed logic/NL samples and packed chunk look structurally correct. It failed before training because `examples/llama/convert_hf_to_nanotron.py` was invoked as a script and hit `ImportError: attempted relative import with no known parent package`. Fix the converter invocation before resubmitting the smoke.

## 2026-06-22

- 20:28 CEST resubmitted the paused repo jobs again. New jobs: normal 32B SFT recovery `3771012_[6-11%1]`, normal 32B eval `3771014_[0-11%1]` with `afterok:3770814:3771012_*`, conditioned-dual 32B SFT recovery `3771013_[1-2%1]`, conditioned-dual eval `3771015_[0-5%1]` with `afterok:3768361:3771013_*`, Qwen2.5 random-init Nanotron probe `3771016_[0-4%1]`, and Qwen2.5 pretrained real-data smoke `3771017`. Slurm currently reports `START_TIME=N/A` for all newly submitted non-dependent rows; evals are dependency-pending.
- 20:10 CEST canceled the six jobs that were briefly resubmitted at 19:50 CEST, at the user's request: `3770854`, `3770855`, `3770856`, `3770857`, `3770858`, and `3770859`. `sacct` reports all six as `CANCELLED` with `00:00:00` elapsed, so none started or wrote useful outputs. Left older running rows `3770814` and `3768361` untouched. Handoff docs now mark the recovery/eval/Nanotron jobs as paused until the user says to resubmit.
- 19:50 CEST resubmitted the queue-relief cancellations after the user finished trying to move other jobs forward. New jobs: normal 32B SFT recovery `3770854_[6-11%1]`, normal 32B eval `3770856_[0-11%1]` with `afterok:3770814:3770854_*`, conditioned-dual 32B SFT recovery `3770855_[1-2%1]`, conditioned-dual eval `3770857_[0-5%1]` with `afterok:3768361:3770855_*`, Qwen2.5 random-init Nanotron probe `3770858_[0-4%1]`, and Qwen2.5 pretrained real-data smoke `3770859`. Slurm currently estimates first non-dependent starts at 2026-06-23 00:32/04:44/05:14/05:37 CEST for `3770854`/`3770858`/`3770859`/`3770855`; evals are dependency-pending with no ETA.
- 19:40 CEST queue relief: canceled pending `synthetic-RLVL` jobs `3758372_[6-11]`, `3758373`, `3768259_[1-2]`, `3768260`, `3768359`, and `3768374` so the user can submit other repo jobs closer in the queue. Left already-running rows untouched: normal 32B SFT row `3758372_5` (raw job `3770814`) and OLMo-3-32B conditioned-dual row `3768259_0` (raw job `3768361`). Qwen2.5 Nanotron probes `3768359` and `3768374` never started and wrote no logs. Resume commands and dependency notes are recorded in `docs/running_experiments.md` and `docs/experiment_backlog.md`.
- 09:35 CEST integrated Qwen real-data smoke: added and submitted `scripts/slurm/jobs/nanotron_qwen25_pretrained_realdata_smoke_2026-06-22.slurm` as job `3768374`. This job exports a tiny weighted normal/logic/NL proof-chain JSONL, prints raw examples and decoded packed chunks under the Qwen2.5 tokenizer, converts `Qwen/Qwen2.5-7B` to a Nanotron checkpoint under `$HPCVAULT` if needed, then trains two packed real-data steps from that checkpoint with optimizer/lr-scheduler loading disabled. The smoke uses conservative TP=1, DP=8, seq len `1024`, microbatch `1`; the separate random-init probe `3768359` remains the TP=4 max-batch check. Local Nanotron converter was patched to use `AutoModelForCausalLM` and instantiate the Qwen2 Nanotron class for Qwen conversion.
- 09:20 CEST Nanotron target decision: canceled pending OLMo-3-7B-shaped proxy `3768322` before start. Qwen3 and OLMo3 both have extra q/k attention norms not represented by this Nanotron checkout's native model paths, while `Qwen/Qwen2.5-7B` matches the native Qwen2 path much more closely. Submitted full-node Qwen2.5-7B random-init batch probe `3768359_[0-4%1]` on 8xA100-80GB with seq len `4096`, TP=4, DP=2, and micro-batches `1/2/4/8/16`. For midtraining mixtures, OLMo/Dolma continuation text can still be used, but it must be re-tokenized with the Qwen2.5 tokenizer before Nanoset mixing with logic/NL proof corpora.
- 09:05 CEST Nanotron feasibility: `$WORK/nanotron` imports `torch 2.6.0+cu124`, `flash_attn 2.7.4.post1`, and `nanotron`. Initial dense launch attempts failed on optional `grouped_gemm` and a Llama initializer mismatch; local Nanotron patches now bypass optional MoE `grouped_gemm` registration when unavailable/broken and pass the full config to the Llama parametrizer, matching Qwen. Tiny 4-GPU dummy-data smoke `3768319` completed 3 train steps and saved a checkpoint. Submitted 8xA100-80GB OLMo-3-7B-shaped random-init memory probe `3768322_[0-4%1]` with seq len `4096`, TP=8, and micro-batches `1/2/4/8/16`. Caveat: this is not true OLMo3 checkpoint support; native/compatible OLMo3 model/conversion remains required before real base `allenai/Olmo-3-1025-7B` midtraining.

## 2026-06-19

- 11:35 CEST official preprint revision: pulled user report-repo edits (`b73995d`) before changing the draft. Rewrote root `../synthetic-RLVL-report/main.tex` to a more NeurIPS-style structure with Introduction, Related Work, Experimental Setup, Results, Discussion, Limitations, Conclusion, and Appendix. The draft now uses descriptive task names ("branching proof chains" and "attribute constraints"), removes paragraph headings that caused double-dot rendering, replaces bar-plot-style evidence with tables, keeps only two line figures where the trend is the claim, adds main/architecture/shortcut/syntax/integrity/hybrid/appendix tables, and includes the OLMo-2-32B short-context sanity rows with an explicit non-OOD caveat. Static checks: `8` tables, `2` figure refs, zero missing figures/citations, zero `\paragraph` commands, and `git diff --check` passes for the report repo. Local TeX compilation remains unavailable.
- 11:10 CEST Overleaf report layout: changed `../synthetic-RLVL-report/main.tex` to the official preprint entrypoint so it renders by default in Overleaf. Moved the older generated informal report to `../synthetic-RLVL-report/informal_report/main.tex`, added `\graphicspath{{../}}` so its root `figures/` references continue to resolve, and updated the report README with the layout plus the Overleaf caveat that compiling a different open tab still requires Menu -> Main document. Static checks found zero missing figure refs for both preprint and informal report and zero missing citation keys for the preprint.
- 10:51 CEST official preprint setup: pulled report-repo commit `6c7ba9c` with the new `hu_new_gen_template/`, kept the old generated `../synthetic-RLVL-report/main.tex` as the informal report, and created `../synthetic-RLVL-report/official_preprint/` with template assets/styles, a new official preprint draft, bibliography, and reproducible figure script. The draft uses `BranchProof` and `AttrCon` as concise dataset names, excludes iGSM/Maze from the main story, frames the claim around correctness rather than proof faithfulness, and integrates hybrid-order plus conditioned-dual as negative-but-informative evidence. Generated eight preprint-specific figures (`main_correctness`, `attribute_correctness`, `shortcut_robustness`, `syntax_controls`, `trace_integrity`, `hybrid_order`, `conditioned_dual`, `architecture_depth50`) with mean/std bands or error bars where seed-level data are available. Static checks found zero missing figure paths and zero missing citation keys; TeX compilation was not run because this node has no available TeX engine.
- 09:30 CEST final paired recovery/report refresh: typed-maze final recovery `3748683_[15-29%15]` completed cleanly on 2026-06-18, bringing typed maze to `30/30` JSONs/sample JSONLs. A live queue check found no remaining `maze_typed_eval` or hard-attribute recovery rows. Regenerated `analysis/logic_cot_report_2026-05-25/` after the final rows landed and patched report-builder prose so the report no longer describes fresh paired readouts as partial.
- 09:30 CEST typed-maze result: the typed-symbol fix did not rescue maze. Logic train-1-to-25 OOD/depth-50 joint@16 is `0.000/0.000`; NL train-1-to-25 OOD/depth-50 correct@16 is `0.111/0.000`; NL translated validity remains unsupported for maze. Sample inspection across train-1-to-25 logic/NL rows shows valid shallow traces, but depth-25/50 logic generations spend the budget on constants/premises/partial derivations and omit `<answer>`, while NL depth-25/50 copies premise chains through roughly move `18..20` and also omits `<answer>`. Treat this as a negative result/generation-budget failure mode rather than a typed-symbol evaluator bug.

## 2026-06-18

- 08:50 CEST live recovery/report refresh: hard-attribute final recovery `3748682_[27-29%3]` completed cleanly, bringing fresh hard attribute to `30/30` JSONs/sample JSONLs. Typed-maze final recovery `3748683_[15-29%15]` advanced to `27/30`; only rows `27..29` (`nl_exact` train-1-to-25 seeds) remain running on A100, all actively sampling around chunks `109..110/112` with only known tokenizer/torch warnings.
- 08:50 CEST result readout: report refresh marks semantic iGSM `30/30`, hard attribute `30/30`, typed maze `27/30`, and all HFSA ablations complete. Hard-attribute logic OOD joint@16 rises with train max (`0.108/0.455/0.500/0.737/0.736`) but depth-50 joint remains `0.000`; hard-attribute NL train-1-to-25 OOD/depth-50 correct@16 is `0.817/0.104`, with translated validity still unsupported (`nl_logic_parse=0`). Typed maze remains poor after typed symbols: logic OOD/depth-50 joint is `0.000` through train-1-to-25, completed NL train-1-to-5/10/15/20 OOD correct@16 is `0.030/0.106/0.111/0.111`, and completed depth-50 rows are `0.000`.
- 08:50 CEST report/handoff update: regenerated `analysis/logic_cot_report_2026-05-25/` and updated `active_experiment_artifact_status.csv`, `active_paired_partial_summary.csv`, and `active_paired_partial_summary.{pdf,png}`. Handoff docs now record hard attribute as complete and typed maze as last-three-running.

## 2026-06-17

- 09:08 CEST live recovery check: hard-attribute final recovery `3748682_[27-29%3]` has started. Rows `27` and `28` are running on A100, with row `27` actively sampling around chunk `68/112`; row `29` remains priority-pending. No new hard-attribute JSONs yet, so coverage remains `27/30`. Typed-maze final recovery `3748683_[15-29%15]` is still priority-pending, so typed-maze coverage remains `15/30`. No fatal log signatures were seen in the running hard-attribute rows.

## 2026-06-16

- 17:55 CEST scheduler edit: raised typed-maze final recovery array throttle from `3748683_[15-29%3]` to `3748683_[15-29%15]` with `scontrol update JobId=3748683 ArrayTaskThrottle=15`. Hard-attribute final recovery `3748682_[27-29%3]` is already at maximum useful concurrency because only three rows remain. Both arrays are pending on A100 for priority; no new artifacts were written by this scheduler edit.
- 17:37 CEST live recovery/report refresh: recovery arrays `3743047` and `3743048` were canceled at 16:07 CEST after writing additional outputs. Typed maze advanced from `9/30` to `15/30` JSONs (logic train-1-to-15 complete, NL train-1-to-10 complete); hard attribute advanced from `21/30` to `27/30` JSONs (all logic rows complete, NL through train-1-to-20 complete). Submitted final targeted recoveries `3748683_[15-29%3]` for typed maze and `3748682_[27-29%3]` for hard attribute. Both are priority-pending on A100. Regenerated the report; active status now records typed maze `15/30`, hard attribute `27/30`, and batch-size still complete at `16/16`.
- 17:37 CEST partial result readout: hard-attribute logic OOD joint@16 improves with train max (`0.108/0.455/0.500/0.737/0.736` for train max `5/10/15/20/25`) but depth-50 joint remains `0.000`. Hard-attribute NL answer correctness remains useful through train-1-to-20, but family-specific translated validity is still unsupported (`nl_logic_parse=0`). Typed maze remains weak after typed symbols: completed logic rows through train-1-to-15 have OOD/depth-50 joint `0.000`, and completed NL train-1-to-5/10 rows have OOD correct@16 `0.030/0.106` with depth-50 `0.000`.

## 2026-06-15

- 09:41 CEST ablation/report deepening: confirmed all HFSA ablation families are complete and report-ingested: trace controls `18/18`, shortcut-rate `18/18`, shortcut-kind `24/24`, hybrid order `30/30`, conditioned dual 10k/final 50k/checkpoint 50k, wordified/symbol-padded/length controls, semantic iGSM, and batch-size `16/16`. Patched `scripts/analysis/build_logic_cot_report.py` to add `hfsa_batch_size_ablation_diagnostics.csv`, `hfsa_batch_size_conditioned_delta.{pdf,png}`, and updated LaTeX prose/captions plus sample-backed interpretation for batch-size. Key insight: the one-seed batch-size diagnostic rejects a simple monotone "larger stratified batch fixes conditioned dual" story; conditioned-NL is best at bsz2 and conditioned-logic is non-monotonic. Added optional backlog item for a two-seed batch-size replication if this becomes a central causal claim, but did not submit broad new science while paired recoveries are active.
- 09:16 CEST live status/recovery: checked required handoff docs, `squeue`, `sacct`, and active artifact roots. Batch-size eval `3722467` completed cleanly at `16/16` JSONs. Typed maze eval `3722471` reached `9/30` JSONs before cancellation at 2026-06-14 10:38 CEST; submitted targeted missing-row recovery `3743047_[9-29%3]`. Hard-attribute eval `3716216` plus row-1 recovery `3739163` reached `21/30` JSONs before the original array cancellation at 2026-06-14 10:38 CEST; submitted targeted missing-row recovery `3743048_[21-29%3]`. Both recoveries are pending on A100 priority. Visible `babylm-*` and `lewm_sudoku_*` jobs are unrelated.
- 09:16 CEST report/result refresh: regenerated `analysis/logic_cot_report_2026-05-25/` after new batch-size, typed-maze, and hard-attribute artifacts. `active_experiment_artifact_status.csv` now records batch-size `12/12` SFT and `16/16` eval, typed maze `9/30`, and hard attribute `21/30`. Batch-size result does not support a simple larger-stratified-batch fix for conditioned dual: conditioned-NL is best at bsz2 on OOD/depth-50 joint (`0.781/0.344`), and conditioned-logic is non-monotonic with OOD joint best at bsz2 (`0.618`) and near-tied at effective bsz16 (`0.587`). Typed maze remains poor on completed rows, while hard-attribute logic OOD joint improves through train-1-to-20 but depth-50 joint remains `0.000`.

## 2026-06-13

- 14:25 CEST live last-24h Slurm audit: active-plan completions since 2026-06-12 14:00 CEST include hard-attribute eval rows that brought fresh hard attribute to `13/30` JSONs, batch-size bsz16 SFT recovery rows `3722466/3722469` that completed SFT at `12/12`, batch-size eval rows that wrote all logic bsz `2/4/8/16` evals plus `nl_exact` bsz2 (`5/16` JSONs), and typed-maze eval rows `3722472/3722473/3722474/3728636/3728638/3728648` that wrote the full train-1-to-5 logic/NL block (`6/30` JSONs). No new active-plan failure appeared in the exact last-24h accounting window; visible `babylm`, `puzzle`, and `grid5` failures are unrelated to this repo handoff. Running active rows are batch-size eval `3722467_5/6/7/8`, typed maze `3722471_6/7/8`, hard attribute `3716216_14/15/16`, and hard-attribute row-1 recovery `3739163_1`.
- 14:11 CEST recovery action: submitted targeted hard-attribute row-1 recovery `3739163_[1%1]` with `PASSK_MAX_NEW_TOKENS=8192` and output subdir `paired_attribute_constraints_hard_full_20260610` to recover original `3716216_1` (`logic`, train-1-to-5, seed `3408`), which had timed out earlier after many `12288`-token generations. This cap is an interpretation caveat for that recovered seed.
- 14:25 CEST result analysis/report refresh: patched `scripts/analysis/build_logic_cot_report.py` to aggregate active typed-maze, hard-attribute, and HFSA batch-size partials across repo-local and `$HPCVAULT` pass@k roots; regenerated `analysis/logic_cot_report_2026-05-25/` and mirrored it to `../synthetic-RLVL-report`. New artifacts include `active_paired_partial_by_seed.csv`, `active_paired_partial_summary.csv`, `hfsa_batch_size_ablation_partial_by_seed.csv`, `hfsa_batch_size_ablation_partial_summary.csv`, `active_paired_partial_summary.{pdf,png}`, and `hfsa_batch_size_ablation_partial.{pdf,png}`. Bundle counts after regeneration: `72` PDF figures and `73` CSV tables; TeX compilation remains unavailable.
- 14:25 CEST sample-backed interpretation: typed maze remains bad despite typed symbols (`logic` train-1-to-5 OOD correct/joint@16 `0.002/0.000`, `nl_exact` OOD correct@16 `0.030`, depth-50 `0.000`), with shallow surfaces correct but longer generations copying/drifting/truncating or omitting answers. Hard attribute is more promising at OOD but not depth-50: logic OOD joint@16 is `0.139` at train-1-to-5 (`n=2`), `0.455` at train-1-to-10 (`n=3`), and `0.540` at train-1-to-15 (`n=2`), while depth-50 joint remains `0.000`; NL answer correctness reaches OOD correct@16 `0.801` at train-1-to-10 but translated validity is unsupported (`nl_logic_parse=0`). Batch-size partials include all logic rows plus `nl_exact` bsz2; logic bsz `2/4/8/16` OOD joint@16 is `0.406/0.562/0.403/0.583`, `nl_exact` bsz2 OOD correct/translated-joint@16 is `0.569/0.503`, and conditioned-dual batch conclusions remain deferred until conditioned rows finish.

## 2026-06-12

- 17:05 CEST live recovery/report refresh: batch-size bsz16 recovery `3722466_7/11` is still running from `checkpoint-8000` at about `8716/10000` and `8729/10000`; dependent eval `3722467_[0-15%4]` remains blocked on `afterok:3722466_*` with `0` JSONs. Typed maze replacement eval `3722471_0/1/2` is running at chunks `78/77/76` of `112` with the new formal `4096`-token cap, much faster than the canceled high-cap run. Hard-attribute eval `3716216` advanced to `8/30` JSONs/sample JSONLs: train-1-to-10 logic is now three-seed complete with OOD correct@16 `0.716-0.912` and OOD joint@16 `0.425-0.578`, but depth-50 joint is still `0.000`; completed NL rows still have `nl_logic_parse@16 = 0`, so hard-attribute NL validity is unsupported. Regenerated `analysis/logic_cot_report_2026-05-25/` and mirrored it to `../synthetic-RLVL-report`; `active_experiment_artifact_status.csv` now records hard attribute `8/30`, typed maze eval running, and batch-size SFT recovery running.
- 10:08 CEST live recovery/report refresh: conditioned-dual 50k final recovery `3716219` completed, bringing final eval to `30/30` JSONs/sample JSONLs. Regenerated `analysis/logic_cot_report_2026-05-25/` and mirrored it to `../synthetic-RLVL-report`; updated active artifact status now shows conditioned-dual 50k complete, hard attribute `5/30`, typed maze replacement running, and batch-size recovery running. Final train-1-to-25 conditioned-dual means: `conditioned_logic` OOD/depth-50 correct@16 `0.833/0.677`, joint@16 `0.348/0.146`; `conditioned_nl` OOD/depth-50 correct@16 `0.675/0.531`, translated-joint@16 `0.531/0.250`. Samples preserve the conditioned prompts, so the result is not a prompt-mismatch artifact; batch-size ablation remains the direct follow-up.
- 10:08 CEST scheduler actions: canceled stale batch-size eval `3715330`, submitted targeted bsz16 resume recovery `3722466_[7,11%2]` from `checkpoint-8000`, and submitted dependent eval `3722467_[0-15%4]` with `afterok:3722466_*`. Typed maze high-cap eval `3705795` had `0/30` JSONs after rows `0..2` timed out, `3..5` node-failed, and `6..8` were on track to miss walltime around chunks `98..100/112`; lowered the typed-maze formal eval cap from `8192` to `4096` tokens while leaving NL at `6144`, canceled `3705795`, and submitted replacement eval `3722471_[0-29%3]`.
- 10:08 CEST hard-attribute partial readout: fresh eval `3716216` has `5/30` JSONs. Completed train-1-to-5 logic seeds `3407/3409` have OOD correct@16 `0.878/0.903`, OOD citation-free joint@16 `0.199/0.276`, and depth-50 correct/joint@16 `0.094/0.000` and `0.000/0.000`; logic seed `3408` timed out. Completed NL seeds have OOD correct@16 about `0.47-0.56` and depth-50 correct@16 `0.000`, but `nl_logic_parse@16` is still `0`, so hard-attribute NL validity is unsupported until a family-specific translator is added.

## 2026-06-11

- 08:14 CEST active recovery oversight/report refresh: checked required handoff/research/report docs, `squeue`, row-level `sacct`, `scontrol show job`, A100 partition state, primary logs, output roots, bsz16 checkpoints, and representative conditioned-dual samples. Typed maze eval `3705795_0/1/2` is still running around chunks `95/112`, `93/112`, and `92/112`, with `0` JSONs and walltime risk. Bsz16 recovery is still `10/12` finals: `3715329_3` is complete, while `3715329_7/11` are running around `8139/10000` and `8253/10000` with latest `checkpoint-8000` and no finals; `3715330_[0-15%4]` remains dependency-pending on `afterok:3715329_*` and the eval root is absent. Conditioned-dual recovery rows `3716219_17/21/22` completed, advancing final eval to `23/30`; rows `23/24/25` are running and rows `26..29` are throttle-pending. Sample inspection of the new conditioned-NL rows confirmed intended natural-language prompts, `<think>/<answer>` surface, and normal answer extraction; raw validity remains zero because formal validation cannot parse natural-language proof lines. Regenerated `analysis/logic_cot_report_2026-05-25/` with `PYTHONPATH=.` and mirrored it to `../synthetic-RLVL-report`; both trees have `70` PDF figures and `69` CSV tables. Hard-attribute eval rows `3716216_0/1/2` are still running around chunks `87/69/105` with `0` JSONs. Focused log/GPU scans found no unrecovered Traceback, CUDA OOM, quota/no-space, dependency-never-satisfied, node failure, timeout, cancellation, vLLM failure, or idle-GPU symptom; no partition widening, cancellation, resubmission, or new oversight scheduling was done. Created local commits in this repo and `../synthetic-RLVL-report`; normal SSH pushes to `git@github.com:22` and fallback pushes to `ssh.github.com:443` both timed out with no output, so both repos remain locally ahead of `origin/main`.
- 04:12 CEST active recovery oversight/report refresh: checked required handoff/research/report docs, `squeue`, row-level `sacct`, `scontrol show job`, A100 partition state, active logs, output roots, and bsz16 checkpoints for typed maze `3705795`, bsz16 recovery/eval `3715329/3715330`, hard attribute `3716216`, and conditioned-dual recovery `3716219`. Typed maze eval rows `3705795_0/1/2` are still running at about chunks `84/112`, `82/112`, and `81/112`, with `0` JSONs and continued walltime risk. Bsz16 SFT is now `10/12` finals after `3715329_3` completed; rows `3715329_7/11` are running around `7709/10000` and `7794/10000`, with latest `checkpoint-7000` and no finals. Edited batch-size eval `3715330` from `afterany:3715329_*` to `afterok:3715329` so it cannot release if rows `7/11` time out without finals. Conditioned-dual rows `3716219_18/20` completed and advanced final eval to `20/30`; report-refresh samples preserve the conditioned-NL prompt/surface, with depth-50 failures from premise-copying, truncation, missing answers, and parse failures. Regenerated and mirrored the report; train-1-to-10 `conditioned_nl` is now three-seed with OOD/depth-50 correct@16 `0.639/0.479` and translated-joint@16 `0.345/0.000`. Hard-attribute rows `3716216_0/1/2` are running around chunks `76/57/90` with `0` JSONs. Focused log scans found no unrecovered Traceback, CUDA OOM, quota/no-space, dependency-never-satisfied, node failure, timeout, cancellation, vLLM failure, or idle-GPU symptom; no partition widening, cancellation, resubmission, or new oversight scheduling was done.
- 04:22 CEST push attempt: local commits were created for this repo and `../synthetic-RLVL-report`, but pushing both repos failed because `git@github.com:22` and fallback `ssh://git@ssh.github.com:443/...` produced no output and hit the 120-second timeout. This repo remains ahead of `origin/main` locally, and the report repo remains ahead of `origin/main` locally.
- 00:04 CEST active recovery oversight: checked required handoff/research/report docs, `squeue`, row-level `sacct`, `scontrol show job`, A100 partition state, active logs, output roots, and bsz16 checkpoint dirs. Typed maze eval `3705795_0/1/2` is still running on `a0633` at about chunks `72/112`, `71/112`, and `69/112`, with many high-depth chunks hitting `max=8192` and `0` JSONs in `$HPCVAULT/synthetic-RLVL/passk_eval/paired_maze_typed_sparse_20260603/`. Bsz16 recovery `3715329_3/7/11` is still running on `a0537` at about `9581/10000`, `7276/10000`, and `7334/10000`; on-disk checkpoints are logic `checkpoint-9000`, NL `checkpoint-7000`, and conditioned-dual `checkpoint-7000`. Batch-size eval `3715330_[0-15%4]` remains dependency-pending on `afterany:3715329_*`, so it is not stale yet and its output root is absent. Hard-attribute eval `3716216_0/1/2` and conditioned-dual final recovery `3716219_17/18/20` are also running with no new JSONs. Focused log scans found no unrecovered Traceback, CUDA OOM, quota/no-space, dependency-never-satisfied, node failure, timeout, cancellation, vLLM failure, or idle-GPU symptom; observed tokenizer/vLLM/model-load lines are startup warnings. No partition edit, cancellation, resubmission, aggregation, report regeneration, or new oversight scheduling was done.

## 2026-06-10

- 20:18 CEST active recovery consolidation: canceled one-off oversight `3715440` after it completed its local audit/commit attempt but stayed alive waiting on a fallback GitHub push that had timed out; later one-off oversight jobs `3715441/3715442/3715443` remain begin-time pending. Hard-attribute eval `3716216_2` started, so fresh hard-attribute recovery is now running at the intended `%3` throttle with rows `0/1/2`; no JSONs yet. Conditioned-dual recovery `3716219` remains priority-pending, typed maze eval `3705795_0/1/2` and bsz16 recovery `3715329_3/7/11` continue running, and batch-size eval `3715330` remains dependency-pending. Updated handoff/backlog wording so hybrid-order is no longer listed as an unrefreshed report todo.
- 20:05 CEST active recovery oversight: live Slurm/accounting/log/output refresh found typed maze eval `3705795_0/1/2` still running on `a0633` with `0` JSONs; row `0` finished chunk `58/112` and is in `59/112`, row `1` finished `57/112` and is in `58/112`, and row `2` finished `56/112` and is in `57/112`, with high-depth chunks still often hitting `max=8192`. Bsz16 recovery `3715329_3/7/11` is still running on `a0537`; row `3` advanced to `checkpoint-9000` and about `9061/10000`, while rows `7/11` are around `6847/10000` and `6876/10000` with latest on-disk `checkpoint-6000`. Eval `3715330` remains dependency-pending on `afterany:3715329_*` and its output root is absent. Hard-attribute eval `3716216_0/1` is running with `0` JSONs; conditioned-dual recovery `3716219` is priority-pending; one-off oversight `3715440` is running. Focused log scans found no unrecovered severe signatures; no partition edit, cancellation, resubmission, aggregation, report regeneration, or new oversight scheduling was done.
- 20:12 CEST report refresh: regenerated `analysis/logic_cot_report_2026-05-25/` with `PYTHONPATH=$PWD` and mirrored it to `../synthetic-RLVL-report`. The report now treats hybrid-order as a completed `30/30` grid, conditions the 50k final table on the current `18/30` row coverage, updates the conditioned-dual checkpoint caption to reflect complete `30/30` checkpoint curves, adds qualitative interpretation for completed hybrid and conditioned checkpoint samples, and expands the active artifact-status table with semantic iGSM, typed maze, fresh hard attribute, and batch-size rows. Verification: report builder completed, `py_compile` passed, both report trees have `70` PDF figures and `69` CSV tables, and the LaTeX sources have `70` figure references with zero missing files. Local TeX compilation remains unavailable.
- 19:58 CEST ablation/constraint recovery: added hard `attribute_constraints` full-suite eval wrapper `scripts/slurm/jobs/posthoc_paired_attribute_constraints_hard_full_eval_2026-06-10.slurm`, validated `30/30` final adapter paths plus row mapping, and submitted fresh eval `3716216_[0-29%3]` under `$HPCVAULT`; rows `0/1` are running and rows `2..29` are pending. Patched `scripts/slurm/jobs/posthoc_hfsa_conditioned_dual_50k_eval_2026-05-29.slurm` so temporary merged checkpoints go under `$HPCVAULT`, then submitted targeted conditioned-dual final recovery `3716219_[17,18,20-29%3]` for the 12 missing final `conditioned_nl` rows. Artifact audit: trace-control `18/18`, shortcut-rate `18/18`, shortcut-kind `24/24`, wordified `3/3`, hybrid-order `30/30`, conditioned-dual checkpoint `30/30`, conditioned-dual final `18/30`, batch-size eval not created yet, typed maze `0/30`, hard attribute `0/30`.
- 19:55 CEST constraint/ablation status clarification: hard `attribute_constraints` / constraint-satisfaction full-suite SFT adapters exist, and a 2026-06-02 audit found the family semantically grounded with explicit slot/value symbols, so it does not need the iGSM or maze generator fixes. However, the current full-suite hard-attribute eval coverage is still `0/30`; only older one-seed train-10 pilot outputs exist under `paired_followup_train10_sparse` and `paired_attribute_constraints_hard_sparse`. If hard-attribute coverage is needed, submit a fresh hard-attribute-only eval under `$HPCVAULT` from existing final adapters instead of resubmitting stale combined recovery `3694619` unchanged.
- 19:42 CEST imminent-action check: typed maze eval `3705795_0/1/2` and bsz16 recovery `3715329_3/7/11` are still running, with no new eval JSONs. Active job inspection shows both families already have `TimeLimit=1-00:00:00`, matching the `a100` partition `MaxTime`, so there is no safe walltime extension to apply. No new submission is useful right now; the next action is to let scheduled oversight `3715440` run at 20:00 CEST and resubmit only rows that actually timeout/fail.
- 19:28 CEST live refresh: first scheduled active-recovery oversight `3715439` completed cleanly (`0:0`) after finding no unrecovered Traceback, CUDA OOM, quota/no-space, dependency, model-load, vLLM, idle-GPU, or new node-failure issue; its handoff commit `2f9e4a3` was pushed successfully on retry. Typed maze eval rows `3705795_0/1/2` are still running on `a0633`, now around chunks `57/112`, `56/112`, and `55/112`, with `0` JSONs in `$HPCVAULT/synthetic-RLVL/passk_eval/paired_maze_typed_sparse_20260603/`. Bsz16 recovery rows `3715329_3/7/11` are still running on `a0537`, around `8984/10000`, `6783/10000`, and `6807/10000`; replacement eval `3715330_[0-15%4]` remains dependency-pending and its output root is not created yet. No scheduler edit, resubmission, aggregation, report regeneration, or new job submission was needed.
- 18:10 CEST active recovery oversight: checked required handoff/research/report docs, live `squeue`, row-level `sacct`, `scontrol show job`, A100 partition state, active logs, final adapters, and eval output roots for typed maze `3705795` plus batch-size `3715329/3715330`. Typed maze eval rows `3705795_0/1/2` are running on `a0633` since 11:42 CEST, with logs around chunks `52/112`, `52/112`, and `51/112`; later chunks often hit `max=8192`, so rows are progressing but walltime-risk. Bsz16 recovery rows `3715329_3/7/11` are running on A100-80GB node `a0537` since 11:43 CEST; row `3` resumed from `checkpoint-8000` and is around `8811/10000`, while rows `7/11` resumed from `checkpoint-6000` and are around `6645/10000` and `6657/10000`. No unrecovered Traceback, CUDA OOM, quota/no-space, dependency-never-satisfied, tokenizer/model-load, vLLM failure, idle-GPU symptom, or new node failure was found; tokenizer/vLLM messages are startup warnings. `3715330` remains dependency-pending on `afterany:3715329_*`. No partition edit, cancellation, resubmission, aggregation, report regeneration, or new oversight scheduling was done because pending rows are array-throttle/dependency blocked and there are still `0` typed maze JSONs plus no batch eval output root.
- 11:50 CEST scheduled oversight: added `scripts/slurm/codex/active_recovery_oversight_2026-06-10.slurm`, a one-off A40 Codex oversight wrapper for typed maze eval `3705795` and batch-size recovery/eval `3715329/3715330`. Initial CPU-only submission was rejected because Alex requires GPU allocation; A40 submission without explicit `--mem` passed. Submitted jobs `3715439` (`2026-06-10 18:00`), `3715440` (`20:00`), `3715441` (`2026-06-11 00:00`), `3715442` (`04:00`), and `3715443` (`08:00`), all pending on `BeginTime`.
- 11:07 CEST live refresh/recovery: typed maze SFT is complete at `30/30` final adapters; eval `3705795_[0-29%3]` is maintenance-held with `0` JSONs. Bsz16 batch-size recovery `3711850_[3,7,11%3]` timed out after 20h but made checkpoint progress: logic reached `checkpoint-8000`, `nl_exact` reached `checkpoint-6000`, and conditioned-dual reached `checkpoint-6000`. Canceled stale eval `3711851` because its `afterany` dependency was satisfied by timeout while the three bsz16 finals were still missing. Submitted new recovery `3715329_[3,7,11%3]` and replacement eval `3715330_[0-15%4]` after `afterany:3695197:3698877:3702079:3705794:3715329`; both are pending behind active maintenance reservation `MAINT20260609` until 2026-06-11 18:00 CEST.

## 2026-06-08

- 09:22 CEST report figure refresh: added `figures/paired_igsm_semantic_summary.{pdf,png}` to the report builder and embedded it in the semantic-iGSM subsection, so the corrected iGSM rerun now has both table and train-depth curves analogous to the original synthetic-task plots. Regenerated and mirrored the report; verification found `70` LaTeX figure references and zero missing figures.
- 09:04 CEST report/scheduler update: added the corrected semantic iGSM table to `scripts/analysis/build_logic_cot_report.py`, regenerated `analysis/logic_cot_report_2026-05-25/`, and mirrored it to `../synthetic-RLVL-report`. New table artifacts are `tables/paired_igsm_semantic_by_seed.csv` and `tables/paired_igsm_semantic_summary.csv`; the LaTeX report now has a dedicated semantic-iGSM subsection. Also inspected maintenance-blocked bsz16 recovery `3711850`: A40/RTX partitions and visible A100-40GB nodes are incompatible with the `a100_80&el9`, 240GB/GPU request, while visible A100-80GB nodes were planned for maintenance. Edited `3711850` walltime from 24h to `20:00:00`; Slurm accepted the edit and rows `3/7/11` started on `a0532`.
- 08:47 CEST live refresh: semantic iGSM clean NL-only re-eval `3705807_[3-5,9-11,15-17,21-23,27-29%4]` completed all `15/15` rows, so all `30/30` semantic iGSM pass@k JSONs are current. Alias canonicalization fixed shallow semantic-NL validation and OOD/depth-50 parser coverage is near-complete, but OOD/depth-50 translated joint remains `0.000`: sampled long NL traces drift/truncate and only small prefixes validate. Current train-1-to-25 corrected means: NL OOD/depth-50 correct@16 `0.873/0.677`; logic OOD/depth-50 correct@16 `0.612/0.281`, internal valid@16 `0.433/0.052`, strict grounded joint `0.000`.
- 08:47 CEST operational refresh: typed maze SFT advanced to `28/30` finals, with only original rows `3695238_28/29` running; eval `3705795_[0-29%3]` remains dependency-pending with `0` JSONs. Batch-size SFT advanced to `9/12` finals after row `10` recovered; bsz16 rows `3/7/11` timed out after saving checkpoints, so stale eval `3705796` was canceled/replaced by bsz16-only recovery `3711850_[3,7,11%3]` and full eval `3711851_[0-15%4]` after `afterany:3695197:3698877:3702079:3705794:3711850`.

## 2026-06-03

- 13:28 CEST semantic iGSM rerun and trace-control validator/report fix: added semantic official-iGSM Slurm chain `build_paired_igsm_semantic_2026-06-03.slurm`, `paired_igsm_semantic_2026-06-03.slurm`, and `posthoc_paired_igsm_semantic_eval_2026-06-03.slurm`. First build `3695464` failed immediately because `WORK` was redirected before the default local iGSM repo path was resolved; patched the scripts to export `IGSM_REPO_PATH=/home/atuin/c107fa/c107fa12/codex_research/iGSM`, canceled broken dependents `3695465`/`3695466`, removed the failed partial dataset root, and submitted replacement build `3695525`, SFT `3695526_[0-29%3]`, and eval `3695527_[0-29%3]`. The replacement build is running and downstream arrays are dependency-pending. The build writes fresh semantic artifacts to `$HPCVAULT/synthetic-RLVL/datasets/materialized_paired_official_igsm_semantic_20260603/` with `100k` train rows per train-depth subset, so the 10k-step batch-size-1 SFT runs do not need to cycle a fixed subset under the normal sampler. Old iGSM artifacts are stale because they were materialized before the semantic grounding fix and therefore contain hidden `v_` constants plus generic `official iGSM relation` NL wording. Also patched `scripts/analysis/build_logic_cot_report.py` so formal trace-control rows use strict grounded joint rather than citation-free joint; citation-free reconstruction can over-credit `invalid_logic` because it ignores deliberately broken citations. Added focused tests showing `terse_nl`, `rule_annotated_nl`, and `pseudocode` gold targets translate to valid logic, while `invalid_logic` is strict-grounded invalid even when citation-free grounded validity succeeds. Restored the OLMo tokenizer cache under `$HPCVAULT/.cache/hf`, regenerated the in-repo report, and mirrored it to `../synthetic-RLVL-report`; refreshed trace-control summary now has strict-grounded formal joint `0.0` for `invalid_logic` and `shuffled_logic`. Verification: Slurm `bash -n`, semantic iGSM logic/NL smoke, `py_compile` for the report builder, and `tests/test_training_stack.py tests/test_logic_symbol_padded_template.py` passed (`34 passed`).
- 12:26-12:29 CEST quota recovery and typed maze rerun: restored `$WORK` directory creation by deleting safe completed-run intermediate `checkpoint-*` dirs with final adapters present, old disposable `$WORK/RLVL` container/W&B cache dirs, and reproducible HF/vLLM/W&B caches; active final adapters, datasets, eval outputs, and current merge dirs were left intact. `$WORK` usage is now about `357G`, and a create/remove probe succeeds. Patched `synthrlvl/datasets/paired_synthetic.py` so maze formal symbols are typed (`r_<room>` and `k_<key>`) while NL and final answers keep natural room/key names, eliminating the old room/key namespace ambiguity. Added regression coverage in `tests/test_paired_synthetic_datasets.py` and new Slurm scripts `build_paired_maze_typed_2026-06-03.slurm`, `paired_maze_typed_2026-06-03.slurm`, and `posthoc_paired_maze_typed_eval_2026-06-03.slurm`. Verification passed: full paired synthetic tests (`13 passed`), generated typed maze depth sweeps through `1/2/5/10/25/50`, tiny parquet materialization smoke, `bash -n` on new/patched Slurm scripts, and `git diff --check`. Submitted typed maze build/SFT/eval chain `3695237 -> 3695238 -> 3695239` under `$HPCVAULT`; build is running and downstream arrays are dependency-pending. Batch-size ablation replacement `3695197_0..2` is running past the original quota failure point; eval `3695199` remains dependency-pending.
- 11:59-12:04 CEST HFSA batch-size ablation launch/recovery: implemented modality tags for conditioned-dual materialized SFT examples and added balanced modality samplers in `train_sft.py` so conditioned-dual can train with 50/50 logic-NL physical batches. Added `scripts/slurm/sweeps/sft/hfsa_batch_size_ablation_2026-06-03.slurm` for 12 SFT runs: batch sizes `2/4/8/16` across `logic`, `nl_exact`, and 50/50-balanced `conditioned_dual`, one train-length setting `train-1-to-20`, seed `3407`, `10k` optimizer steps, `grad_accum=1`, gradient checkpointing enabled. Added dependent eval script `scripts/slurm/jobs/posthoc_hfsa_batch_size_ablation_eval_2026-06-03.slurm` with 16 eval rows because conditioned-dual is evaluated as both `conditioned_logic` and `conditioned_nl`. Verification passed before submit: `bash -n` for both Slurm scripts, `py_compile train_sft.py`, and focused tests for conditioned-dual modality duplication plus accumulation-window balancing (`2 passed`). First submitted SFT `3695143_[0-11%3]` and eval `3695147_[0-15%4]`, but SFT rows `0/1` failed immediately with `OSError: [Errno 122] Disk quota exceeded` when creating new directories under `$WORK`; canceled that attempt, patched both scripts to redirect this ablation's `WORK`, HF caches, W&B scratch, SFT output dirs, merge dirs, and eval outputs to `$HPCVAULT`, then resubmitted replacement SFT `3695197_[0-11%3]` and eval `3695199_[0-15%4]` with `afterok:3695197`. Watch larger physical batch sizes for OOM; if one fails, recover the smallest affected row set.
- 11:21 CEST ablation plot/audit refresh: patched `scripts/analysis/build_logic_cot_report.py` so trace-control and hybrid-order figures now include the normal compact-logic/exact-NL baselines rather than only adding baselines to tables. Added `ablation_shortcut_kind_rate_lines_vs_main.{pdf,png}` plus `shortcut_kind_ablation_vs_main.csv` so position and initial-marker shortcut kinds are plotted as rate curves with baseline rate `0.0`, matching the original shortcut-rate figure style. Added `logic_length_control_surface_examples.csv` and a direct LaTeX subsection showing compact, symbol-padded, and wordified formal surface snippets for the same training item. Audit notes recorded in the report: conditioned-dual trains separate mode-conditioned examples and eval uses the same mode prompt, so the poor conditioned-logic result is not a same-datapoint train-test mismatch; likely confounds are per-modality underexposure at fixed optimizer steps, mode-conditioning overhead, and cross-modality interference. Shortcut-kind sample JSONL aggregation found `active_branch_first=None` for completed eval rows, so the initial-marker NL rate-0.8 improvement is not caused by evaluating with shortcuts still active; treat it as surprising/provisional regularization or distribution interaction, while the original schema shortcut-rate ablation still shows NL degradation with higher shortcut rate. Focused tests passed for padded/wordified gold validity, conditioned-dual row duplication, and shortcut-neutral eval.
- 10:10 CEST ablation report refresh: patched `scripts/analysis/build_logic_cot_report.py` so the targeted-ablation report includes direct full train-sequence examples for normal logic, normal NL, `terse_nl`, `rule_annotated_nl`, `pseudocode`, `shuffled_logic`, `invalid_logic`, `shuffled_nl`, symbol-padded logic, and wordified logic. Added `ablation_training_token_audit_512.csv` with OLMo-tokenized prompt/target/total/proof-body and fixed syntax/operator lexeme diagnostics, plus `trace_control_ablation_with_main_baselines.csv` and `hybrid_order_with_main_baselines.csv` so trace-control and hybrid-order tables show normal compact-logic and exact-NL baselines inline. Added/embedded `ablation_shortcut_kind_summary.{pdf,png}` for position and initial-marker shortcut types. New caveat: current HFSA `terse_nl` is token-identical to `nl_exact` in the audit and sampled sequence because the default NL proof is already terse, so it should not be interpreted as a successful shorter-NL length control. Regenerated and mirrored the report bundle (`68` referenced PDFs, `65` CSVs; zero missing figure refs after escaped-underscore normalization). Verification: `py_compile` passed, report builder reran with local OLMo tokenizer, in-repo and mirrored figure references resolve, and local TeX compilation remains unavailable.
- 09:46 CEST report refresh: patched `scripts/analysis/build_logic_cot_report.py` so paired full-suite tables include train-band metrics and a new `paired_full_suite_family_partial.{pdf,png}` figure covering completed iGSM/maze/attribute slices. Regenerated `analysis/logic_cot_report_2026-05-25/` and mirrored it to `../synthetic-RLVL-report`. The report now records the maze train-1-to-5 failure mode explicitly: perfect train-band logic/NL correctness, near-zero OOD/depth-50 performance, unsupported maze NL translated validity, and sample-inspection evidence of depth-25/50 premise-copying, malformed/truncated traces, and generation-cap pressure. Verification: `py_compile` passed for the report builder, generated paired CSVs include train metrics, the new paired-family figure exists, and TeX figure references resolve after accounting for escaped underscores; local TeX compilation remains unavailable.
- 09:26 CEST paired eval recovery/status: paired full-suite eval output advanced to `36/90` JSONs/sample JSONLs: `official_igsm` `30/30`, `maze_navigation` `6/30` with all train-1-to-5 logic/NL seeds complete, and hard `attribute_constraints` `0/30`. Recovery `3691024_[30-32%3]` completed cleanly and filled the missing maze `logic` train-1-to-5 rows. Maze train-1-to-5 preliminary means are train-band memorization but poor extrapolation: logic train correct/joint@16 `1.000/1.000`, OOD correct/joint@16 `0.039/0.027`, depth-50 `0.000/0.000`; NL train correct@16 `1.000`, OOD correct@16 `0.145`, depth-50 `0.000`, with maze NL parse/validity unsupported by the current translator. Original eval row `3682449_39` is still running, rows `36/37` timed out, row `38` node-failed, and rows `40..89` were canceled or node-failed after the array failure. Submitted targeted recoveries: `3694618_[36-38%3]` with `PASSK_MAX_NEW_TOKENS=4096` for maze `logic` train-1-to-10, and `3694619_[40-89%4]` with default caps for the canceled remainder. No partition edit was needed.

## 2026-06-02

- 15:25 CEST paired-family grounding audit: checked fresh `maze_navigation` and `attribute_constraints` samples across depths `1/2/5/10/25/50` and indices `0..2`. `attribute_constraints` is fine with respect to the iGSM hidden-variable issue: slot/value symbols are explicit in the prompt and proof, strict validation passed, and no duplicate constants appeared. `maze_navigation` is prompt-grounded but has a bounded formal namespace ambiguity because a word can be both a room and a key, e.g. `silver = maze room silver` plus `silver = maze key silver`; strict proof validation still passes because the prover treats constants as untyped symbols. A typed maze-symbol patch was prototyped and passed focused checks, but it was reverted before commit because it would stale current maze artifacts and require reruns. Recorded this as a caveat/future cleanup, not an active generator change.
- 15:25 CEST iGSM semantic/bare-symbol grounding fix: patched `synthrlvl/datasets/paired_synthetic.py` so official_iGSM materialization preserves semantic variable bindings from official solution clauses like `Define Swan's Gallbladder as h`, emits bare one-letter constants such as `h = the number of each Swan's Gallbladder`, labels helper variables as intermediate calculations for the semantic quantity, and replaces generic NL proof prose (`From the official iGSM relation...`) with semantic proof prose (`From the definition of Swan's Gallbladder (h)...`) that does not mention iGSM in new targets. Patched `logic_engine/prover.py` so free one-letter terms are valid in equality formulas, allowing official lowercase `s..z` arithmetic registers while leaving non-equality predicate formulas subject to closure requirements. Patched `synthrlvl/natural_logic.py` to parse both old `v_`/official-relation artifacts and new bare-symbol semantic proof lines, and added dedicated regression/stress tests. Verification passed: focused new/legacy translator tests, generated official_iGSM depth sweeps across depths `1/2/5/10/25/50`, py-compile, and full `pytest -q` (`114 passed`, `3 skipped`). Existing full-suite official_iGSM parquet/SFT/eval artifacts remain old-construction artifacts and should be treated as stale for semantic-grounding claims; backlog now requires a semantic bare-symbol iGSM rebuild/rerun before using iGSM as evidence.
- 06:55 CEST paired/oversight live check: paired recovery `3691024_[30-32%3]` is still running with rows `30/31/32` at chunks `16/13/20` of `112` and no final recovery JSONs yet; paired output count remains `32/90`. Ablation oversight `3690645` completed cleanly (`0:0`) after its report/push pass, and successor `3691029` is begin-time pending. No partition edit, dependency edit, cancellation, new submission, broad launch, generator fix, or evaluator fix was made.
- 06:52 CEST push attempt: local commits were created for this repo (`23121cd`, concurrent paired-recovery note `3b12a3b`, plus this push-status note) and `../synthetic-RLVL-report` (`028339b`). Direct pushes to `git@github.com:Thiggel/synthetic-RLVL.git` and `git@github.com:Thiggel/synthetic-RLVL-report.git` each timed out after 60 seconds with no server response; fallback pushes through `ssh://git@ssh.github.com:443/Thiggel/synthetic-RLVL.git` and `ssh://git@ssh.github.com:443/Thiggel/synthetic-RLVL-report.git` also timed out after 60 seconds with no server response. Both local branches remain ahead and ready to push when SSH connectivity returns.
- 06:46 CEST paired recovery live check: targeted retry `3691024_[30-32%3]` is running on `a100` with `PASSK_MAX_NEW_TOKENS=4096`. Row `32` has sampled through chunk `16/112` with observed chunk maxima around `775..1209`, row `30` through chunk `3/112` with maxima around `775..811`, and row `31` is still in prologue; no recovery JSONs were final yet. This confirms the row-level recovery started sampling with shorter outputs than the timed-out 8192-token capped run. No partition edit, dependency edit, cancellation, new submission, broad launch, generator fix, or evaluator fix was made.
- 06:45 CEST HFSA ablation/report refresh: rechecked required handoff/plan/report docs, live `squeue`, row-level `sacct`, partitions, logs, output roots, eval JSONs, and representative samples for the 2026-05-29 ablation wave. Hybrid-order eval remains `26/30` JSONs with rows `3682461_26..29` running; `formal_think` train-1-to-20 now has two seeds in the report, while train-1-to-25 remains pending/running. Conditioned-dual final eval `3674884` advanced to `5/30` JSONs after row `4` completed; rows `5..8` are running and `9..29` are array-throttle pending. Conditioned checkpoint eval remains `21/30` JSONs with rows `3/4/5` running. Sample checks confirmed intended hybrid `<formal>` then `<think>/<answer>` wrappers, conditioned-logic `<reasoning_mode>formal_logic</reasoning_mode>` prompts with `<formal>/<answer>`, conditioned-NL `<think>/<answer>` outputs, and working answer extraction; formal/translated validity failures remain generation fragility rather than parser breakage. Focused fatal-log scans found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, dependency, tokenizer/model-load, vLLM, node-failure, cancellation, or idle-GPU issue; paired maze rows `3682449_30/31/32` remain covered by live recovery `3691024_[30-32%3]`, so no additional resubmission was made. Pending rows are throttle or begin-time blocked despite compatible idle `a100` nodes, so no partition edit was useful. Regenerated and mirrored the report (`66` PDFs, `61` CSVs, `5` Markdown supplements, zero missing LaTeX figure refs); TeX compilation remains unavailable. Ablation oversight `3690212` completed cleanly, `3690645` is running, and successor `3691029` is begin-time pending. No code/config fix, scheduler edit, cancellation, broad launch, or new experiment submission was made; no visible `tjepa_*`, `seqedit_*`, or `puzzle_*` jobs were present.
- 06:37 CEST paired/report refresh: checked required handoff and plan docs, `squeue`, expanded `sacct`, partition availability, paired logs, manifests, final adapters, eval outputs, and representative samples. Paired eval `3682449_30/31/32` (`maze_navigation`, `logic`, train-1-to-5, seeds `3407/3408/3409`) timed out at the 24h walltime without final JSONs after GPU-active progress around chunks `103-107/112`; logs showed repeated `8192`-token capped chunks, no Traceback, proof-validation failure, OOM/CUDA OOM, context-length error, quota/no-space issue, dependency issue, tokenizer/model-load issue beyond the known Mistral regex warning, vLLM failure, node failure, or idle-GPU symptom. Submitted targeted recovery `3691024_[30-32%3]` with `PASSK_MAX_NEW_TOKENS=4096`; it started on `a100`. Original eval rows `0..29/33/34` are complete, rows `35..38` are running, and rows `39..89` are array-throttle pending. Paired outputs are now `32/90` pass@k JSONs/sample JSONLs: `30` official_iGSM plus two maze `nl_exact` rows; hard `attribute_constraints` remains `0/30`. The new maze `nl_exact` seed-3408 row has OOD correct@16 `0.207`, depth-50 correct@16 `0.000`, and maze NL parse/validity `0.000`; sampled generations preserve `<think>/<answer>` on shallow successes and drift or omit answers at longer depths. Materialized roots still have `55` parquet subsets per family and paired SFT final adapters are `90/90`. Regenerated and mirrored the report (`66` PDFs, `61` CSVs, `5` Markdown supplements, zero missing LaTeX figure refs); TeX compilation remains unavailable. No partition edit, dependency edit, cancellation, broad launch, generator fix, or evaluator fix was made. No visible `tjepa_*`, `seqedit_*`, or `puzzle_*` jobs were present.
- 02:59 CEST push attempt: local commits were created for this repo (`ece496d`, plus this push-status note) and `../synthetic-RLVL-report` (`0cee8fb`). Direct pushes to `git@github.com:Thiggel/synthetic-RLVL.git` and `git@github.com:Thiggel/synthetic-RLVL-report.git` each timed out after 60 seconds with no server response; fallback pushes through `ssh://git@ssh.github.com:443/Thiggel/synthetic-RLVL.git` and `ssh://git@ssh.github.com:443/Thiggel/synthetic-RLVL-report.git` also timed out after 60 seconds with no server response. Both local branches remain ahead and ready to push when SSH connectivity returns.
- 02:55 CEST paired oversight completion: `3690207` completed cleanly (`0:0`) after the 02:39 paired audit and scheduling successor `3690641`. Final output count before commit still showed paired eval `3682449` at `31/90` JSONs, so no report regeneration, scheduler edit, partition edit, cancellation, resubmission, broad launch, or evaluator/generator fix was needed.
- 02:48 CEST HFSA ablation/report refresh: checked `squeue`, expanded `sacct`, active/pending array rows, focused logs, output roots, eval JSONs, sample JSONLs, report artifacts, and live GPU state for the 2026-05-29 ablation wave and still-active predecessors. Hybrid-order eval advanced to `24/30` after `formal_think` train-1-to-15 seeds `3408/3409`; the three-seed train-1-to-15 mean is OOD correct/formal-joint/translated-joint@16 `0.568/0.258/0.264` and depth-50 `0.479/0.083/0.000`. Conditioned-dual 50k SFT is complete; final eval `3674884` has `2/30` JSONs with rows `0/2/4/5` running, and checkpoint eval `3674885` has `15/30` JSONs, completing the `conditioned_logic` train-1-to-25 10k-to-50k curve while `conditioned_nl` rows run. Sample checks confirmed intended hybrid `<formal>` then `<think>/<answer>` wrapping and conditioned-logic `<reasoning_mode>formal_logic</reasoning_mode>` prompts with `<formal>/<answer>` outputs; shallow successes extract/validate, while depth-50 failures are truncation/repetition/validity fragility rather than evaluator breakage. Patched report-builder caption/status text for the newly complete hybrid and conditioned-logic checkpoint slices, `py_compile` passed, regenerated/mirrored the report, and verified `66` PDFs, `61` CSVs, `5` generated Markdown supplements, and zero missing LaTeX figure references. TeX compilation remains unavailable. Focused log/GPU scans found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, dependency, tokenizer/model-load, vLLM failure, node failure, timeout/cancellation, or idle-GPU issue; pending rows are throttle/begin-time blocked, so no partition edit, scheduler edit, cancellation, resubmission, broad launch, or experiment fix was made. No visible `tjepa_*`, `seqedit_*`, or `puzzle_*` jobs were present.
- 02:47 CEST push attempt: committed the paired oversight handoff locally as experiment repo commit `3de33f3`, then attempted direct GitHub SSH push and `ssh.github.com:443` fallback push for `synthetic-RLVL`. Both attempts timed out after 60 seconds with no server response, so the branch remains local-ahead. No report repo push was attempted because this pass did not regenerate or modify the mirrored report bundle.
- 02:39 CEST paired operational audit: `squeue`/expanded `sacct`, pending-row details, active eval logs, output counts, materialized manifests, final adapters, and representative samples were checked for the paired full-suite wave. Paired oversight `3689676` completed cleanly; current paired oversight `3690207` is running after scheduling successor `3690641`. Replacement eval `3682449` remains `31/90` pass@k JSONs/sample JSONLs: `official_igsm` `30/30`, `maze_navigation` first `nl_exact` train-1-to-5 seed-3407 row, and hard `attribute_constraints` `0/30`; rows `30/31/32/34` are running, and rows `35..89` are pending only by `JobArrayTaskLimit`. Active chunks are about `101/112`, `94/112`, `97/112`, and `91/112`; live GPU checks showed `95-98%` utilization with about `67GB` memory. Rows `30/31/32` are walltime risks because logic maze chunks keep hitting the `8192` generation cap, but they are still healthy and progressing. Fatal-log scans found no unrecovered Traceback, proof-validation failure, OOM/CUDA OOM, context-length failure, quota/no-space, dependency, tokenizer/model-load, vLLM, node-failure, timeout/cancellation, or idle-GPU issue. Full-suite manifests remain `55/55` paths and SFT final adapters remain `90/90`. Sample checks re-confirmed 1:1 materialized formal/NL fields and `logic_trace_valid=True`; first maze NL generations use the intended `<think>` wrapper with shallow train-band correctness but zero maze NL translator coverage and deeper drift, while iGSM logic/NL generations preserve intended wrappers and answer extraction but generated translated/grounded validity remains blocked by variable-chain mismatch. No new eval JSONs appeared after the 22:37 report refresh, so no report regeneration, scheduler edit, partition edit, cancellation, resubmission, broad launch, generator fix, or evaluator fix was made.

## 2026-06-01

- 22:51 CEST push attempt: committed the paired/ablation oversight refresh locally as experiment repo commit `c22619a` and report repo commit `3a8f0ea`, then attempted direct GitHub SSH pushes and `ssh.github.com:443` fallback pushes for both repos. All four push attempts timed out after 60 seconds with no server response, so the branches remain local-ahead.
- 22:37 CEST paired/ablation refresh: `squeue`/expanded `sacct`, paired output roots, manifests, active logs, sample JSONLs, and report artifacts were checked. Targeted iGSM NL rerun `3689003` completed all `15/15` official_iGSM `nl_exact` rows; rerun metrics now show OOD parser coverage `1.000/1.000/0.997/1.000/0.994` and depth-50 parser coverage `1.000/1.000/0.990/1.000/0.969` by train range, but OOD/depth-50 translated joint validity remains `0.000` because generated variable chains often do not match the gold formal premises. Paired eval `3682449` remains `31/90` JSONs/sample JSONLs, with rows `30/31/32/34` running and rows `35..89` throttle-pending; active maze rows reached about chunks `90/112`, `83/112`, `86/112`, and `62/112` with `95-97%` GPU utilization and no fatal signatures. Conditioned-dual 50k SFT `3674883` completed, final eval `3674884_0..3` and checkpoint eval `3674885_0..2` released, and checkpoint eval wrote 6 provisional conditioned-logic train-1-to-25 JSONs. Patched `scripts/analysis/build_logic_cot_report.py` to report the completed iGSM rerun and conditioned-50k checkpoint-partial status correctly; `py_compile` passed. Regenerated and mirrored the report (`66` PDFs, `59` CSVs, `5` Markdown supplements). No scheduler edit, partition edit, cancellation, resubmission, broad launch, or experiment fix was made; visible `puzzle_*` jobs were unrelated and no visible `tjepa_*` or `seqedit_*` jobs were present.
- 22:51 CEST push attempt: local branches remain clean but ahead after the 22:37 report refresh (`synthetic-RLVL` through `c22619a`, `../synthetic-RLVL-report` through `3a8f0ea`). Direct pushes to `git@github.com:Thiggel/synthetic-RLVL.git` and `git@github.com:Thiggel/synthetic-RLVL-report.git` failed with `ssh: connect to host github.com port 22: Connection timed out`; fallback pushes through `ssh://git@ssh.github.com:443/Thiggel/synthetic-RLVL.git` and `ssh://git@ssh.github.com:443/Thiggel/synthetic-RLVL-report.git` failed with `ssh: connect to host ssh.github.com port 443: Connection timed out`. No remote update occurred.
- 18:51 CEST push attempt: local commits are present in this repo and `../synthetic-RLVL-report`, but pushing both repos failed. Direct pushes to `git@github.com:Thiggel/synthetic-RLVL.git` and `git@github.com:Thiggel/synthetic-RLVL-report.git` each timed out after 60 seconds with no server response; fallback pushes through `ssh://git@ssh.github.com:443/Thiggel/synthetic-RLVL.git` and `ssh://git@ssh.github.com:443/Thiggel/synthetic-RLVL-report.git` also timed out after 60 seconds with no server response. Local branches remain ahead and ready to push when SSH connectivity returns.
- 18:41 CEST HFSA/paired oversight refresh: `squeue`/expanded `sacct`, focused logs, output roots, manifests, eval JSONs, and sample-generation artifacts were checked for the 2026-05-29 HFSA ablation wave, paired suite, and active predecessors. Conditioned-dual 50k rows `3674883_12/13` completed cleanly and row `14` remains running; final/checkpoint evals `3674884/3674885` are still dependency-pending. Paired eval `3682449` advanced to `31/90` JSONs/sample JSONLs after the first `maze_navigation` row completed; active paired rows are `30/31/32/34`, and hard `attribute_constraints` still has no eval rows. Targeted iGSM NL rerun `3689003` is `8/15` complete; completed rerun rows now parse iGSM NL proof lines, but generated translated validity remains `0.000` because generated variable chains often do not match gold formal premises. First maze `nl_exact` train-1-to-5 seed-3407 has OOD correct@16 `0.088`, NL parse@16 `0.000`, and depth-50 correct@16 `0.000`; maze NL validity is unsupported by the current translator. Regenerated and mirrored the report (`65` PDFs, `57` CSVs, `5` Markdown supplements); `py_compile` passed for `scripts/analysis/build_logic_cot_report.py`, and TeX compilation remains unavailable. No unrecovered fatal log signatures, idle-GPU symptom, scheduler edit, partition edit, cancellation, resubmission, broad launch, or experiment fix was found/made; visible `puzzle_*` jobs were unrelated and no visible `tjepa_*` or `seqedit_*` jobs were present.
- 15:29 CEST iGSM NL validity fix/rerun: patched `synthrlvl/natural_logic.py` to translate official iGSM NL proof lines (`From the official iGSM relation...`, `Substitute ...`, `Evaluate/Reduce ... modulo 23`) into cited formal equality/MOD23 proof lines, and patched `synthrlvl/metrics.py` so translated NL validity can use strict proof validation when citation-free recovery lacks equality/MOD23 support. Added regression tests in `tests/test_training_stack.py`; verification passed with `28 passed`, `tests/test_paired_synthetic_datasets.py` passed (`9 passed`), `py_compile` passed, and sampled materialized gold official_iGSM NL targets at depths `1/10/25/50` now have format/correct/parse/valid all `1.0`. Submitted minimal post-hoc rerun `3689003_[3-5,9-11,15-17,21-23,27-29%4]` with `FORCE_PASSK_EVAL=1` to overwrite only the 15 completed official_iGSM `nl_exact` pass@k JSONs.
- 15:16 CEST paired iGSM validity audit: analyzed completed `official_igsm` pass@k JSONs and representative seed-3409 train-1-to-25 logic/NL sample generations against reconstructed eval records. The `nl_exact` translated-validity `0.000` is an evaluator coverage issue: `synthrlvl/natural_logic.py` does not parse iGSM official-relation, substitution, or modulo-23 proof lines, so even gold-style iGSM NL traces map to `INVALID ; R`. Logic is lower on iGSM answer correctness because many outputs are internally valid but ungrounded invented formal chains; examples include gold `v_a = 0` vs generated `v_Y = 0`, gold `v_g = 1` vs generated `v_m = 18`, and gold `v_r = 0` vs generated `v_y = 0`. Wrote `docs/paired_igsm_validity_audit_2026-06-01.md` and updated handoff/backlog guidance; no code, scheduler, or report-generation change was made.
- 14:57 CEST push attempt: local commits were created for this repo through ablation handoff/report commit `43402c6`, followed by this local push-status note, and for `../synthetic-RLVL-report` through `9ed8411`, but pushing both repos failed. Direct pushes to `git@github.com:Thiggel/synthetic-RLVL.git` and `git@github.com:Thiggel/synthetic-RLVL-report.git` each timed out after 60 seconds with no server response; fallback pushes through `ssh://git@ssh.github.com:443/Thiggel/synthetic-RLVL.git` and `ssh://git@ssh.github.com:443/Thiggel/synthetic-RLVL-report.git` also timed out after 60 seconds with no server response. Local branches remain ahead and ready to push when SSH connectivity returns.
- 14:42 CEST HFSA ablation/report refresh: shortcut-kind eval `3674888_0..23` completed and all `24/24` JSONs plus sample JSONLs are report-ingested. Final `initial_marker` `nl_exact` shortcut-`0.8` three-seed means are OOD correct/translated-joint@16 `0.771/0.702` and depth-50 `0.667/0.500`. Hybrid-order advanced to `22/30` after `3682461_21`; `formal_think` train-1-to-15 seed `3407` has OOD correct/formal-joint/translated-joint@16 `0.656/0.301/0.250` and depth-50 `0.688/0.125/0.000`. Conditioned-dual 50k rows `0..11` are complete and rows `12..14` are running with high GPU use; final/checkpoint evals remain dependency-pending. Regenerated and mirrored the report (`65` PDFs, `57` CSVs, `5` Markdown supplements); TeX compilation remains unavailable. Sample checks confirmed intended shortcut-neutral prompts, wrappers, answer extraction, and expected depth/truncation/validity fragility rather than evaluator breakage. Focused log/GPU scans found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, dependency, tokenizer/model-load, vLLM failure, node failure, timeout, cancellation, or idle-GPU issue. No scheduler edit, partition edit, cancellation, resubmission, broad launch, or fix was made; visible `puzzle_*` jobs are unrelated and no visible `tjepa_*` or `seqedit_*` jobs were present.
- 14:42 CEST paired oversight completion: `3687983` completed cleanly (`0:0`) after recording the 14:35 paired audit and scheduling successor `3688815`. No new paired eval JSONs appeared beyond `official_igsm` `30/30`, and there was no fatal signature, idle-GPU symptom, report trigger, scheduler edit, partition edit, cancellation, resubmission, broad launch, generator fix, or evaluator fix.
- 14:42 CEST push attempt: local paired handoff commit `8dff64b` was created, but pushing this repo failed. Direct push to `git@github.com:Thiggel/synthetic-RLVL.git` and fallback push to `ssh://git@ssh.github.com:443/Thiggel/synthetic-RLVL.git` each timed out after 60 seconds with no server response. This push-status note is local-only until connectivity returns; the local branch remains ahead.
- 14:35 CEST paired operational audit: current paired oversight `3687983` is running and successor `3688815` is begin-time pending. Replacement eval `3682449` remains healthy with rows `0..29` complete, rows `30..33` running, and rows `34..89` pending only by `JobArrayTaskLimit`; no `maze_navigation` or hard `attribute_constraints` eval JSON has finished, so the eval output directory still has `30/90` pass@k JSONs and `30` sample JSONLs, all `official_igsm`. Active maze rows advanced to about chunks `66/112`, `58/112`, `62/112`, and `90/112`; live GPU checks showed `93-98%` utilization with about `67GB` used. Full-suite manifests remain `55/55` with no missing parquet paths for all three families, and paired SFT final adapters remain `90/90`. Focused fatal-log scans found no unrecovered Traceback, proof-validation failure, OOM/CUDA OOM, context-length failure, quota/no-space, dependency, tokenizer/model-load, vLLM failure, node failure, timeout, cancellation, or idle-GPU issue beyond the known Mistral tokenizer regex warning. Materialized row checks re-confirmed matched formal/NL questions, equal proof-line counts, correct answers, and `logic_trace_valid=True`; completed iGSM samples still show intended wrappers and answer extraction, while paired NL validity remains translator-blocked. No report regeneration, scheduler edit, partition edit, cancellation, resubmission, broad launch, generator fix, or evaluator fix was made. Visible `puzzle_*` jobs are unrelated; no visible `tjepa_*` or `seqedit_*` jobs were present.
- 10:55 CEST push attempt: local commits were created for this repo through this push-status note and for `../synthetic-RLVL-report` (`f01ef90`), but pushing both repos failed. Direct pushes to `git@github.com:...` and fallback pushes through `ssh://git@ssh.github.com:443/...` each timed out after 60 seconds with no server response. Local branches remain ahead and ready to push when SSH connectivity returns.
- 10:43 CEST paired oversight completion: `3687377` completed cleanly (`0:0`) after the 10:31 paired audit and scheduling successor `3687983` for 14:29 CEST. It found paired eval still at `30/90` JSONs, all `official_igsm`; no new `maze_navigation` or hard `attribute_constraints` eval JSONs, no fatal signatures, no idle-GPU symptom, and no report regeneration, scheduler edit, partition edit, cancellation, resubmission, broad launch, or fix was needed.
- 10:40 CEST HFSA ablation/report refresh: shortcut-kind eval advanced to `23/24` after rows `3674888_21/22`; the new `initial_marker` `nl_exact` shortcut-`0.8` two-seed partial has OOD correct/translated-joint@16 `0.675/0.572` and depth-50 `0.594/0.344`. Hybrid-order advanced to `21/30` after replacement row `3682461_20`, making `formal_think` train-1-to-10 three-seed with OOD correct/formal-joint/translated-joint@16 `0.626/0.279/0.381` and depth-50 `0.396/0.000/0.000`. Conditioned-dual 50k row `3674883_8` completed cleanly; rows `9..12` are running and rows `13..14` are throttle-pending. Regenerated and mirrored the report (`65` PDFs, `57` CSVs, `5` Markdown supplements); TeX compilation remains unavailable. Sample inspection confirmed intended shortcut-neutral prompts, `<think>/<answer>` and `<formal>/<think>/<answer>` wrappers, working answer extraction, and expected depth/truncation/validity fragility rather than evaluator or scheduler breakage. Focused log and GPU scans found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, dependency, tokenizer/model-load, vLLM failure, node failure, timeout, cancellation, or idle-GPU issue. No scheduler edit, partition edit, cancellation, resubmission, broad launch, or fix was made; visible `puzzle_oversight` is unrelated and no visible `tjepa_*` or `seqedit_*` jobs were present.
- 10:31 CEST paired operational audit: `squeue`/expanded `sacct`, manifests, final adapters, eval outputs, active logs, GPU utilization, and representative completed iGSM sample JSONLs were rechecked. Paired eval remains `30/90` JSONs and `30` sample JSONLs, all `official_igsm`; `maze_navigation` rows `3682449_30..33` are running and sampling around chunks `51/112`, `44/112`, `49/112`, and `56/112`, with about `96-97%` GPU utilization. Full-suite manifests still have `55` subsets per family and no missing parquet paths; paired SFT final adapters remain `90/90`. Focused fatal-log scans found no unrecovered Traceback, proof-validation failure, CUDA OOM/OOM, context-length failure, quota/no-space, dependency, tokenizer/model-load, vLLM failure, node failure, timeout, cancellation, or idle-GPU issue; only the known Mistral tokenizer regex warning appeared. Completed iGSM samples still match intended wrappers and answer extraction, while paired NL validity remains `0.000` and depth-50 logic can be answer-correct but invalid/ungrounded. Paired oversight `3687377` is running and already scheduled `3687983` for 14:29 CEST. No report regeneration was run and no scheduler edit, partition edit, cancellation, resubmission, broad launch, or fix was made. Visible `puzzle_oversight` is unrelated; no visible `tjepa_*` or `seqedit_*` jobs were present.
- 09:15 CEST operational audit: `squeue`/`sacct`, output counts, and active logs were rechecked. Conditioned-dual 50k rows `3674883_6` and `3674883_7` completed cleanly since the 07:00 handoff; rows `8..11` are running and rows `12..14` are throttle-pending. Paired eval remains `30/90` JSONs, all `official_igsm`; `maze_navigation` rows `30..33` are actively sampling but have no completed JSONs yet. Shortcut-kind remains `21/24` with rows `21..23` running, and hybrid remains `20/30` with rows `20..23` running. Ablation oversight `3686897` completed cleanly after the 07:00 refresh. Focused active-log scan found no unrecovered Traceback, CUDA OOM, quota/no-space, dependency, node-failure, timeout, cancellation, or idle-GPU issue. No report regeneration was run because no new eval artifacts appeared; no scheduler edit, partition edit, cancellation, resubmission, broad launch, or fix was made.
- 07:00 CEST paired oversight completion: `3686895` completed cleanly (`0:0`) after the iGSM-complete report refresh; next paired pass `3687377` remains begin-time pending. No scheduler edit, partition edit, cancellation, resubmission, broad launch, or fix was made.
- 06:57 CEST final push attempt: push attempts covered the current local report state (`../synthetic-RLVL-report` through `871ee2e`) and the then-current main-repo handoff/report state, but both repos again timed out after 60 seconds on `git@github.com:...` and after 60 seconds on `ssh://git@ssh.github.com:443/...`. This note is local-only; local branches remain ahead and ready to push when SSH connectivity returns.
- 06:57 CEST paired/report refresh: eval row `3682449_29` completed cleanly, bringing `official_igsm` to `30/30` pass@k JSONs and sample JSONLs. Rows `30..33` are now running, rows `34..89` are throttle-pending, and `maze_navigation` plus hard `attribute_constraints` still have no completed eval JSONs. Regenerated and mirrored the report; active artifact status now marks iGSM complete and leaves maze/attribute eval pending. Final iGSM diagnostics-only means: logic train-1-to-5/10/15/20/25 OOD correct/internal-joint@16 `0.312/0.255`, `0.507/0.377`, `0.546/0.392`, `0.536/0.245`, `0.488/0.106`; `nl_exact` OOD correct@16 `0.366/0.589/0.618/0.576/0.585`, with translated validity still `0.000`. The newly completed `nl_exact` train-1-to-25 seed `3409` row has OOD/depth-50 correct@16 `0.575/0.531`, preserves `<think>/<answer>` formatting, and has `nl_logic_parse=0.000`. Conditioned-dual 50k rows `3674883_4` and `3674883_5` also completed; rows `6..9` are running and final/checkpoint evals remain dependency-pending. No scheduler edit, partition edit, cancellation, resubmission, broad launch, or fix was made.
- 06:46 CEST push attempt: push attempts made before this note covered this repo through `fee306b` and `../synthetic-RLVL-report` through `9be0985`, but pushing both repos failed. `git@github.com:...` timed out with no server response after 60 seconds, and fallback pushes through `ssh://git@ssh.github.com:443/...` also timed out after 60 seconds. This push-status note is local-only until connectivity returns; local branches remain ahead and ready to push.
- 06:35 CEST HFSA ablation/report refresh: shortcut-kind eval advanced to `21/24` JSONs after rows `3674888_17..20`; `initial_marker` logic shortcut `0.8` is three-seed with OOD correct/joint@16 `0.885/0.610` and depth-50 `0.865/0.344`, while `initial_marker` `nl_exact` shortcut `0.5` is three-seed with OOD `0.469/0.421` and depth-50 `0.115/0.094`. Hybrid order advanced to `20/30` after `formal_think` train-1-to-10 seeds `3407/3408`; that partial is OOD correct/formal-joint/translated-joint@16 `0.602/0.242/0.363` with depth-50 translated joint still `0.000`. Paired iGSM partial advanced to `29/30`: logic train-1-to-25 is three-seed OOD correct/internal-joint@16 `0.488/0.106`, and `nl_exact` train-1-to-25 is two-seed OOD correct@16 `0.591` with translated validity still `0.000`; `maze_navigation` rows `30..32` have started but not completed. Conditioned-dual 50k rows `0..3` completed, rows `4..7` are running, and final/checkpoint evals remain dependency-pending. Sample inspection of new shortcut-kind, hybrid, and paired iGSM rows confirmed intended wrappers, shortcut-neutral eval prompts where applicable, working `<answer>` extraction, and expected depth-50 answer/validity fragility. Regenerated and mirrored the report (`65` PDFs, `57` CSVs, `5` Markdown supplements); TeX compilation remains unavailable. Focused log scans found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, dependency, tokenizer/model-load, node-failure, timeout, cancellation, idle-GPU, or free-partition issue. No scheduler edit, partition edit, cancellation, resubmission, broad launch, or fix was made; no visible `tjepa_*`, `seqedit_*`, or `puzzle_*` jobs were present.
- 02:50 CEST paired oversight completion: `3686267` completed cleanly (`0:0`) after recording the `3682449` `official_igsm` `22/30` partial readout and scheduling successor `3686895`. No additional scheduler edit, partition edit, cancellation, resubmission, broad launch, or fix was made.
- 02:49 CEST push attempt: local commits were created for this repo (`0990215`) and `../synthetic-RLVL-report` (`f3b0ba3`), but pushing both repos failed. `git@github.com:...` timed out with no server response after 60 seconds, and fallback pushes through `ssh://git@ssh.github.com:443/...` also timed out after 60 seconds. Local branches remain ahead and ready to push when SSH connectivity returns.
- 02:45 CEST HFSA ablation oversight/report refresh: shortcut-kind eval advanced to `17/24` JSONs after `3674888_15/16`; `initial_marker` `nl_exact` shortcut `0.5` is now two-seed with OOD correct/translated-joint@16 `0.509/0.481` and depth-50 `0.125/0.109`, while `initial_marker` logic `0.5` remains three-seed at OOD `0.883/0.625` and depth-50 `0.854/0.344`. Conditioned-dual 40k completed and 50k chunk `3674883` is running rows `0..3`; final eval `3674884` and checkpoint eval `3674885` remain dependency-pending with no JSONs. Hybrid order remains `18/30` with rows `18..21` running; shortcut-kind rows `17..20` are running. Sample inspection of new shortcut-kind NL rows confirmed shortcut-neutral prompts, intended `<think>`/`<answer>` formatting, working answer extraction, and depth-50 fragility; paired iGSM samples still show intended wrappers and paired NL translator coverage remains zero. Regenerated and mirrored the report with shortcut-kind `17/24` and paired iGSM `22/30` partials (`65` PDFs, `57` CSVs, `5` Markdown supplements); TeX compilation remains unavailable. Focused log, partition, and GPU checks found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, dependency, tokenizer/model-load, vLLM, node-failure, timeout, cancellation, idle-GPU, or free-partition issue. No scheduler edit, partition edit, cancellation, resubmission, broad launch, or fix was made; current ablation oversight `3686268` is running and next pass `3686897` is begin-time pending.
- 02:36 CEST paired full-suite oversight/report refresh: replacement eval `3682449` has rows `0..21` complete, rows `22..25` running, and rows `26..89` pending only by `JobArrayTaskLimit`; paired oversight `3685570` completed cleanly, current pass `3686267` is running, and next pass `3686895` is begin-time pending. The first `22` pass@k JSONs and sample JSONLs are all `official_igsm`; `maze_navigation` and hard `attribute_constraints` still have none. Diagnostics-only partial means: logic train-1-to-5/10/15/20 OOD correct@16 `0.312/0.507/0.546/0.536` and internal-joint@16 `0.255/0.377/0.392/0.245`; `nl_exact` train-1-to-5/10/15 OOD correct@16 `0.366/0.589/0.618`, with one train-1-to-20 seed at `0.589`; NL parse/translated validity remains `0.000`. Sample inspection found intended `<formal>`/`<think>` wrappers and working `<answer>` extraction; shallow logic can be citation-free valid, but grounded iGSM validity remains unreliable beyond trivial retrieval, and deeper logic/NL generations show answer or validity fragility. Regenerated and mirrored the report with updated paired partial tables, figure, and new sample supplement (`65` PDFs, `57` CSVs, `5` Markdown supplements); TeX compilation remains unavailable. Focused log/GPU checks found no unrecovered Traceback, proof-validation failure, OOM/CUDA OOM, context-length failure, quota/no-space, dependency, tokenizer/model-load, node-failure, timeout, cancellation, or idle-GPU issue; the only warning class observed was the known tokenizer regex warning. No scheduler edit, partition edit, cancellation, resubmission, broad launch, or fix was made; no visible `tjepa_*` or `seqedit_*` jobs were present.

## 2026-05-31

- 22:40 CEST paired full-suite oversight/report refresh: replacement eval `3682449` has rows `0..13` complete, rows `14..17` running, and rows `18..89` pending only by `JobArrayTaskLimit`; paired oversight `3685570` is running and next pass `3686267` is begin-time pending. The first `14` pass@k JSONs and sample JSONLs are all `official_igsm`; `maze_navigation` and hard `attribute_constraints` still have none. Diagnostics-only partial means: logic train-1-to-5/train-1-to-10/train-1-to-15 OOD correct@16 `0.312/0.507/0.547` and internal-joint@16 `0.255/0.377/0.400` (`train1to15` two seeds); `nl_exact` train-1-to-5/train-1-to-10 OOD correct@16 `0.366/0.589`, with NL parse/translated validity still `0.000`. Sample inspection found intended `<formal>`/`<think>` wrappers and working `<answer>` extraction; shallow logic can be grounded-valid, while deeper logic/NL generations show grounding, validity, or answer fragility. Regenerated and mirrored the report with new paired partial tables, figure, and sample supplement (`65` PDFs, `57` CSVs); TeX compilation remains unavailable. Focused log/GPU checks found no unrecovered Traceback, proof-validation failure, OOM/CUDA OOM, context-length failure, quota/no-space, dependency, tokenizer/model-load, vLLM, node-failure, timeout, cancellation, or idle-GPU issue. No scheduler edit, partition edit, cancellation, resubmission, broad launch, or fix was made; no visible `tjepa_*` or `seqedit_*` jobs were present.
- 22:45 CEST push attempt: local commits were created for this repo and `../synthetic-RLVL-report`, but pushing both repos failed. `git@github.com:...` timed out on `github.com:22`; fallback pushes through `ssh://git@ssh.github.com:443/...` also timed out. Local branches remain ahead and ready to push when SSH connectivity returns.
- 22:35 CEST HFSA ablation oversight/report refresh: shortcut-kind eval advanced to `15/24` JSONs after `3674888_13/14` completed; `initial_marker` logic shortcut `0.5` is now three-seed with OOD correct/joint@16 `0.883/0.625` and depth-50 `0.854/0.344`. Conditioned-dual 40k `3674882` advanced to rows `0..11` complete with rows `12/13/14` running; hybrid order remains `18/30` with rows `18..21` running. Replacement paired eval `3682449` also has partial `official_igsm` outputs at `14/30`, with rows `14..17` running and later rows throttle-pending; paired metrics remain provisional until the full suite finishes. Regenerated and mirrored the report with `64` PDFs and `55` CSVs; TeX compilation remains unavailable. Sample inspection of new `initial_marker` logic seeds `3408/3409` confirmed shortcut-neutral prompts, `<formal>` wrappers, answer extraction, and expected validity/grounding behavior, with one deeper truncated failure lacking an answer tag. Bounded paired iGSM inspection found intended logic/NL wrappers and answer extraction, shallow grounded-valid logic examples, and fragile deeper examples; paired NL translated validity remains blocked. Focused fatal-log scans and GPU checks found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, dependency, tokenizer/model-load, vLLM, node-failure, timeout, cancellation, or idle-GPU issue. No partition edit, dependency edit, cancellation, resubmission, or new science launch was made.
- 18:51 CEST push attempt: local commits were created for this repo (`4237aa9`) and `../synthetic-RLVL-report` (`7a98df9`), but pushing both repos failed. `git@github.com:...` timed out on `github.com:22`; fallback pushes through `ssh://git@ssh.github.com:443/...` also timed out. Local branches remain ahead and are ready to push when SSH connectivity returns.
- 18:35 CEST HFSA ablation oversight/report refresh: shortcut-kind eval advanced to `13/24` JSONs after `3674888_9..12` completed; trace-control remains `18/18`, hybrid order advanced to `18/30` after `formal_think` train-1-to-5 completed, and conditioned 40k `3674882` has rows `0..8` complete with `9/10/11/12` running. Regenerated and mirrored the report with `64` PDFs and `55` CSVs; TeX compilation remains unavailable. Sample inspection of shortcut-kind `position` `nl_exact` shortcut `0.8` seed `3409`, shortcut-kind `initial_marker` logic shortcut `0.5` seed `3407`, and hybrid `formal_think` seed `3409` confirmed shortcut-neutral eval prompts, intended wrappers, and normal `<answer>` extraction; shallow samples are valid, while depth-50 samples still show truncation/validity fragility. Focused live/recent log scans found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, dependency, tokenizer/model-load, vLLM, node-failure, timeout, cancellation, or idle-GPU issue. No partition edit, dependency edit, cancellation, resubmission, or new science launch was made; pending rows are throttle/dependency/begin-time blocked, and no visible `puzzle_*`, `tjepa_*`, or `seqedit_*` jobs were present.
- 18:35 CEST paired full-suite oversight: row-56 replacement `3683070_56` and replacement SFT rows `3682411_84..89` completed cleanly, bringing paired full-suite SFT final adapters to `90/90` (`official_igsm`, `maze_navigation`, and hard `attribute_constraints` all `30/30`). Replacement eval `3682449_[0-89%4]` released; rows `0..3` are running and rows `4..89` are pending only by `JobArrayTaskLimit`. The eval output directory now exists but has zero pass@k JSONs and zero sample JSONLs, so no aggregation/report regeneration was run. Fresh manifests still have 55 subsets and no missing parquet paths for all three families. Fresh SFT/eval log scans found no unrecovered Traceback, proof-validation failure, OOM/CUDA OOM, context-length failure, quota/no-space, dependency, tokenizer/model-load, vLLM, node-failure, timeout, cancellation, or idle-GPU issue; the only fresh match was a benign tokenizer max-length warning. `3682449_0` was sampling at chunk `25/112` with `92%` GPU utilization, `3682449_1` was entering vLLM startup, and rows `2/3` were still in expected stagger/startup. No partition edit, dependency edit, cancellation, resubmission, broad launch, or report regeneration was made; paired oversight `3685027` was running during this check and next pass `3685570` was begin-time pending.
- 18:43 CEST paired oversight `3685027` completed cleanly (`0:0`) after recording the paired eval release state and scheduling next pass `3685570`. It found zero paired eval JSON/sample outputs, no new severe failures, and made no scheduler or report changes. Its push attempt failed because GitHub SSH connectivity timed out; local paired handoff commit `b51a3b7` remains ahead.
- 14:50 CEST push attempt: local commits were created for this repo (`3228d9d`) and `../synthetic-RLVL-report` (`6334a67`), but pushing both repos failed. `git@github.com:...` timed out on `github.com:22`; fallback pushes through `ssh://git@ssh.github.com:443/...` did not complete within the 45-second timeout. Local branches remain ahead and are ready to push when SSH connectivity returns.
- 14:38 CEST HFSA ablation oversight/report refresh: shortcut-kind eval advanced to `9/24` JSONs after `3674888_5` and `3674888_8` completed; trace-control remains `18/18`, hybrid order remains `15/30`, and conditioned 40k `3674882` has rows `0..5` and `7` complete with `6/8/9/10` running. Regenerated and mirrored the report with `64` PDFs and `55` CSVs; TeX compilation remains unavailable. Sample inspection of shortcut-kind `position` logic `0.8` seed `3409` and `nl_exact` `0.5` seed `3409` confirmed shortcut-neutral prompts, intended wrappers, and normal `<answer>` extraction, with deeper validity/grounding still fragile. Focused live/recent log scans found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, dependency, tokenizer/model-load, vLLM, node-failure, timeout, cancellation, or idle-GPU issue. No partition edit, dependency edit, cancellation, resubmission, or new science launch was made; pending rows are throttle/dependency/begin-time blocked, `puzzle_oversight` is unrelated, and no visible `tjepa_*` or `seqedit_*` jobs were present.
- 14:29 CEST paired full-suite oversight: original SFT rows `3672212_54/58` and replacement rows `3682411_55/57/59/81/82/83` completed cleanly since the 10:45 handoff, increasing paired final adapters to `83/90` (`official_igsm` `30/30`, `maze_navigation` `29/30`, hard `attribute_constraints` `24/30`). Active paired SFT rows are row-56 replacement `3683070_56` and hard-attribute rows `3682411_84..89`; no paired SFT rows are pending. Latest parsed progress is `3683070_56` `8015/10000`, `3682411_84` `6622/10000`, `3682411_85` `5533/10000`, `3682411_86` `4121/10000`, `3682411_87` `1042/10000`, `3682411_88` `475/10000`, and `3682411_89` still in normal startup/stagger. `srun --overlap` checks on `3683070_56`, `3682411_84`, and `3682411_87` showed `100%` GPU utilization, and focused build/SFT/eval log scans found no unrecovered Traceback, proof-validation failure, OOM/CUDA OOM, context-length failure, quota/no-space, dependency, tokenizer/model-load, vLLM, node-failure, timeout, or idle-GPU issue. Full-suite manifests still have 55 subsets and no missing parquet paths for all three families. Bounded materialized-row audit over train-depth-25 and val-depth-50 samples verified matched logic/NL prompts, expected wrappers, answer tags, strict and grounded logic validity, and correct/formatted NL targets; shallow paired NL translation remains `0.0`, so paired NL validity claims remain blocked on translator coverage. Eval `3682449` remains dependency-pending with zero JSONs/sample outputs and no output directory. No partition edit, dependency edit, cancellation, resubmission, report regeneration, or new science launch was made; current paired oversight is `3684369` and next pass `3685027` is begin-time pending. Visible `puzzle_oversight` is unrelated and no visible `tjepa_*` or `seqedit_*` jobs were present.
- 10:45 CEST HFSA ablation oversight/report refresh: trace-control eval completed at `18/18` JSONs after `3682460_8` wrote the final `pseudocode` seed; shortcut-kind eval advanced to `7/24` JSONs after `3674888_7` completed; hybrid order remains `15/30` with `3682461_15/16/17/18` running; conditioned 40k `3674882` has rows `0/1/2` complete, rows `3/4/5/6` running, and `7..14` throttle-pending. Regenerated and mirrored the report with `64` PDFs and `55` CSVs; patched report-builder hybrid table/figure captions and verified `py_compile`. Sample inspection confirmed intended prompt wrappers and answer extraction for pseudocode, shortcut-kind position logic/NL, and hybrid `think_formal`, while strict proof validity and translated validity remain evaluator-sensitive. Focused live/recent log scans found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, dependency, tokenizer/model-load, vLLM, node-failure, timeout, or idle-GPU issue; startup zero-GPU snapshots matched wrapper sleeps. No partition edit, dependency edit, cancellation, resubmission, or new science launch was made. Paired eval `3682449` and conditioned evals remain dependency-pending with no outputs; visible `puzzle_*` jobs are unrelated.
- 10:55 CEST push attempt: local commits were created for this repo and `../synthetic-RLVL-report`, but pushing both repos failed because SSH to `github.com:22` and `ssh.github.com:443` timed out. The local commits remain ready to push when network connectivity is restored.
- 10:47 CEST paired oversight `3683967` completed cleanly after recording the paired full-suite progress below. It made no scheduler edits, report updates, cancellations, resubmissions, or new launches; successor oversight `3684369` remains begin-time pending.
- 10:31 CEST paired full-suite oversight: replacement SFT rows `3682411_78..80` completed cleanly, increasing paired final adapters to `75/90` (`official_igsm` `30/30`, `maze_navigation` `24/30`, hard `attribute_constraints` `21/30`). Active paired SFT rows are original `3672212_54/58`, row-56 replacement `3683070_56`, and replacements `3682411_55/57/59/81/82/83`; rows `3682411_84..89` remain throttle-pending. Latest parsed progress is `3672212_54` `9234/10000`, `3672212_58` `9142/10000`, `3683070_56` `5897/10000`, `3682411_55` `8011/10000`, `3682411_57` `8235/10000`, `3682411_59` `8172/10000`, `3682411_81` `5659/10000`, `3682411_82` `3882/10000`, and `3682411_83` `1553/10000`, with fresh stderr mtimes and no idle-GPU symptom. Focused build/SFT/eval log scans found no unrecovered Traceback, proof-validation failure, OOM/CUDA OOM, context-length failure, quota/no-space, dependency, tokenizer/model-load, vLLM, node-failure, timeout, or idle-GPU signature. Full-suite manifests remain complete with 55 subsets and no missing parquet paths for all three families. Sample audit over paired train-depth-25 and val-depth-50 rows re-confirmed matched logic/NL prompts, correct wrappers, answer tags, strict proof validation, and gold logic validity; sampled gold paired NL targets still have zero NL-to-logic parse/translated validity, so paired NL validity claims remain blocked on the translator backlog item. Eval `3682449` remains dependency-pending and the paired eval output directory still does not exist, so no aggregation/report regeneration was run. No partition edit, dependency edit, cancellation, resubmission, or new science launch was made. Paired oversight `3683562` completed cleanly, pass `3683967` was running at this sample time, and next pass `3684369` was begin-time pending; visible `puzzle_*` jobs are unrelated and no visible `tjepa_*` or `seqedit_*` jobs were present.
- 06:35 CEST HFSA ablation oversight/report refresh: replacement trace-control rows `3682459_16/17` completed cleanly, bringing trace-control artifacts to `17/18` JSONs. `shuffled_nl` is now three-seed complete with OOD correct/translated-joint@16 `0.490/0.000` and depth-50 `0.344/0.000`; samples show high parse coverage but invalid proof order, as intended for the negative control. Remaining trace row is `pseudocode` seed `3409` (`3682460_8`). Shortcut-kind eval rows `3674888_0..3` completed, giving the first `4/24` JSONs: `position` rate `0.5` logic three-seed OOD correct/joint@16 `0.900/0.619`, depth-50 `0.844/0.312`; matched `nl_exact` seed `3407` OOD `0.356/0.300`, depth-50 `0.500/0.438`, so the NL comparison remains one-seed provisional. Sample inspection confirmed shortcut-neutral prompts, expected `<formal>` and `<think>/<proof>` surfaces, normal `<answer>` extraction, and translated-validity failures in some answer-correct NL samples. Conditioned replacement rows `3682457_10/12` completed, with `3682457_13/14` running near `30000` steps; paired progress is recorded in the paired entry below. Focused live/recent log scans found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, dependency, tokenizer/model-load, vLLM, node-failure, timeout, or idle-GPU issue. No partition edit, dependency edit, cancellation, resubmission, or broad new science launch was made. Patched report-builder status prose for dynamic trace/shortcut counts, `py_compile` passed, and regenerated/mirrored the report with `64` PDFs and `55` CSVs; TeX compilation remains unavailable. Push attempts for both repos failed because SSH to `github.com:22` and `ssh.github.com:443` timed out.
- 06:35 CEST paired full-suite oversight: replacement SFT rows `3682411_72..77` completed cleanly, increasing paired final adapters to `72/90` (`official_igsm` `30/30`, `maze_navigation` `24/30`, hard `attribute_constraints` `18/30`). Active paired SFT rows are original `3672212_54/58`, row-56 replacement `3683070_56`, and replacements `3682411_55/57/59/78/79/80`; rows `3682411_81..89` remain throttle-pending. Latest parsed progress is `3672212_54` `7222/10000`, `3672212_58` `7065/10000`, `3683070_56` `3823/10000`, `3682411_55` `5968/10000`, `3682411_57` `6118/10000`, `3682411_59` `6062/10000`, `3682411_78` `2870/10000`, `3682411_79` `1850/10000`, and `3682411_80` `548/10000`, with fresh stderr mtimes and no idle-GPU symptom. Focused build/SFT/eval log scan found no unrecovered Traceback, proof-validation failure, OOM/CUDA OOM, context-length failure, quota/no-space, dependency, tokenizer/model-load, vLLM, node-failure, timeout, or idle-GPU signature. Full-suite manifests remain complete with 55 subsets and no missing parquet paths for all three families. Sample audit over paired train/val rows re-confirmed matched logic/NL prompts, correct wrappers, answer tags, strict proof validation, and gold logic validity; sampled gold paired NL targets still have zero NL-to-logic parse/translated validity, so paired NL validity claims remain blocked on the translator backlog item. Eval `3682449` remains dependency-pending and the paired eval output directory still does not exist, so no aggregation/report regeneration was run. No partition edit, dependency edit, cancellation, resubmission, or new science launch was made. Paired oversight `3683562` is running and scheduled next pass `3683967`; visible `puzzle_*` jobs are unrelated and no visible `tjepa_*` or `seqedit_*` jobs were present.
- 02:38 CEST paired oversight `3683024` completed cleanly (`0:0`) after recording the 02:29 paired full-suite progress and scheduling next pass `3683562`. It found no paired eval JSONs/sample outputs, no new severe failures, and made no additional scheduler/report changes. Push from that job failed because SSH to both `github.com:22` and `ssh.github.com:443` timed out.
- 02:35 CEST HFSA ablation oversight/report refresh: replacement trace-control eval rows `3682459_12/15` and repair rows `3682460_5/6/7` completed cleanly, bringing trace-control artifacts to `15/18` JSONs plus sample JSONLs under `passk_eval/hfsa_ablation_trace_controls_20260525/`. Regenerated and mirrored the report (`64` PDFs, `53` CSVs in both trees; TeX compilation unavailable). Current means: `invalid_logic` OOD correct/formal-joint@16 `0.892/0.427`, depth-50 `0.750/0.146`, with zero grounded validity; repaired `rule_annotated_nl` OOD correct/translated-joint@16 `0.575/0.485`, depth-50 `0.344/0.146`; `pseudocode` two-seed OOD correct/translated-joint@16 `0.406/0.334`, depth-50 `0.188/0.109`; `shuffled_nl` one-seed translated joint `0.000` despite parser coverage. Sample inspection verified intended prompt/answer extraction and trace surfaces: `[rule: ...]` and pseudocode wrappers now parse, shuffled NL is parseable but invalid in order, and invalid-logic citation-free validity is not grounded. Live audit found trace rows `3682459_16/17` and `3682460_8`, shortcut-kind eval rows `3674888_0..3`, hybrid rows `3682461_13/15/16/17`, conditioned rows `3682457_10/12/13/14`, and paired rows still progressing or blocked by throttle/dependencies. Fatal-log scan found no unrecovered Traceback/OOM/CUDA OOM/quota/no-space/dependency/tokenizer/model-load/vLLM/node-failure/timeout/idle-GPU issue; no scheduler edit, partition edit, cancellation, resubmission, or new science launch was made.
- 02:29 CEST paired full-suite progress check: replacement SFT rows `3682411_66..71` completed cleanly, increasing paired final adapters to `66/90` (`official_igsm` `30/30`, `maze_navigation` `24/30`, hard `attribute_constraints` `12/30`). Active paired SFT rows are original `3672212_54/58`, row-56 replacement `3683070_56`, and replacements `3682411_55/57/59/72/73/74`; rows `3682411_75..89` remain throttle-pending. Latest parsed progress is `3672212_54` `5063/10000`, `3672212_58` `4836/10000`, `3683070_56` `1758/10000`, `3682411_55` `3806/10000`, `3682411_57` `3877/10000`, `3682411_59` `3819/10000`, `3682411_72` `6279/10000`, `3682411_73` `5375/10000`, and `3682411_74` `4250/10000`. Focused active/recent paired SFT log scan found no unrecovered Traceback, proof-validation failure, OOM/CUDA OOM, context-length failure, quota/no-space, dependency, tokenizer/model-load, vLLM, node-failure, timeout, or idle-GPU signature. Full-suite manifests remain complete with 55 subsets and no missing parquet paths for all three families. Eval `3682449` remains dependency-pending and the paired eval output directory still does not exist, so no aggregation/report regeneration was run. Pending paired rows are throttle/dependency blocked despite idle compatible `a100` nodes, so no partition edit, cancellation, resubmission, or new science launch was made. Paired oversight `3683024` is running and has scheduled next pass `3683562`; visible `puzzle_*` jobs are unrelated and no visible `tjepa_*` or `seqedit_*` jobs were present.

## 2026-05-30

- 22:58 CEST paired oversight completion: `3682410` completed cleanly (`0:0`) after the already-recorded row-56 recovery. Its log confirms no paired eval JSONs/sample outputs exist yet under `passk_eval/paired_full_suite_sparse_20260528/`, no additional scheduler/report changes were made, and next paired pass `3683024` remains begin-time pending.
- 22:54 CEST trace-control/report refresh: replacement eval row `3682459_14` completed cleanly and wrote `invalid_logic` seed `3409`; `3682459_16` started and `3682459_17` remains throttle-pending. Logs show normal vLLM sampling/W&B sync and only the known Mistral tokenizer warning, with no traceback/OOM/quota/node-failure signature. Report summary now has trace controls `11/18`; `invalid_logic` is two-seed partial with OOD correct/formal-joint@16 `0.906/0.544` and depth-50 `0.734/0.188`. Seed-3409 samples are mostly shallow-correct and citation-free-valid but not grounded-valid; depth-50 sampled rows have zero grounded validity, so treat the result as evaluator-sensitive negative-control evidence. Regenerated and mirrored the report again (`64` PDFs, `53` CSVs in both trees). No scheduler, partition, dependency, cancellation, or resubmission edit was made.
- 22:39 CEST HFSA ablation oversight/report refresh: live `squeue`/expanded `sacct` showed paired row-56 replacement `3683070_56` running, paired eval `3682449` waiting on `3681398/3683070/3681586/3682411`, conditioned replacement row `3682492_5` completed cleanly so `3674882` now waits only on `3682457`, shortcut-kind eval `3674888` waits only on replacement SFT `3682458`, ablation oversight `3682409` running, and next passes `3683023/3683024` begin-time pending; paired oversight `3682410` was confirmed complete in the later 22:58 check. Active log scan found no unrecovered Traceback, CUDA OOM, quota/no-space, dependency-never-satisfied, tokenizer/model-load, vLLM, node-failure, timeout, or idle-GPU signature; OOM matches were benign accelerate memory-reserve INFO lines or oversight prompt text. New report-ingested outputs are trace-control `invalid_logic` seed `3408` and hybrid `think_formal` train-1-to-25 seed `3409`. `invalid_logic` seed `3408` has OOD correct/formal-joint@16 `0.856/0.519` and depth-50 `0.625/0.094`, but sampled depth-50 generations are mostly invalid/wrong, so treat as provisional. Hybrid `think_formal` train-1-to-25 is now two-seed partial with OOD correct/formal-joint/translated-joint@16 `0.584/0.188/0.459` and depth-50 `0.344/0.000/0.172`; samples show intended `<think>` then `<formal>` formatting and fragile formal validity. Patched report-builder prose for the current hybrid partials, `py_compile` passed, regenerated/mirrored the report, and verified `64` PDFs and `53` CSVs in both report trees. No new scheduler edit, partition edit, cancellation, resubmission, or broad science launch was made in this pass.
- 22:31 CEST paired full-suite idle-row recovery: original SFT row `3672212_56` (`maze_navigation`, `logic`, train-1-to-25, seed `3409`) had no stderr updates since 16:59 CEST, GPU utilization `0%` with about `58GB` still allocated, and the Python process waiting in `futex_do_wait`. Canceled that row, submitted targeted replacement `3683070_[56%1]` with `--exclude=a0831`; it launched on `a0833`. Rewired eval `3682449` from `afterok:3681398:3681503:3681586:3682411` to `afterok:3681398:3683070:3681586:3682411`. Replacement `3682411_60..65` completed, raising paired SFT final adapters to `60/90` (`official_igsm` `30/30`, `maze_navigation` `24/30`, hard `attribute_constraints` `6/30`). Eval outputs remain at zero JSON/sample files, so no aggregation or report regeneration was run. No partition widening was needed; visible `puzzle_*` jobs are unrelated.
- 18:47 CEST dependency repair follow-up: a final expanded dependency check showed conditioned-dual 30k original row `3674881_5` had also failed (`3681573`, exit `1:0`), leaving 40k chunk `3674882` effectively blocked. Submitted only missing row-5 replacement `3682492_[5%1]` with `MAX_STEPS=30000,EVAL_STEPS=30001` and rewired `3674882` to `afterok:3681529:3682492:3682457`; by 18:52, `3682492_5` was running and `3674882` was dependency-pending only on live/replacement rows. No partition widening was needed. Regenerated/mirrored the report after correcting the trace-control replacement note; verification found `64` PDFs, `64` unique PDF include references with zero missing, and `53` CSVs in both report trees.
- 18:40 CEST HFSA ablation recovery/report update: expanded `sacct` showed new failures in conditioned-dual 30k (`3674881_3`, `3674881_6`, `3674881_7..14`), shortcut-kind SFT (`3674887_22`), trace-control eval (`3661119_12`, `3661119_14`, `3661119_15..17`), trace repair (`3680004_5..8`), and hybrid eval (`3670783_13`, `3670783_15..29`). Focused log scans found no traceback, OOM/CUDA OOM, quota/no-space, dependency-never-satisfied, tokenizer/model-load, or vLLM signature; failures looked like interrupted rows plus signal-53 killed array tasks, with no single bad node implicated. Submitted targeted replacements `3682457_[3,6-14%4]`, `3682458_[22%1]`, `3682459_[12,14-17%3]`, `3682460_[5-8%3]` with `FORCE_PASSK_EVAL=1`, and `3682461_[13,15-29%4]`. Initial dependency rewires were followed by the 18:47 correction above; no partition edit was appropriate. Hybrid `think_formal` train-1-to-20 is now three-seed complete with mean OOD correct/formal-joint/translated-joint@16 `0.434/0.028/0.148` and depth-50 correct/formal-joint@16 `0.469/0.000`; sample inspection confirmed the intended `<think>` then `<formal>` surface and normal answer extraction, with weak depth-50 validity. Patched the report builder to filter the stale pre-fix `rule_annotated_nl` seed-3409 artifact until repair overwrite, `py_compile` passed, regenerated/mirrored the report, and verified `64` PDFs, `64` unique PDF include references with zero missing after LaTeX-escape normalization, and `53` CSVs in both report trees. TeX compilation was not run because `latexmk`/`pdflatex` are unavailable.
- 18:32 CEST paired full-suite recovery: original SFT array `3672212` advanced rows `48..53` to complete, then rows `55`, `57`, and `59` failed with exit `1:0` and no traceback/OOM/quota/validation signature, while rows `60..89` immediately canceled/failed with signal `53`; rows `54`, `56`, and `58` remained running on `maze_navigation` train-1-to-25. Submitted targeted replacement SFT `3682411_[55,57,59-89%6]`, with rows `55/57/59/60/61/62` running and `63..89` pending by throttle. Canceled stale eval `3672213` because its `afterok:3672212_*` dependency could never satisfy, then submitted replacement eval `3682449_[0-89%4]` depending on original running job IDs `3681398/3681503/3681586` plus replacement SFT `3682411`. Build manifests remain complete at 55/55 subsets for all three paired families, and there are still zero eval JSONs/sample JSONLs. Sample audit over train-depth-25 and val-depth-50 rows for `official_igsm`, `maze_navigation`, and hard `attribute_constraints` verified matched logic/NL prompts, correct wrappers, answer tags, and strict proof validation; iGSM citation-free validation still fails on cited arithmetic substitutions as expected. No partition edit or report regeneration was made.
- 14:36 CEST HFSA ablation oversight: original trace-control `shuffled_logic` eval rows `3661119_9..11` completed cleanly and wrote three pass@k JSONs plus sample JSONLs under `$WORK/synthetic-RLVL/passk_eval/hfsa_ablation_trace_controls_20260525/`. Three-seed mean OOD correct/formal-joint@16 is `0.690/0.002`, and depth-50 correct/formal-joint@16 is `0.510/0.000`. Sample inspection across seeds/depths verified normal `<question>` prompts, `<formal>` generations, and `<answer>` extraction, but higher-depth proofs are usually invalid or unparsable fragments; depth-50 can still be answer-correct while failing citation-free validity, so this is a negative-control result rather than valid reasoning. Regenerated `analysis/logic_cot_report_2026-05-25/` and mirrored it to `../synthetic-RLVL-report`; verification found `64` generated PDFs, `64` PDF include references, and `53` CSV tables in both trees. TeX compilation was not run because `latexmk`/`pdflatex` are unavailable. Fresh `squeue`/expanded `sacct`/log/output scan found no unrecovered Traceback/OOM/CUDA OOM/quota/no-space/dependency/tokenizer/model-load/vLLM/node-failure/timeout/idle-GPU issue. Active rows: repair `3680004_3..5` chunks `50/50/49` of `56`, original trace `3661119_12..14` running, hybrid `3670783_11..14` at sampling/scoring state `112/99/89/90` of `112`, shortcut-kind `3674887_18..20` at `6798/6231/5167` of `10000`, conditioned-dual 30k `3674881_0..3` running from checkpoint-20000. Pending monitored rows are throttle/dependency/begin-time blocked despite idle `a100` nodes, so no partition edit, dependency edit, cancellation, resubmission, or new science launch was made. Current ablation oversight is `3680038`, with next pass `3680772` begin-time pending; visible `puzzle_*` jobs are unrelated, and no visible `tjepa_*` or `seqedit_*` jobs were present.
- 14:31 CEST paired full-suite oversight: build `3672195_0..2` still has complete 55-subset manifests with no missing parquet paths. SFT rows `3672212_0..47` have final adapters; rows `48..53` are still running on `maze_navigation` train-1-to-20 with latest parsed progress `8829/8511/8479/8728/8676/8556` of `10000` and all six have `checkpoint-5000`; rows `54..89` are pending by `JobArrayTaskLimit`. Eval `3672213` remains dependency-pending with no output directory and zero JSON/sample outputs. `squeue`/expanded `sacct`/focused log scans found no failed, node-failed, timed-out, canceled, nonzero-exit, Traceback, proof-validation, OOM/CUDA OOM, context, quota/no-space, dependency, tokenizer/model-load, vLLM, or idle-GPU issue; OOM matches are benign accelerate memory-reserve INFO lines. Sample materialized-row audit over train-1-to-20 and val-depth-50 rows for all three paired families verified matching logic/NL prompts, expected `<formal>`/`<think>` wrappers, final answer tags, and strict proof validation; iGSM depth-50 citation-free validity still fails while strict validation passes because cited arithmetic substitutions are required. Pending paired rows are throttle/dependency-blocked despite idle `a100` nodes, so no partition edit, dependency edit, cancellation, resubmission, report regeneration, or new science launch was made. Oversight `3680037` completed, current paired oversight is `3680039`, and next paired pass `3680777` is begin-time pending. Visible `puzzle_*` jobs are unrelated; no visible `tjepa_*` or `seqedit_*` jobs were present.
- 10:33 CEST HFSA ablation oversight: checked live `squeue`, expanded `sacct`, pending dependencies/partition availability, focused logs, output roots, eval JSON/sample artifacts, and selected sample generations for wordified, trace-control, shortcut-rate, hybrid-order, conditioned-dual 50k, shortcut-kind, and paired full-suite chains. No unrecovered Traceback/OOM/CUDA OOM/quota/no-space/dependency/tokenizer/model-load/vLLM/node-failure/timeout/idle-GPU issue was found. Active progress: trace repair `3680004_3..5` chunk `18/56`; hybrid `3670783_11..14` chunks `96/76/66/63` of `112`; conditioned-dual `3674880_12..14` `14585/10001/11007` of `20000`; shortcut-kind `3674887_15..17` `8424/7922/7107` of `10000`; paired `3672212_48..53` `6256/5929/5906/6014/5977/5833` of `10000`. Row `3674880_13` has stale logs, but `srun --jobid=3679320` showed `97%` GPU utilization and about `63GB` used, so no idle recovery was needed. Pending rows are throttle/dependency/begin-time blocked despite idle A100 nodes; no partition edit, dependency edit, cancellation, resubmission, or new science launch was made. Sample inspection confirmed wordified depth-50 generations often remain in `<formal>` but drift into duplicate predicates/invalid derivations, and old rule-annotated NL samples still show the stale translator artifact, so repair eval outputs are required before using those metrics. Patched `scripts/analysis/build_logic_cot_report.py` so hybrid-order report parsing uses `formal_think` rather than nonexistent `think_natural` and labels `think_formal` as NL-then-formal; `py_compile` and report regeneration passed. Mirrored the report to `../synthetic-RLVL-report`; verification found `64` generated PDFs, `64` PDF include references, zero missing references, and `53` CSV tables in both report trees. TeX compilation was not run because `latexmk`/`pdflatex` are unavailable.
- 10:30 CEST paired full-suite oversight: build `3672195_0..2` still has complete 55-subset manifests with no missing parquet paths. SFT rows `3672212_0..47` are complete with final adapters; rows `48..53` are running on `maze_navigation` train-1-to-20, all have `checkpoint-5000`, and latest parsed progress is `6185/5867/5836/5944/5900/5764` of `10000`; rows `54..89` are pending by array throttle. Eval `3672213` remains dependency-pending with zero JSON/sample outputs and no output directory. `sacct`/log scans found no failed, node-failed, timed-out, canceled, nonzero-exit, Traceback, validation, OOM/CUDA OOM, context, quota/no-space, dependency, tokenizer/model-load, vLLM, or idle-GPU issue; pending rows are throttle/dependency-blocked despite idle `a100` nodes, so no partition edit or resubmission was made. Sample materialized-row audit across train-1-to-5 and val-depth-50 for all paired families verified matching logic/NL prompts, expected target wrappers, final answer tags, and strict proof validation; note that iGSM depth-50 citation-free validation can fail while strict validation passes because cited arithmetic substitutions matter. Backlog analysis remains deferred until `3672213` completes. Current paired oversight is `3680037`, with next pass `3680039` begin-time pending.
- 10:28 CEST oversight automation update: added permanent `AGENTS.md` oversight discipline requiring regular Codex oversight passes to read/update the active plan and backlog, inspect Slurm state/logs/outputs/partitions, inspect representative sample generations across seeds/depths/templates/success/failure cases, question evaluator assumptions before accepting metrics, create justified figures/tables, regenerate/mirror the LaTeX report when results change, and submit only the smallest safe triggered or recovery jobs. Strengthened `scripts/slurm/codex/hfsa_ablation_oversight_2026-05-29.slurm` and `scripts/slurm/codex/paired_full_suite_oversight_2026-05-28.slurm` with those instructions. Validation: `bash -n` passed for both wrappers. Canceled stale queued prompt snapshots `3679878` and `3679358`; submitted fresh plan-driven oversight jobs `3680036` and `3680037`. Both started on `a100` and scheduled next passes `3680038` and `3680039`.
- 10:13 CEST trace-control evaluator fix: inspected `rule_annotated_nl` sample generations and found correct-looking controlled traces, e.g. `a is teal. [rule: R]`, but `nl_logic_parse` was zero because the NL-to-FOL translator treated the trailing `[rule: ...]` text as part of the attribute. Patched `synthrlvl/natural_logic.py` to unwrap `[rule: ...]` annotations and pseudocode `step_i: derive "..." using ...` wrappers before translation. Added regression coverage in `tests/test_training_stack.py`; verification passed (`26 passed`). Canceled stale running pseudocode eval rows `3661119_6..8` and submitted repair eval `3680004_[3-8%3]` with `FORCE_PASSK_EVAL=1` to overwrite stale `rule_annotated_nl` rows `3..5` and rerun pseudocode rows `6..8`. Existing `rule_annotated_nl` translated-validity metrics from `3661119_3..5` should not be used as evidence until the repair eval completes.
- 09:50 CEST report/status refresh: regenerated `analysis/logic_cot_report_2026-05-25/` with active experiment artifact status, trace-control tables/plot, hybrid-order partial tables/plot, shortcut-kind status, and conditioned-dual 50k status; mirrored the full generated bundle into `../synthetic-RLVL-report`. Verification found `64` generated PDFs, `64` `\includegraphics` PDF references, zero missing references, and `53` CSV tables in both report trees. `latexmk`/`pdflatex` are unavailable on this node, so compilation was not run. Fresh queue/log scan found no unrecovered monitored job failures; paired maze rows are at `5767/5447/5424/5495/5461/5326` of `10000`, trace-control eval rows `6..8` are sampling chunks `38/56`, `34/56`, and `25/56`, hybrid eval rows `11..14` are sampling chunks `92/112`, `71/112`, `58/112`, and `54/112`, conditioned-dual 20k rows `12..14` are at `12676/20000`, `10001/20000`, and `10001/20000`, and shortcut-kind rows `15..17` are at `6658/6116/5299` of `10000`. Pending monitored rows are still throttle/dependency/begin-time blocked, so no partition edit, resubmission, dependency edit, or cancellation was made.
- 09:41 CEST HFSA ablation oversight: rechecked live `squeue`, expanded array-row `sacct`, focused logs, output roots, and `a100` partition state for wordified `3674875/3674876`, conditioned-dual `3674879..3674885`, shortcut-kind `3674886..3674888`, trace-control `3661118/3661119`, shortcut-rate `3671431/3671432`, and hybrid `3670782/3670783`. No unrecovered Traceback/OOM/CUDA OOM/quota/no-space/dependency/tokenizer/model-load/vLLM/node-failure/timeout/idle-GPU issue was found. Active rows are progressing: trace eval `3661119_6..8` at `36/31/21` of `56` chunks done; hybrid eval `3670783_11..14` at `90/69/55/50` of `112` chunks done; conditioned-dual 20k `3674880_12..14` at `12224/10001/10001` of `20000`; shortcut-kind SFT `3674887_15..17` at `6232/5686/4887` of `10000`. Pending monitored rows are throttle/dependency/begin-time blocked, so no resubmission, cancellation, dependency edit, or partition edit was made. Current ablation oversight `3679095` is running and next pass `3679878` is begin-time pending; visible `puzzle_*` jobs are unrelated and no visible `tjepa_*` or `seqedit_*` jobs were present.
- 09:29 CEST full job-state audit: checked live `squeue`, expanded `sacct`, active logs, output roots, and partition state for paired full-suite `3672212/3672213`, trace-control `3661118/3661119`, shortcut-rate `3671431/3671432`, hybrid `3670782/3670783`, wordified `3674875/3674876`, conditioned-dual `3674879..3674885`, shortcut-kind `3674886..3674888`, and oversight jobs `3678051/3678335/3679095/3679358`. No failed, node-failed, timed-out, canceled, or nonzero-exit monitored jobs were found. Pending monitored rows are blocked by array throttles, dependencies, or begin time; no partition edit, dependency edit, cancellation, resubmission, or recovery job was needed.
- 09:29 CEST results update: shortcut-rate `0.3` eval is now fully complete with matched logic and NL rows. Logic mean OOD correct/joint@16 is `0.892/0.598` and depth-50 correct/joint@16 is `0.844/0.375`; NL mean OOD correct/translated-joint@16 is `0.588/0.571` and depth-50 correct/translated-joint@16 is `0.458/0.438`. Across rates `0.3/0.5/0.8`, NL depth-50 joint declines `0.438 -> 0.312 -> 0.146`, while logic is `0.375 -> 0.375 -> 0.417`.
- 09:29 CEST partial ablation update: trace-control `rule_annotated_nl` is three-seed complete with OOD correct/translated-joint@16 `0.579/0.000` and depth-50 correct/translated-joint@16 `0.365/0.000`. Hybrid `think_formal` train-1-to-15 is three-seed complete with OOD correct/formal-joint/translated-joint@16 `0.353/0.111/0.111`; train-1-to-20 has two seeds complete with OOD correct/formal-joint/translated-joint@16 `0.419/0.016/0.078`. Treat hybrid train-1-to-20 and train-1-to-25 as partial.
- 09:29 CEST live progress update: paired full-suite SFT rows `48..53` are still running with latest progress `5517/5191/5164/5233/5189/5051` of `10000`; trace-control eval rows `6..8` are running at about `36/56`, `30/56`, and `16/56` vLLM chunks; hybrid eval rows `11..14` are running at about `90/112`, `68/112`, `52/112`, and `48/112` chunks; conditioned-dual 20k rows `12..14` are running at `11676/20000`, `10001/20000`, and `10001/20000`; shortcut-kind SFT rows `15..17` are running at `5724/5188/4385` of `10000`.
- 09:29 CEST regenerated `analysis/logic_cot_report_2026-05-25/` and mirrored the bundle into `../synthetic-RLVL-report`. Verification found `62` generated PDFs, `62` `\includegraphics` PDF references, zero missing references, and `48` CSV tables in the external report repo. Local TeX compilation was not run because `latexmk`/`pdflatex` are unavailable on this node.
- 08:06 CEST paired full-suite oversight: build `3672195_0..2` remains complete with all three manifests at 55/55 paths; SFT rows `3672212_0..47` are complete with final adapters; rows `48..53` are running on `maze_navigation` train-1-to-20 with progress `4581/4246/4220/4236/4195/4072` of `10000`; rows `54..89` are pending by array throttle. Eval `3672213` is still dependency-pending with zero JSON outputs and no output directory. `sacct` shows no failed, node-failed, timed-out, canceled, or nonzero-exit paired rows; focused SFT fatal-log/progress scan found no unrecovered Traceback/OOM/CUDA OOM/context/quota/dependency/tokenizer/model-load/vLLM/node-failure/timeout/idle-GPU issue, with OOM matches limited to benign accelerate memory-reserve INFO lines. No resubmission, cancellation, dependency edit, or partition edit was made. Oversight `3678335` is running and next pass `3679358` is begin-time pending. Visible `puzzle_*` jobs are unrelated; no visible `tjepa_*` or `seqedit_*` jobs were present.
- 05:42 CEST HFSA ablation oversight: checked live `squeue`, expanded `sacct`, focused logs, and output roots for wordified `3674875/3674876`, conditioned-dual `3674879..3674885`, shortcut-kind `3674886..3674888`, trace-control `3661118/3661119`, shortcut-rate `3671431/3671432`, and hybrid `3670782/3670783`. No unrecovered Traceback/OOM/CUDA OOM/quota/no-space/dependency/tokenizer/model-load/vLLM/node-failure/timeout/idle-GPU issue was found, so no resubmission, cancellation, dependency edit, or partition edit was made. New partial outputs since the prior ablation pass: trace-control `terse_nl` is three-seed complete with mean OOD correct/translated-joint@16 `0.348/0.277`, and hybrid `think_formal` train-1-to-15 now has two seeds with mean OOD correct/formal-joint/translated-joint@16 `0.332/0.117/0.111`. Regenerated the in-repo report and mirrored it to `../synthetic-RLVL-report`; the report diff adds the shortcut-rate `0.3` logic rows to the shortcut-rate table/plot. Current ablation oversight is `3678051`, with next pass `3679095` begin-time pending.
- 04:05 CEST paired full-suite oversight: build `3672195_0..2` remains complete with all three manifests at 55/55 paths; SFT rows `3672212_42..47` completed since the prior paired pass, so rows `0..47` have final adapters. Rows `48..53` are running on `maze_navigation` train-1-to-20 with progress `1873/1574/1536/1416/1380/1256` of `10000`; rows `54..89` are pending by array throttle. Eval `3672213` is still dependency-pending with zero JSON outputs and no output directory. Focused fatal-log/progress scan found no unrecovered Traceback/OOM/CUDA OOM/context/quota/dependency/tokenizer/model-load/vLLM/node-failure/timeout/idle-GPU issue, so no resubmission, cancellation, dependency edit, or partition edit was made. Oversight `3677873` is running and next pass `3678335` is begin-time pending.
- 01:41 CEST HFSA ablation oversight: checked `squeue`, expanded `sacct`, logs, and output roots for wordified `3674875/3674876`, conditioned-dual `3674879..3674885`, shortcut-kind `3674886..3674888`, trace-control `3661118/3661119`, shortcut-rate `3671431/3671432`, hybrid `3670782/3670783`, and paired full-suite `3672212/3672213`. No unrecovered Traceback/OOM/CUDA OOM/quota/no-space/dependency/tokenizer/model-load/vLLM/node-failure/timeout/idle-GPU issue was found, so no resubmission, cancellation, dependency edit, or partition edit was made. New partial outputs: shortcut-rate `0.3` logic rows `3671432_0..2` wrote 3 JSONs with mean OOD correct/joint@16 `0.892/0.598`; trace-control `3661119_1` wrote `terse_nl` seed `3408`; hybrid `3670783_6` wrote `think_formal` train-1-to-15 seed `3407`. Paired SFT rows `3672212_42..47` completed and rows `48..53` are running; full-suite eval remains dependency-pending with zero JSONs.
- 00:07 CEST paired full-suite oversight: build `3672195_0..2` remains complete with all three manifests at 55/55 paths; SFT rows `3672212_0..41` are complete with final adapters, rows `42..47` are running on `maze_navigation` train-1-to-15 with all six past `checkpoint-5000` and latest progress `9139/8790/8586/8744/8452/8489` of `10000`, and rows `48..89` are pending by array throttle. Eval `3672213` is still dependency-pending with zero JSON outputs and no output directory. Focused SFT fatal-log scan found no unrecovered Traceback/OOM/CUDA OOM/context/quota/dependency/tokenizer/model-load/vLLM/node-failure/timeout/idle-GPU issue, so no resubmission, cancellation, dependency edit, or partition edit was made. Oversight `3677238` is running and next pass `3677873` is begin-time pending.

## 2026-05-29

- Split the live handoff into `docs/current_system_state.md`, `docs/running_experiments.md`, and `docs/experiment_backlog.md`; preserved the old long handoff in `docs/operational_history_2026-05-29.md`.
- Created the external report repo structure in `../synthetic-RLVL-report` for the ongoing LaTeX report.
- Added project instructions to update and push both the experiment repo and report repo after code/docs/report changes when auth and network permit.
- Added Slurm housekeeping guidance: when jobs are pending, check compatible freer partitions and use `scontrol update JobId=<jobid> Partition=<partition1,partition2>` when safe.
- Removed disposable local Python caches, old tracked smoke/probe artifacts, and old tracked/ignored Slurm logs from the repo tree. Local `logs/` was reduced from about 4.2 GB to about 1.3 GB by retaining logs matching the active job IDs in `docs/running_experiments.md`.
- Active checkpoints and active `$WORK` experiment outputs were left in place; `$WORK/synthetic-RLVL/tmp` is currently dominated by active hybrid-order merged checkpoints.
- Checked partitions during housekeeping. Pending active jobs were blocked by array throttles, dependencies, or begin times rather than partition availability, so no partition widening was applied.
- Verification: `git diff --check` passed; `tests/test_hfsa_shortcut_kinds.py`, `tests/test_logic_symbol_padded_template.py`, and `tests/test_logic_engine.py` passed (`15 passed`). TeX compilation for the external report was not run because `latexmk`/`pdflatex` are not installed on this node.
- Updated report discipline after user clarification: the generated in-repo LaTeX report at `analysis/logic_cot_report_2026-05-25/logic_cot_report_2026-05-25.tex` is the source report, and `../synthetic-RLVL-report` should mirror the full generated bundle for GitHub-facing review.
- Regenerated the in-repo LaTeX report with all current generated PDF figures embedded, an executive insights section, qualitative OOD samples, and an artifact index for CSV/PDF/Markdown supplements. Verification found `62` generated PDFs and `62` `\includegraphics` PDF references; none were missing.
- Mirrored the full generated report bundle into `../synthetic-RLVL-report` after user clarification: `main.tex`, all figures, all CSV tables, and Markdown generation supplements.
- 13:01 CEST ablation oversight: checked `squeue`, `sacct`, active logs, and output roots for wordified `3674875/3674876`, conditioned-dual 50k `3674879..3674885`, shortcut-kind `3674886..3674888`, trace-control `3661118/3661119`, shortcut-rate `3671431/3671432`, and hybrid `3670782/3670783`. No unrecovered failure signatures were found; conditioned-dual `3674879_5` newly completed, shortcut-kind build roots are present, and no partition edit or resubmission was needed.
- 16:05 CEST paired full-suite oversight: build `3672195_0..2` and SFT rows `3672212_0..41` are cleanly complete, rows `42..47` are running, rows `48..89` are pending by throttle, eval `3672213` is still dependency-pending with zero JSON outputs, and next oversight `3676517` is scheduled. Focused log scan found no unrecovered failure signatures; `a100` had idle compatible nodes but pending paired jobs were throttle/dependency-blocked, so no resubmission or partition edit was needed.
- 17:42 CEST ablation oversight: checked `squeue`, `sacct`, logs, and output roots for the 2026-05-29 HFSA ablation chains. Wordified SFT `3674875_0..2` completed and eval `3674876_0..2` is running; conditioned-dual 10k chunk `3674879_0..10` completed with rows `11..14` running; shortcut-kind build `3674886_0..3` and SFT rows `3674887_0..2` completed, rows `3..5` are running; trace-control row `3661118_17`, shortcut-rate rows `3671431_3,5`, and hybrid eval rows `3670783_4..7` are running. Focused log scan found no unrecovered Traceback/OOM/CUDA OOM/quota/no-space/dependency/tokenizer/model-load/vLLM/node-failure/timeout/idle-GPU issue, so no resubmission or partition edit was made.
- 17:42 CEST partial hybrid readout: `3670783_0..3` wrote four JSONs under `$WORK/synthetic-RLVL/passk_eval/hfsa_hybrid_order_full_20260525/`. Completed `think_formal` train-1-to-5 rows average OOD correct@16 `0.480`, formal citation-free joint@16 `0.022`, translated-NL joint@16 `0.297`, depth-50 correct@16 `0.219`, and depth-50 joint@16 `0.000`; `think_formal` train-1-to-10 seed-3407 has OOD correct@16 `0.537`, formal joint@16 `0.275`, translated-NL joint@16 `0.300`, and depth-50 joint@16 `0.000`.
- 17:49 CEST refresh: shortcut-kind SFT advanced cleanly; rows `3674887_0..4` are complete, rows `5..6` are running, and rows `7..23` remain pending by array throttle. Fresh row-4/row-6 log scan found only benign tokenizer/rope/quota/allocator-warning text, so no action was needed.
- Regenerated `analysis/logic_cot_report_2026-05-25/` after the partial hybrid readout and mirrored the bundle into `../synthetic-RLVL-report`. Local TeX compilation was not run because `latexmk`/`pdflatex` are unavailable on this node.
- 20:05 CEST paired full-suite oversight: build `3672195_0..2` remains cleanly complete with all three manifests at 55/55 paths; SFT rows `3672212_0..41` are complete, rows `42..47` are running on `maze_navigation` train-1-to-15 and making optimizer progress, and rows `48..89` remain pending by array throttle. Eval `3672213` is still dependency-pending with zero JSON outputs and no output directory. Focused SFT log scan found no unrecovered Traceback/OOM/CUDA OOM/context/quota/dependency/tokenizer/model-load/vLLM/node-failure/timeout/idle-GPU issue, so no resubmission or partition edit was made. Oversight `3675380` completed, `3676517` is running, and next pass `3677238` is begin-time pending.
- 21:42 CEST ablation oversight: wordified length-control eval `3674876_0..2` completed cleanly and wrote 3 JSONs. Three-seed wordified logic underperforms compact logic and `nl_exact`: OOD correct/joint@16 `0.508/0.323`, depth-50 correct/joint@16 `0.344/0.094`. Regenerated `analysis/logic_cot_report_2026-05-25/` with wordified tables/plots and mirrored the bundle into `../synthetic-RLVL-report`; local TeX compilation was not run because `latexmk`/`pdflatex` are unavailable.
- 21:42 CEST ablation Slurm audit: trace-control SFT `3661118_0..17`, shortcut-rate `0.3` SFT `3671431_0..5`, conditioned-dual 10k chunk `3674879_0..14`, and shortcut-kind SFT rows `3674887_0..7` are complete. Running rows are trace eval `3661119_0..2`, shortcut eval `3671432_0..2`, hybrid eval `3670783_6..9`, conditioned-dual 20k `3674880_0..3`, and shortcut-kind SFT `3674887_8..10`; remaining monitored rows are pending by array throttle or dependencies. Focused log scan found no unrecovered Traceback/OOM/CUDA OOM/quota/no-space/dependency/tokenizer/model-load/vLLM/node-failure/timeout/idle-GPU issue, so no resubmission, cancellation, dependency edit, or partition edit was made. Visible `puzzle_*` jobs are unrelated.
- 21:42 CEST hybrid partial readout: `think_formal` train-1-to-10 is now three-seed complete with OOD correct@16 `0.490`, formal joint@16 `0.249`, translated-NL joint@16 `0.296`, depth-50 correct@16 `0.354`, and depth-50 joint@16 `0.000`. Treat hybrid-order as partial until rows `10..29` finish. Paired-suite oversight `3676517` completed and next `3677238` remains begin-time pending.
- 21:50 CEST queue refresh: shortcut-kind SFT row `3674887_8` completed, rows `9..11` are running, and rows `12..23` remain pending by array throttle. No failure signature, resubmission, dependency edit, or partition edit was found.
- 2026-06-03 13:58 CEST: Queue/backlog refresh found active new chains unchanged: batch-size SFT `3695197_0..2`, typed maze build `3695237`, and semantic iGSM build `3695525` running, with dependent evals pending. Hybrid-order eval is now complete at `30/30`, conditioned-dual checkpoint eval is complete at `30/30`, and conditioned-dual final eval has `17/30` JSONs with one row still running. Old paired-family recovery `3694619_[40-89%4]` failed quickly because it still merged adapters into `$WORK/tmp` and hit quota; do not resubmit unchanged.
- 2026-06-04 09:42 CEST: Live refresh found semantic iGSM build/SFT complete and eval active at `3/30` JSONs; early train-1-to-5 logic rows have high train-band internal joint but zero strict grounded joint. Sample inspection shows semantic labels are now present, but current strict grounded scoring remains brittle to generated variable handles and premise order. Typed maze build is complete with `9/30` SFT finals and eval pending. Batch-size bsz16 logic row OOMed on A100-80GB, so pending bsz16 rows `3695197_7/11` and dead eval `3695199` were canceled; replacement feasible-row eval `3698280_[0-2,4-6,8-10,12-14%4]` was submitted with `afterany:3695197`.
- 2026-06-04 10:04 CEST: Patched HFSA batch-size SFT script so only bsz16 rows use microbatch `8` with `grad_accum=2`, preserving effective batch 16 while avoiding the physical bsz16 OOM. Canceled feasible-only eval `3698280`; submitted bsz16 recovery SFT `3698380_[3,7,11%1]` and full eval `3698381_[0-15%4]` with dependency `afterany:3695197:3698380`.
- 2026-06-04 10:18 CEST: Widened bsz16 effective-batch recovery array `3698380` from `%1` to `%3` with `scontrol update JobId=3698380 ArrayTaskThrottle=3`; Slurm accepted the edit and pending rows `7/11` are eligible when compatible GPUs free up.
- 2026-06-04 13:12 CEST: Live refresh found semantic iGSM eval at `6/30` JSONs with rows `6/7/8` running, typed maze SFT at `12/30` finals with rows `12/13/14` running, and batch-size eval still pending. Batch-size row `3695197_2` (`bsz8 logic`) timed out at the 24h walltime around step `6207` but left `checkpoint-5000`; patched the SFT script to pass `train.resume_from_checkpoint` from `SFT_RESUME_FROM_CHECKPOINT`, submitted row-2 resume recovery `3698877_[2%1]`, canceled stale eval `3698381`, and submitted full eval `3698878_[0-15%4]` after `afterany:3695197:3698380:3698877`.
- 2026-06-05 09:19 CEST: Live refresh found semantic iGSM eval at `22/30` JSONs. Original iGSM eval rows `24..28` failed during LoRA merge after `$HPCVAULT` hit disk/file quota; safe cleanup removed stale merged eval dirs and completed-run intermediate checkpoints, then replacement eval `3702073_[24-28%3]` was submitted and started rows `24..26`. Provisional iGSM readout: logic has nonzero internal valid@16 but zero strict grounded joint@16 through train-1-to-20; NL parser coverage is near-complete on completed rows but translated joint@16 remains zero, so the remaining issue is grounding/equivalence rather than parser coverage.
- 2026-06-05 09:19 CEST: Batch-size recovery was repaired again. Row `3695197_6` (`bsz8 nl_exact`) timed out with `checkpoint-5000`, so recovery `3702079_[6%1]` was submitted with `SFT_RESUME_FROM_CHECKPOINT=auto`. The first bsz16 effective-batch recovery `3698380_[3,7,11]` was canceled near walltime because it had no checkpoints; checkpointable replacement `3702080_[3,7,11%3]` was submitted with `SAVE_STEPS=1000,SAVE_TOTAL_LIMIT=5`. Stale eval `3698878` was canceled and full eval `3702081_[0-15%4]` now waits on `afterany:3695197:3698877:3702079:3702080`.
- 2026-06-05 09:19 CEST: Typed maze SFT is `14/30` final adapters with rows `12/15/16` running and eval `3695239` still dependency-pending at `0` JSONs. Row `12` is near walltime and is the current maze watch item.
- 2026-06-06 10:01 CEST: Semantic iGSM original/recovery eval reached `30/30` JSONs. Sample inspection found the current zero NL translated-validity is an evaluator alias issue: generated semantic NL traces parse and use meaningful lines such as `From the definition of ... (s)`, but generated handles need not match gold formal handles. Patched `synthrlvl/natural_logic.py` to canonicalize iGSM NL definition-line handles by semantic quantity name while preserving explicit `intermediate calculation` helper handles; added regression coverage for alias canonicalization. Verification: `tests/test_training_stack.py` and `tests/test_paired_synthetic_datasets.py` passed (`46 passed`), and `py_compile` passed. Canceled first forced NL re-eval `3705801` because some rows started before the helper-line fix; submitted clean forced NL-only iGSM re-eval `3705807_[3-5,9-11,15-17,21-23,27-29%4]`.
- 2026-06-06 10:01 CEST: Typed maze and batch-size recoveries were refreshed. Maze row `12` timed out with `checkpoint-5000`, row `22` node-failed on `a0531`, and recovery `3705793_[12,22%2]` is running with `--exclude=a0531` after adding `SFT_RESUME_FROM_CHECKPOINT` support to `scripts/slurm/sweeps/sft/paired_maze_typed_2026-06-03.slurm`; replacement eval is `3705795_[0-29%3]` after `afterany:3695238:3705793`. Batch row `10` timed out with `checkpoint-5000`, bsz16 rows `3/7/11` were canceled near walltime after saving frequent checkpoints, recovery `3705794_[3,7,10,11%3]` is running/pending, and replacement eval is `3705796_[0-15%4]` after `afterany:3695197:3698877:3702079:3705794`.
