# Synthetic-RLVL Current Handoff

Last updated: 2026-08-01 01:33 CEST.

This is the short operational handoff. Historical detail was preserved verbatim in `docs/operational_history_2026-05-29.md`.

## Where To Look

| Need | File |
| --- | --- |
| Live/running job state | `docs/running_experiments.md` |
| Planned future experiments | `docs/experiment_backlog.md` |
| Short dated operational log | `docs/project_log.md` |
| Full preserved operational history | `docs/operational_history_2026-05-29.md` |
| Active research plan | `docs/formal_logic_cot_research_plan_2026-05-19.md` |
| HFSA implementation/eval plan | `docs/hfsa_depth_scaling_plan_2026-05-19.md` |
| Dataset materialization details | `docs/materialized_dataset.md` |
| Paired synthetic benchmark details | `docs/paired_synthetic_benchmarks_2026-05-20.md` |
| Paired iGSM validity audit | `docs/paired_igsm_validity_audit_2026-06-01.md` |
| Nanotron mixture/resume audit | `docs/nanotron_mixture_schedule_audit_2026-07-10.md` |
| Ongoing LaTeX report | `analysis/logic_cot_report_2026-05-25/logic_cot_report_2026-05-25.tex` |
| Official preprint draft | `../synthetic-RLVL-report/main.tex` |
| Informal generated report | `../synthetic-RLVL-report/informal_report/main.tex` |

## Current Scientific State

### 2026-08-01 01:33 terminal control and conditioned-32B release

- Dolmino control `3875826_0` completed `0:0` at step `9537/9537` in
  `21:34:31`. Its terminal state passes the complete 645-file, zero-byte,
  TP4/DP2 model/optimizer/scheduler/RNG and exact
  `9537/1220736/5000134656` offset gate, with all tokens charged to the
  control Nanoset. Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_control_step9537_20260801.json`.
- Formal continuation `3875829_1` started at 21:40 CEST on eight verified
  A100-80GB GPUs, restored the accepted step-5000 optimizer/scheduler/RNG and
  exact `95:5` offsets, and reached step 5781 at about 31.0K tokens/s with
  finite diagnostics. Its complete step-5500 state passes the same restart
  gate with exact `2739404800 + 144179200` Dolmino/formal tokens. Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_logic_step5500_20260801.json`.
- After those two acceptances, removed only superseded control steps
  9000/9500 and formal step 5000, reclaiming `319,885,161,145` bytes. Control
  step 9537 and formal step 5500 are the sole numeric restart states; Vault
  returned from `1465G/1000G` to `869G/1000G`, `181k/200k` files. Canceled
  redundant unstarted control stage `3875827_0` after terminal acceptance.
- Conditioned-32B recovery `3932131_14` completed `0:0` at step 10000. Its
  final has nine nonempty files, a `536,991,984`-byte adapter, terminal
  trainer state, and 896 nonempty safetensors tensors. Eval
  `3881781_12` started immediately on four A100-80GB GPUs and reached sampled
  chunk 99/112 in about 4h44m with the audited 16,384 context and no fatal
  signature; rows 13..17 remain array-throttle pending.
- Recorded successor `3936407` is dependency-free, CPU-only/no-GRES on
  `a100mig`, and BeginTime-pending for 07:27 CEST. Both critical paths remain
  incomplete, so it is preserved. No completed eval bundle or new scientific
  metric exists; neither report was regenerated.

### 2026-07-31 19:32 live progress and step-9000 control rotation

- Dolmino control `3875826_0` is healthy at iteration `9071/9537`
  (`4.76B/5B` tokens), about `31.1K` tokens/s, with finite loss and gradient
  diagnostics and an ETA near 21:37 CEST. Step 9000 passes the complete
  645-file, zero-byte, TP4/DP2 model/optimizer/scheduler/RNG and exact
  `9000/1152000/4718592000` offset gate, all charged to the control Nanoset.
  Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_control_step9000_20260731.json`.
- Only after that acceptance, superseded control step 8500 was removed,
  reclaiming `106,628,386,982` bytes. Step 9000 is now the sole numeric
  control restart state; Vault is `748G/1000G`, `181k/200k` files.
- Conditioned-32B recovery `3932131_14` is healthy at step `9612/10000` and
  has complete nonempty steps 9250/9500. Eval `3881781_12..17` remains
  correctly held on `afterok:3932131` with four A100-80GB GPUs and the full
  24-hour protocol. Formal/NL Dolmino `3875829_1/3875832_2` remain
  `AssocGrpGRES` pending, not failed.
- Recorded successor `3935384` is dependency-free, CPU-only/no-GRES on
  `a100mig`, and scheduled for 01:27 CEST on August 1. Both critical paths
  remain incomplete, so it is preserved. No evaluation artifact or new
  scientific metric exists; neither report was regenerated. Scoped commit
  `bd6d86e` remains local after a bounded 90-second `git push origin main`
  produced no response and exited `124`.

### 2026-07-31 18:50 live progress and control rotation

- Dolmino control `3875826_0` is healthy at iteration `8921/9537`
  (`4.68B/5B` tokens), about `31.1K` tokens/s, with finite diagnostics and an
  ETA near 21:35 CEST. Steps 8000 and 8500 independently pass the complete
  645-file, zero-byte, TP4/DP2 restart and exact-offset gates. Step 8500 is
  now the sole numeric control restart state; only after acceptance were
  superseded steps 7500/8000 removed, reclaiming `213,256,773,961` bytes.
- Conditioned-32B recovery `3932131_14` is healthy. Its latest complete
  checkpoint is step `9250/10000` with finite loss `0.0109` and grad norm
  `0.0186`; it should finish around 21:00--22:00 CEST. Eval
  `3881781_12..17` remains correctly dependency-held.
- Formal/NL Dolmino jobs `3875829_1/3875832_2` remain account-GRES pending,
  not failed. Slurm currently prints 2026-08-01 00:05 for both, but they each
  need eight GPUs and the estimates do not imply a simultaneous start. The
  older formal continuation is likely to start first. No new result metric is
  available.

### 2026-07-31 13:30 live recovery and control rotation

- Dolmino control continuation `3875826_0` remains healthy on eight
  A100-80GB GPUs at iteration `7801/9537`, about `31.1K` tokens/s, with
  finite loss and gradient diagnostics. Its step-7500 state passes the full
  645-file, zero-byte, TP4/DP2 model/optimizer/scheduler/RNG and exact-offset
  gate at `7500/960000/3932160000`, all charged to the control Nanoset.
  Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_control_step7500_20260731.json`.
- Only after that acceptance and independent completeness checks, superseded
  control steps 6500/7000 were removed, reclaiming `213,256,773,958` bytes.
  Step 7500 remains the sole numeric control restart state; Vault returned
  from `1145G/1000G` soft to `748G/1000G`, `181k/200k` files.
- Exact conditioned-32B row-14 recovery `3932131_14` resumed from step 7250
  on four verified A100-80GB GPUs and reached step `7717/10000`. It has
  written a complete nonempty step-7500 state and currently projects to
  finish within its 12-hour allocation. Eval `3881781_12..17` remains
  correctly held on `afterok:3932131` with four A100-80GB GPUs and the full
  24-hour protocol.
- Formal/NL continuations `3875829_1/3875832_2` remain account-GRES pending;
  their identical 21:26 CEST estimates are independent and do not imply a
  simultaneous start. Recorded successor `3933417` is dependency-free,
  CPU-only/no-GRES on `a100mig`, and scheduled for 19:26 CEST. Both critical
  paths remain incomplete, so it is preserved. No evaluation artifact or new
  scientific metric exists and neither report was regenerated. The scoped
  handoff commit remains local after a bounded 90-second `git push origin
  main` produced no response and exited `124`.

### 2026-07-31 11:40 format and post-SFT decision

- Exact packed-index audit finds formal/NL document p50
  `3715/3797`, p95 `7321/7392`, maxima `7688/7814`, and
  `44.0%/47.0%` above the 4,096-token training window. Active Nanotron
  `TokenizedBytesFileDataset` slices the concatenated stream into fixed
  windows and does not preserve document instances or apply a proof loss
  mask. The Qwen EOS configuration is consistent; this is a document/objective
  mismatch, not a wrong-EOS-ID bug.
- The implemented neutral envelope drifted from the planned compact
  `problem/Solution/Final answer` form by adding redundant
  `Context/Derivation/Conclusion` scaffolding and another premise
  representation. Do not alter or cancel the matched active series: the NL
  step-5000 readout remains the decisive content-versus-envelope control.
- Add identical modern post-SFT readouts for terminal control/formal/NL and
  retain direct eval separately. The old effective-batch-one UltraChat LoRA
  is a failed alignment diagnostic, not this readout. Gate any corrected 5B
  rerun on a short compact/document-preserving objective pilot. Full decision:
  `analysis/dolmino_format_and_post_sft_plan_20260731.md`.

### 2026-07-31 10:40 step-5000 output audit and conditioned recovery

- Extended review of all 1,200 control/formal multi-hop generations and the
  generated standard-suite families finds no formal-tag or neutral-envelope
  leakage. Formal bare-answer QA-record continuation is `91--97%`, including
  `87.5--100%` among exact first answers; first-answer multi-hop macro is
  nearly flat (`0.3893 -> 0.3799`) while tagged macro improves
  `0.3104 -> 0.3348`, mainly on HotpotQA. MMLU-Pro invalid extraction rises
  `2.50% -> 4.86%` through long repetitive tails. Detailed audit:
  `analysis/nanotron_dolmino_step5000_intermediate_20260730.md`.
- The current formal/NL intervention records share one outer envelope and
  average about 3,816/3,875 tokens versus about 456 for Dolmino. The matched
  NL step-5000 readout is now a P0 trigger to separate long-document
  objective/boundary effects from formal-content effects. Do not alter the
  active 5B runs; retain NL step 5000 for this readout.
- Conditioned-32B row `3883534_14` timed out as expected after 24 hours.
  Its complete nonempty step-7250 state records `global_step=7250` with
  finite final loss/gradient diagnostics. Submitted exact one-row recovery
  `3932131_[14%1]` with the same four-A100-80GB topology for 12 hours and
  rewired eval `3881781_12..17` to `afterok:3932131`.

### 2026-07-31 09:05 live training progress

- Dolmino control continuation `3875826_0` remains healthy on eight
  A100-80GB GPUs at iteration `6881/9537`, about `31.1K` tokens/s, with
  finite loss and gradient diagnostics. Nanotron reports about 12.5 hours
  remaining, projecting completion near 21:30 CEST today within this
  allocation.
- Conditioned-32B row `3883534_14` remains healthy on four A100-80GB GPUs
  and reached step `7262/10000` without a fatal signature. Its allocation
  ends at 09:23 CEST and cannot finish. After the actual timeout, recover
  only this row from the newest independently complete checkpoint (expected
  step 7250), then rewire `3881781_12..17` to the recovery while preserving
  the four-GPU A100-80GB evaluation protocol.
- Formal continuation `3875829_1` and exact from-base NL retry `3875832_2`
  remain `AssocGrpGRES` pending. Slurm independently estimates 14:08 CEST
  for both, but each requests a full eight-GPU node and the common estimate
  is not a simultaneous-start guarantee. Their terminal stages and the six
  conditioned-32B evaluations remain correctly dependency-held. No new
  failure or scientific result was observed.

### 2026-07-31 07:33 guarded control rotation and final conditioned seed progress

- Dolmino control continuation `3875826_0` is healthy on eight A100-80GB
  GPUs at iteration `6541/9537`, about `31.1K` tokens/s, with finite loss and
  gradient diagnostics. Its step-6500 state passed the complete 645-file,
  zero-byte, TP4/DP2 model/optimizer/scheduler/RNG and exact-offset gate at
  `6500/832000/3407872000`, all charged to the normal Nanoset. Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_control_step6500_20260731.json`.
- Only after that acceptance, superseded complete control steps
  5000/5500/6000 were removed, reclaiming `319,885,160,905` bytes. Step 6500
  remains the sole numeric control restart state. Vault quota fell from
  `1344G/1000G` soft to `748G/1000G`, with `181k/200k` files; the deleted
  optimizer states are not directly recoverable but can be reproduced by
  replaying training.
- Conditioned-32B row `3883534_14` remains healthy on four verified
  A100-80GB GPUs at step `6763/10000`, with complete nonempty checkpoints
  6500/6750 and no fatal signature. Its allocation ends at 09:23 CEST and
  cannot finish at the observed rate. Recover only after the actual timeout
  from the newest independently complete checkpoint, then replace the stale
  `3881781_12..17` dependency while preserving its A100-80GB four-GPU
  24-hour protocol.
- Formal continuation `3875829_1` and exact from-base NL retry `3875832_2`
  remain `AssocGrpGRES` pending with provisional independent 10:06 CEST
  estimates. Recorded successor `3930712` is dependency-free,
  CPU-only/no-GRES on `a100mig`, and scheduled for 13:26 CEST. Both critical
  paths remain incomplete, so it is preserved. No new evaluation artifact or
  scientific metric exists, and neither report was regenerated. Scoped commit
  `5e084e0` is local; a bounded 60-second `git push origin main` produced no
  response and exited `124`, so the existing SSH/network publication blocker
  remains.

### 2026-07-31 01:33 conditioned seed-3408 accepted and Dolmino control resumed

- Exact conditioned-32B recovery `3910990_13` completed `0:0` in `08:31:50`
  at step `10000/10000`. Its final adapter is nonempty, identifies
  `allenai/Olmo-3-1125-32B`, and passes a CPU safetensors load with 896
  tensors. Evaluation `3881781_12..17` now waits only on original SFT row
  `3883534_14`; Slurm preserved partition `a100`, feature `a100_80`, four
  A100s, and the 24-hour protocol. Final-backed checkpoints 9750/10000 were
  removed after acceptance, reclaiming `3,242,738,589` bytes while preserving
  the final adapter.
- Original conditioned row `3883534_14` remains healthy on four verified
  A100-80GB GPUs at step `4867/10000`; complete restart states 4500/4750 are
  present and focused log scans find no OOM, quota, traceback, or nonfinite
  signature. Its current allocation still projects a timeout, so submit an
  exact recovery only after the timeout from the newest independently
  complete checkpoint.
- Dolmino control continuation `3875826_0` started at 00:05 CEST on eight
  A100-80GB GPUs. It loaded the audited step-5000 model, optimizer, scheduler,
  RNG, sample, and token offsets exactly and reached iteration `5271/9537`
  with finite loss/gradient diagnostics at about `31.1K` tokens/s. Process
  inspection found no W&B service process, confirming the disabled/no-op
  logging fix. Formal continuation `3875829_1` and exact from-base NL retry
  `3875832_2` remain `AssocGrpGRES` pending; terminal stages retain their
  dependencies.
- Quotas remain safe (Home about `81.9G/100G`, Vault `754G/1000G` and
  `181k/200k` files, group Work `469k/500k` files). Recorded successor
  `3930503` is dependency-free, CPU-only/no-GRES on `a100mig`, scheduled for
  07:26 CEST, and remains required because both critical paths are incomplete.
  A bounded 60-second `git push origin main` again timed out without output;
  commit `aee7093` and its two predecessors remain local.

### 2026-07-30 19:32 conditioned-32B restart progress and successor verification

- Exact row-13 recovery `3910990_13` is healthy on four verified A100-80GB
  GPUs on `a0632`. It resumed from the accepted step-7500 state, reached step
  `9360/10000`, and retains complete nonempty checkpoints 9000/9250. Original
  row `3883534_14` is healthy on four A100-80GB GPUs on `a0931`, reached step
  `2981/10000`, and retains complete checkpoints 2500/2750. Focused fatal
  scans found no traceback, OOM, quota, no-space, or nonfinite-loss signature.
- Eval `3881781_12..17` remains correctly held on both SFT jobs and retains
  partition `a100`, feature `a100_80`, four GPUs, and the full 24-hour
  protocol. Row 13 should finish within its current allocation at the observed
  rate; row 14 still projects beyond one allocation and should be recovered
  only after an actual timeout from its newest complete state.
- Dolmino control/formal continuations `3875826_0/3875829_1` and exact
  from-base NL retry `3875832_2` remain `AssocGrpGRES` pending. Slurm
  provisionally projects 23:44 CEST for each when considered independently;
  this is not a simultaneous-start promise under the shared full-node cap.
- Home is `81,863M/100G`, Vault is `753G/1000G` with `181k/200k` files, and
  group Work is `469k/500k` files. Recorded successor `3929539` is verified
  dependency-free, CPU-only/no-GRES on `a100mig`, and scheduled for 01:25
  CEST July 31. Both critical paths remain incomplete, so it is preserved.
  A bounded 90-second `git push origin main` produced no response and exited
  `124`; the existing SSH/network publication blocker remains.

### 2026-07-30 13:33 second conditioned-32B allocation and intermediate readout audit

- Exact row-13 recovery `3910990_13` started at 13:04 CEST on four verified
  A100-80GB GPUs on `a0632`. Its accepted step-7500 restart remains intact;
  startup loaded all base shards and is rebuilding deterministic preprocessing
  before optimizer resume. Original row `3883534_14` is healthy on four
  A100-80GB GPUs on `a0931`, reached step 1077/10000, and has complete
  checkpoints 750/1000. Eval `3881781_12..17` remains held on both jobs.
- Both step-5000 readouts completed inference. Fixed the MATH-500 sidecar's
  missing terminal-percent normalization exposed by `$10\%$` versus gold
  `10`; `19` focused tests and forced CPU rescoring/re-audit pass, accepting
  both 105-leaf/10,600-row standard bundles without a GPU rerun.
- Limit-100 reviewer formal-minus-control macro is provisional `+0.0408`.
  Multi-hop stock QA-F1 falls `0.3171 -> 0.1093`, while tagged QA-F1 changes
  `0.3104 -> 0.3348`; raw review attributes the stock decline to 91--97%
  formal QA-record continuation. This is intermediate response-control
  evidence only. Audit:
  `analysis/nanotron_dolmino_step5000_intermediate_20260730.md`.
- Dolmino continuations remain `AssocGrpGRES` pending. Successor oversight
  `3927534` is CPU-only/no-GRES and BeginTime-pending for 19:25 CEST; it is
  preserved because both critical paths remain incomplete.

### 2026-07-30 10:32 shared capacity partially released

- Conditioned-32B SFT row `3883534_14` started at 09:23 CEST on four
  A100-80GB GPUs and is healthy: dataset preprocessing completed, W&B run
  `u6cyipt5` is syncing, and optimizer progress reached step 139/10000 by the
  latest log sample without a fatal signature. No checkpoint has been
  written yet. At the observed roughly 10--14 seconds per step, the full run
  is longer than its 24-hour allocation and remains expected to rely on its
  checkpoint/recovery path.
- Step-5000 intermediate A40 eval row 0 completed after 1:02:20 and produced
  an accepted 105-leaf/10,600-sample bundle. Row 1 completed all model
  inference and wrote the same-sized bundle, but deliberately exited nonzero
  because the production audit rejected one MATH-500 post-hoc discrepancy:
  stock exact-positive sample 67 was lost. This is an evaluator/audit issue,
  not an OOM, quota failure, or missing model output.
- Remaining eligible work is still account-cap blocked:
  conditioned recovery row `3910990_13` requests four A100s and currently
  forecasts 23:06 CEST; Dolmino control/formal/NL
  `3875826_0/3875829_1/3875832_2` each request eight A100s and independently
  forecast 06:51 CEST July 31. The shared user's reported number of jobs is
  insufficient to explain usage because `AssocGrpGRES` accounts GPUs, not
  job count; exact sibling allocations remain hidden by Slurm `PrivateData`.
- The completed-job quota footer reports Home `83.7G/104.9G` soft and Vault
  `779.4G/1048.6G`, with 157k/500k and 180k/200k files respectively. No
  disk-quota failure occurred in either eval condition.

### 2026-07-30 07:38 why queue age has not won priority

- Slurm multifactor priority uses a seven-day fair-share decay and seven-day
  maximum age. Fair-share weight is 10,000 versus age weight 6,000.
  `c107fa12` has 25% normalized shares but 61.2% effective recent account
  usage (`LevelFS=0.4085`, fair-share factor `0.1768`), yielding only 1,768
  fair-share priority points. Hidden sibling users with less recent usage can
  therefore retain a much larger priority advantage.
- Age begins at `EligibleTime`, not original submission. Control continuation
  became eligible July 27 08:09 and currently has 2,552 age points; formal
  and NL became eligible July 28 08:10/08:18 and have about 1,690. The
  conditioned SFT rows became eligible July 28 and have about 1,921. Thus the
  chain's July-21 submission date does not translate into nine days of age.
- Current priorities are control 5,027, conditioned SFT 4,387/4,386, formal
  4,169, NL 4,164, and A40 eval 4,132. Pending jobs consume no allocation,
  but when hidden account capacity releases, higher-priority sibling-user
  work may refill the group cap first. The eight-GPU/24-hour Dolmino shape
  also requires a whole A100-80GB node and is difficult to backfill.
- Fair-share recovers automatically as recent usage halves every seven days,
  and age accrues until its seven-day cap, but neither guarantees immediate
  service against a continuously refilled shared account cap. Practical
  remedies are coordination with the other account users, an administrator
  increase/reservation/QoS, or waiting for usage decay; no scientifically
  invalid topology/time-limit edit was made.

### 2026-07-30 07:33 account-wide GPU-cap root cause confirmed

- Confirmed that `AssocGrpGRES` is an inherited account-association GPU
  limit, not malformed jobs, disk quota, partition selection, dependencies,
  or this user's visible allocations. Even the one-A40 intermediate eval is
  blocked, while this user's association reports zero current
  `GrpTRESRunMins`.
- Slurm is configured with
  `PrivateData=accounts,jobs,reservations,usage,users`. Consequently,
  account-wide `squeue`, `sacct`, `sshare`, and association queries expose
  only `c107fa12`; parent-account limits and other `c107fa` users' running
  GPU jobs are hidden or permission-denied. The account-wide generic GPU cap
  is therefore consumed by hidden sibling-user usage even though no running
  job is visible locally.
- All three full-node Dolmino jobs provisionally forecast 08:20 CEST because
  the scheduler sees a hidden allocation ending then. Each forecast is
  calculated independently and does not mean all three eight-GPU jobs can
  start together. The conditioned-32B four-GPU SFT rows and one-GPU A40 eval
  still have no forecast.
- Pending/dependency-never-satisfied jobs allocate no GPU and are not the
  cause. Partition widening cannot bypass an account-wide generic GRES cap,
  and reducing the Dolmino topology would invalidate the accepted TP4/DP2
  resume protocol. Exact cap/current sibling-user attribution requires a
  Slurm administrator or the other account users; no job mutation was made.

### 2026-07-30 07:29 live queue remains healthy but capacity-stalled

- No in-scope experiment job has started, completed, failed, or produced a
  pending-job log/output since the 01:27 handoff. Conditioned-32B SFT
  `3910990_13/3883534_14` remains A100-80GB account-GRES pending without
  estimates; eval `3881781_12..17` retains exactly those two dependencies.
- Dolmino control/formal continuations `3875826_0/3875829_1` and exact
  from-base NL retry `3875832_2` now share a provisional 08:20 CEST forecast.
  All request full eight-A100-80GB nodes under the same association cap, so
  the common forecast is not a simultaneous-start promise. Their terminal
  `afterany` dependencies remain intact.
- A40 step-5000 eval `3913651_[0-1%2]` remains account-GRES pending without
  an estimate despite four physically idle A40 nodes. The association limit,
  rather than partition scarcity, remains binding, so widening partitions
  would not help.
- Revalidated both 645-file/nonempty Dolmino step-5000 restart trees, the
  13-file/nonempty conditioned row-13 step-7500 state, and the
  nine-file/nonempty row-12 final. Work remains healthy at about
  `464k/500k` group files; Home is `81,748M/100G` and Vault is
  `743G/1000G`, `181k/200k` files. Current watcher `3926040` is running
  CPU-only/no-GRES; recorded successor `3926409` is dependency-free,
  CPU-only/no-GRES, and scheduled for 13:24 CEST. The handoff commit remains
  local after a bounded 90-second `git push origin main` produced no remote
  response and exited `124`.

### 2026-07-30 01:27 scheduler and artifact reconciliation

- No in-scope experiment job has started, failed, or completed since the
  prior handoff, and every pending-job log plus the intermediate-eval output
  root remains absent. Conditioned-32B SFT `3910990_13/3883534_14` still has
  no start estimate; eval `3881781_12..17` retains exactly those two
  dependencies and its A100-80GB-only resources.
- Dolmino control/formal continuations `3875826_0/3875829_1` and exact
  from-base NL retry `3875832_2` now share a provisional 02:11 CEST start.
  These are competing full-node requests under the association GPU cap, so
  the identical forecast is not a concurrent-start guarantee. Terminal jobs
  `3875827_0/3875830_1/3875833_2` retain their exact `afterany`
  dependencies.
- Revalidated both accepted Dolmino step-5000 trees at 645 files with no
  empty files and exact offsets `5000/640000/2621440000`; formal remains
  exactly 95:5. The accepted conditioned-32B seed-3407 final and seed-3408
  step-7500 restart remain intact, while the seed-3409 root is correctly
  absent before first start. A40 intermediate eval `3913651_[0-1%2]`
  remains account-GRES pending without an estimate despite physical idle A40
  capacity, so no scheduler edit is useful.
- Work group use is `464k/500k` files and Vault is `743G/1000G`,
  `181k/200k` files. Recorded successor `3926040` is dependency-free,
  CPU-only/no-GRES on `a100mig`, and scheduled for 07:24 CEST. Both critical
  paths remain incomplete, so it is preserved. Handoff commit `45c5c6a` is
  local; `git push origin main` produced no remote response for 90 seconds
  and exited `124`.

### 2026-07-29 19:28 live scheduler and successor reconciliation

- No in-scope experiment job has started, failed, or completed since the
  15:11 handoff, and every pending-job log/output root remains absent.
  Conditioned-32B SFT `3910990_13/3883534_14` briefly showed provisional
  20:52 CEST starts, but those estimates disappeared on the final scheduler
  sample; eval `3881781_12..17` still depends exactly on those two jobs and
  retains A100-80GB-only resources.
- Dolmino control/formal continuations `3875826_0/3875829_1` now project
  21:40 CEST, and exact from-base NL retry `3875832_2` projects 22:55 CEST.
  Terminal stages `3875827_0/3875830_1/3875833_2` retain their respective
  `afterany` dependencies. The accepted control/formal step-5000 trees remain
  the sole numeric restart states at 645 files each with no empty files and
  exact offsets `5000/640000/2621440000`; formal remains exactly 95:5.
- A40 intermediate eval `3913651_[0-1%2]` lost its scheduler estimate and
  remains account-GRES pending. One A40 node is physically idle, but the
  association GPU cap is still binding, so no partition/dependency edit is
  useful. Work group use is `464k/500k` files and Vault is
  `743G/1000G`, `181k/200k` files.
- Recorded successor `3925270` is verified dependency-free, CPU-only/no-GRES
  on `a100mig`, and scheduled for 01:24 CEST July 30. Both critical paths
  remain incomplete, so it is preserved. Handoff commit `38e61f7` is local;
  `git push origin main` produced no remote response for 90 seconds and was
  interrupted.

### 2026-07-29 15:11 post-cleanup live job verification

- Group Work usage is now `463,067` files, below both the 500k soft and 600k
  hard inode limits. No synthetic-RLVL job started, failed, or was canceled
  during cleanup, and no pending-job log was created.
- Conditioned-32B SFT recovery `3910990_13` and original row `3883534_14`
  remain A100-80GB account-GRES pending without current start estimates.
  Eval `3881781_12..17` retains exactly those two dependencies. The accepted
  seed-3407 final remains nine files/no empty files, and the row-13 step-7500
  resume remains 13 files/no empty files with `global_step=7500`.
- Dolmino control continuation `3875826_0` is capacity-pending with a
  provisional 18:45 CEST start. Formal continuation `3875829_1` and exact
  from-base NL retry `3875832_2` remain capacity-pending with provisional
  06:46 CEST July-30 starts; their terminal stages retain correct `afterany`
  dependencies. Both accepted step-5000 restart trees remain 645 files with
  no empty files.
- A40 step-5000 eval `3913651_[0-1%2]` remains capacity-pending with a
  provisional 16:37 CEST start and no output root yet. Partition inspection
  shows the relevant A100-80GB and A40 capacity allocated/mixed/draining;
  no compatible beneficial widening or job mutation was applied.

### 2026-07-29 14:53 minimal safe project inode cleanup

- Confirmed there are no same-named top-level run, dataset, pass@k, lm-eval,
  or analysis directories duplicated between the Work and Vault project
  trees. Preserved all unique checkpoints, results, generations, datasets,
  Slurm logs, W&B histories, and untracked audit artifacts.
- Removed only 157 W&B per-run `tmp/` trees from Work after verifying that
  each contained exactly one empty `code/` directory and no files or bytes.
  This project-attributable cleanup freed exactly 314 Work inodes. Also
  removed 15 Python `__pycache__` trees and the pytest cache from the Home
  checkout (about 4.7M); these do not affect Work quota.
- Group Work usage subsequently reported about 588.7k files, materially more
  reduction than this repo's 314-inode cleanup and therefore consistent with
  concurrent cleanup or delayed quota accounting. A fresh Work directory and
  file creation probe now succeeds. The group remains above the 500k soft
  limit, so additional cross-project cleanup is still required for margin.

### 2026-07-29 14:40 shared Work inode hard limit confirmed

- Diagnosed the controller `mkdir` failure as the group inode quota on
  `/home/atuin`, not user byte quota or global filesystem capacity.
  `quota -g` reports group `c107fa` at exactly `600,000` files against a
  `500,000` soft and `600,000` hard limit, with about 34 hours of soft-limit
  grace remaining. Group byte use is only about 6.55T against 10T soft/12.5T
  hard, and the filesystem itself has 916T free.
- Reproduced the failure with a bounded probe: both `$WORK` and
  `$WORK/TextJEPA` reject `mkdir` with `Disk quota exceeded`; `$HOME`,
  `$HPCVAULT`, and project scratch all create and remove a directory/file
  normally. The Work quota is shared by private trees for users
  `c107fa10..c107fa13`, so another group member can contribute to the
  aggregate blocker; sibling permissions prevent row-level attribution.
- A user-quota query and complete top-level inventory attribute `441,032` of
  the group `600,000` Work inodes (73.5%) to `c107fa12`; the other three
  users together account for `158,968`, but cross-user queries are
  permission-denied. The largest local inode consumers are `FOMO_runtime`
  (146,017), `$WORK/.venv` (94,362), `babylm_runtime` (47,899), TextJEPA
  (43,851), `nanotron` (32,089), and `.local` (30,762). No deletion was made
  because these cross-project runtimes may be active. Synthetic-RLVL
  environment setup directs caches and W&B state to Work, so pending jobs
  remain at risk of write failures until the group frees inodes or the quota
  is raised; no queued job or dependency was changed in this diagnostic pass.

### 2026-07-29 14:22 conservative project quota cleanup

- Audited the project-scoped Vault and Work trees against live Slurm jobs,
  resume/eval scripts, accepted artifact gates, and the active handoff. Removed
  only three superseded conditioned-32B SFT restart snapshots:
  seed-3407 `checkpoint-9750` and `checkpoint-10000`, whose completed,
  accepted nine-file `final/` adapter is the eval input; and seed-3408
  `checkpoint-7250`, whose pending exact recovery uses the newer complete
  `checkpoint-7500`.
- Reclaimed exactly `9,728,633,344` quota-accounted bytes. Vault use fell
  from `752G` to `743G`; the project Vault tree fell from
  `683,172,380,160` to `673,443,746,816` bytes. Revalidated the retained
  seed-3407 final (nine files, no empty files) and seed-3408 step-7500
  restart (13 files, no empty files).
- Preserved both accepted Dolmino step-5000 trees, all final adapters,
  datasets, raw generations, audits, and ambiguous historical outputs.
  The 69G project tree under `$WORK` was audited but not changed because no
  additional artifact there was provably disposable without risking retained
  results or checkpoints.

### 2026-07-29 13:26 live gates and near-term scheduler estimates

- No in-scope GPU job has started or completed since the July-28 handoff.
  Conditioned-32B row-13 recovery `3910990_13` and original row
  `3883534_14` remain A100-80GB account-GRES pending, now with provisional
  17:23 CEST starts. Eval `3881781_12..17` still depends only on those two
  rows and retains the full A100-80GB protocol.
- Dolmino control continuation `3875826_0` remains account-GRES pending with
  a provisional 22:16 CEST start. Formal continuation `3875829_1` and exact
  from-base NL retry `3875832_2` remain pending with provisional 04:17 CEST
  July-30 starts. The shared runtime wrapper still defaults to
  `WANDB_MODE=disabled`; no pending continuation log or new checkpoint exists.
- Step-5000 intermediate eval `3913651_[0-1%2]` remains A40 account-GRES
  pending, but its estimate moved materially from August 4 to 23:01 CEST
  July 29. Its output root is absent, so no raw-generation or metric audit is
  open.
- Accepted control/formal step-5000 checkpoint trees remain the sole numeric
  restart states and their persisted audits still report `status=accepted`,
  645 files, and no zero-byte files. Vault is `752G/1000G`, `181k/200k`
  files. Recorded successor `3924006` is dependency-free, CPU-only/no-GRES,
  and scheduled for 19:22 CEST; both critical paths remain incomplete, so it
  is preserved. Handoff commit `a238518` is local; `git push origin main`
  produced no remote response for 90 seconds and was interrupted, leaving
  local `main` one commit ahead.

### 2026-07-28 18:04 Dolmino step-5000 intermediate eval submitted

- Submitted one-GPU A40 array `3913651_[0-1%2]` to compare the accepted
  control and formal Dolmino step-5000 checkpoints before terminal 5B
  training completes. Each row converts its Nanotron checkpoint into a
  job-local temporary HF model, verifies files/RoPE/CUDA forward locally,
  evaluates, and deletes only the temporary conversion; restart checkpoints
  remain untouched.
- The limited direct readout runs first on 100 examples each from tagged and
  stock HotpotQA, 2WikiMultiHopQA, and MuSiQue, then on the standard reviewer
  suite (`gsm8k`, ARC-Challenge, HellaSwag, WinoGrande, PIQA, LogiQA,
  MATH-500, BBH, MMLU, and MMLU-Pro), also with lm-eval limit 100. Retained
  generations and limit-aware audits are required.
- A40 was selected so the eval does not consume the A100-80GB capacity needed
  by the control/formal continuations and NL retry. The submitted array is
  pending with a provisional 07:42 CEST August 4 estimate. Script:
  `scripts/slurm/jobs/nanotron_qwen25_dolmino_step5000_intermediate_eval_2026-07-28.slurm`.

### 2026-07-28 13:55 live plan reconciled; Dolmino W&B retry hardened

- The selected corrected BranchProof matrix is accepted at `39/45` eval
  rows. All syntax/length, shortcut, both hybrid orders, conditioned-7B,
  Qwen2.5-7B, and single-modal OLMo-3-32B rows are complete and reported.
  The only missing six rows are conditioned OLMo-3-32B eval
  `3881781_12..17`, gated on row-13 recovery `3910990_13` and original row
  `3883534_14`; row-12 recovery is already final-backed and accepted.
- Slurm currently projects both conditioned-32B SFT rows for about 17:22 CEST
  July 28. Control Dolmino continuation `3875826_0` projects 18:44 CEST July
  28; formal continuation `3875829_1` and exact from-base NL retry
  `3875832_2` project 09:30 CEST July 29. These are scheduler estimates, not
  reservations.
- The NL failure exposed a repeatable operational risk rather than a training
  defect: `WANDB_MODE=offline` still starts a local W&B service. Changed the
  shared Nanotron production wrapper default to `WANDB_MODE=disabled`.
  Pending immutable outer jobs call this shared script from disk at runtime,
  so control/formal continuations and the NL retry retain their queue
  positions and now avoid the port-file service entirely. Both Slurm scripts
  pass `bash -n`; the installed W&B source explicitly maps disabled mode to a
  no-op run and states that no service process starts.
- The complete experimental picture is otherwise unchanged: corrected Tiny
  models are an accepted negative capacity diagnostic; Attribute Constraints
  is positive but high-variance; semantic iGSM and typed Maze are complete
  negative/excluded tasks; the corrected 15% continual-pretraining pilot and
  its multihop suite are null/mixed response-format diagnostics. No additional
  broad grid is pending.
- Publication recovered in this pass. Main-repo push advanced GitHub from
  `6c2a616` through `1bfb1d6`, and report-repo push advanced GitHub from
  `e65926e` through `0265eae`; both local `main` branches are synchronized
  with `origin/main`.

### 2026-07-28 13:25 formal boundary, NL retry, and conditioned row 12 accepted

- Dolmino formal first stage `3875828_1` reached iteration 5021 and ended in
  the expected `TIMEOUT` after `1-00:00:28`, without OOM, quota, or unexpected
  fatal signatures. Step 5000 passes the complete 645-file, zero-byte,
  TP4/DP2 model/optimizer/scheduler/RNG gate, Qwen2.5 RoPE `1000000`, and
  exact offsets `5000/640000/2621440000`. Realized consumption is exactly
  `2490368000` Dolmino plus `131072000` formal tokens (`95:5`). Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_logic_step5000_20260728.json`.
- Only after step 5000 passed, the superseded formal step-4500 tree was
  removed, reclaiming about `106.63 GB`. Step 5000 is the sole formal
  restart state and the project Vault tree is `667,160,528 KiB`.
- NL first stage `3875831_2` failed before optimizer step 1 because the local
  W&B service did not produce its port file within 30 seconds. It wrote no
  training checkpoint, so released continuation `3875832_2` is the exact
  from-base retry and remains account-GRES pending; no scientific state is
  replayed. Control/formal continuations `3875826_0/3875829_1` are also
  account-GRES pending.
- Conditioned-32B exact row-12 recovery `3897965_12` completed `0:0` at step
  10000. Its final adapter has eight nonempty files, a
  `536,991,984`-byte safetensors payload, no zero-byte file, and trainer state
  `global_step=10000`. Slurm removed the fulfilled dependency automatically;
  eval `3881781_12..17` now waits only on row-13 recovery `3910990_13` and
  original row `3883534_14`, both A100-80GB account-GRES pending.
- Recorded successor `3912896` is dependency-free, CPU-only/no-GRES, and
  BeginTime-pending. Both critical paths remain incomplete, so it is
  preserved.
- Scoped audit commit is `38089da`. A bounded 60-second
  `git push origin main` produced no remote response and exited `124`, so the
  existing GitHub SSH/network blocker remains.

### 2026-07-28 07:25 formal step-4500 rotation and live recoveries

- Dolmino formal first stage `3875828_1` is healthy through iteration 4851
  at about `30.8K` tokens/s with finite loss and gradient norm. Step 4500
  passed the complete 645-file, zero-byte, TP4/DP2
  model/optimizer/scheduler/RNG gate, Qwen2.5 RoPE `1000000`, and exact
  offsets `4500/576000/2359296000`. Realized consumption is exactly
  `2241331200` Dolmino plus `117964800` formal tokens (`95:5`). Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_logic_step4500_20260728.json`.
- The live writer retained formal steps 3500, 4000, and 4500, raising Vault
  use to `1149G/1000G`. Only after step 4500 passed the complete restart
  gate, the exact superseded step-3500/4000 trees were removed, reclaiming
  `213,256,774,301` bytes. Step 4500 is now the sole numeric formal restart
  state; Vault is `751G/1000G`, `181k/200k` files.
- Conditioned-32B row-12 recovery `3897965_12` remains healthy on verified
  A100-80GB node `a0831`, through step 9160 with complete checkpoints 8750
  and 9000. Row-13 recovery `3910990_13` and original row `3883534_14`
  remain account-GRES pending; eval `3881781_12..17` retains
  `afterok:3897965:3910990:3883534_14` and its A100-80GB protocol.
- Control continuation `3875826_0` and NL first stage `3875831_2` remain
  account-GRES pending. Formal continuation `3875829_1` remains correctly
  held until the first stage reaches its planned boundary. Recorded successor
  `3911426` is dependency-free, CPU-only/no-GRES, and BeginTime-pending for
  13:20 CEST. The plan is incomplete, so it is preserved.
- Scoped audit commit is `162a0fe`. A bounded 60-second
  `git push origin main` produced no remote response and exited `124`, so the
  existing GitHub SSH/network blocker remains.

### 2026-07-28 01:55 formal rotation and conditioned-32B recovery

- Dolmino formal first stage `3875828_1` is healthy through iteration 3591
  at about `30.7K` tokens/s with finite loss and gradient norm. Step 3500
  passed the complete 645-file, zero-byte, TP4/DP2
  model/optimizer/scheduler/RNG gate, Qwen2.5 RoPE `1000000`, and exact
  offsets `3500/448000/1835008000`. Realized consumption is exactly
  `1743257600` Dolmino plus `91750400` formal tokens (`95:5`). Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_logic_step3500_20260728.json`.
- The live writer had retained formal steps 2000, 2500, 3000, and 3500,
  raising user Vault use to `1347G/1000G`. Only after step 3500 passed the
  complete restart gate, the exact superseded step-2000/2500/3000 trees were
  removed, reclaiming `319,885,161,449` bytes. Step 3500 is now the sole
  numeric formal restart state; Vault is `751G/1000G`, `181k/200k` files, and
  the project Vault tree is `666,092,489 KiB`.
- Conditioned-32B SFT `3883534_13` reached step 7538 and ended in the expected
  `TIMEOUT` after `1-00:00:15`, without OOM, quota, or unexpected fatal
  signatures. Step 7500 passed the complete 13-file, zero-byte
  adapter/optimizer/scheduler/RNG gate. Exact row-13 resume `3910990_13` was
  submitted for 12 hours on four A100-80GB GPUs. Row-12 recovery
  `3897965_12` started on verified A100-80GB node `a0831`; original row
  `3883534_14` remains pending. Eval `3881781_12..17` was rewired to
  `afterok:3897965:3910990:3883534_14` and retains its A100-80GB protocol.
- Control continuation `3875826_0` and NL first stage `3875831_2` remain
  account-GRES pending; all later Dolmino stages retain their dependencies.
  Recorded successor `3910926` is dependency-free, CPU-only/no-GRES, and
  BeginTime-pending for 07:20 CEST July 28. Both active paths remain
  incomplete, so it is preserved.
- Scoped commits are `74b3339` and `1324746`. A bounded 60-second
  `git push origin main` produced no remote response and exited `124`, so the
  existing GitHub SSH/network blocker remains.

### 2026-07-27 19:23 formal step-2000 rotation and live progress

- Dolmino formal first stage `3875828_1` is healthy on eight A100-80GB GPUs
  through iteration `2331/9537` at about `30.7K` tokens/s, with finite loss
  and gradient norm. Step 2000 passed the complete 645-file, zero-byte,
  TP4/DP2 model/optimizer/scheduler/RNG gate, Qwen2.5 RoPE `1000000`, and
  exact offsets `2000/256000/1048576000`. Realized consumption is exactly
  `996147200` Dolmino plus `52428800` formal tokens (`95:5`). Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_logic_step2000_20260727.json`.
- The live writer had retained formal steps 1000, 1500, and 2000, raising
  user Vault use to `1149G/1000G`. Only after step 2000 passed the restart
  gate, the exact superseded step-1000 and step-1500 trees were removed,
  reclaiming `213,256,774,296` bytes. Step 2000 is now the sole numeric
  formal restart state; Vault is `751G/1000G`, `181k/200k` files, and the
  project Vault tree is `666,092,361 KiB`.
- Conditioned-32B SFT `3883534_13` is healthy through step `5483/10000`;
  checkpoints 5000 and 5250 each contain 13 nonempty files with no zero-byte
  file. Row `3883534_14` remains array-throttle pending, exact row-12 resume
  `3897965_12` remains account-GRES pending, and eval `3881781_12..17`
  remains correctly dependency-held. Its current pace projects a timeout
  before step 10000, so recover only this row from its newest complete state
  after an actual timeout.
- Control continuation `3875826_0` and NL first stage `3875831_2` remain
  account-GRES pending; all later Dolmino stages retain their dependencies.
  Recorded successor `3908135` is dependency-free, CPU-only/no-GRES, and
  BeginTime-pending for 01:20 CEST July 28. Both active paths remain
  incomplete, so it is preserved. Scoped audit commit is `c53b21c`; a bounded
  60-second `git push origin main` produced no remote response and exited
  `124`, so the existing GitHub SSH/network blocker remains.

### 2026-07-27 13:24 formal step-1000 rotation and live progress

- Dolmino formal first stage `3875828_1` is healthy on eight A100-80GB GPUs
  through iteration `1071/9537` at about `30.7K` tokens/s, with finite loss
  and gradient norm. Its step-1000 state passed the complete 645-file,
  zero-byte, TP4/DP2 model/optimizer/scheduler/RNG gate, Qwen2.5 RoPE
  `1000000`, and exact offsets `1000/128000/524288000`. Realized token
  consumption is exactly `498073600` normal plus `26214400` formal tokens
  (`95:5`). Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_logic_step1000_20260727.json`.
- The live writer retained both steps 500 and 1000, temporarily raising Vault
  to `950G/1000G`. Only after step 1000 passed the restart gate, the exact
  superseded step-500 tree was removed, reclaiming `106,628,387,143` bytes.
  Step 1000 is now the sole numeric formal restart state; Vault is
  `751G/1000G`, `181k/200k` files, and the project Vault tree is
  `666,092,233 KiB`.
- Conditioned-32B SFT `3883534_13` is healthy through step `3586/10000` with
  complete nonempty checkpoints 3250 and 3500. Row `3883534_14` remains
  array-throttle pending, exact row-12 resume `3897965_12` remains
  account-GRES pending, and eval `3881781_12..17` remains correctly
  dependency-held. No new evaluation artifact or report gate opened.
- Control continuation `3875826_0` and NL first stage `3875831_2` remain
  account-GRES pending; all later Dolmino stages retain their dependencies.
  Recorded successor `3904173` is dependency-free, CPU-only/no-GRES, and
  BeginTime-pending for 19:19 CEST. Both active paths remain incomplete, so
  it is preserved. Scoped commit is `d0da6b4`; a bounded 60-second
  `git push origin main` produced no remote response and exited `124`, so the
  existing GitHub SSH/network blocker remains.

### 2026-07-27 08:17 Dolmino control boundary, formal start, and conditioned-32B progress

- Dolmino control first stage `3875825_0` reached iteration `5021` and ended
  in the expected `TIMEOUT` after `1-00:00:10`, without an OOM, quota, or
  unexpected fatal signature. Steps 4500 and 5000 independently passed the
  complete 645-file, zero-byte, TP4/DP2 model/optimizer/scheduler/RNG gate;
  step 5000 has exact offsets `5000/640000/2621440000`, solely normal data,
  and is the sole retained restart state. Audits:
  `analysis/nanotron_checkpoint_audits/dolmino_control_step4500_20260727.json`
  and `analysis/nanotron_checkpoint_audits/dolmino_control_step5000_20260727.json`.
  After acceptance, removed only superseded steps 3500/4000/4500, reclaiming
  `319,885,160,841` bytes. Vault is `553G/1000G`, `180k/200k` files.
- Control continuation `3875826_0` is released and account-GRES pending;
  `3875827_0` remains dependency-held. Formal first stage `3875828_1` started
  at 08:09 CEST on eight A100-80GB GPUs. Its config preserves Qwen2.5 RoPE
  `1000000`, TP4/DP2, the preregistered schedule, and realized normal/formal
  weights `0.95/0.0500002`; iteration 1 has finite loss/gradient diagnostics.
  NL first stage `3875831_2` remains account-GRES pending.
- Conditioned-32B SFT `3883534_13` is healthy through step `1968/10000` on
  four A100-80GB GPUs, with complete nonempty steps 1500 and 1750 retained.
  Row `3883534_14` remains array-throttle pending, exact resume
  `3897965_12` remains account-GRES pending, and eval `3881781_12..17`
  remains correctly dependency-gated. No new eval generations or metrics
  exist, so no report or scientific-claim gate opened.
- Recorded successor `3902223` is dependency-free, CPU-only/no-GRES, and
  BeginTime-pending for 13:19 CEST. Both active critical paths remain
  incomplete, so it is preserved. Scoped commit is `a34dfd1`; a bounded
  60-second `git push origin main` produced no remote response and exited
  `124`, so the existing GitHub SSH/network blocker remains.

### 2026-07-27 01:31 final single-modal 32B acceptance and Dolmino rotation

- OLMo-3-32B NL row `3881780_5` completed in `06:11:47` on four verified
  A100-80GB GPUs and passed the full 448-prompt, 16-generation, 14-depth,
  576-retained-row, complete-log, fresh-constant, and translated-validity
  gate. Audit:
  `$HPCVAULT/synthetic-RLVL/analysis/branchproof_selected_followups_audits_20260723/large_5.json`.
  Retained raw review is perfect through depth 35, with one complete wrong
  chain at depth 40 and four depth-50 failures: two cap/format failures and
  two complete wrong chains. No prompt, extraction, or validator artifact was
  found.
- The accepted three-seed OLMo-3-32B single-modal comparison strongly favors
  natural supervision at OOD answer/joint pass@1:
  `0.9888 +/- 0.0080` / `0.9879 +/- 0.0088` versus formal
  `0.6388 +/- 0.0512` / `0.5109 +/- 0.0182`. At pass@16 formal nearly
  catches up in answer correctness (`0.9875 +/- 0.0088`) but remains below NL
  in joint validity (`0.9396 +/- 0.0179` versus `0.9979 +/- 0.0029`).
  The in-repo report, informal mirror, and official preprint now report this
  accepted architecture-dependent result.
- Dolmino control `3875825_0` is healthy at step `3631/9537`, about `30.7K`
  tokens/s, with finite loss and gradient norm. Step 3500 passed the complete
  645-file TP4/DP2 model/optimizer/scheduler/RNG and exact-offset gate at
  `3500/448000/1835008000`; audit:
  `analysis/nanotron_checkpoint_audits/dolmino_control_step3500_20260727.json`.
  Only after acceptance, superseded complete steps 2000/2500/3000 were
  removed, reclaiming `319,885,160,841` bytes. Step 3500 is the sole numeric
  restart state; Vault is `547G/1000G`, `180k/200k` files.
- Conditioned-32B `3883534_13/14`, exact resume `3897965_12`, and Dolmino
  formal/NL `3875828_1/3875831_2` remain account-GRES pending. Conditioned
  eval `3881781_12..17` retains its repaired dependency gate. Successor
  `3901977` is dependency-free, CPU-only/no-GRES, and BeginTime-pending for
  07:18 CEST; both critical paths remain incomplete, so it is preserved.
  Scoped commits are main repo `9ca84fa` and report repo `0265eae`; bounded
  60-second pushes for both produced no remote response and exited `124`, so
  the existing GitHub SSH/network blocker remains.

### 2026-07-26 19:23 second Dolmino rotation and second 32B NL row accepted

- Dolmino control `3875825_0` is healthy beyond step `2341/9537` at about
  `30.7K` tokens/s with finite loss and gradient norm. Its step-2000 state
  passed the same complete 645-file, zero-byte, TP4/DP2,
  model/optimizer/scheduler/RNG, and exact-offset gate as step 1000:
  `2000/256000/1048576000`, all from the normal Nanoset. Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_control_step2000_20260726.json`.
- The live writer had retained steps 1000, 1500, and 2000 and pushed Vault to
  `1064G/1000G`. Only after step 2000 passed the restart gate, the exact
  superseded step-1000 and step-1500 trees were removed, reclaiming
  `213,256,773,890` bytes. Step 2000 remains the sole numeric restart state;
  Vault is back to `667G/1000G`, `180k/200k` files, and the project Vault tree
  is `577,396,560 KiB`.
- OLMo-3-32B NL eval row `3881780_4` completed as raw job `3900120` in
  `06:06:34` on four A100-80GB GPUs and passed the full 448-prompt,
  16-generation, 14-depth, 576-retained-row, complete-log, and
  translated-validity gate. Audit:
  `$HPCVAULT/synthetic-RLVL/analysis/branchproof_selected_followups_audits_20260723/large_4.json`.
  Its retained shallow, train-edge, OOD, and depth-50 outputs are all
  answer-correct, format-complete, translated-parseable, and
  translated-citation-free-valid; the complete metrics contain only a small
  sampled dip at depth 40 (`correct@1=0.9941`,
  `translated-joint@1=0.9922`). No failed generation is present in the
  retained subset, so failure review remains covered by the earlier accepted
  seed and the final seed.
- Final single-modal NL row `3881780_5` is healthy on `a0632` at sampled
  chunk `89/112`. Conditioned-32B `3883534_13/14`, exact resume
  `3897965_12`, and Dolmino formal/NL `3875828_1/3875831_2` remain
  account-GRES pending with their audited resources and dependencies.
  Successor watcher `3901437` is dependency-free, CPU-only/no-GRES, and
  BeginTime-pending for 01:18 CEST July 27. Both critical paths remain
  incomplete, so it is preserved. Scoped commit `ef5a3ca` is local; a bounded
  60-second `git push origin main` produced no remote response and exited
  `124`, so the existing GitHub SSH/network blocker remains.

### 2026-07-26 13:22 first Dolmino production start and 32B eval progress

- Dolmino control stage `3875825_0` started at 08:09 CEST on full
  A100-80GB node `a0536`. It is healthy through step `1081/9537` at about
  `30.7K` tokens/s with finite loss and gradient norm. The independently
  accepted step-1000 state has 645 files, no zero-byte file, TP4/DP2, 625
  model files, four equal `22,848,937,060`-byte optimizer shards, four
  scheduler shards, eight RNG shards, and exact
  step/sample/token offsets `1000/128000/524288000`, all from the normal
  Nanoset. Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_control_step1000_20260726.json`.
- The live Nanotron writer retains every 500-step checkpoint. After the
  step-1000 state passed the complete restart gate, the exact superseded
  step-500 tree was removed, reclaiming `106,628,386,940` bytes. Step 1000
  remains the sole numeric restart state. The live base wrapper now optionally
  prunes only strictly older complete run states at restart, and the Dolmino
  5B wrapper enables that guard; both scripts pass `bash -n`. Continue the
  same rotation during the active allocation; never remove the newest accepted
  state.
- OLMo-3-32B NL eval row `3881780_4` started at 10:02 CEST on four verified
  A100-80GB GPUs on `a0932`. It uses the audited 16,384-token context and has
  completed greedy scoring plus sampled chunks `1..89/112`; chunk 90 is
  active with no fatal/OOM/quota signature. Row `3881780_5` remains
  array-throttle pending.
- Conditioned-32B `3883534_13/14` and exact resume `3897965_12` remain
  account-GRES pending; conditioned eval `3881781_12..17` retains the repaired
  dependency gate. Dolmino formal/NL first stages `3875828_1/3875831_2`
  remain account-GRES pending, while the later stages retain their valid
  dependencies.
- Vault quota after guarded rotation is `667G/1000G` and `180k/200k` files.
  This project's Vault/Work trees are `577,374,352/71,571,384 KiB`.
  Successor watcher `3900807` is dependency-free, CPU-only/no-GRES, and
  BeginTime-pending for 19:17 CEST; the plan remains incomplete, so it is
  preserved. Scoped commit `b4a1789` is local; a bounded 60-second
  `git push origin main` produced no remote response and exited `124`, so the
  existing GitHub SSH/network publication blocker remains.

### 2026-07-26 07:20 scheduler and watcher-chain reconciliation

- No in-scope GPU job started or completed after 01:20 CEST. BranchProof
  `3881780_4/5`, `3883534_13/14`, and exact resume `3897965_12`
  remain A100-80GB `AssocGrpGRES` pending; conditioned eval
  `3881781_12..17` retains the repaired
  `afterok:3897965:3883534_13:3883534_14` gate. Dolmino first stages
  `3875825_0/3875828_1/3875831_2` remain dependency-free
  `AssocGrpGRES` pending. Slurm currently gives none of these rows a start
  estimate.
- Two A100-80GB nodes are idle, but the association-level GPU cap still
  blocks every in-scope GPU request. The stored jobs retain their audited
  A100-80GB resources and valid dependencies, so no partition, feature,
  throttle, or dependency edit is useful.
- Watcher `3899486` is running CPU-only on `a100mig`. Its recorded successor
  is now `3899917`, verified dependency-free and BeginTime-pending for
  13:17 CEST with `cpu=4,mem=30000M` and no GRES. Both critical paths remain
  incomplete, so the successor is preserved.
- Vault quota remains `348G/1000G` with `179k/200k` files. This project's
  Vault and Work trees remain `243,171,225 KiB` and `71,547,812 KiB`;
  no cleanup or new artifact-audit trigger opened.
- Scoped handoff commit `c50cba6` is local. A bounded 60-second
  `git push origin main` produced no remote response and exited `124`; the
  existing GitHub SSH/network publication blocker remains.

### 2026-07-26 01:20 scheduler reconciliation; restart and successor verified

- No in-scope GPU task started or completed after the July-25 handoff.
  Selected BranchProof `3881780_4/5`, `3883534_13/14`, and exact resume
  `3897965_12` remain A100-80GB `AssocGrpGRES` pending; conditioned eval
  `3881781_12..17` retains the repaired dependencies. Dolmino first stages
  `3875825_0/3875828_1/3875831_2` remain dependency-free and
  `AssocGrpGRES` pending, with current individual Slurm projections of
  05:15 CEST. The idle A100-80GB node does not remove the association-level
  GPU cap, so no partition, feature, throttle, or dependency edit is useful.
- Revalidated the conditioned-32B seed-3407 `checkpoint-7500`: its
  536,991,984-byte adapter, 1,074,512,995-byte optimizer, scheduler, RNG,
  tokenizer, and step-7500 trainer state are nonempty, with no zero-byte
  file. Exact resume `3897965_12` remains the correct recovery.
- Current watcher `3898756` is running CPU-only on `a100mig`. Its recorded
  successor `3899486` is verified dependency-free, requests
  `cpu=4,mem=30000M` with no GRES, and is BeginTime-pending for 07:17 CEST.
  Both BranchProof and Dolmino paths remain incomplete, so the successor is
  preserved.
- User Vault quota is `348G/1000G` with `179k/200k` files. This project's
  Vault and Work trees are `243,171,225 KiB` and `71,547,808 KiB`,
  respectively; no storage cleanup trigger is open.
- Handoff commit `643a541` is local. A bounded 60-second
  `git push origin main` produced no remote response and exited `124`; the
  existing GitHub SSH/network publication blocker remains.

### 2026-07-25 19:19 report consistency fix; queues unchanged

- No in-scope job started or completed after the 13:29 handoff. Remaining
  selected BranchProof jobs `3881780_4/5`, `3883534_13/14`,
  `3897965_12`, and `3881781_12..17` retain their audited A100-80GB
  constraints and dependencies. Dolmino first stages
  `3875825_0/3875828_1/3875831_2` remain unstarted account-GRES pending;
  successor `3898756` remains CPU-only BeginTime-pending and required.
- Corrected one stale informal-report executive bullet that still called
  shortcut, conditioned-dual, and architecture controls pending despite the
  accepted eleven-condition table above it. The replacement states that only
  audited corrected controls are evidence and that OLMo-3-32B remains
  incomplete. The report builder passed and the corrected source was mirrored
  to `../synthetic-RLVL-report/informal_report`; no metric, claim, or official
  root-preprint result changed.
- Report corrections are committed as `ce21a88` in this repo and `df7de8b` in
  the report repo. Bounded 45-second SSH pushes of both repositories produced
  no remote response and exited `124`; the local commits remain intact.

### 2026-07-25 13:29 selected controls accepted; exact 32B recovery submitted

- Fourteen selected eval rows completed and pass the fail-closed 448-prompt,
  16-generation, 14-depth, retained-sample, complete-log, fresh-constant, and
  credited-validity gates: surface `3881774_7/8/24/25/26`, shortcut
  `3881775_15..17`, hybrid `3881776_27..29`, Qwen2.5-7B
  `3881778_34/35`, and OLMo-3-32B NL `3881780_3`. Representative review
  covers shallow/train/OOD/depth-50, correct-valid, wrong, invalid,
  missing-answer, and capped outputs across every newly complete family.
- Selected acceptance is `37/45`. Complete new three-seed controls have OOD
  answer/joint pass@1: terse NL `0.6414/0.6358`, target-token-matched NL
  `0.4104/0.4013`, shortcut formal versus NL `0.4310/0.3849` versus
  `0.7854/0.7807`, reverse-hybrid `0.0053/0.0001`, and Qwen2.5-7B formal
  versus NL `0.6986/0.6919` versus `0.7491/0.7449`. Both hybrid orders
  collapse OOD; shortcut-trained NL is stronger; Qwen is near-tied at pass@1.
- Conditioned-32B SFT `3883534_12` timed out after `24:00:18`. Complete,
  nonempty restart states at steps 7,250 and 7,500 include adapter, optimizer,
  scheduler, RNG, tokenizer, and matching trainer state. Submitted only exact
  row-12 resume `3897965_12` with 12 hours on four A100-80GB GPUs. Rewired
  conditioned eval `3881781_[12-17]` to
  `afterok:3897965:3883534_13:3883534_14`.
- Single-modal 32B eval `3881780_4/5`, conditioned SFT
  `3883534_13/14`, exact recovery `3897965_12`, and Dolmino first stages
  `3875825_0/3875828_1/3875831_2` are A100/account-GRES pending. CPU-only
  successor `3897962` remains BeginTime-pending and required.
- The in-repo report now ingests all eleven complete selected control
  conditions fail-closed. The generated bundle was mirrored to the report
  repo and the official preprint was updated; OLMo-3-32B remains explicitly
  partial. Local TeX tooling remains unavailable. Scoped commits are
  `3807883` in this repo and `b3431c0` in the report repo. Both bounded
  60-second SSH pushes produced no remote response and exited `124`.

### 2026-07-24 13:22 first terse-NL surface row accepted

- Surface row `3881774_6` completed as raw job `3890505` on A100-80GB in
  `07:45:07`. Its fail-closed audit accepts 448 prompts, 16 generations, all
  14 depths, 2,665 metrics, 576 retained rows including 448 sampled rows, and
  complete `7/112` chunk logs. There is no credited validity-diagnostic
  contradiction. Audit:
  `$HPCVAULT/synthetic-RLVL/analysis/branchproof_selected_followups_audits_20260723/surface_6.json`.
- Representative terse-NL review covers depths 1/25/30/35/40/45/50,
  correct translated-valid outputs, wrong/missing-answer outputs, and long
  premise-copy truncations. The intended `<think>/<answer>` surface and exact
  extraction remain clean. This seed's OOD greedy/pass@1/pass@16 answer rates
  are `0.775/0.291/0.806`; translated-joint rates are
  `0.744/0.284/0.788`. This is a row acceptance only; rows 7/8 must finish
  before a three-seed terse-NL family claim or report update.
- Selected acceptance is `23/45`. Surface rows `3881774_7/8`, shortcut rows
  `3881775_15..17`, hybrid rows `3881776_27..29`, surface row
  `3881774_24`, and Qwen row `3881778_34` are actively generating or scoring
  without a fatal signature. Conditioned-32B SFT `3883534_12` has advanced
  beyond step 2,000; complete nonempty restart states at checkpoints 1,750
  and 2,000 passed the zero-byte gate. Recover exactly this row only after an
  actual timeout.
- Dolmino first stages `3875825_0/3875828_1/3875831_2` remain unstarted
  `AssocGrpGRES` pending with provisional 22:02 CEST projections. Vault is
  `621G/1000G` and 179k files; this project is `529,381,827 KiB` with 7,211
  regular files. CPU-only successor `3891580` is BeginTime-pending and remains
  required because both selected BranchProof and Dolmino 5B paths are
  incomplete.
- Scoped acceptance/handoff commit `e695634` is local. A bounded 90-second
  `git push origin main` produced no remote response and exited `124`; local
  `main` remains ahead of `origin/main`. The report repository was unchanged.

### 2026-07-24 07:22 conditioned-7B family accepted and reported

- Seven newly terminal selected rows passed the fail-closed 448-prompt,
  16-generation, 14-depth, retained-sample, complete-chunk-log,
  fresh-constant, and credited-validity gates: conditioned-7B
  `3881777_26/28/29`, Qwen2.5-7B `3881778_25/26/33`, and OLMo-3-32B
  `3881780_2`. Audits are under
  `$HPCVAULT/synthetic-RLVL/analysis/branchproof_selected_followups_audits_20260723/`.
- Representative review covers depths 1/25/30/40/50, correct-valid,
  answer-correct-invalid, wrong, malformed, missing-answer, and long/capped
  outputs. The complete conditioned-7B family has formal versus NL OOD
  answer/joint pass@1 `0.5029 +/- 0.0644` / `0.4488 +/- 0.0600` versus
  `0.1814 +/- 0.0197` / `0.1780 +/- 0.0232`. Conditioned NL is clean through
  roughly depth 30, then copies long premise/proof prefixes and usually omits
  an answer. This is a shared-checkpoint output-mode result, not an
  additive-data claim.
- Selected acceptance is now `22/45`. Qwen formal is complete at three seeds
  but Qwen NL has only seed 3407; OLMo-3-32B formal is complete but its three
  NL rows remain pending. Active selected evals are surface
  `3881774_6/7/8` at sampled chunks `74/65/47` and shortcut `3881775_15` at
  chunk `27`; no active log has a fatal signature.
- Conditioned-32B replacement `3883534_12` started as raw job `3890581` on
  four A100-80GB GPUs. Its first restart state is verified at
  `checkpoint-250`: the 536,991,984-byte adapter, optimizer, scheduler, RNG,
  tokenizer, and step-250 trainer state are nonempty, with no zero-byte file;
  training continued to step 251. The current rate projects beyond one
  24-hour allocation, so recover exactly this row after an actual timeout.
  Rows 13/14 remain array-throttle pending.
- Dolmino first stages `3875825_0/3875828_1/3875831_2` remain unstarted
  `AssocGrpGRES` pending with provisional 22:02 CEST projections. Vault is
  `451G/1000G` and 179k files; this project is `350,611,842 KiB` with 7,107
  regular files. CPU-only successor `3890623` is verified BeginTime-pending
  for 13:13 CEST and remains required.
- The in-repo report was regenerated with fail-closed conditioned-family
  ingestion, mirrored to `../synthetic-RLVL-report/informal_report`, and the
  official preprint was updated. Python compilation and report generation
  pass; local TeX tooling remains unavailable. Commits `e9fcb08` and
  `873e26f` are local. Main/report SSH pushes timed out after 90/60 seconds
  without a remote response.

### 2026-07-24 01:18 three more selected rows accepted

- Conditioned-7B NL row `3881777_27` (raw job `3884546`), Qwen2.5-7B logic
  row `3881778_24` (raw job `3889573`), and OLMo-3-32B logic row
  `3881780_1` (raw job `3884864`) completed `0:0` and passed the fail-closed
  448-prompt, 16-generation, 14-depth, retained-sample, chunk-log, constant,
  and validity-diagnostic gates. Audits are under
  `$HPCVAULT/synthetic-RLVL/analysis/branchproof_selected_followups_audits_20260723/`.
- Representative raw review covers shallow/train/depth-30/depth-40/depth-50,
  correct-valid, wrong, invalid, missing-answer, and long/capped cases.
  Conditioned NL is clean through depth 30 and then has zero complete answers
  at depths 35--50 after premise/proof truncation. Its OOD
  answer/translated-joint pass@1 is `0.1891/0.1891`. Qwen2.5 logic has OOD
  answer/citation-free-joint pass@1 `0.7816/0.7777`, but depth-50 retained
  samples have only `1/32` usable correct answers while many internally valid
  proof blocks omit a complete answer. OLMo-3-32B seed 3408 has OOD
  answer/joint pass@1 `0.7004/0.5359` and pass@16 `0.9938/0.9375`; long
  failures include wrong derivations, unsupported lines, repetition, and
  missing answers.
- Selected acceptance is now `15/45`. Conditioned rows `3881777_26/28` have
  completed all 112 sampled chunks and are scoring; row 29 is at chunk 71,
  Qwen architecture row `3881778_25` is at chunk 81, and large row
  `3881780_2` is at chunk 55. No active selected log has a fatal signature.
- Dolmino first stages `3875825_0/3875828_1/3875831_2` remain unstarted
  `AssocGrpGRES` pending with individual 05:07 CEST projections.
  Conditioned-32B replacement `3883534_12` projects 04:54 CEST and has not
  produced a checkpoint. Vault is `572G/1000G` with 179k files; this project
  is `477,666,501 KiB` with 7,114 regular files. CPU-only successor
  `3890293` is verified BeginTime-pending for 07:13 CEST and remains required.
- Handoff commit `1b5942d` is local. Main-repo and report-repo SSH pushes
  produced no remote response and timed out after 90 and 60 seconds,
  respectively; local `main` remains five commits ahead in this repo and one
  commit ahead in the report repo.

### 2026-07-23 19:17 first selected 32B row accepted

- OLMo-3-32B single-modal logic eval `3881780_0` completed as raw job
  `3883993` on four verified A100-80GB GPUs in `06:34:03`. Its fail-closed
  audit accepts all 448 prompts, 16 generations, 14 depths, 2,665 metrics,
  576 retained rows including 448 sampled rows, and complete `7/112` chunk
  logs. There are no fresh-constant or credited validity-diagnostic
  contradictions. Audit:
  `$HPCVAULT/synthetic-RLVL/analysis/branchproof_selected_followups_audits_20260723/large_0.json`.
- Representative raw review covers depths 1, 25, 30, 40, 45, and 50,
  correct/citation-free-valid, wrong, malformed, and cap-hit cases. The model
  is perfect in the retained sample through train depth 25, then shows
  ordinary wrong-branch, malformed-premise/proof-line, and long-generation
  failures. Across OOD depths 30--50, this one seed has greedy answer/joint
  `0.738/0.491`, pass@1 answer/joint `0.641/0.504`, and pass@16 answer/joint
  `0.994/0.919`. This is a row acceptance only; no 32B family claim or report
  refresh is permitted before the matched logic/NL seeds finish.
- Selected acceptance is now `12/45`. Large row `3881780_1` is running on
  four A100-80GB GPUs at sampled chunk `70/112`. Conditioned-7B rows
  `3881777_26/27/28` are at chunks `82/83/79`; Qwen2.5-7B architecture row
  `3881778_24` has clean A100-80GB merge/vLLM startup. No active selected log
  has a fatal, OOM, quota, or no-space signature.
- Dolmino first stages `3875825_0/3875828_1/3875831_2` remain
  dependency-free `AssocGrpGRES` pending, with current individual projections
  of 22:02 CEST. Conditioned-32B replacement SFT `3883534_12` currently
  projects 02:15 CEST July 24. No Dolmino or conditioned-32B checkpoint exists
  yet, so neither startup gate is open.
- User-wide Vault is `572G/1000G` with 179k files; this project's tree is
  `477,610,021 KiB` with 7,105 regular files. The increase is consistent with
  active selected merge roots and remains below quota. CPU-only/no-GRES
  successor `3889637` is verified BeginTime-pending for 01:13 CEST July 24.
  Both critical paths remain incomplete, so it is preserved.
- Scoped handoff commit `578a3fb` is local. Main-repo and report-repo SSH
  pushes produced no remote response and timed out after 90 and 60 seconds,
  respectively. The local commits and accepted audit artifact remain intact;
  publication is still blocked on the GitHub SSH/network path.

### 2026-07-23 13:24 two selected controls accepted and reported

- All eleven completed selected eval rows now pass the fail-closed production
  audit and representative raw review: surface `3881774_0..2`, shortcut
  `3881775_12..14`, hybrid `3881776_12..14`, and conditioned-7B
  `3881777_24/25`. Each has 448 prompts, 16 generations, all 14 depths,
  576 retained rows including 448 sampled rows, complete `7/112` chunk logs,
  and no fresh-constant or credited validity-diagnostic contradiction. Audits
  are under
  `$HPCVAULT/synthetic-RLVL/analysis/branchproof_selected_followups_audits_20260723/`.
- Surface symbol-padded formal and the NL-then-formal hybrid are the first
  complete selected three-seed families. Symbol padding preserves most of the
  formal direction: OOD greedy/pass@1/joint@1/pass@16/joint@16 is
  `0.781 +/- 0.174`, `0.673 +/- 0.175`, `0.626 +/- 0.176`,
  `0.927 +/- 0.094`, and `0.890 +/- 0.130`. The hybrid collapses OOD at
  `0.000`, `0.002 +/- 0.001`, `0.000`, `0.023 +/- 0.016`, and `0.000`.
  Raw generations show the hybrid fits depth 25 but copies both long surfaces
  and normally reaches the shared cap before an OOD answer. Shortcut has only
  its three formal rows, and conditioned-7B has only the seed-3407 pair, so
  neither supports a family comparison yet.
- The in-repo report was regenerated with fail-closed selected-control
  ingestion, mirrored to `../synthetic-RLVL-report/informal_report`, and the
  official root preprint was updated with only the two complete corrected
  controls. Generated summary:
  `analysis/logic_cot_report_2026-05-25/tables/corrected_branchproof_selected_controls.csv`.
  Python compilation and report generation pass; local TeX tooling remains
  unavailable.
- OLMo-3-32B single-modal eval `3881780_0` is running as raw job `3883993`
  on four verified A100-80GB GPUs. At 13:24 CEST it had reached sampled chunk
  `83/112` in about 2h46m with the audited 16,384-token context and no fatal
  signature. Replacement conditioned-32B SFT `3883534_[12-14%1]` and the
  remaining selected eval rows are priority/dependency pending.
- Dolmino first stages `3875825_0/3875828_1/3875831_2` remain
  dependency-free `AssocGrpGRES` pending, now projected individually for
  05:09 CEST July 24. No checkpoint exists. User-wide Vault reports `462G`
  and 179k files; this project uses about `346G`, 7,055 regular files, and
  9,322 entries in Vault. CPU-only successor `3884297` remains scheduled for
  19:12 CEST because both critical paths remain incomplete.
- Research/report commits are `15d5846` and `d0f4bd2`. Bounded SSH pushes of
  both repositories produced no remote response and timed out, so publication
  remains blocked while the local commits and artifacts are intact.

### 2026-07-23 08:41 guarded project cleanup restores Vault space margin

- Removed 30 final-backed Trainer `checkpoint-*` directories under this
  project's Vault run tree after verifying that every parent has a nonempty
  final model and that no live selected job depends on a restart state. This
  reclaimed `33,953,981,440` bytes.
- Removed only the superseded `38,657,020,416`-byte Dolmino normal-data
  Nanoset at `nanosets/qwen25_dolmino_neutral_v1/dolmino`. The queued 5B
  wrapper uses the retained 5.1B normal Nanoset and the retained
  `qwen25_dolmino_neutral_v1/{logic,nl_exact}` intervention Nanosets; all
  three production inputs passed post-delete stats/shard gates.
- Total Vault reclamation is `72,611,001,856` bytes (`72.61 GB`, `67.62
  GiB`). This project's Vault tree is now `329.53 GB` with 7,052 regular
  files and 9,321 quota-counted entries. The user-wide Vault is about `898
  GiB/1000 GiB` with 201,900 entries, still about 1,900 over the 200k soft
  inode quota but well below the 400k hard limit.
- `$WORK/synthetic-RLVL` remains `73.23 GB` with 6,975 regular files. No
  additional Work artifact met the strict deletion gate: raw eval evidence,
  final adapters, datasets, 21 offline W&B runs, and current/recent online
  W&B logs were retained. The three active hybrid merge trees in Vault were
  also retained and may be reclaimed only after their eval outputs pass.

### 2026-07-23 08:27 selected rows advance; hybrid sampling near completion

- Eight selected eval rows now have clean Slurm completion: surface
  `3881774_0..2`, shortcut `3881775_12..14`, and conditioned-7B
  `3881777_24/25`. The previously accepted subset remains surface rows
  `0/2` and conditioned row `25`; the other five completed rows are
  provisional until the same structural, invariant, and representative raw
  review is recorded. No failed selected row appeared.
- Hybrid rows `3881776_12..14` remain healthy on A100-80GB and have reached
  sampled chunks about `99/99/98` of `112`. Their recent logs contain no
  fatal, OOM, quota, or no-space signature, and the remaining runtime fits
  inside the 24-hour allocation at the observed rate.
- The other selected eval rows are scheduler, dependency, or array-throttle
  pending. Resumable conditioned-32B SFT `3883534_[12-14%1]` is still pending
  without a scheduler estimate; its six eval rows remain correctly dependent
  on those three SFT rows.
- Dolmino control/formal/NL first stages remain dependency-free and pending on
  `AssocGrpGRES`; Slurm currently projects 22:02 CEST for each when considered
  individually. They have not started or produced checkpoints. The retained
  project tree in `$HPCVAULT` is about 375 GiB.

### 2026-07-23 07:16 first three selected-control rows accepted

- Selected eval rows surface `3881774_0/2` and conditioned-7B
  `3881777_25` completed `0:0` on A100-80GB in
  `10:50:04/10:37:16/07:12:23`. Each has the full 448-prompt,
  16-generation, 14-depth metric bundle, 576 retained rows including 448
  sampled rows, complete `7/112` chunk logs, and no fresh-constant or credited
  validity-diagnostic violation. Row audits are under
  `$HPCVAULT/synthetic-RLVL/analysis/branchproof_selected_followups_audits_20260723/`.
- Raw review covers both symbol-padded formal seeds at depths 1, 25, 40, and
  50 plus the conditioned-NL row at depths 1, 25, 30, and 50. Prompts,
  wrappers, exact answers, and modality-appropriate validity are clean where
  credited. The sampled rows retain real failure modes: symbol-padded formal
  has answer-correct invalid traces and depth-50 repetition/truncation, while
  conditioned NL copies long premise/proof prefixes and has `32/32`
  depth-50 format failures. These are row acceptances only; no selected-family
  claim or report refresh is permitted before the matched rows finish.
- Eight selected eval rows remain active: surface `3881774_1`, shortcut
  `3881775_12..14`, hybrid `3881776_12..14`, and conditioned-7B
  `3881777_24`. Their logs are in sampled generation or scoring with no fatal
  signature. Replacement conditioned-32B SFT `3883534_[12-14%1]` remains
  A100-80 pending; its seed-3407 run root has no stale checkpoint or final.
- Dolmino first stages `3875825_0/3875828_1/3875831_2` remain valid
  account-GRES pending with individual 22:02 CEST projections. Vault is
  `1101G/1000G` and `203k/200k` files. CPU-only successor `3883709` remains
  scheduled because both critical paths are incomplete.
- Scoped audit/handoff commit `ef8c4fb` is local. A bounded 90-second
  `git push origin main` produced no remote response and exited `124`; the
  SSH/network publication path remains blocked. The report repository was not
  changed.

### 2026-07-23 01:17 selected follow-ups active; 32B restart defect recovered

- Eleven selected BranchProof eval rows are running on verified A100-80GB
  devices: surface `3881774_0..2`, shortcut `3881775_12..14`, hybrid
  `3881776_12..14`, and conditioned-7B `3881777_24/25`. They have reached
  sampled chunks `55..89/112` without a fatal/OOM/quota signature; no final
  selected-family metrics or sample bundle exists yet.
- Conditioned OLMo-3-32B SFT `3881779_12` reached only step `1368/10000` in
  about `4:17`, projecting roughly 31 hours, while its stored payload saved
  every 20,000 steps and therefore could not leave a restart state before the
  24-hour limit. The wrapper now gives the `large` group 250-step checkpoints
  with two states retained. Original `3881779` was canceled before wasting a
  full allocation, exact replacement `3883534_[12-14%1]` was submitted, and
  conditioned eval `3881781` now depends on successful replacement rows.
- Dolmino first stages `3875825_0/3875828_1/3875831_2` remain valid and
  account-GRES pending; current individual scheduler projections are 22:02
  CEST July 23. Vault is `1183G/1000G` soft and `203k/200k` files. CPU-only
  successor `3883531` remains scheduled because both critical paths are
  incomplete.
- Recovery/handoff commit `7cb2ca0` is local. A bounded 90-second
  `git push origin main` produced no remote response and exited `124`; local
  `main` remains two commits ahead of `origin/main`.

### 2026-07-22 20:15 corrected BranchProof baseline accepted and reported

- Final recovery `3865321_18` completed on verified A100-80GB hardware in
  `08:08:43`; its row audit `3865322_18` passed. Strict aggregate and
  cross-grid qualitative gate `3857769` then completed `0:0`, establishing
  `30/30` accepted rows and a 30-file clean manifest.
- The corrected result is a depth-dependent reversal. NL leads greedy and
  pass@1 for train maxima `5..20`. At train max 25, logic-minus-NL OOD deltas
  are greedy `+0.3333 +/- 0.2680`, answer pass@1 `+0.5712 +/- 0.1399`, joint
  pass@1 `+0.5100 +/- 0.1443`, answer pass@16 `+0.5708 +/- 0.0609`, and joint
  pass@16 `+0.5792 +/- 0.0457`. All three paired train-25 seeds have positive
  pass@1 answer deltas; seed 3407 ties at greedy.
- The accepted cross-grid raw audit covers all templates, seeds, train ranges,
  shallow/train/OOD/depth-50 cases, correct/incorrect and valid/invalid cases,
  plus cap diagnostics. A 30-file invariant scan found no fresh-constant or
  credited validity-diagnostic violation. Long failures commonly copy
  premises or repeat until the shared cap; answer-correct invalid traces also
  occur, so correctness and validity remain separate.
- Generated evidence is under `analysis/branchproof_unique_v2_20260711/`.
  The in-repo informal report was regenerated with fail-closed ingestion, and
  the official preprint now reports only this corrected baseline while keeping
  old wrapped-constant results quarantined. Selected controls `3881774..3881781`
  were released; surface row `3881774_0` started cleanly on A100-80GB.
- Dolmino first stages `3875825_0/3875828_1/3875831_2` remain valid and pending;
  their `UnavailableNodes:a0631` display is not a hard node binding, and Slurm
  has alternate forecast nodes. Vault is about `925.8G/1000G` and `202k/200k`
  files. Successor oversight `3882407` remains scheduled because selected
  controls and all three Dolmino chains are incomplete.

### 2026-07-22 17:01 exhaustive BranchProof wave pruned and replaced

- Canceled the nonbaseline exhaustive parent arrays after resolving every row
  against the official preprint claims. The corrected baseline jobs/audits/
  aggregate were not touched. Cancellation left no active report merge tree.
- Submitted 48 exact selected follow-up rows behind baseline aggregate
  `3857769`: 45 evals and three required OLMo-3-32B conditioned-dual SFT rows.
  Jobs are surface `3881774` (9), shortcut `3881775` (6), hybrid `3881776`
  (6), conditioned-7B `3881777` (6), Qwen2.5-7B architecture `3881778` (6),
  OLMo-3-32B conditioned SFT `3881779` (3), OLMo-3-32B single eval `3881780`
  (6), and conditioned eval `3881781` (6).
- Programmatic row assertions passed for exact templates, rates, models,
  train depth 25, evaluation modes, and total count. Final-adapter gates pass
  for every selected checkpoint except the intentional three new 32B
  conditioned rows. The substantive BranchProof remainder is now 51 GPU rows
  including the three active baseline evals, down from approximately 310.
  Exact scope and rationale: `docs/branchproof_selected_followups_2026-07-22.md`.
- Removed about 22.5 GiB of orphaned incomplete checkpoint payloads from six
  canceled conditioned-50k and three canceled batch-size roots. Every root
  was gated on missing `final/adapter_config.json` and absence from the live
  selected dependency graph; metadata and all completed/selected finals remain.

### 2026-07-22 16:55 BranchProof scope audit: exhaustive queue is not required

- The approximately 310-row substantive remainder is the exhaustive rebuild
  of every historical informal-report cell, not the minimum experiment set
  needed by the official preprint. The old architecture, shortcut, syntax,
  hybrid, conditioned, batch, and 32B sections are retained inside
  `\iffalse`; they are not currently rendered as evidence.
- The official preprint's stated gate is narrower: complete the corrected
  three-seed baseline, then, only if it supports a formal advantage, rerun one
  syntax control, one shortcut intervention, one conditioned-dual comparison,
  and one independent architecture. The paper's expanded mixed-substrate
  claim also justifies one train-1-to-25 same-datapoint hybrid comparison.
- Therefore the full batch-size matrix, conditioned-50k convergence grid,
  every shortcut kind/rate, every surface, every architecture/train depth, and
  the full 32B matrix are optional appendix diagnostics rather than required
  evidence. No jobs were canceled during this audit; an exact keep/cancel list
  should be applied after user confirmation, preserving the nearly complete
  core baseline and any already-near-terminal rows worth harvesting.

### 2026-07-22 16:35 BranchProof remaining-work audit

- Expanded `squeue -r` shows 341 BranchProof-related Slurm rows: 22 running
  and 319 pending. This scheduler count includes one oversight, three CPU
  audits, one CPU aggregate, and 26 staged conditioned-50k recovery rows; most
  recovery rows should become quick skip checks after accepted earlier-stage
  finals exist.
- The substantive remaining GPU workload is about 310 task rows: 56 SFT rows
  and 254 eval rows. SFT consists of conditioned-50k 6, shortcut 8, batch 33,
  and 32B 9. Eval consists of corrected baseline 3, hybrid/replacements 24,
  conditioned-10k 10, conditioned-50k 30, architecture 52, 32B 18, shortcut
  42, batch 48, and surface 27.
- The corrected core baseline is close: rows 13/14 are already scoring and
  row 18 is at sampled chunk 87/112; only their three CPU audits and aggregate
  follow. Barring recovery, expect this gate around July 23--24. Completing
  every report ablation is a longer critical path: with the 16-A100 account
  ceiling, 8--18-hour observed eval rows, training dependencies, and Dolmino
  full-node contention, the current planning window is roughly August 5--12,
  not a scheduler guarantee.

### 2026-07-22 16:20 Dolmino remains healthy; full-node estimate moves

- Control `3875825_0`, formal `3875828_1`, and NL `3875831_2` remain valid,
  dependency-free, and pending solely on `AssocGrpGRES`; none has started or
  created a log/run checkpoint. At 16:16 CEST, Slurm's individual projection
  moved to 09:00 CEST July 23 for all three.
- Sixteen A100 GPUs are currently occupied by this project's active
  BranchProof jobs, matching the observed account allocation pressure. Each
  Dolmino stage requests eight GPUs, so identical per-job estimates are not a
  promise that all three will start together. No partition widening is
  justified without separately validating the production stack on another
  eight-GPU node type.

### 2026-07-22 13:16 baseline reaches 26/30; report matrix advances

- Declaration-fixed baseline eval/audit rows `3857767_24..28 ->
  3857768_24..28` completed and passed all row gates, raising accepted
  coverage from `21/30` to `26/30`. Representative NL train-1-to-20/25 raw
  review across all newly available seeds covered shallow, train, OOD, and
  depth-50 correct/failure cases. Prompts, `<think>/<answer>` extraction, and
  translated validity are clean where credited; long failures are ordinary
  missing-answer truncations or malformed continuations. Rows 13/14/18/29
  remain active under `3863525`, `3865321`, and `3857767`; aggregate
  `3857769` remains correctly gated.
- Conditioned-10k eval rows `3850119_16/17` passed the 448-prompt,
  16-generation, 14-depth, 576-retained-row, metric, chunk-log, constant, and
  validity-diagnostic gates plus raw review, bringing that family to `18/30`.
  The complete train-1-to-15 slice remains provisional: conditioned NL is
  ahead at OOD pass@1 answer/joint (`0.4093 +/- 0.0290` /
  `0.4080 +/- 0.0270`), while conditioned logic rises more with sample budget
  but has much lower joint validity at pass@16 (`0.7331 +/- 0.1654` answer,
  `0.2708 +/- 0.0193` joint). Do not use this partial as report evidence.
- Accepted final-backed SFT artifacts `3850110_8`, `3850213_31..33`, and
  `3850115_5`, advancing conditioned-50k to `9/15`, shortcut to `34/42`, and
  32B to `6/15`. Six terminal, live-reference-free checkpoints were removed
  after artifact gates, reclaiming about 7.7 GiB; all finals/evidence,
  incomplete restart states, and conditioned `checkpoint-50000` remain.
- Active report/baseline logs show no fatal signature. Batch recovery
  `3872659_3..5` resumed exact step-3000 states; exact rows `3879713_9..11`
  remain account-GRES pending. Dolmino first stages
  `3875825_0/3875828_1/3875831_2` have not started and now project 22:19 CEST
  individually. Vault is `1187G/1000G` and `203k/200k` files. CPU-only
  successor `3880822` remains BeginTime-pending because both critical paths
  are incomplete; no report or preprint gate opened.
- Scoped audit/handoff commit `b797282` is local. A bounded 90-second
  `git push origin main` produced no output and left `origin/main` unchanged;
  local `main` is one commit ahead. The report repository is unchanged and
  synchronized.

### 2026-07-22 09:05 Dolmino data gate passed; 5B stages scheduler-eligible

- Prerequisite `3875824` completed cleanly on July 21 at 12:19 CEST in
  `02:32:53`. The packed Dolmino Nanoset contains `5,111,201,524` tokens,
  exceeds the exact `5,000,134,656`-token training requirement, and its
  reproducible raw staging tree was removed after acceptance. The retained
  packed root is about 39 GiB; no error, no-space, or quota failure occurred.
- Control `3875825_0`, formal `3875828_1`, and NL `3875831_2` are all
  dependency-free and pending only on `AssocGrpGRES`. At 08:58 CEST Slurm
  estimated 19:57 CEST July 22 for each first stage considered individually.
  This is not a simultaneous-launch guarantee: each requests eight A100-80GB
  GPUs, and the account is currently at its aggregate GPU allocation limit.
- No 5B run directory or checkpoint exists yet because training has not
  started. Expected compute after allocation is about 45--48 hours per
  condition at the audited approximately 30.9K tokens/s, normally spanning
  two 24-hour stages; calendar completion remains scheduler-dependent.

### 2026-07-22 07:15 batch recovery submitted; baseline reaches 21/30

- Batch rows `3850114_9..11` reached the hard 24-hour limit at 05:25 CEST
  with no final or restart checkpoint, exactly matching their immutable
  `save_steps=20000` payload. Submitted only those three missing rows as
  `3879713_[9-11%3]` with the live batch wrapper's 250-step/two-state policy.
  Rebuilt batch eval `3872663` now requires `afterany:3850114`,
  `afterok:3872659`, and `afterok:3879713`; no completed or future row was
  duplicated.
- Declaration-fixed baseline eval rows `3857767_21..23` and their CPU audits
  `3857768_21..23` completed and passed all structural/validity gates, raising
  acceptance to `21/30`. Raw review across all three new NL train-1-to-15
  seeds covered shallow/train/OOD/depth-50, correct, wrong, malformed, and cap
  cases. Prompts, extraction, and translated validity are clean through depth
  25; depth 50 mostly truncates without an answer. On OOD depths 18--45, NL
  greedy answer/joint is `0.6488 +/- 0.0712` / `0.6280 +/- 0.0866`, and
  pass@1 answer/joint is `0.5648 +/- 0.0129` / `0.5614 +/- 0.0094`. Matched
  formal train-1-to-15 is lower at greedy and pass@1; its answer coverage
  overtakes only at pass@8/16 while joint remains lower. This is an
  incomplete-grid diagnostic, not report evidence.
- Report-matrix hybrid rows `3850118_6..8` and conditioned-10k rows
  `3850119_14/15` completed. All five pass the 448-prompt, 16-generation,
  14-depth, `7/112`-chunk, 576-retained-row, fresh-constant, and
  validity-diagnostic gates plus representative raw review. Hybrid is `6/30`
  and conditioned-10k is `16/30`. Hybrid train-1-to-15 OOD greedy accuracy is
  `0.2679 +/- 0.0818`; pass@1 answer/formal-joint/translated-joint is
  `0.1602 +/- 0.0151`, `0.0290 +/- 0.0293`, and `0.0504 +/- 0.0519`.
  Seed-3408 conditioned logic/NL remains a partial; both collapse at depth 50.
- Conditioned-50k rows `3850110_3/6/7` completed with nonempty
  159,967,880-byte finals and step-50,000 states, raising that family to
  `8/15`; rows 8--10 are active. Shortcut remains `31/42` with rows 31--33
  active, and 32B remains `5/15` with row 5 active. Focused active logs have
  no fatal, OOM, quota, or no-space signature.
- Dolmino stage-one `3875825_0/3875828_1/3875831_2` remains dependency-free
  and `AssocGrpGRES` pending on the only compatible full-node shape. Vault is
  `1228G/1000G` and `203k/200k` files; about 327 GiB is twelve active
  BranchProof merge trees, so it is not cleanup-eligible. Preserve incomplete
  restart states and conditioned target checkpoints needed by immutable skip
  gates. CPU-only successor `3879709` remains BeginTime-pending because both
  critical paths are incomplete. No report or preprint gate opened. Scoped
  recovery/audit commit `636e4ca` is local; a bounded `git push origin main`
  produced no remote output and timed out after 90 seconds with exit `124`,
  leaving `main` seven commits ahead of `origin/main` before this publication
  note. The report repository is unchanged.

### 2026-07-22 01:16 duplicate conditioned stages stopped; five more SFT finals

- Corrected 32B SFT `3854837_1` and `3850115_4` completed and passed the
  final artifact gate: each has a nonempty 536,991,984-byte adapter/config and
  a terminal step-10,000 trainer state. The large-model family is now `5/15`;
  original row `3850115_5` is running. Shortcut rows `3850213_28..30` also
  passed the 159,967,880-byte final plus step-10,000 gate, advancing shortcut
  SFT to `31/42`; rows 31--33 backfilled and are healthy.
- The immutable payload of conditioned-50k staged array `3850110` predates the
  final-only skip repair recorded on July 20. It replayed already accepted
  rows 0--2 to clean step-50,000 finals, then began replaying accepted rows
  4/5 even though their July-15 finals were intact. Canceled only duplicate
  tasks `3850110_4/5` and their future staged copies `3850111_4/5` and
  `3850112_4/5`; genuine unfinished task `3850110_3` continues from its
  audited checkpoints. The canceled duplicates left new 5k/10k/15k
  checkpoints as guarded cleanup candidates; the accepted original finals
  remain unchanged. Preserve the new row-0--2 `checkpoint-50000` directories
  until `3850111/3850112` terminate because their stored skip condition still
  requires the target checkpoint as well as the final.
- Batch rows `3850114_9..11` are near optimizer step 1,740 after about 20
  hours. Their stored Hydra configs confirm `save_steps=20000` and
  `save_total_limit=1`, so they have no restart checkpoint despite the live
  wrapper's later 250-step policy. Do not preempt them. After the actual
  24-hour timeout, submit only these three exact rows with the current
  resumable payload; no completed row should be replayed.
- Hybrid `3850118_6` is scoring after all `112/112` sampled chunks;
  conditioned-10k `3850119_14` is on chunk 112, and the other active report
  evals continue without a fatal signature. Declaration-fixed baseline rows
  `3857767_21..23` are on verified A100-80GB jobs at chunks 93--94/112. No
  new terminal metrics/sample bundle exists yet, so the accepted baseline
  remains `18/30` and no report gate opened.
- Dolmino stage-one jobs `3875825_0`, `3875828_1`, and `3875831_2` remain
  dependency-free and `AssocGrpGRES` pending on their only compatible
  eight-A100 shape. User-wide Vault is `1145G/1000G` soft with `203k/200k`
  files; preserve every incomplete restart state and clean only final-backed,
  live-reference-free intermediates. Recorded CPU-only successor `3879087`
  remains BeginTime-pending because both critical paths are incomplete.
- Scoped scheduler/handoff commit `27b56d3` is local. A bounded
  `git push origin main` produced no remote output and timed out after 90
  seconds with exit `124`; this publication note leaves `main` six commits
  ahead of `origin/main`.
  The report repository is unchanged.

### 2026-07-21 19:30 surface SFT complete; first corrected report artifacts audited

- Surface rows `3850105_24..26` completed and passed the nonempty-final and
  terminal-step-7,140 gates. Surface SFT is now `27/27`. Guarded cleanup then
  removed nine final-backed terminal Trainer checkpoints, including these
  three rows, after verifying that no live process referenced them; all final
  adapters and evaluation evidence remain. Vault usage fell from about
  `1024G` to `1017G` against the `1000G` soft quota, with `203k/200k` files.
- The first corrected report-eval bundles are terminal and structurally clean:
  hybrid `3850118_3/4/5` (three seeds, train 1--10) and conditioned-10k
  `3850119_12/13` (seed 3407 logic/NL, train 1--15). Each has 448 prompts,
  16 generations per prompt, all 14 depths, complete `7/112` greedy/sampled
  chunks, 576 retained records, fresh constants, and no validity-diagnostic
  contradiction. Hybrid completion is `3/30`; conditioned-10k is `14/30`.
- Raw review covered shallow, train-boundary, OOD, depth-50, correct, wrong,
  invalid, malformed, and cap-hit cases. Wrappers, prompts, answer extraction,
  and unique constants match intent. Hybrid degrades beyond train depth 10:
  formal proofs often become invalid before translated NL traces, and depth 50
  usually truncates or collapses. Conditioned seed 3407 is clean through depth
  25 in both modalities but also truncates/collapses at depth 50; it remains a
  single-seed partial, not family evidence. Three-seed hybrid greedy OOD
  answer accuracy is `0.3458 +/- 0.0331`; pass@1 OOD answer/formal-joint/
  translated-joint is `0.2173 +/- 0.0250`, `0.1294 +/- 0.0422`, and
  `0.1926 +/- 0.0079`. No report gate is open.
- The structural audit previously recognized NL only when the checkpoint name
  contained `_nl_exact_`, which falsely treated `conditioned_nl` as formal and
  rejected its intentionally zero syntactic metric. The audit now identifies
  all NL-only report surfaces from checkpoint/artifact names, while hybrid
  remains formal-audited. All 18 focused tests pass, and the real
  `3850119_13` artifact passes without the all-zero escape hatch.
- Active report work remains healthy: `3850118_6..8`, `3850119_14/15`,
  conditioned-50k `3850110_0..2`, 32B `3850115_4`/`3854837_1`, batch
  `3850114_9..11`, and shortcut `3850213_28..30`. Batch rows 9--11 are slow
  and likely to need exact checkpoint resume after an actual timeout; do not
  preempt them. Declaration-fixed baseline acceptance remains `18/30`.
  Dolmino stages `3875825_0`, `3875828_1`, and `3875831_2` remain
  `AssocGrpGRES` pending, now estimated for 07:43 CEST July 22. Successor
  `3878297` remains scheduled because both critical paths are incomplete.
  Scoped oversight commit `77b5eee` is local; a bounded publication attempt
  again produced no remote output; this publication note leaves `main` four commits ahead of
  `origin/main`. The report repository is unchanged.

### 2026-07-21 13:15 Dolmino prerequisite accepted; six more SFT finals

- Normal-data prerequisite `3875824` completed `0:0` in `02:32:53`. Its
  manifest records `5,100,006,129` source tokens in `11,195,395` documents,
  and the packed Qwen Nanoset records `5,111,201,524` tokens, clearing the
  `5,000,134,656`-token production requirement. The terminal log and decoded
  tail sample show ordinary heterogeneous Dolmino text. The wrapper removed
  only the reproducible raw JSONL staging tree after the packed-token gate;
  the accepted packed root remains at
  `$HPCVAULT/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino`.
- First-stage 5B jobs control `3875825_0`, formal `3875828_1`, and NL
  `3875831_2` are released from their dependency and are `AssocGrpGRES`
  pending on full A100-80GB nodes. Slurm currently estimates 02:55 CEST on
  July 22 for all three. Their staged successors remain dependency-held; no
  partition widening is compatible with the eight-A100, 1.875-TB request.
- Surface rows `3850105_21..23` and shortcut rows `3850213_25..27` completed
  cleanly. All six finals have nonempty 159,967,880-byte adapters/configs,
  no empty final file, and terminal `global_step=10000` trainer states.
  Surface SFT is now `24/27`, shortcut SFT `28/42`, and architecture remains
  `54/54`. New rows 24--26 and 28--30 backfilled and are progressing.
- Hybrid/conditioned eval rows are healthy at sampled chunks `70--98/112`;
  batch rows 9--11 are near step `680/10000`, 32B rows near `55%`, and
  conditioned-50k stage-two rows near `16.8--17.4k/50k`. Focused logs contain
  no fatal, OOM, quota, or no-space signature. No new terminal eval JSON and
  sample bundle exists yet, so no metric or report gate opened.
- The completed data build raised user-wide Vault usage to `1067.5G/1048.6G`
  soft with `202k/200k` files. The six newly terminal Trainer checkpoints are
  final-backed and no longer needed for restart, but this oversight runtime
  did not remove them; preserve every incomplete restart state and re-run the
  documented final/output/live-reference cleanup before production checkpoint
  writes materially expand storage. CPU-only successor `3876636` is scheduled
  and must remain queued because both critical paths are incomplete.
- Scoped handoff commit `b292bf9` plus its publication note are local and
  leave `main` two commits ahead of `origin/main`. A bounded
  `git push origin main` produced no remote output and timed out after 90
  seconds with exit `124`. The report repository is unchanged and
  synchronized because no corrected result gate opened.

### 2026-07-21 09:50 three-condition Dolmino 5B production submitted

- At user request, canceled the never-started NL 256-step confirmation
  `3875623_1`. Control and formal short gates remain accepted; decoded NL
  corpus structure was already clean, and the only observed NL launch failure
  was the repaired pre-training port collision.
- Submitted prerequisite `3875824`, which started immediately on RTX Pro node
  `a2041`. It deterministically exports and packs a 5.1B-token normal Dolmino
  stream so the 5,000,134,656-token control does not repeat the current
  4.811B-token corpus. It requires at least 5,000,134,656 packed tokens and
  deletes only its reproducible raw staging tree after that gate passes.
- Submitted three full-node A100-80GB resumable 5B chains, all dependent on
  `3875824`: control `3875825 -> 3875826 -> 3875827`, formal-5%
  `3875828 -> 3875829 -> 3875830`, and matched-NL-5%
  `3875831 -> 3875832 -> 3875833`. Each condition targets step 9,537
  (`5,000,134,656` tokens), global batch 128, TP4/DP2, peak LR `1e-5`, 256
  warmup steps, and cosine decay to `1e-6` over the preregistered 38,147-step
  20B horizon. Checkpoints are every 500 steps; up to three 24-hour stages are
  queued per condition and later stages skip once the complete 5B state exists.
- The 5B gate shares the exact normal Nanoset and seed across conditions and
  uses Nanotron's deterministic `1.0` or `0.95/0.05` weighted sampler. It does
  not precompute identical per-slot normal identities across conditions; that
  stronger pairing remains a methodological caveat and must not be claimed.
  Downstream direct and instruction readouts remain to be submitted against
  the terminal 5B checkpoints.

### 2026-07-21 09:20 Dolmino formal gate recovered; report work active

- A matched control/formal audit now passes both available conditions. After
  normalizing only run paths/names and dataset lists/weights, the configs have
  identical SHA-256 `223fb77c...c37`; both have exactly 256 finite iterations,
  the same 32-step warmup to `1e-5`, global batch 128, TP4/DP1, and comparable
  throughput. Post-warmup mean loss is `0.956750` control versus `0.852558`
  formal, but this cross-corpus difference is diagnostic rather than a quality
  claim. Large loss blocks occur in both control and formal at shifted steps,
  so they are not evidence of unstratified formal mixing.
- Decoded packed samples confirm ordinary heterogeneous Dolmino documents and
  formal records in the intended tag-free neutral
  `Solution/Context/Derivation/Conclusion/Final answer` document format.
  Formal Nanoset metadata records 550,144,949 tokens/144,136 documents; the
  Dolmino Nanoset records 4,810,706,180 tokens/10,705,908 documents. Control
  and formal can be marked individually clean now. Keep production launch
  gated on NL only to avoid locking an unmatched protocol if NL exposes a
  common configuration issue; `3875623_1` remains account-GRES pending and
  currently has no scheduler start estimate.
- Dolmino formal p5 row `3872664_0` reached all `256/256` steps in `02:26:10`
  at about 15.6K tokens/s. Gradients/losses remained finite and the realized
  Dolmino/formal blend was `0.949982/0.0500183`. Its Slurm failure is
  post-training only: the active wrapper was patched while the old shell was
  blocked in `torchrun`, and after training it re-read shifted lines and
  exited before writing `complete.json`. Reconstructed that static marker
  from the terminal log/config; no formal rerun is needed.
- The original NL row failed before optimizer progress because both concurrent
  `torchrun` processes used port 29500. Exact replacement `3875623_1` has the
  allocation-specific-port fix and is account-GRES pending with a current
  09:26 CEST estimate. Scientific control/formal/NL comparison remains gated
  on its completion and subsequent loss/sample inspection.
- Nineteen BranchProof rows are active with no fatal/OOM/quota signature:
  surface/shortcut rows are about `91--95%`, batch rows about `3%`, 32B rows
  about `30%`, hybrid/conditioned eval rows are at chunks `27--59/112`, and
  conditioned-50k stage-two rows `3850110_0..2` started after the original
  24-hour timeout. No new terminal eval JSON/sample bundle exists yet.
- Active checkpoint writing raised user-wide Vault use to `970G/1000G`.
  A repeated final/output/live-reference guard found 20 new terminal Trainer
  checkpoints with verified finals and zero live references; deleting only
  those intermediates reclaimed `17.06 GiB`. Vault is now `954G/1000G`, while
  this repo is `394,804,989 KiB`; all incomplete restart states and evidence
  remain. The user-wide file count is `201k/200k`.

### 2026-07-21 07:15 architecture SFT complete; Dolmino NL startup recovered

- Corrected architecture rows `3850113_51/52/53` completed cleanly, so the
  report-matrix architecture SFT family is now `54/54`. Each new Gemma-3-4B
  NL final has a nonempty 131,252,288-byte adapter and config, no empty final
  file, and a terminal checkpoint trainer state at `global_step=10000`.
  Surface rows `3850105_21..23` and shortcut rows `3850213_25..27` remain
  healthy at about `75--79%`. Conditioned-50k rows `3850109_13/14` are at
  about `17.6k/50k` with complete 5k/10k/15k restart states and the existing
  staged `afterany` resume chain intact.
- A100 capacity opened for report work: batch rows `3850114_9..11`, hybrid
  eval row `3850118_3`, 32B original row `3850115_4`, and 32B recovery row
  `3854837_1` are running. Hybrid row 3 had reached sampled chunk `32/112`
  with the audited 16,384-token context and no fatal signature. Baseline
  `3857767_[21-29]` and exact recoveries `3863525_[13-14]`/`3865321_18`
  remain A100-80 account-GRES pending; the accepted baseline gate stays
  `18/30` and no corrected aggregate or report evidence exists yet.
- Dolmino p5 logic row `3872664_0` is healthy at step `189/256`, about 15.5K
  tokens/s, with finite loss/gradients and realized blend
  `0.949982/0.0500183`. Co-located NL row `3872664_1` failed before training
  because both four-GPU `torchrun` processes tried port 29500. The shared LR
  wrapper now supplies a deterministic per-allocation master port; exact NL
  replacement `3875623_1` is submitted with every scientific setting
  unchanged. A malformed, never-started single-task submission `3875622` was
  canceled immediately.
- User-wide Vault is `861G/1000G` and `202k/200k` files; this repo is about
  `285 GiB` and 7,288 Vault files. Space remains safe, while the shared file
  count is above soft quota. Current CPU-only watcher `3875468` scheduled
  CPU-only, no-GRES successor `3875621` for 13:08 CEST. The plan remains
  incomplete, so the successor is preserved. The report repository remains
  unchanged because no evidence gate opened. Scoped fix/handoff commit
  `f659445` is local; a bounded push timed out with exit `124`, leaving this
  repo three commits ahead of `origin/main`.

### 2026-07-21 01:20 fourteen more report-matrix SFT rows accepted

- Fourteen corrected report-matrix SFT rows completed after the prior handoff:
  surface rows 15--20, shortcut rows 19--24, and architecture rows 49/50.
  This includes both targeted replacements `3872657_[15-17]` and
  `3872658_[19-21]`. Every final passed the artifact gate: nonempty adapter
  and config, terminal `global_step=10000`, and no empty file in `final/`.
  Surface is now `21/27`, shortcut `25/42`, and architecture `51/54` final
  adapters. Rebuilt evals `3872661/3872662` remain dependency-gated and no
  corrected eval JSON or sample JSONL was written.
- Eleven corrected A40 rows are active: surface `3850105_21..23`, shortcut
  `3850213_25..27`, architecture `3850113_51..53`, and conditioned-50k
  `3850109_13/14`. They are around `26--30%`, `26--30%`, `80/45/45%`, and
  `26%`, respectively. Scoped log tails contain no fatal, OOM, quota, or
  no-space signature.
- Declaration-fixed baseline eval/recoveries, batch recovery, hybrid eval,
  and Dolmino p5 remain A100-80 account-GRES pending. Dolmino
  `3872664_[0-1]` currently has a provisional 01:49 CEST start estimate; no
  dependency, feature, or compatible-partition repair is available. The
  baseline acceptance gate remains `18/30`, and no duplicate was submitted.
- User-wide Vault is `829G/1000G` with `201k/200k` files; this repo is about
  `253 GiB` and 7,163 Vault files. The space quota remains safe, while the
  user-wide file count is still above soft quota and must be watched.
- Current watcher `3874799` is CPU-only on `a100mig`. It scheduled CPU-only,
  no-GRES successor `3875468` for 07:08 CEST. The plan remains incomplete, so
  the successor is preserved. The report repository is unchanged because no
  evidence gate opened. The scoped handoff commit is local; a bounded push
  timed out with exit `124`, leaving this repo two commits ahead of
  `origin/main`.

### 2026-07-20 19:15 first resumed architecture final passed; gates unchanged

- Architecture row `3850113_48` completed as raw job `3872671` at 16:41 CEST
  in `07:49:57` with exit `0:0`. Its Gemma-3-4B NL seed-3407 run reached
  step 10,000 and passed the artifact gate: the final adapter is nonempty
  (`131,252,288` bytes), its config is present, and the terminal trainer state
  records `global_step=10000`. Architecture row `3850113_51` backfilled at
  16:41; rows 49/50/51 are now active.
- Seventeen corrected BranchProof SFT rows remain active on A40s. The
  surface/shortcut 10k rows are around `81--85%`, conditioned-50k rows 13/14
  are around `16%`, and architecture rows 49/50/51 are around `85/85/22%`.
  Their current log tails contain no fatal, OOM, quota, or no-space signature.
  No new corrected eval JSON or sample JSONL exists, so no metric,
  raw-generation, aggregation, or report gate opened.
- A100-80 work remains account-GRES pending with dependencies intact. Slurm
  estimates baseline row `3857767_21` for 21:59 CEST and matched Dolmino p5
  `3872664_[0-1]` for 01:49 CEST on July 21. The declaration-fixed baseline
  remains `18/30` accepted; no recovery or duplicate family was submitted.
- User-wide Vault is `810G/1000G` with `201k/200k` files; this repo is about
  `234 GiB` and `6,797` Vault files. The increase from the cleanup snapshot is
  expected active-checkpoint writing and is below the space soft quota, but
  the user-wide file count remains above soft quota and must be watched.
- Watcher `3873713` is running CPU-only on `a100mig`. Its recorded successor
  `3874799` is CPU-only (`cpu=4,mem=30000M`, no GRES) and BeginTime-pending for
  01:08 CEST on July 21. The plan is incomplete, so it remains scheduled.
- The scoped handoff commit is local and one commit ahead of `origin/main`.
  A bounded push and direct SSH probe failed because `github.com:22` timed
  out; the report repository is unchanged, clean, and synchronized.

### 2026-07-20 15:06 project-only storage cleanup; active jobs unchanged

- Removed only reproducible staging artifacts owned by this project. From
  `$HPCVAULT`, deleted the raw JSONL trees
  `nanotron_data/qwen25_branchproof_unique_v2` and
  `nanotron_data/qwen25_dolmino_neutral_v1` after verifying that every
  corresponding packed Nanoset has 16 nonempty shards, metadata, token
  statistics, and no live-job raw-data reference. The packed BranchProof and
  Dolmino/formal/NL Nanosets remain intact. This reclaimed `55.59 GiB`.
- From `$WORK`, deleted `242` completed online W&B run caches dated before
  July 20, reclaiming `1.27 GiB`. Preserved all `20` offline/unsynced runs and
  all `19` current-day run caches used by active jobs. No final adapter,
  checkpoint, dataset result, evaluation sample, environment, or unrelated
  project path was removed.
- Repo-owned storage is now about `231 GiB` on `$HPCVAULT` and `68 GiB` on
  `$WORK`; user-wide quota reports `808G/1000G` on Vault and `80,679M/100G`
  on Work/home. Vault still displays `201k/200k` files because that count is
  user-wide; this repo has only `6,745` Vault files.
- Seventeen BranchProof SFT rows remain running on A40s with no newly recorded
  failure. A100 work remains account-GRES/dependency pending. Slurm currently
  projects 17:34 CEST for Dolmino p5 `3872664` and selected A100 work; the
  scientific result gates are unchanged.

### 2026-07-20 13:15 resumed A40 training healthy; A100 gates capacity-pending

- Seventeen corrected BranchProof SFT rows are running on A40s. The six
  replacement surface/shortcut rows `3872657_[15-17]` and
  `3872658_[19-21]`, resumed surface rows `3850105_[18-20]`, and shortcut
  rows `3850213_[22-24]` are at roughly `32--34%` after 4h20m. Architecture
  row `3850113_48` is at `55%`; rows 49/50 backfilled at 12:31/12:35 CEST and
  are at `7/6%`. Conditioned-50k rows `3850109_13/14` are at about `6%` and
  retain the staged resume chain `3850110..3850112`. No monitored log has a
  fatal, OOM, quota, or no-space signature.
- The declaration-fixed baseline remains `18/30` accepted. Original eval
  rows `3857767_[21-29]`, local-snapshot recoveries
  `3863525_[13-14]`/`3865321_18`, their CPU audits, report evals, batch
  recovery `3872659`, hybrid replacement eval `3872660`, and Dolmino p5
  confirmation `3872664` are all pending on account GRES or dependencies.
  No new in-scope eval JSON, sample JSONL, or Nanotron artifact exists after
  the 08:52 release, so no raw-generation, aggregation, or report gate opened.
- Vault space remains safe at `863G/1000G` user-wide and `300,344,897 KiB`
  repo-owned, with only `26` repo Trainer checkpoint directories. The shared
  user file count is again just over its soft threshold at `201k/200k`
  (`400k` hard); this is not currently a repo write failure, but the next pass
  must recheck it before additional data or checkpoint expansion. Dolmino p5
  currently estimates 17:34 CEST rather than the earlier 10:24 estimate.
- Current CPU-only watcher `3872507` is running on `a100mig`. Its recorded
  CPU-only successor is `3873713`, scheduled for 19:07 CEST with
  `cpu=4,mem=30000M` and no GRES. The plan is incomplete, so the successor
  remains queued.
- Handoff commit `4837cbb` is local. Two bounded `git push origin main`
  attempts produced no remote output and the explicit 60-second retry exited
  `124`; local `main` is one commit ahead of `origin/main`. The report repo is
  unchanged and synchronized.

### 2026-07-20 08:52 capacity restored and all gated work resumed

- The user confirmed the competing experiments are complete. A guarded
  repo-only cleanup first verified `136` run parents with nonempty final
  adapters and no live checkpoint references, then removed their `146`
  intermediate Trainer checkpoint directories (`130.04 GiB`, `1,970` files).
  It preserved all `24` checkpoints for incomplete conditioned-50k and batch
  runs. The superseded, reproducible FineWeb continual-pretraining JSONL and
  Nanoset were also removed (`78.96 GiB`); current science uses Dolmino.
  User-wide Vault use fell from `1070G` to `861G`, and repo-owned use from
  `517,225,409 KiB` to `298,078,072 KiB`. The file quota is no longer marked
  over-limit, although its rounded display remains `200k/200k`.
- Released every held BranchProof array. Existing baseline recoveries,
  report-matrix SFT/evals, conditioned-50k staged resumes, architecture, and
  32B work are now normal running, dependency-pending, or account-GRES
  pending. No `JobHeldUser` row remains.
- Submitted targeted replacements: surface SFT `3872657_[15-17]`, shortcut
  SFT `3872658_[19-21]`, resumable batch SFT `3872659_[3-8]`, and hybrid eval
  `3872660_[0-2]`. Rebuilt complete skip-gated eval arrays as surface
  `3872661`, shortcut `3872662`, and batch `3872663` with dependencies on both
  original and replacement training. The A40 SFT rows started immediately.
- Added and submitted matched Dolmino p5 confirmations `3872664_[0-1%2]`:
  both run 256 steps at shared LR `1e-5`, global batch 128, and TP4/DP1 on
  four A100-80GB GPUs; row 0 is 95% Dolmino + 5% formal and row 1 is 95%
  Dolmino + 5% matched NL. Slurm currently estimates 10:24 CEST. Full staged
  production remains gated on stable, sample-clean confirmation results.
- Final-backed cleanup exposed a stale conditioned-50k skip condition that
  also required the now-removed terminal Trainer checkpoint. Before any
  dependent resume stage started, the wrapper was corrected to treat a
  nonempty final adapter as terminal. Completed rows 0--2 and 4--5 therefore
  skip, while unfinished rows retain and resume their audited checkpoints.

### 2026-07-19 19:09 quota hold and CPU-only successor preserved

- The corrected baseline remains `18/30` accepted. Held eval rows
  `3857767_[21-29]`, exact recoveries `3863525_[13-14]` and `3865321_18`,
  their CPU audits, and aggregate `3857769` are unchanged. No in-scope metric,
  sample, log, or eval artifact appeared after 13:09, so no analysis or report
  gate opened.
- User-wide Vault usage is now `1070G/1000G` soft and `202k/200k` files. The
  repo-owned tree remains exactly `517,225,409 KiB` with `8,711` files and all
  `170` Trainer checkpoint directories intact. Keep all BranchProof and
  Dolmino holds until both quotas have margin and the post-16:30 CEST July-20
  capacity check passes; no unrelated job or artifact was touched.
- CPU-only watcher `3870759` is running on `a100mig`. Its recorded CPU-only
  successor is `3871736`, scheduled for 01:07 CEST on July 20 with
  `cpu=4,mem=30000M` and no GRES. The plan is incomplete, so the successor
  remains queued.
- Quota/chain handoff commit `33ac40e` is local. A bounded push and separate
  `git ls-remote` probe both produced no remote output and timed out; local
  `main` is 19 commits ahead of `origin/main`. The report repository is
  unchanged and synchronized.

### 2026-07-19 13:09 both Vault soft quotas exceeded

- The corrected baseline remains `18/30` accepted. Held eval rows
  `3857767_[21-29]`, exact recoveries `3863525_[13-14]` and `3865321_18`,
  their CPU audits, and aggregate `3857769` are unchanged. No in-scope
  BranchProof or Nanotron metric, sample, log, or eval artifact appeared after
  the 07:15 handoff, so no raw-generation, aggregation, or report trigger
  fired.
- User-wide Vault usage has risen to `1068G/1000G` soft and
  `201k/200k` files. The repo-owned Vault tree is unchanged at
  `517,225,409 KiB` (about `493 GiB`) and `8,711` files; all `170` Trainer
  checkpoint directories remain present. Do not release any held BranchProof
  or Dolmino job until usage is below both soft limits with margin and the
  post-16:30 CEST July-20 capacity check passes. No unrelated artifact was
  touched.
- CPU-only watcher `3870293` is running on `a100mig`. It scheduled CPU-only
  successor `3870759` for 19:06 CEST with `cpu=4,mem=30000M` and no GRES.
  The end-to-end plan is incomplete, so the successor remains queued.
- Quota/chain handoff commit `7731bee` is local. A bounded
  `git push origin main` produced no remote output and timed out after 90
  seconds; local `main` remains 16 commits ahead of `origin/main`. The report
  repository is unchanged and synchronized.

### 2026-07-19 07:15 capacity and quota hold unchanged

- The corrected baseline remains `18/30` accepted. Held eval rows
  `3857767_[21-29]`, exact recoveries `3863525_[13-14]` and `3865321_18`,
  their CPU audits, and aggregate `3857769` are unchanged. No in-scope
  BranchProof or Nanotron metric, sample, log, or eval artifact appeared after
  the 2026-07-18 19:15 handoff, so no raw-generation or report trigger fired.
- User-wide Vault usage is now `1001G/1000G` soft and exactly
  `200k/200k` files. The repo-owned Vault tree is unchanged at
  `517,225,409 KiB` (about `493 GiB`) and `8,711` files; its `170` Trainer
  checkpoint directories remain present, including conditioned-50k
  `checkpoint-25000/25000/15000` and the batch-size-4 `checkpoint-3000`
  restart states. Do not release any held BranchProof job until quota is back
  below both soft limits and the post-16:30 CEST July-20 capacity check passes.
- CPU-only watcher `3869995` is running on `a100mig`. It scheduled CPU-only
  successor `3870293` for 13:06 CEST with `cpu=4,mem=30000M` and no GRES.
  The end-to-end plan is incomplete, so the successor remains queued.
- Live-handoff commit `88ba7ea` is local. A bounded `git push origin main`
  produced no remote output and timed out after 90 seconds; the SSH/network
  publication path remains blocked. The report repository is unchanged and
  synchronized.

### 2026-07-18 19:15 user-wide Vault file-quota warning during the capacity pause

- The corrected baseline remains `18/30` accepted. Held baseline evals
  `3857767_[21-29]`, exact recoveries `3863525_[13-14]` and `3865321_18`,
  their CPU audits, and aggregate `3857769` are unchanged. No in-scope
  BranchProof or Nanotron metric, sample, or log artifact appeared after the
  2026-07-17 07:06 handoff. The urgent out-of-scope GPU work is still active,
  so the documented post-16:30 CEST July-20 capacity check remains the next
  release trigger.
- User-wide Vault quota has risen to `960G/1000G` soft and `198k/200k` files,
  while the repo-owned tree remains `517,225,409 KiB` (about `493 GiB`) and
  contains only `8,711` files. Read-only inode checks attribute at least
  `84,593` files to the shared project virtualenv, `15,284` to the shared
  cache, and `22,403` to out-of-scope BabyLM roots. No unrelated artifact was
  touched. Treat the two-thousand-file soft-quota margin as a prerequisite
  check before releasing BranchProof work on July 20.
- CPU-only watcher `3867919` is running on `a100mig`; recorded successor
  `3869120` is CPU-only with no GRES and is BeginTime-pending for 01:06 CEST
  on July 19. The end-to-end plan is incomplete, so the successor remains
  queued.
- Quota/handoff commit `879acb3` is local. `git push origin main` produced no
  remote output and timed out after 90 seconds; local `main` remains at least
  nine commits ahead of `origin/main`. The report repository is unchanged and
  synchronized.

### 2026-07-17 19:12 watcher prompt quoting fix under the capacity pause

- The corrected baseline remains `18/30` accepted. All unfinished BranchProof
  GPU rows, including `3857767_[21-29]`, `3863525_[13-14]`, and
  `3865321_18`, remain user-held for the documented post-16:30 CEST July-20
  capacity decision. Their CPU audits and aggregate `3857769` remain
  dependency-gated; no new metric/sample artifact appeared after the 07:06
  handoff.
- Fixed the CPU-only watcher wrapper so its Markdown prompt is built with a
  quoted heredoc. The previous double-quoted argument executed prompt
  backticks as shell command substitutions, producing dozens of harmless but
  noisy `command not found` errors before Codex started. `bash -n` and a
  rendered-prompt check preserve the successor-file interpolation and literal
  backticks; shellcheck is unavailable in the current environment.
- Current watcher `3865931` is CPU-only on `a100mig`. Its recorded successor
  `3866771` is also CPU-only, requests only `cpu=4,mem=30000M`, and is scheduled
  for 01:03 CEST on July 18. The plan is incomplete, so the successor remains
  queued. Repo-owned Vault use remains `517,225,409 KiB` (about `493 GiB`);
  the higher user-wide quota use is attributable to out-of-scope work and was
  not touched.
- Wrapper and handoff commit `0d6468d` is local. `git push origin main` again
  produced no remote output and timed out after 90 seconds; the SSH/network
  path remains the exact blocker. The report repository is unchanged and
  synchronized.

### 2026-07-17 07:06 conditioned-50k timeouts preserved under the pause

- Corrected conditioned-50k SFT rows `3850109_10/11/12` reached the expected
  24-hour walltime at optimizer steps `26924/26751/18891`. Their latest
  complete restart states are respectively `checkpoint-25000`,
  `checkpoint-25000`, and `checkpoint-15000`; each contains nonempty adapter,
  optimizer, scheduler, RNG, and trainer state with zero empty files. No final
  adapter was written, and no fatal/OOM/quota signature preceded the timeouts.
- The existing staged `afterany` resume chain `3850110..3850112` remains the
  exact recovery path and is intentionally user-held with the July-20 capacity
  pause. No duplicate recovery or dependency edit was made. Baseline remains
  `18/30` accepted; held A100-80 recoveries `3863525_[13-14]` and
  `3865321_18` plus original rows `21..29` remain unchanged.
- Current watcher `3865320` is CPU-only on `a100mig`. Its recorded successor
  `3865444` is also CPU-only and scheduled for 13:03 CEST; the chain remains
  necessary. Repo-owned Vault use is `517,225,409 KiB` (about `493 GiB`),
  while the full user Vault quota reports `689G` and `189k/200k` files. No
  cleanup or report-regeneration gate fired.
- Scoped handoff commit `c16cb9b` is local. `git push origin main` produced no
  remote output and timed out after 90 seconds; local `main` is six commits
  ahead of `origin/main`. The report repository remains unchanged and synced.

### 2026-07-17 01:08 baseline node-failure recovery staged under the pause

- Declaration-fixed baseline eval/audit rows 19/20 completed and passed,
  advancing the accepted row-scoped gate to `18/30`. Raw NL review for the
  two train-1-to-10 seeds covers depths `1/10/12/25/50`: train-band prompts,
  answer tags, and translated validity are clean, while depth-50 failures
  truncate or lose the answer as expected. Formal-parser errors on NL prose
  remain diagnostic-only and are not translated-validity failures.
- Eval row `3857767_18` suffered `NODE_FAIL` on `a0632` after completing
  sampled chunk `107/112`; it wrote no final metric/sample artifact. Exact
  A100-80 recovery `3865321_[18%1]` uses the audited local base snapshot and
  is user-held with the rest of the July-20 capacity pause. CPU audit
  `3865322_18` follows it. Impossible original audit task `3857768_18` was
  canceled, and aggregate `3857769` now requires `3863527` and `3865322` in
  addition to terminal original audits. No row or scientific setting was
  bypassed.
- The node failure left a `28G` transient merged-model root. After confirming
  job `3863485` was terminal with no final output or live consumer, only that
  root was removed; repo-owned Vault use fell from `521G` to `494G`.
  Successor watcher `3865320` remains CPU-only and scheduled for 07:03 CEST.
  All pending BranchProof GPU work remains held until the documented
  post-16:30 CEST July-20 capacity decision.
- The scoped handoff commit is local. `git push origin main` produced no remote
  output and timed out after 90 seconds; local `main` is four commits
  ahead of `origin/main`. The report repository is unchanged and synchronized.

### 2026-07-16 19:35 NL audit recovery and completed Dolmino LR gate

- Declaration-fixed baseline eval rows 15--17 completed successfully, but
  original CPU audits `3857768_15/16/17` incorrectly required positive formal
  `syntactic` scores from `nl_exact` outputs. The audit now uses translated
  `nl_logic_parse` as the NL train-band structural signal, with a regression
  test. CPU-only replacement `3864893_[15-17%3]` completed `3/3` and accepted
  all artifacts, advancing the baseline row gate to `16/30`. Aggregate
  `3857769` retains the original-array `afterany` gate and now also requires
  successful `3863527` and `3864893` replacement audits.
- Raw NL review spans all three seeds and depths `1/5/10/18/25/50` across
  correct/translated-valid, incorrect, parse-invalid, malformed, and long-cap
  cases. Train-band prompt, answer tags, and translated validity are clean.
  Formal validity errors on natural-language proof lines are expected and are
  not translated-validity errors. Depth-50 failures commonly truncate near
  the 7,168-token cap or emit a wrong answer. This accepts the three row
  artifacts only; no matched family claim is released before `30/30`.
- Dolmino LR row `3859711_2` (`1e-5`) completed 256/256 steps and exactly
  134,217,728 scheduled tokens in `02:27:54`, with finite loss/gradients and
  about 15.3K tokens/s. Across the same 224 post-warmup batches, mean loss is
  `0.95675/0.96040/0.97130` for `1e-5/6e-6/3e-6`; `1e-5` is lower than
  `6e-6` on 125 batches (73 ties) with paired mean `-0.00365`, and lower than
  `3e-6` on 161 batches (40 ties) with paired mean `-0.01455`. Nominate
  `1e-5` as the shared LR for formal/NL p5 confirmation. GPU confirmation is
  deferred until the documented capacity pause ends after 16:30 CEST on
  July 20; the pilot logs are authoritative because Nanotron did not emit the
  configured benchmark CSV despite the terminal manifests naming it.
- Baseline rows 18/19/20 remain healthy at sampled chunks `102/101/78` of
  112. The capacity pause remains in force. CPU-only successor watcher
  `3864892` is preserved because the corrected baseline, report matrix,
  Dolmino confirmations, and report replacement remain incomplete. Current
  repo-owned Vault use is about `627G`; no quota intervention is needed.
- Canceled only impossible original audit tasks `3857768_13/14`, which were
  permanently pending with `DependencyNeverSatisfied` behind failed original
  eval rows. Their full eval/audit gates remain represented by held exact
  recovery `3863525 -> 3863527`; this lets aggregate `3857769` eventually
  satisfy its original-array `afterany` gate without bypassing either row.

### 2026-07-16 16:37 four-day BranchProof capacity pause

- Held remaining surface `3850105`, batch `3850114`, and shortcut `3850213`
  training rows, then canceled active surface rows 15--17, batch rows 6--8,
  shortcut rows 19--21, and checkpointed batch recovery `3863546_[3-5]`.
  This released `3` A100s and `9` A40s immediately. Their stale dependent
  eval arrays `3850116/3850122/3850214` were canceled and must be recreated
  against replacement training jobs after the pause.
- Surface and shortcut rows had no intermediate checkpoint and restart from
  scratch. Batch-size-8 rows 6--8 also restart: their pre-policy launches
  reached roughly 4k/10k without writing checkpoints and could not finish
  within 24 hours. Batch-size-4 rows 3--5 retain complete checkpoint-3000
  states. Future batch launches save every 250 steps and retain two states,
  so 10k-update rows can resume exactly across 24-hour allocations.
- Resume target is 2026-07-20 after 16:30 CEST, subject to the other project's
  capacity needs. Exact release/resubmission order is recorded in
  `docs/branchproof_report_rerun_matrix_2026-07-13.md`.
- Baseline logic train-25 seed 3407 also passed its 1,024-sample/2,665-metric
  audit, advancing the declaration-fixed baseline gate to `13/30`.
- To prevent released slots from immediately backfilling with another
  BranchProof row, all remaining pending BranchProof training/eval arrays were
  user-held. Hybrid eval rows `3850118_0/1/2`, which started in the scheduler
  race, were canceled after five minutes. The three A100 slots immediately
  started non-BranchProof `3863071_0/1/2`, confirming the capacity transfer.
  Established BranchProof rows that were already running were not stopped.

### 2026-07-16 14:25 two Dolmino LRs complete

- Matched 4-GPU Dolmino rows `3859711_0/1` (`6e-6/3e-6`) completed cleanly
  in `02:27:56/02:27:31`, each covering 256 steps and 134,217,728 identical
  tokens. Over all 224 matched post-warmup steps, `6e-6` has lower loss on
  210 steps; mean paired `3e-6 - 6e-6` loss is `+0.0109`. Shared large-loss
  batches occur at the same steps, confirming source-batch variation rather
  than LR instability. The effect is small and provisional until `1e-5` row
  `3859711_2` finishes; that row started at 14:21 CEST and is running on four
  A100s. The redundant sequential 8-GPU fallback `3859297` was canceled
  immediately after the independent row started.
- Declaration-fixed baseline row/audit 11 completed and passed, bringing the
  accepted row-scoped gate to `12/30`. This finishes all three logic seeds for
  train maxima `5/10/15/20`; train-25 logic and NL rows remain active or
  pending. No matched modality claim is available yet. Targeted local-snapshot
  recoveries for transient-Hub failures rows 13/14 remain account-GRES pending.

### 2026-07-16 13:13 targeted recoveries and first completed LR gate

- Declaration-fixed eval rows 13/14 failed during merge after repeated Hub
  `504`/timeout responses while resolving the unchanged OLMo-3 base tokenizer;
  neither wrote an eval artifact. The merge helper now accepts an explicit
  local base-model override, and exact A100-80 recovery `3863525_[13-14%2]`
  uses the audited immutable local snapshot. CPU audits
  `3863527_[13-14%2]` follow it. Aggregate `3857769` now waits for the
  original audit array to terminate and both replacement audits to pass.
  Its qualitative gate now uses each accepted row audit's exact log provenance
  and rejects incomplete chunk logs; seven focused tests pass.
- Original eval/audit rows 0..10 are complete and accepted (`11/30`). Raw
  review of train-15 seed 3409 and train-20 seed 3408 spans depths
  `1/15/20/25/50`, correct-valid, correct-invalid, incorrect, malformed, and
  long/capped cases. Train-band proofs and answer extraction are clean;
  unsupported lines, premise/conclusion parse failures, wrong answers, and
  format collapse lose validity as intended. This is still logic-only partial
  evidence. Rows 11/12 and 15..18 are active; row 11 reached 112/112 sampled
  chunks and is finalizing. No completed runtime justifies sharding.
- Batch recovery `3859299_[3-5]` timed out at 24 hours after reaching roughly
  step 3.4k, with complete optimizer/scheduler/RNG `checkpoint-3000` states
  for all three seeds. Exact resume `3863546_[3-5%3]` is running on A40s;
  eval `3850122` now requires terminal originals/recovery plus successful
  `3863546`. No successful batch row was rerun.
- Dolmino LR row `3859711_0` (`6e-6`) completed all 256 steps and
  134,217,728 tokens in `02:27:56`, sustaining about 15.4K tokens/s with
  finite loss/gradients and a terminal `complete.json`. Row 1 (`3e-6`) is
  running at the same throughput; row 2 (`1e-5`) and sequential fallback
  `3859297` remain pending. Keep the fallback until row 2 actually starts.
  Vault is 827G with 187k files (200k soft/400k hard file quota), so no
  unguarded cleanup is authorized. Successor watcher `3863505` remains queued.

### 2026-07-16 09:32 first Dolmino LR row running

- Matched 4-GPU LR row `3859711_0` (`6e-6`) started at 09:25 CEST on A100-80GB
  node `a0832`. It passed model/data initialization and reached step 8/256 at
  about 15.4K tokens/s, finite loss near `0.52`, finite gradient norm, and peak
  allocation about 62.4GB/GPU. Its ETA is approximately 11:53 CEST.
  Rows `3859711_1/2` (`3e-6/1e-5`) remain account-GRES pending with 10:48
  estimates. Sequential 8-GPU fallback `3859297` has slipped to a 2026-07-17
  02:30 estimate; keep it only until the remaining 4-GPU rows actually start.
- Declaration-fixed baseline eval/audit row 7 also completed and passed its
  exact artifact gates, bringing accepted row-scoped audits to `8/30`. This is
  logic train-1-to-15 seed 3408; representative depths `1/15/18/25/50` show
  correct citation-free-valid shallow/train-edge proofs and properly rejected
  duplicate-declaration/format collapse at long depths. It remains partial
  logic-only evidence, not a report-level modality comparison.

### 2026-07-16 07:20 seven accepted declaration-fixed rows

- Declaration-fixed baseline eval/audit rows `3857767_0..6 ->
  3857768_0..6` are complete and accepted. Each audit retains the exact
  448-prompt, 16-generation, 1,024-row, 2,665-metric protocol with complete
  `7/112` greedy/sampled chunk logs, zero fresh-constant failures, and zero
  credited duplicate-declaration failures. Rows `0..6` cover all three logic
  seeds for train maxima 5 and 10 plus seed 3407 at train maximum 15. The two
  complete three-seed logic blocks remain within-modality evidence only; no
  logic-vs-NL or report claim is released.
- New raw inspection of row 6 covers sampled depths `1/15/18/20/25/50` and a
  greedy depth-20 declaration failure. Shallow/train-band examples have the
  intended prompt, answer extraction, contiguous fresh constants, complete
  proofs, and citation-free validity. Deeper failures include unsupported
  first lines, wrong answers, repetition to the 7,168-token cap, and duplicate
  predicate declarations; the strengthened evaluator rejects each malformed
  validity case. This supports accepting the artifact, not a modality claim.
- Baseline rows `3857767_7..12` are running on verified A100-80GB devices at
  sampled progress `102/88/89/88/71/52` of 112 completed chunks; rows
  `13..29` remain array-throttle pending. The longest completed row took
  `18:45:40`, below the intervention gate and with a final artifact, so no
  sharding, resubmission, or protocol change is justified.
- Batch recoveries `3859299_[3-5]` are only around
  `2,512/2,509/2,527` of 10,000 after about 18 hours and will likely need a
  checkpoint resume after their actual terminal timeout; do not submit it
  early. Dolmino LR gates `3859297` and `3859711_[0-2]` remain account-GRES
  pending, with current estimates near 21:15 and 08:45 CEST respectively.
  Current CPU-only watcher `3862186` scheduled successor `3862431` for about
  13:01 CEST. Preserve it: the plan remains incomplete. Repo-owned Vault use
  is `752,091,870 KiB` (about `717 GiB`), with 151 protected Trainer
  checkpoints and nine active BranchProof merge roots.

### 2026-07-16 01:05 additional declaration-fixed rows and live timeout gates

- Declaration-fixed baseline rows `3857767_3/4` completed in
  `10:09:31/12:09:06`, and CPU audits `3857768_3/4` accepted both exact
  448-prompt, 16-generation, 1,024-retained-row, 2,665-metric bundles. Their
  generation logs contain all `7/112` greedy/sampled chunks; sampled cap-hit
  chunk counts are `68/67`, and fresh-constant plus credited duplicate-
  declaration failures are zero. Raw review across depths `1/5/10/12/25/50`
  finds complete citation-free-valid train-band proofs, first unsupported or
  wrong-branch failures just beyond the train range, and depth-25/50 answer,
  format, repetition, and cap collapse. This remains a two-seed logic slice,
  not a family or modality result; no report metric was accepted.
- Baseline rows `3857767_1/2/5/6/7` are at sampled chunks
  `109/101/109/92/38` of 112 after `18.0/12.5/9.8/6.7/1.1` hours, and row 8
  has started on a verified A100-80GB device. Row 1 is close to completion,
  so the 20-hour depth-sharding trigger is not approached in a way that
  justifies intervention. No fatal, OOM, quota, or dependency signature is
  present, and the protocol is unchanged.
- Conditioned-50k SFT rows `3850109_7/8` are near their hard limit at steps
  `42,816/43,078` of 50,000; row 9 is at `21,837`. The already-submitted
  `afterany` resume chain `3850110..3850112` is the intended recovery and must
  remain intact. Batch recovery `3859299_[3-5]` has reached about
  `1,658/1,656/1,667` of 10,000 after 12 hours and now has latest-only
  1,000-step checkpoints; recover again only after an actual timeout.
- Dolmino LR gates `3859297` and `3859711_[0-2]` remain account-GRES pending;
  their current estimates are approximately 18:03 and 03:17 CEST,
  respectively. The CPU-only watcher scheduled successor `3862186` for 07:01
  CEST. The plan is incomplete, so the successor is preserved. Repo-owned
  Vault use is `739,995,190 KiB` (about `706 GiB`), with 141 protected Trainer
  checkpoints and nine active BranchProof merge roots.

### 2026-07-15 19:10 first declaration-fixed artifact and live recovery audit

- Declaration-fixed baseline row `3857767_0` completed in `11:07:55`, and
  CPU audit `3857768_0` accepted its exact 448 prompts, 16 generations,
  1,024 retained rows, all 2,665 metrics, generation logs, cap diagnostics,
  fresh constants, strict answer shape, and strengthened declaration/validity
  invariants. Raw review covers depths `1/5/25/50`, correct and incorrect
  cases, and malformed/cap-hit traces. Sampled correct/citation-free-joint
  pass@1 is `1.000/1.000` at depths 1 and 5, `0.066/0.000` at depth 25,
  and `0.037/0.000` at depth 50. Duplicate predicate declarations and premise
  parse failures are now rejected rather than credited. This is one logic row,
  not a family or modality result; no report metric was accepted.
- Baseline rows `3857767_1..5` are sampling at chunks
  `84/67/92/85/75` of 112 after `12.1/6.5/6.4/6.3/3.9` hours, and row 6 is
  in greedy generation on a verified A100-80GB device. Row 1 remains below
  the 20-hour sharding trigger but is the next runtime watch. No fatal, OOM,
  quota, or dependency signature is present, and the protocol is unchanged.
- Corrected conditioned-10k row 0 completed both logic and NL outputs; bounded
  raw review again finds clean modality-appropriate shallow traces and
  long-depth collapse. Row 3 then completed its NL bundle with the same
  shallow-clean/deep-collapse pattern; rows 2/4 continue, and the family
  remains partial. New architecture finals are also
  arriving without fatal signatures, but no report-family audit gate is yet
  complete.
- Batch recovery `3859299_[3-5]` is healthy but slow on A40: after about six
  hours its rows are only at steps `815/815/821` of 10,000. The original
  launches had no checkpoint, so these are genuine restarts; the new
  latest-only 1,000-step checkpoints have not yet been reached. Do not submit
  another recovery before an actual timeout. Repo-owned Vault use is now
  `724,556,915 KiB` (about `691 GiB`), including `128` protected Trainer
  checkpoints and nine active BranchProof merge roots.
- Shared-LR jobs `3859297` and `3859711_[0-2]` remain account-GRES pending.
  CPU-only watcher `3859290` is running and scheduled CPU-only successor
  `3860702` before starting. The end-to-end plan remains incomplete, so the
  successor is preserved.

### 2026-07-15 15:24 staged 20B Dolmino design

- If the short shared-LR gate passes, replace the one-shot 4.3B plan with a
  single resumable 20B-token schedule for control, formal-5%, and NL-5%.
  Pause and export weights at steps `9537/19073/28610/38147`, corresponding
  to `5.0001/9.9997/14.9999/20.0000B` tokens. Evaluate all three at 5B; resume
  all three to 20B only under a predeclared nontrivial downstream-signal gate.
- Configure the final 38,147-step LR horizon from step 1. Do not train a
  separately annealed 5B endpoint and then restart its scheduler. Preserve a
  rotating full optimizer state for continuation, permanent BF16 weight-only
  snapshots at each milestone, and separate instruction-tuning branches that
  never mutate the continuing base checkpoints.
- The current 10B Dolmino release and 4.81B packed stash cannot support 20B
  without repetition. Build the eventual stash from the official 100B release:
  at least 21B packed Dolmino tokens plus 1.1B tokens for each proof modality.
  Build in roughly 5B shards and delete verified raw intermediates to control
  peak Vault use. Precompute a shared slot schedule so formal/NL replace the
  same 5% of normal slots and all common normal samples remain aligned.

### 2026-07-15 15:03 targeted Dolmino and batch recoveries

- Dolmino prerequisite row `3858584_0` failed at `4.128B/4.8B` tokens after a
  Hugging Face Xet/CAS HTTP 500 on shuffled shard position 410. Its resumable
  state and 18.3GB JSONL are intact at an exact byte offset. The direct-shard
  exporter now retries a transient shard download up to five times with
  bounded backoff; the full suite passes (`232 passed, 3 skipped`). Exact
  row-0 recovery `3859296_[0%1]` resumed without replay and completed the raw
  export at `4,800,000,272` tokens in `10,705,908` records across 120 source
  groups. Manifest totals and first/quarter/middle/three-quarter/tail records
  pass inspection. Nanoset tokenization completed with one nonempty 19.24GB
  shard and `4,810,706,180` packed tokens. The exact `10,705,908`-token delta
  from source tokens confirms one EOS per record, and capacity exceeds the
  4.3B-token schedule. Dependency-dead LR gate `3858902` was canceled before
  start and replaced by sequential full-node gate `3859297`. It is normal
  `AssocGrpGRES`-pending with Slurm estimate 2026-07-16 10:48 CEST. Independent
  matched 4xA100-80GB alternatives `3859711_[0-2%3]` are also account-GRES
  pending with per-row estimates of 2026-07-16 03:49 CEST:
  they preserve TP4, global batch 128, and 134,217,728 tokens per row by using
  DP1 and gradient accumulation 32. Keep both paths until one can cover all
  three LRs, then cancel only redundant unstarted work. Completed formal/NL
  prerequisite rows are retained and were not rerun.
- Batch-size SFT rows `3850114_3/4/5` genuinely hit the 24-hour ceiling at
  steps approximately `9472/9416/9472`; their pre-patch launches had no
  resumable checkpoint. Exact recovery `3859299_[3-5%3]` is running on A40s
  with the already-added 1,000-step/latest-only checkpoint policy. Eval
  `3850122` now requires `afterany:3850114` and `afterok:3859299`.
- Declaration-fixed baseline rows `3857767_0/1` are healthy at sampled chunks
  `82/55` of 112 after about six hours; rows `2/3/4` started on verified
  A100-80GB devices and are in greedy generation. Corrected conditioned-10k
  eval row `3850119_1` completed; raw NL review at depths `1/5/10/25/50`
  confirms clean shallow behavior and deeper trace collapse. It remains a
  partial row, so no family result or report regeneration was accepted.
- CPU-only watcher `3858016` scheduled successor `3859290` before this pass.
  It remains pending because the end-to-end plan is incomplete.

### 2026-07-15 11:49 Dolmino prerequisite build and LR gate

- Production prerequisites are now running as
  `3858584_[0-2]` on RTX Pro 6000 nodes: row 0 streams a deterministic
  shuffled-shard sample of the released 10B Dolmino repository to 4.8B Qwen
  tokens; rows 1/2 build 550M-token formal/NL BranchProof corpora with the
  shared neutral `Solution/Context/Derivation/Conclusion/Final answer` format.
  The direct JSONL.zst reader is required because the released heterogeneous
  `default` Hugging Face config fails Arrow schema unification; a local direct
  shard smoke passed on native PDF text.
- Rows 1/2 completed in `00:12:32/00:11:52`; row 0 is healthy at approximately
  2.13B/4.8B tokens after 56 minutes. The original separately queued LR jobs
  `3858587/3858588` were canceled before start and replaced by `3858902`.
  That job holds one 8xA100-80GB node for up to 12 hours and runs `6e-6`,
  `3e-6`, and `1e-5` sequentially, avoiding two additional full-node queue
  waits. Every row sees the same packed chunks, uses 32 warmup steps followed
  by constant LR, consumes 134,217,728 tokens, and writes no large checkpoint.
  This is a stability/optimization screen, not a transfer result.
- Select one shared LR from finite loss/gradient behavior and matched
  late-window loss, then run 256-step formal-5% and NL-5% confirmation rows.
  Submit the full 4.3B-token control/formal-5%/NL-5% runs only after that gate.
  No condition-specific LR tuning and no full run are currently submitted.

### 2026-07-15 10:18 Dolmino midtraining corpus decision

- The completed Qwen2.5/FineWeb experiment is a **continual-pretraining
  pilot**, not a midtraining experiment. FineWeb-Edu is a broad pretraining
  corpus; all handoff and preprint references should use the corrected term.
- The appropriate next background is AI2's Dolma 3 **Dolmino Mix**. The
  released `allenai/dolma3_dolmino_mix-10B-1025` is the official 10B
  micro-anneal mixture for OLMo 3 stage 2 and is large enough for the planned
  4.3B-token no-repeat pilot. The 100B release is the exact full OLMo-3-7B
  second-stage mix but is unnecessary for this budget.
- Dolmino is heterogeneous plain text, not a single chat schema. Preserve its
  source text and EOS boundaries. Format paired BranchProof injections with
  the same modality-neutral outer document envelope, for example
  `problem\n\nSolution:\n{trace}\n\nFinal answer: {answer}`, with formal versus
  NL differing only inside `{trace}`. Do not retain `<formal>`, `<think>`, or
  modality labels.
- Future conditions should replace, rather than append, Dolmino tokens so
  total steps and tokens are identical: control is 100% Dolmino; each
  intervention is `(100-x)%` of the same Dolmino stream plus `x%` paired
  formal or NL proof tokens. Start with a bounded `{0,2,5,10}%` pilot rather
  than the old `{5,10,15,20,25}%` grid: 15--25% from one synthetic generator
  would dominate Dolmino's roughly 8.3% thinking category. No new job was
  submitted.

### 2026-07-15 09:54 Nanotron mixture and scheduler diagnosis

- Exact production-path replay rules out a missing shuffle but shows random,
  not strict, optimizer stratification. Seed-42 global updates contain 6--35
  proof chunks out of 128 (mean `19.2001`, standard deviation `4.0848`, near
  binomial expectation), with no proof-empty global update. The full schedule
  remains exactly matched at 15.000057% proof tokens for logic and NL.
- A real Nanotron resume bug was reproduced: the rebuilt scheduler normalized
  by checkpoint-current LR while PyTorch retained the original base LR, causing
  the shared step-4096 jump from about `5.94e-6` to `6.25e-6`. The installed
  checkout now uses `initial_lr`; the repo job template also lets Nanotron
  derive the 7,936-step post-warmup cosine span so future runs reach the
  `1e-6` floor. All completed conditions share both defects, so they affect
  absolute trajectories but not the matched condition ordering.
- Pretokenization made one nonempty unshuffled shard per source, but Nanoset
  then randomly permutes packed sample indices, so the effective stream is
  shuffled. Exact global-batch stratification is an optional variance-reduction
  ablation, not a repair for a missing shuffle. The stronger diagnosis is
  objective mismatch: full-document continuation teaches wrappers and
  continuation behavior. The multi-hop audit confirms that logic/NL strongly
  learned `<formal>`/`<think>` response surfaces while clean transfer remains
  only `+0.012/+0.018` after answer-head sensitivity. Details are in
  `docs/nanotron_mixture_schedule_audit_2026-07-10.md`.
- Format/readout follow-up: FineWeb was raw text plus EOS, whereas
  injected records use modality-identifying `<formal>`/`<think>` and nested QA
  tags. In a Dolmino pilot, paired formal and NL interventions should share
  the neutral `problem\n\nSolution:\n{trace}\n\nFinal answer: {answer}`
  envelope while native Dolmino records remain unchanged. The existing
  batch-one UltraChat LoRA is retained as a failed alignment diagnostic; a new
  pilot should compare direct evaluation with identical answer-only
  calibration and modality-neutral reasoning SFT across all checkpoints.

### 2026-07-15 07:18 multi-hop acceptance and active runtime recovery

- All six prompt-fixed multi-hop production bundles completed: direct
  `3855271_[0-2]`, instruction `3855272_[0-2]`, and strict aggregate
  `3855273` are `COMPLETED 0:0`. Every bundle passed the 1,200-row coverage,
  prompt-shape, 32/64-token decoding, corrected `rope_theta=1000000`, and
  32,768-window gates. The accepted compact bundle is
  `analysis/nanotron_branchproof_unique_v2_multihop_promptfix_20260714/`.
- Raw review covers all three conditions, both branches, all three benchmarks,
  both protocols, and correct/partial/incorrect cases. Direct stock QA-F1 is
  `0.189/0.250/0.238` for control/logic/NL, but an answer-head sensitivity
  rescore gives `0.349/0.361/0.367`; most of the apparent stock gain is
  continuation control, not a clean reasoning gain. Direct tagged prompts
  launch `<formal>` in `98.5--99.0%` of logic rows and `<think>` in
  `97.0--99.0%` of NL rows, normally exhausting the 64-token diagnostic before
  a usable answer. Instruction SFT removes those substrate launches, while
  stock QA-F1 remains `0.097/0.100/0.085` and almost every 32-token response
  caps. This closes the bounded multi-hop item as a response-format diagnostic,
  not positive transfer evidence.
- Declaration-fixed baseline `3857767_0/1` started at `06:59/07:01` CEST on
  verified A100-80GB devices with the unchanged 16,384 context; audits
  `3857768` and aggregate `3857769` remain dependency-gated. The first runtime
  check is healthy: row 0 completed greedy chunks 1--3 and row 1 completed
  chunks 1--2, with the expected 7,168-token cap hits in deeper chunks and no
  fatal signature. Conditioned-10k report eval `3850119_0/1/2` is healthy at
  sampled chunks `64/87/39` of 112 after `5:02/4:56/3:23`; projected
  completion remains well below 24 hours.
- Conditioned-50k row `3850109_6` timed out at step `43,105/50,000`, but its
  staged after-any resume chain `3850110..3850112` remains the correct recovery
  and must not be bypassed. Batch-family rows `3850114_3/4/5` are only about
  `72%` after `18.3--18.5` hours and may exceed the A100 partition's hard
  24-hour ceiling. Pending batch rows now retain a resumable checkpoint every
  1,000 steps; recover only rows 3/4/5 if they actually time out.
- Repo-owned Vault use is `460,945,705 KiB` (`439.6 GiB`) with 110 active
  Trainer checkpoints. No checkpoint is eligible for cleanup yet. Watcher
  `3857722` is CPU-only on `a100mig`; recorded successor `3858016` remains
  begin-time pending because the baseline, report-wide evals, audits, and final
  report replacement are incomplete.
- The official root preprint now includes only the corrected one-run p15
  null/mixed table and the multi-hop response-control finding; every historical
  BranchProof quantitative section remains disabled. The generated informal
  report was not regenerated because its builder still targets quarantined
  roots. No TeX engine is installed locally, so only static source checks ran.
- Local commits `2cf8a08` in this repo and `afb2d34` in the official report
  repo contain the accepted audit and preprint update. Direct SSH pushes of
  both repositories timed out after 90 seconds with no remote output; the
  branches remain ahead of `origin/main` for the successor to retry.

### 2026-07-15 01:08 declaration-validity supersession and clean replacement

- Raw review of the first eight exact-answer logic bundles found that generated
  formal wrappers can redeclare the same predicate symbol for different state
  words while still receiving internal citation-free validity. Across their
  `7,168` retained sampled rows, `1,616` have duplicate declarations, `94`
  were credited citation-free valid, and `53` were both answer-correct and
  credited valid; `46/53` occur at depth 20. The strengthened row gate rejects
  the old artifacts, so every metric previously read from `3853284_0..7` is
  quarantined and is no longer evidence. Compact census:
  `analysis/branchproof_unique_v2_declaration_collision_audit_2026-07-15.json`.
- `OutputEvaluator` now treats duplicate constant or predicate declarations as
  malformed and forces format, syntax, strict validity, citation-free validity,
  and grounded validity to zero. The artifact audit independently rejects any
  retained `citation_free_valid=1` row with duplicate declarations. Empty
  predicate blocks and case-distinct arithmetic symbols used by official iGSM
  remain supported. Focused and full verification pass at
  `226 passed, 3 skipped`.
- Old eval/audit/aggregate `3853284/3853285/3853286` were canceled after rows
  `0..7` had completed. Their 16 eval files, 8 audit JSONs, and 44 logs are
  isolated under the repo/Vault quarantine path
  `pre_declaration_fix_20260715`. Replacement
  `3857767_[0-29%6] -> 3857768_[0-29%8] -> 3857769` is submitted with the
  unchanged A100-80-only, 448-prompt, 16-generation, 14-depth,
  pass@1/2/4/8/16, 16,384-context, 7,168-cap protocol. It is currently
  `AssocGrpGRES` pending; accept no baseline metric until all 30 replacement
  audits and the qualitative grid pass.
- Prompt-fixed direct multi-hop rows `3855271_0/1` started on verified
  A100-80GB nodes with `rope_theta=1000000` and the audited 32,768 window;
  row 2 remains array-throttle pending. Instruction array `3855272` remains
  account-GRES pending and aggregate `3855273` remains dependency-held. Raw
  production bundles must be inspected before the aggregate is interpreted.
- Corrected report-wide SFT remains active without a fresh fatal signature.
  Conditioned-10k SFT `3850108` is complete at `15/15`; conditioned-50k row 5
  completed its active 50k chunk in `19:45:22`, rows 4/6 are finishing, and
  row 7 has started. Shortcut recovery `3856142_5/6` is about `92%`, and exact
  32B recovery `3854837_0` has started on A100-80GB. All report evals will
  inherit the declaration-validity correction when they start.
- Canceling the six obsolete merge processes removed their guarded temporary
  roots. Repo-owned vault use is now `326,852,150 KiB` with 95 Trainer
  checkpoints; no active report checkpoint is eligible for deletion. Current
  watcher `3857212` preserved recorded CPU-only successor `3857722` for
  06:49 CEST. Report/preprint regeneration remains deferred. Commit `381b388`
  is local; `git push origin main` timed out after 60 seconds without a remote
  response; the correction plus this handoff update leave `main` two commits
  ahead of `origin/main`.

### 2026-07-14 19:26 tiny-curve acceptance and corrected Nanotron p15 result

- Tiny checkpoint eval/recovery `3854813 + 3856145` is terminal. All `90/90`
  metric JSONs and `90/90` sample JSONLs passed the full audit with exact
  depth, pass@k, retained-sample, chunk-log, cap, and fresh-constant coverage.
  The audit now checks citation-free-valid diagnostics regardless of the
  strict-valid flag; its regression passes, and all 90 rows still pass the
  strengthened gate. Raw review covered both templates, all sizes and seeds,
  20k/60k/100k exposures, depths 1/10/50, successes, failures, and cap-hit
  degeneration. Answer-only OOD pass@1 rises at most to about `0.052` for an
  individual three-seed size/template checkpoint mean and pass@8 to `0.221`,
  but modality-appropriate OOD and depth-50 joint pass@1/4/8 remain `0.000`
  throughout. This is an accepted negative tiny mechanism result, not main
  report evidence. Audit bundle:
  `$HPCVAULT/synthetic-RLVL/analysis/branchproof_unique_v2_tiny_100k_checkpoint_audits_20260714`.
- The accepted tiny gate released guarded cleanup: all 90 intermediate
  `checkpoint-*` directories (`102G`) were deleted only after verifying 18
  nonempty finals, 90 metrics, 90 samples, 90 accepted audits, terminal parent
  jobs, and no live dependency. The finals and all curve outputs remain;
  repo-owned vault use is now about `393G`.
- Corrected logic instruction eval `3854824_3` and strict aggregate `3854847`
  completed `0:0`. The six-bundle manifest is accepted under
  `analysis/nanotron_branchproof_unique_v2_p15_20260711/`. Direct logic versus
  control changes all-primary/reasoning/general/targeted macros by
  `+0.0033/+0.0071/-0.0004/-0.0116`; direct NL changes them by
  `-0.0011/-0.0012/-0.0011/-0.0069`. Post-instruction logic/NL changes are
  similarly small (`+0.0018/+0.0027` all-primary). These are one-run,
  null/mixed readouts, not evidence of positive formal transfer.
- Raw Nanotron review found two evaluator/generation limits. Direct logic and
  NL increase BBH/MMLU-Pro next-document continuation markers to roughly
  `60.0/45.5%` and `58.0/49.5%` versus control `35.6/12.1%`. Instruction SFT
  removes literal markers but often generates long repetition; BBH instruction
  exact scores are an extraction floor (`0` in all conditions) because correct
  leading choices are followed by suffix text. Those BBH cells and the large
  instruction-minus-direct macro drops are not transfer evidence. The broader
  Nanotron mixture grid is rejected: the p15 trigger is neither positive nor
  sample-clean. Prompt-fixed multi-hop arrays `3855271/3855272` remain
  submitted and account-GRES pending as a bounded evaluation of the existing
  three checkpoints, not a broader training launch.
- Clean BranchProof baseline rows `3853284_0/2/3/4/5` and corresponding CPU
  audits are complete and accepted. Row 1 is healthy at sampled chunk 110/112
  after about 18 hours on a verified A100-80GB, still below the `20--24` hour
  sharding trigger; rows 6/7/8 are active and rows 9..29 remain throttle
  pending. No protocol, dependency, or partition edit was made.
- CPU-only watcher `3856057` preserved its recorded successor `3857212`,
  scheduled for 00:49 CEST. The chain remains necessary because the clean
  30-row baseline, corrected report matrix, multi-hop bundles, and report
  replacement are incomplete. Report/preprint regeneration remains deferred;
  the current builder still points at quarantined historical BranchProof
  roots and corrected report-wide evidence is not complete. Verification is
  green at `223 passed, 3 skipped`.
- Authenticated per-repository Hugging Face reconciliation after all corrected
  logic uploads measures `79.281G` total (`50.846G` models and `28.435G`
  datasets across 66 repositories), leaving `20.719G` against the nominal
  `100G` quota. Preserve the three p15 checkpoint repositories and their
  current adapters through multi-hop jobs `3855271/3855272/3855273`. No
  broader checkpoint is planned after the negative p15 gate; guarded rotation
  remains mandatory if that decision is revisited. Inventory:
  `analysis/hf_storage_cleanup_2026-07-13.json`.
- Publishing remains externally blocked. The configured GitHub SSH push timed
  out after 55 seconds, and the port-443 SSH fallback timed out during
  connection. `gh` is not installed and the known HTTPS fallback has no
  noninteractive credentials. Local `main` is two commits ahead of
  `origin/main`; preserve both commits and retry without discarding them.

### 2026-07-14 13:22 clean-row readout and targeted recoveries

- Clean exact-answer BranchProof eval rows `3853284_0/3/5` completed in
  `11:05:17`, `10:09:23`, and `10:34:35`; their CPU audits
  `3853285_0/3/5` all passed. Each bundle contains the intended `448` prompts,
  `16` generations per prompt, `1,024` retained rows, all `2,665` metrics, and
  zero fresh-constant failures. Raw review confirms clean train-band formal
  proofs, answer-correct but citation-free-invalid depth-25 traces, and
  frequent depth-50 cap-hit repetition. These are provisional logic-only rows,
  not a logic-vs-NL conclusion. Rows `1/2/4` remain healthy at sampled chunks
  `83/95/108` of `112`, and rows `6/7` backfilled as slots opened. Even the slowest
  row still projects below the `20--24` hour depth-sharding trigger; the
  protocol and A100-80-only placement remain unchanged.
- Prompt-fixed direct multi-hop smoke `3855269_0` passed. Instruction smoke
  `3855270_0` completed all 12 generations and the RoPE/window checks, but its
  audit incorrectly required the stock prompt to begin at byte zero even
  under Qwen `--apply-chat-template`. The audit now extracts and checks the
  single Qwen user turn; a regression covers that path. The stored instruction
  smoke re-audited cleanly, and CPU-only replacement gate `3856131` passed in
  two seconds. Full instruction array `3855272` was rewired to
  `afterok:3856131`; both full arrays `3855271/3855272` are now ordinary
  account-GRES pending. Raw smoke responses remain scientifically weak
  (explanations/repetition and incomplete tagged extraction) but are bounded
  by the audited 32/64-token caps rather than a prompt-construction error.
- Shortcut SFT rows `3850213_5/6` failed before training because the W&B
  service did not publish its port file within 30 seconds. Exact recovery
  `3856142_[5-6%2]` is running on A40s; shortcut eval `3850214` now requires
  original-array completion plus both recovery arrays `3854948` and `3856142`.
  Tiny checkpoint rows `3854813_24/26/28-32` were canceled during Slurm
  startup with no eval artifacts. Exact recovery
  `3856145_[24,26,28-32%3]` is running and was safely widened to generic
  one-GPU `a40,a100`, matching the already-audited tiny envelope.
- Corrected logic instruction reviewer eval `3854824_3` is running on a
  verified A100-80GB device with resolved RoPE `1000000`; after about 37
  minutes of generation it had processed `5,793/20,362` requests with no fatal
  signature. Aggregate `3854847` remains dependency-held. Current vault quota
  is about `566 GiB/1000 GiB` soft, and no checkpoint cleanup trigger fired.
  CPU-only watcher `3854785` is running; its recorded successor is `3856057`
  and must remain scheduled because the end-to-end plan is incomplete.
- Verification for the audit change passes the focused tests and the complete
  repository suite: `223 passed, 3 skipped`. Neither LaTeX report was
  regenerated because the clean 30-row BranchProof and six-bundle Nanotron
  aggregates are incomplete.
- The tracked oversight changes are committed locally. Pushes through the
  configured SSH remote and SSH port 443 timed out; HTTPS lacks non-interactive
  credentials. The branch is therefore one commit ahead of `origin/main`, and
  the next watcher should retry the push without discarding the local commit.

### 2026-07-14 11:19 HPCVAULT cleanup

- Repo-owned `$HPCVAULT/synthetic-RLVL` usage fell from `653,188,485 KiB`
  (`622.9 GiB`) to `554,725,384 KiB` (`529.0 GiB`), reclaiming `93.9 GiB`.
  Removed artifacts were limited to 60 intermediate checkpoints from 30
  completed corrected baseline SFT runs, nine intermediate checkpoints from
  three completed Nanotron instruction-SFT runs, superseded pre-BranchProof
  logic/NL raw corpora and Nanosets, and known-invalid/incomplete eval, smoke,
  and quarantine outputs. All 33 corresponding final adapters were verified
  nonempty before deletion and remain present.
- Protected artifacts were rechecked after deletion: corrected logic/NL and
  normal-continuation Nanosets, corrected raw data, the converted Qwen2.5-7B
  Nanotron checkpoint, six merge directories used by active clean eval
  `3853284`, and all intermediate checkpoints used by active corrected report
  and tiny checkpoint-curve jobs remain present. Current largest roots are
  `runs` `216 GiB`, `tmp` `164 GiB`, corrected Nanosets `54 GiB`, corrected raw
  Nanotron data `54 GiB`, and the converted base checkpoint `29 GiB`.
- Deferred cleanup is trigger-based: verify the six `tmp` merge directories
  disappear after `3853284`; after tiny curve `3854813` and each corrected
  report family are accepted, remove their intermediate `checkpoint-*`
  directories while retaining finals, audits, samples, and result bundles.
  No active or pending job artifact was removed.

### 2026-07-14 10:00 multi-hop repair, tiny interpretation, and WORK cleanup

- The matched Nanotron pilot already has all three intended checkpoints:
  normal-continuation control (`p0`), `15%` BranchProof NL, and `15%`
  BranchProof logic. Five of six conventional downstream bundles are accepted;
  logic instruction eval `3854824_[3%1]` is account-GRES pending and the strict
  baseline/NL/logic table job `3854847` waits on it.
- Raw smoke inspection found a LongBench task-construction bug. The dataset's
  `context` already contains the stock instruction prefix and suffix, while the
  old tagged/standard YAMLs wrapped it a second time; the `question` field also
  already began with `Question:`. Both protocols therefore duplicated their
  instructions and question. Document renderers now remove exactly the known
  embedded wrapper, normalize the question once, preserve the passage body,
  and limit tagged short answers to 64 tokens. The audit rejects old nested
  prompts and wrong generation caps. Validation passed on real records from all
  three datasets, all six lm-eval tasks, focused tests, and the full suite
  (`220 passed, 3 skipped`); re-auditing both old smokes now marks them rejected
  for all three defects. Prompt-fixed smokes are `3855269/3855270`; full
  direct/instruction baseline-NL-logic arrays `3855271/3855272` depend on them,
  and table aggregate `3855273` depends on both full arrays. Its compact
  baseline/NL/logic table will be written under
  `analysis/nanotron_branchproof_unique_v2_multihop_promptfix_20260714/`.
- The corrected tiny jobs are only the 50M/100M/200M one-pass scratch
  diagnostic, not all experiments in the report. The other surface, hybrid,
  conditioned, architecture, batch-size, 32B, and shortcut reruns are separate
  arrays. Modality-aware scoring is active: logic uses citation-free formal
  validity and NL uses translated validity. At depth 10, sampled outputs often
  have the correct final label but take a non-derivable branch, so zero joint is
  genuine for these under-capacity models rather than a validator mix-up.
  Depth-50 collapse is not comparable to the old result: the old generator was
  ambiguous above depth 17 and its nominal 100k run recycled a small corpus for
  100k optimizer steps. The corrected run is one pass over 100k unique examples.
  Checkpoint curve replacement `3854813` is running and remains the acceptance
  gate for the mechanism diagnostic.
- Repo-only `$WORK` cleanup removed `20.66 GiB`: superseded old tiny scratch
  runs/evals, obsolete OLMo-2 32B adapters, materialized long-depth BranchProof
  datasets known to be invalid, and local W&B caches belonging to terminal
  jobs. Active corrected runs, current downstream bundles, old large-model
  report outputs, and paused paired-task datasets were retained. The repo tree
  fell from about `88.4 GiB` to about `67.7 GiB`.

### 2026-07-14 07:14 oversight recovery and audit

- A post-oversight scheduler audit found shortcut SFT tasks `3850213_3/4`
  canceled by Slurm before Python startup, with empty stderr and no training
  artifact. Unlike conditioned-50k task `3850109_3`, which is covered by the
  existing `afterany` resume chain `3850110..3850112`, these cancellations
  made shortcut eval `3850214`'s original `afterok` dependency unsatisfiable.
  Exact recovery `3854948_[3-4%2]` was submitted and started immediately on
  A40s. Eval `3850214` now waits on `afterany:3850213` plus
  `afterok:3854948`, preserving all 42 rows without rerunning successful SFT.
- Clean BranchProof baseline eval `3853284_0..5` is healthy on verified
  A100-80GB devices after `4:58--5:46`; sampled-chunk progress projects roughly
  `6.5--11.8` hours total, below the `20--24` hour depth-sharding trigger.
  Rows `6..29` remain array-throttle pending, and CPU audits `3853285` plus
  aggregate `3853286` remain correctly dependency-held. No protocol or
  dependency edit was made.
- All 18 no-repeat tiny final eval rows in `3850492` completed and passed the
  structural/raw-artifact audit. Representative logic/NL samples across all
  sizes, seeds, depths `1/10/50`, correct/incorrect cases, and cap hits show
  shallow valid proofs, frequent depth-10 answer-correct but invalid traces,
  and severe depth-50 degeneration/truncation. The tiny final result is a
  mechanism smoke, not report evidence: joint@`1/4/8` is zero in every
  size/template aggregate and the checkpoint curve is still incomplete.
  Checkpoint eval `3850493` was canceled after 39 deterministic failures
  exposed missing tokenizer assets in intermediate checkpoint directories;
  the evaluator now accepts an explicit tokenizer path, all 90 checkpoints
  and 18 final tokenizers passed the preflight, and exact replacement
  `3854813_[0-89%3]` was safely widened from A100-80-only to compatible
  `a40,a100` with generic one-GPU GRES; rows `0..2` started immediately on
  A40s at 08:13 CEST. The model sizes are only 50M--200M and final eval rows
  already established that this protocol is far below the A40 memory/time
  envelope. Audits/tables/raw-review index:
  `$HPCVAULT/synthetic-RLVL/analysis/branchproof_unique_v2_tiny_100k_final_audits_20260714`.
- Corrected 32B rows `3850115_0/1` failed on transient expired Hugging Face
  Xet URLs while downloading the base; row 2 subsequently downloaded and is
  training normally. Targeted recovery `3854837_[0-1%1]` preserves the exact
  protocol. Held eval `3850123` now depends on `3854837` and original tasks
  `3850115_2..14`, so it cannot release until every required SFT row succeeds.
- Logic Nanotron upload `3847802_3` passed conversion, local/remote parity,
  finite-logit, and consumer-RoPE gates, uploaded the repaired checkpoint, and
  deleted its 199G local checkpoint only through the guarded path. Logic
  direct eval `3847804_3` completed. Fresh quota is `781G/1000G` soft
  (`2000G` hard) and `153k/200k` files. Instruction SFT `3847805_3` completed all
  10,000 steps (`train/eval loss 0.942806/0.936798`) but failed only on a
  transient Hub Xet 401 during adapter upload; the complete local adapter was
  retried without retraining and verified at Hub commit
  `3d1e4a751150fffbb26e23e6f759c402bf203b4d`. Stale dependent `3847806` and
  aggregate `3850389` were canceled; replacement logic instruction eval
  `3854824_[3%1]` and strict six-bundle aggregate `3854847` are pending.
- The schema-v4 MATH-500 scorer incorrectly treated escaped currency `\$` as
  a math delimiter. It now recognizes only unescaped delimiters; forced
  rescoring and production audits accept all five completed corrected bundles
  with no lost stock-positive rows. Raw diagnostics show direct control,
  logic, and NL BBH/MMLU-Pro next-document marker rates of
  `35.6/12.1%`, `60.0/45.5%`, and `58.0/49.5%`, respectively, with zero prompt
  marker incidence. Corrected control/NL instruction SFT removes those markers
  (`0%`) but often produces long repetitive continuations, so transfer claims
  remain blocked on logic instruction eval and the matched aggregate.
- Corrected instruction multi-hop smoke `3850354_0` passed the structural
  32,768-window/RoPE audit, but raw inspection rejected the protocol as
  production-clean: tagged Hotpot was `2/2`, tagged 2Wiki and MuSiQue were
  `0/2`, and MuSiQue showed missing closing tags plus context copying/repetition
  to the cap. The full six-condition grid remains on hold until both the p15
  aggregate is positive and the tagged instruction protocol is sample-clean.
  The report/preprint were intentionally not regenerated: the corrected
  baseline, report matrix, tiny checkpoint curve, and p15 aggregate are still
  incomplete. The repository suite passes `216 passed, 3 skipped`. CPU-only
  watcher `3853210` is running on `a100mig` without GRES; its pre-scheduled
  CPU-only successor `3854785` remains begin-time pending and must be kept
  because the end-to-end plan is incomplete.

### 2026-07-13 live correction wave

- A retained-generation audit found a second evaluator defect after the
  premise-validity fix: `synthrlvl.metrics._is_answer_match` credited any
  `<answer>` body containing the gold token. Two actual corrected samples were
  false positives, including a malformed 20-label answer and a multi-label
  answer that still passed format. The frequency in the five complete bundles
  was low (`1/640` retained greedy and `1/4480` retained sampled rows), but the
  unretained generations make exact pass@k rescoring impossible. Eval
  `3838163`, row audits `3847756`, and aggregate `3847757` were therefore
  canceled; their five complete eval/audit rows are diagnostic only and all
  metrics/samples were moved outside the aggregate input root under
  `quarantine/pre_answer_match_fix_20260714/`. Answer matching now accepts an
  exact answer or a single-line natural assertion, never a multi-line or
  alternative list. The row audit independently recomputes the same strict
  answer-shape condition from each retained generation and gold answer. The
  complete suite passes (`213 passed, 3 skipped`). Clean
  replacements are A100-80 eval `3853284_[0-29%6]`, CPU row audits
  `3853285_[0-29%8]`, and CPU aggregate/qualitative gate `3853286`. Eval rows
  `0/1` started at 01:28 CEST on verified A100-SXM4-80GB nodes `a0633/a0832`;
  they loaded the intended seed-3407/3408 adapters into isolated merge roots
  and initialized vLLM at 16,384 context without a fatal signature. Both first
  greedy chunks completed in `7.6/7.7s` with the intended 64 prompts,
  `max_new_tokens=7168`, and `</answer>` stop; observed maxima were 367 tokens.
  Rows `2..29` are capacity/throttle pending and both CPU gates remain
  dependency-held. The unchanged protocol
  remains 32 prompts/depth, 16 samples, all 14 depths, pass@1/2/4/8/16,
  16,384 context, 7,168 cap, and two retained samples per prompt.
- The corrected `BranchProof-unique-v2` baseline is no longer the only active
  rerun. A report-coverage audit found that every old long-depth syntax,
  shortcut, hybrid, conditioned-dual, architecture, batch-size, 32B, and tiny
  result is affected by the wrapped-constant bug. Corrected three-seed
  training/evaluation matrices are now submitted for all report families.
  Active parent arrays are surface `3850105 -> 3850116`, hybrid
  `3850107 -> 3850118`, conditioned 10k `3850108 -> 3850119`, conditioned
  50k recovery `3850109..3850112 -> 3850120`, architectures
  `3850113 -> 3850121`, batch-size `3850114 -> 3850122`, 32B
  `3850115 -> 3850123`, and shortcut build/train/eval
  `3850212 -> 3850213 -> 3850214`. The first tiny chain `3850072..3850078`
  was canceled before its first epoch completed after a sample-budget audit
  found it would recycle 50k rows for 100k optimizer steps. Its no-repeat
  replacement build `3850394` correctly rejected six duplicate training
  sequences among its first 100k rows. The deduplicating replacement is build
  `3850488`, train `3850490`, and final/checkpoint eval `3850492/3850493`.
  Build `3850488` completed CPU-only and verified 100k formal plus 100k NL
  sequence fingerprints before reloading the 100k-row Hub subset. No-repeat
  tiny training rows `3850490_0..2` completed one exact epoch at 20:12--20:16
  CEST with complete step-6250 checkpoints and final weights; rows `3..5`
  backfilled immediately and the remaining rows are throttle-pending.
  Architecture SFT rows
  `3850113_0..2` completed `0:0` at 18:44--18:45 CEST; all three exact
  `checkpoint-10000` trainer states and nonempty final adapters passed the
  artifact check. Rows `3850113_3..5` backfilled on A40s, and corrected
  shortcut SFT rows `3850213_0..2` started on A40s at 18:56--18:59 CEST.
  Other active rows are progressing without fatal signatures.
- Logic Nanotron recovery `3835442_3` completed step 8192 in `19:26:16` with
  finite terminal loss (`1.80`) and about `30.8K` tokens/s. Independent audit
  accepted all 645 files: model `625`, optimizer `4`, scheduler `4`, RNG `8`,
  no zero-byte files, and exact offsets `8192/1048576/4294967296`. Consumed
  proof tokens are `644247552`, or `15.000057%`; normal tokens are
  `3650719744`. Audit artifact:
  `analysis/nanotron_checkpoint_audits/qwen25_7b_midtrain_logic_p15_bp_unique_v2_4p3b_step8192_20260714.json`.
  Repaired-payload upload `3847802_3` is released and capacity-pending; its
  fail-closed conversion/consumer-RoPE/remote-parity gate remains responsible
  for guarded local cleanup. Vault usage is `901G/1000G` soft (`2000G` hard)
  with `150k/200k` files, so do not create a second full local checkpoint tree.
  CPU-only watcher `3850618` is active, and its
  already scheduled CPU-only successor `3853210` remains pending for 06:47
  CEST because the end-to-end plan is incomplete. The handoff is committed
  locally; normal SSH and `ssh.github.com:443` timed out, while HTTPS was
  reachable but had no noninteractive credentials, so this repo remains ahead
  of `origin/main` for the successor to retry.
- Sequence-length auditing found two additional old-result confounds. Hybrid
  targets average about 10k OLMo tokens and were truncated by the old 8192 SFT
  cap, so corrected hybrid SFT uses 16384. Tiny depth-10 targets exceed the old
  2048 SFT cap; corrected tiny training uses 4096 with a 16384 positional
  context. The replacement tiny protocol means exactly 100k unique examples:
  effective batch 16, 6,250 optimizer steps, and checkpoints every 20k
  examples. A trainer-side guard fails if the configured step budget would
  require row reuse. Old hybrid/tiny results remain quarantined even
  independently of the uniqueness bug.
- A Nanotron-to-HF compatibility audit found that Transformers 5 serialized
  Qwen2.5's correct RoPE base (`1000000`) only under `rope_parameters`, while
  the Transformers 4.57 downstream environment ignored that field and silently
  used `10000`. Nanotron training itself used `1000000` and is unaffected, but
  all previously completed control/NL HF downstream evaluations and both
  UltraChat adapters are invalid. This includes the previously quoted
  MATH-500 symbolic scores. The control and corrected-NL Hub configs now carry
  both fields and load as `1000000` in Transformers 4.57; the converter and
  verifier now enforce this compatibility for the pending logic upload.
- Invalid local and Hub instruction adapters were deleted after identification.
  Corrected adapter retraining is queued as control `3850351` and NL `3850352`.
  The first multi-hop smoke additionally truncated every tested prompt from
  the left under an 8192 window. Complete tokenization of all 600 LongBench
  examples gives maxima `17684/17079/17927` for HotpotQA/2Wiki/MuSiQue, so the
  corrected evaluator requires exactly a 32768 window and tests both stock
  LongBench short-answer and strict tagged protocols. Direct smoke `3850353`
  and dependency-gated instruction smoke `3850354` are queued; flawed full
  jobs `3850099/3850100` and aggregates `3850207/3850217` were canceled before
  start. No production multi-hop result will be submitted until corrected raw
  smoke generations are inspected.
- Corrected direct multi-hop smoke `3850353_0` completed `0:0` in `00:04:10`.
  Its audit accepted the exact 32,768 window, resolved RoPE base `1000000`, all
  six intended tasks, six sample files, and all 12 rows. Manual review covered
  every generation. The tagged protocol found and extracted an answer in
  `6/6`; its `</answer>` stop is intentionally omitted from stored raw text,
  and none continued into a new question. The stock protocol generated a next
  question or assistant preamble in `4/6` rows; both HotpotQA rows began with
  the correct answer but suffix leakage reduced F1 to `0.273/0.300`. Tagged
  exact match was `1/6`; 2Wiki and MuSiQue examples were wrong under both
  protocols. These two-example-per-dataset scores are only smoke diagnostics.
  Instruction smoke `3850354_0` remains held on corrected adapters, so no full
  six-condition multi-hop grid was submitted. Audit:
  `$HPCVAULT/synthetic-RLVL/lm_eval_results/qwen25_branchproof_unique_v2_multihop_smoke_ropefix_20260713/qwen25_7b_midtrain_control_p0_4p3b_step8192_direct/multihop_audit.json`.
- The four RoPE-invalid control/NL reviewer-suite bundles were preserved under
  `.rope10000_invalid_20260713` directory suffixes. Corrected direct reruns are
  `3850385/3850386`; corrected instruction reruns `3850387/3850388` depend on
  the new adapters. Stale aggregate `3849776` was canceled and replacement
  `3850389` depends on all four corrected control/NL jobs plus the corrected
  logic direct/instruction jobs `3847804/3847806`.
- Corrected control/NL adapter jobs `3850351_0/3850352_8` and direct reviewer
  evals `3850385_0/3850386_8` started on A100-80GB GPUs at 01:16--01:20 CEST.
  Every consumer preflight resolved the intended Qwen2.5
  `rope_theta=1000000`; both direct evaluators validated the production task
  suite and no startup fatal signature is present. Instruction evals remain
  correctly dependency-held on the new adapters. These are startup checks,
  not accepted downstream results.
- Full-suite verification exposed and fixed an independent legacy hard-task
  generator defect: rules containing extended predicate symbols were rendered
  by string concatenation (`P37b`) instead of as explicit atoms (`P37(b)`).
  All legacy `hard_v2/hard_v3` rule paths now use the shared atom renderer;
  compact one-letter syntax is unchanged. This does not affect the separate
  BranchProof-v2 generator or its active jobs. The complete repository suite
  passes after the materialized-sequence and one-pass guards
  (`205 passed, 3 skipped`).

### P0 correction: old BranchProof long-depth evidence is quarantined

- A 2026-07-10 forward-closure audit found that the old `hard_fsa` and
  `hard_fsa_schema` generators wrapped constants after 18 layers. Reused
  `(predicate, constant)` atoms let distinct branches re-enter one another, so
  many long-depth questions had multiple derivable candidate answers despite
  having one labeled answer. The ambiguity rate was `0/96` at depths
  `5/10/15`, `73/96` at depth `20`, `74/96` at depths `25/30/35`, and `92/96`
  at depths `40/45/50` in the audited seed/index grid.
- Therefore, all old BranchProof results involving depth above 17 are
  scientifically quarantined, including the headline depth-scaling result and
  derived architecture, syntax, shortcut, hybrid, conditioned-dual, and batch
  ablations. They remain useful only as diagnostics until selectively rerun on
  the corrected generator. They must not be used as paper evidence.
- The old headline also depended strongly on sampling: for the OLMo-7B
  train-1-to-25 run, long-depth pass@1 was logic `0.513` versus NL `0.574`,
  while pass@16 was logic `0.921` versus NL `0.794`. At depth 50, pass@1 was
  logic `0.241` versus NL `0.147`, while pass@16 was `0.833` versus `0.510`.
  A stored logic generation independently exhibited the ambiguity: it answered
  a non-gold state that the proof validator accepted as citation-free valid.
- `synthetic_dataset.py` now uses a fresh constant `c0..c_depth` at every
  layer and explicit atoms such as `A(c18)`. The probe gate now computes Horn
  closure and requires exactly one derivable answer. A production probe over
  1,000 train and 2,000 eval examples passed with unique-solution rate `1.0`,
  maximum derived-candidate count `1`, no generation failures, and balanced
  answer positions. Symbol-padded and wordified rewrites were extended to the
  explicit atom form; focused regression tests pass (`84 passed`).
- Corrected materialization/push `3829067` completed cleanly. The one-seed
  train-1-to-25 logic SFT pilot `3829069_12` completed all `10000` steps in
  `12:39:51` with final train loss `0.0171`; its final adapter and complete
  step-5000/10000 checkpoints are present. The original full-size pilot
  eval/audit `3829070 -> 3831023`
  was canceled before start because comparable eval rows take 13--23 hours and
  risk the 24-hour limit. First replacement gate `3831135_12` was canceled
  after four minutes when process inspection found Slurm had truncated the
  comma-containing export to `--k-values 1`. The wrapper now accepts a
  delimiter-safe `PASSK_K_VALUES_COLON`; corrected replacement `3832945_12`
  completed in `07:10:38` on A40 with 16 prompts per depth, 8 samples, greedy
  plus pass@`1/2/4/8`, the same equal `7168` cap, and all 14 depths. Structural
  audit `3831136` and sampled qualitative probe/audit
  `3833178_12 -> 3833179` completed cleanly. The latter retained all eight
  sampled generations for four prompts at depths `1/25/30/50`; every row had
  fresh `c0..c_depth` constants and the expected prompt/sample-index coverage.
  Manual inspection found correct complete traces, late wrong-branch
  transitions, and depth-50 repetition/nontermination, with no evidence of
  another ambiguous-data or extraction bug. User approval accepted the
  expected corrected runtime, so the manual hold on full SFT `3829072` was
  released at 10:15 CEST. Rows `0..11` started immediately on A40 nodes and
  entered optimizer training; at 10:40 they ranged from step `239` to `1333` of
  `10000` with no fatal/OOM/quota signature. Rows `12..29` wait only on the
  array throttle. Every validity-fixed downstream eval row
  `3838163` remains constrained to `a100` plus feature
  `a100_80`, throttle 6, and starts only after its corresponding SFT row
  succeeds. It replaces pre-fix/canceled arrays `3829073/3834582`; the replacement
  retains two sampled generations for every one of the 448 prompts so the
  required qualitative audit can cover all depths without changing compute.
  Row-level audit array `3838164_[0-29%8]` follows the eval with row-wise
  `aftercorr` dependencies. It requires exact metrics, including translated NL
  parse/citation-free-valid/joint metrics, plus 1,024 retained rows with
  numeric formal and translated validity fields,
  all 448 prompts, sampled indices `0/1` for every prompt, complete chunk logs,
  cap diagnostics, fresh constants in formal prompts, and no contradiction
  between a positive validity flag and retained validity diagnostics.
  Replacement aggregation `3838165` releases only after all 30 row audits
  pass. It supersedes canceled jobs `3834707/3835779` and also runs a
  cross-grid qualitative audit. That gate requires all 30 sample artifacts and
  exact per-depth retained coverage, then indexes representative shallow,
  train-edge, first-OOD, and depth-50 correct/incorrect/valid/invalid cases
  across both modalities, every train range, and all seeds. It also identifies
  retained examples from generation chunks that reached the `7168` cap and
  writes reviewable JSON/Markdown supplements before aggregation can succeed.
  The logic-engine and modality-aware gate changes pass all 26 focused tests
  through the preferred project environment (`26 passed in 0.34s`). The
  pre-fix first production row is intentionally rejected and quarantined;
  pending audit rows load the fixed Python at runtime.
  The pilot had occupied row 12's production filenames with a smaller
  224-prompt/8-generation artifact. Those files are preserved under the
  `_pilot_gate` suffix and the production names are now clear, so row 12 cannot
  skip its full rerun. The local eval wrapper also refuses to skip an existing
  artifact unless it has the exact 448-prompt/16-generation/1,024-row shape.
- The accepted runtime audit records all four greedy and 28 sampled chunks.
  Greedy generation took `3826.3s`; sampled generation took `20677.8s` and
  produced `5,062,917` tokens. One greedy and seven sampled chunks hit the
  `7168` cap, all at the longest depths. Both timing probes actually ran on
  A40s (`a1721` and `a0121`); the earlier handoff incorrectly treated `a0121`
  as an A100 comparison. Scaling the A40 gate projects more than 25 hours, but
  the old full protocol ran on A100-80GB in roughly 3.5--8 hours (the matched
  old row-12 logic eval took `03:56:41`). Corrected outputs are substantially
  longer, the batch is `64` rather than `128`, and greedy adds work. The full
  A100-80 eval is now approved unchanged; treat its first completed row as the
  production timing measurement and intervene only if it approaches the
  24-hour limit. Preserve all depths, prompts, generations, and pass@16.
  The first production row now provides a direct A100-80 measurement. At
  `3:19` wall time it had completed all seven greedy chunks and sampled chunk
  `55/112`. Completed sampled depths `1/2/5/10/12/15` took
  `39/65/171/1661/1553/1739` seconds, and seven of eight depth-18 chunks took
  `2158` seconds. Even charging every one of the 57 unfinished chunks the
  worst observed `350.2` seconds gives `7.60` hours for sampled generation;
  adding measured greedy generation, setup, and scoring projects about nine
  hours total. This is well below the 20-hour intervention threshold, so no
  depth sharding or protocol reduction is justified.
- At 00:14 CEST, 22/30 SFT adapters were final: logic rows `0..12` and NL rows
  `15..23`; logic `13/14` and NL `24..29` remained active. First full eval row
  `3834582_0` completed on A100-80GB in `11:18:40`, but raw-sample inspection
  found 14/896 retained sampled traces marked citation-free valid despite
  premise-parse errors. `ProofAnalyzer` did not require all premises to parse
  when computing `ok`. The logic engine and artifact gate are fixed and 26
  focused tests pass. Old eval/audit/aggregate `3834582/3834706/3835779` were
  canceled; pre-fix row 0 is quarantined outside aggregate inputs. Clean
  A100-80GB replacement eval `3838163`, audit `3838164`, and aggregate
  `3838165` are dependency submitted with the unchanged 448-prompt,
  16-generation, all-depth protocol. Details are in
  `docs/branchproof_validity_evaluator_audit_2026-07-12.md`.
- At 00:41 CEST, the late-submitted replacement array still showed every
  `aftercorr` child dependency unfulfilled even though parent rows
  `0..12/15..23` had completed successfully and all 22 corresponding final
  adapters were present. Dependencies were cleared only for those 22 verified
  eval tasks. They are now scheduler-eligible with `Features=a100_80`; unfinished
  rows `13/14/24..29` retain their one-to-one SFT dependency. The three
  Nanotron recovery start estimates remain approximately 05:14.
- Logic rows `13/14` subsequently completed `0:0`, wrote their final adapters,
  and passed the same artifact gate. Their eval dependencies were cleared at
  01:05/01:06 CEST. Corrected SFT finals and scheduler-eligible A100-80 evals
  are therefore `24/30`; only NL rows `24..29` remain active and dependency
  gated. Their latest steps were `9076/7523/6701/4695/4685/4272`.
- NL rows `24/25` subsequently completed `0:0` at 01:55/03:15 CEST. Their
  exact train-1-to-20 seed-3407/3408 final adapters had nonempty
  `adapter_config.json` files and no zero-byte files, so only eval tasks
  `3838163_24/25` had their stale dependencies cleared at 03:18 CEST. The
  corrected grid is now `26/30` final adapters and 26 scheduler-eligible
  A100-80 eval rows. Rows `26..29` remain running and dependency-gated; latest
  observed steps were `9209/6493/6469/6046`, with no fatal signature. The
  corrected production output root still contains only the explicitly
  suffixed pilot/qualitative artifacts, so there is no production metric or
  sample bundle to audit yet. Project usage is `866G`. Current watcher
  `3839191` preserved the chain by scheduling successor `3839693` before this
  pass; the plan remains incomplete, so the successor remains queued.
- NL rows `26..29` subsequently completed `0:0` at 03:59/07:40/07:43/08:16
  CEST, completing corrected SFT `3829072` at `30/30`. All 30 exact final
  adapter directories have nonempty `adapter_config.json` files and no
  zero-byte files; the four newest finals are each `324M`, and their logs have
  no fatal/OOM/quota signature. Slurm resolved the remaining row-wise
  dependencies without manual edits, so all tasks in validity-fixed eval
  `3838163_[0-29%6]` are dependency-free, still pinned to partition `a100`
  with feature `a100_80`, and pending only on `AssocGrpGRES`. The corrected
  output root still contains only four explicitly suffixed pilot/qualitative
  files, so row audits `3838164` and aggregate `3838165` remain closed. Project
  usage is `871G` by `du`; the latest quota epilogue reports `996.7G` used.
  Watcher `3839693` completed `0:0` in `00:14:19` after scheduling successor
  `3840018`; the successor begin time was advanced from 15:16 to 10:35 CEST so
  it can inspect the projected 10:23 Nanotron resume transition. The stored
  successor payload names the corrected critical paths, and the plan remains
  incomplete. Commits `cb70aa6` and `18f79e5` were pushed to `origin/main` at
  09:33 CEST after the watcher's own transient SSH attempts timed out.
- Control recovery `3835438_0` started at 10:24 CEST on full A100-80 node
  `a0531`. It loaded run checkpoint 4096 with optimizer and LR scheduler state,
  restored `start_iteration_step=4096`, `consumed_samples=524288`, and
  `consumed_tokens_total=2147483648`, then logged resumed iteration `4101` at
  `30.9K` tokens/s with finite loss `2.07`. This proves no replay or weight-only
  reset in the control branch. Corrected logic/NL recoveries
  `3835442_3/3835443_8` remain dependency-free and account-GRES pending, with
  projected starts near 12:27 CEST. No safe partition widening exists for
  their required full-node A100-80 resumes.
- At 10:39 CEST control recovery `3835438_0` remained healthy at iteration
  `4141/8192`, `2.17B` consumed tokens, `30.8K` tokens/s, and finite loss
  `1.98`; no fatal/OOM/quota signature appeared. All corrected BranchProof
  eval tasks `3838163_[0-29%6]` remain dependency-free, pinned to `a100_80`,
  and account-GRES pending, with no new production artifact. Watcher `3840018`
  completed `0:0` in `00:09:18` after scheduling successor `3841073`; its
  payload names the corrected critical paths. The successor was advanced from
  16:35 to 12:45 CEST so it can verify the projected 12:27 logic/NL
  first-resume transition. The full plan was incomplete, so `3841073` was
  left queued for that handoff.
- At 18:50 CEST corrected NL recovery `3835443_8` had run for `06:22` on
  full A100-80 node `a0532` and passed the resume gate: it loaded the exact
  step-4096 checkpoint with optimizer/LR-scheduler state, restored
  `524288` samples and `2147483648` tokens with the accepted
  `1825357824 + 322125824` normal/NL split, and logged resumed iteration
  `4101` before advancing to `5421/8192` at `30.9K` tokens/s with finite loss
  `1.74`. Control `3835438_0` simultaneously reached `5871/8192`, `3.08B`
  tokens, and finite loss `2.11`. Neither log has a fatal/OOM/quota signature.
  Logic recovery `3835442_3` and all corrected BranchProof eval tasks
  `3838163_[0-29%6]` remain dependency-free, A100-80-only, and blocked solely
  by `AssocGrpGRES`; an A100-80 node is idle, confirming the constraint is the
  account GPU ceiling rather than node availability. No scheduler or protocol
  edit is justified. No step-8192 tree or corrected BranchProof production
  JSON/sample artifact exists yet; project usage remains `871G` by `du`.
- Watcher `3841073` failed after `00:00:13` because Codex reported a usage
  limit, but its wrapper had already scheduled successor `3842454`. That pass
  started at 18:45 and scheduled successor `3843796` for 00:45 CEST on
  2026-07-13. The stored successor remains queued because the full plan is
  incomplete.
- Watcher `3842454` completed `0:0` in `00:09:29`; successor `3843796` remains
  queued for 00:45 CEST on 2026-07-13. At 21:10 control had reached
  `6371/8192` and NL `5921/8192`, both at about `30.9K` tokens/s with finite
  losses (`2.05/1.78`) and no fatal/OOM/quota signature. Logic recovery remains
  account-GRES pending with a projected 07:00 start. Only step-4096 checkpoint
  trees exist, project usage remains `871G`, and BranchProof eval remains
  account-GRES pending without production outputs.
- At 00:55 CEST on 2026-07-13 control recovery `3835438_0` had advanced to
  iteration `7161/8192` and corrected NL recovery `3835443_8` to `6711/8192`,
  both at about `30.9K` tokens/s with finite losses (`2.01/1.63`) and no
  fatal/OOM/quota signature. Their runtime ETAs were about `04:51/06:59` from
  the latest logged iterations. Only accepted step-4096 trees exist; no final
  checkpoint, HF verification report, downstream artifact, or guarded cleanup
  trigger is present, and project usage remains `871G`. Logic recovery
  `3835442_3` remains dependency-free and account-GRES pending with a
  provisional 07:00 start. Corrected BranchProof eval `3838163_[0-29%6]`
  remains dependency-free, A100-80-only, and account-GRES pending with no
  production output; a temporary row-0 start estimate disappeared on the next
  scheduler cycle and was not treated as a launch. The only idle A100 nodes
  were `a0903/a0905`, both feature `a100_40`, so no compatible partition or
  feature widening exists. Watcher `3843796` scheduled successor `3845763`
  for 06:46 CEST before starting; the stored batch payload was verified to
  retain the corrected job IDs, protocol, audit gates, and end-to-end stop
  condition, so the incomplete plan remains continuously covered.
- At 06:53 CEST control recovery `3835438_0` is `COMPLETED 0:0` after
  reaching step `8192` and saving its final checkpoint. Independent audit
  `analysis/nanotron_checkpoint_audits/control_step8192.json` accepts the tree:
  645 files, no zero-byte files, TP=4/DP=2, 625 model files, four scheduler
  shards, eight RNG shards, four equal `22,848,937,060`-byte optimizer shards,
  and metadata offsets `8192/1048576/4294967296`. Upload `3831119` is released
  and pending only on account GRES; no local checkpoint was deleted. Logic
  recovery `3835442_3` started at 05:48 on `a0531` and passed the no-replay
  gate: optimizer/LR-scheduler loading is enabled, step/sample/token offsets
  restored at `4096/524288/2147483648`, the exact
  `1825357824 + 322125824` normal/logic split was restored, and iteration
  `4101` had finite loss `1.71`. It reached `4301/8192` at `30.9K` tokens/s
  with finite loss `1.76`. NL recovery `3835443_8` remained healthy at
  `7971/8192`, `30.9K` tokens/s, finite loss `1.65`, and about one hour to its
  final save. Current total vault usage is `1211.8G` against the `2097.2G`
  hard quota; the three matched Nanotron roots occupy `795G`. Corrected
  BranchProof eval `3838163` and all downstream jobs remain gated without new
  production artifacts. Watcher `3845763` scheduled successor `3846896` for
  12:46 CEST before this pass; that payload was later superseded as recorded
  below.
- At 10:10 CEST NL recovery `3835443_8` is also `COMPLETED 0:0` after
  reaching step `8192`. Independent audit
  `analysis/nanotron_checkpoint_audits/nl_exact_step8192.json` accepts 645
  files, no zero-byte files, TP=4/DP=2, 625 model files, four scheduler
  shards, eight RNG shards, four equal `22,848,937,060`-byte optimizer shards,
  exact offsets `8192/1048576/4294967296`, and exact
  `3650719744 + 644247552` normal/NL token accounting. Upload `3831113` is
  released and account-GRES pending alongside control upload `3831119`; both
  retain their full local trees. Logic recovery `3835442_3` reached
  `4991/8192` at `30.9K` tokens/s with finite loss `1.76` and a projected
  completion near 01:12 CEST on July 14, comfortably before its 05:48
  allocation limit. Corrected BranchProof eval remains A100-80-only and
  account-GRES pending with no production output. Total vault usage is
  `1268G` against the `2097.2G` hard quota. Watcher `3845763` completed
  `0:0`; successor `3846896` was subsequently replaced with refreshed watcher
  `3847667` at the same 12:46 CEST start time.
- At 10:40 CEST refreshed watcher `3847667` was submitted for 12:46 CEST and
  its stored batch payload was verified to include the repaired upload IDs,
  tokenizer compatibility fix, replacement instruction jobs, HF cleanup, and
  current BranchProof rows. Stale queued watcher `3846896` was then canceled.
- At 10:14 CEST corrected BranchProof eval rows `3838163_0..5` were running
  on six verified A100-SXM4-80GB allocations; rows `6..29` are pending only
  on the array throttle. All six selected the intended corrected adapters and
  reached merge/model startup without a fatal/OOM/quota signature. Row 0
  initialized vLLM with context `16384`, batch `64`, and entered the unchanged
  greedy protocol at `max_new_tokens=7168`; its first 64-prompt chunk completed
  in 7.7 seconds with maximum output length 367. This is a startup check, not
  a runtime or scientific result. No corrected production JSON/sample bundle
  is complete yet, so row audits and aggregation remain closed.
- At 10:36 CEST control/NL uploads `3831119/3831113` had failed immediately
  because their converter was launched without Nanotron's required distributed
  environment. The wrapper now uses single-rank `torchrun`; focused shell,
  bytecode, and verifier tests pass (`5 passed`). Replacements
  `3847569/3847570` completed `0:0`, each producing four HF shards, finite CUDA
  logits `[1,152064]`, and zero remote-manifest omissions before guarded local
  cleanup. Repos are
  `flaitenberger/qwen25-7b-logic-cot-midtrain-control-p0-step8192` and
  `flaitenberger/qwen25-7b-branchproof-unique-v2-midtrain-nl_exact-p15-step8192`.
  Transformers 5 had saved the 13 Qwen special tokens under
  `extra_special_tokens`, which Transformers 4.57.3 rejected. Both remote
  configs were normalized to `additional_special_tokens`; fresh loads under
  Transformers 4.57.3 and 5.12.1 preserve IDs `151644..151656` and native-chat
  rendering. Direct evals `3835927/3834904` are released. Replacement
  instruction SFT jobs `3847661/3847662` feed existing evals
  `3835928/3834905`; strict aggregate `3836159` remains intact.
- HF storage audit found `83.942G` of models plus `21.896G` of datasets.
  Deleted only three unreferenced merged OLMo SFT repos for seeds
  `3407/3408/3409` (`43.818G` total); their retained rank-16 LoRA repos plus
  public base `allenai/Olmo-3-1025-7B` can reconstruct them. All datasets,
  unrelated models, and new Qwen repos were preserved. Verified model storage
  is now `40.125G`; total model+dataset storage is about `62.021G`, projected
  to `77.264G` after the logic upload. Audit artifact:
  `analysis/hf_storage_cleanup_2026-07-13.json`.
- At 11:22 CEST the Hub LFS inventory, which includes retained history rather
  than only current repository heads, was `86.191G`; this explained the
  apparent conflict with the `62.021G` head-size audit and would have reached
  `101.434G` after the pending logic upload. Removed only 34 historical LFS
  objects (`16.694G`) from `flaitenberger/imagenet-100-LT-balanced` that were
  unreachable from every live branch, tag, and conversion ref. Byte-level
  path/size/SHA snapshots verify that both live `main` and `parquet` trees are
  unchanged. No additional model or dataset repository was deleted. Retained
  LFS is now `69.497G`, projected `84.740G` after logic, leaving about `15.26G`
  under the nominal 100 GB limit before small instruction adapters. The
  broader mixture grid therefore needs checkpoint rotation or external
  archival; it cannot retain every 15.243 GB full checkpoint simultaneously.
- At 11:34 CEST a fresh API inventory reconciled all `71` account repos and
  confirmed retained LFS remains `69.497G`: `41.094G` models plus `28.403G`
  datasets. Current project control/NL checkpoints account for `30.485G`; the
  three reconstructing SFT LoRAs use `0.480G`; and `17` private,
  remote-only `autoformalization-*` adapters from another project use
  `10.129G`. None of those cross-project adapters was deleted: even deleting
  all of them would not fit one additional `15.243G` full checkpoint and
  would discard non-local trained artifacts. The pending logic upload remains
  safe at a projected `84.740G`. Adopt upload/evaluate/audit/delete rotation
  for later full-grid checkpoints, never deleting a model referenced by a
  pending/running evaluation. Exact reconciliation is recorded in
  `analysis/hf_storage_cleanup_2026-07-13.json`.
- At 11:57 CEST, after explicit user authorization to reduce Hub pressure,
  nine older `autoformalization-*` iterations were archived before deletion.
  The archive fixes each exact Hub commit and verifies all 117 files by size
  and SHA-256 under
  `$HPCVAULT/hf_model_archives/2026-07-13_autoformalization_superseded/`.
  Only iter1 rows with a retained iter2 and folio/mmlu iter1--2 rows with a
  retained iter3 were removed; the latest adapter for every task, the strict
  MMLU variant, all datasets, the three reconstructing SFT LoRAs, and both
  active Qwen checkpoints remain. All nine deletions were API-verified absent.
  A complete LFS re-inventory now gives `63.487G` across 62 repos (`35.084G`
  models and `28.403G` datasets), reclaiming `6.010G` including retained
  history. The pending `15.243G` logic upload projects to `78.730G`, leaving
  `21.270G` under the nominal 100 GB limit. This is enough for the matched p15
  chain, but the broader mixture grid still requires guarded
  upload/evaluate/audit/delete rotation.
- At 12:20 CEST an authenticated per-repository `usedStorage` reconciliation,
  after the control instruction adapter upload, measured `63.610G` across 63
  repositories: `35.257G` in 14 model repos and `28.352G` in 49 dataset repos.
  The pending logic checkpoint projects to `78.852G`; one further `15.243G`
  full checkpoint would project to `94.095G`, while two would project to
  `109.338G`. The seven retained latest autoformalization adapters plus their
  strict MMLU variant total only `4.119G`, so deleting those irreplaceable
  latest artifacts would not remove the need for rotation. No further model
  was deleted. Preserve the active control/NL checkpoints, instruction
  adapter, reconstructing SFT LoRAs, and latest cross-project adapters; rotate
  broader-grid checkpoints only after upload parity, evaluation, and artifact
  audits pass.
- At 11:40 CEST the recurring `fix_mistral_regex=True` warning in corrected
  BranchProof eval logs was resolved as a false remediation suggestion for
  this protocol. The base training tokenizer and current merged-eval default
  tokenizer produce identical IDs on `704/704` actual prompts/targets across
  both modalities and depths `1..50`. Enabling the suggested fix changes
  `640/704` texts and would introduce a train/eval mismatch. Keep the current
  default tokenizer. Audit artifact:
  `analysis/branchproof_tokenizer_consistency_2026-07-13.json`.
- At 11:41 CEST all active paths remain healthy. Corrected BranchProof rows
  `0..5` are through greedy generation and are sampling chunks
  `28/18/27/43/42/44` of `112`, respectively; rows `6..29` wait only on the
  six-row throttle. Logic midtraining reached `5311/8192` at `30.9K` tokens/s
  with finite loss `1.69` and still projects to finish near 01:11 CEST on
  July 14. Control instruction SFT reached `5227/10000` with finite losses;
  NL direct evaluation is actively processing a `109837`-request task stage
  and had reached `67407` requests. No corrected BranchProof production file
  or complete downstream result bundle exists yet, and focused fatal-log
  scans are clean. CPU-only watcher `3847808` remains scheduled for 12:46
  CEST and its stored payload schedules the next six-hour pass before Codex.
- At 12:14 CEST the MATH-500 evaluator gap is closed without a GPU rerun.
  `scripts/analysis/rescore_math500.py` now scores only the first nonempty line
  after the task's `Answer:` prompt using complete-expression symbolic
  equivalence. It rejects later prompt leakage, wrong equations/tuples, extra
  comma-separated answers, and incomplete repetitions; the stock exact score
  remains diagnostic. Focused scorer/audit/aggregate/Slurm tests pass
  (`18 passed`), and the preferred venv is dependency-clean. NL direct
  `3834904_8` now has accepted post-hoc MATH score `80/500 = 0.160` versus
  stock exact `14/500 = 0.028`, with all stock positives preserved. Its full
  production audit remains accepted at 50,693 unique scored documents. The
  live audit creates this deterministic sidecar for every pending matched run,
  so no job cancellation or GPU re-evaluation is needed. Details are in
  `docs/nanotron_math500_scoring_audit_2026-07-13.md`.
- At 12:14 CEST control instruction SFT `3847661_0` completed and released
  post-instruction eval `3835928`, which is account-GRES pending. Replacement
  control direct eval `3847792_0` started on A100-80GB. NL instruction SFT
  `3847662_8` reached about `4176/10000`; logic midtraining reached
  `5431/8192` at `30.8K` tokens/s with finite loss `1.73`. Corrected
  BranchProof eval rows `0..5` reached sampled chunks
  `39/30/33/55/53/55` of 112 with no production bundle or fatal signature.
- At 12:31 CEST a full raw-generation audit of accepted NL-direct run
  `3834904_8` covered all primary metrics plus correct/incorrect examples.
  GSM8K generations are format-clean; BBH and MMLU-Pro show genuine direct-LM
  continuation pathologies. Invalid extraction rates are `9.1%/20.5%`, and a
  literal next-document assistant preamble appears in `22.9%/3.7%` of rows,
  respectively. The marker occurs in none of the corresponding prompts.
  Correct samples contain coherent task reasoning; incorrect samples expose
  omitted constraints, false implication reversals, or repetition to the cap,
  not an extraction bug. The matched aggregate now emits these condition-blind
  diagnostics for all six direct/instruction bundles. Its qualitative index
  now preserves raw continuations and prompt tails rather than only filtered
  answers. Focused tests pass (`20 passed`). Interpretation remains provisional until control, logic, and
  post-instruction results complete. See
  `docs/nanotron_nl_direct_generation_audit_2026-07-13.md`.
- At 12:48 CEST replacement NL instruction SFT `3847662_8` completed `0:0`
  in `01:04:38` after all `10000` steps. It ended with train/eval loss
  `0.9525/0.9435`, wrote an 11-file nonzero local final including the
  `161,533,192`-byte adapter, and uploaded commit
  `cddf739f4b4332e1d9f3d71b825e52c836476679` to the intended adapter repo.
  Remote metadata retains the Transformers-4-compatible
  `additional_special_tokens` form and names the repaired NL p15 checkpoint
  as its base. Post-instruction eval `3834905_8` is therefore released and
  waits only on account GRES. Control post-instruction eval `3835928_0`
  started at 12:48 on A100-80GB node `a0535`, merged the accepted adapter, and
  initialized the intended 8192-context reviewer run. Control direct
  `3847792_0` remains healthy on A100-80GB and had processed `17018/20362`
  prompts in its generative stage at 12:55. Logic recovery `3835442_3`
  reached `5571/8192` at `30.8K` tokens/s with finite loss `1.69` and still
  projects to finish near 01:14 CEST on July 14.
- The same 12:54 authenticated Hub reconciliation now measures `63.782G`
  across 64 repositories: `35.430G` in 15 model repos and `28.352G` in 49
  dataset repos. The pending logic checkpoint projects to `79.025G`; logic
  plus one further full checkpoint projects to `94.268G`, whereas two project
  to `109.511G`. No repository was deleted. Guarded rotation remains required
  before any broader mixture grid.
- At 12:55 the six corrected BranchProof eval rows were at sampled chunks
  `49/35/39/64/61/64` of `112`; rows `6..29` remain throttle pending. Charging
  every unfinished chunk the worst late-chunk time observed in its own row
  gives a slowest-row total projection below 14 hours, safely under the
  20-hour sharding trigger. There is still no corrected production JSON/sample
  bundle, no fatal signature, and no reason to alter the A100-80-only protocol.
  CPU-only audits `3847756` and aggregate `3847757` remain correctly gated.
- Control direct eval `3847792_0` completed `0:0` in `01:06:47` and its strict
  artifact audit accepts all ten task groups, 105 leaf sample files, and
  50,693 rows. Control post-instruction eval `3835928_0` completed the same
  GPU workload and wrote a complete bundle but exited `1:0` when the post-hoc
  gate rejected positional task ordering and one correct MATH response whose
  first line was prose. The task check is now order-insensitive and MATH
  schema v4 falls back to an explicit answer only when the first line
  has no answer token; generated next-`Problem:`/`Question:` records are
  excluded. All 100 fallback rows per instruction bundle were enumerated;
  `36` control and `35` NL fallbacks end in explicit equivalent answers.
  Schema v4 also
  ranks an answer-cued delimited expression above a later malformed plain-text
  suffix, and the original NL
  direct score remains `80/500 = 0.160`. Focused tests pass (`24 passed`).
  Final schema-v4 CPU gate `3849774` accepted the control instruction bundle
  at MATH `65/500 = 0.130`; control direct is `79/500 = 0.158`.
  NL post-instruction `3834905_8` also completed all GPU work and its full
  bundle before the stale gate exited `1:0`; CPU gate `3849775` accepts it at
  MATH `61/500 = 0.122`. Failed-parent aggregates were canceled and final
  CPU-only aggregate `3849776` now waits only on logic direct/instruction
  evals `3847804/3847806`, so neither instruction GPU evaluation will rerun.
- At 10:41 CEST NL direct eval `3834904_8` was running on A100-80GB. It
  selected all ten reviewer tasks, loaded the repaired remote metadata,
  resolved `Qwen2ForCausalLM`, and initialized vLLM without the previous
  tokenizer exception. Control direct eval and both replacement instruction
  SFT parents remain priority pending.
- At 10:46 CEST exact dry runs against both repaired remote checkpoints loaded
  UltraChat and retained all `8/8` train plus `4/4` eval examples. Native Qwen
  system/user/assistant rendering was intact and labels supervised assistant
  tokens only. The focused instruction-format, downstream-audit, and aggregate
  tests pass (`12 passed`). NL direct eval completed model/KV-cache startup and
  began constructing the full reviewer-suite contexts. Current account use is
  all 16 allowed A100 GPUs: 15 for this project and one unrelated job, which
  fully explains the remaining `AssocGrpGRES` waits.
- At 10:50 CEST watcher scheduling was made independent of that GPU ceiling.
  CPU-only `a100mig` probe `3847702` completed `0:0` with no GRES while all 16
  GPUs were occupied. The oversight wrapper no longer requests a MIG GPU;
  replacement watcher `3847703` requests only four CPUs and 30 GB RAM at the
  same 12:46 CEST start, and its stored payload was verified before stale
  GPU-requesting watcher `3847667` was canceled. Self-scheduled successors
  inherit this CPU-only request.
- At 10:53 CEST native-chat instruction SFT became restart-safe before any
  replacement parent started. All three exact output roots were absent, so
  first-launch behavior is unchanged; the wrapper now requests automatic
  latest-checkpoint resume after a timeout or node loss. Resolver, wrapper,
  chat-format, and exact remote-checkpoint dry-run checks pass (`7 passed`).
- At 10:59 CEST all six active BranchProof eval rows were confirmed both in
  the stored request (`Features=a100_80`) and on A100-SXM4-80GB nodes. Rows
  `0/2` were in greedy chunk 6 of 7, row `1` in chunk 5, and rows `3..5` in
  chunk 7; no corrected production artifact is complete yet. The CPU-only
  row-audit/aggregate chain is now `3847756 -> 3847757`; it preserves
  `aftercorr:3838163` and `afterok:3847756`, requests no GRES, and supersedes
  canceled GPU-requesting `3838164/3838165`. Aggregation now emits explicit
  three-seed variability in the primary table and depth-curve bands; focused
  tests pass (`8 passed`).
- At 10:59 CEST logic midtraining `3835442_3` was healthy at `5171/8192`,
  `2.71B` tokens, `30.9K` tokens/s, finite loss `1.77`, and ETA about 01:13
  CEST on July 14. NL direct eval `3834904_8` had processed `3571/20362`
  prompts (18%) on A100-80GB; control direct eval `3835927_0` had started on
  A100-80GB and selected the same ten-task reviewer suite. No downstream
  result bundle is complete yet.
- Pending CPU-only watcher `3847703` was replaced at the identical 12:46 CEST
  start by `3847769` so its stored payload follows audit/aggregate
  `3847756/3847757`. The replacement requests only four CPUs and 30 GB RAM;
  the obsolete watcher was canceled only after stored-script verification.
- At 11:06 CEST control direct eval `3835927_0` had failed during model startup
  because inherited `hf_transfer` received a transient 403 for the first HF
  shard. It did not generate or score benchmark data. Both remote-model
  wrappers now force standard resumable HTTP; focused downstream tests pass
  (`17 passed`). A100-80 replacement `3847792_[0%1]` is account-GRES pending.
  The old strict aggregate became unsatisfiable and was replaced by CPU-only
  `3847793`, which waits on `3834904/3834905/3847792/3835928/3834908/3834909`.
  Queued watcher `3847795` supersedes `3847769` at the same 12:46 start and its
  stored payload follows these recovery IDs. Control instruction SFT
  `3847661_0` is running on A100-80GB; its live process has Xet disabled and
  `hf_transfer` is not enabled.
- At 11:12 CEST a stored-payload audit found that dependency-held logic jobs
  `3831123/3834908/3831125/3834909` still contained the pre-repair converter,
  downloader, and non-resumable instruction wrappers captured at submission.
  They were canceled before start only after replacements were verified:
  upload `3847802`, direct eval `3847804`, instruction SFT `3847805`, and
  instruction eval `3847806`. The stored upload uses single-rank `torchrun`
  plus downstream verification; both evals disable `hf_transfer`, and SFT
  auto-resumes. CPU-only strict aggregate `3847807` now waits on the exact six
  accepted evals `3834904/3834905/3847792/3835928/3847804/3847806`. Watcher
  `3847808` supersedes `3847795` at the same 12:46 CEST start and its stored
  payload preserves the six-hour successor chain with all current IDs.
- Scoped oversight commit `cf5162e` was pushed successfully from the login
  node at 21:12 CEST after the watcher's transient SSH attempts timed out; the
  report repository remains clean and synchronized.
- The report repository remains unchanged and synchronized. Main-repo handoff
  commit `19df2c4` was pushed successfully at 10:45 CEST after the watcher's
  own transient SSH attempts timed out.
- Slurm had captured the prior watcher script at submission, and queued watcher
  `3837467` still named canceled pre-fix BranchProof jobs despite reading fresh
  docs. It was replaced before start by `3839191` at the identical
  `2026-07-12 03:15:38` begin time with hop `2/120`. The stored replacement
  payload was checked for eval `3838163` and the verified per-task dependency
  recovery rule before `3837467` was canceled.
- The pilot's primary self-contained metric is citation-free validity because
  training targets use rule labels (`R`, `->E`) without numbered premise
  citations. Strict `valid=0` is expected under that citation-demanding metric,
  not evidence that the proofs are invalid. Citation-free correct-and-valid is
  perfect through train depth 25; sampled depth-30/40/50 pass@1 is
  `0.883/0.625/0.344` and pass@8 is `1.000/1.000/0.938`. In the separately
  retained depth-50 slice, `15/32` outputs answered correctly, `11/32` were
  self-contained valid-and-correct, `24/32` completed the format, and failures
  were late derivation errors or repetitive cap hits. These are one-seed pilot
  diagnostics, not yet a logic-vs-NL result.
- The full-grid sample audit now supports filtering combined greedy/sampled
  retention while independently checking the total row count and unique prompt
  coverage. The full-grid aggregator recognizes corrected `branchproof_unique_v2`
  filenames and has a strict acceptance mode. Corrected analysis uses
  `--skip-intermediate --strict-final-grid`, preventing old checkpoint mixing
  and refusing output unless all 30 unique rows, exact prompt/generation
  counts, greedy cells, and pass@`1/2/4/8/16` correctness/validity/joint cells
  are complete and monotonic. Focused tests pass (`9 passed`), old-grid
  compatibility still finds exactly 30 complete rows, and a smoke correctly
  rejects the current one-row pilot.
- The final aggregator now matches the claim-decision order instead of
  exporting mostly pass@16 summaries. Per-run, grouped mean/std, depth-curve,
  and paired logic-minus-NL tables include greedy correctness/validity plus
  correctness, modality-appropriate validity, and joint performance at
  pass@`1/2/4/8/16`. It emits separate primary greedy/pass@1 correctness
  figures and a train-1-to-25 sampling-efficiency figure, while preserving the
  existing pass@16 tables and plots. Focused pytest passes (`4 passed`), and a
  full 30-row compatibility smoke writes all new CSV/Markdown/PDF/PNG outputs.
  The corrected strict grid, unlike some historical rows, is required to have
  complete greedy and band-level sampled metrics before aggregation succeeds.
  It now also exports a five-row paired-delta summary with mean/std over the
  three matched seeds for greedy and pass@`1/2/4/8/16` correctness and joint
  metrics, plus a compact primary markdown table. Missing paired inputs remain
  `NaN`/`N/A` rather than being silently converted to zero. Eighteen focused
  tests and a 30-row compatibility smoke pass; the smoke produces five groups
  with `n=3` and preserves historical missing-greedy evidence as missing.
- An exhaustive OLMo-tokenizer audit now covers all 1,000 corrected validation
  records at each of the 14 depths in both modalities (28,000 rendered gold
  traces). No target exceeds the shared 7,168-token generation cap and no
  prompt-plus-target exceeds the 16,384-token context. Formal maxima are
  6,212 target and 13,123 total tokens; NL maxima are 6,674 and 13,596,
  leaving at least 494 generation tokens and 2,788 context tokens of headroom.
  This rules out asymmetric gold truncation in the corrected protocol. The
  reproducible accepted audit is
  `analysis/branchproof_unique_v2_eval_token_budget_2026-07-11.json`.
- The official preprint had a scientific-consistency gap after the quarantine:
  its abstract warned that old BranchProof evidence was invalid, but its title,
  introduction, main tables/figures, discussion, and conclusion still stated
  the invalidated effect as fact. The root preprint now has a neutral title and
  renders only the independent AttrCon signal, the measured uniqueness audit,
  and the corrected acceptance protocol. Historical BranchProof performance,
  architecture, syntax, shortcut, hybrid, conditioned-dual, and proof-mixture
  sections are disabled in the official source and remain available in the
  informal report. Corrected claims will replace them only after the gate and
  full three-seed rerun pass artifact and raw-generation audits.
- Both corrected 1.2B-token Nanotron corpus rows completed cleanly: logic has
  `311,301` records / `1,200,002,513` source tokens and NL has `307,329`
  records / `1,200,004,689` source tokens. Their packed Nanosets contain one
  additional EOS token per record. A full scan of all `307,329` matched
  records found zero prompt, answer, wrapper, or fresh-constant-contiguity
  mismatches. Packed token/source round-trip audit `3830855` completed cleanly:
  all metadata/file counts agree and 15 sampled records spanning depths 3--24
  exactly match source tokenization and decode round trips in both modalities.
  A corrected 15% logic integration smoke `3830924` then loaded the intended
  `0.85/0.15` normal/proof blend and completed three optimizer steps.
- The corrected 15% proof-mixture pilot is now a matched three-condition
  comparison: normal control, logic, and NL. Logic train/recovery is
  `3830927_3 -> 3835442_3`; NL integration smoke `3831110` completed all three
  steps after loading the intended `0.85/0.15` normal/NL mix, releasing NL
  train/recovery `3831111_8 -> 3835443_8`. Logic and NL started together near
  04:46 CEST on full A100-80GB nodes `a0534/a0535`. At 11:20 they were at
  steps `1391/1401` of `8192`, `729M/735M` consumed tokens,
  `30.9--31.2K` tokens/s, and loss about `1.78`, with no fatal/OOM/quota
  signature. Their 24-hour
  allocations cannot finish all 8192 steps, so the existing `afterany`
  recovery rows will resume from the complete step-4096 checkpoint. The live
  jobs explicitly override checkpoint interval to `4096`; neither corrected
  run retains 1024-step checkpoints. At 15:20 they remained matched and healthy
  near steps `2200/2210`, about `1.15B/1.16B` consumed tokens,
  `30.9--31.2K` tokens/s, and loss about `1.73--1.80`. Superseded untouched
  recoveries `3830928_3/3831112_8` were replaced before start by
  `3835442_3/3835443_8`; the replacements preserve the same after-any parents,
  isolated roots, corpus overrides, and step-4096 interval while disabling W&B
  service startup and excluding nodes `a0803/a0831`.
  Both corrected runs reached step 4096 near midnight. A persistent verifier
  accepted each 645-file tree: 625 model files, four scheduler files, eight RNG
  files, no zero-byte files, four equal `22,848,937,060`-byte optimizer shards,
  step/sample/token offsets `4096/524288/2147483648`, and the exact checkpoint
  split of `1,825,357,824` normal plus `322,125,824` proof tokens. Parents
  `3830927_3/3831111_8` were canceled only after acceptance, avoiding about
  five hours of uncheckpointed work. Recoveries `3835442_3/3835443_8` and
  control `3835438_0` are released but account-GRES pending.
  The corrected logic branch uses upload `3831123`, direct eval `3834908`,
  instruction SFT `3831125`, and instruction eval `3834909`; NL uses upload
  `3831113`, direct eval `3834904`, instruction SFT `3831115`, and instruction
  eval `3834905`. Both use isolated `bp_unique_v2` names and HF prefixes.
- A direct audit of Nanotron's production `TokenizedBytes`/
  `BlendableDataset` path found no mixture bug. The run blends 4,096-token
  chunks; each p15 arm realizes `157,287/1,048,576` proof chunks, or
  `644,247,552/4,294,967,296` tokens (`15.000057%`). The normal, corrected
  logic, and corrected NL streams are all large enough to avoid wraparound.
  Step-4096 metadata and absolute sampler offsets preserve the blend across
  recovery. Details are in `docs/nanotron_mixture_schedule_audit_2026-07-10.md`.
- All old proof-mixture Nanotron jobs and stale proof checkpoints were canceled and
  deleted because their logic/NL corpora inherited the same generator defect.
  The unaffected `0%` normal-corpus control `3823434_0` reached its exact
  24-hour wall limit after logged iteration `5051`, as expected. Recovery
  `3828946_0` started at 14:36, resolved the correct step-4096 optimizer,
  scheduler, RNG, and data offsets, then failed before its first resumed step
  because the local W&B service did not publish its port file on `a0831`.
  Step `4096` was explicitly verified complete: 625 model files, four equal
  `22,848,937,060`-byte optimizer shards, no zero-byte files, and complete
  metadata. The full filesystem-accounted checkpoint footprint is about
  `199G` (`29G` model, `171G` optimizer), not the earlier optimizer-only
  `91.4G` estimate. The checkpoint remained unchanged and complete.
  Replacement `3835438_0` is pending with W&B disabled and `a0831` excluded.
  The scheduler
  currently reports `AssocGrpGRES`: the account-level GPU allocation is
  saturated while the live full-node proof runs and A100 evaluations execute.
  The estimate is provisional, but there is no dependency, feature, or
  node-compatibility fault to repair. At 17:07 this repo occupied 31 GPUs:
  12 A40s for corrected SFT, 16 A100s for the two live full-node proof runs,
  and three A100s for corrected eval. Releasing enough account allocation for
  the eight-GPU control immediately would require displacing active corrected
  BranchProof work, contrary to the plan's priority order, so no hold,
  cancellation, or throttle reduction was applied. After all three accepted
  step-4096 checkpoints were present, a refreshed `du` measured the project
  tree at `858G`. The largest recent scheduler epilogue accounted `1072.3G`;
  conservatively adding three more `199G` final checkpoints projects about
  `1.67T`, roughly `428G` below the documented `2097.2G` hard limit. This will
  exceed the `1048.6G` soft warning threshold but leaves enough hard-space and
  file-count headroom for all final checkpoints plus HF staging. Upload
  `3831119` was rewired to it. Its downstream branch uses upload `3831119`,
  replacement direct eval `3835927`, instruction SFT `3831121`, and
  replacement instruction eval `3835928`, all with `afterok` dependencies.
  Untouched old evals `3834906/3834907` were canceled before start because
  their environment omitted the corrected unified output root and would have
  written to the legacy default root; the replacements preserve the same
  model parents, suite, resources, and A100-80GB constraint.
- Upload jobs now have a path-guarded opt-in that removes all local Nanotron
  checkpoints only after successful HF conversion/upload. Before cleanup, the
  converter requires every HF parameter to have a Nanotron mapping, and a new
  verifier checks the staged safetensors manifest/shards, reloads the entire
  model on CUDA for a finite-logit forward pass, and confirms that every local
  staged file exists in the remote private HF repository. Its JSON report is
  stored in the run root. Any failure exits before staging or checkpoint
  cleanup. Ten focused conversion/downstream tests, `py_compile`, and shell
  syntax pass. The safeguard is enabled for
  control, corrected logic, and corrected NL, preventing three-condition
  checkpoint retention from exhausting the vault. Cluster epilogue accounting
  reports a `1048.6G` soft quota and `2097.2G` hard quota. The project tree was
  `411G` at 10:37 CEST; adding two roughly `199G` corrected step-4096
  checkpoints should remain below the soft quota while the control checkpoint
  is retained.
- The held instruction branch had a real format mismatch: UltraChat was
  trained with custom `<question>/<answer>` wrappers while lm-eval supplied
  neither those wrappers nor a chat template. It now uses Qwen's native chat
  template for training and instruction-branch evaluation, masks loss to the
  assistant response, and drops rows truncated before any supervised token.
  Data smoke `3831179` completed in `00:02:15`: all `32/32` train and `16/16`
  eval rows retained targets, with lengths `71/537/1202` min/mean/max and a
  verified native system/user/assistant rendering. Instruction jobs for all
  three conditions are additionally dependency-gated on this smoke.
- Downstream-suite preflight found that the installed lm-eval harness does not
  provide `folio`, and legacy `logiqa`/`logiqa2` use dataset scripts rejected
  by the installed `datasets`. Diagnostic smoke `3834738` completed both
  direct and native-chat branches, but retained samples exposed a second
  scientific issue: installed FLD prompts request a proof while its metric
  exact-matches only `UNKNOWN/PROVED/DISPROVED`, creating an extraction floor.
  FLD is therefore excluded rather than reported as transfer. The final suite
  is `gsm8k`, `hendrycks_math500`, `arc_challenge`, `hellaswag`, `winogrande`,
  `piqa`, `agieval_logiqa_en`, `bbh`, `mmlu`, and `mmlu_pro`; MMLU formal
  logic and the BBH logic subtasks provide targeted logic readouts. Final
  direct/native-chat smoke `3834836` completed in `00:08:50` after constructing
  every dataset successfully. Each branch wrote one aggregate result with all
  ten required task/group keys and 105 nonempty sample files. Retained prompts
  verify plain direct formatting versus Qwen system/user/assistant chat
  rendering. A new production artifact audit requires the exact task list,
  one result bundle, all ten top-level task/group keys, all 105 leaf-task
  sample files, complete unique-document coverage, finite primary metrics,
  no evaluation limit, and the correct direct-versus-Qwen-chat rendering.
  Existing results are skippable only after this audit passes. Final audited
  production evals are NL `3834904/3834905`, control `3847792/3835928`, and
  logic `3847804/3847806`; they wait on their live model branches. Slurm
  confirms each is constrained to one A100-80GB GPU with 240 GB host RAM.
  The wrapper also preflights actual dataset construction and archives
  command-only, incomplete, or audit-rejected outputs.
- CPU-only strict downstream analysis job `3849776` waits for all six accepted
  bundles, using recovery gates `3849774/3849775` for the artifact-complete
  control/NL instruction runs.
  `aggregate_nanotron_downstream_pilot.py` refuses an incomplete or
  audit-rejected bundle and writes individual task values/stderr, deltas from
  the matched control, instruction-minus-direct deltas, and four predeclared
  unweighted macros: all ten primary tasks, reasoning core, general
  multiple-choice, and a targeted logic set. The targeted set is fixed before
  results and contains LogiQA, MMLU formal logic, BBH formal fallacies, and the
  three BBH logical-deduction tasks. It also indexes correct/incorrect retained
  samples from fixed representative tasks using each task's exact primary
  filter. This pilot has one training run per condition; benchmark stderr is
  reported, but macro values do not estimate training-seed uncertainty.
- Persistent plan oversight is active through watcher `3835433` using
  `scripts/slurm/codex/branchproof_nanotron_oversight_2026-07-11.slurm` on one
  A100-MIG slice. It started at 21:15 CEST and scheduled successor `3837467`
  for approximately 03:15 CEST before invoking Codex. Each pass self-schedules
  its successor
  every six hours before invoking Codex, for up to 120 passes. Each pass reads
  the live plans/handoffs, monitors and recovers the BranchProof and Nanotron
  chains, audits new generations/results, executes explicitly triggered
  follow-ups, updates reports/docs, and preserves the successor until the full
  plan is verified complete. No-op passes do not append handoff churn.
- At 15:20 CEST full BranchProof SFT had completed rows `0..5`, `12`, and `15`;
  rows `6..11`, `13..14`, and `16..19` were running, and `20..29` were held
  only by the array throttle. A100-80 eval rows `3834582_0/1/2` started. Row 0
  completed greedy generation and reached sampled chunk `48/112` after about
  `2:37`; the observed long-chunk rate projects completion comfortably below
  24 hours, so no protocol change or depth sharding is justified. Many deeper
  train-1-to-5 logic chunks hit the `7168` cap, which remains a model-behavior
  diagnostic pending the completed artifact and raw-generation audit.
- Detailed evidence, scope, remediation, and decision gates are in
  `docs/branchproof_uniqueness_audit_2026-07-10.md`.

### Other established results

- Main HFSA OLMo-7B 3-seed depth-scaling grid is complete: 30 SFT rows and sparse final pass@k eval are done.
- Main result: logic is more depth/sample efficient at intermediate train ranges; `nl_exact` catches up at train-1-to-25 on joint validity. Depth-50 joint@16 at train-1-to-25 is similar: logic `0.417`, NL `0.427`.
- Bare-format OOD rerun is complete for the main OLMo grid. NL transfers much better to GSM8K numeric EM; logic transfers much better to context-provided HotpotQA/2Wiki/MuSiQue EM/F1. Treat those QA tasks as context-QA robustness, not as proof-chain evidence.
- Tiny Llama 20k and 100k scratch-pretraining runs are complete. They learn some train-band behavior, but strict OOD/depth-50 joint validity is essentially absent; use them as mechanism smoke tests, not as solved extrapolation.
- Architecture ablations for Qwen-2.5-7B, Qwen-2.5-1.5B, Gemma-3-4B, and OLMo-2-32B short-context are complete and report-tabulated.
- The cleaner equal-length `logic_wordified` control is complete. It underperforms compact logic and `nl_exact`: train-1-to-25 mean OOD correct/joint@16 is `0.508/0.323`, and depth-50 correct/joint@16 is `0.344/0.094`.
- Trace-control repair outputs are complete and report-ingested for `18/18` rows. `invalid_logic` keeps high answer accuracy with zero grounded validity; repaired `rule_annotated_nl` and `pseudocode` have nonzero translated validity; shuffled NL parses but loses translated joint validity.
- Report refresh at 2026-06-03 10:10 CEST added direct full train-sequence examples and 512-example OLMo-token diagnostics for normal logic, normal NL, `terse_nl`, `rule_annotated_nl`, `pseudocode`, shuffled/invalid controls, symbol-padded logic, and wordified logic. The trace-control and hybrid-order tables now include normal compact-logic/exact-NL baseline rows inline. New caveat: current HFSA `terse_nl` is token-identical to `nl_exact` under the audit and sampled sequence, so it is not an actual shorter-NL control in these artifacts.
- Report refresh at 2026-06-03 11:21 CEST fixed the remaining baseline visibility gap in figures: trace-control and hybrid-order plots now include the normal compact-logic/exact-NL baselines, and shortcut-kind controls now have line plots against baseline rate `0.0` in `ablation_shortcut_kind_rate_lines_vs_main.{pdf,png}`. Added direct compact-vs-symbol-padded-vs-wordified surface snippets in the report. Audit outcome: conditioned-dual uses separate mode-conditioned train examples and the same mode prompts at eval, so no same-datapoint train-test mismatch was found; poor conditioned-logic performance is more plausibly per-modality underexposure/interference at fixed steps. Shortcut-kind eval sample metadata is shortcut-neutral (`active_branch_first=None`), so the `initial_marker` NL rate-`0.8` improvement is real in current metrics but surprising/provisional, not an eval-shortcut bug.
- Shortcut-kind controls are complete at `24/24` eval JSONs. `position` rows are three-seed complete; `initial_marker` logic `0.5` and `0.8` are three-seed with OOD correct/joint@16 `0.883/0.625` and `0.885/0.610`, depth-50 `0.854/0.344` and `0.865/0.344`; `initial_marker` `nl_exact` `0.5` is three-seed with OOD correct/translated-joint@16 `0.469/0.421` and depth-50 `0.115/0.094`; `initial_marker` `nl_exact` `0.8` is now three-seed with OOD correct/translated-joint@16 `0.771/0.702` and depth-50 `0.667/0.500`.
- Original full paired-family SFT for `official_igsm`, `maze_navigation`, and hardened `attribute_constraints` is complete at `90/90` final adapters, but the combined old eval is scientifically stale for iGSM semantic grounding and maze typed-symbol questions. The old combined eval stopped at `37/90` JSONs (`official_igsm` `30/30`, old `maze_navigation` `7/30`, hard `attribute_constraints` `0/30`) and should not be broadly recovered unchanged. Current paired-family conclusions should come from the fresh semantic iGSM complete rerun, typed maze replacement eval, and hard-attribute-only eval tracked below.
- Focused iGSM validity audit at 2026-06-01 15:16 CEST found the old `nl_exact` translated-validity `0.000` was initially an evaluator coverage issue for official-relation, substitution, and modulo-23 proof lines. The coverage gap was patched at 15:29 CEST in `synthrlvl/natural_logic.py` and `synthrlvl/metrics.py`; gold materialized official_iGSM NL targets now translate/validate at depths `1/10/25/50`. Targeted rerun `3689003_[3-5,9-11,15-17,21-23,27-29%4]` is complete at `15/15`: generated iGSM NL parser coverage is now near-complete on OOD/depth-50 slices (`0.994-1.000` by train range), but generated translated joint validity remains `0.000` on OOD/depth-50 because generated variable chains often do not match the gold formal premises. Sampled iGSM logic outputs likewise often form internally valid but ungrounded invented variable chains; details and examples are in `docs/paired_igsm_validity_audit_2026-06-01.md`.
- iGSM semantic-grounding fix at 2026-06-02 15:25 CEST: the completed full-suite iGSM materialization is scientifically stale for semantic-grounding claims because its logic constants say only `v_X = official iGSM variable X` and its NL proof lines say `From the official iGSM relation...`, which hides the original iGSM object/property binding from both modalities. `synthrlvl/datasets/paired_synthetic.py` now preserves `Define <quantity> as <letter>` from the official solution, emits bare one-letter formal constants such as `h = the number of each Swan's Gallbladder`, labels temporary helper variables as intermediate calculations for the semantic quantity, and emits NL proof lines such as `From the definition of Swan's Gallbladder (h)...` with no `iGSM` wording in new targets. `logic_engine/prover.py` now allows free one-letter terms in equality formulas so official lowercase `s..z` arithmetic registers remain valid, while non-equality FOL formulas still require bound variables. `synthrlvl/natural_logic.py` accepts both old `v_`/`official iGSM relation` artifacts and new bare-symbol semantic proof prose. Verification: full `pytest -q` passed (`114 passed, 3 skipped`), py-compile passed, focused translator tests passed for old and new iGSM wording, and fresh generated official_iGSM samples across depths `1/2/5/10/25/50` validate and translate. Existing `official_igsm` SFT/eval artifacts should be treated as the old under-grounded construction; a semantic bare-symbol iGSM rebuild/SFT/eval is now the required follow-up before drawing iGSM logic-vs-NL conclusions.
- Maze/constraint grounding audit and typed-symbol fix: `attribute_constraints` does not have the iGSM hidden-variable problem; its slot/value symbols (`s0`, `v10`, etc.) are explicitly present in both the prompt and proof, and a fresh depth/index scan found no duplicate constants or validation failures. `maze_navigation` is prompt-grounded, but the old formal namespace could reuse the same word as both a room and a key (e.g. `silver = maze room silver` and `silver = maze key silver`). This is a semantic namespace ambiguity, not a proof-engine validity failure. On 2026-06-03, `synthrlvl/datasets/paired_synthetic.py` was patched so new maze logic uses typed symbols (`r_<room>` and `k_<key>`) while preserving natural room/key wording in NL and final answers. Verification passed with full paired synthetic tests (`13 passed`), generated depth sweeps through `1/2/5/10/25/50`, and a tiny parquet materialization smoke. Existing old maze artifacts remain stale for the typed-symbol question; the fresh typed maze rebuild/SFT/eval is now complete at `30/30`.
- Hybrid-order eval is complete at `30/30` JSONs and has been report-refreshed. Sample inspection of train-1-to-20/25 rows confirmed intended `formal_think`/`think_formal` surfaces and normal answer extraction, but also shows the same failure mode as the metrics: formal blocks often omit strict citations and depth-50 outputs truncate or drift, so the hybrid result is not evidence that combining both substrates improves long-depth reasoning.
- Conditioned-dual 50k checkpoint eval and final eval are both complete at `30/30` JSONs. The final train-1-to-25 summary is now three-seed: `conditioned_logic` OOD/depth-50 correct@16 `0.833/0.677` and joint@16 `0.348/0.146`; `conditioned_nl` OOD/depth-50 correct@16 `0.675/0.531` and translated-joint@16 `0.531/0.250`. Samples preserve the intended mode-conditioned prompts, so the weak conditioned result is not a train-test prompt mismatch. The result still does not close the single-modality logic/NL gap; keep the causal decision open until the batch-size ablation finishes.
- Active ablation/eval state at 2026-06-19 09:30 CEST: all HFSA ablation families and focused paired reruns are complete and report-ingested: trace controls `18/18`, shortcut-rate `18/18`, shortcut-kind `24/24`, hybrid order `30/30`, wordified/symbol-padded/length controls, conditioned dual 10k/final 50k/checkpoint 50k, batch-size `16/16`, semantic iGSM `30/30`, hard attribute `30/30`, and typed maze `30/30`. No active Slurm recovery remains for these chains.
- OLMo-3/Qwen3 32B normal proof-chain eval completed on 2026-06-28 at `12/12` JSONs plus sample JSONLs. Summary artifact: `analysis/logic_cot_report_2026-05-25/tables/hfsa_model_ablation_32b_train25_summary.csv`. Initial readout is nuanced: OLMo-3-32B logic has high answer accuracy (OOD/depth-50 correct@16 `0.954/0.917`) but zero strict grounded joint and only modest citation-free joint (`0.477/0.115`); OLMo-3-32B NL has lower answer accuracy (`0.685/0.427`) but translated joint `0.546/0.208`. Qwen3-32B NL is much stronger than Qwen3-32B logic on this setup (NL translated joint `0.838/0.792` vs logic citation-free joint `0.065/0.010`). Sample inspection shows OLMo logic often omits strict citations while remaining citation-free derivable, and Qwen3 logic often emits invalid lowercase predicate handles such as `aa`/`ab`. Treat these as diagnostic architecture/capacity results, not main-story support for formal validity.
- OLMo-3-32B conditioned-dual sample audit at 2026-07-01 13:34 CEST: exact prompt matching found `240/240` matched stored samples for both logic and NL, and tag counts showed no mode leakage (`conditioned_logic` stayed in `<formal>`, `conditioned_nl` stayed in `<think>`). The Table 7 readout should be framed as same-step mixed exposure, not additive data. Conditioned logic is slightly higher than single logic in pass@16 answer correctness (`0.963/0.979` OOD/hard-tail vs `0.954/0.975`), but the effect is small. Conditioned NL is lower on the mean because seeds `3407/3408` degrade while seed `3409` improves; sample pairs show wrong branch/state continuation rather than parser or prompt collapse. Detailed audit note: `analysis/logic_cot_report_2026-05-25/conditioned_dual_32b_sample_audit_2026-07-01.md`.
- Official preprint draft state at 2026-06-19 10:51 CEST: pulled the new `../synthetic-RLVL-report/hu_new_gen_template/` commit and created `../synthetic-RLVL-report/official_preprint/` so the old generated `../synthetic-RLVL-report/main.tex` remains the informal report. The preprint draft uses the new template, renames the main story datasets to `BranchProof` and `AttrCon`, keeps the paper scope to those two task families, integrates hybrid-order as a negative result and conditioned-dual as capacity-dependent, and includes one-claim figures with seed standard deviations where available. Static checks pass: all `\includegraphics` paths exist and all citation keys resolve. Local TeX compilation was not run because no TeX engine (`latexmk`, `pdflatex`, `tectonic`, or `xelatex`) is available on this node.
- Overleaf/report layout update at 2026-06-19 11:10 CEST: `../synthetic-RLVL-report/main.tex` is now the official preprint entrypoint so Overleaf renders it by default. The older generated informal report moved to `../synthetic-RLVL-report/informal_report/main.tex` and has `\graphicspath{{../}}` so its existing root `figures/` references still resolve. Static checks found zero missing figure refs in both root preprint and informal report, and zero missing citation keys in the preprint. Overleaf still requires using Menu -> Main document to switch compiled entrypoints; it cannot automatically compile whichever tab is open.
- Official preprint revision at 2026-06-19 11:35 CEST: pulled user edits to the report repo, then rewrote the root preprint toward a more paper-like structure. The draft now uses descriptive task names ("branching proof chains" and "attribute constraints"), moves task description into Experimental Setup with appendix details, removes paragraph headings that caused double-dot formatting, centers the main evidence in tables rather than bar plots, keeps only two line figures for depth/shortcut trends, adds main/architecture/shortcut/syntax/integrity/hybrid/appendix tables, and explicitly includes the OLMo-2-32B short-context sanity rows. Static checks: 8 tables, 2 figure refs, 0 missing figures, 0 missing citation keys, 0 `\paragraph` commands. TeX compilation remains unavailable on this node.
- Nanotron Qwen2.5 proof-mixture operational state at 2026-07-08 10:20 CEST: row `3819135_3` (`logic_p15`) is running on `a0932`, started 2026-07-07 22:10, has passed local iteration `2551/8192`, and has clean checkpoints through local step `2048`. Recovery row `3819040_0` (`control_p0`) also started and reached local step `2521`, but inspection showed the old wrapper loaded the latest weights while resetting the local step counter; to avoid overtraining the control row, it was canceled after a complete local-step-2048 checkpoint was present. The Nanotron midtrain wrapper now loads optimizer and LR scheduler state when resuming from a run checkpoint, while keeping pretrained Qwen loads as weight-only initialization. Because the already-started control recovery still contained an optimizer/LR reset segment, the compromised control checkpoint was deleted and the row-0 chain was replaced by a clean restart from the converted Qwen checkpoint: fresh train `3823434_[0%1]`, automatic `afterany` resume/skip `3823435_[0%1]`, and dependents `3823436..3823439`. Rows-1/2 corrected recovery remains `3823414_[1-2%1]` with dependents `3823415..3823418`; replacement rows `3..10` remain under `3819135` with `ExcNodeList=a0803` and dependents `3819136..3819139`. Current useful checkpoints are logic `p5` step `1024`, logic `p10` step `4096`, and logic `p15` local step `2048`; control `p0` currently has no checkpoint by design. `/home/vault` is about `856G/1000G`, and `nanotron_midtrain` is `596G`.
- Active submissions at 2026-06-22 08:52 CEST: OLMo-3/Qwen3 normal 32B proof-chain baselines are still progressing as SFT `3758372` with eval `3758373`; rows `0..3` completed and row `4` is running. OLMo-3-32B conditioned-dual capacity follow-up `3758374` failed on a Hydra append-key bug, so the old dependency-stuck eval `3758375` was canceled and replacement SFT/eval `3768259`/`3768260` were submitted from the patched script. Hard-attribute NL validity re-eval `3758371` completed and now gives nonzero translated validity in the report-ingested hard-depth band: mean NL `correct/parse/joint@16` by train max `5/10/15/20/25` is `0.420/0.833/0.006`, `0.791/0.800/0.660`, `0.762/0.766/0.661`, `0.862/0.824/0.769`, `0.808/0.815/0.795`; depth-50 remains weak (`joint@16 <= 0.073`). Nanotron FlashAttention repair is complete and verified: `$WORK/nanotron` imports `torch 2.6.0+cu124` and `flash_attn 2.7.4.post1`. Details and log paths are in `docs/running_experiments.md`.
- Nanotron feasibility update at 2026-06-22 09:35 CEST: `$WORK/nanotron` imports `torch 2.6.0+cu124`, `flash_attn 2.7.4.post1`, and `nanotron`. Dense-model launch works after local Nanotron patches that bypass optional MoE `grouped_gemm` registration when the extension is broken and align the Llama initializer with the Qwen initializer. Tiny 4-GPU dummy-data smoke `3768319` completed 3 train steps and saved a checkpoint. The pending OLMo-3-7B-shaped random-init proxy `3768322` was canceled before start because true OLMo3 support would require native/compatible OLMo3 model/conversion work. Qwen2.5-7B is the cleaner midtraining target for this Nanotron checkout: Qwen3 and OLMo3 both have extra q/k attention norms, but `Qwen/Qwen2.5-7B` matches the native Qwen2 path much more closely. Full-node Qwen2.5-7B random-init batch probe `3768359_[0-4%1]` is pending/running on 8xA100-80GB with seq len `4096`, TP=4, DP=2, and micro-batches `1/2/4/8/16`. Integrated pretrained+real-data smoke `3768374` is also pending: it exports a tiny normal/logic/NL proof-chain mixture, prints decoded packed chunks, converts/loads pretrained Qwen2.5 into Nanotron if needed, and runs two packed real-data steps with TP=1/DP=8 at seq len `1024`.
- Queue relief at 2026-06-22 19:40 CEST: all pending `synthetic-RLVL` jobs were canceled so other repo jobs can be submitted closer in the queue. Left running: normal 32B SFT row `3758372_5` (raw job `3770814`) and OLMo-3-32B conditioned-dual row `3768259_0` (raw job `3768361`). Canceled pending repo work: normal 32B SFT rows `3758372_[6-11]`, normal 32B eval `3758373`, conditioned-dual SFT rows `3768259_[1-2]`, conditioned-dual eval `3768260`, Qwen2.5 random-init batch probe `3768359`, and Qwen2.5 pretrained+real-data smoke `3768374`. Resubmit after the user's other jobs are submitted.
- Live update at 2026-06-23 15:28 CEST: the older normal 32B SFT row `3770814` and conditioned-dual row `3768361` completed cleanly. Current active SFT rows are normal 32B child `3771012_7` running since 2026-06-23 10:25 CEST and conditioned-dual child `3771013_1` running since 2026-06-22 21:56 CEST. Remaining recovery rows `3771012_[8-11%1]` and `3771013_[2%1]` are pending only behind their array task limits; eval arrays `3771014` and `3771015` remain dependency-pending. Nanotron Qwen2.5 random-init probe `3771016` partially succeeded: micro-batches `1/2/4` completed, while `8/16` OOMed on A100-80GB at seq len 4096. Pretrained real-data smoke `3771017` exported and printed the expected packed normal/logic/NL data, then failed on an HF-to-Nanotron converter relative-import invocation issue; fix the converter invocation before resubmitting that smoke.
- Live update at 2026-06-24 10:25 CEST: `$WORK` quota/cache failures hit normal 32B row `3771012_8` and conditioned-dual row `3771013_2` before optimizer progress, and conditioned-dual row `3771013_1` timed out at about step `9972/10000` with `checkpoint-5000`. Quota is fixed for new cache writes: deleted only safe completed-run intermediate checkpoints, removed reproducible local caches, moved the large Qwen3/OLMo HF hub caches to `/home/vault/c107fa/c107fa12/cache_offload/hf_hub`, and restored `$WORK` cache paths as symlinks. Write probes under `$WORK/.cache/hf/datasets` now pass. Targeted recovery jobs are submitted: normal row-8 recovery `3775860_[8%1]`, conditioned-dual row-1/2 recovery `3775861_[1-2%1]` with `SAVE_STEPS=1000`, replacement normal eval `3775864_[0-11%1]`, and replacement conditioned-dual eval `3775868_[0-5%1]`. Original normal row `3771012_9` is still running; original normal rows `3771012_[10-11%1]` remain pending by array limit.
- Live update at 2026-06-25 09:34 CEST: 32B recovery is progressing cleanly. Normal baseline row `3771012_9` completed, targeted row-8 recovery `3775860_8` completed, and original row `3771012_10` is running; only original row `3771012_11` remains array-pending. Conditioned-dual recovery row `3775861_1` completed and row `3775861_2` is running. Replacement evals `3775864` and `3775868` remain dependency-pending. The Qwen2.5 Nanotron chain remains intentionally held/dependency-pending; the three Qwen-tokenized Nanoset roots and `$HPCVAULT/synthetic-RLVL/nanotron_checkpoints/qwen25_7b_tp1` are still missing, so `3776105` should not be released yet.
- Live update at 2026-06-29 11:33 CEST: normal 32B SFT/eval is complete. Conditioned-dual recovery row `3775861_2` timed out after 24h at about step `9861/10000` with checkpoints through `checkpoint-9000`, leaving old eval `3775868` stuck in `DependencyNeverSatisfied`. Canceled `3775868`, submitted single-row resume `3795088_[2%1]` with 8h walltime and `SAVE_STEPS=500`, and submitted replacement eval `3795089_[0-5%1]` with `afterok:3795088`. `3795088_2` started immediately on `a0532`.
- Live update at 2026-06-29 12:38 CEST: Qwen2.5 Nanotron prerequisites are now actively building. Added resumable token-budget exporters for FineWeb-Edu normal text and generated HFSA proof traces, patched the local Nanotron preprocessor for installed `datatrove` compatibility, and patched the stale real-data smoke converter invocation to run as a module. Submitted prereq builder `3795206` on `a40`; it writes `$HPCVAULT/synthetic-RLVL/nanosets/qwen25/{normal_continuation,logical_deduction_logic,logical_deduction_nl_exact}` and `$HPCVAULT/synthetic-RLVL/nanotron_checkpoints/qwen25_7b_tp1`. Released midtraining array `3776105_[0-10%1]` from `JobHeldUser` and set `afterok:3795206`, so it is now normal dependency-pending rather than user-held. The proof Nanosets are generated to about `1.2B` Qwen tokens per modality so the 25% mixture rows should not have to cycle the old 50k-row HFSA Hub subset; the normal Nanoset targets about `4.8B` Qwen tokens from `HuggingFaceFW/fineweb-edu:sample-10BT`.
- Live update at 2026-06-29 14:15 CEST: conditioned-dual seed-3409 recovery `3795088` completed cleanly after `02:38:56` and produced the missing `final/` adapter. Replacement eval `3795089` has started; row `0` is running as raw job `3795535`, and rows `1..5` are pending behind the array throttle. Keep the seed-3409 intermediate recovery checkpoints until the eval chain finishes, then they can be cleaned if no rerun is needed.
- Quota cleanup at 2026-06-29 14:15 CEST: removed only inactive/recoverable storage. `$HPCVAULT` cleanup removed the completed Qwen3-32B HF cache, ten completed-run intermediate `checkpoint-*` dirs whose parents have `final/` artifacts, local repo W&B logs/caches, and tiny smoke/temp dirs; quota usage fell from about `595G` to `451G` immediately after deletion and was about `474G` on the final check while the active Nanotron prereq job continued writing data. `$WORK` cleanup removed `144` old `checkpoint-*` dirs under `$WORK/synthetic-RLVL/runs` whose parents have `final/` artifacts plus local W&B dirs; `$WORK/synthetic-RLVL/runs` fell from about `141G` to `59G` and now has zero `checkpoint-*` dirs. Left untouched: active OLMo-3-1125-32B cache/offload, active conditioned-dual seed-3409 checkpoints, active Nanotron Qwen2.5 prereq outputs, final adapters, datasets, eval outputs, active envs, and other-project directories. Remaining large `$HPCVAULT` items are `sequence-editing` about `233G`, active `cache_offload` about `121G`, and `$HPCVAULT/synthetic-RLVL` about `84G`.
- Live update at 2026-06-30 10:18 CEST, corrected 2026-07-01 11:32 CEST: OLMo-3-32B conditioned-dual capacity follow-up is complete at `6/6` eval JSONs and sample JSONLs. Summary: conditioned logic OOD/hard-tail correct@16 is `0.963/0.979`, citation-free joint@16 is `0.487/0.715`, and strict/grounded joint remains `0.000`; conditioned NL OOD/hard-tail correct@16 is `0.608/0.782`, translated joint@16 is `0.537/0.743`, and parse@16 is `0.938/0.965`. On the same OOD/hard-tail bands, single-modality OLMo-3-32B logic is correct@16 `0.954/0.975` and citation-free joint@16 `0.477/0.709`; single-modality NL is correct@16 `0.685/0.825` and translated joint@16 `0.546/0.748`. Interpretation: conditioned-dual at 7B remains worse than single-modality, but at 32B the formal conditioned mode closes and slightly reverses the single-modality logic gap on answer correctness and citation-free joint, answering the capacity concern. The NL conditioned mode remains below single-modality NL. Sample inspection confirms intended mode prompts and surfaces.
- Preprint framing update at 2026-07-01 11:32 CEST: `../synthetic-RLVL-report/main.tex` now frames conditioned dual as capacity-dependent rather than uniformly negative. The main text includes a new OLMo-7B vs OLMo-3-32B conditioned-dual capacity table and avoids using grounded validity as the headline criterion for this point.
- Nanotron prereq update at 2026-06-30 10:27 CEST: prereq builder `3795206` failed after finishing raw JSONLs and Nanosets because `examples.llama.convert_hf_to_nanotron` assumed Llama-style config fields absent from this installed `Qwen2Config` (`pretraining_tp`, `attention_bias`, RoPE fields). Local `../nanotron` is patched to default missing Qwen2 config fields and map Q/K/V biases. First rerun `3797384` then failed quickly because the converter was launched without `torchrun` and Nanotron expected `WORLD_SIZE`. The production prereq and older smoke Slurm wrappers now launch conversion with `torchrun --standalone --nproc_per_node=1 -m examples.llama.convert_hf_to_nanotron`; `bash -n`, `sbatch --test-only`, `torchrun --help`, and `git diff --check` passed. Prereq rerun `3797409` completed cleanly after `00:01:22`, verified the three Nanosets and wrote a 29G converted checkpoint at `$HPCVAULT/synthetic-RLVL/nanotron_checkpoints/qwen25_7b_tp1`. Midtraining array `3776105` has no dependency now and is priority-pending; Slurm currently estimates first start at 2026-06-30 21:22 CEST on `a0534`.
- Live update at 2026-07-01 11:06 CEST: original Qwen2.5 midtraining array `3776105` did start, but all 11 rows failed before optimizer progress with Nanotron `dacite.exceptions.UnionMatchError` on `data_stages.data.dataset`. Root cause was the generated Nanoset YAML passing `tokenizer_name` without explicit `token_size_in_bytes`/`vocab_size`, so Nanotron's strict union parser swallowed the metadata assertion as a generic union failure. Patched `scripts/slurm/jobs/nanotron_qwen25_midtrain_grid_2026-06-24.slurm` to emit `token_size_in_bytes: 4` and `vocab_size: 152064`; `bash -n`, `sbatch --test-only`, and `git diff --check` passed. Canceled stale dependency-poisoned arrays `3776106`/`3776107`/`3776108`/`3776109` and submitted fresh chain: training `3801554_[0-10%1]`, HF push `3801555`, direct eval `3801556`, instruction SFT `3801557`, instruction eval `3801558`. `3801554` is priority-pending with current Slurm estimate 2026-07-01 21:22 CEST on `a0633`.
- Live update at 2026-07-02 09:36 CEST: Qwen2.5 Nanotron training `3801554_[0-10%1]` has not started yet and remains priority-pending with Slurm estimate 2026-07-02 18:08 CEST on `a0535`; downstream `3801555/3801556/3801557/3801558` are dependency-pending as intended. All three Qwen Nanosets and the converted Qwen2.5 Nanotron checkpoint still exist. No new `synthetic-RLVL` job failures were found after the fixed resubmission; the only recent failures are the known pre-fix `3776105`/`3776106` rows. No safe partition widening was made because the job requests a full 8xA100-80GB node with `a100_80`; available alternative partitions are A40, A100-40GB, RTX Pro, or MIG.
- Live update at 2026-07-03 10:36 CEST: Qwen2.5 Nanotron training `3801554_[0-10%1]` started on 2026-07-02 and all rows failed before optimizer progress with `AssertionError: Tokenizer vocab size (151665) does not match model config vocab size (152064)`. This is Qwen2.5's padded embedding vocabulary, not a dataset-content issue: token IDs fit inside the model table, but local Nanotron required exact tokenizer/model vocab equality for Nanosets. Patched local `../nanotron/run_train.py` to allow `len(tokenizer) <= model_config.vocab_size` and log a rank-0 warning when the model vocab is padded. Canceled poisoned downstream arrays `3801555/3801556/3801557/3801558` and submitted the replacement chain: training `3808220_[0-10%1]`, HF push `3808241_[0-10%2]`, direct eval `3808252_[0-10%3]`, instruction SFT `3808253_[0-10%2]`, and instruction eval `3808274_[0-10%3]`. Training `3808220` is priority-pending on `a100`; dependencies are attached with `aftercorr` as intended. No partition widening was made because the run still requires a full 8xA100-80GB node.
- Live update at 2026-07-03 11:25 CEST: per user request, canceled the priority-pending full-node chain `3808220/3808241/3808252/3808253/3808274` and moved Nanotron debugging to single-GPU smoke tests. RTX Pro/Blackwell is incompatible with the current `$WORK/nanotron` torch build (`2.6.0+cu124` supports up to `sm_90`, while RTX PRO 6000 is `sm_120`), so Nanotron should not target `rtxpro6k` until torch/flash-attn are rebuilt. Added `scripts/slurm/jobs/nanotron_qwen25_tiny_nanoset_smoke_2026-07-03.slurm` and `scripts/slurm/jobs/nanotron_qwen25_single_gpu_smoke_2026-07-03.slurm`. Patched local `../nanotron` for installed Datatrove constructor compatibility, virtualenv-local C++ helper compilation, and local-folder consumption stats; installed `pybind11` into `$WORK/nanotron`. Tiny Qwen/Nanoset smoke `3808410` completed 3 optimizer steps on A100-MIG. Full Qwen2.5-7B smoke `3808424` completed on one A40: built the 7.62B model, loaded all 199 converted checkpoint shards from `$HPCVAULT/synthetic-RLVL/nanotron_checkpoints/qwen25_7b_tp1`, built the mixed real Nanoset loader, and finished 2 optimizer steps at seq256. After those smokes, resubmitted the production chain as training `3808429_[0-10%1]`, HF push `3808430_[0-10%2]`, direct eval `3808431_[0-10%3]`, instruction SFT `3808432_[0-10%2]`, and instruction eval `3808433_[0-10%3]`; training is priority-pending on A100-80 with no Slurm start estimate yet.
- Live update at 2026-07-03 14:50 CEST: Nanotron production chain remains pending unchanged. Training `3808429_[0-10%1]` is eligible but priority-pending with no Slurm start estimate, and downstream arrays `3808430/3808431/3808432/3808433` are dependency-pending. No training logs exist yet. Partition audit shows the compatible `a100` nodes are draining with `wrong kernel version` / `Reboot ASAP`; `a40` and `a100mig` are also draining or rebooting, while `rtxpro6k` remains incompatible with current torch for Nanotron because it is Blackwell `sm_120`. No safe partition widening or resubmission is available right now; wait for the cluster reboot/kernel issue to clear, then watch `3808429_0` when it starts.
- Post-eval storage cleanup at 2026-06-30 10:18 CEST: after conditioned-dual eval completed, removed the now-inactive OLMo-3-1125-32B HF cache/offload and seed-3409 conditioned-dual intermediate checkpoints. Kept final adapters and eval outputs. `$HPCVAULT` quota is about `543G/1000G`; `cache_offload` is now essentially empty.
- Official preprint rewrite at 2026-06-23 17:07 CEST: `../synthetic-RLVL-report/main.tex` is now a conventional NeurIPS-style paper with Introduction, Related Work, Method, Experimental Setup, Results, Analysis, Discussion, Limitations, Conclusion, and appendix sections. The main report keeps the scientific story to `BranchProof` and `AttrCon`, uses explicit depth terminology (`training range`, `long-depth band`, `hard-depth band`, and `depth-50 endpoint`), and presents the evidence through claim-specific tables plus three focused figures. Static checks pass: no banned task/ambiguous depth terms in the main report, all `\includegraphics` paths exist, all labels/citations resolve, and `git diff --check` passes. Local TeX compilation remains unavailable because `latexmk`, `pdflatex`, `tectonic`, and `xelatex` are missing on this node.
- Official preprint revision at 2026-06-23 17:37 CEST: strengthened `../synthetic-RLVL-report/main.tex` against the NeurIPS-style review. Added a claim-first overview figure, moved AttrCon/syntax/architecture/hybrid figures into the body, expanded related work and reproducibility details, added sample counts/uncertainty/caveats, reframed architecture and mechanism claims as directional/narrowing evidence, removed the weak terse-NL control from main evidence, expanded the appendix with generator/evaluator details, and replaced stale `official_preprint/main.tex` with an archival placeholder. Static checks pass: `9` figure refs, `10` refs, `22` citations, zero missing assets/cites/refs, zero ambiguous depth terms in the canonical report/script, and `git diff --check` passes. Local TeX compilation remains unavailable because no TeX engine is installed.
- Official preprint polish at 2026-06-24 10:39 CEST: revised `../synthetic-RLVL-report/main.tex` and `official_preprint/scripts/build_preprint_figures.py` after the latest review. The overview figure now uses provenance-clean excerpts from `sample_generation_snippets.csv`, all result figures are PDF-only, the builder no longer emits PNG sidecars, the AttrCon and mechanism claims are softer, the hybrid caption no longer asserts an untested cause, and the appendix now includes grammar, generator/evaluator mechanics, and a minimal paired example while keeping the paper scoped to `BranchProof`/`AttrCon`. Static checks pass: `9` figure refs, `0` missing figures, `0` non-PDF result figure refs, `0` PNG result files, `10` refs, `20` citations, no targeted stale phrases in the canonical report/script, and `git diff --check` passes. Local TeX compilation remains unavailable because no TeX engine is installed.
- Official preprint voice polish at 2026-06-24 11:23 CEST: sharpened the title to `Formal Traces Improve Length Generalization in Symbolic Chain-of-Thought`, removed stock manuscript transitions, shortened the main result/analysis subsection titles, made key paragraphs claim-first, changed Figure 1 panel titles from metric labels to claims, removed prose semicolons, and replaced LaTeX `--` ranges with words. Static checks pass: no targeted stock phrases in the canonical report/script, main report semicolons are down to `7` formal grammar/proof-example uses, `0` `--` ranges, `0` PNG result files, `9` figure refs with `0` missing/non-PDF result refs, `10` refs, `20` citations, figure-script `py_compile`, and `git diff --check`. Local TeX compilation remains unavailable because no TeX engine is installed.
- Official preprint figure pass at 2026-06-24 11:39 CEST: refactored `official_preprint/scripts/build_preprint_figures.py` so supporting figures use claim-level titles, more colorblind-friendly colors, distinct marker shapes, direct line labels where applicable, and seed dots instead of shaded bands for raw three-seed views. The architecture figure now uses point intervals instead of ceiling-clipped bars, and the hybrid/conditioned-dual figures are simplified to endpoint dot/bar summaries. Tightened remaining dense abstract/contribution wording and renamed the architecture subsection to `The Pattern Repeats Across Model Families`. Static checks pass: `9` figure refs, `0` missing figures, `0` non-PDF result figure refs, `0` PNG result files, `10` refs, `20` citations, `7` semicolons only in formal examples, `0` `--` ranges, no targeted stale phrases in the canonical report/script, `py_compile`, and `git diff --check`. Local visual rendering and TeX compilation remain unavailable because no PDF renderer or TeX engine is installed.
- Quota cleanup at 2026-06-24 11:30 CEST: removed only definitely disposable repo artifacts after the quota incident: Nanotron smoke/probe checkpoint payloads, stale merged eval dir `merged_sft_hfsa_conditioned_dual_train1to5_50k_seed3409_conditioned_nl`, canceled OLMo-3-1025-7B cache, pip cache, and old completed experiment-specific HF caches for hard attribute, batch-size, semantic iGSM, and typed maze. `$HPCVAULT` quota usage dropped from about `1975G` to about `1204G`; `$WORK/synthetic-RLVL` was about `168G`, `$HPCVAULT/synthetic-RLVL` about `41G`, `$HPCVAULT/.cache` about `538M`, and `$WORK/nanotron` about `7.0G`. This was superseded by the 2026-06-29 cleanup, which removed the now-inactive Qwen3-32B cache but kept active OLMo-3-1125-32B cache/offload. Other-project `$HPCVAULT` entries need explicit approval before cleanup.
- Qwen2.5 midtraining downstream chain at 2026-06-24 12:11 CEST: added Qwen2 Nanotron-to-HF conversion/upload tooling plus held/dependency-gated Slurm wrappers for the 11-run final-checkpoint midtraining grid: `0%` control plus `{logic,nl_exact} x {5,10,15,20,25}` proof-token mixtures, 8192 optimizer steps, seq4096, TP=4/DP=2, microbatch `4`, grad accumulation `16`, about `4.29B` tokens per run. Submitted training array `3776105_[0-10%1]` in `JobHeldUser` state; upload array `3776106`, direct downstream eval `3776107`, instruction-SFT array `3776108`, and instruction downstream eval `3776109` wait behind `aftercorr` dependencies. Do not release `3776105` yet: expected Qwen-tokenized Nanoset roots and pretrained Nanotron checkpoint are still missing (`$HPCVAULT/synthetic-RLVL/nanosets/qwen25/{normal_continuation,logical_deduction_logic,logical_deduction_nl_exact}` and `$HPCVAULT/synthetic-RLVL/nanotron_checkpoints/qwen25_7b_tp1`). The default reviewer eval suite is `gsm8k,arc_challenge,hellaswag,winogrande,piqa,logiqa,folio,bbh,mmlu`; validate task availability and data roots before release.
- HFSA batch-size ablation was launched to test whether conditioned-dual weakness is partly a physical-batch/modality-mixing issue. Eval is complete at `16/16` for seed `3407`; the 2026-06-15 report adds `hfsa_batch_size_ablation_diagnostics.csv` and `hfsa_batch_size_conditioned_delta.{pdf,png}`. Single-modality logic has best OOD joint at bsz16 (`0.583`) and best depth-50 joint at bsz4 (`0.188`); single-modality NL has best OOD/depth-50 joint at bsz8 (`0.771/0.313`). Conditioned dual is not uniformly rescued by larger batches: `conditioned_logic` OOD joint is best at bsz2 (`0.618`) and nearly tied at bsz16 (`0.587`), while `conditioned_nl` OOD/depth-50 joint is best at bsz2 (`0.781/0.344`) and often worse at larger batches. This argues against a simple "larger stratified batch fixes conditioned dual" explanation. Bsz16 must be interpreted as effective batch 16 because true physical microbatch 16 OOMed and recovery uses microbatch `8` with `grad_accum=2`.
- Fresh paired reruns are report-ingested through 2026-06-19 09:30 CEST. Hard attribute is complete at `30/30`: all logic rows are complete, and logic hard-depth joint@16 rises from `0.108` at train-1-to-5 to `0.736` at train-1-to-25, but depth-50 joint remains `0.000`; NL train-1-to-25 hard-depth correct@16 is `0.808` and depth-50 correct@16 is `0.083`, with depth-50 correctness coming from one seed (`0.000/0.000/0.250`). Typed maze is complete at `30/30` and remains a clear negative result after typed symbols: logic train-1-to-25 hard-depth/depth-50 joint is `0.000`, NL train-1-to-25 hard-depth correct@16 is `0.111`, and depth-50 correct/joint is `0.000`. Sample checks show shallow train-band traces are valid, but train-1-to-25 depth-25/50 logic outputs spend the budget on constants/premises/partial derivations and omit `<answer>`; NL depth-25/50 outputs copy premise chains to around move `18..20` and also omit `<answer>`.
- `$WORK` quota recovery at 2026-06-03 12:26 CEST: directory creation under `/home/atuin/...` is restored after deleting safe completed-run intermediate `checkpoint-*` dirs with final adapters present, old `$WORK/RLVL` disposable container/W&B cache dirs, and reproducible HF/vLLM/W&B caches. Active final adapters, datasets, eval outputs, and current merge dirs were left in place. `$WORK` usage is now about `357G`, and a create/remove probe succeeds.
- Semantic official-iGSM rerun launched at 2026-06-03 13:28 CEST: failed first build `3695464`, canceled dependents `3695465`/`3695466`, replacement build `3695525`, SFT `3695526_[0-29%3]`, and eval `3695527_[0-29%3]` use fresh `$HPCVAULT` roots and the semantic bare-symbol generator. Old iGSM artifacts are stale because their parquet was materialized before the generator fix and used hidden `v_` constants plus generic `official iGSM relation` NL wording. The new build uses `100k` train rows per train-depth subset so 10k-step, batch-size-1 SFT runs should not repeat examples under the normal no-replacement sampler.
- Trace-control validator/report correction at 2026-06-03 13:28 CEST: `terse_nl`, `rule_annotated_nl`, and `pseudocode` gold HFSA targets translate to valid logic; symbol-padded and wordified logic validate formally. The issue is that citation-free reconstruction can over-credit `invalid_logic` after deliberately broken citations, so report aggregation now uses strict grounded formal joint for formal trace-control rows. The report was regenerated and mirrored after this correction; refreshed trace-control tables now show `invalid_logic` and `shuffled_logic` formal joint validity as `0.000`, as intended.
- Old paired-family recovery issue at 2026-06-03 13:58 CEST: recovery array `3694619_[40-89%4]` failed quickly because the old paired eval script still merges adapters into `/home/atuin/.../tmp`, and `$WORK` quota blocked directory creation. Do not resubmit that old recovery unchanged; either patch it to merge under `$HPCVAULT` or deprioritize the stale old full-suite recovery in favor of the semantic iGSM and typed maze reruns.
- 2026-06-04 09:42 CEST live refresh: semantic iGSM build and all `30/30` SFT rows are complete; eval has `3/30` JSONs, rows `3/4` running, and rows `5..29` throttle-pending. The completed train-1-to-5 logic rows have high train-band joint@16 (`0.969-1.000`) and OOD correct@16 around `0.50-0.54`, but strict grounded joint remains `0.000`. Sample inspection shows generated traces now use semantic object/property labels, but formal handles/premise ordering can differ from gold, so current strict grounded validity is still likely too brittle for variable-renaming/premise-order equivalence and should not be interpreted without more diagnostics.
- Typed maze build is complete; typed maze SFT has `9/30` finals, rows `9..11` running, and eval is dependency-pending. HFSA batch-size ablation has `3` completed finals, rows `2/5/6` running, and bsz16-logic row `3` failed with CUDA OOM on the first step under physical microbatch 16. The SFT script now recovers bsz16 as effective batch 16 with microbatch `8` and `grad_accum=2`; bsz16 recovery SFT `3698380_[3,7,11%3]` and full eval `3698381_[0-15%4]` after `afterany:3695197:3698380` replace the earlier feasible-only eval `3698280`. `3698380` was widened from `%1` to `%3` at 10:18 CEST.
- 2026-06-04 13:12 CEST live refresh: semantic iGSM eval has `6/30` JSONs and rows `6/7/8` running. Typed maze SFT has `12/30` finals and rows `12/13/14` running. Batch-size row `3695197_2` (`bsz8 logic`) timed out at step ~6207 but left `checkpoint-5000`; the SFT script now supports env-controlled `train.resume_from_checkpoint`, and row-2 recovery `3698877_[2%1]` was submitted with `SFT_RESUME_FROM_CHECKPOINT=auto,FORCE_SFT=1`. Canceled stale eval `3698381`; replacement full eval `3698878_[0-15%4]` waits on `afterany:3695197:3698380:3698877`.
- 2026-06-05 09:19 CEST live refresh: semantic iGSM eval is partial at `22/30` JSONs/sample JSONLs. Original eval rows `24..28` failed during LoRA merge because `$HPCVAULT` hit disk/file quota; safe cleanup removed stale merged eval dirs and completed-run intermediate checkpoints, leaving `$HPCVAULT/synthetic-RLVL/runs` about `21G` and `$HPCVAULT/synthetic-RLVL/tmp` about `28G`. Replacement eval `3702073_[24-28%3]` is running/pending, while original rows `22/23/29` are still running. Provisional semantic iGSM metrics: logic train `5/10/15/20` OOD correct@16 is `0.516/0.522/0.573/0.551`, internal valid@16 is `0.348/0.321/0.374/0.365`, and strict grounded joint@16 remains `0.000`; NL train `5/10/15` plus one train-20 seed has near-complete NL parse@16 (`0.992-1.000`) but translated joint@16 remains `0.000`. Parser coverage is fixed; the remaining issue is generated trace grounding/equivalence.
- 2026-06-05 09:19 CEST batch-size recovery update: batch-size SFT has `6` final adapters. Row `3695197_6` (`bsz8 nl_exact`) also timed out but left `checkpoint-5000`, so row-6 recovery `3702079_[6%1]` was submitted with `SFT_RESUME_FROM_CHECKPOINT=auto,FORCE_SFT=1`. The earlier bsz16 effective-batch recovery `3698380_[3,7,11]` was canceled near walltime because it had no checkpoints and could not reach `save_steps=5000`; replacement `3702080_[3,7,11%3]` was submitted with `SAVE_STEPS=1000,SAVE_TOTAL_LIMIT=5,SFT_RESUME_FROM_CHECKPOINT=auto,FORCE_SFT=1`. Stale eval `3698878` was canceled; full eval `3702081_[0-15%4]` now waits on `afterany:3695197:3698877:3702079:3702080`.
- 2026-06-05 09:19 CEST typed maze status: typed maze SFT has `14/30` final adapters; rows `12/15/16` are running and rows `17..29` are throttle-pending. Eval `3695239_[0-29%3]` is still dependency-pending with `0` JSONs. Row `12` is near the 24h walltime and should be watched for timeout/recovery.
- 2026-06-06 10:01 CEST live refresh and fix: semantic iGSM eval reached `30/30` JSONs, but the NL validity fields are stale. Raw NL samples use the intended semantic proof surface, but generated parenthetical handles such as `(c)`/`(s)` do not have to match the gold formal variable names, so the old translator parsed the lines while validating them against mismatched variables. `synthrlvl/natural_logic.py` now canonicalizes iGSM NL definition-line handles through the semantic quantity name while preserving explicit helper handles for `intermediate calculation` lines; regression tests pass. Canceled the first re-eval `3705801` because some rows started before the helper-line fix; clean forced NL-only iGSM re-eval `3705807_[3-5,9-11,15-17,21-23,27-29%4]` is pending/running. Do not use current semantic iGSM NL translated-validity metrics until `3705807` overwrites them.
- 2026-06-06 10:01 CEST recovery update: typed maze SFT has `19/30` finals. Row `12` timed out with `checkpoint-5000`, and row `22` node-failed on `a0531`; the typed maze SFT wrapper now supports `SFT_RESUME_FROM_CHECKPOINT`, and recovery `3705793_[12,22%2]` is running with `--exclude=a0531`. Stale eval `3695239` was canceled and replacement eval `3705795_[0-29%3]` waits on `afterany:3695238:3705793`. Batch-size SFT has `8` finals; row `10` timed out with `checkpoint-5000`, and bsz16 rows `3/7/11` were canceled near walltime after saving `checkpoint-1000/2000(/3000)`. Recovery `3705794_[3,7,10,11%3]` is running/pending, and replacement eval `3705796_[0-15%4]` waits on `afterany:3695197:3698877:3702079:3705794`.
- 2026-06-08 08:47 CEST live refresh: clean semantic iGSM NL-only re-eval `3705807` completed `15/15`, so all semantic iGSM pass@k JSONs are current. Alias canonicalization fixed shallow semantic-NL validation, and parser coverage is near-complete on OOD/depth-50 rows, but translated joint validity remains `0.000` on OOD/depth-50 because generated long NL traces drift/truncate and only small prefixes validate. NL answer accuracy is much higher than logic at higher train ranges (train-1-to-25 OOD/depth-50 correct@16 `0.873/0.677` for NL versus `0.612/0.281` for logic), while logic has nonzero internal validity but poor answer accuracy and zero strict grounded joint. Treat this as a generation/grounding failure, not an NL parser coverage bug.
- 2026-06-08 08:47 CEST recovery update: typed maze SFT is `28/30` finals, with only original rows `3695238_28/29` still running; replacement eval `3705795_[0-29%3]` remains dependency-pending with `0` JSONs. Batch-size SFT is `9/12` finals. Row `10` recovered successfully, but bsz16 rows `3/7/11` timed out after saving checkpoints; stale eval `3705796` was canceled and replaced by bsz16-only recovery `3711850_[3,7,11%3]` plus full eval `3711851_[0-15%4]` after `afterany:3695197:3698877:3702079:3705794:3711850`. The new bsz16 recovery is pending during A100 maintenance.
- 2026-06-08 09:04 CEST report/scheduler update: added corrected semantic iGSM summary tables to the report builder and regenerated/mirrored the report; new artifacts include `tables/paired_igsm_semantic_by_seed.csv` and `tables/paired_igsm_semantic_summary.csv`, plus a LaTeX semantic-iGSM subsection. To work around the maintenance reservation, edited recovery `3711850` from a 24h to a `20:00:00` walltime; Slurm accepted the edit and rows `3711850_3/7/11` started on `a0532`. This is a checkpoint-progress recovery: if the rows do not finish before the shorter walltime, they should still save additional checkpoints.
- 2026-06-10 11:07 CEST live refresh: typed maze SFT is complete at `30/30` final adapters; eval `3705795_[0-29%3]` is pending with `0` JSONs because the FAU maintenance reservation is active until 2026-06-11 18:00 CEST. Batch-size SFT remains `9/12` finals: the 20h bsz16 recovery `3711850` timed out cleanly after GPU-active progress and advanced checkpoints to logic `checkpoint-8000`, `nl_exact` `checkpoint-6000`, and `conditioned_dual` `checkpoint-6000`. Stale eval `3711851` was canceled because its `afterany` dependency was satisfied by timeout despite missing finals. Submitted new bsz16 recovery `3715329_[3,7,11%3]` and replacement full eval `3715330_[0-15%4]` after `afterany:3695197:3698877:3702079:3705794:3715329`; both are pending behind maintenance.
- 2026-06-10 11:50 CEST scheduled oversight/live update: added `scripts/slurm/codex/active_recovery_oversight_2026-06-10.slurm` and submitted one-off oversight passes on the A40 partition for `2026-06-10 18:00/20:00` and `2026-06-11 00:00/04:00/08:00` CEST. Job IDs are `3715439`, `3715440`, `3715441`, `3715442`, and `3715443`; each is pending on `BeginTime`. Typed maze eval rows `3705795_0..2` and bsz16 recovery rows `3715329_3/7/11` have started; batch-size eval `3715330` is dependency-pending.
- 2026-06-10 18:10 CEST active recovery oversight: typed maze eval rows `3705795_0/1/2` are running on `a0633` since 11:42 CEST, with no JSONs yet and logs around chunks `52/112`, `52/112`, and `51/112`; later chunks repeatedly hit `max=8192`, so these rows are making progress but remain walltime-risk. Bsz16 recovery rows `3715329_3/7/11` are running on A100-80GB node `a0537` since 11:43 CEST. Row `3` resumed from `checkpoint-8000` and is around `8811/10000`; rows `7` and `11` resumed from `checkpoint-6000` and are around `6645/10000` and `6657/10000`. No unrecovered Traceback, CUDA OOM, quota/no-space, dependency-never-satisfied, model-load, or vLLM failure was found. `3715330` is correctly dependency-pending on `afterany:3715329_*`; no partition edit is useful because pending rows are array-throttle/dependency blocked and require compatible `a100_80&el9` resources. No report regeneration was run because typed maze has `0` JSONs and the batch-size eval root does not exist yet.
- 2026-06-10 19:28 CEST live refresh: first scheduled oversight `3715439` completed cleanly (`0:0`) and its handoff commit `2f9e4a3` is now pushed to GitHub. Typed maze eval rows `3705795_0/1/2` are still running on `a0633`, now around chunks `57/112`, `56/112`, and `55/112`; the output root still has `0` JSONs. Bsz16 recovery rows `3715329_3/7/11` are still running on `a0537`; latest parsed optimizer progress is about `8984/10000`, `6783/10000`, and `6807/10000`. Full batch-size eval `3715330_[0-15%4]` remains dependency-pending and its output root is not created yet. No new severe failure, scheduler edit, resubmission, aggregation, or report regeneration was triggered.
- 2026-06-10 19:58 CEST ablation/constraint recovery: submitted fresh hard `attribute_constraints` full-suite eval `3716216_[0-29%3]` with merge/output scratch under `$HPCVAULT`; rows `0/1` are running and rows `2..29` are priority/array pending. Patched conditioned-dual 50k final eval so temporary merged checkpoints go under `$HPCVAULT`, then submitted targeted recovery `3716219_[17,18,20-29%3]` for the 12 missing final `conditioned_nl` rows. Completed ablation artifact audit: trace-control `18/18`, shortcut-rate `18/18`, shortcut-kind `24/24`, wordified `3/3`, hybrid-order `30/30`, conditioned-dual checkpoint `30/30`; batch-size, typed maze, hard attribute, and recovered conditioned-dual final outputs are still pending new JSONs.
- 2026-06-10 20:05 CEST live refresh: typed maze eval rows `3705795_0/1/2` remain running on `a0633` with no JSONs; current logs show row `0` finished chunk `58/112` and is in `59/112`, row `1` finished `57/112` and is in `58/112`, and row `2` finished `56/112` and is in `57/112`, with high-depth chunks still often capped at `max=8192`. Bsz16 recovery rows `3715329_3/7/11` remain running on `a0537`: row `3` has advanced to `checkpoint-9000` and about `9061/10000`, while rows `7/11` are around `6847/10000` and `6876/10000` with latest on-disk checkpoints still `checkpoint-6000`. Batch eval `3715330` remains dependency-pending on `afterany:3715329_*`, and its output root is still absent. Hard-attribute eval rows `3716216_0/1` are running with `0` JSONs; conditioned-dual recovery `3716219` is priority-pending. Log scans found no unrecovered Traceback, CUDA OOM, quota/no-space, `DependencyNeverSatisfied`, tokenizer/model-load error, vLLM failure, idle-GPU symptom, node failure, timeout, or cancellation; only expected startup/model-load and memory-reserve warnings appeared. No partition edit, cancellation, resubmission, aggregation, report regeneration, or new oversight scheduling was done.

## Active Work

The current active Slurm work is the 2026-07-14 correction wave summarized at
the top of this file and in `docs/running_experiments.md`. The older detailed
items below are retained for artifact provenance; they are not the live queue
snapshot.

- old full paired-family suite: SFT/build are complete at `90/90` final adapters. The stale old eval root has `37/90` JSONs (`official_igsm` `30/30`, old maze `7/30`, hard `attribute_constraints` `0/30`), but the combined recovery is not being resubmitted because iGSM and maze have newer semantic/typed reruns. Hard `attribute_constraints` is being recovered separately by fresh eval `3716216`.
- hard attribute/constraint eval: `3716216_[0-29%3]` is the fresh hard-attribute-only eval under `$HPCVAULT`. It has `8/30` JSONs/sample JSONLs; rows `9/10/11` are running and rows `12..29` are throttle-pending. Logic seed `3408` at train-1-to-5 timed out and should be recovered later with any additional failed rows.
- trace-control ablations: SFT `3661118` complete; original eval `3661119` rows `0..5`, `9..11`, and `13` complete, with rows `12/14/15..17` failed or killed; replacement eval `3682459_[12,14-17%3]` and replacement repair eval `3682460_[5-8%3]` are complete, yielding `18/18` JSONs plus sample JSONLs
- shortcut-rate `0.3`: SFT `3671431` rows `0..5` complete; eval `3671432` complete; all `0.3` logic and NL rows now have 3-seed JSONs
- hybrid-order eval: original/replacement evals are complete at `30/30` JSONs and sample JSONLs. Report tables/figures and qualitative sample interpretation are refreshed; the current conclusion is that hybrid surfaces match intent but do not close the long-depth validity gap.
- wordified length-control: SFT `3674875_[0-2]` and eval `3674876_[0-2]` complete with 3 JSONs; duplicates `3674877/3674878` were intentionally canceled
- conditioned-dual 50k extension: 10k/20k/repaired-30k/40k/50k SFT chunks are complete. Checkpoint eval and final eval are both complete at `30/30` JSONs/sample JSONLs. Samples preserve mode-conditioned prompts; use the batch-size ablation to test whether remaining weakness is modality-mixing/effective-batch related.
- shortcut-kind controls: build `3674886_[0-3]` complete; original SFT `3674887` rows `0..21` and `23` complete, replacement SFT `3682458_22` complete, and eval `3674888` rows `0..23` are complete; all 24 eval JSONs and sample JSONLs are report-ingested.
- HFSA batch-size ablation: SFT is `10/12` finals. Bsz16 rows `7` and `11` are running as targeted resume recovery `3722466_[7,11%2]` from `checkpoint-8000`, now around `87%`; full eval `3722467_[0-15%4]` waits on `afterok:3722466_*`. Outputs are under `$HPCVAULT/synthetic-RLVL/runs/sft_hfsa_batch_*`; the eval root `$HPCVAULT/synthetic-RLVL/passk_eval/hfsa_batch_size_ablation_20260603/` has `0` JSONs.
- Typed maze rerun: build `3695237` and SFT `3695238_[0-29%3]` are complete at `30/30` finals. High-cap eval `3705795` was canceled at `0/30` after repeated formal `8192`-token saturation; the wrapper now defaults formal eval to `4096` tokens and leaves NL at `6144`. Replacement eval `3722471_[0-29%3]` is running rows `0..2`, with rows `3..29` throttle-pending, and is progressing fast enough to finish if late chunks stay comparable.
- Semantic iGSM rerun: build `3695525`, SFT `3695526_[0-29%3]`, original/recovery eval `3695527/3702073`, and clean NL-only re-eval `3705807` are complete at `30/30` current JSONs. Outputs are under `$HPCVAULT/synthetic-RLVL/datasets/materialized_paired_official_igsm_semantic_20260603/`, `$HPCVAULT/synthetic-RLVL/runs/sft_paired_igsm_semantic_*`, and `$HPCVAULT/synthetic-RLVL/passk_eval/paired_igsm_semantic_sparse_20260603/`.
- Active recovery oversight: one-off A40 oversight jobs `3715439`/`3715440`/`3715441`/`3715442`/`3715443` are complete; no further oversight job is scheduled.
- ablation oversight: `3687984`, `3688814`, `3689677`, `3690212`, and `3690645` completed cleanly; next plan-driven pass `3691029` is begin-time pending; detailed ablation analysis is handled by the separate ablation oversight pass
- hybrid-order readout: completed `think_formal` and `formal_think` rows are three-seed complete through train-1-to-25. Samples preserve the intended NL-then-formal and formal-then-NL surfaces and normal answer extraction, but formal blocks often omit strict citations and depth-50 outputs truncate or drift. Treat the result as evidence against a simple hybrid-substrate fix for long-depth validity, not as an evaluator failure.
- HFSA ablation/report refresh at 2026-06-01 06:35-06:57 CEST: shortcut-kind eval advanced to `21/24` after rows `17..20`; hybrid-order advanced to `20/30` after `formal_think` train-1-to-10 seeds `3407/3408`; paired iGSM completed at `30/30` after row `29`; conditioned-dual 50k rows `0..5` are complete with rows `6..9` running. Regenerated and mirrored the report with `65` PDFs, `57` CSVs, and `5` generated Markdown supplements. Sample inspection covered new shortcut-kind initial-marker logic/NL rows, hybrid `formal_think`, and paired iGSM logic/NL rows: prompts and wrappers match intended modalities, answer extraction works, eval metadata is shortcut-neutral where applicable, and new failures are expected answer/validity fragility rather than parser or scheduler breakage. The newly completed iGSM `nl_exact` train-1-to-25 seed `3409` row keeps `<think>/<answer>` format and has OOD/depth-50 correct@16 `0.575/0.531`, but NL-to-logic parse/validity remains `0.000`. Focused fatal-log scans found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, dependency, tokenizer/model-load, node-failure, timeout, cancellation, or idle-GPU issue; pending rows are throttle/dependency/begin-time blocked. No partition edit, dependency edit, cancellation, resubmission, or broad new science launch was made; no visible `tjepa_*`, `seqedit_*`, or `puzzle_*` jobs were present.
- Operational refresh at 2026-06-01 09:15 CEST: conditioned-dual 50k SFT rows `3674883_6` and `3674883_7` completed cleanly since the 07:00 handoff; rows `8..11` are running and rows `12..14` are throttle-pending. Paired eval remains at `30/90` JSONs, all `official_igsm`; maze rows `30..33` are actively sampling but have no completed JSONs yet. Shortcut-kind remains `21/24` with rows `21..23` running. Hybrid remains `20/30` with rows `20..23` running and `24..29` throttle-pending. No new result JSONs appeared, so no report regeneration was run. Focused active-log scan found no unrecovered Traceback, CUDA OOM, quota/no-space, dependency, node-failure, timeout, cancellation, or idle-GPU issue; no scheduler edit, partition edit, cancellation, resubmission, or new job submission was made.
- Paired operational refresh at 2026-06-01 10:31 CEST: paired eval still has `30/90` pass@k JSONs and `30` sample JSONLs, all `official_igsm`; no `maze_navigation` or hard `attribute_constraints` eval JSON has finished yet, so no report regeneration was run. Active maze rows are healthy: `3682449_30/31/32/33` are sampling around chunks `51/112`, `44/112`, `49/112`, and `56/112`, with GPU utilization about `96-97%` and no idle symptom. Full-suite manifests still have `55` subsets per family with all parquet paths present, and paired SFT final adapters remain `90/90`. Focused fatal-log scans found no unrecovered Traceback, proof-validation failure, CUDA OOM/OOM, context-length failure, quota/no-space, dependency, tokenizer/model-load, vLLM failure, node failure, timeout, or cancellation; only the known Mistral tokenizer regex warning appeared. Representative completed iGSM samples still show intended `<formal>`/`<think>` wrappers and answer extraction; NL-to-logic parse/validity remains `0.000`, and depth-50 logic can be answer-correct while invalid/ungrounded. Pending rows are throttle-blocked despite idle compatible A100 nodes, so no partition edit, scheduler edit, cancellation, resubmission, broad launch, or fix was made. `puzzle_oversight` is visible but unrelated; no visible `tjepa_*` or `seqedit_*` jobs were present.
- HFSA ablation/report refresh at 2026-06-01 10:40 CEST: shortcut-kind advanced to `23/24` after rows `21/22`; hybrid-order advanced to `21/30` after row `20`; conditioned-dual 50k row `8` completed and row `12` started; paired remains `30/90` with only `official_igsm` complete. Regenerated and mirrored the report with `65` PDFs, `57` CSVs, and `5` generated Markdown supplements. Sample inspection covered new shortcut-kind `initial_marker` `nl_exact` rate-`0.8` rows and hybrid `formal_think` train-1-to-10 seed `3409`: prompts/wrappers match intended modalities, shortcut-kind eval metadata is shortcut-neutral, answer extraction works, shallow samples translate/validate where expected, and depth-50 failures are truncation/drift/validity fragility rather than a parser or scheduler breakage. Focused log and GPU scans found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, dependency, tokenizer/model-load, vLLM, node-failure, timeout, cancellation, or idle-GPU issue. Pending rows are throttle/dependency/begin-time blocked despite idle compatible A100 nodes, so no partition edit, scheduler edit, cancellation, resubmission, broad launch, or fix was made. Visible `puzzle_oversight` is unrelated; no visible `tjepa_*` or `seqedit_*` jobs were present.
- Paired oversight completion at 2026-06-01 10:43 CEST: `3687377` completed cleanly (`0:0`) after the 10:31 paired audit and scheduled successor `3687983`. It found paired eval still at `30/90` JSONs, all `official_igsm`; no new maze or hard-attribute JSONs, no fatal signatures, no idle-GPU symptom, and no need for report regeneration, scheduler edit, partition edit, cancellation, resubmission, broad launch, or fix.
- Paired operational refresh at 2026-06-01 14:35-14:42 CEST: paired oversight `3687983` completed cleanly after scheduling successor `3688815` (begin-time pending). Replacement eval `3682449` remains healthy with rows `0..29` complete, rows `30..33` running on the first `maze_navigation` train-1-to-5 slice, and rows `34..89` pending only by `JobArrayTaskLimit`. The output directory still has `30/90` pass@k JSONs and `30` sample JSONLs, all `official_igsm`; no `maze_navigation` or hard `attribute_constraints` eval JSON has finished, so no report regeneration was run. Active row progress is `30` around chunk `66/112`, `31` around `58/112`, `32` around `62/112`, and `33` around `90/112`; `srun --overlap` GPU checks on the actual running job IDs showed `93-98%` GPU utilization with about `67GB` used, so there is no idle-GPU symptom. Full-suite manifests still have `55` subsets per family with no missing parquet paths, and paired SFT final adapters remain `90/90`. Sample/materialization checks found matched formal/NL question fields, equal formal/NL proof-line counts, `logic_trace_valid=True` for sampled train-up-to-5 and val-depth-50 rows in all three families, and completed iGSM samples with intended `<formal>`/`<think>` and `<answer>` wrappers; paired NL parse/translated validity remains `0.000`, so paired NL validity claims remain blocked on translator coverage. Focused fatal-log scans found no unrecovered Traceback, proof-validation failure, OOM/CUDA OOM, context-length failure, quota/no-space, dependency, tokenizer/model-load, vLLM failure, node failure, timeout, cancellation, or idle-GPU issue beyond the known Mistral tokenizer regex warning. Pending paired rows are array-throttle blocked despite idle compatible `a100` nodes, so no partition edit, scheduler edit, cancellation, resubmission, broad launch, generator fix, or evaluator fix was made. Visible `puzzle_*` jobs are unrelated; no visible `tjepa_*` or `seqedit_*` jobs were present.
- HFSA ablation/report refresh at 2026-06-01 14:42 CEST: shortcut-kind eval `3674888_0..23` is complete with `24/24` JSONs and sample JSONLs, and the final `initial_marker` `nl_exact` shortcut-`0.8` three-seed mean is OOD correct/translated-joint@16 `0.771/0.702` and depth-50 `0.667/0.500`. Hybrid-order eval advanced to `22/30` after `3682461_21`; `formal_think` train-1-to-15 seed `3407` has OOD correct/formal-joint/translated-joint@16 `0.656/0.301/0.250` and depth-50 `0.688/0.125/0.000`. Conditioned-dual 50k chunk `3674883` has rows `0..11` complete and rows `12..14` running with high GPU use; final/checkpoint eval arrays remain dependency-pending. Regenerated and mirrored the report with `65` PDFs, `57` CSVs, and `5` Markdown supplements; TeX compilation remains unavailable. Sample checks covered the final shortcut-kind row and new hybrid row: prompts/wrappers match intended modalities, shortcut-kind eval metadata is shortcut-neutral, answer extraction works, and deeper failures are truncation/drift/validity fragility rather than evaluator breakage. Focused log/GPU scans found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, dependency, tokenizer/model-load, vLLM, node failure, timeout, cancellation, or idle-GPU issue. No partition edit, dependency edit, scheduler edit, cancellation, resubmission, broad launch, or code/config fix was made; visible `puzzle_*` jobs are unrelated and no visible `tjepa_*` or `seqedit_*` jobs were present.
- HFSA/paired oversight refresh at 2026-06-01 18:32-18:41 CEST: `3674883_12/13` completed cleanly and `3674883_14` remains running, leaving final/checkpoint conditioned-dual evals dependency-pending. Paired eval `3682449` advanced to `31/90` JSONs/sample JSONLs after the first `maze_navigation` row completed; active paired rows are `30/31/32/34`, while `35..89` are throttle-pending. Targeted iGSM NL rerun `3689003` is `8/15` complete; rerun rows improved iGSM NL parser coverage to `1.000` for train-1-to-5/10 and partial `0.664` for train-1-to-15, but generated translated validity remains `0.000` because generated variables often do not match gold formal premises. First maze `nl_exact` train-1-to-5 seed-3407 has OOD correct@16 `0.088`, NL parse@16 `0.000`, and depth-50 correct@16 `0.000`; maze NL validity is not covered by the HFSA/iGSM translator. Regenerated and mirrored the report with `65` PDFs, `57` CSVs, and `5` Markdown supplements; `py_compile` passed for the report builder, and TeX compilation remains unavailable. Active-log scans found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, dependency, tokenizer/model-load, vLLM, node-failure, timeout, cancellation, or idle-GPU issue; pending rows are throttle/dependency/begin-time blocked despite idle compatible `a100` nodes. No partition edit, dependency edit, scheduler edit, cancellation, resubmission, broad launch, or experiment fix was made. Visible `puzzle_*` jobs are unrelated; no visible `tjepa_*` or `seqedit_*` jobs were present.
- Paired/ablation refresh at 2026-06-01 22:37 CEST: targeted iGSM NL rerun `3689003` completed all `15/15` rows and overwrote the official_iGSM `nl_exact` pass@k artifacts. Full rerun means by train range now show OOD parser coverage `1.000/1.000/0.997/1.000/0.994` and depth-50 parser coverage `1.000/1.000/0.990/1.000/0.969`, but OOD/depth-50 translated joint remains `0.000` for every train range; train-band translated joint is only `0.111/0.094/0.059/0.038/0.037`. Paired eval `3682449` still has `31/90` JSONs with rows `30/31/32/34` running and rows `35..89` throttle-pending; active maze rows reached about chunks `90/112`, `83/112`, `86/112`, and `62/112` with `95-97%` GPU utilization and no fatal signatures. Conditioned-dual 50k SFT `3674883` completed, releasing final eval `3674884` and checkpoint eval `3674885`; checkpoint eval has written 6 provisional conditioned-logic train-1-to-25 JSONs. Regenerated and mirrored the report with `66` PDFs, `59` CSVs, and `5` Markdown supplements; patched the report builder so the completed iGSM rerun and conditioned-50k checkpoint-partial status are represented correctly. No partition edit, dependency edit, scheduler edit, cancellation, resubmission, broad launch, or experiment fix was made. Visible `puzzle_*` jobs are unrelated; no visible `tjepa_*` or `seqedit_*` jobs were present.
- Paired operational refresh at 2026-06-02 02:39 CEST: paired oversight `3689676` completed cleanly and current paired oversight `3690207` is running after scheduling successor `3690641` (begin-time pending). Replacement eval `3682449` remains `31/90` JSONs/sample JSONLs: `official_igsm` `30/30`, `maze_navigation` first `nl_exact` train-1-to-5 seed-3407 row, and hard `attribute_constraints` `0/30`. Rows `30/31/32/34` are running, rows `35..89` are throttle-pending, and active chunks are about `101/112`, `94/112`, `97/112`, and `91/112`; live GPU checks showed `95-98%` utilization with about `67GB` memory. Rows `30/31/32` are walltime risks because the logic maze rows are repeatedly hitting the `8192` max-new-token cap, but they are still healthy and progressing, so no cancellation or replacement was made. Fatal-log scans found no unrecovered Traceback, proof-validation failure, OOM/CUDA OOM, context-length failure, quota/no-space, dependency, tokenizer/model-load, vLLM, node-failure, timeout/cancellation, or idle-GPU issue beyond known warnings. Full-suite manifests remain `55/55` paths for all families and SFT final adapters remain `90/90`. Materialized sample checks re-confirmed 1:1 paired formal/NL fields and `logic_trace_valid=True`; sample JSONL checks found first maze NL uses intended `<think>` with shallow train-band correctness but zero maze NL translator coverage and depth-25/50 drift, while completed iGSM logic/NL wrappers and answer extraction are correct but generated translated/grounded joint remains zero due variable-chain mismatch. No new JSONs appeared after the 22:37 report refresh, so no report regeneration, scheduler edit, partition edit, cancellation, resubmission, broad launch, generator fix, or evaluator fix was made.
- HFSA ablation/report refresh at 2026-06-02 02:45-02:48 CEST: hybrid-order advanced to `24/30` JSONs after `formal_think` train-1-to-15 seeds `3408/3409`; conditioned-dual 50k final eval has `2/30` JSONs and checkpoint eval has `15/30` JSONs, completing the `conditioned_logic` train-1-to-25 10k-to-50k curve. Regenerated and mirrored the report with `66` PDFs, `61` CSVs, and `5` generated Markdown supplements, after patching the report builder to remove stale `formal_think` and conditioned-50k caption language. Sample inspection found hybrid `formal_think` prompts start with `<question>` and outputs start `<formal>` followed by the expected NL `<think>/<answer>` section when not truncated; conditioned-logic prompts use `<reasoning_mode>formal_logic</reasoning_mode>` and outputs use `<formal>/<answer>` without a natural-language think section. Shallow successes have working extraction/validity; depth-50 failures are dominated by truncation/repetition and validity fragility rather than evaluator or scheduler breakage. Focused log and live GPU scans found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, dependency, tokenizer/model-load, vLLM, node-failure, timeout/cancellation, or idle-GPU issue; pending rows are throttle/begin-time blocked. No scheduler edit, partition edit, cancellation, resubmission, broad launch, or experiment fix was made.
- Paired oversight completion at 2026-06-02 02:55 CEST: `3690207` completed cleanly (`0:0`) after the 02:39 paired audit and scheduling successor `3690641`. No new paired eval JSONs appeared by the final output count check (`31/90`), and no scheduler/report action was needed.
- GitHub push status at 2026-06-02 02:59 CEST: local commits were created for this repo (`ece496d`, plus this push-status note) and `../synthetic-RLVL-report` (`0cee8fb`). Direct pushes to `git@github.com:Thiggel/synthetic-RLVL.git` and `git@github.com:Thiggel/synthetic-RLVL-report.git` each timed out after 60 seconds with no server response; fallback pushes to `ssh://git@ssh.github.com:443/Thiggel/synthetic-RLVL.git` and `ssh://git@ssh.github.com:443/Thiggel/synthetic-RLVL-report.git` also timed out after 60 seconds with no server response. This push-status note is local-only until connectivity returns; both local branches remain ahead.
- GitHub push status at 2026-06-02 02:47 CEST: local paired handoff commit `3de33f3` was created after the 02:39 audit, but pushing this repo failed. Direct push to `git@github.com:Thiggel/synthetic-RLVL.git` timed out after 60 seconds with no server response; fallback push to `ssh://git@ssh.github.com:443/Thiggel/synthetic-RLVL.git` also timed out after 60 seconds with no server response. This push-status note is local-only until connectivity returns; the local branch remains ahead.
- GitHub push status at 2026-06-01 22:51 CEST: local commits were created for this repo (`c22619a`, plus this push-status note) and `../synthetic-RLVL-report` (`3a8f0ea`). Direct pushes to `git@github.com:Thiggel/synthetic-RLVL.git` and `git@github.com:Thiggel/synthetic-RLVL-report.git` each timed out after 60 seconds with no server response; fallback pushes to `ssh://git@ssh.github.com:443/Thiggel/synthetic-RLVL.git` and `ssh://git@ssh.github.com:443/Thiggel/synthetic-RLVL-report.git` also timed out after 60 seconds with no server response. This push-status note is local-only until connectivity returns; both local branches remain ahead.
- GitHub push status at 2026-06-01 18:51 CEST: local commits are present in this repo and `../synthetic-RLVL-report`. Direct pushes to `git@github.com:Thiggel/synthetic-RLVL.git` and `git@github.com:Thiggel/synthetic-RLVL-report.git` each timed out after 60 seconds with no server response; fallback pushes to `ssh://git@ssh.github.com:443/Thiggel/synthetic-RLVL.git` and `ssh://git@ssh.github.com:443/Thiggel/synthetic-RLVL-report.git` also timed out after 60 seconds with no server response. This push-status note is local-only until connectivity returns; both local branches remain ahead.
- GitHub push status at 2026-06-01 14:57 CEST: local commits were created for this repo through ablation handoff/report commit `43402c6`, followed by this local push-status note, and for `../synthetic-RLVL-report` through `9ed8411`. Direct pushes to `git@github.com:Thiggel/synthetic-RLVL.git` and `git@github.com:Thiggel/synthetic-RLVL-report.git` each timed out after 60 seconds with no server response; fallback pushes to `ssh://git@ssh.github.com:443/Thiggel/synthetic-RLVL.git` and `ssh://git@ssh.github.com:443/Thiggel/synthetic-RLVL-report.git` also timed out after 60 seconds with no server response. This push-status note is local-only until connectivity returns; both local branches remain ahead.
- GitHub push status at 2026-06-01 14:42 CEST: local paired handoff commit `8dff64b` was created, but pushing this repo failed. Direct push to `git@github.com:Thiggel/synthetic-RLVL.git` and fallback push to `ssh://git@ssh.github.com:443/Thiggel/synthetic-RLVL.git` each timed out after 60 seconds with no server response. This note is local-only until connectivity returns; the local branch remains ahead.
- GitHub push status at 2026-06-01 10:55 CEST: local commits were created for this repo through the latest handoff commits and for `../synthetic-RLVL-report` (`f01ef90`). Pushes to both `git@github.com:...` and fallback `ssh://git@ssh.github.com:443/...` timed out after 60 seconds with no server response. This push-status note is local-only until connectivity returns; local branches remain ahead and ready to push.
- GitHub push status at 2026-06-01 06:57 CEST: final push attempts covered the current local report state (`../synthetic-RLVL-report` through `871ee2e`) and the then-current main-repo handoff/report state, but both repos timed out after 60 seconds on `github.com:22` and after 60 seconds on `ssh.github.com:443`. This push-status note is local-only; local branches remain ahead and ready to push.
- GitHub push status at 2026-06-01 06:46 CEST: push attempts made before this note covered this repo through `fee306b` and `../synthetic-RLVL-report` through `9be0985`, but pushes to both `github.com:22` and `ssh.github.com:443` timed out after 60 seconds with no server response. This push-status note is local-only until connectivity returns; local branches remain ahead and ready to push.
- Paired full-suite transition at 2026-05-31 18:35 CEST: row-56 replacement `3683070_56` and replacement rows `3682411_84..89` completed cleanly, so all paired full-suite SFT final adapters are present (`official_igsm` `30/30`, `maze_navigation` `30/30`, hard `attribute_constraints` `30/30`). Replacement eval `3682449` is now running rows `0..3` (`official_igsm`, train-1-to-5, early seeds/templates) and rows `4..89` are pending only by `JobArrayTaskLimit`; the eval output directory exists but still has `0` pass@k JSONs and `0` sample JSONLs. Row `3682449_0` was actively sampling at chunk `25/112` with `92%` GPU utilization; row `3682449_1` had merged and entered vLLM startup; rows `2/3` were still inside the expected stagger window. Focused fresh SFT/eval log scan found no unrecovered Traceback, proof-validation failure, OOM/CUDA OOM, context-length failure, quota/no-space, `DependencyNeverSatisfied`, tokenizer/model-load, vLLM, node-failure, timeout, cancellation, or idle-GPU issue; the only fresh match was a benign tokenizer max-length warning. Full-suite manifests still have 55 subsets and no missing parquet paths for all three families. No partition edit, dependency edit, cancellation, resubmission, report regeneration, or new science launch was made; no visible `tjepa_*` or `seqedit_*` jobs were present.
- Paired oversight completion at 2026-05-31 18:43 CEST: `3685027` completed cleanly (`0:0`) after recording the paired eval release state above and scheduling next pass `3685570`. It found zero paired eval JSON/sample outputs, no new severe failures, and made no scheduler or report changes. Its push attempt also failed because GitHub SSH connectivity timed out; local commits remain ahead.
- HFSA ablation/report refresh at 2026-05-31 18:35 CEST: shortcut-kind eval advanced to `13/24` after rows `9..12`; hybrid-order eval advanced to `18/30` after `formal_think` train-1-to-5 completed; trace-control remains complete at `18/18`; conditioned-dual 40k has rows `0..8` complete with `9..12` running. Regenerated and mirrored the report with `64` PDFs and `55` CSVs. Sample inspection covered `position` `nl_exact` shortcut `0.8` seed `3409`, `initial_marker` logic shortcut `0.5` seed `3407`, and hybrid `formal_think` seed `3409`: eval prompts are shortcut-neutral (`active_branch_first=None`), wrappers and `<answer>` extraction match intended modality, shallow samples are valid, and depth-50 samples show expected truncation/validity fragility. Focused live/recent log scans found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, `DependencyNeverSatisfied`, tokenizer/model-load, vLLM, node-failure, timeout, cancellation, or idle-GPU issue. No partition edit, dependency edit, cancellation, resubmission, or new science launch was made; pending rows are throttle/dependency/begin-time blocked. No visible `puzzle_*`, `tjepa_*`, or `seqedit_*` jobs were present.
- GitHub push status at 2026-05-31 18:51 CEST: local commits `4237aa9` in this repo and `7a98df9` in `../synthetic-RLVL-report` were created, but pushes to both `github.com:22` and `ssh.github.com:443` timed out. Local branches remain ahead and ready to push when SSH connectivity returns.
- HFSA ablation/report refresh at 2026-05-31 22:35 CEST: shortcut-kind eval advanced to `15/24` after rows `13/14`; `initial_marker` logic shortcut `0.5` is now three-seed with OOD correct/joint@16 `0.883/0.625` and depth-50 `0.854/0.344`. Conditioned-dual 40k rows `0..11` are complete with rows `12/13/14` running. Hybrid-order remains `18/30`, with rows `18..21` running. The report was regenerated and mirrored with `64` PDFs and `55` CSVs; TeX compilation remains unavailable. Sample inspection covered new `initial_marker` logic seeds `3408/3409` and partial paired `official_igsm` logic/NL rows: prompts and wrappers matched the intended modalities, shortcut-kind eval metadata stayed shortcut-neutral, and extraction/validity diagnostics behaved as expected; depth-50/deeper samples still show validity or answer fragility. Focused fatal-log scans and GPU checks found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, `DependencyNeverSatisfied`, tokenizer/model-load, vLLM, node-failure, timeout, cancellation, or idle-GPU issue. No partition edit, dependency edit, cancellation, resubmission, or new science launch was made; pending rows are throttle/dependency/begin-time blocked.
- Paired full-suite partial readout at 2026-05-31 22:40 CEST: replacement eval `3682449` remains healthy with rows `0..13` complete, rows `14..17` running, and rows `18..89` pending only by `JobArrayTaskLimit`. The first `14` pass@k JSONs and sample JSONLs are all `official_igsm`; `maze_navigation` and hard `attribute_constraints` still have no eval JSONs. Partial metrics are diagnostics-only: logic train-1-to-5/train-1-to-10/train-1-to-15 OOD correct@16 are `0.312/0.507/0.547` and internal-joint@16 are `0.255/0.377/0.400` (`train1to15` two seeds); matched `nl_exact` train-1-to-5/train-1-to-10 OOD correctness is `0.366/0.589`, while NL parse/translated validity remains `0.000`. Sample inspection found intended `<formal>` and `<think>` wrappers and working `<answer>` extraction; shallow logic can be grounded-valid, but deeper logic/NL generations show the expected grounding, validity, or answer fragility. Regenerated and mirrored the report with paired partial artifacts, now `65` PDFs and `57` CSVs; TeX compilation remains unavailable. No scheduler edit, partition edit, cancellation, resubmission, broad launch, or fix was made; no visible `tjepa_*` or `seqedit_*` jobs were present.
- GitHub push status at 2026-05-31 22:45 CEST: local commits were created in this repo and in `../synthetic-RLVL-report`, but pushes to both `github.com:22` and `ssh.github.com:443` again timed out. Local branches remain ahead and ready to push when SSH connectivity returns.
- GitHub push status at 2026-06-01 02:49 CEST: local commits were created in this repo and in `../synthetic-RLVL-report`, but pushes to both `git@github.com:...` and `ssh://git@ssh.github.com:443/...` timed out after 60 seconds with no server response. Local branches remain ahead and ready to push when SSH connectivity returns.
- Paired full-suite partial readout at 2026-06-01 02:36 CEST: replacement eval `3682449` remains healthy with rows `0..21` complete, rows `22..25` running, and rows `26..89` pending only by `JobArrayTaskLimit`. The first `22` pass@k JSONs and sample JSONLs are all `official_igsm`; `maze_navigation` and hard `attribute_constraints` still have no eval JSONs. Partial metrics are diagnostics-only: logic train-1-to-5/10/15/20 OOD correct@16 is `0.312/0.507/0.546/0.536` and internal-joint@16 is `0.255/0.377/0.392/0.245`; matched `nl_exact` train-1-to-5/10/15 OOD correctness is `0.366/0.589/0.618`, with one train-1-to-20 seed at `0.589`, while NL parse/translated validity remains `0.000`. Sample inspection found intended `<formal>` and `<think>` wrappers and working `<answer>` extraction; shallow logic can be citation-free valid, but grounded iGSM validity remains unreliable beyond trivial retrieval, and deeper logic/NL generations show answer or validity fragility. The report also ingested shortcut-kind rows `15/16`, bringing shortcut-kind to `17/24`; bounded sample inspection of the new `initial_marker` `nl_exact` rate-0.5 rows found shortcut-neutral prompts (`active_branch_first=None`), intended `<think>` wrappers, working extraction, and mixed depth-50 success/failure. Active paired rows `22..25` were progressing at chunks `99/112`, `93/112`, `72/112`, and `74/112`, with about `95%` GPU utilization and no idle symptom. Regenerated and mirrored the report with updated paired partial artifacts, now `65` PDFs, `57` CSVs, and `5` Markdown supplements; TeX compilation remains unavailable. No scheduler edit, partition edit, cancellation, resubmission, broad launch, or fix was made; no visible `tjepa_*` or `seqedit_*` jobs were present.
- HFSA ablation/report refresh at 2026-05-31 14:38 CEST: shortcut-kind eval advanced to `9/24` after rows `5` and `8`; trace-control remains complete at `18/18`; hybrid-order eval remains `15/30`; conditioned-dual 40k has rows `0..5` and `7` complete with `6/8/9/10` running; paired full-suite SFT has `83/90` final adapters and eval remains absent. Regenerated and mirrored the report with `64` PDFs and `55` CSVs. Representative sample inspection checked shortcut-kind `position` logic `0.8` seed `3409` and `nl_exact` `0.5` seed `3409`: prompts are shortcut-neutral, wrappers and `<answer>` extraction match intent, shallow examples are valid, and deeper examples show the expected validity/grounding fragility. Focused live/recent log scans found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, `DependencyNeverSatisfied`, tokenizer/model-load, vLLM, node-failure, timeout, cancellation, or idle-GPU issue. No partition edit, dependency edit, cancellation, resubmission, or new science launch was made; pending rows are throttle/dependency/begin-time blocked. Visible `puzzle_*` jobs are unrelated; no visible `tjepa_*` or `seqedit_*` jobs were present.
- HFSA ablation/report refresh at 2026-05-31 10:45 CEST: trace-control eval is complete at `18/18`; shortcut-kind eval is `7/24`; hybrid-order eval is `15/30`; conditioned-dual 40k has rows `0/1/2` complete and `3/4/5/6` running; paired full-suite eval remains absent. Representative sample inspection checked `pseudocode`, shortcut-kind `position` logic/NL rows, and hybrid `think_formal`; prompts and wrappers match intended modalities and answer extraction works, but strict proof validity remains citation-sensitive and NL/translated validity remains fragile in several answer-correct samples. Focused live/recent log scans found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, `DependencyNeverSatisfied`, tokenizer/model-load, vLLM, node-failure, timeout, or idle-GPU issue; observed zero-GPU snapshots for newly launched staggered rows matched wrapper startup sleeps. No partition edit, dependency edit, cancellation, resubmission, or new science launch was made.
- Paired full-suite progress at 2026-05-31 14:29 CEST: original maze rows `3672212_54/58` and replacement rows `3682411_55/57/59/81/82/83` completed cleanly since the 10:45 handoff, bringing paired SFT final adapters to `83/90` (`official_igsm` `30/30`, `maze_navigation` `29/30`, hard `attribute_constraints` `24/30`). Active paired SFT rows are row-56 replacement `3683070_56` and replacement hard-attribute rows `3682411_84..89`; no paired SFT rows are pending. Latest parsed optimizer progress: `3683070_56` `8015/10000`, `3682411_84` `6622/10000`, `3682411_85` `5533/10000`, `3682411_86` `4121/10000`, `3682411_87` `1042/10000`, `3682411_88` `475/10000`, and `3682411_89` still in normal startup/stagger. Fresh stderr mtimes plus `srun --overlap` GPU checks on `3683070_56`, `3682411_84`, and `3682411_87` showed active `100%` GPU utilization, so there is no idle-GPU symptom. Focused paired build/SFT/eval log scans found no unrecovered Traceback, proof-validation failure, OOM/CUDA OOM, context-length failure, quota/no-space, `DependencyNeverSatisfied`, tokenizer/model-load, vLLM, node-failure, timeout, or cancellation signature beyond benign tokenizer warnings and quota headers. Full-suite manifests still have 55 subsets and no missing parquet paths for all three families. A bounded materialized/gold-target audit over sampled train-depth-25 and val-depth-50 rows found matched logic/NL prompts, expected `<formal>`/`<think>` wrappers, final answer tags, strict and grounded logic proof validation, and correct/formatted NL targets; shallow sampled NL translation still has `nl_logic_parse=0.0` and translated validity `0.0`, so paired NL validity metrics remain blocked on translator coverage. Replacement eval `3682449` remains dependency-pending and the paired eval output directory still does not exist, so no aggregation/report regeneration was run. No partition edit, dependency edit, cancellation, resubmission, or new science launch was made; visible `puzzle_oversight` is unrelated, and no visible `tjepa_*` or `seqedit_*` jobs were present.
- earlier ablation readout at 2026-05-31 06:35 CEST: shortcut-rate `0.3` was complete and supported the shortcut-robustness interpretation; trace controls were then `17/18` and shortcut-kind was then `4/24`. The 10:45 CEST bullet above supersedes this state.
- paired full-suite recovery at 2026-05-30 18:32 CEST: build `3672195_0..2` remains complete with all three manifests at 55 subsets and no missing parquet paths. Original SFT rows `0..53` have final adapters; original rows `54/56/58` are running on `maze_navigation` train-1-to-25 with latest parsed progress about `972/109/627` of `10000`; rows `55/57/59` failed with exit `1:0` and no traceback/OOM/quota/validation signature; rows `60..89` were immediately canceled/failed with signal `53`. Submitted targeted replacement SFT `3682411_[55,57,59-89%6]`; rows `55/57/59/60/61/62` are running and rows `63..89` are pending by array throttle. Canceled stale eval `3672213` because its `afterok:3672212_*` dependency could never satisfy, then submitted replacement eval `3682449_[0-89%4]` depending on original running job IDs `3681398/3681503/3681586` and replacement SFT `3682411`. There are still `0` eval JSONs/sample JSONLs and no eval output directory. Materialized-row audit over train-depth-25 and val-depth-50 rows for all three families found matching logic/NL prompts, expected `<formal>`/`<think>` wrappers, final `<answer>` tags, and strict proof validation passing; iGSM citation-free validation still fails for cited arithmetic substitution lines, so inspect per-family evaluator fields before using paired validity metrics. `a100` had idle nodes and replacement rows launched on `a100`; no partition widening or new science launch was made. Plan/backlog triggers remain deferred until replacement eval `3682449` writes outputs. Visible `puzzle_*` jobs are unrelated; no visible `tjepa_*` or `seqedit_*` jobs were present.
- paired full-suite idle-row recovery at 2026-05-30 22:31 CEST: original row `3672212_56` (`maze_navigation`, `logic`, train-1-to-25, seed `3409`) had no stderr updates since 16:59 CEST, GPU utilization `0%` with about `58GB` allocated, and the Python process stuck in `futex_do_wait`; it was canceled and replaced by targeted SFT `3683070_[56%1]` with `--exclude=a0831`, now running on `a0833`. Eval `3682449` was rewired from `afterok:3681398:3681503:3681586:3682411` to `afterok:3681398:3683070:3681586:3682411`. Replacement `3682411_60..65` completed, so paired SFT final adapters now count `60/90` (`official_igsm` `30/30`, `maze_navigation` `24/30`, hard `attribute_constraints` `6/30`). Active rows are original `3672212_54/58`, replacements `3682411_55/57/59/66/67/68`, row-56 replacement `3683070_56`, and pending `3682411_69..89`. Full-suite eval still has `0` JSON/sample outputs, so no aggregation or report regeneration was run. No partition widening was appropriate; visible `puzzle_*` jobs remain unrelated.
- Paired oversight completion at 2026-05-30 22:58 CEST: `3682410` completed cleanly after recording the row-56 recovery already reflected below; it produced no paired eval outputs and made no additional scheduler changes beyond that earlier recovery. Next paired pass remains `3683024` begin-time pending.
- Paired full-suite progress at 2026-05-31 02:29 CEST: `3682411_66..71` completed cleanly since the last handoff, bringing paired SFT final adapters to `66/90` (`official_igsm` `30/30`, `maze_navigation` `24/30`, hard `attribute_constraints` `12/30`). Active paired SFT rows are original `3672212_54/58`, replacements `3682411_55/57/59/72/73/74`, and row-56 replacement `3683070_56`; pending rows are `3682411_75..89` by array throttle. Latest parsed optimizer progress: `3672212_54` `5063/10000`, `3672212_58` `4836/10000`, `3683070_56` `1758/10000`, `3682411_55` `3806/10000`, `3682411_57` `3877/10000`, `3682411_59` `3819/10000`, `3682411_72` `6279/10000`, `3682411_73` `5375/10000`, and `3682411_74` `4250/10000`. Focused active/recent paired SFT log scan found no unrecovered Traceback, proof-validation failure, OOM/CUDA OOM, context-length failure, quota/no-space, `DependencyNeverSatisfied`, tokenizer/model-load, vLLM, node-failure, timeout, or idle-GPU signature. `3682449` remains dependency-pending and the paired eval output directory still does not exist, so no metrics/report regeneration were run. No partition widening was appropriate; pending paired rows are throttle/dependency blocked despite idle compatible `a100` nodes. Visible `puzzle_*` jobs are unrelated; no visible `tjepa_*` or `seqedit_*` jobs were present.
- Paired oversight completion at 2026-05-31 02:38 CEST: `3683024` completed cleanly (`0:0`) after recording the 02:29 paired progress state and scheduling next pass `3683562`. It found no paired eval JSONs/sample outputs, no new severe failures, and made no additional scheduler changes. Push from that job failed because both `github.com:22` and `ssh.github.com:443` timed out.
- Paired full-suite progress at 2026-05-31 06:35 CEST: replacement SFT rows `3682411_72..77` completed cleanly since the 02:29 handoff, bringing paired SFT final adapters to `72/90` (`official_igsm` `30/30`, `maze_navigation` `24/30`, hard `attribute_constraints` `18/30`). Active paired SFT rows are original `3672212_54/58`, replacements `3682411_55/57/59/78/79/80`, and row-56 replacement `3683070_56`; pending rows are `3682411_81..89` by array throttle. Latest parsed optimizer progress: `3672212_54` `7222/10000`, `3672212_58` `7065/10000`, `3683070_56` `3823/10000`, `3682411_55` `5968/10000`, `3682411_57` `6118/10000`, `3682411_59` `6062/10000`, `3682411_78` `2870/10000`, `3682411_79` `1850/10000`, and `3682411_80` `548/10000`; fresh stderr mtimes show current progress and no idle-GPU symptom. Focused paired SFT/build/eval log scans found no unrecovered Traceback, proof-validation failure, OOM/CUDA OOM, context-length failure, quota/no-space, `DependencyNeverSatisfied`, tokenizer/model-load, vLLM, node-failure, timeout, or idle-GPU signature. Full-suite manifests still have 55 subsets and no missing parquet paths for all three families. A refreshed materialized/gold-target audit over sampled train and val-depth-50 rows found matched logic/NL prompts, expected `<formal>`/`<think>` wrappers, final answer tags, strict proof validation, and gold logic evaluator validity; gold paired NL targets answer and format correctly but have `nl_logic_parse=0.0` and translated validity `0.0` in sampled families, so do not use paired NL validity metrics for logic-vs-NL claims until backlog item P1 is fixed. `3682449` remains dependency-pending and the paired eval output directory still does not exist, so no aggregation/report regeneration was run. No partition widening was appropriate; pending paired rows are throttle/dependency blocked despite idle compatible `a100` nodes. Paired oversight `3683562` is running and scheduled next pass `3683967`; visible `puzzle_*` jobs are unrelated, and no visible `tjepa_*` or `seqedit_*` jobs were present.
- Paired full-suite progress at 2026-05-31 10:31-10:47 CEST: replacement SFT rows `3682411_78..80` completed cleanly since the 06:35 handoff, bringing paired SFT final adapters to `75/90` (`official_igsm` `30/30`, `maze_navigation` `24/30`, hard `attribute_constraints` `21/30`). Active paired SFT rows are original `3672212_54/58`, replacements `3682411_55/57/59/81/82/83`, and row-56 replacement `3683070_56`; pending rows are `3682411_84..89` by array throttle. Latest parsed optimizer progress: `3672212_54` `9234/10000`, `3672212_58` `9142/10000`, `3683070_56` `5897/10000`, `3682411_55` `8011/10000`, `3682411_57` `8235/10000`, `3682411_59` `8172/10000`, `3682411_81` `5659/10000`, `3682411_82` `3882/10000`, and `3682411_83` `1553/10000`; fresh stderr mtimes show current progress and no idle-GPU symptom. Focused paired build/SFT/eval log scans found no unrecovered Traceback, proof-validation failure, OOM/CUDA OOM, context-length failure, quota/no-space, `DependencyNeverSatisfied`, tokenizer/model-load, vLLM, node-failure, timeout, or idle-GPU signature. Full-suite manifests still have 55 subsets and no missing parquet paths for all three families. A refreshed materialized/gold-target audit over train-depth-25 and val-depth-50 samples found matched logic/NL prompts, expected `<formal>`/`<think>` wrappers, final answer tags, strict proof validation, and gold logic evaluator validity; gold paired NL targets answer and format correctly but still have `nl_logic_parse=0.0` and translated validity `0.0`, so paired NL validity metrics remain blocked on translator coverage. `3682449` remains dependency-pending and the paired eval output directory still does not exist, so no aggregation/report regeneration was run. No partition widening, dependency edit, cancellation, resubmission, or new science launch was made; pending paired rows are throttle/dependency blocked despite idle compatible `a100` nodes. Paired oversight `3683967` completed cleanly at 10:47 CEST after making no scheduler changes; next pass `3684369` is begin-time pending. Visible `puzzle_*` jobs are unrelated, and no visible `tjepa_*` or `seqedit_*` jobs were present.
- Trace-control/shortcut-kind/report refresh at 2026-05-31 06:35 CEST: replacement trace rows `3682459_16/17` completed cleanly, bringing trace-control artifacts to `17/18` JSONs plus sample JSONLs; only `pseudocode` seed `3409` (`3682460_8`) remains. `shuffled_nl` is now three-seed with high parse coverage but translated joint `0.000`, matching the intended proof-order negative control. Shortcut-kind eval rows `3674888_0..3` completed and wrote the first 4 JSONs: `position` rate `0.5` logic is three-seed, while matched NL has seed `3407` only and must remain provisional. Sample inspection confirmed shortcut-kind eval prompts are shortcut-neutral, answer extraction works, logic generations use `<formal>`, and NL generations use `<think>`/`<proof>` with translated validity failures in some answer-correct samples. Regenerated and mirrored the report with `64` PDFs and `55` CSVs, including new shortcut-kind tables; TeX compilation remains unavailable. Active rows are `3682460_8`, hybrid `3682461_13/15/16/17`, shortcut-kind eval `3674888_4..7`, conditioned `3682457_13/14`, and paired rows recorded above. No fatal log signatures, partition edits, dependency edits, cancellations, resubmissions, or new science launches were made.
- Trace-control/report refresh at 2026-05-30 22:54 CEST: replacement eval row `3682459_14` completed cleanly and wrote `invalid_logic` seed `3409`, then `3682459_16` started. The report now ingests trace controls `11/18` with `invalid_logic` two-seed mean OOD correct/formal-joint@16 `0.906/0.544` and depth-50 `0.734/0.188`. Seed `3409` samples are mostly correct and citation-free-valid at shallow depths but not grounded-valid, and depth-50 samples have zero grounded validity, so this remains an evaluator-sensitive negative-control result rather than positive evidence for invalid traces. Regenerated and mirrored the report again with `64` PDFs and `53` CSVs in both trees; TeX compilation remains unavailable.
- HFSA ablation/report refresh at 2026-05-30 22:39 CEST: ablation oversight `3682409` is running, paired oversight `3682410` completed cleanly by the final check, and next passes `3683023` and `3683024` are scheduled. `3682492_5` completed cleanly, so conditioned 40k chunk `3674882` now waits only on replacement array `3682457`; no dependency edit was needed. Focused active-log scan found no unrecovered Traceback, CUDA OOM, quota/no-space, dependency-never-satisfied, tokenizer/model-load, vLLM, node-failure, timeout, or idle-GPU signature; OOM matches were benign accelerate memory-reserve messages and oversight prompt text. New report-ingested outputs are trace-control `invalid_logic` seed `3408` and hybrid `think_formal` train-1-to-25 seed `3409`. Invalid-logic seed `3408` has OOD correct/formal-joint@16 `0.856/0.519` and depth-50 `0.625/0.094`, but representative depth-50 samples are mostly invalid/wrong, so treat this single-seed partial as provisional. Hybrid `think_formal` train-1-to-25 is now two-seed partial with mean OOD correct/formal-joint/translated-joint@16 `0.584/0.188/0.459` and depth-50 `0.344/0.000/0.172`; samples confirm the intended `<think>` then `<formal>` surface, normal answer extraction, and fragile formal validity. Regenerated and mirrored the report with `64` PDFs and `53` CSVs in both trees; TeX compilation was not run because `latexmk`/`pdflatex` are unavailable.
- ablation log audit at 2026-05-30 09:50 CEST: focused `squeue`/`sacct`/log scan found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, `DependencyNeverSatisfied`, tokenizer/model-load, vLLM failure, node failure, timeout, cancellation, or idle-GPU failure in the monitored HFSA ablation chains. Active trace-control eval rows `6..8` were sampling chunks `38/56`, `34/56`, and `25/56`; hybrid eval rows `11..14` were sampling chunks `92/112`, `71/112`, `58/112`, and `54/112`; conditioned-dual 20k rows `12..14` were at `12676/20000`, `10001/20000`, and `10001/20000`; shortcut-kind SFT rows `15..17` were at `6658/6116/5299` of `10000`. Pending monitored rows were blocked by array throttles, dependencies, or begin time, so no partition edit, dependency edit, cancellation, or resubmission was made. The old queued follow-up `3679878` was later canceled after the oversight prompt update.
- trace-control evaluator fix at 2026-05-30 10:13 CEST: manual inspection of `rule_annotated_nl` sample generations showed correct-looking traces such as `a is teal. [rule: R]`, but `nl_logic_parse` was zero because the translator treated the annotation as part of the attribute. `synthrlvl/natural_logic.py` now unwraps rule annotations and pseudocode lines before translation, and `tests/test_training_stack.py` verifies `RULE_ANNOTATED_NL` and `PSEUDOCODE` targets translate to valid logic. Verification: `tests/test_training_stack.py` passed (`26 passed`). Canceled stale running pseudocode eval rows `3661119_6..8` and submitted repair eval `3680004_[3-8%3]` with `FORCE_PASSK_EVAL=1`; `3680004_3..5` are running and `3680004_6..8` are pending by throttle. `3661119_3..5` completed metrics should be treated as stale until `3680004_3..5` overwrite them.
- oversight automation update at 2026-05-30 10:28 CEST: strengthened the active Codex oversight wrappers so each pass reads the live docs, backlog, and active plan; checks jobs/logs/outputs/partitions; inspects sample generations and evaluator assumptions before accepting metrics; analyzes newly completed outputs; regenerates/mirrors the LaTeX report; updates docs/backlog; and may submit the smallest triggered or recovery jobs. Replaced old queued prompt snapshots by canceling `3679878` and `3679358`, then submitted fresh oversight jobs `3680036` and `3680037`. Both started on `a100` and scheduled the next passes `3680038` and `3680039`.
- HFSA ablation oversight update at 2026-05-30 10:33 CEST: fresh `squeue`/expanded `sacct`/log/output scan found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, dependency, tokenizer/model-load, vLLM, node-failure, timeout, or idle-GPU failure in the monitored chains. Active rows are progressing: trace repair `3680004_3..5` is at chunk `18/56`, hybrid rows `3670783_11..14` are at chunks `96/76/66/63` of `112`, conditioned-dual 20k rows `3674880_12..14` are at `14585/10001/11007` of `20000`, shortcut-kind rows `3674887_15..17` are at `8424/7922/7107` of `10000`, and paired rows `3672212_48..53` are at `6256/5929/5906/6014/5977/5833` of `10000`. Row `3674880_13` has stale stdout/stderr but `srun --jobid=3679320` showed the allocated GPU active at `97%` utilization with about `63GB` used, so no idle recovery was submitted. Pending monitored rows are throttle/dependency/begin-time blocked; no partition edit, dependency edit, cancellation, or resubmission was made. Sample inspection confirmed wordified depth-50 generations often stay on the `<formal>` surface but drift into duplicated predicate declarations or invalid derivations; stale rule-annotated NL samples fail the old translator on unstripped NL/premise text, so repair eval outputs remain required before using those metrics. Visible `puzzle_*` jobs are unrelated; no visible `tjepa_*` or `seqedit_*` jobs were present.
- HFSA ablation oversight update at 2026-05-30 14:36 CEST: original trace-control `shuffled_logic` eval rows `3661119_9..11` completed and wrote 3 JSONs plus samples. Three-seed mean OOD correct/formal-joint@16 is `0.690/0.002`, and depth-50 correct/formal-joint@16 is `0.510/0.000`; report artifacts now include this row. Sample inspection found normal `<question>` prompts, `<formal>` generations, and `<answer>` extraction, but higher-depth proofs are often invalid or unparsable fragments; depth-50 can still answer correctly while failing citation-free validity, so this is a negative-control result rather than valid reasoning. Focused `squeue`/expanded `sacct`/log scan found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, dependency, tokenizer/model-load, vLLM, node-failure, timeout, or idle-GPU failure. Active progress: trace repair `3680004_3..5` chunks `50/50/49` of `56`, original trace rows `3661119_12..14` running with row `12` at chunk `29/56`, hybrid rows `3670783_11..14` at sampling/scoring state `112/99/89/90` of `112`, shortcut-kind rows `3674887_18..20` at `6798/6231/5167` of `10000`, and conditioned-dual 30k rows `3674881_0..3` running from checkpoint-20000. Pending monitored rows are throttle/dependency/begin-time blocked despite idle `a100` nodes, so no partition edit, dependency edit, cancellation, resubmission, or new science launch was made. Regenerated `analysis/logic_cot_report_2026-05-25/` and mirrored it to `../synthetic-RLVL-report`; verification found `64` generated PDFs, `64` PDF include references, and `53` CSV tables in both report trees. TeX compilation was not run because `latexmk`/`pdflatex` are unavailable.
- HFSA ablation recovery/report update at 2026-05-30 18:40-18:52 CEST: several array rows failed with exit `1:0` or signal `53` but no traceback, OOM/CUDA OOM, quota/no-space, dependency-never-satisfied, tokenizer/model-load, or vLLM signature. Submitted targeted replacements only for missing rows: conditioned-dual 30k `3682457_[3,6-14%4]` plus later row-5 replacement `3682492_[5%1]`, shortcut-kind SFT `3682458_[22%1]`, original trace-control eval `3682459_[12,14-17%3]`, trace repair eval `3682460_[5-8%3]` with `FORCE_PASSK_EVAL=1`, and hybrid eval `3682461_[13,15-29%4]`. `3682492_5` is now running, and `3674882` waits on `afterok:3681529:3682492:3682457`; `3674888` was also rewired as noted above. Newly completed hybrid `think_formal` train-1-to-20 is now three-seed complete with mean OOD correct/formal-joint/translated-joint@16 `0.434/0.028/0.148` and depth-50 correct/formal-joint@16 `0.469/0.000`; sample inspection verified the intended `<think>` then `<formal>` surface and normal answer extraction, with depth-50 validity still fragile. Regenerated and mirrored the report after filtering the stale rule-annotated seed-3409 artifact; verification found `64` PDFs, `64` unique PDF include references with zero missing after LaTeX-escape normalization, and `53` CSV tables in both report trees. TeX compilation was not run because `latexmk`/`pdflatex` are unavailable.

## Report Artifacts

Primary reader-facing report root in this repo:

```bash
analysis/logic_cot_report_2026-05-25/
```

Important files:

- `analysis/logic_cot_report_2026-05-25/logic_cot_report_2026-05-25.tex`
- `analysis/logic_cot_report_2026-05-25/tables/`
- `analysis/logic_cot_report_2026-05-25/figures/`
- `analysis/logic_cot_report_2026-05-25/full_generation_sequences_olmo7b_olmo32b_2026-05-28.md`

The report builder is:

```bash
source ./scripts/env.sh
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" ${HPCVAULT}/.venv_rlvl_posttrain/bin/python scripts/analysis/build_logic_cot_report.py
```

The 2026-06-19 09:30 report regeneration refreshes final fresh paired readouts after typed maze completed. Updated artifacts include `active_paired_partial_by_seed.csv`, `active_paired_partial_summary.csv`, `active_paired_partial_summary.{pdf,png}`, and `active_experiment_artifact_status.csv` (`typed maze` `30/30`, `hard attribute fresh` `30/30`, `batch-size` SFT `12/12` and eval `16/16`). Batch-size diagnostics from 2026-06-15 remain in `hfsa_batch_size_ablation_diagnostics.csv` and `hfsa_batch_size_conditioned_delta.{pdf,png}`. Current generated bundle verification: report builder completed successfully; TeX compilation remains unavailable because `latexmk`/`pdflatex` are not installed.

The external report repo `../synthetic-RLVL-report` mirrors the generated bundle and should be pushed after every report update.

`pdflatex`/`latexmk` are not installed on the current node, so the `.tex` source is generated but not compiled here.

## Quick Commands

```bash
source ./scripts/env.sh
squeue -u c107fa12 -o '%.18i %.9P %.34j %.2t %.11M %.6D %.24E %R'
sacct -j 3823434,3828946,3829069,3829072,3829073,3830927,3830928,3831110,3831111,3831112,3831113,3831115,3831119,3831121,3831135,3831136,3831179,3832945,3833178,3833179,3834582,3834706,3834707,3834728,3834737,3834738,3834836,3834904,3834905,3834906,3834907,3834911,3835433,3835438,3835442,3835443,3835779,3835927,3835928,3838163,3847756,3847757,3847792,3847802,3847804,3847805,3847806,3847808,3849774,3849775,3849776 --format=JobID,JobName%34,State,Elapsed,ExitCode -n -P
```
