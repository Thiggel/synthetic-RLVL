# Running Experiments

Last updated: 2026-08-03 15:55 CEST.

This file is the live Slurm dashboard. Historical details live in `docs/operational_history_2026-05-29.md`; planned-but-not-running work lives in `docs/experiment_backlog.md`.

## Live Delta At 15:55 CEST On August 3

- Offload `3946030` completed cleanly and both post-move full checkpoint audits
  pass. Final epilogue usage is Vault 611.7 GB and Atuin 995.5 GB; both
  control/formal optimizer paths are valid `$WORK` symlinks.
- Replacement smoke `3945777_0` is running on four A100-80GB GPUs. Training
  reached `32/32`; eval loss is finite and improved `0.9334 -> 0.9202`. Full
  FSDP checkpoint serialization is actively progressing. Large checkpoint
  output is temporary and is removed only after the smoke audit is written.
- LR rows remain correctly dependency-held. Terminal NL `3875833_2` remains
  priority-pending with the current August 4 02:19 estimate; its matched eval
  remains dependency-held.

## Live Delta At 14:55 CEST On August 3

- Optimizer offload `3946030` is running cleanly and approximately 72%
  complete by logical bytes. Control is fully transferred and symlinked;
  logic has one complete shard and a mostly copied second shard. Vault is
  already `754G/1000G`, down from `949G` before cleanup/offload.
- Projected final `$WORK/synthetic-RLVL` footprint is about 238.7 GiB. Recent
  epilogues put the whole user Atuin tree near 806 GB before the move; adding
  about 183 GB projects approximately 989 GB total, safely below the user's
  3 TB planning ceiling. The 906 TB filesystem-free figure is cluster-wide,
  not the user's allocation. Expected completion is roughly 20--35 minutes.
- Post-SFT smoke `3945777_0` remains correctly held on `afterok:3946030`.
  Terminal NL remains independent and priority-pending with the current
  August 4 02:19 estimate.

## Live Delta At 14:10 CEST On August 3

- Vault cleanup reclaimed 25,985,174,016 bytes and reduced user usage from
  `949G` to `924G/1000G`. Removed only completed/invalid project artifacts;
  all active Dolmino Nanosets, model shards, the live NL restart, accepted
  results, and final adapters remain protected.
- Optimizer offload `3946030` is running CPU-only on `a0605`. It moves the
  accepted control/formal step-9537 optimizer shards to `$WORK`, verifies
  exact file manifests and both full checkpoints, then leaves symlinks at the
  original paths. Expected additional Vault recovery is 365,583,141,888 bytes.
- Replacement smoke `3945777_0` now has `afterok:3946030`; its previous start
  estimate is superseded by this safety dependency. Terminal NL `3875833_2`
  remains independent and priority-pending, currently estimated for 23:50.
  NL terminal eval and LR-pilot dependencies remain valid.
- Detailed cross-project usage and cleanup gates are in
  `analysis/storage_audit_20260803.md`.

## Live Delta At 13:40 CEST On August 3

- Smoke `3944070_0` failed `126:0` after six minutes. Its full base-checkpoint
  audit and Nanotron-to-HF conversion passed; the first post-SFT launch then
  hit a stale `torchrun` shebang pointing to a deleted Atuin Python. The wrapper
  now uses the verified posttrain venv Python with
  `-m torch.distributed.run`. `bash -n`, module-launch, and Slurm test-only
  checks pass.
- Submitted only the exact replacement smoke `3945777_0`, preserving the
  32-step, 1,024-train/128-eval, full-parameter four-A100-80GB protocol and
  guarded output cleanup. It is priority-pending with a current 19:02 start
  estimate. Rewired LR array `3944071_[0-1%2]` from the failed predecessor to
  `afterok:3945777:3875833_2`; it is dependency-pending, not dead.
- Terminal NL `3875833_2` remains dependency-free and priority-pending with a
  current 21:32 estimate. Terminal NL eval `3944017_2` remains held on it.
  Current watcher `3943787` is running CPU-only/no-GRES; recorded successor
  `3945763` remains BeginTime-pending and must be preserved.
- Smoke epilogue quota is `994.1G/1048.6G` user Vault soft usage and about
  `181k/200k` files; the project subtree is about `833G`. No artifact has a
  sufficiently proven safe-delete gate in this pass, so preserve all current
  checkpoints and use only the documented guarded cleanup paths.

## Live Delta At 10:20 CEST On August 3

- Terminal direct control/formal `3944016_0/1` completed `0:0` and passed all
  checkpoint, conversion, RoPE, task, symbolic-MATH, and retained-sample
  gates. Ten-task macros are `0.6246/0.6169`; formal improves every multi-hop
  cell, including tagged HotpotQA `0.4404 -> 0.4937` and MuSiQue
  `0.1810 -> 0.2210`. Raw review finds coherent GSM8K/MATH outputs and confirms
  untagged multi-hop continuation contamination. Partial audit:
  `analysis/nanotron_dolmino_terminal_partial_20260803.md`.
- Dolci preparation `3944069` completed `0:0` in 11 minutes. Deterministic
  fail-closed retention is 99,817 train and 2,044 eval examples; fingerprints,
  decoded native-Qwen chat, and assistant-only labels are recorded under
  `$HPCVAULT/synthetic-RLVL/analysis/dolci_control_sft_20260803`.
- Full-parameter smoke `3944070_0` is priority-pending, currently estimated
  for August 3 17:22. Terminal NL `3875833_2` is priority-pending, currently
  estimated for August 4 02:19. NL terminal eval `3944017_2` and LR pilot
  `3944071_[0-1%2]` remain dependency-held as designed. No row needs recovery.

## Live Delta At 09:30 CEST On August 3

- Terminal direct control/formal eval `3944016_0/1` is running on two A40s;
  both rows passed full Nanotron checkpoint verification, local conversion,
  and vLLM startup. NL row `3944017_2` is held by
  `afterok:3875833_2`. Output root:
  `$HPCVAULT/synthetic-RLVL/lm_eval_results/qwen25_dolmino_terminal_5b_20260803`.
- Deterministic Dolci subset/tokenization preparation `3944069` is running
  CPU-only/no-GRES. Full-parameter four-A100 smoke `3944070_0` depends on it;
  LR pilot `3944071_[0-1%2]` depends on both the smoke and terminal NL. The LR
  rows use `2e-6/5e-6`, effective batch 128, one epoch, 3% warmup, and the
  exact same pinned 100K data order. The smoke removes its final model after
  writing an acceptance audit.
- Terminal NL continuation `3875833_2` remains dependency-free and
  priority-pending. Its provisional start moved to August 4 02:19 CEST. No
  resource reduction is safe under the measured host-RSS gate.

## Live Delta At 08:35 CEST On August 3

- Terminal Dolmino NL continuation `3875833_2` is the only pending GPU job for
  this project. It is dependency-free, priority-pending, and Slurm projects a
  `12:58:36` CEST start on high-memory A100-80GB node `a0535`.
- Resource reduction is unsafe. Completed control/formal batch steps peaked at
  about 1.21--1.25 TiB RSS and the NL stage at about 1.05 TiB, so the idle
  approximately 1-TB A100 node cannot host this accepted TP4/DP2 run. Keep the
  eight-GPU, 1.875-TB request and exact step-5000 restart unchanged.
- No failed synthetic-RLVL row needs resubmission. Conditioned SFT timeout
  `3883534_14` was recovered by completed `3932131_14`; all six downstream
  evals passed. NL timeout `3875832_2` was the expected wall-time boundary and
  is already covered by `3875833_2`. Visible dead-dependency and `tj-*` rows
  are owned by other repositories and were left untouched.
- Current CPU-only watcher successor `3943787` remains BeginTime-pending for
  `13:32:24` CEST. It should verify startup/restart or safely recover the queue
  if the provisional start changes.

## Live Delta At 07:36 CEST On August 3

- Terminal NL continuation `3875833_2` remains dependency-free and
  priority-pending with the unchanged one-node TP4/DP2, eight-A100-80GB
  request. Slurm now projects `12:58:36` CEST on `a0535`; every healthy
  A100-80GB node is allocated or mixed, so there is no compatible beneficial
  widening or resource mutation.
- Its accepted step-5000 restart remains intact at 645 files, zero empty
  files, and `106,628,387,172` bytes. The accepted audit still records exact
  `5000/640000/2621440000` offsets and exact 95:5 token accounting. No
  pending-job stdout/stderr exists. Vault is about `948G/1000G`, 182k files;
  group Work is about `471k/500k` files.
- Current watcher `3943558` is running CPU-only/no-GRES. Recorded successor
  `3943787` is dependency-free, CPU-only/no-GRES on `a100mig`, and scheduled
  for `13:32:24` CEST. It remains required because terminal training and the
  direct/post-SFT gate are incomplete. No broader grid or report trigger is
  open. Bounded main/report GitHub SSH pushes produced no response and exited
  `124`; the scoped handoff commit remains local.

## Live Delta At 01:36 CEST On August 3

- Matched NL step-5000 limited direct eval `3942598_2` completed `0:0` in
  `00:54:31` on A40 node `a1721`. Local HF conversion/finite-forward and
  legacy/modern RoPE-`1000000` gates pass. Production audits accept six
  multi-hop files/600 rows, 105 standard leaf files/10,600 rows, and the
  schema-v4 MATH sidecar with zero lost stock-exact positives.
- Matched-NL ten-task macro is `0.6327`, versus control `0.5932` and formal
  `0.6341`. Its stock/tagged/first-head multi-hop macros are
  `0.1067/0.3420/0.3863`, and stock continuation is `93%/94%/96%` across
  2Wiki/HotpotQA/MuSiQue, essentially formal `91%/94%/97%`. Representative
  correct/incorrect raw review confirms intact prompts and extraction,
  ordinary task errors, tagged-boundary adherence, and long repetitive
  MMLU-Pro invalid tails. This supports a shared long-document/full-loss
  mechanism, not a formal-modality advantage.
- Terminal NL continuation `3875833_2` remains priority-pending from the
  audited 645-file step-5000 restart, with a provisional 16:59 CEST start.
  No compatible partition widening is useful: it requires one full
  A100-80GB node and all healthy matching nodes are allocated. Vault is about
  `948G/1000G`, 182k files; group Work is about `471k/500k` files.
  Dependency-free CPU-only/no-GRES successor `3943558` remains scheduled for
  07:31 because terminal training and downstream/post-SFT gates are incomplete.
- No broader mixture grid, duplicate BranchProof family, report regeneration,
  or official-preprint update was triggered. Detailed intermediate audit:
  `analysis/nanotron_dolmino_step5000_intermediate_20260730.md`.

## Live Delta At 20:05 CEST On August 2

- Conditioned-32B NL eval row `3881781_17` completed `0:0` in `06:17:22`
  on four A100-80GB GPUs and passed the full 448-prompt, 16-generation,
  14-depth, 576-row artifact and representative raw-generation gates. Its
  row-scoped OOD greedy/pass@1/joint@1/pass@16/joint@16 is
  `0.9563/0.9383/0.8824/0.9938/0.9938`; depth-50 answer/joint pass@1 is
  `0.9375/0.9141`. Complete `7/112` logs have no cap hit. Audit:
  `$HPCVAULT/synthetic-RLVL/analysis/branchproof_selected_followups_audits_20260723/large_17.json`.
- The six-row conditioned OLMo-3-32B family is accepted. Formal versus natural
  OOD answer/joint pass@1 is `0.8501/0.7504` versus `0.7724/0.7505`, and
  answer/joint pass@16 is `0.9979/0.9771` versus `0.8875/0.8667`. Reports
  were refreshed only after the raw gate passed.
- Dolmino NL `3875832_2` reached step 5061 and ended in the expected 24-hour
  `TIMEOUT`. Step 5000 passes the full 645-file restart/RoPE/exact-offset gate
  at `5000/640000/2621440000`, with exact
  `2490368000 + 131072000` Dolmino/NL tokens. After acceptance, removed only
  steps 3500/4000/4500 (`319,885,161,515` bytes); retained step 5000 re-audits
  unchanged. Terminal continuation `3875833_2` is priority-pending.
- Matched NL step-5000 limited direct eval `3942598_2` started on A40 node
  `a1721`. Its local four-shard conversion, finite forward pass, and consumer
  RoPE `1000000` preflight passed; it is running the same reviewer and
  tagged/stock multi-hop limit-100 protocol as the accepted control/formal
  rows. Terminal continuation `3875833_2` remains priority-pending. Successor `3942568`
  remains dependency-free, CPU-only/no-GRES, and scheduled for 01:31 CEST
  August 3 because terminal training and downstream/post-SFT gates remain.

## Live Delta At 13:40 CEST On August 2

- Exact from-base Dolmino NL `3875832_2` is healthy beyond step 3861 at about
  31.1K tokens/s with finite diagnostics. Its complete step-3500 state passed
  the 645-file restart gate with Qwen2.5 RoPE `1000000`, exact
  `3500/448000/1835008000` offsets, and exact
  `1743257600 + 91750400` Dolmino/NL token accounting. After acceptance,
  removed only superseded steps 2500/3000 (`213,256,774,342` bytes); retained
  step 3500 re-audits unchanged. Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_nl_exact_step3500_20260802.json`.
  User Vault is about `1068G/1000G`, `182k/200k` files, so guarded rotation
  remains required and step 5000 must be preserved for the matched readout.
- Conditioned-32B NL eval row `3881781_17` is healthy on four A100-80GB GPUs
  at sampled chunk 98/112 after about 4h11m, with no fatal signature and the
  full 24-hour protocol unchanged. It is the sole remaining selected eval.
- Successor `3941585` is dependency-free, CPU-only/no-GRES, and scheduled for
  19:30 CEST. It remains required. No scheduler edit, new GPU submission, or
  report refresh is yet justified. Scoped commit `1783749` remains local after
  a silent bounded 90-second GitHub SSH push exited `124`.

## Live Delta At 07:35 CEST On August 2

- Conditioned-32B formal eval row `3881781_16` completed as raw job
  `3940286` `0:0` in `06:28:35` and passed the full 448-prompt,
  16-generation, 14-depth, 576-row artifact and representative raw-generation
  gates. Intended formal prompts/extraction are clean, and answer-correct
  invalid long traces are rejected by the cited diagnostics. Row-scoped OOD
  greedy/pass@1/joint@1/pass@16/joint@16 is
  `0.9688/0.8999/0.8428/1.0000/1.0000`; depth-50 answer/joint pass@1 is
  `0.7051/0.5195`. Audit:
  `$HPCVAULT/synthetic-RLVL/analysis/branchproof_selected_followups_audits_20260723/large_16.json`.
- Eval row `3881781_17` is the sole remaining selected row. It is
  priority-pending with no dependency, partition `a100`, feature `a100_80`,
  four A100 GPUs, 16,384 context, and the full 24-hour protocol unchanged.
- Exact from-base Dolmino NL `3875832_2` is healthy beyond step 2581 at about
  31.1K tokens/s with finite diagnostics. Its complete step-2500 state passed
  the 645-file restart gate with exact `2500/320000/1310720000` offsets and
  exact `1245184000 + 65536000` Dolmino/NL token accounting. After acceptance,
  removed only superseded steps 1000/1500/2000 (`319,885,161,508` bytes);
  retained step 2500 re-audits unchanged. Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_nl_exact_step2500_20260802.json`.
  Vault is back below its soft quota at about `948G/1000G`, `182k/200k` files.
- Successor `3941017` is dependency-free, CPU-only/no-GRES, and scheduled for
  13:30 CEST. It remains required. No family-level report, broader submission,
  scheduler edit, or additional GPU job is justified. The scoped commit
  remains local after a silent bounded 90-second GitHub SSH push exited `124`.

## Live Delta At 01:35 CEST On August 2

- Conditioned-32B NL eval row `3881781_15` completed as raw job `3938243`
  `0:0` in `06:23:40` and passed the 448-prompt, 16-generation, 14-depth,
  576-row artifact and representative raw-generation gates. Its row-scoped
  OOD greedy/pass@1/joint@1/pass@16/joint@16 is
  `0.7063/0.6840/0.6742/0.8562/0.7937`; depth-50 answer/joint pass@1 is
  `0.0332/0.0000`. Long-depth failures are format/cap-limited generation
  behavior, not prompt, extraction, or translated-validity artifacts. Audit:
  `$HPCVAULT/synthetic-RLVL/analysis/branchproof_selected_followups_audits_20260723/large_15.json`.
- Eval `3881781_16` is healthy on four A100-80GB GPUs at sampled chunk
  68/112 after about 1h56m; row 17 remains throttle-pending. The full
  24-hour, 16,384-context protocol is unchanged.
- Exact from-base Dolmino NL `3875832_2` is healthy beyond step 1301 at about
  31.1K tokens/s with finite diagnostics. Step 1000 passed the complete
  645-file restart gate with exact `1000/128000/524288000` offsets and exact
  `498073600 + 26214400` Dolmino/NL token accounting. After acceptance,
  removed only superseded step 500 (`106,628,387,164` bytes); step 1000
  re-audits unchanged. Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_nl_exact_step1000_20260802.json`.
  Vault remains above its soft space quota at about `1068G/1000G`, so the
  next complete checkpoint must be audited and rotated promptly.
- Successor `3940687` is dependency-free, CPU-only/no-GRES, and scheduled for
  07:29 CEST. It remains required. No family-level report or broader
  submission is justified. Scoped commit `d000c73` remains local after a
  silent bounded 90-second GitHub SSH push exited `124`.

## Live Delta At 19:35 CEST On August 1

- Conditioned-32B formal eval row `3881781_14` completed as raw job
  `3937636` `0:0` in `06:37:56` and passed the full 448-prompt,
  16-generation, 14-depth, 576-row artifact and raw-review gates. Intended
  formal prompts/extraction and credited validity are clean; deeper
  answer-correct-invalid and wrong chains are genuine generation failures.
  Row-scoped OOD greedy/pass@1/joint@1/pass@16/joint@16 is
  `0.8250/0.7387/0.5859/0.9938/0.9375`. Audit:
  `$HPCVAULT/synthetic-RLVL/analysis/branchproof_selected_followups_audits_20260723/large_14.json`.
- Eval `3881781_15` is healthy on four verified A100-80GB GPUs at sampled
  chunk 75/112 after about 2h20m; rows 16/17 remain throttle-pending. The full
  24-hour, 16,384-context protocol is unchanged.
- Formal Dolmino `3875829_1` completed at terminal step 9537 and passed the
  complete 645-file restart gate with exact `9537/1220736/5000134656`
  offsets and exact `95:5` accounting. Canceled only redundant unstarted
  stage `3875830_1`; after acceptance removed superseded formal steps
  8000/8500/9000/9500, reclaiming `426,513,548,736` bytes. Retained step
  9537 re-audits unchanged. Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_logic_step9537_20260801.json`.
- Exact from-base NL retry `3875832_2` is running on eight A100-80GB GPUs,
  beyond step 41 at about 31.2K tokens/s with the exact 95:5 blend and finite
  diagnostics. Successor `3939882` is CPU-only/no-GRES, dependency-free, and
  scheduled for 01:29 CEST August 2. It remains required; no family-level
  report or broader submission is justified. Scoped commit `c499ba8` remains
  local after a silent bounded 90-second GitHub SSH push exited `124`; local
  `main` is nine commits ahead of `origin/main`.

## Live Delta At 13:35 CEST On August 1

- Conditioned-32B NL eval row `3881781_13` completed as raw job `3936773`
  `0:0` in `06:45:28` on four A100-80GB GPUs and passed the fail-closed
  448-prompt, 16-generation, 14-depth, 576-retained-row, complete-log, and
  credited-diagnostic gates. Representative review covers clean intended NL
  prompts/extraction, shallow/train/OOD successes, and depth-50 cap-limited
  failures. Row-scoped OOD greedy/pass@1/joint@1/pass@16/joint@16 is
  `0.6938/0.6949/0.6949/0.8125/0.8125`. Audit:
  `$HPCVAULT/synthetic-RLVL/analysis/branchproof_selected_followups_audits_20260723/large_13.json`.
- Eval `3881781_14` is healthy on four verified A100-80GB GPUs at sampled
  chunk 84/112 after about three hours with no fatal signature. Rows 15..17
  remain array-throttle pending; the full 24-hour, 16,384-context protocol is
  unchanged.
- Formal Dolmino `3875829_1` is healthy beyond step 8331 at about 31.0K
  tokens/s with finite diagnostics. Accepted its complete step-8000 restart
  state with exact `8000/1024000/4194304000` offsets and exact `95:5` token
  accounting. After acceptance, removed only superseded steps 7000/7500,
  reclaiming `213,256,774,362` bytes. Step 8000 is the sole numeric formal
  restart state and re-audits unchanged. Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_logic_step8000_20260801.json`.
- NL `3875832_2` remains account-GRES pending and formal terminal stage
  `3875830_1` remains dependency-held. Successor `3937914` is CPU-only,
  no-GRES, dependency-free, and scheduled for 19:28 CEST. It remains required.
  No report regeneration or broader submission is justified. Scoped commit
  `9979d5d` remains local after a silent bounded 90-second push and a
  port-443 connection timeout; local `main` is eight commits ahead of
  `origin/main`.

## Live Delta At 07:33 CEST On August 1

- Conditioned-32B eval row `3881781_12` completed as raw job `3935515`
  `0:0` in `07:05:27` on four A100-80GB GPUs and passed the fail-closed
  448-prompt, 16-generation, 14-depth, 576-retained-row, complete-log,
  fresh-constant, and credited-validity gates. Representative raw review
  covers intended formal prompts, correct-valid, answer-correct-invalid, and
  wrong cases. Its row-scoped OOD greedy/pass@1/joint@1/pass@16/joint@16 is
  `0.9375/0.9508/0.8871/1.0000/1.0000`. Audit:
  `$HPCVAULT/synthetic-RLVL/analysis/branchproof_selected_followups_audits_20260723/large_12.json`.
- Eval `3881781_13` is running on four verified A100-80GB GPUs at sampled
  chunk 89/112 after about 3h38m with no fatal signature. Rows 14..17 remain
  array-throttle pending; the full 24-hour, 16,384-context protocol is
  unchanged.
- Formal Dolmino `3875829_1` is healthy at step 7051, about 31.0K tokens/s,
  with finite diagnostics. Accepted its complete step-7000 restart state with
  exact `7000/896000/3670016000` offsets and exact `95:5` token accounting.
  After acceptance, removed only superseded steps 5500/6000/6500, reclaiming
  about 319.9 GB. Step 7000 is the sole numeric formal restart state; Vault
  is `869G/1000G`, 181k files. Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_logic_step7000_20260801.json`.
- NL `3875832_2` remains account-GRES pending; formal terminal stage
  `3875830_1` remains dependency-held. Successor `3937058` is CPU-only,
  no-GRES, dependency-free, and scheduled for 13:28 CEST. It remains required.
  No report regeneration or broader submission is justified.

## Live Delta At 01:33 CEST On August 1

- Dolmino control `3875826_0` completed `0:0` at step 9537. Its terminal
  checkpoint passes the full 645-file restart gate and exact
  `9537/1220736/5000134656` offsets, solely from the control Nanoset. Formal
  continuation `3875829_1` started on eight A100-80GB GPUs, restored the
  audited step-5000 state exactly, and reached step 5781 at about 31.0K
  tokens/s with finite diagnostics. Its accepted step-5500 state contains
  exact `2739404800 + 144179200` Dolmino/formal tokens.
- Audits are
  `analysis/nanotron_checkpoint_audits/dolmino_control_step9537_20260801.json`
  and `analysis/nanotron_checkpoint_audits/dolmino_logic_step5500_20260801.json`.
  After acceptance, removed only superseded control 9000/9500 and formal 5000
  states (`319,885,161,145` bytes). Retained terminal control 9537 and live
  formal 5500; Vault is `869G/1000G`, `181k/200k` files. Redundant unstarted
  control stage `3875827_0` was canceled.
- Conditioned-32B recovery `3932131_14` completed at step 10000 and its final
  passed the nine-file, zero-byte, terminal-state, and 896-tensor adapter
  gates. Eval `3881781_12` is running on four verified A100-80GB GPUs at
  sampled chunk 99/112 after about 4h44m; rows 13..17 are throttle-pending.
  The full 24-hour, 16,384-context protocol remains unchanged.
- NL `3875832_2` remains account-GRES pending and formal terminal stage
  `3875830_1` remains correctly dependency-held. Successor `3936407` is
  dependency-free, CPU-only/no-GRES, and scheduled for 07:27 CEST; it remains
  required. No final eval artifact, scientific metric, or report update is
  available yet.

## Live Delta At 19:32 CEST On July 31

- Dolmino control `3875826_0` is healthy at iteration `9071/9537`, or
  `4.76B/5B` tokens, at about 31.1K tokens/s with finite diagnostics and an
  ETA near 21:37 CEST. Step 9000 passed the complete 645-file, zero-byte,
  TP4/DP2, optimizer/scheduler/RNG, and exact
  `9000/1152000/4718592000` offset gates. Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_control_step9000_20260731.json`.
- After acceptance, removed only superseded control step 8500
  (`106,628,386,982` bytes). Step 9000 is the sole numeric restart state;
  Vault is `748G/1000G`, `181k/200k` files.
- Conditioned-32B recovery `3932131_14` is healthy at step `9612/10000`
  with complete nonempty steps 9250/9500. Eval `3881781_12..17` remains
  held on `afterok:3932131` with its four-A100-80GB 24-hour protocol.
  Formal/NL `3875829_1/3875832_2` remain `AssocGrpGRES` pending.
- Successor `3935384` is dependency-free, CPU-only/no-GRES, and BeginTime
  pending for 01:27 CEST August 1. It remains required. No eval artifact,
  scientific metric, report update, or scheduler edit was available. Scoped
  commit `bd6d86e` remains local after a silent bounded 90-second push exited
  `124`.

## Live Delta At 18:50 CEST On July 31

- Dolmino control `3875826_0` is healthy at iteration `8921/9537`, or
  `4.68B/5B` tokens, with an ETA near 21:35 CEST. Complete steps 8000/8500
  passed all restart, shard, zero-byte, topology, and exact-offset gates.
  Retained step 8500 and removed only accepted superseded steps 7500/8000,
  reclaiming `213,256,773,961` bytes. Audits are under
  `analysis/nanotron_checkpoint_audits/`.
- Conditioned-32B recovery `3932131_14` is healthy with a complete step-9250
  checkpoint (`loss=0.0109`, `grad_norm=0.0186`). Expected completion is
  approximately 21:00--22:00 CEST; eval `3881781_12..17` remains held on
  `afterok:3932131`.
- Formal/NL `3875829_1/3875832_2` remain `AssocGrpGRES` pending. Slurm's
  identical 00:05 CEST estimates are optimistic/independent because each row
  requests a full eight-GPU node. Neither row has failed or started.

## Live Delta At 13:30 CEST On July 31

- Dolmino control `3875826_0` is healthy at iteration `7801/9537` on eight
  A100-80GB GPUs, sustaining about 31.1K tokens/s with finite diagnostics.
  Step 7500 passed the complete 645-file, zero-byte, TP4/DP2
  model/optimizer/scheduler/RNG and exact `7500/960000/3932160000` offset
  gate. Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_control_step7500_20260731.json`.
- After accepting step 7500 and independently checking both older states,
  removed only superseded control steps 6500/7000
  (`213,256,773,958` bytes). Step 7500 is the sole numeric control restart
  state; Vault is back to `748G/1000G`, `181k/200k` files.
- Conditioned-32B recovery `3932131_14` resumed from step 7250 and is healthy
  at step `7717/10000` on four A100-80GB GPUs. Its complete nonempty
  step-7500 checkpoint is intact, and current throughput fits the 12-hour
  allocation. Eval `3881781_12..17` retains `afterok:3932131`, four
  A100-80GB GPUs, and 24 hours.
- Formal/NL `3875829_1/3875832_2` remain `AssocGrpGRES` pending with
  independent 21:26 estimates. Successor `3933417` is dependency-free,
  CPU-only/no-GRES, and BeginTime-held for 19:26 CEST; it remains required.
  No evaluation artifact, new metric, report update, or scheduler edit was
  justified. A bounded 90-second main-repo push produced no response and
  exited `124`; the scoped commit remains local.

## Live Delta At 10:40 CEST On July 31

- Conditioned-32B row `3883534_14` reached the expected 24-hour timeout.
  Step 7250 is complete and nonempty with `global_step=7250` and finite final
  diagnostics. Exact row-14 recovery `3932131_[14%1]` is submitted for 12
  hours on four A100-80GB GPUs. Eval `3881781_12..17` now depends on
  `afterok:3932131` and retains its full A100-80GB protocol.
- The deep step-5000 control/formal output audit is complete. It changes no
  active job: preserve the future NL step-5000 checkpoint for the exact
  matched limited readout recorded in the backlog.

## Live Delta At 09:05 CEST On July 31

- Dolmino control `3875826_0` is healthy at iteration `6881/9537` on eight
  A100-80GB GPUs, sustaining about 31.1K tokens/s with finite diagnostics.
  Its current ETA is about 12.5 hours, near 21:30 CEST today.
- Conditioned-32B row `3883534_14` is healthy at step `7262/10000` on four
  A100-80GB GPUs, but its allocation ends at 09:23 CEST. Recover only this
  row after the actual timeout from the newest complete checkpoint (expected
  step 7250), then replace the dependency for eval `3881781_12..17`.
- Formal/NL `3875829_1/3875832_2` remain account-GRES pending with
  independent 14:08 CEST estimates. Their identical estimates do not imply
  simultaneous starts because both request full eight-GPU nodes. Terminal
  stages remain dependency-held. No new failure or result artifact appeared.

## Live Delta At 07:33 CEST On July 31

- Dolmino control `3875826_0` reached iteration 6541 with finite diagnostics
  at about 31.1K tokens/s. Step 6500 passed the complete 645-file,
  zero-byte, TP4/DP2 model/optimizer/scheduler/RNG and exact
  `6500/832000/3407872000` offset gate. Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_control_step6500_20260731.json`.
- After accepting step 6500, removed only superseded complete control steps
  5000/5500/6000 (`319,885,160,905` bytes). Step 6500 remains intact as the
  sole numeric restart state; Vault is back to `748G/1000G`,
  `181k/200k` files.
- Conditioned-32B row `3883534_14` is healthy at step 6763 with complete
  checkpoints 6500/6750 on four A100-80GB GPUs. It projects past its 09:23
  CEST walltime; submit an exact one-row recovery only after the actual
  timeout, then rewire `3881781_12..17` to the recovery without changing its
  A100-80GB four-GPU 24-hour protocol.
- Formal/NL `3875829_1/3875832_2` remain account-GRES pending with
  provisional independent 10:06 starts. Successor `3930712` is
  dependency-free, CPU-only/no-GRES, and scheduled for 13:26 CEST. It remains
  required. No evaluation output or report gate opened. Scoped commit
  `5e084e0` remains local after a silent 60-second push timed out.

## Live Delta At 01:33 CEST On July 31

- Conditioned-32B recovery `3910990_13` completed `0:0` at step 10000 and its
  final OLMo-3-32B adapter passed nonempty-file and 896-tensor safetensors
  load gates. Final-backed checkpoints 9750/10000 were removed, reclaiming
  `3,242,738,589` bytes; the final adapter remains intact.
- Original conditioned row `3883534_14` is healthy at step 4867 with complete
  checkpoints 4500/4750 on four A100-80GB GPUs. Eval `3881781_12..17` now
  depends only on that row and retains partition `a100`, feature `a100_80`,
  four GPUs, and 24 hours. Recover row 14 only after its expected actual
  timeout from the newest complete state.
- Dolmino control continuation `3875826_0` started on eight A100-80GB GPUs,
  loaded the audited step-5000 restart offsets exactly, and reached iteration
  5271/9537 at about 31.1K tokens/s with finite diagnostics. No W&B service
  process is running. Formal `3875829_1` and NL `3875832_2` remain
  `AssocGrpGRES` pending; terminal `3875827_0/3875830_1/3875833_2` remain
  dependency-held.
- Successor `3930503` is dependency-free, CPU-only/no-GRES on `a100mig`, and
  BeginTime-pending for 07:26 CEST. It is preserved because both critical
  paths remain incomplete. No evaluation output or new scientific metric was
  available, so no report or broader-grid update was made.

## Live Delta At 19:32 CEST On July 30

- Conditioned-32B recovery `3910990_13` resumed cleanly from step 7500 and is
  healthy at step `9360/10000` with complete checkpoints 9000/9250. Original
  row `3883534_14` is healthy at step `2981/10000` with complete checkpoints
  2500/2750. Both use four verified A100-80GB GPUs, and focused fatal scans
  found no OOM, quota, no-space, traceback, or nonfinite-loss signature.
- Eval `3881781_12..17` remains dependency-held on exactly those two SFT jobs
  and retains its A100-80GB-only full protocol. Recover row 14 only after an
  actual timeout; row 13 currently fits its 12-hour recovery allocation.
- Dolmino `3875826_0/3875829_1/3875832_2` remains account-GRES pending, with
  provisional independent 23:44 CEST starts. Home is `81,863M/100G`, Vault
  is `753G/1000G` and `181k/200k` files, and group Work is `469k/500k`
  files.
- Successor `3929539` is dependency-free, CPU-only/no-GRES, and BeginTime
  pending for 01:25 CEST July 31. Both critical paths remain incomplete, so
  it is preserved. A bounded 90-second main-repo push timed out silently.

## Live Delta At 13:33 CEST On July 30

- Conditioned-32B recovery `3910990_13` started at 13:04 on four A100-80GB
  GPUs; its accepted step-7500 restart remains intact during preprocessing.
  Original `3883534_14` is healthy at step 1077/10000 with complete
  checkpoints 750/1000. Eval `3881781_12..17` remains dependency-held.
- Repaired the MATH-500 sidecar's terminal-percent normalization. Focused
  tests pass (`19 passed`); CPU-only rescoring and re-audit accept both
  step-5000 bundles, so no GPU rerun is needed.
- Limit-100 raw review is complete. Reviewer macro delta is provisional
  `+0.0408`; stock multi-hop QA-F1 falls `0.3171 -> 0.1093` because formal
  rows continue into new QA records, while tagged QA-F1 changes
  `0.3104 -> 0.3348`. Artifact:
  `analysis/nanotron_dolmino_step5000_intermediate_20260730.md`.
- Dolmino continuation/retry jobs remain account-GRES pending. CPU-only
  successor `3927534` remains BeginTime-pending and required.

## Live Delta At 10:32 CEST On July 30

- Conditioned-32B SFT `3883534_14` is now running on four A100-80GB GPUs
  after starting at 09:23. Preprocessing completed and optimizer progress is
  active at step 139/10000 with no fatal log signature; W&B run `u6cyipt5`
  is syncing. No checkpoint exists yet. Current throughput projects beyond
  the 24-hour walltime, so the established checkpoint/recovery chain remains
  necessary.
- A40 intermediate eval row 0 completed cleanly and produced an accepted
  105-leaf/10,600-sample bundle. Row 1 also finished inference and wrote
  105 leaf files/10,600 samples, but exited 1 because its production audit
  found one MATH-500 stock-exact positive (sample 67) lost by the post-hoc
  scorer. This is a result-validation failure, not compute, OOM, or quota
  failure; no automatic resubmission was made.
- Eligible pending A100 work remains blocked by `AssocGrpGRES`.
  `3910990_13` requests four GPUs and forecasts 23:06 CEST; each Dolmino
  continuation requests eight and forecasts 06:51 CEST July 31. Job count
  from another account user does not reveal shared consumption because two
  multi-GPU jobs can consume most or all of the inherited GPU cap, and
  sibling GPU allocations remain hidden by `PrivateData`.
- Latest job footer reports Home at 83.7 GB and 157k files against the
  104.9-GB/500k soft limits, and Vault at 779.4 GB and 180k files against the
  1048.6-GB/200k soft limits. No scheduler or job mutation was made.

## Live Delta At 07:38 CEST On July 30

- Slurm intentionally deprioritizes `c107fa12` under fair-share: 25% account
  shares versus 61.2% effective recent usage gives `LevelFS=0.4085` and only
  1,768/10,000 fair-share priority points. Usage decays with a seven-day
  half-life; fair-share weight 10,000 exceeds age weight 6,000.
- Scheduler age starts when a job becomes eligible. Control only became
  eligible July 27 and formal/NL plus conditioned SFT on July 28, despite
  older chain submission dates. They therefore have about 1,690--2,552 age
  points rather than the seven-day maximum 6,000.
- Hidden sibling-user jobs can take newly released account GPU capacity when
  their total priority is higher. Full-node eight-GPU Dolmino jobs are also
  harder to backfill. Coordination or administrator QoS/reservation/cap
  changes are the actionable remedies; no job setting was changed.

## Live Delta At 07:33 CEST On July 30

- Root cause of the prolonged queue is confirmed as an inherited
  account-wide generic GPU association cap. It blocks every GPU type,
  including the one-GPU A40 eval, while `c107fa12` has zero visible running
  GPU allocation.
- Cluster `PrivateData` hides accounts/jobs/usage/users, so other `c107fa`
  users' jobs and the parent association limit are not visible to this user.
  Cross-user/account-limit queries are permission-denied. Hidden sibling-user
  GPU usage is consuming the group cap.
- The shared 08:20 forecast for the three eight-A100 Dolmino jobs reflects a
  hidden allocation's projected release, not three simultaneous starts.
  Pending or dependency-never-satisfied jobs consume no GPU and do not cause
  the cap. No partition edit can bypass it, and no job mutation was made.

## Live Delta At 07:29 CEST On July 30

- No in-scope start, completion, failure, log, or output appeared.
  Conditioned-32B SFT `3910990_13/3883534_14` remains A100-80GB
  account-GRES pending without estimates; eval `3881781_12..17` retains its
  exact two-job dependency.
- Dolmino control/formal continuations `3875826_0/3875829_1` and exact NL
  retry `3875832_2` now provisionally forecast 08:20 CEST. They compete for
  full A100-80GB nodes under one association cap, and their terminal
  `afterany` dependencies remain correct.
- A40 intermediate eval `3913651_[0-1%2]` has no forecast despite idle
  physical nodes because the association GRES cap is binding. All required
  final/restart artifacts remain complete and nonempty.
- Work group use is about 464k files, below quota. Current oversight
  `3926040` is running CPU-only/no-GRES; recorded successor `3926409` is
  dependency-free, CPU-only/no-GRES, and scheduled for 13:24 CEST. No
  partition, dependency, cancellation, or resubmission edit was made. The
  handoff commit remains local after a silent 90-second push timed out.

## Live Delta At 01:27 CEST On July 30

- No in-scope experiment start, completion, failure, log, or output artifact
  appeared. Conditioned OLMo-3-32B SFT `3910990_13/3883534_14` remains
  A100-80GB account-GRES pending without estimates; eval
  `3881781_12..17` retains exactly those dependencies and the full protocol.
- Dolmino control/formal continuations `3875826_0/3875829_1` and exact NL
  retry `3875832_2` now provisionally project 02:11 CEST. All three request a
  full eight-A100-80GB node under the same association cap, so the common
  estimate does not imply concurrent starts. Terminal stages
  `3875827_0/3875830_1/3875833_2` retain their correct `afterany`
  dependencies.
- Both accepted Dolmino step-5000 trees remain 645-file, nonempty,
  exact-offset restart states. The accepted conditioned-32B seed-3407 final
  and seed-3408 step-7500 state remain intact; seed 3409 has correctly written
  no run root before starting. A40 step-5000 eval `3913651_[0-1%2]` remains
  account-GRES pending without an estimate, so no scheduler mutation was made.
- Work group quota is `464k/500k` files; Vault is `743G/1000G`,
  `181k/200k` files. Successor oversight `3926040` is dependency-free,
  CPU-only/no-GRES, and scheduled for 07:24 CEST; it remains required.
  Handoff commit `45c5c6a` remains local after a silent 90-second push
  attempt exited `124`.

## Live Delta At 19:28 CEST On July 29

- No in-scope experiment start, completion, or failure occurred. Conditioned
  OLMo-3-32B SFT `3910990_13/3883534_14` briefly projected 20:52 CEST, but
  both estimates disappeared on the final scheduler sample. Eval
  `3881781_12..17` retains only those two dependencies and its full
  A100-80GB protocol.
- Dolmino control/formal continuations `3875826_0/3875829_1` provisionally
  project 21:40 CEST, while exact from-base NL retry `3875832_2` projects
  22:55 CEST. Terminal jobs `3875827_0/3875830_1/3875833_2` retain their
  correct `afterany` dependencies. Both accepted step-5000 restart trees
  remain 645-file, nonempty, exact-offset states.
- A40 step-5000 eval `3913651_[0-1%2]` remains account-GRES pending and no
  longer has a start estimate. Physical A40 availability does not clear the
  account association cap, so no scheduler edit was made. No output root or
  pending-job log exists.
- Work group quota is `464k/500k` files; Vault is `743G/1000G`,
  `181k/200k` files. Successor oversight `3925270` is dependency-free,
  CPU-only/no-GRES, and scheduled for 01:24 CEST July 30; it remains required.
  Handoff commit `38e61f7` remains local after a silent 90-second push attempt.

## Live Delta At 15:11 CEST On July 29

- Work group inode use is now 463,067, below the 500k soft quota. No in-scope
  job started, failed, or was canceled during cleanup; all pending logs remain
  absent.
- Synthetic conditioned-32B SFT `3910990_13/3883534_14` remains A100-80GB
  account-GRES pending without current estimates. Eval `3881781_12..17`
  retains its exact two-job dependency. Required final/resume artifacts remain
  complete and nonempty.
- Dolmino control continuation `3875826_0` provisionally estimates 18:45
  CEST. Formal continuation `3875829_1` and exact NL retry `3875832_2`
  estimate 06:46 CEST July 30. Terminal dependencies are intact, and both
  accepted 645-file step-5000 restart states have no empty files.
- A40 intermediate eval `3913651_[0-1%2]` provisionally estimates 16:37 CEST
  and has not created an output root. Current compatible capacity is
  allocated/mixed/draining, so no partition or dependency edit was made.

## Live Delta At 14:53 CEST On July 29

- Removed exactly 314 provably disposable Work inodes: 157 W&B run-local
  `tmp/` directories and their 157 empty `code/` children. They contained no
  files or bytes. All W&B histories and scientific artifacts remain intact.
- Group Work usage now reports about 588.7k files and a fresh mkdir/file
  probe succeeds. The roughly 11k total reduction from the prior hard limit
  exceeds this repo's 314-inode cleanup, so it is not attributed to this
  action. Work remains above the 500k soft limit and needs more margin.

## Live Delta At 14:40 CEST On July 29

- `$WORK` is currently unable to allocate an inode: group `c107fa` is at its
  exact 600,000-file hard limit on `/home/atuin` (500,000 soft limit; about
  34 hours of grace shown). Direct `mkdir` probes fail with `Disk quota
  exceeded` in `$WORK` and succeed in Home, Vault, and scratch. Byte quota is
  not the constraint.
- This group quota is shared across `c107fa10..c107fa13`. Exact user quota
  reports attribute 441,032 Work inodes (73.5%) to `c107fa12`; the other
  three users together account for 158,968, but private sibling trees and
  quota permissions prevent their individual split. The largest local
  contributors are `FOMO_runtime` (146,017), `.venv` (94,362),
  `babylm_runtime` (47,899), TextJEPA (43,851), `nanotron` (32,089), and
  `.local` (30,762). No ambiguous runtime, checkpoint, result, or
  other-project deletion was made.
- All in-scope GPU jobs remain pending. Because `scripts/env.sh` points
  runtime caches and W&B directories to Work, they may fail on any new Work
  file until inodes are freed or the group quota is raised. W&B remains
  disabled for the pending Dolmino jobs, but no job/dependency mutation was
  authorized or applied.

## Live Delta At 14:22 CEST On July 29

- Project-scoped quota audit removed only superseded conditioned-32B SFT
  restart snapshots: seed-3407 steps 9750/10000 after its accepted final, and
  seed-3408 step 7250 after validating the newer resume step 7500. Exact
  reclaim was `9,728,633,344` bytes; Vault is now `743G/1000G`.
- The eval-required seed-3407 final and recovery-required seed-3408
  checkpoint-7500 remain complete and nonempty. Both Dolmino step-5000
  restart trees and every other checkpoint/final remain untouched. The 69G
  `$WORK/synthetic-RLVL` tree was audited but conservatively left unchanged.

## Live Delta At 13:26 CEST On July 29

- No in-scope GPU job started or completed. Conditioned-32B SFT
  `3910990_13/3883534_14` remains A100-80GB account-GRES pending with
  provisional 17:23 CEST starts; full-protocol eval `3881781_12..17`
  retains its two-job dependency.
- Dolmino continuation/retry `3875826_0/3875829_1/3875832_2` remains
  account-GRES pending. Current provisional starts are 22:16 CEST for control
  and 04:17 CEST July 30 for formal/NL. No pending-job log or new checkpoint
  exists; the shared wrapper still supplies disabled/no-op W&B.
- A40 intermediate eval `3913651_[0-1%2]` remains pending, but its estimate
  advanced materially to 23:01 CEST July 29. The output root remains absent,
  so there are no generations or metrics to inspect.
- Accepted control/formal step-5000 trees remain intact as the sole numeric
  restart checkpoints. Vault is `752G/1000G`, `181k/200k` files.
  Successor `3924006` is dependency-free, CPU-only/no-GRES, and scheduled for
  19:22 CEST; it remains required.

## Live Delta At 18:04 CEST On July 28

- Submitted Dolmino step-5000 intermediate direct eval
  `3913651_[0-1%2]` for accepted control/formal checkpoints. Each row uses
  one A40, converts to a job-local HF checkpoint, locally verifies the
  conversion, evaluates, and removes only the temporary HF copy.
- Each condition first runs 100-example tagged and stock
  HotpotQA/2WikiMultiHopQA/MuSiQue, then the normal reviewer suite at
  lm-eval limit 100. Both outputs retain raw generations and must pass
  limit-aware artifact/prompt/metric audits. The array is pending with a
  provisional 07:42 CEST August 4 estimate.
- Validation passed: both task registries, shell syntax, Python compilation,
  `git diff --check`, Slurm test submission, and relevant tests (`11 passed`).
  The A40 placement avoids competing with the full-node A100-80GB Dolmino
  continuation and NL retry.

## Live Delta At 13:55 CEST On July 28

- Selected corrected BranchProof acceptance is `39/45`; only conditioned
  OLMo-3-32B eval `3881781_12..17` remains. Row-12 SFT recovery is accepted;
  row-13 recovery `3910990_13` and original row `3883534_14` are pending with
  individual 17:22 CEST estimates.
- Dolmino control/formal each retain audited step-5000 states at
  `2,621,440,000` consumed tokens. Continuations `3875826_0/3875829_1` are
  pending with current estimates of 18:44 CEST July 28 and 09:30 CEST July
  29. NL retry `3875832_2` has no scientific state to resume and currently
  shares the 09:30 CEST July 29 estimate.
- Patched the shared Nanotron wrapper from W&B offline mode to disabled mode.
  This preserves all training settings and queued jobs while preventing the
  local service/port-file failure that killed the first NL attempt. Shell
  syntax checks pass, and installed W&B source confirms disabled mode uses a
  no-op run without starting a service.
- There are no other active project GPU jobs. Successor oversight `3912896`
  remains CPU-only/no-GRES and scheduled for 19:21 CEST.
- The prior GitHub SSH timeout cleared: main repo through `1bfb1d6` and report
  repo through `0265eae` are now pushed to `origin/main`.

## Live Delta At 13:25 CEST On July 28

- Formal Dolmino `3875828_1` ended in the expected 24-hour `TIMEOUT` after
  iteration 5021. Its step-5000 state passed the 645-file, zero-byte, TP4/DP2
  model/optimizer/scheduler/RNG, RoPE-`1000000`, and exact
  `5000/640000/2621440000` offset gates. Exact realized consumption is
  `2490368000` normal plus `131072000` formal tokens (`95:5`). Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_logic_step5000_20260728.json`.
- After acceptance, removed only superseded formal step 4500, reclaiming
  about `106.63 GB`. Formal step 5000 is the sole numeric restart state;
  project Vault use is `667,160,528 KiB`.
- NL first stage `3875831_2` failed before step 1 when the local W&B service
  timed out waiting for its port file. It created no checkpoint. Released
  continuation `3875832_2` is therefore an exact from-base retry and remains
  account-GRES pending alongside control/formal continuations
  `3875826_0/3875829_1`; terminal stages retain their dependencies.
- Conditioned-32B row-12 recovery `3897965_12` completed at step 10000 and
  passed the final-adapter gate: eight nonempty files, a
  `536,991,984`-byte adapter payload, no zero-byte file, and trainer
  `global_step=10000`. Eval `3881781_12..17` now waits only on A100-80GB rows
  `3910990_13` and `3883534_14`.
- Successor `3912896` is dependency-free, CPU-only/no-GRES, and
  BeginTime-pending. The active plan remains incomplete, so it is preserved.
- Scoped audit commit is `38089da`; a bounded 60-second push timed out with no
  remote response (`124`), preserving the known publication blocker.

## Live Delta At 07:25 CEST On July 28

- Dolmino formal `3875828_1` is healthy through iteration 4851 at about
  `30.8K` tokens/s. Its step-4500 state passed the complete 645-file,
  zero-byte, TP4/DP2 model/optimizer/scheduler/RNG, RoPE-`1000000`, and exact
  `4500/576000/2359296000` offset gates. Exact realized consumption is
  `2241331200` normal plus `117964800` formal tokens (`95:5`). Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_logic_step4500_20260728.json`.
- After that acceptance, removed only superseded formal steps 3500/4000,
  reclaiming `213,256,774,301` bytes. Formal step 4500 is the sole numeric
  restart state; Vault is back to `751G/1000G`, `181k/200k` files.
- Conditioned-32B row-12 exact resume `3897965_12` is healthy through step
  9160 on A100-80GB node `a0831`, with complete checkpoints 8750/9000.
  Row-13 resume `3910990_13` and original row `3883534_14` are account-GRES
  pending. Eval `3881781_12..17` retains its repaired three-job dependency
  and A100-80GB protocol.
- Control continuation `3875826_0` and NL first stage `3875831_2` remain
  account-GRES pending; later Dolmino stages retain their dependencies.
  Successor `3911426` is dependency-free, CPU-only/no-GRES, and scheduled for
  13:20 CEST. The active plan remains incomplete, so it is preserved.
- Scoped audit commit is `162a0fe`; a bounded 60-second push timed out with no
  remote response (`124`), preserving the known publication blocker.

## Live Delta At 01:55 CEST On July 28

- Dolmino formal `3875828_1` is healthy through iteration 3591 at about
  `30.7K` tokens/s. Its step-3500 state passed the complete 645-file,
  zero-byte, TP4/DP2 model/optimizer/scheduler/RNG, RoPE-`1000000`, and exact
  `3500/448000/1835008000` offset gates. Exact realized consumption is
  `1743257600` normal plus `91750400` formal tokens (`95:5`). Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_logic_step3500_20260728.json`.
- After that acceptance, removed only superseded formal steps 2000/2500/3000,
  reclaiming `319,885,161,449` bytes. Formal step 3500 is the sole numeric
  restart state; Vault is back to `751G/1000G`, `181k/200k` files.
- Conditioned-32B `3883534_13` reached step 7538 and ended in the expected
  `TIMEOUT` after `1-00:00:15`, with no OOM, quota, or unexpected fatal
  signature. Its step-7500 checkpoint has 13 nonempty files and complete
  adapter/optimizer/scheduler/RNG state. Submitted only exact row-13 recovery
  `3910990_13` for 12 hours on four A100-80GB GPUs. Exact row-12 resume
  `3897965_12` started on verified A100-80GB node `a0831`; row
  `3883534_14` remains pending. Eval `3881781_12..17` now depends on
  `afterok:3897965:3910990:3883534_14` and retains A100-80GB.
- Control continuation `3875826_0` and NL first stage `3875831_2` remain
  account-GRES pending. Successor `3910926` is dependency-free,
  CPU-only/no-GRES, and scheduled for 07:20 CEST; the active plan is
  incomplete, so it is preserved.
- Scoped commits are `74b3339` and `1324746`; a bounded 60-second push timed
  out with no remote response (`124`), preserving the known publication
  blocker.

## Live Delta At 19:23 CEST On July 27

- Formal Dolmino `3875828_1` is healthy through iteration `2331/9537` at
  about `30.7K` tokens/s. Step 2000 passed the complete 645-file, zero-byte,
  TP4/DP2 model/optimizer/scheduler/RNG, RoPE-`1000000`, and exact
  `2000/256000/1048576000` offset gate. Its realized stream is exactly
  `996147200` normal plus `52428800` formal tokens. Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_logic_step2000_20260727.json`.
- The writer retained steps 1000/1500/2000 and raised Vault to
  `1149G/1000G`. After step 2000 passed, removed only superseded steps
  1000/1500, reclaiming `213,256,774,296` bytes. Step 2000 is the sole
  numeric restart state; Vault is `751G/1000G`, `181k/200k` files, and
  project Vault/Work use is `666,092,361/71,701,364 KiB`.
- Conditioned-32B `3883534_13` is healthy through step `5483/10000`;
  checkpoints 5000/5250 each have 13 nonempty files and no zero-byte file.
  Row `3883534_14` is array-throttle pending, exact row-12 resume
  `3897965_12` and Dolmino control/NL `3875826_0/3875831_2` are
  account-GRES pending, and conditioned eval `3881781_12..17` plus later
  Dolmino stages remain dependency-held. The next watcher should recover row
  13 from its newest complete state only after its projected timeout occurs.
- Watcher `3904173` is CPU-only on `a100mig`. Its recorded successor
  `3908135` is verified dependency-free, CPU-only/no-GRES, and scheduled for
  01:20 CEST July 28. Both critical paths remain incomplete, so it is
  preserved. Scoped audit commit `c53b21c` is local; a bounded 60-second push
  produced no remote response and exited `124`.

## Live Delta At 13:24 CEST On July 27

- Formal Dolmino `3875828_1` is healthy through iteration `1071/9537` at
  about `30.7K` tokens/s. Step 1000 passed the 645-file, zero-byte, TP4/DP2,
  model/optimizer/scheduler/RNG, RoPE-`1000000`, and exact
  `1000/128000/524288000` offset gate. Its realized stream is exactly
  `498073600` normal plus `26214400` formal tokens. Audit:
  `analysis/nanotron_checkpoint_audits/dolmino_logic_step1000_20260727.json`.
- After that acceptance, removed only superseded formal step 500
  (`106,628,387,143` bytes). Step 1000 is the sole numeric formal restart
  state. Vault fell from `950G` to `751G/1000G`, with `181k/200k` files;
  project Vault use is `666,092,233 KiB`.
- Conditioned-32B `3883534_13` is healthy through step `3586/10000`; its
  complete nonempty checkpoints 3250 and 3500 are retained. Row
  `3883534_14` is array-throttle pending, exact resume `3897965_12` and
  Dolmino control/NL `3875826_0/3875831_2` are account-GRES pending, and
  conditioned eval `3881781_12..17` plus later Dolmino stages remain
  dependency-held. No scheduler edit is useful.
- Watcher `3902223` is CPU-only on `a100mig`. Its recorded successor
  `3904173` is verified dependency-free, CPU-only/no-GRES, and scheduled for
  19:19 CEST. Both critical paths remain incomplete, so it is preserved.
  Scoped commit `d0da6b4` is local; a bounded 60-second push produced no
  remote response and exited `124`.

## Live Delta At 08:17 CEST On July 27

- Dolmino control first stage `3875825_0` ended in the expected `TIMEOUT`
  after `1-00:00:10`, last logging iteration 5021. Step 5000 passed the full
  645-file, zero-byte, TP4/DP2 model/optimizer/scheduler/RNG and exact
  `5000/640000/2621440000` offset gate, solely normal data. After accepting
  audited steps 4500 and 5000, removed only superseded complete steps
  3500/4000/4500, reclaiming `319,885,160,841` bytes; step 5000 is the sole
  restart state. `3875826_0` is released and account-GRES pending, while
  `3875827_0` retains its dependency.
- Formal first stage `3875828_1` started at 08:09 CEST on eight A100-80GB
  GPUs. Config and first-iteration review confirms RoPE `1000000`, TP4/DP2,
  4096 context, target 9537, the preregistered LR schedule, realized
  normal/formal weights `0.95/0.0500002`, and finite loss/gradient values.
  Formal continuations `3875829_1/3875830_1` remain dependency-held. NL first
  stage `3875831_2` is account-GRES pending, with `3875832_2/3875833_2`
  dependency-held.
- Conditioned-32B SFT `3883534_13` is running on `a0532` through step
  `1968/10000`, with complete steps 1500/1750 retained and no severe
  signature. `3883534_14` is array-throttle pending, `3897965_12` is
  account-GRES pending, and `3881781_12..17` remains dependency-held.
- User-wide quotas are home `80835M/100G`, `158k/500k` files and Vault
  `553G/1000G`, `180k/200k` files. Watcher `3901977` is CPU-only on
  `a100mig`; successor `3902223` is dependency-free, no-GRES, and scheduled
  for 13:19 CEST. It remains required.

## Live Delta At 01:31 CEST On July 27

- BranchProof single-modal OLMo-3-32B is complete and accepted at `6/6`.
  Final NL row `3881780_5` completed in `06:11:47` and passed its complete
  artifact/log/fresh-constant/translated-validity audit plus representative
  raw review. The matched three-seed OOD answer/joint pass@1 result is formal
  `0.6388 +/- 0.0512` / `0.5109 +/- 0.0182` versus natural
  `0.9888 +/- 0.0080` / `0.9879 +/- 0.0088`. Formal answer pass@16 recovers
  to `0.9875 +/- 0.0088`, but joint remains `0.9396 +/- 0.0179`.
- Dolmino control `3875825_0` is healthy through step `3631/9537` at about
  `30.7K` tokens/s. Step 3500 passed the 645-file, zero-byte, TP4/DP2,
  model/optimizer/scheduler/RNG, and exact
  `3500/448000/1835008000` offset gate. After that acceptance, removed only
  superseded complete steps 2000/2500/3000, reclaiming
  `319,885,160,841` bytes. Step 3500 is the sole numeric restart state.
- User-wide Vault is back to `547G/1000G` with `180k/200k` files; project
  Vault/Work use is `451,487,404/71,621,608 KiB`.
- Conditioned-32B `3883534_13/14`, exact resume `3897965_12`, and Dolmino
  formal/NL `3875828_1/3875831_2` remain account-GRES pending.
  `3881781_12..17` retains its valid dependency gate. Recorded successor
  `3901977` is CPU-only/no-GRES, dependency-free, and scheduled for 07:18
  CEST; it remains required.

## Live Delta At 19:23 CEST On July 26

- Dolmino control `3875825_0` remains healthy beyond step `2341/9537` at
  about `30.7K` tokens/s. Step 2000 independently passed the 645-file,
  zero-byte, TP4/DP2, model/optimizer/scheduler/RNG, and exact
  step/sample/token-offset gate at `2000/256000/1048576000`; audit:
  `analysis/nanotron_checkpoint_audits/dolmino_control_step2000_20260726.json`.
- The writer retained steps 1000, 1500, and 2000, temporarily raising Vault
  to `1064G/1000G`. After accepting step 2000, only superseded complete steps
  1000 and 1500 were removed (`213,256,773,890` bytes). Step 2000 is the sole
  numeric restart state. Vault is back to `667G/1000G`, `180k/200k` files;
  project Vault use is `577,396,560 KiB`.
- BranchProof 32B NL row `3881780_4` completed as raw job `3900120` in
  `06:06:34` and passed the full production audit. Its 448 retained sampled
  rows span all 14 depths and are all answer-correct, format-complete,
  translated-parseable, and translated-citation-free-valid. Complete sampled
  metrics are perfect except for depth-40 answer/joint pass@1
  `0.9941/0.9922`; the row audit is
  `$HPCVAULT/synthetic-RLVL/analysis/branchproof_selected_followups_audits_20260723/large_4.json`.
  Final seed row `3881780_5` is running on four A100-80GB GPUs at sampled
  chunk `89/112` with no fatal signature.
- Conditioned-32B `3883534_13/14` and exact resume `3897965_12` remain
  account-GRES pending; `3881781_12..17` retains
  `afterok:3897965:3883534_13:3883534_14`. Dolmino formal/NL first stages
  `3875828_1/3875831_2` remain account-GRES pending. Recorded successor
  `3901437` is dependency-free, CPU-only/no-GRES, and scheduled for 01:18
  CEST July 27; it remains required.

## Live Delta At 13:22 CEST On July 26

- Dolmino control `3875825_0` started at 08:09 CEST on eight A100-80GB GPUs
  on `a0536`. It is healthy through step `1081/9537` at about `30.7K`
  tokens/s with finite optimization diagnostics. Its step-1000 checkpoint
  passed the 645-file model/optimizer/scheduler/RNG, zero-byte, topology, and
  exact-offset gate; persisted audit:
  `analysis/nanotron_checkpoint_audits/dolmino_control_step1000_20260726.json`.
- After that newer state was accepted, removed only the superseded complete
  step-500 tree, reclaiming `106,628,386,940` bytes. The live writer does not
  prune older 500-step states itself. The later restart stages now enable a
  live-wrapper guard that removes only strictly older complete states after
  resolving the newest accepted resume checkpoint; both edited wrappers pass
  `bash -n`. During a running stage, each pass must still retain the latest
  accepted state and rotate only strictly older complete states.
- BranchProof 32B NL eval `3881780_4` started at 10:02 CEST on four
  A100-80GB GPUs on `a0932`. Greedy scoring is complete and sampled chunks
  `1..89/112` finished; chunk 90 is active without a fatal signature.
  Row `3881780_5` is correctly held by the one-row array throttle.
- Conditioned-32B `3883534_13/14` and exact resume `3897965_12` remain
  account-GRES pending; `3881781_12..17` retains the repaired dependencies.
  Dolmino formal/NL first stages `3875828_1/3875831_2` remain account-GRES
  pending and all downstream stages remain dependency-held.
- Vault quota is `667G/1000G` and `180k/200k` files after rotation; project
  Vault/Work use is `577,374,352/71,571,384 KiB`. Current watcher `3899917`
  is CPU-only on `a100mig`; recorded successor `3900807` is verified
  CPU-only/no-GRES and BeginTime-pending for 19:17 CEST. It remains required.
  Commit `b4a1789` is local; a bounded 60-second push returned no output and
  exited `124`.

## Live Delta At 07:20 CEST On July 26

- No in-scope GPU task started or completed after 01:20 CEST. BranchProof
  `3881780_4/5`, `3883534_13/14`, and `3897965_12` remain
  A100-80GB `AssocGrpGRES` pending; conditioned eval `3881781_12..17`
  retains its repaired dependencies. Dolmino
  `3875825_0/3875828_1/3875831_2` remains dependency-free
  `AssocGrpGRES` pending. Slurm currently provides no start estimate.
- Two A100-80GB nodes are idle, but all in-scope GPU requests remain blocked
  at the association GRES limit. Their resources and dependencies are valid,
  so no partition, feature, throttle, or dependency edit is justified.
- Watcher `3899486` is running CPU-only on `a100mig`; recorded successor
  `3899917` is verified CPU-only/no-GRES, dependency-free, and
  BeginTime-pending for 13:17 CEST. It remains required.
- Vault quota is unchanged at `348G/1000G` and `179k/200k` files. Project
  Vault/Work usage is `243,171,225/71,547,812 KiB`; no cleanup or artifact
  audit was triggered.
- Handoff commit `c50cba6` is local; a bounded 60-second SSH push produced no
  remote response and exited `124`.

## Live Delta At 01:20 CEST On July 26

- No in-scope GPU task started or completed after the prior handoff.
  BranchProof `3881780_4/5`, `3883534_13/14`, and `3897965_12`
  remain A100-80GB `AssocGrpGRES` pending; `3881781_12..17` retains its
  repaired dependencies. Dolmino `3875825_0/3875828_1/3875831_2`
  remains dependency-free `AssocGrpGRES` pending, with current individual
  Slurm projections of 05:15 CEST. No partition or dependency edit is
  justified.
- Revalidated the complete conditioned-32B step-7500 restart state, including
  nonempty adapter, optimizer, scheduler, RNG, tokenizer, and trainer state
  with no zero-byte file. Exact resume `3897965_12` remains correct.
- Watcher `3898756` is running CPU-only on `a100mig`; recorded successor
  `3899486` is verified CPU-only/no-GRES and BeginTime-pending for 07:17
  CEST. It remains required.
- Vault quota is `348G/1000G` and `179k/200k` files. The project uses
  `243,171,225 KiB` in Vault and `71,547,808 KiB` in Work, so no guarded
  cleanup is currently needed.
- Handoff commit `643a541` is local; a bounded 60-second SSH push produced no
  remote response and exited `124`.

## Live Delta At 19:19 CEST On July 25

- No in-scope task started or completed after 13:29. Selected BranchProof
  `3881780_4/5`, `3883534_13/14`, and exact resume `3897965_12` remain
  A100-80GB account-GRES pending; conditioned eval `3881781_12..17` retains
  its repaired dependencies. Dolmino first stages
  `3875825_0/3875828_1/3875831_2` remain unstarted account-GRES pending.
- Successor `3898756` is verified CPU-only on `a100mig`, dependency-free, and
  BeginTime-pending for 01:16 CEST July 26. It remains required.
- Fixed a stale informal-report executive bullet that contradicted the
  accepted eleven-control table. Report generation and informal mirroring
  passed. This is a consistency correction only; no metric, scheduler state,
  or scientific acceptance changed.
- Commits are `ce21a88` and report-repo `df7de8b`. Both bounded SSH pushes
  timed out after 45 seconds without a remote response.

## Live Delta At 13:29 CEST On July 25

- Selected acceptance is `37/45`. Newly accepted rows are surface
  `3881774_7/8/24/25/26`, shortcut `3881775_15..17`, hybrid
  `3881776_27..29`, architecture `3881778_34/35`, and large
  `3881780_3`. Every row passes the full artifact/log/retained-sample,
  fresh-constant, and credited-validity audit plus representative raw review.
- Complete new family OOD answer/joint pass@1 is terse NL
  `0.6414/0.6358`, target-token-matched NL `0.4104/0.4013`,
  shortcut formal versus NL `0.4310/0.3849` versus `0.7854/0.7807`,
  reverse-hybrid `0.0053/0.0001`, and Qwen2.5-7B formal versus NL
  `0.6986/0.6919` versus `0.7491/0.7449`.
- Conditioned-32B row `3883534_12` timed out after 24 hours with complete
  restart states through step 7,500. Exact resume `3897965_12` is pending on
  four A100-80GB GPUs with a 12-hour limit. Conditioned eval
  `3881781_[12-17]` now depends on
  `3897965:3883534_13:3883534_14`; the stale failed dependency was removed.
- Remaining selected work is single-modal 32B eval `3881780_4/5`, conditioned
  SFT `3883534_13/14` plus recovery `3897965_12`, and conditioned eval
  `3881781_12..17`. Dolmino first stages
  `3875825_0/3875828_1/3875831_2` remain dependency-free account-GRES
  pending. Successor `3897962` remains scheduled on CPU-only `a100mig`.

## Live Delta At 13:22 CEST On July 24

- Selected acceptance is `23/45`. Newly accepted terse-NL surface row
  `3881774_6`, raw job `3890505`, completed `0:0` in `07:45:07` and passed
  the complete metric, retained-sample, generation-log, and
  validity-consistency gates. Raw review spans depths 1/25/30/35/40/45/50,
  correct translated-valid cases, wrong/missing-answer cases, and long
  premise-copy truncations.
- The one accepted terse-NL seed has OOD greedy/pass@1/pass@16 answer
  `0.775/0.291/0.806` and translated joint `0.744/0.284/0.788`. Rows 7/8
  remain active, so this is not yet a three-seed surface-family result and
  does not trigger report regeneration.
- Active selected evals are surface `3881774_7/8/24`, shortcut
  `3881775_15..17`, hybrid `3881776_27..29`, and Qwen `3881778_34`.
  Focused logs have no fatal/OOM/quota/no-space signature. Conditioned-32B
  SFT `3883534_12` has passed step 2,000 with complete nonempty checkpoints
  1,750 and 2,000; wait for an actual timeout before exact recovery.
- Dolmino first stages `3875825_0/3875828_1/3875831_2` remain unstarted
  account-GRES pending with provisional 22:02 CEST projections. Vault is
  `621G/1000G` and 179k files; this project is `529,381,827 KiB` with 7,211
  regular files. CPU-only successor `3891580` remains scheduled because both
  critical paths are incomplete.

## Live Delta At 07:22 CEST On July 24

- Selected acceptance is `22/45`. Newly accepted rows are conditioned-7B
  `3881777_26/28/29`, Qwen2.5-7B `3881778_25/26/33`, and OLMo-3-32B
  `3881780_2`. All seven pass the complete metrics/sample/log, fresh-constant,
  and credited-validity consistency gates plus representative raw review.
- Conditioned-7B is now a complete matched three-seed family. Formal versus NL
  OOD answer/joint pass@1 is `0.5029 +/- 0.0644` /
  `0.4488 +/- 0.0600` versus `0.1814 +/- 0.0197` /
  `0.1780 +/- 0.0232`. Raw NL outputs are clean through roughly depth 30 and
  then copy long premise/proof prefixes or omit answers. The report and
  official preprint now include this accepted shared-checkpoint mode
  comparison.
- Qwen formal and OLMo-3-32B formal each have all three seeds accepted, but
  their matched NL rows remain partial or pending. Active evals are surface
  `3881774_6/7/8` at chunks `74/65/47` and shortcut `3881775_15` at chunk
  `27`; all have clean focused logs.
- Conditioned-32B replacement `3883534_12` is running as raw job `3890581`
  and wrote a complete first restart state at `checkpoint-250`: nonempty
  adapter, optimizer, scheduler, RNG, tokenizer, and step-250 trainer state,
  with no zero-byte file. Training continued to step 251. Its projected
  runtime exceeds 24 hours; wait for an actual timeout before exact recovery.
- Dolmino first stages `3875825_0/3875828_1/3875831_2` remain unstarted
  account-GRES pending with provisional 22:02 CEST projections. Vault is
  `451G/1000G` and 179k files; this project is `350,611,842 KiB` with 7,107
  regular files. CPU-only successor `3890623` is verified BeginTime-pending
  for 13:13 CEST and remains required.

## Live Delta At 01:18 CEST On July 24

- Selected acceptance is `15/45`. Newly accepted rows are conditioned-7B NL
  `3881777_27` (raw `3884546`), Qwen2.5-7B logic `3881778_24` (raw
  `3889573`), and OLMo-3-32B logic `3881780_1` (raw `3884864`). All three
  passed the full production artifact, log, retained-sample, fresh-constant,
  and credited-validity consistency gates plus representative raw review.
- Conditioned NL seed 3408 is clean through depth 30 but truncates without an
  answer in every retained depth-35--50 sample; OOD answer/translated-joint
  pass@1 is `0.1891/0.1891`. Qwen2.5 logic seed 3407 has OOD answer/joint
  pass@1 `0.7816/0.7777`, with depth-50 answer completion the dominant
  retained-sample failure. OLMo-3-32B logic seed 3408 has OOD answer/joint
  pass@1 `0.7004/0.5359` and pass@16 `0.9938/0.9375`. No newly complete row
  closes a three-seed matched family.
- Active selected rows are conditioned `3881777_26/28` in scoring after
  `112/112` sampled chunks, conditioned row 29 at chunk 71, Qwen architecture
  row `3881778_25` at chunk 81, and large row `3881780_2` at chunk 55. No
  focused log has a fatal/OOM/quota/no-space signature.
- Dolmino first stages `3875825_0/3875828_1/3875831_2` remain unstarted
  account-GRES pending with individual 05:07 CEST projections.
  Conditioned-32B replacement `3883534_12` projects 04:54 CEST. Vault is
  `572G/1000G` and 179k files; the project tree is `477,666,501 KiB` with
  7,114 regular files. CPU-only successor `3890293` is scheduled for 07:13
  CEST and remains required.

## Live Delta At 19:17 CEST On July 23

- Selected eval completion and acceptance are now `12/45`. OLMo-3-32B logic
  row `3881780_0`, raw job `3883993`, completed on four A100-80GB GPUs in
  `06:34:03` and passed the full 448-prompt, 16-generation, 14-depth,
  576-retained-row, metric, chunk-log, fresh-constant, and
  validity-consistency audit.
- Raw review at depths 1/25/30/40/45/50 covers correct-valid, wrong,
  malformed, and cap-hit cases. Retained samples are perfect through depth 25
  and degrade through ordinary wrong branches and malformed/long traces OOD.
  One-seed OOD greedy answer/joint is `0.738/0.491`, pass@1 is
  `0.641/0.504`, and pass@16 is `0.994/0.919`. Keep this row provisional at
  family level until all matched 32B logic/NL seeds complete.
- Active selected rows are large `3881780_1` at sampled chunk `70/112`,
  conditioned-7B `3881777_26/27/28` at `82/83/79`, and Qwen2.5-7B
  architecture `3881778_24` in clean startup. Remaining rows are
  array-throttle, account-GRES, or valid conditioned-SFT dependency pending.
- Dolmino first stages `3875825_0/3875828_1/3875831_2` remain unstarted
  `AssocGrpGRES` pending with current 22:02 CEST projections.
  Conditioned-32B replacement `3883534_12` currently projects 02:15 CEST July
  24. No checkpoint exists for either startup gate.
- Vault is `572G/1000G` and 179k/200k files; this project is
  `477,610,021 KiB` with 7,105 regular files. Verified CPU-only successor
  `3889637` is scheduled for 01:13 CEST July 24 and remains required. No
  recovery, scheduler edit, cleanup, or report regeneration was justified.

## Live Delta At 13:24 CEST On July 23

- Selected eval completion and row acceptance are `11/45`: surface
  `3881774_0..2`, shortcut `3881775_12..14`, hybrid `3881776_12..14`, and
  conditioned-7B `3881777_24/25`. All pass the full structural, log,
  retained-sample, constant, and validity-consistency gates plus
  representative raw review.
- Surface symbol-padded formal and NL-then-formal hybrid are complete
  three-seed families and are now report evidence. Their OOD answer/joint
  pass@1 is respectively `0.673/0.626` and `0.002/0.000`; hybrid generations
  fit depth 25 but copy both trace surfaces and truncate without answers OOD.
  Shortcut formal-only and the conditioned seed-3407 pair remain partial.
- OLMo-3-32B logic eval `3881780_0` is running as raw job `3883993` on four
  A100-80GB GPUs at sampled chunk `83/112` after about 2h46m. The remaining
  selected rows are array-throttle, priority, or valid conditioned-SFT
  dependency pending. Replacement SFT `3883534_[12-14%1]` has not started.
- Dolmino first stages `3875825_0/3875828_1/3875831_2` remain
  `AssocGrpGRES` pending with individual 05:09 CEST July-24 projections and no
  checkpoint. Successor oversight `3884297` is CPU-only/no-GRES and scheduled
  for 19:12 CEST; it remains required.
- User-wide Vault is `462G/1000G` with 179k/200k files. This project's Vault
  tree is about 346G with 7,055 regular files and 9,322 total entries. No
  cleanup, partition edit, resubmission, or new experiment launch was needed.

## Live Delta At 08:41 CEST On July 23

- Guarded cleanup reclaimed `72.61 GB` from this project's Vault tree: 30
  final-backed Trainer restart checkpoints and the superseded first Dolmino
  normal-data Nanoset. Final models, eval evidence, active merge trees,
  current Nanotron base/restart state, the 5.1B normal Nanoset, and the
  formal/NL intervention Nanosets were retained and verified.
- User-wide Vault is now about `898 GiB/1000 GiB` and 201,900 entries versus
  the 200k soft/400k hard inode limits. This project uses `329.53 GB`, 7,052
  regular files, and 9,321 quota-counted entries in Vault; its Work tree uses
  `73.23 GB` and 6,975 files.
- Active job state is unchanged by cleanup. Hybrid `3881776_12..14` is at
  `99/99/98` of `112` sampled chunks, and all queued Dolmino inputs referenced
  by the immutable jobs remain present.

## Live Delta At 08:27 CEST On July 23

- Selected eval completion is now `8/45`: surface `3881774_0..2`, shortcut
  `3881775_12..14`, and conditioned-7B `3881777_24/25` all exited `0:0`.
  Three rows were already fully accepted; the five newer completions remain
  provisional pending artifact, invariant, and representative raw-generation
  review. No selected eval failed.
- Hybrid rows `3881776_12..14` are running on A100-80GB at approximately
  `99/99/98` of `112` sampled chunks. They are progressing at the expected
  rate with no fatal/OOM/quota/no-space signature. The rest of the selected
  eval matrix is scheduler, array-throttle, or valid SFT-dependency pending.
- Replacement conditioned-32B SFT `3883534_[12-14%1]` remains pending without
  a start estimate. Dolmino control/formal/NL first stages remain
  dependency-free `AssocGrpGRES` pending; the current individual Slurm
  projection is 22:02 CEST July 23. No Dolmino checkpoint exists yet.

## Live Delta At 07:16 CEST On July 23

- Selected rows surface `3881774_0/2` and conditioned-7B `3881777_25`
  completed `0:0` in `10:50:04/10:37:16/07:12:23`. All three pass the
  production structural/log/retained-sample/constant/validity audit and
  representative raw review. Audits are in
  `$HPCVAULT/synthetic-RLVL/analysis/branchproof_selected_followups_audits_20260723/`.
- Symbol-padded formal rows show clean correct-and-valid train-edge examples,
  answer-correct invalid long traces, and depth-50 repetition/truncation.
  Conditioned NL has clean translated-valid examples through depth 30 but all
  32 retained depth-50 samples fail format after long premise/proof copying.
  No family aggregate is accepted from these partial rows.
- Active selected evals are surface `3881774_1`, shortcut
  `3881775_12..14`, hybrid `3881776_12..14`, and conditioned-7B
  `3881777_24`; they are generating or scoring without a fatal signature.
  Remaining selected evals are array-throttle or scheduler/dependency pending.
  Conditioned-32B replacement `3883534_[12-14%1]` remains A100-80 pending,
  and its first run root contains no stale checkpoint/final.
- Dolmino first stages `3875825_0/3875828_1/3875831_2` remain account-GRES
  pending with individual 22:02 CEST projections. No partition widening is
  compatible with the validated full-node A100-80 stack. Vault is
  `1101G/1000G` and `203k/200k` files. CPU-only successor `3883709` remains
  BeginTime-pending and required.

## Live Delta At 01:17 CEST On July 23

- Selected evals currently running are `3881774_0..2`, `3881775_12..14`,
  `3881776_12..14`, and `3881777_24/25`. Their sampled-generation progress is
  `55..89/112` chunks, all on A100-80GB, with no fatal/OOM/quota signature.
  No selected-family final JSON/sample artifact has appeared yet.
- OLMo-3-32B conditioned SFT `3881779_12` reached step `1368/10000` after
  about `4:17`; the measured rate projected beyond 24 hours. Its immutable
  payload used `save_steps=20000`, so it could neither finish nor provide a
  restart checkpoint. Added a `large`-group policy of `save_steps=250` and
  `save_total_limit=2`, canceled original `3881779`, and submitted exact
  replacement `3883534_[12-14%1]`. Rewired only conditioned eval `3881781`
  to `afterok:3883534_12:3883534_13:3883534_14`.
- Dolmino first stages `3875825_0/3875828_1/3875831_2` remain account-GRES
  pending with current individual projections of 22:02 CEST July 23. Vault is
  `1183G/1000G` and `203k/200k` files. Successor oversight `3883531` remains
  BeginTime-pending on CPU-only `a100mig`.

## Live Delta At 20:15 CEST On July 22

- Corrected baseline recovery/audit `3865321_18 -> 3865322_18` and aggregate
  `3857769` completed `0:0`. All `30/30` row audits and the cross-grid
  qualitative audit are accepted. The final recovery took `08:08:43` and did
  not require sharding.
- Accepted artifacts are `analysis/branchproof_unique_v2_20260711/` plus 30
  row JSONs in `analysis/branchproof_unique_v2_full_grid_audits/`. The primary
  train-25 OOD logic-minus-NL deltas are greedy `+0.3333`, answer pass@1
  `+0.5712`, joint pass@1 `+0.5100`, answer pass@16 `+0.5708`, and joint
  pass@16 `+0.5792`; NL is stronger at greedy/pass@1 for train maxima 5--20.
- Dependency release started selected surface eval `3881774_0` on A100-80GB;
  merge, 16,384-token vLLM startup, and early greedy chunks are clean. Other
  selected rows `3881774..3881780` are scheduler-pending, while conditioned
  32B eval `3881781` additionally waits for SFT `3881779_12..14`.
- Dolmino `3875825_0/3875828_1/3875831_2` remain pending. The displayed
  unavailable node `a0631` is not a requested-node binding; each job has an
  alternate scheduler candidate, so no partition or node-list edit was made.
- Successor oversight `3882407` is dependency-free and scheduled at 01:11 CEST
  on CPU-only `a100mig`. It remains required.

## Live Delta At 17:01 CEST On July 22

- Canceled the exhaustive nonbaseline BranchProof arrays and preserved the
  three active corrected-baseline evals plus their audits/aggregate.
- Selected follow-ups are dependency-held on baseline aggregate `3857769`:
  surface `3881774` (9 eval), shortcut `3881775` (6), hybrid `3881776` (6),
  conditioned-7B `3881777` (6), Qwen2.5-7B `3881778` (6), OLMo-3-32B
  conditioned SFT `3881779` (3), single-modal 32B eval `3881780` (6), and
  conditioned 32B eval `3881781` (6). The last also waits for all three new
  32B SFT rows.
- Exact matrix assertions and existing-final gates passed. Remaining
  substantive BranchProof GPU work is 51 rows including the baseline three,
  rather than approximately 310. No canceled merge staging tree remains.
- Guarded cleanup removed about 22.5 GiB of incomplete checkpoint payloads
  from canceled conditioned-50k/batch roots with no final and no live
  reference. Completed and selected artifacts were preserved.

## Live Delta At 16:35 CEST On July 22

- Expanded BranchProof queue count is 22 running plus 319 pending Slurm rows.
  After separating CPU oversight/audits/aggregate and 26 staged recovery rows,
  approximately 310 substantive GPU rows remain: 56 SFT and 254 eval.
- Remaining SFT by family is conditioned-50k 6, shortcut 8, batch 33, and
  32B 9. Remaining eval by family is baseline 3, hybrid/replacements 24,
  conditioned-10k 10, conditioned-50k 30, architecture 52, 32B 18, shortcut
  42, batch 48, and surface 27.
- Baseline rows 13/14 are scoring and row 18 is at chunk 87/112, making the
  corrected baseline gate likely July 23--24 absent recovery. The full
  ablation wave is provisionally August 5--12 under current capacity and
  observed runtimes; dependencies and Dolmino contention make this uncertain.

## Live Delta At 16:20 CEST On July 22

- Dolmino first stages `3875825_0/3875828_1/3875831_2` remain dependency-free
  `AssocGrpGRES` pending, with no logs or checkpoints because they have never
  started. Slurm now projects 09:00 CEST July 23 for each job individually.
- This project's running BranchProof jobs currently occupy sixteen A100 GPUs.
  Each Dolmino stage needs eight, so the equal projections do not guarantee
  three simultaneous starts. No job, data, OOM, or storage failure occurred.

## Live Delta At 13:16 CEST On July 22

- Baseline eval/audit rows `24..28` completed and passed, bringing strict
  declaration-fixed acceptance to `26/30`. Raw NL train-1-to-20/25 review is
  clean where credited and shows ordinary truncation/malformed failures at
  depth 50. Rows 13/14/18/29 and their dependent audits remain active; full
  aggregate `3857769` is still gated.
- Conditioned-10k `3850119_16/17` passed structural and raw-generation review,
  bringing that family to `18/30`. Its complete train-1-to-15 provisional
  slice has OOD pass@1 answer/joint `0.4093/0.4080` for conditioned NL and
  `0.3214/0.1919` for conditioned logic; logic pass@16 answer rises to
  `0.7331` but joint remains `0.2708`. This partial is not report evidence.
- Accepted conditioned-50k `3850110_8`, shortcut `3850213_31..33`, and 32B
  `3850115_5` finals. Counts are now `9/15`, `34/42`, and `6/15`. Guarded
  cleanup removed six final-backed, live-reference-free checkpoints (about
  7.7 GiB) while retaining all finals, incomplete restart states, and the
  conditioned step-50,000 skip checkpoint.
- Hybrid row 9, conditioned rows 18--20, architecture eval rows 0/1,
  conditioned-50k rows 9--11, shortcut rows 34--36, baseline/recovery rows,
  and batch recovery rows `3872659_3..5` are active without a fatal signature.
  Exact batch recovery `3879713_9..11` remains account-GRES pending.
- Dolmino `3875825_0/3875828_1/3875831_2` remain full-node account-GRES
  pending with current individual projections of 22:19 CEST. Vault is
  `1187G/1000G` and `203k/200k` files. Successor `3880822` remains scheduled;
  no report regeneration is justified.

## Live Delta At 09:05 CEST On July 22

- Dolmino prerequisite `3875824` completed cleanly in `02:32:53`. Its packed
  gate accepted `5,111,201,524` tokens; retained packed data is about 39 GiB,
  and raw staging was deleted only after acceptance.
- First stages control `3875825_0`, formal `3875828_1`, and NL `3875831_2`
  are dependency-free `AssocGrpGRES` pending. Slurm's current per-job start
  projection is 19:57 CEST July 22. Because each requests a full eight-GPU
  node, identical projections do not guarantee simultaneous starts.
- No training checkpoint exists yet. Once allocated, each condition should
  require about 45--48 hours of compute and normally two 24-hour stages.

## Live Delta At 07:15 CEST On July 22

- Batch `3850114_9..11` timed out at 05:25 CEST with no final/checkpoint.
  Exact recovery `3879713_[9-11%3]` is account-GRES pending under the repaired
  250-step checkpoint policy. Eval `3872663` now depends on original batch,
  prior recovery `3872659`, and `3879713`.
- Baseline `3857767_21..23 -> 3857768_21..23` completed and audited cleanly,
  raising accepted coverage to `21/30`. The new NL train-1-to-15 three-seed
  raw review is clean through depth 25 and collapses at depth 50. Its OOD
  greedy answer/joint is `0.6488/0.6280`; pass@1 is `0.5648/0.5614`.
  Matched formal is lower at greedy/pass@1 and only overtakes answer coverage
  at pass@8/16; the full 30-row gate remains closed.
- Hybrid `3850118_6..8` and conditioned-10k `3850119_14/15` passed structural
  and raw-generation review. Counts are hybrid `6/30` and conditioned-10k
  `16/30`; later rows are active. Conditioned-50k finals are `8/15` after
  `3850110_3/6/7`; rows 8--10 are active. Shortcut remains `31/42`, 32B
  remains `5/15`, and focused live logs have no fatal signature.
- Dolmino stage-one `3875825_0/3875828_1/3875831_2` remains `AssocGrpGRES`
  pending. Vault is `1228G/1000G` and `203k/200k`; twelve active merge trees
  account for about 327 GiB and cannot be removed. CPU-only successor
  `3879709` remains scheduled; no report regeneration is justified.

## Live Delta At 01:16 CEST On July 22

- Accepted new 32B finals `3854837_1` and `3850115_4` and shortcut finals
  `3850213_28..30`. All have nonempty adapters/configs and terminal
  step-10,000 trainer states. Large-model SFT is `5/15`, shortcut is `31/42`;
  `3850115_5` and shortcut rows 31--33 are active without fatal signatures.
- Conditioned stage `3850110` used an immutable pre-repair payload and replayed
  final-backed rows. Rows 0--2 reached fresh clean step-50,000 finals. Stopped
  duplicate active tasks `3850110_4/5` after confirming their accepted
  July-15 finals were intact, and canceled their future duplicates
  `3850111_4/5` and `3850112_4/5`. Genuine unfinished row 3 remains active;
  six new partial duplicate checkpoints are cleanup candidates only. Retain
  row-0--2 `checkpoint-50000` until both later staged arrays terminate because
  their immutable payload still needs it to skip completed finals.
- Batch rows `3850114_9..11` are around step 1,740 at 20 hours. Their stored
  configs use `save_steps=20000`, so there is no checkpoint to preserve before
  timeout. Let them reach a terminal timeout, then recover only rows 9--11
  under the current 250-step checkpoint policy.
- Hybrid `3850118_6` completed sampled generation and is scoring;
  conditioned-10k `3850119_14` reached chunk 112. Baseline rows
  `3857767_21..23` are at chunks 93--94/112 on A100-80GB. No new terminal
  eval bundle is available, so artifact/raw-review/aggregate gates are
  unchanged at hybrid `3/30`, conditioned-10k `14/30`, and baseline `18/30`.
- Dolmino `3875825_0/3875828_1/3875831_2` remain dependency-free
  `AssocGrpGRES` pending. Vault is `1145G/1000G` and `203k/200k` files.
  CPU-only successor `3879087` is BeginTime-pending and remains required.

## Live Delta At 19:30 CEST On July 21

- Surface rows `3850105_24..26` completed and passed final/terminal-step
  gates, bringing surface SFT to `27/27`. Nine guarded final-backed Trainer
  checkpoints were removed with no live references; finals and evidence
  remain. Vault is now about `1017G/1000G` soft and `203k/200k` files.
- Hybrid eval `3850118_3/4/5` and conditioned-10k eval `3850119_12/13`
  completed. All five bundles pass the 448-prompt, 16-generation, 14-depth,
  `7/112`-chunk, 576-retained-row, fresh-constant, and validity-diagnostic
  structural gates. Completion counts are hybrid `3/30` and conditioned-10k
  `14/30`. Representative raw review found correct wrappers/extraction and no
  constant reuse; hybrid validity degrades OOD and both conditioned modalities
  collapse at depth 50. These partials do not open report acceptance.
- Active eval rows are hybrid `3850118_6..8` and conditioned-10k
  `3850119_14/15`. Conditioned-50k resumes `3850110_0..2` are around
  41.5k--43k/50k; 32B `3850115_4` and recovery `3854837_1` are around
  93%/95%; shortcut rows `3850213_28..30` are around 72--74%. Batch rows
  `3850114_9..11` are only around step 1,214 after 13.7 hours and may require
  exact post-timeout resume; no early cancellation or duplicate recovery was
  made. Focused logs show no fatal/OOM/quota/no-space signature.
- Dolmino first stages `3875825_0`, `3875828_1`, and `3875831_2` remain
  `AssocGrpGRES` pending on their only compatible full-node shape. The current
  shared start estimate is 07:43 CEST July 22. Baseline eval/recovery rows also
  remain account-GRES pending and acceptance is still `18/30`. CPU-only
  successor `3878297` is BeginTime-pending and must remain queued.

## Live Delta At 13:15 CEST On July 21

- Prerequisite `3875824` completed `0:0` in `02:32:53` and accepted a packed
  Dolmino stream with `5,111,201,524` tokens from `11,195,395` documents.
  The reproducible raw staging tree is absent after the gate; the packed
  Nanoset and stats remain. Production stage-one jobs `3875825_0`,
  `3875828_1`, and `3875831_2` are now dependency-free and `AssocGrpGRES`
  pending, with a current shared estimate of 02:55 CEST on July 22. Their
  full-node A100-80GB resource shape has no safe partition-widening option.
- Surface rows `3850105_21..23` and shortcut rows `3850213_25..27` completed
  and passed final adapter/config, zero-empty-file, and step-10,000 gates.
  Counts are now surface `24/27`, shortcut `28/42`, architecture `54/54`.
  Their next six training rows are active at roughly `22--34%`.
- Active hybrid/conditioned eval rows `3850118_3/4/5` and
  `3850119_12/13` reached sampled chunks `70--98/112`; batch SFT rows 9--11
  are around step 680, 32B rows around 55%, and conditioned-50k resumes around
  16.8--17.4k. Focused logs have no fatal/OOM/quota/no-space signature and no
  new final eval JSON/sample bundle is available.
- Vault is above both user soft quotas after the packed-data build at
  `1067.5G/1048.6G` and `202k/200k` files. Six terminal Trainer checkpoints
  are final-backed cleanup candidates; all incomplete restart states remain
  protected. Watcher `3875621` is CPU-only; CPU-only successor `3876636`
  remains BeginTime-pending because the plan is incomplete.

## Live Delta At 09:50 CEST On July 21

- Canceled unstarted NL smoke `3875623_1` and submitted the three 5B Dolmino
  production conditions. Normal-data extension/audit `3875824` is running on
  RTX Pro node `a2041`. It builds 5.1B packed normal tokens and deletes raw
  staging only after the packed-token gate.
- Full-node A100-80GB chains are control `3875825 -> 3875826 -> 3875827`,
  formal `3875828 -> 3875829 -> 3875830`, and NL
  `3875831 -> 3875832 -> 3875833`. Each targets 9,537 steps/5,000,134,656
  tokens with global batch 128, TP4/DP2, shared LR/schedule, 500-step restart
  states, and up to three 24-hour stages.
- All conditions use the same deterministic normal Nanoset/seed, but the 5B
  gate uses Nanotron weighted blending rather than a precomputed exactly
  slot-aligned stream. Do not claim exact per-normal-chunk pairing. Direct and
  instruction-tuned downstream dependency chains are not yet submitted.

## Live Delta At 09:20 CEST On July 21

- Matched control/formal audit passes: normalized configs are identical;
  both runs have 256 finite steps, the same warmup/LR/batch/topology, and
  comparable throughput. Post-warmup mean loss is `0.956750/0.852558`
  (control/formal). Decoded Nanoset documents match ordinary Dolmino and the
  intended tag-free neutral formal format. Shared but shifted loss blocks
  rule out a formal-mixture-specific batching failure. NL replacement remains
  account-GRES pending and no longer has a scheduler start estimate.
- Dolmino formal p5 completed `256/256` finite steps with realized mixture
  `0.949982/0.0500183` and about 15.6K tokens/s. Its nonzero Slurm exit was a
  post-training manifest failure caused by patching the wrapper while the old
  shell was active; the completion marker was reconstructed from the terminal
  log/config. Exact NL replacement `3875623_1` contains the port fix and
  currently estimates 09:26 CEST.
- Nineteen BranchProof jobs are active without a fatal signature. Six
  surface/shortcut rows are at `91--95%`; batch rows 9--11 are near `3%`;
  32B rows are near `30%`; five hybrid/conditioned eval rows have reached
  chunks `27--59/112`; and conditioned-50k stage-two rows 0--2 are running
  after the original timeout. Corrected baseline acceptance remains `18/30`.
- A final-backed/live-reference cleanup removed 20 newly terminal Trainer
  checkpoints (`17.06 GiB`) and preserved every incomplete restart state.
  User-wide Vault use fell from `970G` to `954G`; repo-owned use is now
  `394,804,989 KiB`. The user-wide file count remains `201k/200k`.

## Live Delta At 07:15 CEST On July 21

- Architecture rows `3850113_51/52/53` completed and passed the final artifact
  gate, bringing architecture SFT to `54/54`. Surface `3850105_21..23` and
  shortcut `3850213_25..27` are around `75--79%`; conditioned-50k
  `3850109_13/14` is around `17.6k/50k` with complete checkpoints through
  step 15,000 and the staged resume chain intact.
- A100 work now running is batch SFT `3850114_[9-11]`, hybrid eval
  `3850118_3`, 32B original SFT `3850115_4`, and 32B recovery `3854837_1`.
  Hybrid row 3 reached sampled chunk `32/112` without a fatal signature.
  Declaration-fixed baseline rows/recoveries remain account-GRES pending and
  the acceptance gate remains `18/30`.
- Dolmino logic confirmation `3872664_0` is healthy at step `189/256`, about
  15.5K tokens/s, finite loss/gradients, and realized `0.949982/0.0500183`
  mixing. Original NL row `3872664_1` failed before training with port-29500
  `EADDRINUSE` after co-location on the same node. The wrapper now derives a
  per-allocation torchrun port, and exact NL-only replacement `3875623_1` is
  submitted. Malformed pending submission `3875622` never started and was
  canceled.
- Vault is `861G/1000G` user-wide and about `285 GiB`/7,288 files repo-owned;
  the shared file count is `202k/200k`. Watcher `3875468` is CPU-only and its
  no-GRES successor `3875621` remains scheduled for 13:08 CEST.

## Live Delta At 01:20 CEST On July 21

- Fourteen report-matrix SFT tasks completed after 19:15: surface rows 15--20,
  shortcut rows 19--24, and architecture rows 49/50. Both targeted replacement
  arrays `3872657_[15-17]` and `3872658_[19-21]` are cleanly complete. All 14
  finals have nonempty adapter/config files, terminal step-10,000 trainer
  states, and zero empty final files. Current completion is surface `21/27`,
  shortcut `25/42`, and architecture `51/54`.
- Eleven A40 rows remain active: surface 21--23 and shortcut 25--27 are around
  `26--30%`, architecture 51/52/53 is around `80/45/45%`, and conditioned-50k
  13/14 is around `26%`. Scoped tails have no fatal, OOM, quota, or no-space
  signature. No new corrected eval JSON/sample JSONL exists.
- A100-80 baseline `3857767_[21-29]`, recoveries `3863525_[13-14]` and
  `3865321_18`, batch recovery `3872659`, hybrid eval `3872660`, and Dolmino
  p5 `3872664` remain account-GRES pending. Dolmino currently projects 01:49
  CEST. Dependent eval/audit/aggregate jobs remain correctly gated; no
  duplicate, dependency edit, or partition edit was made.
- Vault is `829G/1000G` user-wide and `253 GiB`/7,163 files repo-owned; the
  shared file count remains `201k/200k`. Watcher `3874799` is running CPU-only
  and its no-GRES successor `3875468` remains scheduled for 07:08 CEST.

## Live Delta At 19:15 CEST On July 20

- Architecture row `3850113_48` completed cleanly as raw job `3872671` at
  16:41 CEST (`07:49:57`, exit `0:0`). The Gemma-3-4B NL seed-3407 run has a
  step-10,000 trainer state and a nonempty `131,252,288`-byte final adapter.
  Row `3850113_51` backfilled immediately; architecture rows 49/50/51 are
  active around `85/85/22%`.
- Seventeen corrected BranchProof A40 rows remain active. Surface and shortcut
  10k rows are around `81--85%`; conditioned-50k rows 13/14 are around `16%`.
  Current log tails have no fatal, OOM, quota, or no-space signature. No new
  corrected eval JSON/sample JSONL exists, so no evidence gate opened.
- Baseline eval/recovery, batch recovery, hybrid eval, and Dolmino p5 remain
  A100 account-GRES pending or dependency-gated. Slurm estimates
  `3857767_21` at 21:59 CEST and `3872664_[0-1]` at 01:49 CEST on July 21.
  No job was duplicated or dependency-edited.
- Vault is `810G/1000G` user-wide with `201k/200k` files; the repo uses about
  `234 GiB` and 6,797 files. CPU-only watcher `3873713` is running and its
  no-GRES successor `3874799` remains scheduled for 01:08 CEST on July 21.

## Live Delta At 15:06 CEST On July 20

- Seventeen corrected BranchProof SFT rows remain active on A40s and no new
  project failure is recorded. Baseline/recovery/report evaluation and batch
  training remain account-GRES pending on A100; dependent jobs remain gated.
  Slurm currently estimates 17:34 CEST for Dolmino p5 `3872664` and selected
  A100 work.
- Project-only cleanup removed `55.59 GiB` of reproducible raw Nanotron JSONL
  staging data from Vault after validating all packed replacements, and
  `1.27 GiB` from 242 completed online W&B caches on Work. All packed Nanosets,
  offline W&B runs, current run caches, checkpoints, finals, and result
  evidence are retained.
- Repo-owned usage is about `231 GiB` on Vault and `68 GiB` on Work. User-wide
  Vault usage is `808G/1000G`; its file display remains `201k/200k`, but only
  `6,745` Vault files belong to this repo. No unrelated path was inspected for
  deletion or modified.

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
