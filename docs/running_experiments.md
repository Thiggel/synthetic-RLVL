# Running Experiments

Last updated: 2026-05-31 02:42 CEST.

This file is the live Slurm dashboard. Historical details live in `docs/operational_history_2026-05-29.md`; planned-but-not-running work lives in `docs/experiment_backlog.md`.

## Active Slurm Chains

| Experiment | Jobs | State | Expected outputs | Notes |
| --- | --- | --- | --- | --- |
| Full paired-family suite | original SFT `3672212_[0-89%6]`, replacement SFT `3682411_[55,57,59-89%6]`, row-56 replacement SFT `3683070_[56%1]`, replacement eval `3682449_[0-89%4]`, last oversight `3683024`, next oversight `3683562` | Original SFT rows `0..53` complete, `54/58` running, `56` canceled after idle-GPU/stale-log diagnosis, `55/57/59` failed with exit `1:0`, `60..89` canceled/failed with signal `53`; replacement `3682411` rows `60..71` complete, `55/57/59/72/73/74` running, and `75..89` pending by throttle; targeted replacement `3683070_56` running on `a0833` with `--exclude=a0831`; stale eval `3672213` canceled; replacement eval `3682449` dependency-pending on `afterok:3681398:3683070:3681586:3682411`; paired oversight `3683024` completed cleanly and next pass `3683562` is begin-time pending | `$WORK/synthetic-RLVL/passk_eval/paired_full_suite_sparse_20260528/` | Covers `official_igsm`, `maze_navigation`, hardened `attribute_constraints`, templates `logic,nl_exact`, train ranges `1..5/10/15/20/25`, seeds `3407..3409`. Build is complete with 55/55 manifest paths present for all three families; `66/90` SFT rows currently have final adapters (`official_igsm` `30/30`, `maze_navigation` `24/30`, hard `attribute_constraints` `12/30`). Active progress at the last paired check was `3672212_54` `5063/10000`, `3672212_58` `4836/10000`, `3683070_56` `1758/10000`, `3682411_55` `3806/10000`, `3682411_57` `3877/10000`, `3682411_59` `3819/10000`, `3682411_72` `6279/10000`, `3682411_73` `5375/10000`, and `3682411_74` `4250/10000`. Focused active/recent paired SFT scans found no unrecovered Traceback, proof-validation failure, OOM/CUDA OOM, context error, quota/no-space, `DependencyNeverSatisfied`, tokenizer/model-load, vLLM, node-failure, timeout, or idle-GPU signature. Full-suite eval still has `0` JSON/sample outputs and no output directory, so analysis/report triggers remain deferred. A 2026-05-30 materialized-row audit across train-depth-25 and val-depth-50 for all three families found matching logic/NL prompts, correct target wrappers, strict proof validation passing, and the expected iGSM citation-free caveat for cited arithmetic substitutions. |
| Trace-control ablations | SFT `3661118_[0-17%3]`, original eval `3661119_[0-17%3]`, original repair `3680004_[3-8%3]`, replacement evals `3682459_[12,14-17%3]` and `3682460_[5-8%3]` | SFT complete. Original eval rows `0..5`, `9..11`, and `13` complete; rows `12/14/15..17` failed/killed. Original repair rows `3..4` complete and `5..8` failed/killed. Replacement `3682459` rows `12/14/15` complete and `16/17` running; replacement repair `3682460` rows `5/6/7` complete and `8` running. | `passk_eval/hfsa_ablation_trace_controls_20260525/` | `15/18` eval JSONs plus sample JSONLs are present. Complete controls: `terse_nl`, `rule_annotated_nl`, `shuffled_logic`, `invalid_logic`; partial controls: `pseudocode` seeds `3407/3408`, `shuffled_nl` seed `3407`. Means: `invalid_logic` OOD correct/formal-joint@16 `0.892/0.427`, depth-50 `0.750/0.146`, but grounded validity is zero; repaired `rule_annotated_nl` OOD correct/translated-joint@16 `0.575/0.485`, depth-50 `0.344/0.146`; `pseudocode` two-seed OOD correct/translated-joint@16 `0.406/0.334`, depth-50 `0.188/0.109`; `shuffled_nl` one-seed translated joint is `0.000` despite high parser coverage. |
| Shortcut-rate `0.3` | SFT `3671431_[0-5%3]`, eval `3671432_[0-5%3]` | complete; all 18 shortcut-rate JSONs exist across `0.3/0.5/0.8` and `logic/nl_exact` | `$WORK/synthetic-RLVL/passk_eval/hfsa_shortcut_rate_ablation_20260525/` | The `0.3` row is now fully matched: logic OOD correct/joint@16 `0.892/0.598`, depth-50 correct/joint@16 `0.844/0.375`; NL OOD correct/translated-joint@16 `0.588/0.571`, depth-50 correct/translated-joint@16 `0.458/0.438`. Across rates `0.3/0.5/0.8`, NL depth-50 joint falls `0.438 -> 0.312 -> 0.146`, while logic depth-50 joint is `0.375 -> 0.375 -> 0.417`. |
| Hybrid order | targeted SFT `3670782`, original eval `3670783_[0-29%4]`, replacement eval `3682461_[13,15-29%4]` | SFT complete; 14 eval JSONs written. Original eval rows `0..12` and `14` complete, row `13` failed, and rows `15..29` failed/killed. Replacement `3682461` is running rows `13/15/16/17` with `18..29` pending by throttle. | `$WORK/synthetic-RLVL/passk_eval/hfsa_hybrid_order_full_20260525/` | Completed `think_formal` is three-seed complete through train-1-to-20 and two-seed partial at train-1-to-25. Means: train-1-to-5 OOD correct/formal-joint/translated-joint@16 `0.480/0.022/0.297`, train-1-to-10 `0.490/0.249/0.296`, train-1-to-15 `0.353/0.111/0.111`, train-1-to-20 `0.434/0.028/0.148`, train-1-to-25 partial `0.584/0.188/0.459`; depth-50 formal joint remains `0.000` for these rows. Samples confirm the intended `<think>` then `<formal>` surface and normal answer extraction, but formal validity remains fragile. |
| Wordified length-control logic | SFT `3674875_[0-2%3]`, eval `3674876_[0-2%3]` | complete; 3 eval JSONs written | `$WORK/synthetic-RLVL/passk_eval/hfsa_logic_wordified_20260529/` | Cleaner equal-length control: predicates become word names such as `Teal(a)`, constants stay compact. Duplicate `3674877/3674878` was canceled. Mean OOD correct/joint@16 `0.508/0.323`; depth-50 correct/joint@16 `0.344/0.094`. |
| Conditioned dual 50k | SFT chunks `3674879 -> 3674880 -> 3674881/3682457/3682492 -> 3674882 -> 3674883`, final eval `3674884`, checkpoint eval `3674885` | 10k and 20k chunks complete. Original 30k chunk `3674881` has rows `0..2` and `4` complete, rows `3/5/6` failed, and `7..14` killed. Replacement rows `3682457_3/6/7/8/9/11` and `3682492_5` are complete; `3682457_10/12/13/14` are running. `3674882` waits only on `afterok:3682457`. | `$WORK/synthetic-RLVL/passk_eval/hfsa_conditioned_dual_50k_20260529/`, `$WORK/synthetic-RLVL/passk_eval/hfsa_conditioned_dual_50k_intermediate_20260529/` | Five-stage resume chain because `a100` max walltime is one day. Replacement rows use `MAX_STEPS=30000,EVAL_STEPS=30001` and resume from existing checkpoints where present; no final or checkpoint eval JSONs yet. |
| Shortcut-kind controls | build `3674886_[0-3%2]`, original SFT `3674887_[0-23%3]`, replacement SFT `3682458_[22%1]`, eval `3674888_[0-23%4]` | build and SFT complete after replacement `3682458_22`; eval rows `3674888_0..3` are running and `4..23` are pending by array throttle. | `$WORK/synthetic-RLVL/passk_eval/hfsa_shortcut_kind_ablation_20260529/` | Tests `position` and `initial_marker` shortcuts at rates `0.5` and `0.8`, both templates, three seeds. Eval is shortcut-neutral. All four materialized roots have train-25 and val-50 parquet present; no eval JSONs yet. |
| New ablation oversight | current oversight `3683023`, next oversight `3683563` | current plan-driven oversight `3683023` running; next pass `3683563` begin-time pending | handoff updates, result analysis, report updates, targeted recovery jobs, and triggered backlog submissions if appropriate | Monitors the 2026-05-29 ablation wave and still-active predecessor ablations. This pass regenerated/mirrored the report for new trace-control JSONs and made no scheduler edit, partition edit, cancellation, resubmission, or broad new science launch. |

## Partition Audit

Checked at 2026-05-31 02:42 CEST. Pending paired, shortcut-kind, hybrid, and trace-control rows are blocked by array throttles; paired eval `3682449` and conditioned 40k/eval rows are blocked by dependencies; oversight follow-ups are blocked by begin time. `a100` has idle compatible nodes, but there is no compatible freer partition issue to solve and no `scontrol update JobId=<jobid> Partition=<partition1,partition2>` edit was appropriate.

Unrelated visible `puzzle_*` jobs are not part of this handoff. No visible `tjepa_*` or `seqedit_*` jobs were present in the queue check.

## Watch Rules

- If a row fails with `NODE_FAIL`, submit the smallest replacement and exclude the bad node if one is implicated.
- If a row fails with OOM or context length, patch the wrapper and resubmit only failed rows.
- During housekeeping, inspect whether compatible partitions are freer or likely to start sooner. If a pending job can safely run on more partitions, widen it with `scontrol update JobId=<jobid> Partition=<partition1,partition2>` and record the edit here plus in `docs/current_system_state.md`.
- If an eval family completes, rerun `scripts/analysis/build_logic_cot_report.py` and update `docs/current_system_state.md`, this file, and the relevant experiment doc.
- Do not launch broad new science jobs from oversight unless needed to recover these chains.

## Commands

```bash
source ./scripts/env.sh
squeue -u c107fa12 -o '%.18i %.9P %.34j %.2t %.11M %.6D %.24E %R'
sacct -j 3672212,3682411,3683070,3682449,3680777,3682410,3683024,3683562,3661118,3661119,3680004,3682459,3682460,3671431,3671432,3670783,3682461,3674875,3674876,3674879,3674880,3674881,3682457,3682492,3674882,3674883,3674884,3674885,3674886,3674887,3682458,3674888,3682409,3683023,3683563 --format=JobID,JobName%34,State,Elapsed,ExitCode -n -P
```

Useful log tails:

```bash
for f in \
  logs/sft_pair_full_3672212_*.out logs/sft_pair_full_3682411_*.out logs/sft_pair_full_3683070_*.out logs/pair_full_eval_3682449_*.out \
  logs/sft_hfsa_trace_ctl_3661118_*.out logs/hfsa_trace_ctl_eval_3661119_*.out \
  logs/hfsa_trace_ctl_eval_3682459_*.out logs/hfsa_trace_ctl_eval_3682460_*.out \
  logs/sft_hfsa_shortcut_3671431_*.out logs/hfsa_shortcut_eval_3671432_*.out \
  logs/hfsa_hybrid_eval_3670783_*.out logs/hfsa_hybrid_eval_3682461_*.out \
  logs/sft_hfsa_word_3674875_*.out logs/sft_hfsa_word_3674875_*.err logs/hfsa_word_eval_3674876_*.out \
  logs/sft_hfsa_cond50k_3674879_*.out logs/sft_hfsa_cond50k_3674879_*.err logs/sft_hfsa_cond50k_3682457_*.out logs/sft_hfsa_cond50k_3682492_*.out logs/hfsa_cond50k_eval_3674884_*.out logs/hfsa_cond50k_ckpt_3674885_*.out \
  logs/build_hfsa_shkind_3674886_*.out logs/sft_hfsa_shortkind_3674887_*.out logs/sft_hfsa_shortkind_3674887_*.err logs/sft_hfsa_shortkind_3682458_*.out logs/hfsa_shortkind_eval_3674888_*.out \
  logs/hfsa_ablate_oversight_3678051.* logs/hfsa_ablate_oversight_3679095.* logs/hfsa_ablate_oversight_3680036.* logs/hfsa_ablate_oversight_3680038.* logs/hfsa_ablate_oversight_3682409.* \
  logs/paired_full_oversight_3680037.* logs/paired_full_oversight_3680039.* logs/paired_full_oversight_3682410.*; do
  [ -f "$f" ] && echo "### $f" && tail -n 20 "$f"
done
```
