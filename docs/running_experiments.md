# Running Experiments

Last updated: 2026-05-31 10:45 CEST.

This file is the live Slurm dashboard. Historical details live in `docs/operational_history_2026-05-29.md`; planned-but-not-running work lives in `docs/experiment_backlog.md`.

## Active Slurm Chains

| Experiment | Jobs | State | Expected outputs | Notes |
| --- | --- | --- | --- | --- |
| Full paired-family suite | original SFT `3672212_[0-89%6]`, replacement SFT `3682411_[55,57,59-89%6]`, row-56 replacement SFT `3683070_[56%1]`, replacement eval `3682449_[0-89%4]`, completed oversight `3683967`, next oversight `3684369` | Original SFT rows `0..53` complete, `54/58` running, `56` canceled after idle-GPU/stale-log diagnosis, `55/57/59` failed with exit `1:0`, `60..89` canceled/failed with signal `53`; replacement `3682411` rows `60..80` complete, `55/57/59/81/82/83` running, and `84..89` pending by throttle; targeted replacement `3683070_56` running on `a0833` with `--exclude=a0831`; stale eval `3672213` canceled; replacement eval `3682449` dependency-pending on `afterok:3681398:3683070:3681586:3682411`; paired oversight `3683967` completed cleanly and next pass `3684369` is begin-time pending | `$WORK/synthetic-RLVL/passk_eval/paired_full_suite_sparse_20260528/` | Covers `official_igsm`, `maze_navigation`, hardened `attribute_constraints`, templates `logic,nl_exact`, train ranges `1..5/10/15/20/25`, seeds `3407..3409`. Build is complete with 55/55 manifest paths present for all three families; `75/90` SFT rows currently have final adapters (`official_igsm` `30/30`, `maze_navigation` `24/30`, hard `attribute_constraints` `21/30`). Active progress at the latest paired check was `3672212_54` `9234/10000`, `3672212_58` `9142/10000`, `3683070_56` `5897/10000`, `3682411_55` `8011/10000`, `3682411_57` `8235/10000`, `3682411_59` `8172/10000`, `3682411_81` `5659/10000`, `3682411_82` `3882/10000`, and `3682411_83` `1553/10000`. Focused active/recent paired scans found no unrecovered Traceback, proof-validation failure, OOM/CUDA OOM, context error, quota/no-space, `DependencyNeverSatisfied`, tokenizer/model-load, vLLM, node-failure, timeout, or idle-GPU signature. Full-suite eval still has `0` JSON/sample outputs and no output directory, so analysis/report triggers remain deferred. A 2026-05-31 materialized/gold-target audit across sampled train-depth-25 and val-depth-50 rows for all three families found matched logic/NL prompts, correct target wrappers, strict proof validation, and gold logic validity. Gold paired NL targets answer and format correctly but currently have `nl_logic_parse=0.0` and translated validity `0.0` in sampled families, so paired NL validity metrics remain blocked on the backlog translator-improvement item. |
| Trace-control ablations | SFT `3661118_[0-17%3]`, original eval `3661119_[0-17%3]`, original repair `3680004_[3-8%3]`, replacement evals `3682459_[12,14-17%3]` and `3682460_[5-8%3]` | SFT complete. Original eval rows `0..5`, `9..11`, and `13` complete; rows `12/14/15..17` failed/killed. Original repair rows `3..4` complete and `5..8` failed/killed. Replacement `3682459` rows `12/14/15/16/17` and replacement repair `3682460` rows `5/6/7/8` are complete. | `passk_eval/hfsa_ablation_trace_controls_20260525/` | `18/18` eval JSONs plus sample JSONLs are present. Means: `invalid_logic` OOD correct/formal-joint@16 `0.892/0.427`, depth-50 `0.750/0.146`, but grounded validity is zero; repaired `rule_annotated_nl` OOD correct/translated-joint@16 `0.575/0.485`, depth-50 `0.344/0.146`; `pseudocode` OOD correct/translated-joint@16 `0.544/0.479`, depth-50 `0.208/0.104`; `shuffled_nl` OOD correct/translated-joint@16 `0.490/0.000`, depth-50 `0.344/0.000`, with high parser coverage but invalid proof order. |
| Shortcut-rate `0.3` | SFT `3671431_[0-5%3]`, eval `3671432_[0-5%3]` | complete; all 18 shortcut-rate JSONs exist across `0.3/0.5/0.8` and `logic/nl_exact` | `$WORK/synthetic-RLVL/passk_eval/hfsa_shortcut_rate_ablation_20260525/` | The `0.3` row is now fully matched: logic OOD correct/joint@16 `0.892/0.598`, depth-50 correct/joint@16 `0.844/0.375`; NL OOD correct/translated-joint@16 `0.588/0.571`, depth-50 correct/translated-joint@16 `0.458/0.438`. Across rates `0.3/0.5/0.8`, NL depth-50 joint falls `0.438 -> 0.312 -> 0.146`, while logic depth-50 joint is `0.375 -> 0.375 -> 0.417`. |
| Hybrid order | targeted SFT `3670782`, original eval `3670783_[0-29%4]`, replacement eval `3682461_[13,15-29%4]` | SFT complete; 15 eval JSONs written. Original eval rows `0..12` and `14` complete, row `13` failed, and rows `15..29` failed/killed. Replacement `3682461_13` completed; rows `15/16/17/18` are running and `19..29` are pending by throttle. | `$WORK/synthetic-RLVL/passk_eval/hfsa_hybrid_order_full_20260525/` | Completed `think_formal` is three-seed complete through train-1-to-25. Means: train-1-to-5 OOD correct/formal-joint/translated-joint@16 `0.480/0.022/0.297`, train-1-to-10 `0.490/0.249/0.296`, train-1-to-15 `0.353/0.111/0.111`, train-1-to-20 `0.434/0.028/0.148`, train-1-to-25 `0.573/0.204/0.419`; depth-50 formal joint remains `0.000` for these rows, with train-1-to-25 depth-50 translated joint `0.135`. Samples confirm the intended `<think>` then `<formal>` surface and normal answer extraction, but formal validity remains fragile. |
| Wordified length-control logic | SFT `3674875_[0-2%3]`, eval `3674876_[0-2%3]` | complete; 3 eval JSONs written | `$WORK/synthetic-RLVL/passk_eval/hfsa_logic_wordified_20260529/` | Cleaner equal-length control: predicates become word names such as `Teal(a)`, constants stay compact. Duplicate `3674877/3674878` was canceled. Mean OOD correct/joint@16 `0.508/0.323`; depth-50 correct/joint@16 `0.344/0.094`. |
| Conditioned dual 50k | SFT chunks `3674879 -> 3674880 -> 3674881/3682457/3682492 -> 3674882 -> 3674883`, final eval `3674884`, checkpoint eval `3674885` | 10k, 20k, and repaired 30k chunks complete. 40k chunk `3674882` is active: rows `0/1/2` complete, rows `3/4/5/6` running, and `7..14` pending by throttle. 50k chunk `3674883`, final eval `3674884`, and checkpoint eval `3674885` are dependency-pending. | `$WORK/synthetic-RLVL/passk_eval/hfsa_conditioned_dual_50k_20260529/`, `$WORK/synthetic-RLVL/passk_eval/hfsa_conditioned_dual_50k_intermediate_20260529/` | Five-stage resume chain because `a100` max walltime is one day. Row `3674882_2` reached `40000/40000` and completed cleanly during this pass; active rows show current optimizer progress and no fatal signatures. No final or checkpoint eval JSONs yet. |
| Shortcut-kind controls | build `3674886_[0-3%2]`, original SFT `3674887_[0-23%3]`, replacement SFT `3682458_[22%1]`, eval `3674888_[0-23%4]` | build and SFT complete after replacement `3682458_22`; eval rows `3674888_0..4/6/7` complete, rows `5/8/9/10` running, and `11..23` pending by array throttle. | `$WORK/synthetic-RLVL/passk_eval/hfsa_shortcut_kind_ablation_20260529/` | Tests `position` and `initial_marker` shortcuts at rates `0.5` and `0.8`, both templates, three seeds. Eval is shortcut-neutral. First `7/24` JSONs are report-ingested: `position` rate `0.5` logic three-seed OOD correct/joint@16 `0.900/0.619`, depth-50 `0.844/0.312`; `position` rate `0.8` logic two-seed OOD `0.884/0.647`, depth-50 `0.766/0.328`; matched `nl_exact` rate `0.5` two-seed OOD `0.459/0.303`, depth-50 `0.422/0.219`. Treat as provisional until remaining rows finish. |
| New ablation oversight | current oversight `3683966`, next oversight `3684370` | previous plan-driven pass `3683563` completed; current pass `3683966` running and next pass `3684370` begin-time pending | handoff updates, result analysis, report updates, targeted recovery jobs, and triggered backlog submissions if appropriate | Monitors the 2026-05-29 ablation wave and still-active predecessor ablations. This pass regenerated/mirrored the report for trace-control `18/18`, shortcut-kind `7/24`, and hybrid `15/30`, patched report-builder hybrid caption prose, and made no scheduler edit, partition edit, cancellation, resubmission, or broad new science launch. |

## Partition Audit

Checked at 2026-05-31 10:45 CEST. Pending paired SFT and shortcut-kind/hybrid/conditioned SFT rows are blocked by array throttles, paired and conditioned evals are blocked by dependencies, and oversight follow-ups `3684369/3684370` are blocked by begin time. `a100` has idle compatible nodes, but there is no compatible freer partition issue to solve and no `scontrol update JobId=<jobid> Partition=<partition1,partition2>` edit was appropriate.

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
sacct -j 3672212,3682411,3683070,3682449,3680777,3682410,3683024,3683562,3683967,3684369,3661118,3661119,3680004,3682459,3682460,3671431,3671432,3670783,3682461,3674875,3674876,3674879,3674880,3674881,3682457,3682492,3674882,3674883,3674884,3674885,3674886,3674887,3682458,3674888,3682409,3683023,3683563,3683966,3684370 --format=JobID,JobName%34,State,Elapsed,ExitCode -n -P
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
  logs/hfsa_ablate_oversight_3678051.* logs/hfsa_ablate_oversight_3679095.* logs/hfsa_ablate_oversight_3680036.* logs/hfsa_ablate_oversight_3680038.* logs/hfsa_ablate_oversight_3682409.* logs/hfsa_ablate_oversight_3683563.* logs/hfsa_ablate_oversight_3683966.* logs/hfsa_ablate_oversight_3684370.* \
  logs/paired_full_oversight_3680037.* logs/paired_full_oversight_3680039.* logs/paired_full_oversight_3682410.* logs/paired_full_oversight_3683024.* logs/paired_full_oversight_3683562.* logs/paired_full_oversight_3683967.* logs/paired_full_oversight_3684369.*; do
  [ -f "$f" ] && echo "### $f" && tail -n 20 "$f"
done
```
