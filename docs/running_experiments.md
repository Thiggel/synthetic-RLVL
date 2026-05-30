# Running Experiments

Last updated: 2026-05-30 14:31 CEST.

This file is the live Slurm dashboard. Historical details live in `docs/operational_history_2026-05-29.md`; planned-but-not-running work lives in `docs/experiment_backlog.md`.

## Active Slurm Chains

| Experiment | Jobs | State | Expected outputs | Notes |
| --- | --- | --- | --- | --- |
| Full paired-family suite | SFT `3672212_[0-89%6]`, eval `3672213_[0-89%4]`, previous oversight `3680037`, current oversight `3680039`, next oversight `3680777` | SFT rows `0..47` complete, `48..53` running, `54..89` pending by throttle; eval dependency-pending on `afterok:3672212_*`; oversight `3680037` complete; current plan-driven oversight `3680039` running; next pass `3680777` begin-time pending | `$WORK/synthetic-RLVL/passk_eval/paired_full_suite_sparse_20260528/` | Covers `official_igsm`, `maze_navigation`, hardened `attribute_constraints`, templates `logic,nl_exact`, train ranges `1..5/10/15/20/25`, seeds `3407..3409`. Build is complete with 55/55 manifest paths present for all three families; completed SFT rows `0..47` have final adapters. Running rows `48..53` are `maze_navigation` train-1-to-20 (`logic` and `nl_exact`, seeds `3407..3409`) with latest parsed progress `8829/8511/8479/8728/8676/8556` of `10000`, and all six have `checkpoint-5000`. Full-suite eval still has `0` JSON outputs and no output directory yet. A 2026-05-30 14:31 sample materialized-row audit across train-1-to-20 and val-depth-50 for all three families found matching logic/NL prompts, correct target wrappers, strict proof validation passing, and the expected iGSM citation-free caveat for cited arithmetic substitutions. |
| Trace-control ablations | SFT `3661118_[0-17%3]`, original eval `3661119_[0-17%3]`, repair eval `3680004_[3-8%3]` | SFT rows `0..17` complete; original eval rows `0..5` complete, rows `6..8` intentionally canceled, rows `9..11` running, rows `12..17` pending by throttle; repair eval `3680004_3..5` running at chunk `18/56`, `3680004_6..8` pending by throttle | `passk_eval/hfsa_ablation_trace_controls_20260525/` | Templates: `terse_nl`, `rule_annotated_nl`, `pseudocode`, `shuffled_logic`, `invalid_logic`, `shuffled_nl`. Manual sample inspection found the `rule_annotated_nl` zero translated-joint result was an evaluator artifact: lines like `a is teal. [rule: R]` were parsed as attribute `teal. [rule: r]`. The translator now strips `[rule: ...]` and unwraps pseudocode `derive "..." using ...`; tests pass. Treat `rule_annotated_nl` translated-validity numbers from `3661119_3..5` as stale until `3680004_3..5` overwrite them. `3680004_6..8` reruns pseudocode with the fixed evaluator. Stale samples still demonstrate the old parser failure, so do not cite the zero translated-joint rows. |
| Shortcut-rate `0.3` | SFT `3671431_[0-5%3]`, eval `3671432_[0-5%3]` | complete; all 18 shortcut-rate JSONs exist across `0.3/0.5/0.8` and `logic/nl_exact` | `$WORK/synthetic-RLVL/passk_eval/hfsa_shortcut_rate_ablation_20260525/` | The `0.3` row is now fully matched: logic OOD correct/joint@16 `0.892/0.598`, depth-50 correct/joint@16 `0.844/0.375`; NL OOD correct/translated-joint@16 `0.588/0.571`, depth-50 correct/translated-joint@16 `0.458/0.438`. Across rates `0.3/0.5/0.8`, NL depth-50 joint falls `0.438 -> 0.312 -> 0.146`, while logic depth-50 joint is `0.375 -> 0.375 -> 0.417`. |
| Hybrid order | targeted SFT `3670782`, eval `3670783_[0-29%4]` | SFT complete; 11 eval JSONs written, rows `11..14` running, `15..29` pending by throttle | `$WORK/synthetic-RLVL/passk_eval/hfsa_hybrid_order_full_20260525/` | Completed `think_formal` train-1-to-5 rows average OOD correct@16 `0.480`, formal joint@16 `0.022`, translated-NL joint@16 `0.297`, depth-50 correct@16 `0.219`; train-1-to-10 rows average OOD correct@16 `0.490`, formal joint@16 `0.249`, translated-NL joint@16 `0.296`, depth-50 correct@16 `0.354`; train-1-to-15 is now three-seed complete with mean OOD correct/formal-joint/translated-joint@16 `0.353/0.111/0.111`, depth-50 correct/formal-joint@16 `0.312/0.000`; train-1-to-20 has seeds `3407/3408` complete with OOD correct/formal-joint/translated-joint@16 `0.419/0.016/0.078`, depth-50 correct/formal-joint@16 `0.594/0.000`. Running rows `11..14` are sampling chunks `96/112`, `76/112`, `66/112`, and `63/112`. `think_formal` is NL then formal; the report builder now labels it correctly and parses the pending `formal_think` rows. |
| Wordified length-control logic | SFT `3674875_[0-2%3]`, eval `3674876_[0-2%3]` | complete; 3 eval JSONs written | `$WORK/synthetic-RLVL/passk_eval/hfsa_logic_wordified_20260529/` | Cleaner equal-length control: predicates become word names such as `Teal(a)`, constants stay compact. Duplicate `3674877/3674878` was canceled. Mean OOD correct/joint@16 `0.508/0.323`; depth-50 correct/joint@16 `0.344/0.094`. |
| Conditioned dual 50k | SFT chunks `3674879 -> 3674880 -> 3674881 -> 3674882 -> 3674883`, final eval `3674884`, checkpoint eval `3674885` | 10k chunk rows `0..14` complete; 20k chunk rows `0..11` complete, `12..14` running; later chunks/evals dependency-pending | `$WORK/synthetic-RLVL/passk_eval/hfsa_conditioned_dual_50k_20260529/`, `$WORK/synthetic-RLVL/passk_eval/hfsa_conditioned_dual_50k_intermediate_20260529/` | Five-stage resume chain because `a100` max walltime is one day. Running 20k rows are progressing from checkpoint-10000; latest parsed progress is `14585/20000`, `10001/20000`, and `11007/20000`. Row `3674880_13` has stale stdout/stderr, but `srun --jobid=3679320` showed the allocated GPU active at `97%` utilization and about `63GB` used, so no idle recovery was submitted. No final or checkpoint eval JSONs yet. Checkpoint eval will support convergence curves at `10k..50k`. |
| Shortcut-kind controls | build `3674886_[0-3%2]`, SFT `3674887_[0-23%3]`, eval `3674888_[0-23%4]` | build complete; SFT rows `0..14` complete, `15..17` running, `18..23` pending by throttle; eval dependency-pending | `$WORK/synthetic-RLVL/passk_eval/hfsa_shortcut_kind_ablation_20260529/` | Tests `position` and `initial_marker` shortcuts at rates `0.5` and `0.8`, both templates, three seeds. Eval is shortcut-neutral. All four materialized roots have train-25 and val-50 parquet present. Running rows `15..17` are making optimizer progress with latest parsed progress `8424/7922/7107` of `10000`. |
| New ablation oversight | latest completed `3679095`, current oversight `3680036`, next oversight `3680038` | `3679095` completed cleanly; stale queued oversight `3679878` canceled after the prompt update; fresh plan-driven oversight `3680036` running; next pass `3680038` begin-time pending | handoff updates, result analysis, report updates, targeted recovery jobs, and triggered backlog submissions if appropriate | Monitors the 2026-05-29 ablation wave and still-active predecessor ablations. The wrapper now requires every pass to read the backlog/active plan, inspect sample generations/evaluator assumptions before accepting metrics, create justified figures/tables when results appear, update handoff docs/report, and submit only the smallest safe triggered or recovery jobs. |

## Partition Audit

Checked at 2026-05-30 14:31 CEST. Paired pending rows were blocked by the `3672212` array task limit or the eval dependency, not by a scarce compatible partition; `a100` also had idle nodes. Oversight `3680039` is running on `a100` and next paired pass `3680777` is begin-time pending. No `scontrol update JobId=<jobid> Partition=<partition1,partition2>` edit was appropriate.

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
sacct -j 3672212,3672213,3678335,3680037,3680039,3680777,3661118,3661119,3680004,3671431,3671432,3670783,3674875,3674876,3674879,3674880,3674881,3674882,3674883,3674884,3674885,3674886,3674887,3674888,3678051,3679095,3680036,3680038 --format=JobID,JobName%34,State,Elapsed,ExitCode -n -P
```

Useful log tails:

```bash
for f in \
  logs/sft_pair_full_3672212_*.out logs/pair_full_eval_3672213_*.out \
  logs/sft_hfsa_trace_ctl_3661118_*.out logs/hfsa_trace_ctl_eval_3661119_*.out \
  logs/sft_hfsa_shortcut_3671431_*.out logs/hfsa_shortcut_eval_3671432_*.out \
  logs/hfsa_hybrid_eval_3670783_*.out \
  logs/sft_hfsa_word_3674875_*.out logs/sft_hfsa_word_3674875_*.err logs/hfsa_word_eval_3674876_*.out \
  logs/sft_hfsa_cond50k_3674879_*.out logs/sft_hfsa_cond50k_3674879_*.err logs/hfsa_cond50k_eval_3674884_*.out logs/hfsa_cond50k_ckpt_3674885_*.out \
  logs/build_hfsa_shkind_3674886_*.out logs/sft_hfsa_shortkind_3674887_*.out logs/sft_hfsa_shortkind_3674887_*.err logs/hfsa_shortkind_eval_3674888_*.out \
  logs/hfsa_ablate_oversight_3678051.* logs/hfsa_ablate_oversight_3679095.* logs/hfsa_ablate_oversight_3680036.* \
  logs/paired_full_oversight_3680037.* logs/paired_full_oversight_3680039.*; do
  [ -f "$f" ] && echo "### $f" && tail -n 20 "$f"
done
```
