# Running Experiments

Last updated: 2026-05-29 20:05 CEST.

This file is the live Slurm dashboard. Historical details live in `docs/operational_history_2026-05-29.md`; planned-but-not-running work lives in `docs/experiment_backlog.md`.

## Active Slurm Chains

| Experiment | Jobs | State | Expected outputs | Notes |
| --- | --- | --- | --- | --- |
| Full paired-family suite | SFT `3672212_[0-89%6]`, eval `3672213_[0-89%4]`, oversight `3676517`, next `3677238` | SFT rows `0..41` complete, `42..47` running, `48..89` pending by throttle; eval dependency-pending on `afterok:3672212_*`; oversight `3676517` running, next `3677238` begin-time pending | `$WORK/synthetic-RLVL/passk_eval/paired_full_suite_sparse_20260528/` | Covers `official_igsm`, `maze_navigation`, hardened `attribute_constraints`, templates `logic,nl_exact`, train ranges `1..5/10/15/20/25`, seeds `3407..3409`. Build is complete with 55/55 manifest paths present for all three families; completed SFT rows `0..41` have final adapters; running rows `42..47` are `maze_navigation` train-1-to-15 and are making optimizer progress, with row `42` past checkpoint-5000; full-suite eval still has `0` JSON outputs and no output directory yet. |
| Trace-control ablations | SFT `3661118_[0-17%3]`, eval `3661119_[0-17%3]` | SFT rows `0..16` complete, row `17` running; eval dependency-pending | `passk_eval/hfsa_ablation_trace_controls_20260525/` | Templates: `terse_nl`, `rule_annotated_nl`, `pseudocode`, `shuffled_logic`, `invalid_logic`, `shuffled_nl`. |
| Shortcut-rate `0.3` | SFT `3671431_[0-5%3]`, eval `3671432_[0-5%3]` | SFT rows `0..2,4` complete, rows `3,5` running; eval dependency-pending | `$WORK/synthetic-RLVL/passk_eval/hfsa_shortcut_rate_ablation_20260525/` | Adds low/intermediate point to the existing `0.5/0.8` shortcut-rate curve. Existing `0.5/0.8` output count is `12`; no `0.3` eval outputs yet. |
| Hybrid order | targeted SFT `3670782`, eval `3670783_[0-29%4]` | SFT complete; 4 eval JSONs written, rows `4..7` running, `8..29` pending by throttle | `$WORK/synthetic-RLVL/passk_eval/hfsa_hybrid_order_full_20260525/` | Completed `think_formal` train-1-to-5 rows average OOD correct@16 `0.480`, formal joint@16 `0.022`, translated-NL joint@16 `0.297`, depth-50 correct@16 `0.219`; train-1-to-10 seed-3407 has OOD correct@16 `0.537`, formal joint@16 `0.275`, translated-NL joint@16 `0.300`. |
| Wordified length-control logic | SFT `3674875_[0-2%3]`, eval `3674876_[0-2%3]` | all three SFT rows complete; all three eval rows running | `$WORK/synthetic-RLVL/passk_eval/hfsa_logic_wordified_20260529/` | Cleaner equal-length control: predicates become word names such as `Teal(a)`, constants stay compact. Duplicate `3674877/3674878` was canceled. No eval JSONs yet. |
| Conditioned dual 50k | SFT chunks `3674879 -> 3674880 -> 3674881 -> 3674882 -> 3674883`, final eval `3674884`, checkpoint eval `3674885` | 10k chunk rows `0..10` complete, `11..14` running; later chunks/evals dependency-pending | `$WORK/synthetic-RLVL/passk_eval/hfsa_conditioned_dual_50k_20260529/`, `$WORK/synthetic-RLVL/passk_eval/hfsa_conditioned_dual_50k_intermediate_20260529/` | Five-stage resume chain because `a100` max walltime is one day. Checkpoint eval will support convergence curves at `10k..50k`. |
| Shortcut-kind controls | build `3674886_[0-3%2]`, SFT `3674887_[0-23%3]`, eval `3674888_[0-23%4]` | build complete; SFT rows `0..4` complete, `5..6` running, `7..23` pending by throttle; eval dependency-pending | `$WORK/synthetic-RLVL/passk_eval/hfsa_shortcut_kind_ablation_20260529/` | Tests `position` and `initial_marker` shortcuts at rates `0.5` and `0.8`, both templates, three seeds. Eval is shortcut-neutral. All four materialized roots have train-25 and val-50 parquet present. |
| New ablation oversight | current `3675833`, next `3676880` | `3675833` running; `3676880` begin-time pending | handoff updates and targeted recovery jobs if needed | Monitors the 2026-05-29 ablation wave and still-active predecessor ablations. |

## Partition Audit

Checked at 2026-05-29 20:05 CEST. Monitored paired pending rows were blocked by array task limits or dependencies, not partition availability. No `scontrol update JobId=<jobid> Partition=<partition1,partition2>` edit was made.

Unrelated visible `puzzle_*` jobs are not part of this handoff.

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
sacct -j 3672212,3672213,3675380,3676517,3677238,3661118,3661119,3671431,3671432,3670783,3674875,3674876,3674879,3674880,3674881,3674882,3674883,3674884,3674885,3674886,3674887,3674888,3675833,3676880 --format=JobIDRaw,JobName%34,State,Elapsed,ExitCode -n -P
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
  logs/hfsa_ablate_oversight_3675833.* logs/hfsa_ablate_oversight_3676880.*; do
  [ -f "$f" ] && echo "### $f" && tail -n 20 "$f"
done
```
