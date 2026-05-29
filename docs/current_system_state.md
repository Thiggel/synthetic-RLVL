# Synthetic-RLVL Current Handoff

Last updated: 2026-05-29 16:05 CEST.

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
| Ongoing LaTeX report | `analysis/logic_cot_report_2026-05-25/logic_cot_report_2026-05-25.tex` |

## Current Scientific State

- Main HFSA OLMo-7B 3-seed depth-scaling grid is complete: 30 SFT rows and sparse final pass@k eval are done.
- Main result: logic is more depth/sample efficient at intermediate train ranges; `nl_exact` catches up at train-1-to-25 on joint validity. Depth-50 joint@16 at train-1-to-25 is similar: logic `0.417`, NL `0.427`.
- Bare-format OOD rerun is complete for the main OLMo grid. NL transfers much better to GSM8K numeric EM; logic transfers much better to context-provided HotpotQA/2Wiki/MuSiQue EM/F1. Treat those QA tasks as context-QA robustness, not as proof-chain evidence.
- Tiny Llama 20k and 100k scratch-pretraining runs are complete. They learn some train-band behavior, but strict OOD/depth-50 joint validity is essentially absent; use them as mechanism smoke tests, not as solved extrapolation.
- Architecture ablations for Qwen-2.5-7B, Qwen-2.5-1.5B, Gemma-3-4B, and OLMo-2-32B short-context are complete and report-tabulated.
- Full paired-family suites for `official_igsm`, `maze_navigation`, and hardened `attribute_constraints` are still running; use pilots only as diagnostics until the full suite eval finishes.

## Active Work

The active Slurm work is summarized in `docs/running_experiments.md`. Current high-priority active chains are:

- full paired-family suite: SFT `3672212` rows `0..41` complete, rows `42..47` running, rows `48..89` pending by throttle; eval `3672213` dependency-pending; oversight `3675380` running, next `3676517` begin-time pending
- trace-control ablations: `3661118 -> 3661119`
- shortcut-rate `0.3`: `3671431 -> 3671432`
- hybrid-order eval: `3670783`
- wordified length-control: SFT `3674875_[0-2]` running, eval `3674876` dependency-pending
- conditioned-dual 50k extension: 10k chunk `3674879` has rows `0..5` complete, `6..9` running, `10..14` pending; later chunks `3674880..3674883` and evals `3674884/3674885` pending
- shortcut-kind controls: build `3674886_[0-3]` complete and materialized roots exist, SFT `3674887_0..2` running, `3..23` pending, eval `3674888` dependency-pending
- ablation oversight: `3674892` running; next pass `3675833` begin-time pending
- paired full-suite audit at 16:05 CEST: build `3672195_0..2` remains complete with all three manifests present and 55/55 parquet paths per family; completed SFT rows `0..41` all have final adapter checkpoints; focused SFT/eval log scan found no Traceback/proof-validation failure/OOM/CUDA OOM/context failure/quota/no-space/DependencyNeverSatisfied/tokenizer/model-load/vLLM/node-failure/idle-GPU failure. Full-suite eval output directory has not been created yet, so there are still `0` eval JSONs. `a100` had idle compatible nodes, but paired pending rows were blocked by array throttle or dependency, so no partition edit was made.
- partition/log audit at 13:01 CEST: `a100` had idle nodes, but pending active jobs were blocked by array task limits, dependencies, or begin times rather than partition availability. Log scan found no unrecovered Traceback/OOM/CUDA OOM/quota/no-space/DependencyNeverSatisfied/tokenizer/model-load/vLLM/node-failure issues. No `scontrol update ... Partition=...` edit was made.

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
${HPCVAULT}/.venv_rlvl_posttrain/bin/python scripts/analysis/build_logic_cot_report.py
```

The 2026-05-29 report regeneration now includes an executive insight section, every generated PDF figure in the report bundle, OOD qualitative samples, and an index of generated CSV/PDF/Markdown artifacts.

The external report repo `../synthetic-RLVL-report` mirrors the generated bundle and should be pushed after every report update.

`pdflatex`/`latexmk` are not installed on the current node, so the `.tex` source is generated but not compiled here.

## Quick Commands

```bash
source ./scripts/env.sh
squeue -u c107fa12 -o '%.18i %.9P %.34j %.2t %.11M %.6D %.24E %R'
sacct -j 3672212,3672213,3675380,3676517,3661118,3661119,3671431,3671432,3670783,3674875,3674876,3674879,3674880,3674881,3674882,3674883,3674884,3674885,3674886,3674887,3674888,3674892 --format=JobIDRaw,JobName%34,State,Elapsed,ExitCode -n -P
```
