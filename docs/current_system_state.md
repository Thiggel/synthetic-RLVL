# Synthetic-RLVL Current Handoff

Last updated: 2026-05-30 18:32 CEST.

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
- The cleaner equal-length `logic_wordified` control is complete. It underperforms compact logic and `nl_exact`: train-1-to-25 mean OOD correct/joint@16 is `0.508/0.323`, and depth-50 correct/joint@16 is `0.344/0.094`.
- Trace-control `rule_annotated_nl` translated-validity metrics from `3661119_3..5` are stale: the NL-to-FOL translator did not strip `[rule: ...]` suffixes, so parse/joint were artificially zero. The translator now unwraps rule annotations and pseudocode `derive "..." using ...` lines; repair eval `3680004_[3-8%3]` is queued to overwrite `rule_annotated_nl` rows and rerun pseudocode rows.
- Full paired-family suites for `official_igsm`, `maze_navigation`, and hardened `attribute_constraints` are still running; use pilots only as diagnostics until the full suite eval finishes.

## Active Work

The active Slurm work is summarized in `docs/running_experiments.md`. Current high-priority active chains are:

- full paired-family suite: original SFT `3672212` rows `0..53` complete, rows `54/56/58` running, rows `55/57/59` failed silently, and rows `60..89` were canceled with signal `53`; replacement SFT `3682411_[55,57,59-89%6]` is running rows `55/57/59/60/61/62` with `63..89` pending by throttle; stale eval `3672213` was canceled and replacement eval `3682449_[0-89%4]` is dependency-pending on original running job IDs `3681398/3681503/3681586` plus replacement SFT `3682411`; current paired oversight `3680777` is running and next pass `3682410` is begin-time pending
- trace-control ablations: SFT `3661118` rows `0..17` complete; stale eval rows `3661119_6..8` were intentionally canceled after an evaluator bug was found; original eval `3661119` rows `0..5` and `9..11` complete, rows `12..14` running, rows `15..17` pending by throttle; repair eval `3680004_3..5` running at chunks `50/50/49` of `56`, with `3680004_6..8` pending by throttle to rerun `rule_annotated_nl` and `pseudocode` with the fixed translator
- shortcut-rate `0.3`: SFT `3671431` rows `0..5` complete; eval `3671432` complete; all `0.3` logic and NL rows now have 3-seed JSONs
- hybrid-order eval: `3670783` rows `0..10` complete, rows `11..14` running, rows `15..29` pending by throttle
- wordified length-control: SFT `3674875_[0-2]` and eval `3674876_[0-2]` complete with 3 JSONs; duplicates `3674877/3674878` were intentionally canceled
- conditioned-dual 50k extension: 10k chunk `3674879` rows `0..14` complete; 20k chunk `3674880` rows `0..14` complete; 30k chunk `3674881_0..3` running and `4..14` pending by throttle; later chunks `3674882..3674883` and evals `3674884/3674885` dependency-pending
- shortcut-kind controls: build `3674886_[0-3]` complete and materialized roots exist, SFT `3674887_0..17` complete, `18..20` running, `21..23` pending by throttle, eval `3674888` dependency-pending
- ablation oversight: `3679095` and refreshed pass `3680036` completed cleanly; stale next pass `3679878` canceled after the prompt update; current plan-driven pass `3680038` running and next pass `3680772` begin-time pending
- hybrid-order partial readout: completed `think_formal` train-1-to-5 rows average OOD correct@16 `0.480`, formal citation-free joint@16 `0.022`, translated-NL joint@16 `0.297`, depth-50 correct@16 `0.219`, depth-50 joint@16 `0.000`. Completed `think_formal` train-1-to-10 rows average OOD correct@16 `0.490`, formal joint@16 `0.249`, translated-NL joint@16 `0.296`, depth-50 correct@16 `0.354`, and depth-50 joint@16 `0.000`. Treat as partial until the remaining hybrid rows finish. `think_formal` is NL then formal; the report builder now labels this correctly and parses the pending `formal_think` rows.
- ablation readout at 2026-05-30 09:29 CEST: shortcut-rate `0.3` is now fully complete. Logic mean OOD correct/joint@16 is `0.892/0.598` and depth-50 correct/joint@16 is `0.844/0.375`; matched NL mean OOD correct/translated-joint@16 is `0.588/0.571` and depth-50 correct/translated-joint@16 is `0.458/0.438`. Across rates `0.3/0.5/0.8`, NL depth-50 joint falls `0.438 -> 0.312 -> 0.146`, while logic depth-50 joint is `0.375 -> 0.375 -> 0.417`; this supports the shortcut-robustness interpretation. Trace-control `rule_annotated_nl` is now three-seed complete with mean OOD correct/translated-joint@16 `0.579/0.000` and depth-50 correct/translated-joint@16 `0.365/0.000`, so it gets answers but fails the translated-validity check. Hybrid `think_formal` train-1-to-15 is now three-seed complete with mean OOD correct/formal-joint/translated-joint@16 `0.353/0.111/0.111` and depth-50 correct/formal-joint@16 `0.312/0.000`; train-1-to-20 has two seeds with mean OOD correct/formal-joint/translated-joint@16 `0.419/0.016/0.078` and depth-50 correct/formal-joint@16 `0.594/0.000`. Treat hybrid train-1-to-20 and train-1-to-25 as partial until the running rows finish.
- paired full-suite recovery at 2026-05-30 18:32 CEST: build `3672195_0..2` remains complete with all three manifests at 55 subsets and no missing parquet paths. Original SFT rows `0..53` have final adapters; original rows `54/56/58` are running on `maze_navigation` train-1-to-25 with latest parsed progress about `972/109/627` of `10000`; rows `55/57/59` failed with exit `1:0` and no traceback/OOM/quota/validation signature; rows `60..89` were immediately canceled/failed with signal `53`. Submitted targeted replacement SFT `3682411_[55,57,59-89%6]`; rows `55/57/59/60/61/62` are running and rows `63..89` are pending by array throttle. Canceled stale eval `3672213` because its `afterok:3672212_*` dependency could never satisfy, then submitted replacement eval `3682449_[0-89%4]` depending on original running job IDs `3681398/3681503/3681586` and replacement SFT `3682411`. There are still `0` eval JSONs/sample JSONLs and no eval output directory. Materialized-row audit over train-depth-25 and val-depth-50 rows for all three families found matching logic/NL prompts, expected `<formal>`/`<think>` wrappers, final `<answer>` tags, and strict proof validation passing; iGSM citation-free validation still fails for cited arithmetic substitution lines, so inspect per-family evaluator fields before using paired validity metrics. `a100` had idle nodes and replacement rows launched on `a100`; no partition widening or new science launch was made. Plan/backlog triggers remain deferred until replacement eval `3682449` writes outputs. Visible `puzzle_*` jobs are unrelated; no visible `tjepa_*` or `seqedit_*` jobs were present.
- ablation log audit at 2026-05-30 09:50 CEST: focused `squeue`/`sacct`/log scan found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, `DependencyNeverSatisfied`, tokenizer/model-load, vLLM failure, node failure, timeout, cancellation, or idle-GPU failure in the monitored HFSA ablation chains. Active trace-control eval rows `6..8` were sampling chunks `38/56`, `34/56`, and `25/56`; hybrid eval rows `11..14` were sampling chunks `92/112`, `71/112`, `58/112`, and `54/112`; conditioned-dual 20k rows `12..14` were at `12676/20000`, `10001/20000`, and `10001/20000`; shortcut-kind SFT rows `15..17` were at `6658/6116/5299` of `10000`. Pending monitored rows were blocked by array throttles, dependencies, or begin time, so no partition edit, dependency edit, cancellation, or resubmission was made. The old queued follow-up `3679878` was later canceled after the oversight prompt update.
- trace-control evaluator fix at 2026-05-30 10:13 CEST: manual inspection of `rule_annotated_nl` sample generations showed correct-looking traces such as `a is teal. [rule: R]`, but `nl_logic_parse` was zero because the translator treated the annotation as part of the attribute. `synthrlvl/natural_logic.py` now unwraps rule annotations and pseudocode lines before translation, and `tests/test_training_stack.py` verifies `RULE_ANNOTATED_NL` and `PSEUDOCODE` targets translate to valid logic. Verification: `tests/test_training_stack.py` passed (`26 passed`). Canceled stale running pseudocode eval rows `3661119_6..8` and submitted repair eval `3680004_[3-8%3]` with `FORCE_PASSK_EVAL=1`; `3680004_3..5` are running and `3680004_6..8` are pending by throttle. `3661119_3..5` completed metrics should be treated as stale until `3680004_3..5` overwrite them.
- oversight automation update at 2026-05-30 10:28 CEST: strengthened the active Codex oversight wrappers so each pass reads the live docs, backlog, and active plan; checks jobs/logs/outputs/partitions; inspects sample generations and evaluator assumptions before accepting metrics; analyzes newly completed outputs; regenerates/mirrors the LaTeX report; updates docs/backlog; and may submit the smallest triggered or recovery jobs. Replaced old queued prompt snapshots by canceling `3679878` and `3679358`, then submitted fresh oversight jobs `3680036` and `3680037`. Both started on `a100` and scheduled the next passes `3680038` and `3680039`.
- HFSA ablation oversight update at 2026-05-30 10:33 CEST: fresh `squeue`/expanded `sacct`/log/output scan found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, dependency, tokenizer/model-load, vLLM, node-failure, timeout, or idle-GPU failure in the monitored chains. Active rows are progressing: trace repair `3680004_3..5` is at chunk `18/56`, hybrid rows `3670783_11..14` are at chunks `96/76/66/63` of `112`, conditioned-dual 20k rows `3674880_12..14` are at `14585/10001/11007` of `20000`, shortcut-kind rows `3674887_15..17` are at `8424/7922/7107` of `10000`, and paired rows `3672212_48..53` are at `6256/5929/5906/6014/5977/5833` of `10000`. Row `3674880_13` has stale stdout/stderr but `srun --jobid=3679320` showed the allocated GPU active at `97%` utilization with about `63GB` used, so no idle recovery was submitted. Pending monitored rows are throttle/dependency/begin-time blocked; no partition edit, dependency edit, cancellation, or resubmission was made. Sample inspection confirmed wordified depth-50 generations often stay on the `<formal>` surface but drift into duplicated predicate declarations or invalid derivations; stale rule-annotated NL samples fail the old translator on unstripped NL/premise text, so repair eval outputs remain required before using those metrics. Visible `puzzle_*` jobs are unrelated; no visible `tjepa_*` or `seqedit_*` jobs were present.
- HFSA ablation oversight update at 2026-05-30 14:36 CEST: original trace-control `shuffled_logic` eval rows `3661119_9..11` completed and wrote 3 JSONs plus samples. Three-seed mean OOD correct/formal-joint@16 is `0.690/0.002`, and depth-50 correct/formal-joint@16 is `0.510/0.000`; report artifacts now include this row. Sample inspection found normal `<question>` prompts, `<formal>` generations, and `<answer>` extraction, but higher-depth proofs are often invalid or unparsable fragments; depth-50 can still answer correctly while failing citation-free validity, so this is a negative-control result rather than valid reasoning. Focused `squeue`/expanded `sacct`/log scan found no unrecovered Traceback, OOM/CUDA OOM, quota/no-space, dependency, tokenizer/model-load, vLLM, node-failure, timeout, or idle-GPU failure. Active progress: trace repair `3680004_3..5` chunks `50/50/49` of `56`, original trace rows `3661119_12..14` running with row `12` at chunk `29/56`, hybrid rows `3670783_11..14` at sampling/scoring state `112/99/89/90` of `112`, shortcut-kind rows `3674887_18..20` at `6798/6231/5167` of `10000`, and conditioned-dual 30k rows `3674881_0..3` running from checkpoint-20000. Pending monitored rows are throttle/dependency/begin-time blocked despite idle `a100` nodes, so no partition edit, dependency edit, cancellation, resubmission, or new science launch was made. Regenerated `analysis/logic_cot_report_2026-05-25/` and mirrored it to `../synthetic-RLVL-report`; verification found `64` generated PDFs, `64` PDF include references, and `53` CSV tables in both report trees. TeX compilation was not run because `latexmk`/`pdflatex` are unavailable.

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

The 2026-05-30 14:36 report regeneration added the completed `shuffled_logic` trace-control row to the trace-control table/figure, kept active experiment artifact status, and mirrored the bundle to `../synthetic-RLVL-report`. Verification found `64` generated PDFs, `64` `\includegraphics` PDF references, zero missing references, and `53` CSV tables in both report trees.

The external report repo `../synthetic-RLVL-report` mirrors the generated bundle and should be pushed after every report update.

`pdflatex`/`latexmk` are not installed on the current node, so the `.tex` source is generated but not compiled here.

## Quick Commands

```bash
source ./scripts/env.sh
squeue -u c107fa12 -o '%.18i %.9P %.34j %.2t %.11M %.6D %.24E %R'
sacct -j 3672212,3682411,3682449,3680777,3682410,3661118,3661119,3680004,3671431,3671432,3670783,3674875,3674876,3674879,3674880,3674881,3674882,3674883,3674884,3674885,3674886,3674887,3674888,3678051,3679095,3680036,3680038,3680772 --format=JobID,JobName%34,State,Elapsed,ExitCode -n -P
```
