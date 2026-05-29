# Project Log

Short dated notes for useful operational events, cleanup decisions, results updates, and handoff changes. Keep this concise; move bulky history to experiment-specific docs or archives.

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
