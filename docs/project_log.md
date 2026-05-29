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
