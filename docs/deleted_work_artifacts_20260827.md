# $WORK cleanup 2026-08-27 — deleted artifacts and how to regenerate

Context: $HOME was over its 100 G soft quota and $WORK/synthetic-RLVL had grown to
865 G, blocking headroom for the long-window midtrain arms. Same discipline as
`deleted_sft_finals_20260826.md`: record before removing.

All deletions below were checked against `squeue` first — no running or pending
job referenced any of these paths.

| Path (under `$WORK/synthetic-RLVL/`) | Size | Why removed | Regeneration |
|---|---|---|---|
| `post_sft_dolci_20260804/qwen25_7b_dolmino_control_5b_dolci_100k_lr5em6/checkpoint-{750,780}` | 227 G | Orphaned optimizer state. Run completed 2026-08-06 and `final/` was intact; this job predates the post-run `rm -rf checkpoint-*` guard, so two 114 G mid-run checkpoints were never pruned. No scientific content. | Not worth regenerating; mid-run resume of a finished run is meaningless. |
| `checkpoint_state_offload/nanotron_dolmino_5b` | 171 G | Offloaded optimizer state from the 5 B dolmino midtrains (control/logic/nl_exact), last written 2026-08-01/03. Those midtrains completed and the transfer study they belong to is out of the paper (scope decision 2026-08-26). | Re-run the 5 B dolmino midtrains (~2 x 24 h x 8 A100 per arm). Not planned. |
| `post_sft_dolci_20260804/*/final` | 87 G (3 x 29 G, fp32) | The 5 B dolmino three-way Dolci SFT finals. Readout completed 2026-08-06 and again under pass@k 2026-08-07; the transfer study is excluded from the paper. Same rationale as the 13 finals deleted 2026-08-26. | `qwen25_dolmino_threeway_post_sft_*.slurm` from the retained 5 B midtrain checkpoints (~6 h x 4 A100 each). |
| `runs/` | 51 G | 325 SFT run directories from the superseded HFSA / paired-full / maze / iGSM ablation lines, newest file 2026-06-01. Predates BranchProof-unique-v2; none of it is cited. | Re-run the corresponding sweeps under `scripts/slurm/sweeps/`. Not planned. |

**Retained deliberately** (unchanged): `post_sft_dolci_20260804/*_base_checkpoint_audit.json`
(3 x 4 KB, the provenance record for the deleted finals), `lm_eval_results/`,
`nanosets/`, `nanotron_checkpoints/`, `data/`, `datasets/`, `nanotron_data/`,
and everything under `nanotron_longwin_midtrain/` and `nanosets_longwin_20260826/`
(the running arms).

Also cleaned the same day, outside `$WORK`:
- `$HOME/synthetic-RLVL/logs/` gzipped (3.4 G -> 490 M, lossless) and 937 pre-July
  logs from the superseded HFSA/paired lines deleted; `git gc` (394 M -> 240 M).
- `$HOME/.cache/uv` (14 G) and `$HOME/.codex` (5.6 G of sessions, db-backups and
  `*.codex-repair-*.bak`) removed. Keep the uv cache off `$HOME` in future:
  set `UV_CACHE_DIR` under `$WORK`.
- `analysis/hfsa_easy_validity_2026-05-14{,_smoke}` and
  `analysis/logic_cot_report_2026-05-25` (328 M) removed from the working tree;
  all three are git-tracked and clean in HEAD, so `git checkout -- <path>` restores them.

Net: `$WORK` 1.4 T -> 976 G, `$WORK/synthetic-RLVL` 865 G -> 418 G, `$HOME` 133 G -> 122 G.

**Still outstanding:** `$HPCVAULT/.venv_rlvl_posttrain` is 84,612 files = 46% of the
200 k vault inode soft quota (184 k used) for only 22 G. Move it to `$WORK` and
update `VENV_PATH` in `scripts/slurm/sweeps/posttrain_hard_*.slurm` rather than
deleting it, since RL post-training is deferred rather than abandoned.
