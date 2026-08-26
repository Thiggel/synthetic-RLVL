# Vault / storage inventory — large artifacts

Last updated: 2026-08-26 (created during the long-window depth-25 midtrain prep).

Purpose: when space is cleared, make it obvious at a glance what is cheap to
regenerate and what is not. Sort your deletions by the **Regeneration cost**
column, cheapest first, and never delete a row whose *Needed until* has not
passed without checking the dependents column.

Context for this file: the 2026-08-24..26 user-initiated cleanup removed
`$HPCVAULT/synthetic-RLVL/nanosets/`, the `qwen25_7b_tp1` base checkpoint, and
the `$WORK/nanotron` venv. All three were cheap to regenerate (≈3.5 GPU-hours
total) but they blocked the midtrain prep for half a day because nothing
recorded that they were prerequisites. Hence this table.

Cost legend: **GPU-h** = GPU-hours of a single A100/A40 (build jobs are
effectively single-GPU or CPU-only); wall-clock includes queue-free runtime
only — add 1–4 days of queue for anything needing a full 8-GPU A100 node.

## Prerequisites of the long-window (8192) depth-25 midtrain

| Artifact | Path | Size | Produced by | Regeneration cost | Needed until | Notes |
|---|---|---|---|---|---|---|
| Dolmino 5.1B Qwen2.5 nanoset | `$WORK/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino` | ~20.5G | job 3875824 → `nanotron_qwen25_build_dolmino_5p1b_2026-07-21.slurm` (re-run via `longwin_rebuild_prereqs_2026-08-26.slurm`) | ~2.5 GPU-h (2h33m observed) + ~50G transient raw JSONL | **all five long-window midtrains complete** (~2026-09) | Deterministic recipe (HF `allenai/dolma3_dolmino_mix-10B-1025`, shuffle seed 42) but the **HF revision is not pinned** — a re-pull could differ from the accepted 5,111,201,524-token build. Cheap to rebuild, but rebuilding mid-study breaks comparability with the 4096 rerun. Do not delete while any arm is unfinished. |
| Qwen2.5-7B → Nanotron tp1 base checkpoint | `$WORK/synthetic-RLVL/nanotron_checkpoints/qwen25_7b_tp1` | ~29G | `nanotron_qwen25_build_prereqs_2026-06-29.slurm` (conversion step; re-run via `longwin_rebuild_prereqs_2026-08-26.slurm`) | ~0.5 GPU-h | all midtrains complete | Pure function of the HF weights (cached under `$HPCVAULT/.cache*/hf`). Safe to delete between studies. |
| nanotron training venv | `$WORK/nanotron` | ~5.7G | `longwin_rebuild_nanotron_env_2026-08-26.slurm` (python 3.11.14 / torch 2.6.0+cu124 / flash-attn 2.7.4.post1) | ~0.5 GPU-h (flash-attn source build dominates) | all midtrains + post-SFT complete | Cheap in time, but **transitive versions are not pinned** — a rebuild is not bit-identical to the accepted env and requires re-running the tiny-nanoset smoke before release. Prefer keeping it over rebuilding mid-study. |
| Patched nanotron checkout | `/home/hpc/c107fa/c107fa12/nanotron` (HEAD 13625f34) | small | git; carries local patches (padding-label masking dd6aae2e, LR-resume fix, datatrove compat) | **not reproducible from upstream** | indefinitely | **NEVER DELETE.** The local patches are the study's correctness fixes and exist only here. |
| BranchProof-unique-v2 source dataset (seed 3407) | `$HPCVAULT/synthetic-RLVL/datasets/branchproof_unique_v2_20260710` | 689M | job → `build_materialized_branchproof_unique_v2_2026-07-10.slurm` | ~1 GPU-h | indefinitely | Source of every accepted synthetic result (bands 5–25). Small; keep. |
| Dolci prepared SFT subset | `$HPCVAULT/synthetic-RLVL/datasets/dolci_no_tools_single_turn_100k_seed3407_20260803` | 21G | job 3944069 | ~0.5 GPU-h, but pinned to HF revision `9156a5a5…` | all post-SFT runs complete | Deterministic + revision-pinned, so safe to regenerate. Every post-SFT arm reads it. |

## Artifacts this midtrain produces

| Artifact | Path | Size | Produced by | Regeneration cost | Needed until | Notes |
|---|---|---|---|---|---|---|
| Fresh band-25 corpus (72k rows, seed 20260830) | `$WORK/synthetic-RLVL/datasets/branchproof_unique_v2_longwin_20260826` | ~1G | `longwin_build_band25_2026-08-26.slurm` (materialize step) | ~5 min CPU | all five arms + any readout that needs to re-render traces | Fully deterministic from seed 20260830. Cheapest thing here — delete freely once packs exist, **but** keep the manifest so the seed is recoverable. |
| Rendered trace JSONLs + length stats (logic / nl_exact / condensed_logic) | `$WORK/synthetic-RLVL/nanotron_data/longwin_band25_20260826` | ~3G | same job (render step) | ~20 min CPU | until the packs are audited | Intermediate staging, same status as the raw JSONL trees the Dolmino builder deletes after its gate. Safe to delete after the audits pass; the `.stats.json` files are the length-matching evidence and should be **copied into `analysis/longwin_midtrain_prep_20260826/` before deleting**. |
| Arm 2 pack: long-doc control | `$WORK/synthetic-RLVL/nanosets_longwin_20260826/longdoc_control` | ~1.1G | `longwin_build_longdoc_2026-08-26.slurm` | ~15 min CPU, but **requires the Dolmino nanoset** (2.5 GPU-h if that is gone too) | arm 2 midtrain complete | Selection seed 20260830 + the band-25 histogram make it reproducible only if both the Dolmino nanoset and the logic stats JSON survive. |
| Arm 3 pack: logic_band25 | `$WORK/synthetic-RLVL/nanosets_longwin_20260826/logic_band25` | ~1.1G | `longwin_build_band25_2026-08-26.slurm` | ~10 min CPU | arm 3 midtrain complete | |
| Arm 4 pack: nl_exact_band25 | `$WORK/synthetic-RLVL/nanosets_longwin_20260826/nl_exact_band25` | ~1.1G | same | ~10 min CPU | arm 4 midtrain complete | |
| Arm 5 pack: condensed_logic_band25 | `$WORK/synthetic-RLVL/nanosets_longwin_20260826/condensed_logic_band25` | ~0.5G | same | ~10 min CPU | arm 5 midtrain complete | |
| Audit bundle (4 arms) | `analysis/longwin_midtrain_prep_20260826/` (in-repo, git) | <10M | `longwin_docpack_audit_2026-08-26.slurm` | ~30 min CPU | indefinitely | In git. Each midtrain is **fail-closed on its audit JSON**, so deleting the bundle blocks the runs. |
| Midtrain checkpoints (5 arms × 2385 steps) | `$WORK/synthetic-RLVL/nanotron_longwin_midtrain/<arm>/checkpoints` | ~199G per retained checkpoint; in-job pruner keeps **2 newest** ⇒ ≤~400G live per running arm; ~199G per finished arm | `nanotron_qwen25_longwin_midtrain_2026-08-26.slurm` | **~400 GPU-h per arm** (8 A100 × ~2 days) + 1–4 days queue | until the arm's post-SFT `final/` is verified, then the terminal checkpoint only | **The expensive rows.** ~2000 GPU-h for the five arms. Never delete a terminal checkpoint before its post-SFT succeeded, and only after the 645-nonempty-file / 625-model-file / four-equal-optimizer-shard gate passes on the retained successor. Model-only is ~29G if optimizer state is dropped after the study. |
| Post-SFT models (5 arms) | `$WORK/synthetic-RLVL/post_sft_dolci_longwin_20260826/<run>/final` | ~57G per arm (fp32 saves; ~285G total) | `qwen25_longwin_post_sft_2026-08-26.slurm` | ~24 GPU-h per arm (4 A100 × ~6h) | until the eval bundles are accepted | fp32 `final/` is ~2× a bf16 save; inherited from the accepted protocol. `checkpoint-*` dirs are deleted in-job. |

## Existing large artifacts (not produced by this study)

| Artifact | Path | Size | Regeneration cost | Needed until | Notes |
|---|---|---|---|---|---|
| Docpack rerun midtrain checkpoints (control/logic/nl_exact @2.5B) | `$HPCVAULT/synthetic-RLVL/nanotron_docpack_rerun` | 86G | ~1200 GPU-h | paper accepted | The accepted P0 answer. Effectively irreplaceable on the ICLR timeline. |
| mixdepth post-SFT models (10 runs) | `$HPCVAULT/synthetic-RLVL/post_sft_reasoning_mixture_20260821` | 568G | ~250 GPU-h (array 4075658, 10 × ~5h on 4 A100) | eval bundles accepted (they are — 2026-08-26) | **Largest deletable block on vault.** Greedy + pass@k readouts are complete and accepted, so these are now only needed for re-eval. First candidate if vault must be freed. |
| lm_eval result bundles | `$HPCVAULT/synthetic-RLVL/lm_eval_results` | varies | cheap to re-run only if the models above survive | paper accepted | Small; the actual scientific output. Keep. |

## Deletion order if space is needed (cheapest loss first)

1. Rendered trace JSONLs (`nanotron_data/longwin_band25_20260826`) — after copying the `.stats.json` files into the analysis bundle.
2. Fresh band-25 parquet corpus — regenerable in 5 min from seed 20260830.
3. `qwen25_7b_tp1` base checkpoint — 0.5 GPU-h, only needed at midtrain start.
4. mixdepth post-SFT models (568G) — readouts already accepted.
5. Dolci prepared subset — 0.5 GPU-h, revision-pinned.
6. Dolmino 5.1B nanoset — 2.5 GPU-h, but only between studies (HF revision unpinned).
7. Superseded midtrain checkpoints — only via the in-job pruner's discipline.
Never: the patched nanotron checkout, the audit bundles, terminal midtrain
checkpoints of an unfinished study, or anything under `analysis/`.
