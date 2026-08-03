# Storage Audit and Cleanup

Date: 2026-08-03 14:10 CEST

Progress update at 14:55 CEST: job `3946030` is about 72% complete by logical
bytes. Control is fully transferred and symlinked; logic has one complete shard
and most of the second. Vault is already `754 GiB/1000 GiB`. Atuin reports
906 TB free, 1% inode use, and no visible per-user quota; projected final
`$WORK/synthetic-RLVL` usage is about 238.7 GiB.

## Vault ownership

Sizes use filesystem-allocated bytes. File counts count regular files.

| Vault subtree | Bytes | GiB | Files | Ownership/status |
| --- | ---: | ---: | ---: | --- |
| `synthetic-RLVL` | 867,830,053,376 | 808.3 | 9,475 | This project; dominant byte user |
| `sequence-editing` | 71,839,092,224 | 66.9 | 53,998 | Other project; untouched |
| `.venv_rlvl_posttrain` | 22,820,245,504 | 21.3 | 76,410 | Shared active environment; untouched |
| `.cache` | 17,989,185,024 | 16.8 | 13,763 | Shared cache; untouched because live symlink/reference ownership is mixed |
| `hf_model_archives` | 9,953,673,216 | 9.3 | 127 | Other-project archive; untouched |
| `babylm` | 1,659,117,056 | 1.5 | 2,807 | Other project; untouched |

The byte problem is this project: after immediate cleanup it still accounts
for about 87.5% of measured Vault bytes. The file-count pressure is different:
the active shared venv and `sequence-editing` account for most regular files,
while this project has fewer than 10,000.

## Immediate deletion

The following paths had completed or invalidated lifecycle gates and no live
Slurm reference. Deletion reclaimed exactly 25,985,174,016 bytes (24.2 GiB):

| Removed artifact | Bytes | Gate |
| --- | ---: | --- |
| `nanosets/qwen25_branchproof_unique_v2` | 19,215,099,392 | Old p15 experiment accepted; broader old-format grid rejected; future compact/document-preserving pilot requires a rebuild |
| `quarantine/pre_declaration_fix_20260715` | 284,264,448 | Explicitly invalid pre-fix outputs |
| conditioned-32B seed-3409 `checkpoint-9750` | 3,242,888,704 | Training/evaluation accepted; final adapter retained |
| conditioned-32B seed-3409 `checkpoint-10000` | 3,242,921,472 | Adapter SHA-256 exactly matches retained `final/`; optimizer restart is no longer needed |

User Vault usage moved from `949 GiB` to `924 GiB` against the `1000 GiB`
soft quota. This fixes the immediate quota condition.

## Reversible optimizer offload

Control and formal step-9537 optimizer states are each 182,791,570,944 bytes.
They are not safe to delete because the staged 10B/15B/20B design may still
resume from 5B after the complete three-way gate. Job `3946030` is therefore
moving both optimizer directories to
`$WORK/synthetic-RLVL/checkpoint_state_offload/nanotron_dolmino_5b`, verifying
file names/sizes and the full checkpoint gate, and replacing each original
directory with a symlink. Replacement post-SFT smoke `3945777_0` depends on
successful offload.

When complete, this preserves exact resume capability while reclaiming another
365,583,141,888 bytes (340.4 GiB) from Vault. `$WORK/synthetic-RLVL` was only
73,510,830,080 bytes (68.5 GiB) before this move; the filesystem reports ample
physical capacity and no Atuin user quota in `quota -s`.

## Protected large data

- Keep all three current Dolmino model shards and the NL step-5000 optimizer:
  NL continuation is live, and all three weight states feed matched readouts.
- Keep `nanosets/qwen25_dolmino_neutral_v1_5p1b` and
  `nanosets/qwen25_dolmino_neutral_v1`: the pending NL continuation references
  their Dolmino and NL folders directly.
- Keep `nanotron_checkpoints/qwen25_7b_tp1`: the corrected compact-objective
  pilot may require the common pretrained initialization.
- Keep accepted raw evaluation bundles and final SFT adapters until report
  tables and qualitative evidence are frozen.
