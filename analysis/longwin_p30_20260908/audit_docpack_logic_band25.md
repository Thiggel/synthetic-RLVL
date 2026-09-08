# Document-preserving loader audit

- packed folder: `/home/atuin/c107fa/c107fa12/synthetic-RLVL/nanosets_longwin_p30_20260908/logic_band25`
- Dolmino folder (unchanged): `/home/atuin/c107fa/c107fa12/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino`
- verdict: **ALL GATES PASS**

## zero_overlength: PASS

- overlength_count: 0
- doc_len_min: 409
- doc_len_mean: 3857.262428846154
- doc_len_max: 7719
- window_len: 8193
- depth_range: [1, 25]

## decoded_batch: PASS

- windows_decoded: 32
- documents_seen: 62
- split_documents_found: 0
- examples_file: analysis/longwin_p30_20260908/decoded_windows_logic_band25.md
- dataset_windows: 256247
- windows_match_stats: True

## padding_loss_mask: PASS

- windows_checked: 64
- masked_labels: 24023
- pad_labels: 24023
- mask_equals_padding_everywhere: True
- failing_windows: []
- dolmino_mask_unchanged_all_ones: True

## exact_mixture: PASS

- target_loss_token_ratio: 0.3
- realized_loss_token_ratio: 0.29999878695787086
- abs_error: 1.2130421291334237e-06
- tolerance: 0.0002
- proof_weight: 0.30965444724334135
- normal_weight: 0.6903455527566587
- blend_size_samples: 305280
- synthetic_samples: 94531
- dolmino_samples: 210749
- synthetic_sample_ratio: 0.3096534329140461
- synthetic_loss_tokens: 739905358
- dolmino_loss_tokens: 1726455808
- synthetic_epochs_consumed: 0.3689057823115978
- note: loss tokens = label positions contributing loss (padding labels masked; Dolmino windows contribute all seq_len labels, matching the original runs)

