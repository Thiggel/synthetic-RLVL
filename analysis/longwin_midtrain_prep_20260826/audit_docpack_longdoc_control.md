# Document-preserving loader audit

- packed folder: `/home/atuin/c107fa/c107fa12/synthetic-RLVL/nanosets_longwin_20260826/longdoc_control`
- Dolmino folder (unchanged): `/home/atuin/c107fa/c107fa12/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino`
- verdict: **ALL GATES PASS**

## zero_overlength: PASS

- overlength_count: 0
- doc_len_min: 251
- doc_len_mean: 3831.6408561904204
- doc_len_max: 7750
- window_len: 8193
- depth_range: [None, None]

## decoded_batch: PASS

- windows_decoded: 32
- documents_seen: 65
- split_documents_found: 0
- examples_file: analysis/longwin_midtrain_prep_20260826/decoded_windows_longdoc_control.md
- dataset_windows: 33138
- windows_match_stats: True

## padding_loss_mask: PASS

- windows_checked: 64
- masked_labels: 17091
- pad_labels: 17091
- mask_equals_padding_everywhere: True
- failing_windows: []
- dolmino_mask_unchanged_all_ones: True

## exact_mixture: PASS

- target_loss_token_ratio: 0.1
- realized_loss_token_ratio: 0.10000106392191897
- abs_error: 1.0639219189673144e-06
- tolerance: 0.0002
- proof_weight: 0.10324981410447501
- normal_weight: 0.896750185895525
- blend_size_samples: 305280
- synthetic_samples: 31521
- dolmino_samples: 273759
- synthetic_sample_ratio: 0.10325275157232705
- synthetic_loss_tokens: 249184471
- dolmino_loss_tokens: 2242633728
- synthetic_epochs_consumed: 0.9512040557667935
- note: loss tokens = label positions contributing loss (padding labels masked; Dolmino windows contribute all seq_len labels, matching the original runs)

