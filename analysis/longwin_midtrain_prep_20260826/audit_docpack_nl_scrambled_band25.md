# Document-preserving loader audit

- packed folder: `/home/atuin/c107fa/c107fa12/synthetic-RLVL/nanosets_longwin_20260826/nl_scrambled_band25`
- Dolmino folder (unchanged): `/home/atuin/c107fa/c107fa12/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino`
- verdict: **ALL GATES PASS**

## zero_overlength: PASS

- overlength_count: 0
- doc_len_min: 344
- doc_len_mean: 3907.954513888889
- doc_len_max: 7852
- window_len: 8193
- depth_range: [1, 25]

## decoded_batch: PASS

- windows_decoded: 32
- documents_seen: 61
- split_documents_found: 0
- examples_file: analysis/longwin_midtrain_prep_20260826/decoded_windows_nl_scrambled_band25.md
- dataset_windows: 35979
- windows_match_stats: True

## padding_loss_mask: PASS

- windows_checked: 64
- masked_labels: 25318
- pad_labels: 25318
- mask_equals_padding_everywhere: True
- failing_windows: []
- dolmino_mask_unchanged_all_ones: True

## exact_mixture: PASS

- target_loss_token_ratio: 0.1
- realized_loss_token_ratio: 0.10000181520113068
- abs_error: 1.8152011306699078e-06
- tolerance: 0.0002
- proof_weight: 0.10427183003088988
- normal_weight: 0.8957281699691101
- blend_size_samples: 305280
- synthetic_samples: 31833
- dolmino_samples: 273447
- synthetic_sample_ratio: 0.1042747641509434
- synthetic_loss_tokens: 248902556
- dolmino_loss_tokens: 2240077824
- synthetic_epochs_consumed: 0.884766113566247
- note: loss tokens = label positions contributing loss (padding labels masked; Dolmino windows contribute all seq_len labels, matching the original runs)

