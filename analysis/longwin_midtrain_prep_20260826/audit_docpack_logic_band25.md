# Document-preserving loader audit

- packed folder: `/home/atuin/c107fa/c107fa12/synthetic-RLVL/nanosets_longwin_20260826/logic_band25`
- Dolmino folder (unchanged): `/home/atuin/c107fa/c107fa12/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino`
- verdict: **ALL GATES PASS**

## zero_overlength: PASS

- overlength_count: 0
- doc_len_min: 409
- doc_len_mean: 3857.4017083333333
- doc_len_max: 7728
- window_len: 8193
- depth_range: [1, 25]

## decoded_batch: PASS

- windows_decoded: 32
- documents_seen: 66
- split_documents_found: 0
- examples_file: analysis/longwin_midtrain_prep_20260826/decoded_windows_logic_band25.md
- dataset_windows: 35492
- windows_match_stats: True

## padding_loss_mask: PASS

- windows_checked: 64
- masked_labels: 21717
- pad_labels: 21717
- mask_equals_padding_everywhere: True
- failing_windows: []
- dolmino_mask_unchanged_all_ones: True

## exact_mixture: PASS

- target_loss_token_ratio: 0.1
- realized_loss_token_ratio: 0.10000120024391
- abs_error: 1.2002439099961792e-06
- tolerance: 0.0002
- proof_weight: 0.104206316189453
- normal_weight: 0.895793683810547
- blend_size_samples: 305280
- synthetic_samples: 31813
- dolmino_samples: 273467
- synthetic_sample_ratio: 0.10420925052410901
- synthetic_loss_tokens: 248919060
- dolmino_loss_tokens: 2240241664
- synthetic_epochs_consumed: 0.8963428378226079
- note: loss tokens = label positions contributing loss (padding labels masked; Dolmino windows contribute all seq_len labels, matching the original runs)

