# Document-preserving loader audit

- packed folder: `/home/atuin/c107fa/c107fa12/synthetic-RLVL/nanosets_longwin_p30_20260908/nl_exact_band25`
- Dolmino folder (unchanged): `/home/atuin/c107fa/c107fa12/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino`
- verdict: **ALL GATES PASS**

## zero_overlength: PASS

- overlength_count: 0
- doc_len_min: 344
- doc_len_mean: 3907.6716038461536
- doc_len_max: 7843
- window_len: 8193
- depth_range: [1, 25]

## decoded_batch: PASS

- windows_decoded: 32
- documents_seen: 62
- split_documents_found: 0
- examples_file: analysis/longwin_p30_20260908/decoded_windows_nl_exact_band25.md
- dataset_windows: 259766
- windows_match_stats: True

## padding_loss_mask: PASS

- windows_checked: 64
- masked_labels: 21753
- pad_labels: 21753
- mask_equals_padding_everywhere: True
- failing_windows: []
- dolmino_mask_unchanged_all_ones: True

## exact_mixture: PASS

- target_loss_token_ratio: 0.3
- realized_loss_token_ratio: 0.30000123982519883
- abs_error: 1.2398251988399167e-06
- tolerance: 0.0002
- proof_weight: 0.3098018533865743
- normal_weight: 0.6901981466134257
- blend_size_samples: 305280
- synthetic_samples: 94577
- dolmino_samples: 210703
- synthetic_sample_ratio: 0.3098041142557652
- synthetic_loss_tokens: 739752500
- dolmino_loss_tokens: 1726078976
- synthetic_epochs_consumed: 0.36408536913991824
- note: loss tokens = label positions contributing loss (padding labels masked; Dolmino windows contribute all seq_len labels, matching the original runs)

