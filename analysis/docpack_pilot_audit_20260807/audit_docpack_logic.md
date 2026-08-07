# Document-preserving loader audit

- packed folder: `/home/vault/c107fa/c107fa12/synthetic-RLVL/nanosets/qwen25_dolmino_compact_docpack_v1/logic`
- Dolmino folder (unchanged): `/home/vault/c107fa/c107fa12/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino`
- verdict: **ALL GATES PASS**

## zero_overlength: PASS

- overlength_count: 0
- doc_len_min: 362
- doc_len_mean: 2117.368763557484
- doc_len_max: 4082
- window_len: 4097
- depth_range: [1, 14]

## decoded_batch: PASS

- windows_decoded: 32
- documents_seen: 59
- split_documents_found: 0
- examples_file: analysis/docpack_pilot_audit_20260807/decoded_windows_logic.md
- dataset_windows: 10290
- windows_match_stats: True

## padding_loss_mask: PASS

- windows_checked: 64
- masked_labels: 12937
- pad_labels: 12937
- mask_equals_padding_everywhere: True
- failing_windows: []
- dolmino_mask_unchanged_all_ones: True

## exact_mixture: PASS

- target_loss_token_ratio: 0.05
- realized_loss_token_ratio: 0.049997892264386584
- abs_error: 2.107735613418593e-06
- tolerance: 0.0002
- proof_weight: 0.05250141264914705
- normal_weight: 0.9474985873508529
- blend_size_samples: 122112
- synthetic_samples: 6411
- dolmino_samples: 115701
- synthetic_sample_ratio: 0.05250098270440252
- synthetic_loss_tokens: 24941593
- dolmino_loss_tokens: 473911296
- synthetic_epochs_consumed: 0.6230320699708455
- note: loss tokens = label positions contributing loss (padding labels masked; Dolmino windows contribute all seq_len labels, matching the original runs)

