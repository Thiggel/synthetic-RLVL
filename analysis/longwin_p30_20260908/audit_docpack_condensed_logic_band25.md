# Document-preserving loader audit

- packed folder: `/home/atuin/c107fa/c107fa12/synthetic-RLVL/nanosets_longwin_p30_20260908/condensed_logic_band25`
- Dolmino folder (unchanged): `/home/atuin/c107fa/c107fa12/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino`
- verdict: **ALL GATES PASS**

## zero_overlength: PASS

- overlength_count: 0
- doc_len_min: 195
- doc_len_mean: 1501.3807557692307
- doc_len_max: 3017
- window_len: 8193
- depth_range: [1, 25]

## decoded_batch: PASS

- windows_decoded: 32
- documents_seen: 177
- split_documents_found: 0
- examples_file: analysis/longwin_p30_20260908/decoded_windows_condensed_logic_band25.md
- dataset_windows: 96124
- windows_match_stats: True

## padding_loss_mask: PASS

- windows_checked: 64
- masked_labels: 4778
- pad_labels: 4778
- mask_equals_padding_everywhere: True
- failing_windows: []
- dolmino_mask_unchanged_all_ones: True

## exact_mixture: PASS

- target_loss_token_ratio: 0.3
- realized_loss_token_ratio: 0.3000000390488439
- abs_error: 3.9048843936129174e-08
- tolerance: 0.0002
- proof_weight: 0.3018288188837097
- normal_weight: 0.6981711811162903
- blend_size_samples: 305280
- synthetic_samples: 92143
- dolmino_samples: 213137
- synthetic_sample_ratio: 0.301831105870021
- synthetic_loss_tokens: 748293698
- dolmino_loss_tokens: 1746018304
- synthetic_epochs_consumed: 0.9585847447047564
- note: loss tokens = label positions contributing loss (padding labels masked; Dolmino windows contribute all seq_len labels, matching the original runs)

