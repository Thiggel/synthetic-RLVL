# Document-preserving loader audit

- packed folder: `/home/vault/c107fa/c107fa12/synthetic-RLVL/nanosets/qwen25_dolmino_compact_docpack_v1/nl_exact`
- Dolmino folder (unchanged): `/home/vault/c107fa/c107fa12/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino`
- verdict: **ALL GATES PASS**

## zero_overlength: PASS

- overlength_count: 0
- doc_len_min: 179
- doc_len_mean: 1212.5477950084808
- doc_len_max: 2382
- window_len: 4097
- depth_range: [1, 14]

## decoded_batch: PASS

- windows_decoded: 32
- documents_seen: 110
- split_documents_found: 0
- examples_file: analysis/docpack_rerun_audit_20260807/decoded_windows_nl_exact.md
- dataset_windows: 9942
- windows_match_stats: True

## padding_loss_mask: PASS

- windows_checked: 64
- masked_labels: 4612
- pad_labels: 4612
- mask_equals_padding_everywhere: True
- failing_windows: []
- dolmino_mask_unchanged_all_ones: True

## exact_mixture: PASS

- target_loss_token_ratio: 0.05
- realized_loss_token_ratio: 0.049998698329067344
- abs_error: 1.301670932658816e-06
- tolerance: 0.0002
- proof_weight: 0.0508271927856276
- normal_weight: 0.9491728072143724
- blend_size_samples: 610560
- synthetic_samples: 31033
- dolmino_samples: 579527
- synthetic_sample_ratio: 0.05082710953878407
- synthetic_loss_tokens: 124930397
- dolmino_loss_tokens: 2373742592
- synthetic_epochs_consumed: 3.1214041440354054
- note: loss tokens = label positions contributing loss (padding labels masked; Dolmino windows contribute all seq_len labels, matching the original runs)

