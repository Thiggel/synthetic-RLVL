# Document-preserving loader audit

- packed folder: `/home/vault/c107fa/c107fa12/synthetic-RLVL/nanosets/qwen25_real_logic_docpack_v1/nl`
- Dolmino folder (unchanged): `/home/vault/c107fa/c107fa12/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino`
- verdict: **ALL GATES PASS**

## zero_overlength: PASS

- overlength_count: 0
- doc_len_min: 61
- doc_len_mean: 473.9494372309725
- doc_len_max: 973
- window_len: 4097
- depth_range: [0, 5]

## decoded_batch: PASS

- windows_decoded: 32
- documents_seen: 287
- split_documents_found: 0
- examples_file: analysis/real_logic_docpack_audit_20260807/decoded_windows_real_nl.md
- dataset_windows: 55113
- windows_match_stats: True

## padding_loss_mask: PASS

- windows_checked: 64
- masked_labels: 13708
- pad_labels: 13708
- mask_equals_padding_everywhere: True
- failing_windows: []
- dolmino_mask_unchanged_all_ones: True

## exact_mixture: PASS

- target_loss_token_ratio: 0.05
- realized_loss_token_ratio: 0.05000147883675023
- abs_error: 1.4788367502238664e-06
- tolerance: 0.0002
- proof_weight: 0.05264683675123945
- normal_weight: 0.9473531632487605
- blend_size_samples: 610560
- synthetic_samples: 32145
- dolmino_samples: 578415
- synthetic_sample_ratio: 0.05264838836477988
- synthetic_loss_tokens: 124697979
- dolmino_loss_tokens: 2369187840
- synthetic_epochs_consumed: 0.5832562190408797
- note: loss tokens = label positions contributing loss (padding labels masked; Dolmino windows contribute all seq_len labels, matching the original runs)

