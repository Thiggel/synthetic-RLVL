# Document-preserving loader audit

- packed folder: `/home/vault/c107fa/c107fa12/synthetic-RLVL/nanosets/qwen25_real_logic_docpack_v1/formal`
- Dolmino folder (unchanged): `/home/vault/c107fa/c107fa12/synthetic-RLVL/nanosets/qwen25_dolmino_neutral_v1_5p1b/dolmino`
- verdict: **ALL GATES PASS**

## zero_overlength: PASS

- overlength_count: 0
- doc_len_min: 91
- doc_len_mean: 663.4777618682424
- doc_len_max: 1355
- window_len: 4097
- depth_range: [0, 5]

## decoded_batch: PASS

- windows_decoded: 32
- documents_seen: 222
- split_documents_found: 0
- examples_file: analysis/real_logic_docpack_audit_2p5b_20260818/decoded_windows_real_logic.md
- dataset_windows: 78497
- windows_match_stats: True

## padding_loss_mask: PASS

- windows_checked: 64
- masked_labels: 18551
- pad_labels: 18551
- mask_equals_padding_everywhere: True
- failing_windows: []
- dolmino_mask_unchanged_all_ones: True

## exact_mixture: PASS

- target_loss_token_ratio: 0.05
- realized_loss_token_ratio: 0.049998662081001725
- abs_error: 1.3379189982773432e-06
- tolerance: 0.0002
- proof_weight: 0.053516531571887335
- normal_weight: 0.9464834684281127
- blend_size_samples: 610560
- synthetic_samples: 32675
- dolmino_samples: 577885
- synthetic_sample_ratio: 0.05351644392033543
- synthetic_loss_tokens: 124576331
- dolmino_loss_tokens: 2367016960
- synthetic_epochs_consumed: 0.41625794616354767
- note: loss tokens = label positions contributing loss (padding labels masked; Dolmino windows contribute all seq_len labels, matching the original runs)

