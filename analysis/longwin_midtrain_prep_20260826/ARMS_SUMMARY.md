# Long-window arms: packing + length-matching summary

## Per-arm packed corpus (10% replacement slice)

| arm | docs | windows | real tokens | pad | eff | doc p50 | doc p99 | doc max | overlength excluded | split docs |
|---|---|---|---|---|---|---|---|---|---|---|
| longdoc_control | 68,396 | 33,138 | 262,000,512 | 9,499,122 | 0.9650 | 3712.0 | 7684.050000000003 | 7750 | 0 | 0 |
| logic_band25 | 72,000 | 35,492 | 277,732,923 | 13,053,033 | 0.9551 | 3750.0 | 7674.0 | 7728 | 0 | 0 |
| nl_exact_band25 | 72,000 | 35,979 | 281,372,725 | 13,403,222 | 0.9545 | 3820.0 | 7745.0 | 7852 | 0 | 0 |
| condensed_logic_band25 | 72,000 | 13,311 | 108,099,330 | 957,693 | 0.9912 | 1451.0 | 3008.0 | 3015 | 0 | 0 |

## Audit gates + realized mixture (2.5B blend, seq 8192)

| arm | all_pass | zero_overlen | decoded | pad_mask | mixture | proof_weight | realized ratio | epochs | blend tokens |
|---|---|---|---|---|---|---|---|---|---|
| longdoc_control | True | True | True | True | True | 0.10324981 | 0.100001 | 0.951 | 2,500,853,760 |
| logic_band25 | True | True | True | True | True | 0.10420632 | 0.100001 | 0.896 | 2,500,853,760 |
| nl_exact_band25 | True | True | True | True | True | 0.10427183 | 0.100002 | 0.885 | 2,500,853,760 |
| condensed_logic_band25 | True | True | True | True | True | 0.10079632 | 0.100000 | 2.312 | 2,500,853,760 |

## Rendered-document token lengths (Qwen2.5-7B tokenizer)

| corpus | n | p50 | p95 | p99 | max | mean | total tokens | >4096 | >8192 |
|---|---|---|---|---|---|---|---|---|---|
| logic | 72,000 | 3749.0 | 7357.0 | 7673.0 | 7727 | 3856 | 277,660,923 | 0.442 | 0.0000 |
| nl_exact | 72,000 | 3819.0 | 7417.0 | 7744.0 | 7851 | 3907 | 281,300,725 | 0.477 | 0.0000 |
| condensed_logic | 72,000 | 1450.0 | 2886.0 | 3007.0 | 3014 | 1500 | 108,027,330 | 0.000 | 0.0000 |
| longdoc_control | 68,396 | 3712.0 | 7113.0 | 7684.050000000003 | 7750 | 3832 | 262,000,512 | - | 0.0000 |

### Long-doc control histogram match (250-token bins)

| bin | band-25 logic (target, normalized) | longdoc achieved (normalized) |
|---|---|---|
| 250-500 | 0.0400 | 0.0415 |
| 500-750 | 0.0400 | 0.0388 |
| 750-1000 | 0.0400 | 0.0408 |
| 1000-1250 | 0.0400 | 0.0392 |
| 1250-1500 | 0.0400 | 0.0406 |
| 1500-1750 | 0.0388 | 0.0391 |
| 1750-2000 | 0.0317 | 0.0311 |
| 2000-2250 | 0.0256 | 0.0267 |
| 2250-2500 | 0.0280 | 0.0280 |
| 2500-2750 | 0.0359 | 0.0364 |
| 2750-3000 | 0.0400 | 0.0404 |
| 3000-3250 | 0.0400 | 0.0394 |
| 3250-3500 | 0.0400 | 0.0406 |
| 3500-3750 | 0.0203 | 0.0201 |
| 3750-4000 | 0.0197 | 0.0189 |
| 4000-4250 | 0.0400 | 0.0402 |
| 4250-4500 | 0.0400 | 0.0388 |
| 4500-4750 | 0.0399 | 0.0411 |
| 4750-5000 | 0.0047 | 0.0045 |
| 5000-5250 | 0.0354 | 0.0340 |
| 5250-5500 | 0.0400 | 0.0409 |
| 5500-5750 | 0.0400 | 0.0389 |
| 5750-6000 | 0.0381 | 0.0389 |
| 6000-6250 | 0.0020 | 0.0020 |
| 6250-6500 | 0.0399 | 0.0401 |
| 6500-6750 | 0.0400 | 0.0410 |
| 6750-7000 | 0.0002 | 0.0004 |
| 7000-7250 | 0.0398 | 0.0396 |
| 7250-7500 | 0.0400 | 0.0387 |
| 7500-7750 | 0.0400 | 0.0392 |
| 7750-8000 | 0.0000 | 0.0002 |

Note: the document-preserving packer never splits a document across windows by construction (docs longer than one window are EXCLUDED and counted as `overlength`), so the split-document fraction is 0 by design; the decoded-batch audit gate independently verifies this on real loader windows.
