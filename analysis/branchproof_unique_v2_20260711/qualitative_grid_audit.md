# Corrected BranchProof qualitative audit

Structural acceptance: **True**.

Each run contributes shallow, train-edge, first-OOD, and depth-50 selections. Labels distinguish correct+valid, correct+invalid, and incorrect retained generations when those cases exist. For each generation chunk whose observed maximum reached the configured cap, the audit references the longest of the retained generations; that sample is a cap-hit diagnostic, not proof that it was the exact generation that reached the cap.

| modality | train max | seed | correct | incorrect | valid | invalid | cap chunks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `logic` | 5 | 3407 | 303 | 593 | 233 | 663 | 88 |
| `logic` | 5 | 3408 | 218 | 678 | 207 | 689 | 88 |
| `logic` | 5 | 3409 | 244 | 652 | 202 | 694 | 88 |
| `logic` | 10 | 3407 | 439 | 457 | 410 | 486 | 68 |
| `logic` | 10 | 3408 | 446 | 450 | 406 | 490 | 67 |
| `logic` | 10 | 3409 | 467 | 429 | 409 | 487 | 65 |
| `logic` | 15 | 3407 | 500 | 396 | 449 | 447 | 58 |
| `logic` | 15 | 3408 | 531 | 365 | 463 | 433 | 51 |
| `logic` | 15 | 3409 | 533 | 363 | 456 | 440 | 58 |
| `logic` | 20 | 3407 | 567 | 329 | 514 | 382 | 48 |
| `logic` | 20 | 3408 | 560 | 336 | 543 | 353 | 48 |
| `logic` | 20 | 3409 | 668 | 228 | 527 | 369 | 45 |
| `logic` | 25 | 3407 | 815 | 81 | 804 | 92 | 28 |
| `logic` | 25 | 3408 | 843 | 53 | 832 | 64 | 22 |
| `logic` | 25 | 3409 | 778 | 118 | 760 | 136 | 38 |
| `nl_exact` | 5 | 3407 | 611 | 285 | 603 | 293 | 34 |
| `nl_exact` | 5 | 3408 | 576 | 320 | 566 | 330 | 6 |
| `nl_exact` | 5 | 3409 | 498 | 398 | 545 | 351 | 13 |
| `nl_exact` | 10 | 3407 | 668 | 228 | 639 | 257 | 20 |
| `nl_exact` | 10 | 3408 | 554 | 342 | 522 | 374 | 8 |
| `nl_exact` | 10 | 3409 | 665 | 231 | 675 | 221 | 36 |
| `nl_exact` | 15 | 3407 | 636 | 260 | 637 | 259 | 4 |
| `nl_exact` | 15 | 3408 | 643 | 253 | 642 | 254 | 29 |
| `nl_exact` | 15 | 3409 | 630 | 266 | 632 | 264 | 7 |
| `nl_exact` | 20 | 3407 | 659 | 237 | 660 | 236 | 29 |
| `nl_exact` | 20 | 3408 | 611 | 285 | 611 | 285 | 0 |
| `nl_exact` | 20 | 3409 | 628 | 268 | 630 | 266 | 0 |
| `nl_exact` | 25 | 3407 | 641 | 255 | 640 | 256 | 1 |
| `nl_exact` | 25 | 3408 | 617 | 279 | 625 | 271 | 0 |
| `nl_exact` | 25 | 3409 | 640 | 256 | 649 | 247 | 24 |

## Selected generations

### logic train 1..5 seed 3407
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=515
- `train_edge/correct_valid`: depth 5, line 131, correct=1.0, format=1.0, chars=1264
- `ood_edge/correct_valid`: depth 10, line 342, correct=1.0, format=1.0, chars=2280
- `ood_edge/correct_invalid`: depth 10, line 244, correct=1.0, format=0.0, chars=1766
- `ood_edge/incorrect`: depth 10, line 132, correct=0.0, format=1.0, chars=1157
- `depth50/correct_invalid`: depth 50, line 338, correct=1.0, format=1.0, chars=3671
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=5794
- `cap_hit_chunk_25_longest_retained`: depth 35, line 321, correct=0.0, format=0.0, chars=20063
- `cap_hit_chunk_26_longest_retained`: depth 40, line 336, correct=0.0, format=0.0, chars=8594
- `cap_hit_chunk_27_longest_retained`: depth 50, line 338, correct=1.0, format=1.0, chars=3671

### logic train 1..5 seed 3408
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=515
- `train_edge/correct_valid`: depth 5, line 131, correct=1.0, format=1.0, chars=1264
- `ood_edge/correct_valid`: depth 10, line 188, correct=1.0, format=1.0, chars=2209
- `ood_edge/correct_invalid`: depth 10, line 132, correct=1.0, format=1.0, chars=2367
- `ood_edge/incorrect`: depth 10, line 146, correct=0.0, format=0.0, chars=11896
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=11948
- `cap_hit_chunk_25_longest_retained`: depth 40, line 322, correct=0.0, format=0.0, chars=12793
- `cap_hit_chunk_26_longest_retained`: depth 35, line 335, correct=0.0, format=0.0, chars=13200
- `cap_hit_chunk_27_longest_retained`: depth 45, line 337, correct=0.0, format=0.0, chars=11946

### logic train 1..5 seed 3409
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=515
- `train_edge/correct_valid`: depth 5, line 131, correct=1.0, format=1.0, chars=1264
- `ood_edge/correct_valid`: depth 10, line 160, correct=1.0, format=1.0, chars=2278
- `ood_edge/correct_invalid`: depth 10, line 146, correct=1.0, format=1.0, chars=4217
- `ood_edge/incorrect`: depth 10, line 132, correct=0.0, format=0.0, chars=15790
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=21533
- `cap_hit_chunk_25_longest_retained`: depth 50, line 324, correct=0.0, format=0.0, chars=22864
- `cap_hit_chunk_26_longest_retained`: depth 12, line 329, correct=0.0, format=0.0, chars=21242
- `cap_hit_chunk_27_longest_retained`: depth 50, line 338, correct=0.0, format=0.0, chars=20385

### logic train 1..10 seed 3407
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=515
- `train_edge/correct_valid`: depth 10, line 132, correct=1.0, format=1.0, chars=2277
- `ood_edge/correct_valid`: depth 12, line 133, correct=1.0, format=1.0, chars=2726
- `ood_edge/correct_invalid`: depth 12, line 315, correct=1.0, format=0.0, chars=2482
- `ood_edge/incorrect`: depth 12, line 217, correct=0.0, format=1.0, chars=2720
- `depth50/correct_invalid`: depth 50, line 408, correct=1.0, format=1.0, chars=2769
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=15921
- `cap_hit_chunk_35_longest_retained`: depth 45, line 407, correct=1.0, format=1.0, chars=6761
- `cap_hit_chunk_39_longest_retained`: depth 45, line 435, correct=0.0, format=0.0, chars=11945
- `cap_hit_chunk_41_longest_retained`: depth 45, line 449, correct=0.0, format=0.0, chars=14218

### logic train 1..10 seed 3408
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=515
- `train_edge/correct_valid`: depth 10, line 132, correct=1.0, format=1.0, chars=2277
- `ood_edge/correct_valid`: depth 12, line 133, correct=1.0, format=1.0, chars=2726
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=22849
- `cap_hit_chunk_34_longest_retained`: depth 50, line 394, correct=0.0, format=0.0, chars=12118
- `cap_hit_chunk_42_longest_retained`: depth 40, line 462, correct=0.0, format=0.0, chars=14439
- `cap_hit_chunk_46_longest_retained`: depth 45, line 491, correct=0.0, format=0.0, chars=12113

### logic train 1..10 seed 3409
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=515
- `train_edge/correct_valid`: depth 10, line 132, correct=1.0, format=1.0, chars=2277
- `ood_edge/correct_valid`: depth 12, line 133, correct=1.0, format=1.0, chars=2726
- `ood_edge/correct_invalid`: depth 12, line 525, correct=1.0, format=1.0, chars=2724
- `depth50/correct_invalid`: depth 50, line 408, correct=1.0, format=0.0, chars=8038
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=1.0, chars=5956
- `cap_hit_chunk_42_longest_retained`: depth 50, line 464, correct=0.0, format=0.0, chars=11955
- `cap_hit_chunk_49_longest_retained`: depth 50, line 520, correct=0.0, format=0.0, chars=12009
- `cap_hit_chunk_50_longest_retained`: depth 18, line 527, correct=0.0, format=0.0, chars=9723

### logic train 1..15 seed 3407
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=515
- `train_edge/correct_valid`: depth 15, line 134, correct=1.0, format=1.0, chars=3395
- `ood_edge/correct_valid`: depth 18, line 135, correct=1.0, format=1.0, chars=4068
- `ood_edge/correct_invalid`: depth 18, line 1003, correct=1.0, format=1.0, chars=2572
- `ood_edge/incorrect`: depth 18, line 765, correct=0.0, format=1.0, chars=4061
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=2719
- `cap_hit_chunk_53_longest_retained`: depth 50, line 548, correct=0.0, format=0.0, chars=13220
- `cap_hit_chunk_54_longest_retained`: depth 35, line 559, correct=0.0, format=0.0, chars=14627
- `cap_hit_chunk_57_longest_retained`: depth 20, line 584, correct=1.0, format=0.0, chars=4513

### logic train 1..15 seed 3408
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=515
- `train_edge/correct_valid`: depth 15, line 134, correct=1.0, format=1.0, chars=3395
- `ood_edge/correct_valid`: depth 18, line 135, correct=1.0, format=1.0, chars=4068
- `ood_edge/correct_invalid`: depth 18, line 513, correct=1.0, format=0.0, chars=4158
- `depth50/correct_invalid`: depth 50, line 436, correct=1.0, format=0.0, chars=10607
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=13213
- `cap_hit_chunk_57_longest_retained`: depth 20, line 584, correct=0.0, format=0.0, chars=4550
- `cap_hit_chunk_58_longest_retained`: depth 50, line 590, correct=0.0, format=0.0, chars=15226
- `cap_hit_chunk_59_longest_retained`: depth 30, line 600, correct=0.0, format=0.0, chars=6181

### logic train 1..15 seed 3409
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=515
- `train_edge/correct_valid`: depth 15, line 134, correct=1.0, format=1.0, chars=3395
- `ood_edge/correct_valid`: depth 18, line 135, correct=1.0, format=1.0, chars=4068
- `ood_edge/correct_invalid`: depth 18, line 667, correct=1.0, format=0.0, chars=3989
- `ood_edge/incorrect`: depth 18, line 723, correct=0.0, format=1.0, chars=4020
- `depth50/correct_invalid`: depth 50, line 156, correct=1.0, format=0.0, chars=10791
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=13120
- `cap_hit_chunk_48_longest_retained`: depth 50, line 506, correct=0.0, format=0.0, chars=13232
- `cap_hit_chunk_52_longest_retained`: depth 30, line 544, correct=0.0, format=0.0, chars=13222
- `cap_hit_chunk_57_longest_retained`: depth 20, line 584, correct=0.0, format=0.0, chars=4474

### logic train 1..20 seed 3407
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=515
- `train_edge/correct_valid`: depth 20, line 136, correct=1.0, format=1.0, chars=4513
- `ood_edge/correct_valid`: depth 25, line 417, correct=1.0, format=1.0, chars=5565
- `ood_edge/correct_invalid`: depth 25, line 137, correct=1.0, format=0.0, chars=5595
- `ood_edge/incorrect`: depth 25, line 179, correct=0.0, format=0.0, chars=14472
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=12496
- `cap_hit_chunk_65_longest_retained`: depth 30, line 642, correct=1.0, format=0.0, chars=6582
- `cap_hit_chunk_66_longest_retained`: depth 25, line 655, correct=0.0, format=0.0, chars=5432
- `cap_hit_chunk_67_longest_retained`: depth 50, line 660, correct=0.0, format=0.0, chars=13416

### logic train 1..20 seed 3408
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=515
- `train_edge/correct_valid`: depth 20, line 136, correct=1.0, format=1.0, chars=4513
- `ood_edge/correct_valid`: depth 25, line 137, correct=1.0, format=1.0, chars=5552
- `ood_edge/correct_invalid`: depth 25, line 179, correct=1.0, format=1.0, chars=5446
- `ood_edge/incorrect`: depth 25, line 165, correct=0.0, format=1.0, chars=5530
- `depth50/correct_invalid`: depth 50, line 548, correct=1.0, format=0.0, chars=10360
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=13175
- `cap_hit_chunk_65_longest_retained`: depth 40, line 644, correct=0.0, format=0.0, chars=8304
- `cap_hit_chunk_66_longest_retained`: depth 30, line 656, correct=0.0, format=0.0, chars=12520
- `cap_hit_chunk_67_longest_retained`: depth 40, line 658, correct=0.0, format=0.0, chars=11980

### logic train 1..20 seed 3409
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=515
- `train_edge/correct_valid`: depth 20, line 136, correct=1.0, format=1.0, chars=4513
- `ood_edge/correct_valid`: depth 25, line 165, correct=1.0, format=1.0, chars=5602
- `ood_edge/correct_invalid`: depth 25, line 179, correct=1.0, format=0.0, chars=5598
- `ood_edge/incorrect`: depth 25, line 137, correct=0.0, format=1.0, chars=3507
- `depth50/correct_valid`: depth 50, line 968, correct=1.0, format=1.0, chars=10843
- `depth50/correct_invalid`: depth 50, line 562, correct=1.0, format=0.0, chars=10794
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=13261
- `cap_hit_chunk_65_longest_retained`: depth 45, line 645, correct=0.0, format=0.0, chars=13231
- `cap_hit_chunk_66_longest_retained`: depth 30, line 656, correct=1.0, format=1.0, chars=6225
- `cap_hit_chunk_68_longest_retained`: depth 35, line 671, correct=0.0, format=0.0, chars=7594

### logic train 1..25 seed 3407
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=515
- `train_edge/correct_valid`: depth 25, line 137, correct=1.0, format=1.0, chars=5715
- `ood_edge/correct_valid`: depth 30, line 138, correct=1.0, format=1.0, chars=6863
- `ood_edge/correct_invalid`: depth 30, line 152, correct=1.0, format=1.0, chars=6864
- `ood_edge/incorrect`: depth 30, line 334, correct=0.0, format=1.0, chars=6948
- `depth50/correct_valid`: depth 50, line 184, correct=1.0, format=1.0, chars=11249
- `depth50/correct_invalid`: depth 50, line 142, correct=1.0, format=1.0, chars=11032
- `depth50/incorrect`: depth 50, line 156, correct=0.0, format=1.0, chars=10928
- `cap_hit_chunk_75_longest_retained`: depth 40, line 728, correct=1.0, format=1.0, chars=8798
- `cap_hit_chunk_76_longest_retained`: depth 50, line 730, correct=1.0, format=1.0, chars=11028
- `cap_hit_chunk_77_longest_retained`: depth 50, line 744, correct=1.0, format=1.0, chars=11080

### logic train 1..25 seed 3408
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=515
- `train_edge/correct_valid`: depth 25, line 137, correct=1.0, format=1.0, chars=5715
- `ood_edge/correct_valid`: depth 30, line 138, correct=1.0, format=1.0, chars=6777
- `ood_edge/correct_invalid`: depth 30, line 320, correct=1.0, format=1.0, chars=6784
- `ood_edge/incorrect`: depth 30, line 194, correct=0.0, format=1.0, chars=6687
- `depth50/correct_valid`: depth 50, line 142, correct=1.0, format=1.0, chars=11030
- `depth50/correct_invalid`: depth 50, line 520, correct=1.0, format=1.0, chars=13317
- `depth50/incorrect`: depth 50, line 184, correct=0.0, format=1.0, chars=10904
- `cap_hit_chunk_81_longest_retained`: depth 50, line 772, correct=1.0, format=1.0, chars=11166
- `cap_hit_chunk_86_longest_retained`: depth 40, line 812, correct=0.0, format=0.0, chars=13044
- `cap_hit_chunk_88_longest_retained`: depth 50, line 828, correct=1.0, format=1.0, chars=11028

### logic train 1..25 seed 3409
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=515
- `train_edge/correct_valid`: depth 25, line 137, correct=1.0, format=1.0, chars=5715
- `ood_edge/correct_valid`: depth 30, line 138, correct=1.0, format=1.0, chars=6865
- `ood_edge/correct_invalid`: depth 30, line 250, correct=1.0, format=1.0, chars=6534
- `ood_edge/incorrect`: depth 30, line 180, correct=0.0, format=1.0, chars=6485
- `depth50/correct_valid`: depth 50, line 142, correct=1.0, format=1.0, chars=11030
- `depth50/correct_invalid`: depth 50, line 156, correct=1.0, format=1.0, chars=11025
- `depth50/incorrect`: depth 50, line 170, correct=0.0, format=1.0, chars=11029
- `cap_hit_chunk_73_longest_retained`: depth 30, line 712, correct=1.0, format=1.0, chars=6781
- `cap_hit_chunk_74_longest_retained`: depth 50, line 716, correct=1.0, format=1.0, chars=11036
- `cap_hit_chunk_75_longest_retained`: depth 40, line 728, correct=1.0, format=1.0, chars=8798

### nl_exact train 1..5 seed 3407
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=519
- `train_edge/correct_valid`: depth 5, line 131, correct=1.0, format=1.0, chars=1966
- `ood_edge/correct_valid`: depth 10, line 132, correct=1.0, format=1.0, chars=3771
- `depth50/correct_valid`: depth 50, line 170, correct=1.0, format=1.0, chars=5577
- `depth50/correct_invalid`: depth 50, line 870, correct=1.0, format=0.0, chars=4441
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=11958
- `cap_hit_chunk_62_longest_retained`: depth 45, line 617, correct=0.0, format=0.0, chars=12107
- `cap_hit_chunk_68_longest_retained`: depth 35, line 671, correct=0.0, format=0.0, chars=12069
- `cap_hit_chunk_69_longest_retained`: depth 50, line 674, correct=0.0, format=0.0, chars=12638

### nl_exact train 1..5 seed 3408
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=519
- `train_edge/correct_valid`: depth 5, line 131, correct=1.0, format=1.0, chars=1966
- `ood_edge/correct_valid`: depth 10, line 132, correct=1.0, format=1.0, chars=3771
- `depth50/correct_valid`: depth 50, line 744, correct=1.0, format=1.0, chars=7034
- `depth50/correct_invalid`: depth 50, line 310, correct=1.0, format=1.0, chars=6199
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=10858
- `cap_hit_chunk_56_longest_retained`: depth 35, line 573, correct=0.0, format=0.0, chars=11610
- `cap_hit_chunk_87_longest_retained`: depth 25, line 823, correct=1.0, format=1.0, chars=9535
- `cap_hit_chunk_95_longest_retained`: depth 50, line 884, correct=0.0, format=0.0, chars=11891

### nl_exact train 1..5 seed 3409
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=519
- `train_edge/correct_valid`: depth 5, line 131, correct=1.0, format=1.0, chars=1966
- `ood_edge/correct_valid`: depth 10, line 132, correct=1.0, format=1.0, chars=3771
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=10326
- `cap_hit_chunk_57_longest_retained`: depth 20, line 584, correct=1.0, format=1.0, chars=7619
- `cap_hit_chunk_73_longest_retained`: depth 30, line 712, correct=0.0, format=0.0, chars=11130
- `cap_hit_chunk_74_longest_retained`: depth 35, line 713, correct=0.0, format=0.0, chars=11890

### nl_exact train 1..10 seed 3407
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=519
- `train_edge/correct_valid`: depth 10, line 132, correct=1.0, format=1.0, chars=3771
- `ood_edge/correct_valid`: depth 12, line 133, correct=1.0, format=1.0, chars=4542
- `depth50/correct_valid`: depth 50, line 184, correct=1.0, format=1.0, chars=19001
- `depth50/correct_invalid`: depth 50, line 548, correct=1.0, format=0.0, chars=17585
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=12857
- `cap_hit_chunk_85_longest_retained`: depth 20, line 808, correct=1.0, format=1.0, chars=7630
- `cap_hit_chunk_88_longest_retained`: depth 40, line 826, correct=0.0, format=0.0, chars=14042
- `cap_hit_chunk_90_longest_retained`: depth 45, line 841, correct=0.0, format=0.0, chars=12705

### nl_exact train 1..10 seed 3408
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=519
- `train_edge/correct_valid`: depth 10, line 132, correct=1.0, format=1.0, chars=3771
- `ood_edge/correct_valid`: depth 12, line 133, correct=1.0, format=1.0, chars=4542
- `depth50/correct_invalid`: depth 50, line 786, correct=1.0, format=0.0, chars=12087
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=11701
- `cap_hit_chunk_74_longest_retained`: depth 45, line 715, correct=0.0, format=0.0, chars=20283
- `cap_hit_chunk_85_longest_retained`: depth 20, line 808, correct=1.0, format=1.0, chars=7630
- `cap_hit_chunk_89_longest_retained`: depth 35, line 839, correct=0.0, format=0.0, chars=11656

### nl_exact train 1..10 seed 3409
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=519
- `train_edge/correct_valid`: depth 10, line 132, correct=1.0, format=1.0, chars=3771
- `ood_edge/correct_valid`: depth 12, line 133, correct=1.0, format=1.0, chars=4542
- `depth50/correct_valid`: depth 50, line 310, correct=1.0, format=1.0, chars=19102
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=11033
- `cap_hit_chunk_65_longest_retained`: depth 40, line 644, correct=0.0, format=0.0, chars=12839
- `cap_hit_chunk_67_longest_retained`: depth 35, line 657, correct=0.0, format=0.0, chars=20821
- `cap_hit_chunk_73_longest_retained`: depth 30, line 712, correct=1.0, format=1.0, chars=11412

### nl_exact train 1..15 seed 3407
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=519
- `train_edge/correct_valid`: depth 15, line 134, correct=1.0, format=1.0, chars=5719
- `ood_edge/correct_valid`: depth 18, line 135, correct=1.0, format=1.0, chars=6884
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=11718
- `cap_hit_chunk_91_longest_retained`: depth 45, line 855, correct=0.0, format=0.0, chars=11809
- `cap_hit_chunk_97_longest_retained`: depth 45, line 897, correct=0.0, format=0.0, chars=12023
- `cap_hit_chunk_105_longest_retained`: depth 50, line 968, correct=0.0, format=0.0, chars=11703

### nl_exact train 1..15 seed 3408
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=519
- `train_edge/correct_valid`: depth 15, line 134, correct=1.0, format=1.0, chars=5719
- `ood_edge/correct_valid`: depth 18, line 135, correct=1.0, format=1.0, chars=6884
- `depth50/correct_invalid`: depth 50, line 366, correct=1.0, format=0.0, chars=17710
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=11949
- `cap_hit_chunk_83_longest_retained`: depth 45, line 785, correct=0.0, format=0.0, chars=20775
- `cap_hit_chunk_84_longest_retained`: depth 50, line 800, correct=0.0, format=0.0, chars=20904
- `cap_hit_chunk_85_longest_retained`: depth 20, line 808, correct=1.0, format=1.0, chars=7630

### nl_exact train 1..15 seed 3409
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=519
- `train_edge/correct_valid`: depth 15, line 134, correct=1.0, format=1.0, chars=5719
- `ood_edge/correct_valid`: depth 18, line 135, correct=1.0, format=1.0, chars=6884
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=11813
- `cap_hit_chunk_92_longest_retained`: depth 20, line 864, correct=1.0, format=1.0, chars=7632
- `cap_hit_chunk_93_longest_retained`: depth 40, line 868, correct=0.0, format=0.0, chars=11970
- `cap_hit_chunk_100_longest_retained`: depth 50, line 926, correct=0.0, format=0.0, chars=11963

### nl_exact train 1..20 seed 3407
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=519
- `train_edge/correct_valid`: depth 20, line 136, correct=1.0, format=1.0, chars=7619
- `ood_edge/correct_valid`: depth 25, line 137, correct=1.0, format=1.0, chars=9465
- `depth50/correct_valid`: depth 50, line 156, correct=1.0, format=1.0, chars=19116
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=11804
- `cap_hit_chunk_23_longest_retained`: depth 50, line 310, correct=0.0, format=0.0, chars=13351
- `cap_hit_chunk_81_longest_retained`: depth 40, line 770, correct=0.0, format=0.0, chars=12783
- `cap_hit_chunk_82_longest_retained`: depth 35, line 783, correct=0.0, format=0.0, chars=11926

### nl_exact train 1..20 seed 3408
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=519
- `train_edge/correct_valid`: depth 20, line 136, correct=1.0, format=1.0, chars=7619
- `ood_edge/correct_valid`: depth 25, line 137, correct=1.0, format=1.0, chars=9465
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=11831

### nl_exact train 1..20 seed 3409
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=519
- `train_edge/correct_valid`: depth 20, line 136, correct=1.0, format=1.0, chars=7619
- `ood_edge/correct_valid`: depth 25, line 137, correct=1.0, format=1.0, chars=9465
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=11771

### nl_exact train 1..25 seed 3407
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=519
- `train_edge/correct_valid`: depth 25, line 137, correct=1.0, format=1.0, chars=9465
- `ood_edge/correct_valid`: depth 30, line 138, correct=1.0, format=1.0, chars=11388
- `ood_edge/incorrect`: depth 30, line 264, correct=0.0, format=0.0, chars=11277
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=11755
- `cap_hit_chunk_105_longest_retained`: depth 35, line 965, correct=1.0, format=1.0, chars=13286

### nl_exact train 1..25 seed 3408
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=519
- `train_edge/correct_valid`: depth 25, line 137, correct=1.0, format=1.0, chars=9465
- `ood_edge/correct_valid`: depth 30, line 138, correct=1.0, format=1.0, chars=11388
- `ood_edge/incorrect`: depth 30, line 152, correct=0.0, format=0.0, chars=11218
- `depth50/correct_valid`: depth 50, line 926, correct=1.0, format=1.0, chars=19084
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=13464

### nl_exact train 1..25 seed 3409
- `shallow/correct_valid`: depth 1, line 129, correct=1.0, format=1.0, chars=519
- `train_edge/correct_valid`: depth 25, line 137, correct=1.0, format=1.0, chars=9465
- `ood_edge/correct_valid`: depth 30, line 138, correct=1.0, format=1.0, chars=11388
- `ood_edge/incorrect`: depth 30, line 180, correct=0.0, format=0.0, chars=11448
- `depth50/incorrect`: depth 50, line 142, correct=0.0, format=0.0, chars=11791
- `cap_hit_chunk_82_longest_retained`: depth 35, line 783, correct=0.0, format=0.0, chars=20768
- `cap_hit_chunk_83_longest_retained`: depth 45, line 785, correct=0.0, format=0.0, chars=12520
- `cap_hit_chunk_84_longest_retained`: depth 45, line 799, correct=0.0, format=0.0, chars=15897
