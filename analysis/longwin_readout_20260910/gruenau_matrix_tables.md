
### base / deduction / seed 3407   (midtrained base, no chat template; separate table)
| task | metric | Control | Formal | English |
|---|---|---|---|---|
| deduction_bp_cot_d10 | exact_match | 0.030 | 0.120 | 0.140 |
| deduction_bp_cot_d15 | exact_match | 0.025 | 0.055 | 0.055 |
| deduction_bp_cot_d20 | exact_match | 0.015 | 0.055 | 0.020 |
| deduction_bp_cot_d25 | exact_match | 0.040 | 0.035 | 0.030 |
| deduction_bp_cot_d5 | exact_match | 0.035 | 0.100 | 0.320 |
| deduction_pw_d0 | exact_match | 0.362 | 0.490 | 0.440 |
| deduction_pw_d1 | exact_match | 0.342 | 0.356 | 0.350 |
| deduction_pw_d2 | exact_match | 0.508 | 0.476 | 0.490 |
| deduction_pw_d3 | exact_match | 0.506 | 0.484 | 0.504 |
| deduction_pw_d5 | exact_match | 0.576 | 0.494 | 0.530 |

### base / deduction_cot / seed 3407   (midtrained base, no chat template; separate table)
| task | metric | Control | Formal | English |
|---|---|---|---|---|
| deduction_pw_cot_d0 | exact_match | 0.526 | 0.536 | 0.504 |
| deduction_pw_cot_d1 | exact_match | 0.404 | 0.498 | 0.436 |
| deduction_pw_cot_d2 | exact_match | 0.348 | 0.326 | 0.286 |
| deduction_pw_cot_d3 | exact_match | 0.352 | 0.286 | 0.240 |
| deduction_pw_cot_d5 | exact_match | 0.316 | 0.158 | 0.118 |

### base / deduction_mc / seed 3407   (midtrained base, no chat template; separate table)
| task | metric | Control | Formal | English |
|---|---|---|---|---|
| deduction_pw_mc_d0 | acc_norm | 0.610 | 0.732 | 0.684 |
| deduction_pw_mc_d1 | acc_norm | 0.508 | 0.544 | 0.518 |
| deduction_pw_mc_d2 | acc_norm | 0.456 | 0.450 | 0.488 |
| deduction_pw_mc_d3 | acc_norm | 0.456 | 0.430 | 0.484 |
| deduction_pw_mc_d5 | acc_norm | 0.416 | 0.354 | 0.388 |

### base / deduction_native / seed 3407   (midtrained base, no chat template; separate table)
| task | metric | Control | Formal | English |
|---|---|---|---|---|
| deduction_bp_native_d10 | exact_match | 0.000 | 1.000 | 1.000 |
| deduction_bp_native_d15 | exact_match | 0.000 | 1.000 | 1.000 |
| deduction_bp_native_d20 | exact_match | 0.000 | 0.995 | 1.000 |
| deduction_bp_native_d25 | exact_match | 0.000 | 0.240 | 0.000 |
| deduction_bp_native_d5 | exact_match | 0.000 | 0.990 | 1.000 |

### base / deduction_native_long / seed 3407   (midtrained base, no chat template; separate table)
| task | metric | Formal | English |
|---|---|---|---|
| deduction_bp_native_long_d25 | exact_match | 1.000 | 0.985 |

### base / deduction_pert / seed 3407   (midtrained base, no chat template; separate table)
| task | metric | Control | Formal | English |
|---|---|---|---|---|
| deduction_pw_ablate | exact_match | 0.115 | 0.246 | 0.164 |
| deduction_pw_flip | exact_match | 0.639 | 0.585 | 0.618 |

### base / multihop / seed 3407   (midtrained base, no chat template; separate table)
| task | metric | Control | Formal | English |
|---|---|---|---|---|
| longbench_2wikimqa_standard | qa_f1_score | 0.349 | 0.001 | 0.036 |
| longbench_2wikimqa_tagged | exact_match | 0.240 | 0.275 | 0.265 |
| longbench_hotpotqa_standard | qa_f1_score | 0.542 | 0.002 | 0.016 |
| longbench_hotpotqa_tagged | exact_match | 0.395 | 0.460 | 0.435 |
| longbench_musique_standard | qa_f1_score | 0.306 | 0.001 | 0.008 |
| longbench_musique_tagged | exact_match | 0.125 | 0.150 | 0.150 |

### base / standard / seed 3407   (midtrained base, no chat template; separate table)
| task | metric | Control | English |
|---|---|---|---|
| agieval_logiqa_en | acc_norm | 0.415 | 0.419 |
| arc_challenge | acc_norm | 0.599 | 0.613 |
| bbh | exact_match | 0.674 | 0.671 |
| gsm8k | exact_match | 0.805 | 0.826 |
| hellaswag | acc_norm | 0.774 | 0.777 |
| hendrycks_math500 | exact_match | 0.202 | 0.184 |
| mmlu | acc | 0.695 | 0.688 |
| piqa | acc_norm | 0.802 | 0.801 |
| winogrande | acc | 0.734 | 0.731 |

### it / deduction / seed 3407
| task | metric | Control | LongDoc | Formal | English | Condensed | Ctl+FormalSFT | Ctl+EnglishSFT | Formal+FormalSFT | English+EnglishSFT | Formal+ModeCond |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deduction_bp_cot_d10 | exact_match | 0.040 | 0.115 | 0.175 | 0.360 | 0.290 | 0.430 | 0.960 | 0.230 | 0.370 | 0.830 |
| deduction_bp_cot_d15 | exact_match | 0.100 | 0.135 | 0.245 | 0.335 | 0.200 | 0.215 | 0.955 | 0.250 | 0.285 | 0.765 |
| deduction_bp_cot_d20 | exact_match | 0.100 | 0.185 | 0.195 | 0.260 | 0.210 | 0.230 | 0.900 | 0.215 | 0.195 | 0.780 |
| deduction_bp_cot_d25 | exact_match | 0.070 | 0.115 | 0.185 | 0.310 | 0.220 | 0.260 | 0.870 | 0.205 | 0.305 | 0.745 |
| deduction_bp_cot_d5 | exact_match | 0.045 | 0.080 | 0.285 | 0.515 | 0.295 | 0.475 | 0.995 | 0.285 | 0.620 | 0.985 |
| deduction_pw_d0 | exact_match | 0.442 | 0.448 | 0.476 | 0.486 | 0.456 | 0.550 | 0.510 | 0.496 | 0.492 | 0.488 |
| deduction_pw_d1 | exact_match | 0.328 | 0.320 | 0.362 | 0.388 | 0.352 | 0.380 | 0.378 | 0.386 | 0.392 | 0.380 |
| deduction_pw_d2 | exact_match | 0.436 | 0.438 | 0.496 | 0.554 | 0.462 | 0.550 | 0.526 | 0.520 | 0.546 | 0.524 |
| deduction_pw_d3 | exact_match | 0.444 | 0.446 | 0.500 | 0.570 | 0.482 | 0.558 | 0.524 | 0.536 | 0.562 | 0.524 |
| deduction_pw_d5 | exact_match | 0.452 | 0.434 | 0.506 | 0.570 | 0.464 | 0.546 | 0.526 | 0.522 | 0.562 | 0.528 |

### it / deduction_cot / seed 3407
| task | metric | Control | LongDoc | Formal | English | Condensed | Ctl+FormalSFT | Ctl+EnglishSFT | Formal+FormalSFT | English+EnglishSFT |
|---|---|---|---|---|---|---|---|---|---|---|
| deduction_pw_cot_d0 | exact_match | 0.516 | 0.654 | 0.464 | 0.548 | 0.562 | 0.518 | 0.388 | 0.472 | 0.468 |
| deduction_pw_cot_d1 | exact_match | 0.420 | 0.496 | 0.394 | 0.464 | 0.520 | 0.446 | 0.358 | 0.420 | 0.424 |
| deduction_pw_cot_d2 | exact_match | 0.516 | 0.588 | 0.442 | 0.510 | 0.504 | 0.484 | 0.386 | 0.502 | 0.496 |
| deduction_pw_cot_d3 | exact_match | 0.470 | 0.540 | 0.494 | 0.526 | 0.498 | 0.468 | 0.422 | 0.436 | 0.486 |
| deduction_pw_cot_d5 | exact_match | 0.480 | 0.514 | 0.474 | 0.488 | 0.532 | 0.446 | 0.420 | 0.476 | 0.442 |

### it / deduction_cot / seed 3408
| task | metric | Control | LongDoc | Formal | English | Condensed |
|---|---|---|---|---|---|---|
| deduction_pw_cot_d0 | exact_match | 0.538 | 0.674 | 0.482 | 0.574 | 0.552 |
| deduction_pw_cot_d1 | exact_match | 0.432 | 0.486 | 0.412 | 0.474 | 0.454 |
| deduction_pw_cot_d2 | exact_match | 0.522 | 0.562 | 0.410 | 0.530 | 0.484 |
| deduction_pw_cot_d3 | exact_match | 0.464 | 0.526 | 0.476 | 0.534 | 0.518 |
| deduction_pw_cot_d5 | exact_match | 0.464 | 0.496 | 0.488 | 0.512 | 0.472 |

### it / deduction_mc / seed 3407
| task | metric | Control | LongDoc | Formal | English | Condensed | Ctl+FormalSFT | Ctl+EnglishSFT | Formal+FormalSFT | English+EnglishSFT |
|---|---|---|---|---|---|---|---|---|---|---|
| deduction_pw_mc_d0 | acc_norm | 0.564 | 0.548 | 0.662 | 0.674 | 0.538 | 0.654 | 0.618 | 0.704 | 0.682 |
| deduction_pw_mc_d1 | acc_norm | 0.380 | 0.378 | 0.482 | 0.506 | 0.390 | 0.428 | 0.456 | 0.528 | 0.544 |
| deduction_pw_mc_d2 | acc_norm | 0.470 | 0.480 | 0.496 | 0.464 | 0.494 | 0.536 | 0.520 | 0.430 | 0.464 |
| deduction_pw_mc_d3 | acc_norm | 0.476 | 0.472 | 0.448 | 0.448 | 0.500 | 0.558 | 0.526 | 0.410 | 0.412 |
| deduction_pw_mc_d5 | acc_norm | 0.436 | 0.442 | 0.416 | 0.384 | 0.456 | 0.558 | 0.480 | 0.334 | 0.344 |

### it / deduction_mc / seed 3408
| task | metric | Control | LongDoc | Formal | English | Condensed |
|---|---|---|---|---|---|---|
| deduction_pw_mc_d0 | acc_norm | 0.592 | 0.552 | 0.628 | 0.666 | 0.540 |
| deduction_pw_mc_d1 | acc_norm | 0.382 | 0.366 | 0.448 | 0.488 | 0.382 |
| deduction_pw_mc_d2 | acc_norm | 0.462 | 0.480 | 0.512 | 0.480 | 0.498 |
| deduction_pw_mc_d3 | acc_norm | 0.466 | 0.476 | 0.514 | 0.474 | 0.486 |
| deduction_pw_mc_d5 | acc_norm | 0.408 | 0.452 | 0.470 | 0.434 | 0.448 |

### it / deduction_native / seed 3407
| task | metric | Control | LongDoc | Formal | English | Condensed | Ctl+FormalSFT | Ctl+EnglishSFT | Formal+FormalSFT | English+EnglishSFT |
|---|---|---|---|---|---|---|---|---|---|---|
| deduction_bp_native_d10 | exact_match | 0.000 | 0.000 | 0.000 | 0.230 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| deduction_bp_native_d15 | exact_match | 0.000 | 0.000 | 0.000 | 0.235 | 0.000 | 0.990 | 1.000 | 1.000 | 1.000 |
| deduction_bp_native_d20 | exact_match | 0.000 | 0.000 | 0.000 | 0.255 | 0.000 | 0.995 | 1.000 | 1.000 | 1.000 |
| deduction_bp_native_d25 | exact_match | 0.000 | 0.000 | 0.010 | 0.000 | 0.000 | 0.165 | 0.250 | 0.210 | 0.265 |
| deduction_bp_native_d5 | exact_match | 0.000 | 0.000 | 0.000 | 0.240 | 0.070 | 1.000 | 1.000 | 1.000 | 1.000 |

### it / deduction_native_long / seed 3407
| task | metric | Ctl+FormalSFT | Ctl+EnglishSFT | Formal+FormalSFT | English+EnglishSFT |
|---|---|---|---|---|---|
| deduction_bp_native_long_d25 | exact_match | 0.980 | 0.995 | 1.000 | 1.000 |

### it / deduction_pert / seed 3407
| task | metric | Control | LongDoc | Formal | English | Condensed | Ctl+FormalSFT | Ctl+EnglishSFT | Formal+FormalSFT | English+EnglishSFT |
|---|---|---|---|---|---|---|---|---|---|---|
| deduction_pw_ablate | exact_match | 0.214 | 0.195 | 0.219 | 0.182 | 0.179 | 0.274 | 0.265 | 0.219 | 0.213 |
| deduction_pw_flip | exact_match | 0.562 | 0.558 | 0.630 | 0.712 | 0.596 | 0.691 | 0.639 | 0.674 | 0.711 |

### it / deduction_pert / seed 3408
| task | metric | Control | LongDoc | Formal | English | Condensed |
|---|---|---|---|---|---|---|
| deduction_pw_ablate | exact_match | 0.240 | 0.201 | 0.210 | 0.178 | 0.187 |
| deduction_pw_flip | exact_match | 0.554 | 0.560 | 0.634 | 0.711 | 0.604 |

### it / folio_gpqa / seed 3407
| task | metric | Control | LongDoc | Formal | English | Condensed | Ctl+FormalSFT | Ctl+EnglishSFT | Formal+FormalSFT | English+EnglishSFT | Formal+ModeCond |
|---|---|---|---|---|---|---|---|---|---|---|---|
| folio | exact_match | 0.557 | 0.567 | 0.606 | 0.591 | 0.591 | 0.581 | 0.542 | 0.606 | 0.591 | 0.611 |
| gpqa_diamond | exact_match | 0.348 | 0.308 | 0.333 | 0.278 | 0.354 | 0.359 | 0.343 | 0.359 | 0.338 | 0.354 |

### it / folio_gpqa / seed 3408
| task | metric | Control | LongDoc | Formal | English | Condensed |
|---|---|---|---|---|---|---|
| folio | exact_match | 0.552 | 0.581 | 0.606 | 0.586 | 0.581 |
| gpqa_diamond | exact_match | 0.338 | 0.374 | 0.313 | 0.318 | 0.308 |

### it / multihop / seed 3407
| task | metric | Control | LongDoc | Formal | English | Condensed | Ctl+FormalSFT | Ctl+EnglishSFT | Formal+FormalSFT | English+EnglishSFT |
|---|---|---|---|---|---|---|---|---|---|---|
| longbench_2wikimqa_standard | qa_f1_score | 0.389 | 0.353 | 0.364 | 0.382 | 0.343 | 0.378 | 0.373 | 0.381 | 0.382 |
| longbench_2wikimqa_tagged | exact_match | 0.040 | 0.040 | 0.130 | 0.155 | 0.025 | 0.305 | 0.210 | 0.280 | 0.180 |
| longbench_hotpotqa_standard | qa_f1_score | 0.557 | 0.576 | 0.571 | 0.559 | 0.570 | 0.551 | 0.553 | 0.562 | 0.553 |
| longbench_hotpotqa_tagged | exact_match | 0.280 | 0.260 | 0.380 | 0.310 | 0.215 | 0.410 | 0.110 | 0.420 | 0.145 |
| longbench_musique_standard | qa_f1_score | 0.299 | 0.302 | 0.273 | 0.272 | 0.268 | 0.285 | 0.307 | 0.256 | 0.272 |
| longbench_musique_tagged | exact_match | 0.180 | 0.175 | 0.140 | 0.125 | 0.095 | 0.195 | 0.040 | 0.170 | 0.025 |

### it / standard / seed 3407
| task | metric | Control | LongDoc | Formal | English | Ctl+FormalSFT | Ctl+EnglishSFT | Formal+FormalSFT | English+EnglishSFT | Formal+ModeCond |
|---|---|---|---|---|---|---|---|---|---|---|
| agieval_logiqa_en | acc_norm | 0.364 | 0.363 | 0.350 | 0.347 | 0.367 | 0.367 | 0.373 | 0.366 | 0.366 |
| arc_challenge | acc_norm | 0.550 | 0.568 | 0.574 | 0.540 | 0.549 | 0.561 | 0.560 | 0.543 | 0.571 |
| bbh | exact_match | 0.677 | 0.681 | 0.683 | 0.682 | 0.689 | 0.691 | 0.693 | 0.683 | 0.694 |
| gsm8k | exact_match | 0.787 | 0.786 | 0.788 | 0.789 | 0.788 | 0.784 | 0.785 | 0.788 | 0.784 |
| hellaswag | acc_norm | 0.734 | 0.741 | 0.730 | 0.731 | 0.734 | 0.731 | 0.732 | 0.731 | 0.733 |
| hendrycks_math500 | exact_match | 0.000 | 0.002 | 0.002 | 0.000 | 0.002 | 0.002 | 0.000 | 0.000 | 0.002 |
| mmlu | acc | 0.697 | 0.695 | 0.691 | 0.692 | 0.687 | 0.692 | 0.689 | 0.689 | 0.686 |
| piqa | acc_norm | 0.800 | 0.805 | 0.797 | 0.801 | 0.804 | 0.804 | 0.803 | 0.799 | 0.801 |
| winogrande | acc | 0.698 | 0.703 | 0.700 | 0.704 | 0.690 | 0.681 | 0.693 | 0.702 | 0.691 |

### it / switch_english / seed 3407
| task | metric | Formal+ModeCond |
|---|---|---|
| deduction_bp_switch_english_d10 | exact_match | 1.000 |
| deduction_bp_switch_english_d15 | exact_match | 1.000 |
| deduction_bp_switch_english_d20 | exact_match | 1.000 |
| deduction_bp_switch_english_d25 | exact_match | 1.000 |
| deduction_bp_switch_english_d5 | exact_match | 1.000 |

### it / switch_formal / seed 3407
| task | metric | Formal+ModeCond |
|---|---|---|
| deduction_bp_switch_formal_d10 | exact_match | 1.000 |
| deduction_bp_switch_formal_d15 | exact_match | 1.000 |
| deduction_bp_switch_formal_d20 | exact_match | 1.000 |
| deduction_bp_switch_formal_d25 | exact_match | 0.995 |
| deduction_bp_switch_formal_d5 | exact_match | 1.000 |

* = bundle not yet marked complete
