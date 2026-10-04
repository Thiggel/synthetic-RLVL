# Stage 3: frozen-backbone value probe on e4

Model `/vol/tmp2/laitenbf/rlvl_data/formal_mixture_sft_20260925/qwen35_2b_lc_libext_e4_lr5em6_seed3407/final`; rollouts from `2b_e2_G16_cvffmt_overlong_noproof:500-1000`, `2b_e3_G17_cvffmt_overlong_noproof`; 16000 completions, 165302 prefix positions; prompt-disjoint split (1459 train / 359 test prompts). Base rates: correct 0.475, cvf 0.334, valid 0.544.

AUC of a logistic probe at the end of the prompt (`prompt`), after proof lines by relative depth, and after the last line (`end`). *Within-prompt* AUC compares completions of the same prompt and step at the same depth bucket (0.5 = no help for search).

| features:target | AUC | prompt | 0-25% | 25-50% | 50-75% | 75-100% | end | within-prompt | wp 0-25% | wp 25-50% | wp 50-75% | wp 75-100% | wp end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| last:correct | 0.904 | 0.907 | 0.906 | 0.901 | 0.901 | 0.905 | 0.911 | 0.562 | 0.583 | 0.515 | 0.549 | 0.634 | 0.583 |
| last:cvf | 0.905 | 0.882 | 0.882 | 0.905 | 0.911 | 0.917 | 0.927 | 0.655 | 0.617 | 0.600 | 0.698 | 0.742 | 0.755 |
| last:valid | 0.862 | 0.755 | 0.811 | 0.861 | 0.885 | 0.901 | 0.902 | 0.672 | 0.581 | 0.663 | 0.710 | 0.748 | 0.760 |
| mid:correct | 0.909 | 0.906 | 0.910 | 0.902 | 0.910 | 0.914 | 0.913 | 0.519 | 0.505 | 0.451 | 0.553 | 0.572 | 0.577 |
| mid:cvf | 0.904 | 0.887 | 0.884 | 0.899 | 0.913 | 0.920 | 0.920 | 0.584 | 0.501 | 0.473 | 0.672 | 0.681 | 0.673 |
| mid:valid | 0.865 | 0.755 | 0.811 | 0.868 | 0.890 | 0.909 | 0.886 | 0.662 | 0.574 | 0.651 | 0.720 | 0.789 | 0.741 |
