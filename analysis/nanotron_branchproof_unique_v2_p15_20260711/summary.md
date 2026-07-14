# Corrected Nanotron p15 downstream comparison

Each condition is one continuation-training run. Task-level stderr values come from lm-eval, except MATH-500, which uses answer-prefix symbolic equivalence. The stock MATH exact score remains in the CSV as a format-sensitive diagnostic. Macro rows are unweighted task means and do not estimate training-seed variance.

| condition | branch | macro | score | delta vs control | instruction - direct |
| --- | --- | --- | ---: | ---: | ---: |
| `control` | `direct` | `all_primary` | 0.6052 | +0.0000 | -0.1060 |
| `control` | `direct` | `reasoning_core` | 0.4946 | +0.0000 | -0.1895 |
| `control` | `direct` | `general_multiple_choice` | 0.7158 | +0.0000 | -0.0226 |
| `control` | `direct` | `logic_targeted` | 0.5316 | +0.0000 | -0.3685 |
| `control` | `instruction` | `all_primary` | 0.4992 | +0.0000 | -0.1060 |
| `control` | `instruction` | `reasoning_core` | 0.3051 | +0.0000 | -0.1895 |
| `control` | `instruction` | `general_multiple_choice` | 0.6932 | +0.0000 | -0.0226 |
| `control` | `instruction` | `logic_targeted` | 0.1631 | +0.0000 | -0.3685 |
| `logic` | `direct` | `all_primary` | 0.6085 | +0.0033 | -0.1075 |
| `logic` | `direct` | `reasoning_core` | 0.5017 | +0.0071 | -0.1927 |
| `logic` | `direct` | `general_multiple_choice` | 0.7154 | -0.0004 | -0.0223 |
| `logic` | `direct` | `logic_targeted` | 0.5200 | -0.0116 | -0.3575 |
| `logic` | `instruction` | `all_primary` | 0.5010 | +0.0018 | -0.1075 |
| `logic` | `instruction` | `reasoning_core` | 0.3090 | +0.0038 | -0.1927 |
| `logic` | `instruction` | `general_multiple_choice` | 0.6931 | -0.0001 | -0.0223 |
| `logic` | `instruction` | `logic_targeted` | 0.1626 | -0.0005 | -0.3575 |
| `nl_exact` | `direct` | `all_primary` | 0.6041 | -0.0011 | -0.1022 |
| `nl_exact` | `direct` | `reasoning_core` | 0.4935 | -0.0012 | -0.1787 |
| `nl_exact` | `direct` | `general_multiple_choice` | 0.7147 | -0.0011 | -0.0256 |
| `nl_exact` | `direct` | `logic_targeted` | 0.5247 | -0.0069 | -0.3629 |
| `nl_exact` | `instruction` | `all_primary` | 0.5019 | +0.0027 | -0.1022 |
| `nl_exact` | `instruction` | `reasoning_core` | 0.3147 | +0.0096 | -0.1787 |
| `nl_exact` | `instruction` | `general_multiple_choice` | 0.6891 | -0.0041 | -0.0256 |
| `nl_exact` | `instruction` | `logic_targeted` | 0.1618 | -0.0013 | -0.3629 |

## Targeted task results

| condition | branch | task | score | stderr | delta vs control | instruction - direct |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `control` | `direct` | `agieval_logiqa_en` | 0.4178 | 0.0193 | +0.0000 | -0.0108 |
| `control` | `direct` | `mmlu_formal_logic` | 0.5238 | 0.0447 | +0.0000 | +0.0476 |
| `control` | `direct` | `bbh_cot_fewshot_boolean_expressions` | 0.9120 | 0.0180 | +0.0000 | -0.9120 |
| `control` | `direct` | `bbh_cot_fewshot_formal_fallacies` | 0.5520 | 0.0315 | +0.0000 | -0.5520 |
| `control` | `direct` | `bbh_cot_fewshot_logical_deduction_three_objects` | 0.8400 | 0.0232 | +0.0000 | -0.8400 |
| `control` | `direct` | `bbh_cot_fewshot_logical_deduction_five_objects` | 0.4800 | 0.0317 | +0.0000 | -0.4800 |
| `control` | `direct` | `bbh_cot_fewshot_logical_deduction_seven_objects` | 0.3760 | 0.0307 | +0.0000 | -0.3760 |
| `control` | `instruction` | `agieval_logiqa_en` | 0.4071 | 0.0193 | +0.0000 | -0.0108 |
| `control` | `instruction` | `mmlu_formal_logic` | 0.5714 | 0.0443 | +0.0000 | +0.0476 |
| `control` | `instruction` | `bbh_cot_fewshot_boolean_expressions` | 0.0000 | 0.0000 | +0.0000 | -0.9120 |
| `control` | `instruction` | `bbh_cot_fewshot_formal_fallacies` | 0.0000 | 0.0000 | +0.0000 | -0.5520 |
| `control` | `instruction` | `bbh_cot_fewshot_logical_deduction_three_objects` | 0.0000 | 0.0000 | +0.0000 | -0.8400 |
| `control` | `instruction` | `bbh_cot_fewshot_logical_deduction_five_objects` | 0.0000 | 0.0000 | +0.0000 | -0.4800 |
| `control` | `instruction` | `bbh_cot_fewshot_logical_deduction_seven_objects` | 0.0000 | 0.0000 | +0.0000 | -0.3760 |
| `logic` | `direct` | `agieval_logiqa_en` | 0.4163 | 0.0193 | -0.0015 | -0.0123 |
| `logic` | `direct` | `mmlu_formal_logic` | 0.5079 | 0.0447 | -0.0159 | +0.0635 |
| `logic` | `direct` | `bbh_cot_fewshot_boolean_expressions` | 0.8920 | 0.0197 | -0.0200 | -0.8920 |
| `logic` | `direct` | `bbh_cot_fewshot_formal_fallacies` | 0.5240 | 0.0316 | -0.0280 | -0.5240 |
| `logic` | `direct` | `bbh_cot_fewshot_logical_deduction_three_objects` | 0.8200 | 0.0243 | -0.0200 | -0.8200 |
| `logic` | `direct` | `bbh_cot_fewshot_logical_deduction_five_objects` | 0.4600 | 0.0316 | -0.0200 | -0.4600 |
| `logic` | `direct` | `bbh_cot_fewshot_logical_deduction_seven_objects` | 0.3920 | 0.0309 | +0.0160 | -0.3920 |
| `logic` | `instruction` | `agieval_logiqa_en` | 0.4040 | 0.0192 | -0.0031 | -0.0123 |
| `logic` | `instruction` | `mmlu_formal_logic` | 0.5714 | 0.0443 | +0.0000 | +0.0635 |
| `logic` | `instruction` | `bbh_cot_fewshot_boolean_expressions` | 0.0000 | 0.0000 | +0.0000 | -0.8920 |
| `logic` | `instruction` | `bbh_cot_fewshot_formal_fallacies` | 0.0000 | 0.0000 | +0.0000 | -0.5240 |
| `logic` | `instruction` | `bbh_cot_fewshot_logical_deduction_three_objects` | 0.0000 | 0.0000 | +0.0000 | -0.8200 |
| `logic` | `instruction` | `bbh_cot_fewshot_logical_deduction_five_objects` | 0.0000 | 0.0000 | +0.0000 | -0.4600 |
| `logic` | `instruction` | `bbh_cot_fewshot_logical_deduction_seven_objects` | 0.0000 | 0.0000 | +0.0000 | -0.3920 |
| `nl_exact` | `direct` | `agieval_logiqa_en` | 0.4163 | 0.0193 | -0.0015 | -0.0169 |
| `nl_exact` | `direct` | `mmlu_formal_logic` | 0.4921 | 0.0447 | -0.0317 | +0.0794 |
| `nl_exact` | `direct` | `bbh_cot_fewshot_boolean_expressions` | 0.9080 | 0.0183 | -0.0040 | -0.9080 |
| `nl_exact` | `direct` | `bbh_cot_fewshot_formal_fallacies` | 0.5320 | 0.0316 | -0.0200 | -0.5320 |
| `nl_exact` | `direct` | `bbh_cot_fewshot_logical_deduction_three_objects` | 0.8360 | 0.0235 | -0.0040 | -0.8360 |
| `nl_exact` | `direct` | `bbh_cot_fewshot_logical_deduction_five_objects` | 0.4680 | 0.0316 | -0.0120 | -0.4680 |
| `nl_exact` | `direct` | `bbh_cot_fewshot_logical_deduction_seven_objects` | 0.4040 | 0.0311 | +0.0280 | -0.4040 |
| `nl_exact` | `instruction` | `agieval_logiqa_en` | 0.3994 | 0.0192 | -0.0077 | -0.0169 |
| `nl_exact` | `instruction` | `mmlu_formal_logic` | 0.5714 | 0.0443 | +0.0000 | +0.0794 |
| `nl_exact` | `instruction` | `bbh_cot_fewshot_boolean_expressions` | 0.0000 | 0.0000 | +0.0000 | -0.9080 |
| `nl_exact` | `instruction` | `bbh_cot_fewshot_formal_fallacies` | 0.0000 | 0.0000 | +0.0000 | -0.5320 |
| `nl_exact` | `instruction` | `bbh_cot_fewshot_logical_deduction_three_objects` | 0.0000 | 0.0000 | +0.0000 | -0.8360 |
| `nl_exact` | `instruction` | `bbh_cot_fewshot_logical_deduction_five_objects` | 0.0000 | 0.0000 | +0.0000 | -0.4680 |
| `nl_exact` | `instruction` | `bbh_cot_fewshot_logical_deduction_seven_objects` | 0.0000 | 0.0000 | +0.0000 | -0.4040 |

## Generation diagnostics

The next-document marker is a literal continuation into the Qwen assistant preamble after an answer. Character lengths are diagnostics, not token counts.

| condition | branch | family | rows | invalid extraction | response marker | prompt marker | chars p50 | chars p95 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `control` | `direct` | `gsm8k` | 1319 | 0.0000 | 0.0000 | 0.0000 | 263 | 621 |
| `control` | `direct` | `math500` | 500 | 0.0000 | 0.0000 | 0.0000 | 444 | 931 |
| `control` | `direct` | `bbh` | 6511 | 0.0438 | 0.3560 | 0.0000 | 660 | 1892 |
| `control` | `direct` | `mmlu_pro` | 12032 | 0.0455 | 0.1207 | 0.0000 | 266 | 1438 |
| `control` | `instruction` | `gsm8k` | 1319 | 0.0000 | 0.0000 | 0.0000 | 676 | 945 |
| `control` | `instruction` | `math500` | 500 | 0.0000 | 0.0000 | 0.0000 | 719 | 1031 |
| `control` | `instruction` | `bbh` | 6511 | 0.0366 | 0.0000 | 0.0000 | 3958 | 4970 |
| `control` | `instruction` | `mmlu_pro` | 12032 | 0.0122 | 0.0000 | 0.0000 | 8839 | 11051 |
| `logic` | `direct` | `gsm8k` | 1319 | 0.0000 | 0.0000 | 0.0000 | 254 | 552 |
| `logic` | `direct` | `math500` | 500 | 0.0000 | 0.0000 | 0.0000 | 405 | 946 |
| `logic` | `direct` | `bbh` | 6511 | 0.0424 | 0.5996 | 0.0000 | 694 | 1819 |
| `logic` | `direct` | `mmlu_pro` | 12032 | 0.0398 | 0.4548 | 0.0000 | 359 | 1477 |
| `logic` | `instruction` | `gsm8k` | 1319 | 0.0000 | 0.0000 | 0.0000 | 642 | 980 |
| `logic` | `instruction` | `math500` | 500 | 0.0000 | 0.0000 | 0.0000 | 744 | 1053 |
| `logic` | `instruction` | `bbh` | 6511 | 0.0347 | 0.0000 | 0.0000 | 1374 | 4473 |
| `logic` | `instruction` | `mmlu_pro` | 12032 | 0.0133 | 0.0000 | 0.0000 | 8660 | 11087 |
| `nl_exact` | `direct` | `gsm8k` | 1319 | 0.0000 | 0.0000 | 0.0000 | 263 | 637 |
| `nl_exact` | `direct` | `math500` | 500 | 0.0000 | 0.0000 | 0.0000 | 424 | 937 |
| `nl_exact` | `direct` | `bbh` | 6511 | 0.0439 | 0.5801 | 0.0000 | 705 | 1862 |
| `nl_exact` | `direct` | `mmlu_pro` | 12032 | 0.0474 | 0.4951 | 0.0000 | 375 | 2212 |
| `nl_exact` | `instruction` | `gsm8k` | 1319 | 0.0000 | 0.0000 | 0.0000 | 581 | 922 |
| `nl_exact` | `instruction` | `math500` | 500 | 0.0000 | 0.0000 | 0.0000 | 742 | 1046 |
| `nl_exact` | `instruction` | `bbh` | 6511 | 0.0370 | 0.0000 | 0.0000 | 996 | 4807 |
| `nl_exact` | `instruction` | `mmlu_pro` | 12032 | 0.0126 | 0.0000 | 0.0000 | 8736 | 11123 |

## Qualitative index

- `control/direct/gsm8k/correct`: line 1320, doc 0, score=1.0
- `control/direct/gsm8k/incorrect`: line 1324, doc 4, score=0.0
- `control/direct/agieval_logiqa_en/correct`: line 1, doc 0, score=1.0
- `control/direct/agieval_logiqa_en/incorrect`: line 3, doc 2, score=0.0
- `control/direct/mmlu_formal_logic/correct`: line 2, doc 1, score=1.0
- `control/direct/mmlu_formal_logic/incorrect`: line 1, doc 0, score=0.0
- `control/direct/bbh_cot_fewshot_logical_deduction_three_objects/correct`: line 1, doc 0, score=1.0
- `control/direct/bbh_cot_fewshot_logical_deduction_three_objects/incorrect`: line 6, doc 5, score=0.0
- `control/direct/mmlu_pro_computer_science/correct`: line 2, doc 1, score=1.0
- `control/direct/mmlu_pro_computer_science/incorrect`: line 1, doc 0, score=0.0
- `control/instruction/gsm8k/correct`: line 1321, doc 1, score=1.0
- `control/instruction/gsm8k/incorrect`: line 1320, doc 0, score=0.0
- `control/instruction/agieval_logiqa_en/correct`: line 1, doc 0, score=1.0
- `control/instruction/agieval_logiqa_en/incorrect`: line 3, doc 2, score=0.0
- `control/instruction/mmlu_formal_logic/correct`: line 4, doc 3, score=1.0
- `control/instruction/mmlu_formal_logic/incorrect`: line 1, doc 0, score=0.0
- `control/instruction/bbh_cot_fewshot_logical_deduction_three_objects/incorrect`: line 1, doc 0, score=0.0
- `control/instruction/mmlu_pro_computer_science/correct`: line 1, doc 0, score=1.0
- `control/instruction/mmlu_pro_computer_science/incorrect`: line 3, doc 2, score=0.0
- `logic/direct/gsm8k/correct`: line 1320, doc 0, score=1.0
- `logic/direct/gsm8k/incorrect`: line 1325, doc 5, score=0.0
- `logic/direct/agieval_logiqa_en/correct`: line 1, doc 0, score=1.0
- `logic/direct/agieval_logiqa_en/incorrect`: line 3, doc 2, score=0.0
- `logic/direct/mmlu_formal_logic/correct`: line 2, doc 1, score=1.0
- `logic/direct/mmlu_formal_logic/incorrect`: line 1, doc 0, score=0.0
- `logic/direct/bbh_cot_fewshot_logical_deduction_three_objects/correct`: line 1, doc 0, score=1.0
- `logic/direct/bbh_cot_fewshot_logical_deduction_three_objects/incorrect`: line 6, doc 5, score=0.0
- `logic/direct/mmlu_pro_computer_science/correct`: line 1, doc 0, score=1.0
- `logic/direct/mmlu_pro_computer_science/incorrect`: line 3, doc 2, score=0.0
- `logic/instruction/gsm8k/correct`: line 1320, doc 0, score=1.0
- `logic/instruction/gsm8k/incorrect`: line 1321, doc 1, score=0.0
- `logic/instruction/agieval_logiqa_en/correct`: line 1, doc 0, score=1.0
- `logic/instruction/agieval_logiqa_en/incorrect`: line 3, doc 2, score=0.0
- `logic/instruction/mmlu_formal_logic/correct`: line 4, doc 3, score=1.0
- `logic/instruction/mmlu_formal_logic/incorrect`: line 1, doc 0, score=0.0
- `logic/instruction/bbh_cot_fewshot_logical_deduction_three_objects/incorrect`: line 1, doc 0, score=0.0
- `logic/instruction/mmlu_pro_computer_science/correct`: line 1, doc 0, score=1.0
- `logic/instruction/mmlu_pro_computer_science/incorrect`: line 3, doc 2, score=0.0
- `nl_exact/direct/gsm8k/correct`: line 1320, doc 0, score=1.0
- `nl_exact/direct/gsm8k/incorrect`: line 1324, doc 4, score=0.0
- `nl_exact/direct/agieval_logiqa_en/correct`: line 1, doc 0, score=1.0
- `nl_exact/direct/agieval_logiqa_en/incorrect`: line 3, doc 2, score=0.0
- `nl_exact/direct/mmlu_formal_logic/correct`: line 2, doc 1, score=1.0
- `nl_exact/direct/mmlu_formal_logic/incorrect`: line 1, doc 0, score=0.0
- `nl_exact/direct/bbh_cot_fewshot_logical_deduction_three_objects/correct`: line 1, doc 0, score=1.0
- `nl_exact/direct/bbh_cot_fewshot_logical_deduction_three_objects/incorrect`: line 6, doc 5, score=0.0
- `nl_exact/direct/mmlu_pro_computer_science/correct`: line 2, doc 1, score=1.0
- `nl_exact/direct/mmlu_pro_computer_science/incorrect`: line 1, doc 0, score=0.0
- `nl_exact/instruction/gsm8k/correct`: line 1320, doc 0, score=1.0
- `nl_exact/instruction/gsm8k/incorrect`: line 1325, doc 5, score=0.0
- `nl_exact/instruction/agieval_logiqa_en/correct`: line 1, doc 0, score=1.0
- `nl_exact/instruction/agieval_logiqa_en/incorrect`: line 3, doc 2, score=0.0
- `nl_exact/instruction/mmlu_formal_logic/correct`: line 4, doc 3, score=1.0
- `nl_exact/instruction/mmlu_formal_logic/incorrect`: line 1, doc 0, score=0.0
- `nl_exact/instruction/bbh_cot_fewshot_logical_deduction_three_objects/incorrect`: line 1, doc 0, score=0.0
- `nl_exact/instruction/mmlu_pro_computer_science/correct`: line 1, doc 0, score=1.0
- `nl_exact/instruction/mmlu_pro_computer_science/incorrect`: line 3, doc 2, score=0.0
