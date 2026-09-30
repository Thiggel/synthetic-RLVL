# 0.8b-fp32m: format-tagged benchmarks (% of items)

cells: grammatical / valid / correct / in-system (valid proof whose own `ans` is right);
`answerable` restricts to items with a yes/no or numeric reference.


## all

| bench | n | 0% | 25% |
|---|---:|---|---|
| overall | 9524 | 0 / 0 / 0 / 0 | 30 / 6 / 0 / 4 |
| deduction | 2703 | 0 / 0 / 24 / 0 | 80 / 18 / 44 / 15 |
| pw_all | 2500 | 0 / 0 / 24 / 0 | 82 / 19 / 44 / 16 |
| pw_d0 | 500 | 0 / 0 / 16 / 0 | 72 / 17 / 22 / 13 |
| pw_d1 | 500 | 0 / 0 / 22 / 0 | 83 / 20 / 35 / 16 |
| pw_d2 | 500 | 0 / 0 / 26 / 0 | 86 / 24 / 52 / 21 |
| pw_d3 | 500 | 0 / 0 / 29 / 0 | 83 / 21 / 55 / 18 |
| pw_d5 | 500 | 0 / 0 / 29 / 0 | 84 / 12 / 55 / 10 |
| folio | 203 | 0 / 0 / 18 / 0 | 55 / 7 / 41 / 4 |
| bbh | 2700 | 0 / 0 / 9 / 0 | 13 / 1 / 21 / 0 |
| bbh_web_of_lies | 100 | 0 / 0 / 29 / 0 | 94 / 3 / 53 / 0 |
| bbh_formal_fallacies | 100 | 0 / 0 / 18 / 0 | 47 / 3 / 46 / 1 |
| bbh_boolean_expressions | 100 | 0 / 0 / 48 / 0 | 7 / 0 / 55 / 0 |
| bbh_navigate | 100 | 0 / 0 / 12 / 0 | 23 / 0 / 39 / 0 |
| bbh_multistep_arithmetic_two | 100 | 0 / 0 / 84 / 0 | 16 / 0 / 30 / 0 |
| bbh_object_counting | 100 | 0 / 0 / 1 / 0 | 33 / 0 / 26 / 0 |
| gsm8k | 500 | 0 / 0 / 53 / 0 | 62 / 7 / 25 / 4 |
| gpqa_diamond | 198 | 0 / 0 / 21 / 0 | 2 / 0 / 16 / 0 |
| gpqa_quant | 73 | 0 / 0 / 18 / 0 | 0 / 0 / 11 / 0 |
| standard | 3323 | 0 / 0 / 14 / 0 | 11 / 1 / 30 / 1 |
| arc_challenge | 1172 | 0 / 0 / 6 / 0 | 3 / 0 / 44 / 0 |
| logiqa | 651 | 0 / 0 / 10 / 0 | 3 / 0 / 18 / 0 |
| mmlu | 1000 | 0 / 0 / 7 / 0 | 2 / 0 / 24 / 0 |
| multihop | 600 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| hotpotqa | 200 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| 2wikimqa | 200 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| musique | 200 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

## answerable

| bench | n | 0% | 25% |
|---|---:|---|---|
| overall | 3187 | 0 / 0 / 35 / 0 | 67 / 14 / 52 / 13 |
| deduction | 1887 | 0 / 0 / 34 / 0 | 81 / 22 / 63 / 21 |
| pw_all | 1753 | 0 / 0 / 35 / 0 | 83 / 23 / 63 / 22 |
| pw_d0 | 226 | 0 / 0 / 35 / 0 | 72 / 31 / 48 / 29 |
| pw_d1 | 255 | 0 / 0 / 43 / 0 | 86 / 33 / 69 / 32 |
| pw_d2 | 382 | 0 / 0 / 34 / 0 | 87 / 27 / 68 / 27 |
| pw_d3 | 413 | 0 / 0 / 35 / 0 | 83 / 23 / 67 / 22 |
| pw_d5 | 477 | 0 / 0 / 30 / 0 | 83 / 12 / 58 / 11 |
| folio | 134 | 0 / 0 / 27 / 0 | 55 / 8 / 63 / 7 |
| bbh | 800 | 0 / 0 / 26 / 0 | 36 / 1 / 45 / 0 |
| bbh_web_of_lies | 100 | 0 / 0 / 29 / 0 | 94 / 3 / 53 / 0 |
| bbh_formal_fallacies | 100 | 0 / 0 / 18 / 0 | 47 / 3 / 46 / 1 |
| bbh_boolean_expressions | 100 | 0 / 0 / 48 / 0 | 7 / 0 / 55 / 0 |
| bbh_navigate | 100 | 0 / 0 / 12 / 0 | 23 / 0 / 39 / 0 |
| bbh_multistep_arithmetic_two | 100 | 0 / 0 / 84 / 0 | 16 / 0 / 30 / 0 |
| bbh_object_counting | 100 | 0 / 0 / 1 / 0 | 33 / 0 / 26 / 0 |
| gsm8k | 500 | 0 / 0 / 53 / 0 | 62 / 7 / 25 / 4 |
| standard | 500 | 0 / 0 / 53 / 0 | 62 / 7 / 25 / 4 |
