# TL midtraining eval (2026-10-09)

Held-out generator prompts `datasets/tl_mid_20261009/tl_eval.jsonl`; base_ext was run on the first 200.

| model | mode | n | valid | correct | valid∧correct | dead end | mean tokens | first-200 valid∧correct |
|---|---|---|---|---|---|---|---|---|
| base_ext | free | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 1466 | 0.000 |
| base_ext | guided | 200 | 0.000 | 0.000 | 0.000 | 0.000 | 2048 | 0.000 |
| mid (guide 11:43) | free | 1000 | 0.975 | 0.987 | 0.975 | 0.000 | 213 | 0.990 |
| mid (guide 11:43) | guided | 1000 | 0.985 | 0.985 | 0.985 | 0.000 | 242 | 0.995 |
| mid | free | 1000 | 0.975 | 0.987 | 0.975 | 0.000 | 213 | 0.990 |
| mid | guided | 1000 | 0.998 | 0.998 | 0.998 | 0.001 | 215 | 1.000 |

Free-decoding failures (mid): calc (answer wrong): 13, →E (answer right): 4, ∨E (answer right): 4, new (answer right): 2, calc (answer right): 1, =E (answer right): 1

mid free failures by family: {'calc': 8, 'logic': 11, 'K14': 4, 'K3': 2}
mid guided failures by family: {'logic': 1, 'calc': 1}
