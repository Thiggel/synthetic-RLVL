# W3.5 tl_mid3 vs v1-data runs (0.8B, lr 2e-5, batch 64, 45% replay where noted)

## TL evals (n = 1000 each): valid ∧ correct, free / guided

| model | data | head | v1 free | v1 guided | v2 free | v2 guided | r3 free | r3 guided |
|---|---|---|---|---|---|---|---|---|
| r45 | v1 | full | 0.979 | 0.996 | 0.560 | 0.578 | 0.059 | 0.107 |
| h60 | v1 | rows60 | 0.987 | 0.998 | 0.566 | 0.606 | 0.010 | 0.049 |
| mid3 | v2 | rows60 | 0.971 | 0.994 | 0.959 | 0.992 | 0.049 | 0.113 |

## General ability

| model | Dolci CE/tok | GSM8K (1319) | GSM8K mean tokens | hit cap | TL tokens in answer |
|---|---|---|---|---|---|
| r45 | 0.941 | 0.561 | 263 | 5.5% | 0.1% |
| h60 | 0.980 | 0.525 | 231 | 2.9% | 0.0% |
| mid3 | 0.940 | 0.561 | 205 | 2.6% | 0.0% |

GSM8K mid3 − r45: +0.000 [-0.028, +0.027] (paired bootstrap, 2000 resamples)

GSM8K mid3 − h60: +0.036 [+0.008, +0.064] (paired bootstrap, 2000 resamples)

## r3 (realistic themes, held out): per theme

| model | theme | n | free valid | free correct | guided valid∧correct | guided dead end | guided hit cap |
|---|---|---|---|---|---|---|---|
| r45 | R:count | 129 | 0.000 | 0.264 | 0.023 | 0.264 | 0.504 |
| r45 | R:graph | 117 | 0.000 | 0.598 | 0.137 | 0.051 | 0.761 |
| r45 | R:gsm | 219 | 0.324 | 0.648 | 0.388 | 0.237 | 0.132 |
| r45 | R:mc | 151 | 0.000 | 0.053 | 0.000 | 0.430 | 0.450 |
| r45 | R:prob | 161 | 0.000 | 0.335 | 0.000 | 0.000 | 0.994 |
| r45 | R:rates | 223 | 0.049 | 0.462 | 0.013 | 0.278 | 0.466 |
| h60 | R:count | 129 | 0.023 | 0.326 | 0.163 | 0.543 | 0.171 |
| h60 | R:graph | 117 | 0.000 | 0.462 | 0.120 | 0.427 | 0.410 |
| h60 | R:gsm | 219 | 0.027 | 0.626 | 0.059 | 0.347 | 0.288 |
| h60 | R:mc | 151 | 0.007 | 0.053 | 0.000 | 0.377 | 0.199 |
| h60 | R:prob | 161 | 0.000 | 0.329 | 0.000 | 0.193 | 0.733 |
| h60 | R:rates | 223 | 0.013 | 0.417 | 0.004 | 0.413 | 0.247 |
| mid3 | R:count | 129 | 0.132 | 0.450 | 0.395 | 0.217 | 0.078 |
| mid3 | R:graph | 117 | 0.000 | 0.393 | 0.000 | 0.761 | 0.197 |
| mid3 | R:gsm | 219 | 0.123 | 0.639 | 0.174 | 0.292 | 0.342 |
| mid3 | R:mc | 151 | 0.033 | 0.536 | 0.086 | 0.417 | 0.470 |
| mid3 | R:prob | 161 | 0.000 | 0.410 | 0.006 | 0.280 | 0.621 |
| mid3 | R:rates | 223 | 0.013 | 0.430 | 0.045 | 0.202 | 0.547 |

## r3 guided outcomes

| model | valid ∧ correct | valid, wrong answer | dead end (calc) | hit 2048 cap | invalid |
|---|---|---|---|---|---|
| r45 | 107 | 159 | 219 | 515 | 0 |
| h60 | 49 | 239 | 376 | 336 | 0 |
| mid3 | 113 | 152 | 334 | 401 | 0 |

## r3 free: top check errors (invalid rows)

- r45: new: no such word 405, type mismatch 176, other 101, trailing tokens 100, application of 88, expected 25
- h60: type mismatch 234, trailing tokens 232, new: no such word 201, other 153, application of 109, expected 49
- mid3: type mismatch 277, new: no such word 255, trailing tokens 218, application of 71, calc: not entailed 48, other 41
