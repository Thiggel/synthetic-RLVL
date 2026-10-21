# W3.7 tl_mid5 (25% TL / 75% Dolci) vs mid4 / mid3 (55% TL) (0.8B, lr 2e-5, batch 64, rows60 head)

## TL evals (n = 1000 each): valid ∧ correct, free / guided

| model | data | TL share | v1 free | v1 guided | v2 free | v2 guided | r3 free | r3 guided | v4 free | v4 guided |
|---|---|---|---|---|---|---|---|---|---|---|
| mid3 | v2 | 55% | 0.971 | 0.994 | 0.959 | 0.992 | 0.049 | 0.113 | – | – |
| mid4 | v2 + 25% gen3 | 55% | 0.963 | 0.992 | 0.948 | 0.990 | 0.905 | 0.991 | 0.943 | 0.989 |
| mid5 | v2 + 25% gen3 | 25% | 0.936 | 0.983 | 0.902 | 0.968 | 0.867 | 0.982 | 0.872 | 0.959 |

v1 free mid5 − mid4: -0.027 [-0.042, -0.012] (paired bootstrap, n = 1000)

v1 guided mid5 − mid4: -0.009 [-0.017, -0.001] (paired bootstrap, n = 1000)

v2 free mid5 − mid4: -0.046 [-0.063, -0.029] (paired bootstrap, n = 1000)

v2 guided mid5 − mid4: -0.022 [-0.032, -0.013] (paired bootstrap, n = 1000)

r3 free mid5 − mid4: -0.038 [-0.057, -0.021] (paired bootstrap, n = 1000)

r3 guided mid5 − mid4: -0.009 [-0.017, -0.002] (paired bootstrap, n = 1000)

v4 free mid5 − mid4: -0.071 [-0.090, -0.053] (paired bootstrap, n = 1000)

v4 guided mid5 − mid4: -0.030 [-0.042, -0.019] (paired bootstrap, n = 1000)

## General ability (GSM8K with the <|im_end|> stop)

| model | Dolci CE/tok | GSM8K (1319) | GSM8K mean tokens | hit cap | TL tokens in answer |
|---|---|---|---|---|---|
| mid3 | 0.940 | 0.561 | 195 | 2.1% | 0.0% |
| mid4 | 0.941 | 0.555 | 180 | 1.4% | 0.0% |
| mid5 | 0.931 | 0.563 | 195 | 1.9% | 0.0% |

GSM8K mid5 − mid4: +0.008 [-0.020, +0.035] (paired bootstrap, 2000 resamples)

GSM8K mid5 − mid3: +0.002 [-0.027, +0.029] (paired bootstrap, 2000 resamples)

## r3 per theme

| model | theme | n | free valid | free v∧c | guided v∧c | guided dead end |
|---|---|---|---|---|---|---|
| mid3 | R:count | 129 | 0.132 | 0.132 | 0.395 | 0.217 |
| mid3 | R:graph | 117 | 0.000 | 0.000 | 0.000 | 0.761 |
| mid3 | R:gsm | 219 | 0.123 | 0.114 | 0.174 | 0.292 |
| mid3 | R:mc | 151 | 0.033 | 0.033 | 0.086 | 0.417 |
| mid3 | R:prob | 161 | 0.000 | 0.000 | 0.006 | 0.280 |
| mid3 | R:rates | 223 | 0.013 | 0.009 | 0.045 | 0.202 |
| mid4 | R:count | 129 | 0.907 | 0.907 | 1.000 | 0.000 |
| mid4 | R:graph | 117 | 0.966 | 0.966 | 0.983 | 0.017 |
| mid4 | R:gsm | 219 | 0.799 | 0.799 | 0.991 | 0.009 |
| mid4 | R:mc | 151 | 0.940 | 0.940 | 0.967 | 0.020 |
| mid4 | R:prob | 161 | 1.000 | 1.000 | 1.000 | 0.000 |
| mid4 | R:rates | 223 | 0.883 | 0.883 | 1.000 | 0.000 |
| mid5 | R:count | 129 | 0.829 | 0.829 | 1.000 | 0.000 |
| mid5 | R:graph | 117 | 0.923 | 0.923 | 0.949 | 0.051 |
| mid5 | R:gsm | 219 | 0.795 | 0.795 | 0.973 | 0.027 |
| mid5 | R:mc | 151 | 0.934 | 0.934 | 0.960 | 0.020 |
| mid5 | R:prob | 161 | 0.981 | 0.981 | 1.000 | 0.000 |
| mid5 | R:rates | 223 | 0.803 | 0.803 | 1.000 | 0.000 |

## r3 guided outcomes

| model | valid ∧ correct | valid, wrong answer | dead end (calc) | hit 2048 cap | invalid |
|---|---|---|---|---|---|
| mid3 | 113 | 152 | 334 | 401 | 0 |
| mid4 | 991 | 0 | 7 | 2 | 0 |
| mid5 | 982 | 0 | 15 | 3 | 0 |

## r3 coverage by mid4 training (29514 R: rows; numbers / names masked)

| theme | n | same template in train | all sentences in train | free v∧c, seen template | unseen template (n) |
|---|---|---|---|---|---|
| R:count | 129 | 0.96 | 1.00 | 0.903 | 1.000 (5) |
| R:graph | 117 | 0.98 | 1.00 | 0.974 | 0.500 (2) |
| R:gsm | 219 | 0.17 | 1.00 | 0.947 | 0.768 (181) |
| R:mc | 151 | 0.25 | 0.85 | 0.921 | 0.947 (113) |
| R:prob | 161 | 0.42 | 0.99 | 1.000 | 1.000 (93) |
| R:rates | 223 | 0.81 | 1.00 | 0.917 | 0.738 (42) |
| all | 1000 | 0.56 | 0.98 | 0.938 | 0.862 (436) |

## r3 coverage by mid5 training (8064 R: rows; numbers / names masked)

| theme | n | same template in train | all sentences in train | free v∧c, seen template | unseen template (n) |
|---|---|---|---|---|---|
| R:count | 129 | 0.87 | 1.00 | 0.812 | 0.941 (17) |
| R:graph | 117 | 0.90 | 1.00 | 0.933 | 0.833 (12) |
| R:gsm | 219 | 0.08 | 1.00 | 0.941 | 0.782 (202) |
| R:mc | 151 | 0.11 | 0.77 | 0.941 | 0.933 (134) |
| R:prob | 161 | 0.25 | 0.88 | 1.000 | 0.975 (120) |
| R:rates | 223 | 0.65 | 1.00 | 0.849 | 0.714 (77) |
| all | 1000 | 0.44 | 0.94 | 0.881 | 0.856 (562) |
