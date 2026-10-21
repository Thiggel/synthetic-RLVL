# W4.1 prep: AZ eval-only baseline of tl_mid4 (fresh value head, n = 200 per cell, reward mul)

| set | decoding | sims | r (valid ∧ correct) | valid | correct | dead end | tokens | decisions | s / problem | sims64 − sims0 |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 | guided | 0 | 1.000 | 1.000 | 1.000 | 0.000 | 201 | 84 | 0.6 | – |
| v1 | guided | 64 | 1.000 | 1.000 | 1.000 | 0.000 | 199 | 83 | 2.6 | +0.000 [+0.000, +0.000] |
| v1 | free | 0 | 0.975 | 0.975 | 0.985 | 0.000 | 203 | 204 | 0.6 | – |
| v1 | free | 64 | 1.000 | 1.000 | 1.000 | 0.000 | 198 | 199 | 16.8 | +0.025 [+0.005, +0.050] |
| r3 | guided | 0 | 0.990 | 0.990 | 0.990 | 0.010 | 254 | 114 | 0.7 | – |
| r3 | guided | 64 | 1.000 | 1.000 | 1.000 | 0.000 | 246 | 111 | 3.7 | +0.010 [+0.000, +0.025] |
| r3 | free | 0 | 0.915 | 0.915 | 0.915 | 0.000 | 254 | 255 | 0.7 | – |
| r3 | free | 64 | 1.000 | 1.000 | 1.000 | 0.000 | 243 | 244 | 26.7 | +0.085 [+0.050, +0.125] |

## Greedy (sims 0) failures and what search does with them

| set | decoding | family | n | sims0 r | sims64 r | sims0 failures: valid wrong / invalid right / invalid wrong / dead end |
|---|---|---|---|---|---|---|
| v1 | free | calc | 20 | 0.850 | 1.000 | 0 / 0 / 3 / 0 |
| v1 | free | chain | 2 | 0.500 | 1.000 | 0 / 1 / 0 / 0 |
| v1 | free | logic | 34 | 0.971 | 1.000 | 0 / 1 / 0 / 0 |
| r3 | guided | R:gsm | 48 | 0.979 | 1.000 | 0 / 0 / 1 / 1 |
| r3 | guided | R:mc | 31 | 0.968 | 1.000 | 0 / 0 / 1 / 1 |
| r3 | free | R:count | 30 | 0.867 | 1.000 | 0 / 0 / 4 / 0 |
| r3 | free | R:gsm | 48 | 0.854 | 1.000 | 0 / 0 / 7 / 0 |
| r3 | free | R:mc | 31 | 0.935 | 1.000 | 0 / 0 / 2 / 0 |
| r3 | free | R:rates | 34 | 0.882 | 1.000 | 0 / 0 / 4 / 0 |
