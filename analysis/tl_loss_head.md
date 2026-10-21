# W3.3 loss-head ablation (0.8B, v1 r25 mix, lr 2e-5, batch 64, 1 epoch)

| TL loss head | TL free v∧c | free valid | free dead end | TL guided v∧c | TL CE/tok (full softmax) | Dolci CE/tok | GSM8K (1319) | GSM8K TL leak | GSM8K hit 2048 cap | bench ms/1k tok | bench peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 0.975 | 0.975 | 0.000 | 0.998 | 0.0015 | 0.984 | 0.486 | 0.000 | 0.027 | 224 | 9.9 |
| rows60 | 0.987 | 0.987 | 0.000 | 0.998 | 5.8261 | 0.980 | 0.525 | 0.000 | 0.029 | 222 | 9.9 |
| legal | 0.000 | 0.000 | 0.000 | 0.997 | 6.2431 | 0.981 | 0.506 | 0.000 | 0.168 | 239 | 9.9 |
