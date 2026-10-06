# Greedy vs test-time MCTS, 300-item gate subset

valid = valid with premise numbers stated (greedy) / valid_s2 (MCTS); cvf = correct x that validity.

| model | decoding | valid | cvf | math cvf | dapo cvf | wordprob cvf | yesno cvf | knowledge cvf |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| e6 init | greedy | 0.250 | 0.043 | 0.000 | 0.017 | 0.167 | 0.033 | 0.000 |
| e6 init | MCTS, probe terminal | 0.693 | 0.103 | 0.017 | 0.017 | 0.150 | 0.333 | 0.000 |
| e6 init | MCTS, gold terminal | 0.677 | 0.113 | 0.017 | 0.017 | 0.167 | 0.367 | 0.000 |
| r6 @12 (AZ ExIt) | greedy | 0.283 | 0.060 | 0.033 | 0.000 | 0.183 | 0.083 | 0.000 |
| r6 @12 (AZ ExIt) | MCTS, probe terminal | 0.640 | 0.107 | 0.067 | 0.017 | 0.183 | 0.267 | 0.000 |
| r6 @12 (AZ ExIt) | MCTS, gold terminal | 0.657 | 0.120 | 0.067 | 0.017 | 0.250 | 0.267 | 0.000 |
| G19 @750 (GRPO) | greedy | 0.600 | 0.190 | 0.050 | 0.017 | 0.317 | 0.567 | 0.000 |
| G19 @750 (GRPO) | MCTS, gold terminal | 0.810 | 0.217 | 0.100 | 0.017 | 0.300 | 0.667 | 0.000 |
