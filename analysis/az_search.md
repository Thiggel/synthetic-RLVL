# Stage 3: search decoders vs sampling (e4, 300 clean gate items)

valid_s2 = Stage-2 valid (rlvl strict + hardening + stated premise numbers); cvf = valid_s2 · correct; tokens = sampled tokens per item.

| decoder | valid_s2 | correct | cvf | tokens |
|---|---:|---:|---:|---:|
| greedy | 0.187 | 0.207 | 0.020 | 251 |
| k16 mean | 0.128 | 0.204 | 0.018 | 217 |
| k16 majority | 0.120 | 0.243 | 0.013 | 3471 |
| k16 first-valid | 0.540 | 0.197 | 0.090 | 3471 |
| k16 pass@16 (oracle) | 0.540 | 0.437 | 0.100 | 3471 |
| guided DFS | 0.520 | 0.067 | 0.067 | 2924 |
| guided DFS + fallback | 0.520 | 0.207 | 0.067 | 3102 |
| MCTS none/one | 0.470 | 0.060 | 0.060 | 6357 |
| MCTS none/one + fallback | 0.490 | 0.207 | 0.060 | 6552 |

Per bench: valid_s2 / correct / cvf

| decoder | dapo | knowledge | math | wordprob | yesno |
|---|---|---|---|---|---|
| greedy | 0.23 / 0.08 / 0.02 | 0.22 / 0.00 / 0.00 | 0.27 / 0.07 / 0.00 | 0.13 / 0.38 / 0.07 | 0.08 / 0.50 / 0.02 |
| k16 mean | 0.14 / 0.04 / 0.01 | 0.15 / 0.00 / 0.00 | 0.19 / 0.05 / 0.00 | 0.11 / 0.38 / 0.05 | 0.05 / 0.55 / 0.03 |
| k16 majority | 0.15 / 0.03 / 0.00 | 0.13 / 0.00 / 0.00 | 0.22 / 0.07 / 0.00 | 0.08 / 0.43 / 0.05 | 0.02 / 0.68 / 0.02 |
| k16 first-valid | 0.53 / 0.07 / 0.07 | 0.80 / 0.00 / 0.00 | 0.55 / 0.05 / 0.02 | 0.40 / 0.35 / 0.12 | 0.42 / 0.52 / 0.25 |
| k16 pass@16 (oracle) | 0.53 / 0.27 / 0.07 | 0.80 / 0.05 / 0.00 | 0.55 / 0.27 / 0.02 | 0.40 / 0.67 / 0.15 | 0.42 / 0.93 / 0.27 |
| guided DFS | 0.55 / 0.02 / 0.02 | 0.65 / 0.00 / 0.00 | 0.62 / 0.02 / 0.02 | 0.40 / 0.10 / 0.10 | 0.38 / 0.20 / 0.20 |
| guided DFS + fallback | 0.55 / 0.08 / 0.02 | 0.65 / 0.00 / 0.00 | 0.62 / 0.07 / 0.02 | 0.40 / 0.38 / 0.10 | 0.38 / 0.50 / 0.20 |
| MCTS none/one | 0.53 / 0.00 / 0.00 | 0.72 / 0.00 / 0.00 | 0.50 / 0.02 / 0.02 | 0.30 / 0.12 / 0.12 | 0.30 / 0.17 / 0.17 |
| MCTS none/one + fallback | 0.53 / 0.07 / 0.00 | 0.75 / 0.00 / 0.00 | 0.50 / 0.08 / 0.02 | 0.33 / 0.40 / 0.12 | 0.33 / 0.48 / 0.17 |

- **guided DFS**: found 156/300; correct on found 0.128 vs greedy on the same items 0.128 (greedy on not-found items 0.292)
- **MCTS none/one**: found 141/300; correct on found 0.128 vs greedy on the same items 0.128 (greedy on not-found items 0.277)
