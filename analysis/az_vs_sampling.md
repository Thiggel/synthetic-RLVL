# Search vs sampling on the round-1 self-play prompts (e4 student)

1370 prompts covered by sampling (n=32) and MCTS (commit); fraction of prompts with >= 1 correct x valid_s2 proof. difficulty = G16@750 teacher at n=16 (zero: never solved).

| bench / difficulty | n | sampling pass@1 | sampling pass@8 | sampling pass@32 | MCTS (commit) | MCTS (solve) |
|---|---:|---:|---:|---:|---:|---:|
| all / mixed | 765 | 0.208 | 0.435 | 0.573 | 0.277 | – |
| all / zero | 605 | 0.007 | 0.032 | 0.068 | 0.018 | 0.019 (n=311) |
| dolci_math / mixed | 277 | 0.185 | 0.493 | 0.675 | 0.354 | – |
| dolci_math / zero | 223 | 0.008 | 0.032 | 0.067 | 0.018 | 0.033 (n=122) |
| dolci_wordprob / mixed | 265 | 0.194 | 0.385 | 0.525 | 0.230 | – |
| dolci_wordprob / zero | 235 | 0.005 | 0.023 | 0.064 | 0.017 | 0.016 (n=125) |
| gsm8k_train / mixed | 223 | 0.254 | 0.423 | 0.502 | 0.238 | – |
| gsm8k_train / zero | 147 | 0.009 | 0.045 | 0.075 | 0.020 | 0.000 (n=64) |

Search sampled tokens per prompt: MCTS (commit) 7720, MCTS (solve) 9418

Zero prompts, head to head (prompts solved by one method only): MCTS (commit): search-only 4, sampling-only 34 of 605; MCTS (solve): search-only 3, sampling-only 22 of 311

