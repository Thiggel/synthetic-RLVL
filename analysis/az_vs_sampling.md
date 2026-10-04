# Search vs sampling on the round-1 self-play prompts (e4 student)

370 prompts covered by sampling (n=32) and MCTS (commit); fraction of prompts with >= 1 correct x valid_s2 proof. difficulty = G16@750 teacher at n=16 (zero: never solved).

| bench / difficulty | n | sampling pass@1 | sampling pass@8 | sampling pass@32 | MCTS (commit) |
|---|---:|---:|---:|---:|---:|
| all / mixed | 223 | 0.254 | 0.423 | 0.502 | 0.238 |
| all / zero | 147 | 0.009 | 0.045 | 0.075 | 0.020 |
| gsm8k_train / mixed | 223 | 0.254 | 0.423 | 0.502 | 0.238 |
| gsm8k_train / zero | 147 | 0.009 | 0.045 | 0.075 | 0.020 |

Search sampled tokens per prompt: MCTS (commit) 9237

Zero prompts, head to head (prompts solved by one method only): MCTS (commit): search-only 1, sampling-only 9 of 147

