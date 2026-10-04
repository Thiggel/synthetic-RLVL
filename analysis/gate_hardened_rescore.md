# Dolci gate (greedy) rescored with the hardened GRPO reward

valid = rlvl strict; valid_prem = valid with premise numbers stated (the cvf reward's validity); cvf = correct · valid_prem. Clean = the 713 gate items without a near-duplicate in the training pool.

| model | step | valid | valid_prem | cvf | correct | clean valid_prem | clean cvf | clean correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| e2 (SFT) | 0 | 0.101 | 0.035 | 0.025 | 0.244 | 0.041 | 0.031 | 0.274 |
| e3 (SFT) | 0 | 0.166 | 0.068 | 0.020 | 0.267 | 0.074 | 0.021 | 0.299 |
| e4 (SFT) | 0 | 0.374 | 0.194 | 0.040 | 0.257 | 0.182 | 0.043 | 0.282 |
| e5 (SFT) | 0 | 0.420 | 0.234 | 0.051 | 0.259 | 0.229 | 0.055 | 0.282 |
| G16 (from e2)@250 | 250 | 0.174 | 0.084 | 0.042 | 0.267 | 0.098 | 0.049 | 0.296 |
| G16 (from e2)@500 | 500 | 0.328 | 0.168 | 0.081 | 0.276 | 0.182 | 0.098 | 0.299 |
| G16 (from e2)@750 | 750 | 0.623 | 0.295 | 0.113 | 0.248 | 0.325 | 0.130 | 0.264 |
| G16 (from e2)@1000 | 1000 | 0.732 | 0.251 | 0.128 | 0.256 | 0.297 | 0.151 | 0.281 |
| G17 (from e3)@250 | 250 | 0.197 | 0.113 | 0.038 | 0.266 | 0.122 | 0.045 | 0.303 |
| G17 (from e3)@500 | 500 | 0.362 | 0.286 | 0.088 | 0.274 | 0.268 | 0.090 | 0.292 |
| G17 (from e3)@750 | 750 | 0.687 | 0.570 | 0.142 | 0.223 | 0.572 | 0.150 | 0.229 |
