# GRPO repetition-loop collapse

Trainer metrics averaged over the first and the last 25 logged steps; loop = a non-empty line repeated >= 8 times (digits masked) in a training rollout (scripts/analysis/grpo_loop_collapse.py).

| arm | steps | clipped ratio | entropy | zero-std groups | mean length (tok) | loop share | reward |
|---|---|---|---|---|---|---|---|
| L1 correct (old lib) | 1–2070 | 0.0441 → 0.237 | 0.316 → 0.309 | 0.59 → 0.713 | 437 → 869 | 0.052 → 0.151 | 0.447 → 0.599 |
| L1 cvf (old lib) | 1–1701 | 0.033 → 0.0567 | 0.343 → 0.0227 | 0.731 → 0.93 | 411 → 250 | 0.047 → 0.055 | 0.225 → 0.254 |
| G12 cvf_fmt (G10@500, old lib) | 1–150 | 0.236 → 0.137 | 0.0307 → 0.0396 | 0.898 → 0.89 | 710 → 534 | 0.214 → 0.131 | 0.239 → 0.236 |
| G13 cvf_fmt (le SFT, new lib) | 1–440 | 0.165 → 0.356 | 0.135 → 0.0255 | 0.726 → 0.866 | 687 → 930 | 0.125 → 0.192 | 0.236 → 0.272 |
| G14 = G13 + overlong penalty, unmasked | 1–819 | 0.106 → 0.000781 | 0.28 → 0.174 | 0.415 → 0.889 | 557 → 263 | 0.096 → 0.009 | 0.175 → 0.302 |
| G15 = G14 recipe from e2 SFT | 1–190 | 0.0738 → 0.00125 | 0.122 → 0.0862 | 0.478 → 0.806 | 463 → 263 | 0.063 → 0.008 | 0.243 → 0.314 |
| G16 = G15 + no-proof penalty | 1–346 | 0.0756 → 0.000312 | 0.124 → 0.0619 | 0.464 → 0.846 | 468 → 203 | 0.062 → 0.002 | 0.242 → 0.264 |
