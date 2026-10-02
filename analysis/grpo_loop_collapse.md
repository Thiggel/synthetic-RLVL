# GRPO repetition-loop collapse

Trainer metrics averaged over the first and the last 25 logged steps; loop = a non-empty line repeated >= 8 times (digits masked) in a training rollout (scripts/analysis/grpo_loop_collapse.py).

| arm | steps | clipped ratio | entropy | zero-std groups | mean length (tok) | loop share | reward |
|---|---|---|---|---|---|---|---|
| L1 correct (old lib) | 1–1760 | 0.0441 → 0.143 | 0.316 → 0.231 | 0.59 → 0.764 | 437 → 985 | 0.052 → 0.045 | 0.447 → 0.668 |
| L1 cvf (old lib) | 1–1370 | 0.033 → 0.0748 | 0.343 → 0.0294 | 0.731 → 0.935 | 411 → 297 | 0.047 → 0.073 | 0.225 → 0.384 |
| G12 cvf_fmt (G10@500, old lib) | 1–150 | 0.236 → 0.137 | 0.0307 → 0.0396 | 0.898 → 0.89 | 710 → 534 | 0.214 → 0.131 | 0.239 → 0.236 |
| G13 cvf_fmt (le SFT, new lib) | 1–440 | 0.165 → 0.356 | 0.135 → 0.0255 | 0.726 → 0.866 | 687 → 930 | 0.125 → 0.192 | 0.236 → 0.272 |
| G14 = G13 + overlong penalty, unmasked | 1–602 | 0.106 → 0.000469 | 0.28 → 0.38 | 0.415 → 0.872 | 557 → 230 | 0.096 → 0.014 | 0.175 → 0.263 |
| G15 = G14 recipe from e2 SFT | 1–72 | 0.0738 → 0.000937 | 0.122 → 0.0967 | 0.478 → 0.764 | 463 → 261 | 0.063 → 0.010 | 0.243 → 0.343 |
| G16 = G15 + no-proof penalty | 1–102 | 0.0756 → 0.000625 | 0.124 → 0.0816 | 0.464 → 0.8 | 468 → 259 | 0.062 → 0.002 | 0.242 → 0.352 |
