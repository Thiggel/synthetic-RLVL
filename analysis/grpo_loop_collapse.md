# GRPO repetition-loop collapse

Trainer metrics averaged over the first and the last 25 logged steps; loop = a non-empty line repeated >= 8 times (digits masked) in a training rollout (scripts/analysis/grpo_loop_collapse.py).

| arm | steps | clipped ratio | entropy | zero-std groups | mean length (tok) | loop share | reward |
|---|---|---|---|---|---|---|---|
| L1 correct (old lib) | 1–1117 | 0.0441 → 0.253 | 0.316 → 0.324 | 0.59 → 0.735 | 437 → 1.01e+03 | 0.052 → 0.013 | 0.447 → 0.64 |
| L1 cvf (old lib) | 1–658 | 0.033 → 0.188 | 0.343 → 0.135 | 0.731 → 0.884 | 411 → 650 | 0.047 → 0.182 | 0.225 → 0.22 |
| G12 cvf_fmt (G10@500, old lib) | 1–150 | 0.236 → 0.137 | 0.0307 → 0.0396 | 0.898 → 0.89 | 710 → 534 | 0.214 → 0.131 | 0.239 → 0.236 |
| G13 cvf_fmt (le SFT, new lib) | 1–222 | 0.165 → 0.156 | 0.135 → 0.0581 | 0.726 → 0.832 | 687 → 657 | 0.125 → 0.108 | 0.236 → 0.302 |
| G14 = G13 + overlong penalty, unmasked | 1–132 | 0.106 → 0.00203 | 0.28 → 0.421 | 0.415 → 0.79 | 557 → 253 | 0.096 → 0.014 | 0.175 → 0.289 |
