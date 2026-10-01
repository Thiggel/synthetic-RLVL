# GRPO repetition-loop collapse

Trainer metrics averaged over the first and the last 25 logged steps; loop = a non-empty line repeated >= 8 times (digits masked) in a training rollout (scripts/analysis/grpo_loop_collapse.py).

| arm | steps | clipped ratio | entropy | zero-std groups | mean length (tok) | loop share | reward |
|---|---|---|---|---|---|---|---|
| L1 correct (old lib) | 1–876 | 0.0441 → 0.264 | 0.316 → 0.318 | 0.59 → 0.761 | 437 → 1.02e+03 | 0.052 → 0.082 | 0.447 → 0.635 |
| L1 cvf (old lib) | 1–401 | 0.033 → 0.139 | 0.343 → 0.148 | 0.731 → 0.889 | 411 → 558 | 0.047 → 0.143 | 0.225 → 0.174 |
| G12 cvf_fmt (G10@500, old lib) | 1–147 | 0.236 → 0.141 | 0.0307 → 0.0388 | 0.898 → 0.89 | 710 → 542 | 0.214 → 0.130 | 0.239 → 0.25 |
| G13 cvf_fmt (le SFT, new lib) | 1–106 | 0.165 → 0.154 | 0.135 → 0.105 | 0.726 → 0.801 | 687 → 641 | 0.125 → 0.103 | 0.236 → 0.305 |
