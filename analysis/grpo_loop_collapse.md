# GRPO repetition-loop collapse

Trainer metrics averaged over the first and the last 25 logged steps; loop = a non-empty line repeated >= 8 times (digits masked) in a training rollout (scripts/analysis/grpo_loop_collapse.py).

| arm | steps | clipped ratio | entropy | zero-std groups | mean length (tok) | loop share | reward |
|---|---|---|---|---|---|---|---|
| L1 correct (old lib) | 1–1574 | 0.0441 → 0.114 | 0.316 → 0.293 | 0.59 → 0.717 | 437 → 873 | 0.052 → 0.027 | 0.447 → 0.659 |
| L1 cvf (old lib) | 1–1162 | 0.033 → 0.165 | 0.343 → 0.0453 | 0.731 → 0.908 | 411 → 482 | 0.047 → 0.162 | 0.225 → 0.361 |
| G12 cvf_fmt (G10@500, old lib) | 1–150 | 0.236 → 0.137 | 0.0307 → 0.0396 | 0.898 → 0.89 | 710 → 534 | 0.214 → 0.131 | 0.239 → 0.236 |
| G13 cvf_fmt (le SFT, new lib) | 1–437 | 0.165 → 0.359 | 0.135 → 0.0269 | 0.726 → 0.871 | 687 → 937 | 0.125 → 0.194 | 0.236 → 0.263 |
| G14 = G13 + overlong penalty, unmasked | 1–462 | 0.106 → 0.00141 | 0.28 → 0.358 | 0.415 → 0.884 | 557 → 230 | 0.096 → 0.014 | 0.175 → 0.263 |
