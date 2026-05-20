# Old State: RL Validity-Reward Direction - 2026-05-19

Status: archived / not the active research direction.

## What We Tried

The earlier project direction tested whether GRPO posttraining with formal-proof validity reward improves correctness and OOD proof quality.

Reward families included:

- `correct_plus_0p1_format`
- `correct_plus_valid_plus_0p1_format`
- validity scaling sweeps such as `0.25`, `0.5`, `0.75`, `1.0`
- sparse indicator rewards such as `indicator_all`
- citation-free validity rewards
- dense line-validity rewards
- gated validity/correctness rewards

Datasets evolved through hard-v2, hard-v3, hard-v5, hard-FSA, hard-FSA-schema, easy HFSA, and fixed-target HFSA.

## Main Findings

- Early jobs were contaminated by unmerged/raw LoRA checkpoint loading issues; later runs used verified merged SFT checkpoints.
- Citation bookkeeping was a major confound, so citation-free proof validation was added.
- A target bug was found and fixed: earlier HFSA gold traces concluded the final marker atom instead of the queried final state atom.
- On the easy fixed/citation-free HFSA curriculum, GRPO saturated training-depth correctness but did not robustly improve long-depth valid+correct reasoning.
- Validity-gated or validity-shaped rewards were at most slightly better and not strong relative to seed variance.
- Long-depth failures were dominated by invalid or incomplete proof chains, generation-length issues, and step-skipping/post-hoc proof failures.

## Why This Is Archived

The RL experiments did not cleanly show that validity reward teaches a better reasoning algorithm. The strongest positive signal now comes from pure SFT: logic CoT appears to extrapolate better than matched natural-language CoT under length extrapolation, especially in pass@k.

Therefore the active paper direction is now supervised/midtraining comparison of reasoning substrates:

- logic CoT vs deterministic natural-language CoT;
- depth extrapolation on controlled synthetic datasets;
- real-benchmark transfer;
- model-regime and trace-length controls.

## Pointers To Old Detailed Docs

- `docs/posttrain_status_2026-04-18.md`
- `docs/posttrain_recovery_passk_checklist_2026-04-24.md`
- `docs/hard_v3_experiment_status_2026-04-30.md`
- `docs/hard_v5_experiment_status_2026-05-03.md`
- `docs/hfsa_easy_validity_diagnostic_2026-05-14.md`
- `docs/hard_fsa_schema_fixedtarget_2026-05-14.md`
