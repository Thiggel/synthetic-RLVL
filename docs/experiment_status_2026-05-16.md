# Experiment Status - 2026-05-16

Snapshot time: 2026-05-16 10:26 CEST.

## Executive Summary

- The `hard_fsa_schema_easy500` GRPO experiment completed and is the strongest completed evidence so far.
- On `easy500`, correctness and citation-free validity saturate on train depths `1..5`, but OOD joint correct+valid proof quality collapses at long depths.
- Validity-gated reward gives at most a small depth-10 improvement, mostly in the shortcut split; it does not materially improve depth-15/20 correctness or valid reasoning.
- The fixed-target dataset repair is in progress. It fixes the previous target mismatch where the formal proof concluded a marker while the natural answer was a state.
- The fixed-target GRPO continuation is still running. Three rows are alive; one validity-gated row failed from a Ray worker death and is covered by the queued `afterany` continuation waves.
- The pure SFT logic-vs-natural-language first wave completed all four SFT jobs. Its post-hoc pass@k eval is running now.
- Preliminary SFT online eval says `nl_exact` traces are much easier for OLMo-7B to imitate than formal logic traces, but both collapse beyond training depth. The large pass@k eval is required before making firm claims.

## Completed: Easy HFSA GRPO

Artifacts:

- Metrics table: `analysis/hfsa_easy_validity_2026-05-14/tables/easy500_passk_condition_summary_latest.csv`
- Report: `docs/hfsa_easy_validity_diagnostic_2026-05-14.md`

Mean/std over three seeds:

| train shortcut | reward | train correct@1 | train cf-joint@1 | OOD correct@1 | OOD cf-joint@1 | step10 correct@1 | step10 cf-joint@1 | step20 correct@1 | step20 cf-joint@1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.0` | `correct_plus_0p1_format` | `0.998 +/- 0.000` | `0.998 +/- 0.000` | `0.461 +/- 0.048` | `0.269 +/- 0.025` | `0.702 +/- 0.077` | `0.524 +/- 0.114` | `0.071 +/- 0.025` | `0.002 +/- 0.002` |
| `0.0` | `indicator_correct_and_citation_free_valid_plus_0p1_format` | `0.999 +/- 0.001` | `0.998 +/- 0.001` | `0.462 +/- 0.048` | `0.269 +/- 0.027` | `0.710 +/- 0.071` | `0.528 +/- 0.116` | `0.066 +/- 0.036` | `0.002 +/- 0.001` |
| `0.5` | `correct_plus_0p1_format` | `0.999 +/- 0.001` | `0.998 +/- 0.001` | `0.458 +/- 0.045` | `0.274 +/- 0.026` | `0.707 +/- 0.076` | `0.536 +/- 0.108` | `0.064 +/- 0.027` | `0.004 +/- 0.002` |
| `0.5` | `indicator_correct_and_citation_free_valid_plus_0p1_format` | `0.998 +/- 0.001` | `0.998 +/- 0.001` | `0.463 +/- 0.043` | `0.282 +/- 0.028` | `0.727 +/- 0.064` | `0.555 +/- 0.102` | `0.070 +/- 0.024` | `0.003 +/- 0.001` |

Interpretation:

- The dataset is learnable at train depths.
- Long-depth OOD failure is primarily proof-chain validity, not answer-format failure.
- The validity reward has weak positive signs at depth 10 under shortcut pressure, but no convincing depth-15/20 effect.
- This result is not clean enough for the main claim because the old target mismatch made the formal conclusion not exactly match the answer proposition.

## Live: Fixed-Target GRPO

Dataset:

- HF dataset: `flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget`
- Report: `docs/hard_fsa_schema_fixedtarget_2026-05-14.md`

Current Slurm chain:

- Original GRPO array `3606770` timed out after about 24h, as expected.
- Continuation wave `3608684` is active.
- Further continuation waves `3608685 -> 3608686` are dependency-held with `afterany` and `RESUME_MODE=auto`.
- Merge/pass@k eval `3608687` is dependency-held after `3608686`.

Current row status:

| row | regime | reward | latest observed progress | status |
| --- | --- | --- | ---: | --- |
| `0` | `sft1to3_rl1to10` | `correct_plus_0p1_format` | `454/500` | running |
| `1` | `sft1to3_rl1to10` | `correct_plus_citation_free_valid_plus_0p1_format` | `300/500` checkpoint | failed early in continuation; will resume in queued wave |
| `2` | `sft1to5_rl1to15` | `correct_plus_0p1_format` | `195/500` | running |
| `3` | `sft1to5_rl1to15` | `correct_plus_citation_free_valid_plus_0p1_format` | `193/500` | running |

Failure note:

- Row `1` failed after resuming at `global_step_300`.
- The visible error is a Ray worker death with `Worker unexpectedly exits with a connection error code 2`.
- Slurm job stats showed high but not obviously fatal GPU memory use; no quota issue was visible.
- Because the next waves use `afterany` and `RESUME_MODE=auto`, this row is already queued for another resume attempt.

## Live: Pure SFT Logic vs NL

Report:

- `docs/pure_sft_logic_vs_nl_2026-05-15.md`

SFT jobs:

- SFT array `3612413_[0-3%4]` completed successfully for all four rows.
- Dependent merge + pass@k eval array `3612414_[0-3%4]` is running.

Rows:

| row | template | train depths | SFT steps | SFT status | eval status |
| --- | --- | ---: | ---: | --- | --- |
| `0` | `logic` | `1..10` | `10000` | completed | running, sampled chunks progressing |
| `1` | `nl_exact` | `1..10` | `10000` | completed | running, vLLM initialized; no sampled chunk printed yet |
| `2` | `logic` | `1..15` | `10000` | completed | running, sampled chunks progressing |
| `3` | `nl_exact` | `1..15` | `10000` | completed | running, vLLM initialized; no sampled chunk printed yet |

Preliminary online SFT eval:

| row | template | train depths | correct `1..10` | correct `11..15` | correct `16..25` | translated/formal joint `1..10` | translated/formal joint `11..15` | translated/formal joint `16..25` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `logic` | `1..10` | `0.319` | `0.338` | `0.138` | `0.119` | `0.075` | `0.038` |
| `1` | `nl_exact` | `1..10` | `0.481` | `0.563` | `0.144` | `0.350` | `0.475` | `0.131` |
| `2` | `logic` | `1..15` | `0.175` | `0.113` | `0.081` | `0.113` | `0.075` | `0.038` |
| `3` | `nl_exact` | `1..15` | `0.769` | `0.300` | `0.056` | `0.763` | `0.300` | `0.056` |

Important caveat:

- This online SFT eval uses only 16 prompts per depth and greedy-ish validation. It is a sanity signal, not the main result.
- The running pass@k eval uses 128 prompts per proof length and 16 generations per prompt, and should be the authoritative SFT comparison.

## Analysis So Far

- The strongest completed result is negative/weak for RL validity reward: it changes some training-depth route metrics but does not yet create robust long-chain extrapolation.
- The old target mismatch explains why some previous validity-reward results were hard to interpret.
- After fixing the target, the fixed-target GRPO rows are still too slow to evaluate yet.
- Pure SFT is now testing a simpler question: whether supervised formal traces or controlled natural-language traces extrapolate better without the GRPO noise.
- Early SFT logs suggest OLMo-7B can memorize the natural-language trace format much more readily than the formal trace format under the current LoRA SFT setup.
- If the large pass@k eval confirms this, then formal logic CoT may need either stronger SFT, smaller syntax burden, or a hybrid trace before GRPO can expose a validity-reward advantage.

## Current Risks

- `hfsa_pure_eval` NL rows may be slow because natural-language generations can run long before hitting EOS. If they still show no sampled chunk after more wall time, resubmit those rows with lower `PASSK_BATCH_SIZE` and/or lower `PASSK_MAX_NEW_TOKENS`.
- Fixed-target GRPO row `1` had a transient Ray worker failure. It is already covered by continuation waves, but the retry should be checked after `3608685` starts.
- The `sft1to5_rl1to15` fixed-target rows are slow enough that they likely require multiple continuation waves before final pass@k.

## Recommended Next Steps

1. Wait for `3612414` pass@k outputs before deciding whether formal logic or natural-language CoT is better under pure SFT.
2. If NL eval rows remain silent, cancel and resubmit only rows `1` and `3` with a smaller token budget as a diagnostic.
3. Let fixed-target GRPO continuation waves run; do not interpret the fixed-target RL experiment until pass@k JSONs exist.
4. Add incremental flushing to `scripts/evaluate_checkpoint_passk.py` so long eval jobs do not lose all progress on timeout.
5. If pure SFT still fails ID, run a small controlled ablation before more GRPO: easier branching factor `K=2`, shorter proofs, and/or larger SFT batch/effective steps.
6. If pure SFT gets high ID but weak OOD, compare logic, NL, and hybrid traces on the same fixed-target dataset before returning to expensive RL sweeps.
