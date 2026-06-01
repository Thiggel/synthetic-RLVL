# HFSA Depth-Scaling SFT Plan - 2026-05-19

## Motivation

The strongest current signal is from the pure-SFT fixed-target HFSA comparison: logic CoT appears to extrapolate better than controlled natural-language CoT when training depth is extended from `1..10` to `1..15`, especially under sampled pass@k and joint correct+valid metrics. That result is still only one seed and only evaluates to depth 25.

This wave tests whether the trend is robust across seeds and over longer reasoning ranges.

This experiment is now part of the broader active plan in:

```bash
docs/formal_logic_cot_research_plan_2026-05-19.md
```

The earlier RL-validity-reward direction is archived as old state in:

```bash
docs/old_rl_validity_reward_direction_2026-05-19.md
```

## Current Dataset

Dataset family: `hard_fsa_schema` in shortcut-neutral fixed-target mode.

HF dataset for this wave:

```bash
flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget-depth50
```

The task is a finite-state automaton proof problem:

- each example has depth `D`;
- the prompt contains natural-language facts and transition rules;
- each step has `K=4` locally plausible branch transitions;
- exactly one branch is reachable from the current state-marker pair;
- wrong branches remain coherent, so shallow branch selection is not trivially detectable;
- the gold proof derives alternating state and marker atoms and stops at the queried final state;
- proof length is `2D + 1` lines;
- proof validity can be checked citation-free by the logic engine;
- the final `<answer>` is the natural state label corresponding to the final proof conclusion.

The current depth-scaling dataset is intentionally shortcut-neutral: `shortcut_rate=0.0` for train and eval. Shortcut-rich SFT remains a planned follow-up, not part of the first depth-scaling grid.

## Materialized Subsets

The materializer now supports explicit train subsets up to depths 20 and 25 in addition to the previous 3/5/10/15 subsets.

Planned subsets:

| subset | depths | rows |
| --- | ---: | ---: |
| `train_fixedtarget_up_to_5_50k` | `1..5` | 50,000 |
| `train_fixedtarget_up_to_10_50k` | `1..10` | 50,000 |
| `train_fixedtarget_up_to_15_50k` | `1..15` | 50,000 |
| `train_fixedtarget_up_to_20_50k` | `1..20` | 50,000 |
| `train_fixedtarget_up_to_25_50k` | `1..25` | 50,000 |
| `val_step_01_1k` ... `val_step_50_1k` | fixed `D` | 1,000 each |

Build script:

```bash
scripts/slurm/jobs/build_materialized_hfsa_fixedtarget_depth50_2026-05-19.slurm
```

## Length Audit

A generated fixed-target sample with the local OLMo tokenizer has the following approximate lengths:

| template | depth | prompt tokens | target tokens | total SFT tokens |
| --- | ---: | ---: | ---: | ---: |
| `logic` | 25 | 2,927 | 1,915 | 4,843 |
| `logic` | 50 | 5,838 | 3,509 | 9,348 |
| `nl_exact` | 25 | 2,927 | 2,766 | 5,694 |
| `nl_exact` | 50 | 5,838 | 5,489 | 11,328 |

Training only uses depths up to 25, so the existing SFT `data.max_length=8192` is sufficient for the primary train sweep. Depth-50 post-hoc eval uses vLLM with `max_new_tokens=6144`. OLMo-3-1025-7B config reports `max_position_embeddings=65536`, so the depth-50 prompt+generation budget is within model context.

## Main 3-Seed Sweep

Script:

```bash
scripts/slurm/sweeps/sft/hfsa_depth_scaling_logic_vs_nl_2026-05-19.slurm
```

Grid:

| axis | values |
| --- | --- |
| trace template | `logic`, `nl_exact` |
| train depth range | `1..5`, `1..10`, `1..15`, `1..20`, `1..25` |
| seed | `3407`, `3408`, `3409` |
| SFT steps | 10,000 |
| effective batch | 1 sequence/update |
| model | `allenai/Olmo-3-1025-7B` + LoRA |
| checkpoint retention in patched script | save every 1000 steps, keep up to 12 checkpoints |

Total: `2 x 5 x 3 = 30` SFT runs.

W&B groups:

```bash
sft_hfsa_depth_scaling/<template>_train1to<max_depth>
```

## Post-Hoc Evaluation

Script:

```bash
scripts/slurm/jobs/posthoc_hfsa_depth_scaling_merge_eval_2026-05-19.slurm
```

Eval protocol:

| setting | value |
| --- | ---: |
| eval depths | `1..50` |
| prompts per depth | 32 |
| sampled generations per prompt | 16 |
| pass@k | `1,2,4,8,16` |
| max new tokens | `4096` for `logic`, `6144` for `nl_exact` |
| vLLM max model length | 16384 |
| constrained line-level eval | disabled by default |

The eval job merges each LoRA checkpoint into a temporary standalone HF model, runs pass@k, writes JSON/JSONL outputs, logs to W&B, and deletes the merged model by default to avoid accumulating hundreds of GB in `tmp/`.

Outputs:

```bash
${WORK}/synthetic-RLVL/passk_eval/hfsa_depth_scaling/
```

Intermediate-checkpoint eval script:

```bash
scripts/slurm/jobs/posthoc_hfsa_depth_scaling_intermediate_eval_2026-05-19.slurm
```

Default checkpoint steps:

```bash
CHECKPOINT_STEPS=1000,3000,10000
```

Dense backfill submitted 2026-05-25:

```bash
3660813_[0,3,6,9,12,15,18,21,24,27%2]
CHECKPOINT_STEPS=1000,2000,3000,4000,5000,6000,7000,8000,9000,10000
```

Oversight correction 2026-05-25 18:51 CEST: `3660813` was malformed because the comma-separated `CHECKPOINT_STEPS` value was passed through Slurm `--export`, so Slurm split it and the job only saw `CHECKPOINT_STEPS=1000`. Rows `0,3,6,9,12` exited `0:0` after skipping existing 1000-step outputs and produced no new dense JSONs; rows `15+` were canceled. Replacement `3661090_[0,3,6,9,12,15,18,21,24,27%2]` was submitted with `CHECKPOINT_STEPS` in the environment and `--export=ALL`, preserving the full `1000,2000,...,10000` list; row `3661090_0` has already reached checkpoint-2000 merge in its log.

Follow-up 2026-05-26 09:22 CEST: `3661090_3` completed the full `logic_train1to10_seed3407` 10-point grid. That row's OOD correct@16 ranges `0.396..0.688`, OOD joint@16 ranges `0.083..0.344`, and depth-50 joint@16 is near-zero across checkpoints except `0.062` at checkpoint 9000. `3661090_0` is still running `logic_train1to5_seed3407` at checkpoint 9000 after writing missing 2k/4k/5k/6k/7k/8k JSONs; remaining dense rows are pending by throttle.

Follow-up 2026-05-26 13:25 CEST: `3661090_0` also completed the full `logic_train1to5_seed3407` 10-point grid. That row's OOD correct@16 ranges `0.161..0.554`, OOD joint@16 ranges `0.045..0.098`, and depth-50 joint@16 is nonzero only at checkpoints 7000 and 9000 (`0.062`). Rows `3661090_6` and `3661090_9` are now running the `logic_train1to15_seed3407` and `logic_train1to20_seed3407` dense grids.

Follow-up 2026-05-26 17:27 CEST: no new dense-eval failures were found. The intermediate output directory now has 50 main checkpoint JSONs. Partial dense rows are available for `logic_train1to15/20/25_seed3407`: current OOD correct@16 ranges are `0.362..0.825`, `0.562..0.906`, and `0.833..0.979`, and OOD joint@16 ranges are `0.075..0.225`, `0.094..0.531`, and `0.438..0.667`. `3661090_6` and `3661090_9` are still running; targeted train-25 job `3664473_0` has written `logic_train1to25` checkpoint-2000 and is evaluating checkpoint-4000, while `3664473_1` is still evaluating `nl_exact_train1to25` checkpoint-2000.

Follow-up 2026-05-26 21:22 CEST: no new dense-eval failures were found. The intermediate output directory now has 59 main checkpoint JSONs. Targeted train-25 job `3664473_0` has written `logic_train1to25` checkpoints `2000/4000/5000` and is evaluating `6000`; `3664473_1` has written `nl_exact_train1to25` checkpoints `2000/4000` and is evaluating `5000`. Rows `3661090_6` and `3661090_9` continue the broader dense backfill for `logic_train1to15/20_seed3407`.

Follow-up 2026-05-27 01:28 CEST: no new dense-eval failures were found. The intermediate output directory now has 68 main checkpoint JSONs. Rows `3661090_6` and `3661090_9` completed the full `logic_train1to15/20_seed3407` 10-point grids. The remaining broad backfill rows are `3661090_15,18` running for `nl_exact_train1to5/10_seed3407` and `3661090_21,24` pending for `nl_exact_train1to15/20_seed3407`. Targeted train-25 `3664473_0,1` is still running; outputs now include `logic_train1to25` checkpoints `2000/4000/5000/6000/7000/8000` and `nl_exact_train1to25` checkpoints `2000/4000/5000/6000`.

Follow-up 2026-05-27 05:25 CEST: no new dense-eval failures were found. The intermediate output directory now has 75 main checkpoint JSONs. All five logic seed-3407 train ranges have full `1000..10000` grids; `nl_exact_train1to5/10` have `1000/2000/3000/4000/10000`; `nl_exact_train1to15/20` retain `1000/3000/10000`; and targeted `nl_exact_train1to25` has `1000/2000/3000/4000/5000/6000/7000/8000/10000`. Broad rows `3661090_15,18` and targeted row `3664473_1` are running; broad rows `3661090_21,24` are pending.

Follow-up 2026-05-27 05:36 CEST: targeted train-25 row `3664473_1` completed exit `0:0`, so the matched `logic_train1to25` and `nl_exact_train1to25` pair now both have full `1000..10000` seed-3407 grids. The intermediate output directory has 76 main checkpoint JSONs; remaining dense work is broad rows `3661090_15,18` running and `3661090_21,24` pending.

Follow-up 2026-05-28 10:01 CEST: dense seed-3407 intermediate eval is complete. The intermediate output directory now has 100 main checkpoint JSONs: logic and `nl_exact` for train ranges `1..5/10/15/20/25`, seed `3407`, each at checkpoints `1000,2000,...,10000`. This is the full grid recoverable from the completed training runs; true 500-step curves cannot be recovered because the SFT jobs saved every 1000 optimizer steps.

The main SFT sweep saved every `1000` optimizer steps, so true 500-step checkpoint curves cannot be recovered from the completed runs. Future reruns should use `train.save_steps=500` and a larger `save_total_limit` if 500-step curves are required.

Intermediate checkpoint eval is intentionally lighter than final eval:

| setting | value |
| --- | ---: |
| eval depths | `1..50` |
| prompts per depth | 16 |
| sampled generations per prompt | 8 |
| pass@k | `1,2,4,8,16` |
| max new tokens | `4096` for `logic`, `6144` for `nl_exact` |
| vLLM max model length | 16384 |

The intermediate eval job merges one LoRA checkpoint at a time into a job-local tmp directory, evaluates it, writes JSON/JSONL outputs under:

```bash
${WORK}/synthetic-RLVL/passk_eval/hfsa_depth_scaling_intermediate_sparse/
```

and deletes the merged model immediately after each checkpoint.

Primary metrics:

- `correct_pass@k`
- `citation_free_valid_pass@k` for logic traces
- `nl_logic_citation_free_valid_pass@k` for NL traces translated back to FOL
- joint correct+valid pass@k
- `valid_given_correct@k`
- depth threshold curves: largest depth with metric above 80%, 50%, and 25%
- area under the depth curve over `1..50`

Metric caveat found on 2026-05-22: current HFSA `grounded_valid` / `citation_free_grounded_valid` numbers are not interpretable for logic traces because the prompt does not expose canonical predicate letters. Generated formal traces can copy the natural facts with a different but semantically equivalent predicate mapping, so exact-symbol grounded validity is near zero even when internal citation-free proof validity and the final answer are correct. Until semantic/canonicalized grounding is implemented, use internal citation-free validity for logic traces and `nl_logic_*` translated validity for NL traces.

## OOD lm-eval Add-On

Implementation added 2026-05-25:

```bash
${HPCVAULT}/.venv_rlvl_posttrain/bin/python scripts/evaluate_lm_eval.py \
  --checkpoint <model_or_path> \
  --suite synthrlvl_ood \
  --output-path <out> \
  --include-path lm_eval_tasks/synthrlvl_ood \
  --confirm-run-unsafe-code
```

The suite contains GSM8K plus context-provided LongBench variants of HotpotQA, 2WikiMultiHopQA, and MuSiQue. Scoring is format-aware: GSM8K extracts explicit `<answer>...</answer>` or answer-marker text before numeric normalization and does not fall back to arbitrary raw-trace numbers, while the multi-hop QA tasks use strict answer extraction before F1/EM so context-copy text does not receive accidental credit. HotpotQA/2Wiki/MuSiQue now report both QA F1 and exact-match/`qa_exact_match`; GSM8K remains numeric exact-match accuracy.

Pilot wiring jobs `3659344` and strict rerun `3659348` completed. The broad submitted evals are:

| job | scope | note |
| --- | --- | --- |
| `3659356_[0-89%4]` | main OLMo-7B, paired pilots, hard attribute, Qwen-7B, Qwen-1.5B, Gemma-4B | one-GPU LoRA merge/eval; missing checkpoints skip cleanly |
| failed `3659357_[0-1%1]`, replacement `3660240_[0-1%1]` | OLMo-32B pilot | full LongBench OOD failed because OLMo-2 32B enforces a 4096-position limit; replacement is GSM8K-only short-context vLLM eval |
| `3659392_[0-5%3]`, replacement `3659488_[0-5%3]`, EM rerun `3659634_[0-5%3]` | seed-3407 tiny Llama scratch-pretraining checkpoints | direct checkpoint eval; original failed at 32k context, replacement succeeded at 8192 context, EM rerun completed `0:0` and populated new LongBench EM fields |
| `3659629_[0-5%3]`, `3659633_[0-5%3]` | seed-3408/3409 tiny Llama scratch-pretraining checkpoints | dependency-pending on new tiny pretraining arrays `3659626`/`3659630`; separate from the larger-model automatic hook |

The larger-model lm-eval Slurm hook `scripts/slurm/jobs/lm_eval_hfsa_depth_scaling_2026-05-19.slurm` now defaults to this OOD suite so future larger training runs can attach the same downstream eval after training. Tiny scratch-pretraining runs are excluded from the automatic downstream hook.

OOD update 2026-05-26 13:29 CEST: broad array `3659356` has completed rows `0..22`, covering all main OLMo-7B logic rows, all `nl_exact_train1to5/10` rows, and two `nl_exact_train1to15` seeds. Three-seed logic means by train depth `1..5/10/15/20/25` are GSM8K EM `0.056/0.079/0.074/0.086/0.056` and mean strict LongBench F1 `0.395/0.391/0.383/0.408/0.404`. Three-seed `nl_exact` means for `1..5/10` are GSM8K EM `0.493/0.479` and mean LongBench F1 `0.171/0.179`; the two completed `nl_exact_train1to15` seeds have mean GSM8K EM `0.253` and mean LongBench F1 `0.267`. This is a partial downstream transfer readout only: shallow NL is much stronger on GSM8K, while logic is stronger on strict context-QA F1 so far.

OOD update 2026-05-26 17:27 CEST: `3659356` now has 40 completed result JSONs: all 30 main OLMo-7B rows, six paired/hard-attribute rows, and four Qwen-7B logic rows. Main OLMo `nl_exact` is now complete; three-seed `nl_exact` GSM8K EM by train depth `1..5/10/15/20/25` is `0.493/0.479/0.326/0.411/0.257`, and mean strict LongBench F1 is `0.171/0.179/0.214/0.185/0.145`. Paired rows continue the same downstream tradeoff: maze `logic/nl_exact` GSM8K EM is `0.114/0.597` but mean LongBench F1 is `0.403/0.179`; hard attribute `logic/nl_exact` GSM8K EM is `0.139/0.197` but mean LongBench F1 is `0.273/0.044`. The first Qwen-7B logic rows are not yet a matched modality comparison: `logic_train1to10` has three-seed GSM8K EM `0.044` and mean LongBench F1 `0.343`; the single completed `logic_train1to20_seed3407` row has GSM8K EM `0.061` and mean LongBench F1 `0.356`.

OOD update 2026-05-26 21:22 CEST: `3659356` now has 57 completed result JSONs: all main OLMo rows, all six paired rows, all 18 Qwen-7B rows, and three Qwen-1.5B logic rows. Three-seed Qwen-7B logic train ranges `1..10/20/25` get GSM8K EM `0.044/0.105/0.084`, mean LongBench F1 `0.343/0.360/0.342`, and mean LongBench EM `0.267/0.283/0.263`; matched Qwen-7B `nl_exact` gets GSM8K EM `0.559/0.532/0.785`, mean LongBench F1 `0.100/0.240/0.269`, and mean LongBench EM `0.079/0.183/0.211`. The first Qwen-1.5B OOD slice, `logic_train1to10`, has GSM8K EM `0.143`, mean LongBench F1 `0.146`, and mean LongBench EM `0.109`; matched Qwen-1.5B NL/Gemma rows are still pending/running.

OOD update 2026-05-27 01:28 CEST: `3659356` now has 72 completed result JSONs: all main OLMo rows, all six paired rows, all 18 Qwen-7B rows, and all 18 Qwen-1.5B rows. Qwen-1.5B logic train ranges `1..10/20/25` get GSM8K EM `0.143/0.063/0.053`, mean LongBench F1 `0.146/0.119/0.184`, and mean LongBench EM `0.109/0.091/0.136`; matched `nl_exact` gets GSM8K EM `0.163/0.258/0.326`, mean LongBench F1 `0.063/0.100/0.080`, and mean LongBench EM `0.047/0.073/0.059`. This keeps the downstream pattern task-dependent, but Qwen-1.5B is weaker than Qwen-7B on both strict context-QA and GSM8K. Gemma OOD rows `72..75` are running and rows `76..89` are pending.

OOD update 2026-05-27 05:25 CEST: `3659356` now has 85 completed result JSONs. Completed Gemma logic rows are very weak on the OOD suite: train ranges `1..10/20/25` get GSM8K EM `0.010/0.013/0.010`, mean LongBench F1 `0.016/0.057/0.009`, and mean LongBench EM `0.012/0.041/0.004`. Completed Gemma `nl_exact_train1to10` gets GSM8K EM `0.114`, LongBench F1 `0.034`, and LongBench EM `0.025`; the single completed `nl_exact_train1to20_seed3407` gets GSM8K EM `0.150` and LongBench F1/EM `0.000`. Rows `85..88` are running and row `89` is pending.

OOD update 2026-05-27 05:39 CEST: `3659356` now has 87 completed result JSONs, with rows `87..89` running. Gemma `nl_exact_train1to20` is now three-seed complete with GSM8K EM `0.069`, mean LongBench F1 `0.077`, and mean LongBench EM `0.058`; the only remaining Gemma OOD slice is `nl_exact_train1to25`.

OOD update 2026-05-27 09:29 CEST: `3659356` completed all 90 result JSONs. Full Gemma OOD remains weak: logic train ranges `1..10/20/25` get GSM8K EM `0.010/0.013/0.010`, mean LongBench F1 `0.016/0.057/0.009`, and mean LongBench EM `0.012/0.041/0.004`; matched `nl_exact` gets GSM8K EM `0.114/0.069/0.179`, mean LongBench F1 `0.034/0.077/0.027`, and mean LongBench EM `0.025/0.058/0.019`.

OOD update 2026-05-27 11:30 CEST: GSM8K scoring was tightened to explicit-answer-only extraction. The report builder now recomputes main OLMo and tiny GSM8K from sample JSONL where available, and `analysis/logic_cot_report_2026-05-25/tables/ood_gsm8k_strict_recompute_from_samples.csv` contains strict recomputes for all 90 broad OOD rows. Main OLMo-7B logic train ranges `1..5/10/15/20/25` have strict GSM8K EM `0.051/0.072/0.070/0.079/0.049`; matched `nl_exact` has `0.491/0.478/0.322/0.409/0.256`. The qualitative split is unchanged: NL is much better on GSM8K arithmetic, while logic is stronger on strict context-provided QA. Paired examples are written to `analysis/logic_cot_report_2026-05-25/ood_generation_examples_olmo7b_train1to25_seed3407.md`. OLMo-7B UltraChat instruction-control SFT `3666639_0` and OOD eval `3666640_0` completed: GSM8K EM `0.755`, Hotpot EM/F1 `0.050/0.343`, 2Wiki `0.005/0.207`, MuSiQue `0.010/0.195`.

OOD prompt update 2026-05-27 13:15 CEST: sample inspection showed that the answer-only LongBench prompt is off the training target manifold and does not test explicit reasoning. Added format-matched CoT suites `synthrlvl_ood_cot_bare` and `synthrlvl_ood_cot_prompted`; submitted pilot `3667055_[0-3%2]` on OLMo-7B train-1-to-25 seed-3407 logic/NL checkpoints with `LM_EVAL_LIMIT=8`. Inspect generations before using these tasks broadly.

Pilot readout 2026-05-27 13:36 CEST: all four rows completed exit `0:0`. Prompted rows improve LongBench answer-tag adherence for NL and produce nonzero tiny-sample GSM8K EM for both modalities, but completed LongBench samples still mostly do not provide explicit multi-hop reasoning traces.

Full bare-format rerun 2026-05-27 14:04 CEST: submitted full `synthrlvl_ood_cot_bare` arrays after sample inspection. `3667168_[0-90%3]` covers the previous 90 broad non-tiny rows plus UltraChat instruction control; `3667167_[0-17%3]` and `3667169_[0-17%3]` cover tiny 20k and 100k checkpoints; `3667166_[0-1%1]` covers OLMo-32B GSM8K only because LongBench is invalid for OLMo-2 32B's 4096-position limit.

Bare-format readout 2026-05-28 10:01 CEST: the 30-row main OLMo-7B slice is complete. Logic train ranges `1..5/10/15/20/25` get GSM8K EM `0.046/0.043/0.064/0.066/0.025` and mean LongBench F1 about `0.404/0.412/0.411/0.416/0.407`; matched `nl_exact` gets GSM8K EM `0.369/0.341/0.287/0.277/0.242` and mean LongBench F1 about `0.254/0.261/0.235/0.263/0.114`. Tiny 20k and tiny 100k bare arrays completed with strict EM/F1 `0.000` across GSM8K and LongBench. OLMo-32B GSM8K-only bare completed for both templates: logic EM `0.2335`, NL EM `0.6755`.

## Runtime Expectation

Prior 10k OLMo-7B LoRA SFT jobs on the fixed-target dataset took roughly 13-15 hours on one A100-80GB for train depths `1..10` and `1..15`. Depth `1..20` and `1..25` should still usually fit the 24h allocation because the max training sequence length remains below 8192 tokens, but they are expected to be slower.

Depth-50 pass@k eval is the higher-risk runtime component. The original `50 x 128 x 16 = 102,400` generations/checkpoint setting was too slow for the 24h window. A first runtime fix reduced final eval to `50 x 32 x 16 = 25,600` sampled generations/checkpoint, plus a greedy pass, and intermediate eval to `50 x 16 x 8 = 6,400` sampled generations/checkpoint.

Second runtime patch applied on 2026-05-22 10:30 CEST for future/resubmitted jobs:

- Final sparse protocol: depths `{1,2,5,10,12,15,18,20,25,30,35,40,45,50}`, `32` prompts/depth, `16` samples/prompt, no separate greedy pass, vLLM stop string `</answer>`, output subdir `passk_eval/hfsa_depth_scaling_sparse/`.
- Intermediate sparse protocol: depths `{1,5,10,15,20,25,30,40,50}`, `16` prompts/depth, `16` samples/prompt, no separate greedy pass, vLLM stop string `</answer>`, output subdir `passk_eval/hfsa_depth_scaling_intermediate_sparse/`.
- The final sparse protocol is `14*32*16 = 7,168` sampled generations/checkpoint versus `25,600` sampled generations/checkpoint in the old current protocol, and skips `1,600` greedy generations. This is about `3.6x` fewer sampled generations and `3.8x` fewer total generations before accounting for stop-string token savings.
- The intermediate sparse protocol is `9*16*16 = 2,304` sampled generations/checkpoint, gives correct@16 over time, and skips `800` greedy generations. This is about `3.1x` fewer total generations than the previous `6,400 sampled + 800 greedy` intermediate protocol, before stop-string token savings.

Eval jobs are separate dependent jobs, and the scripts are idempotent: they skip existing JSON outputs unless `FORCE_PASSK_EVAL=1` is set. Already-running Slurm jobs still use the script snapshot from their submission time; the sparse protocol affects new/resubmitted jobs only.

## Follow-Up Plan

Do not immediately run the full factorial grid. The staged plan is:

1. Main 3-seed LoRA depth-scaling sweep above.
2. Batch-size ablation on representative train depths, not all depths.
3. Shortcut-rich SFT ablation with `shortcut_rate in {0.0, 0.5, 0.8}` after the shortcut-neutral trend is confirmed.
4. Full/non-LoRA or smaller-model pretraining experiments to check whether the LoRA result is an adapter-specific artifact.
5. Later dataset-family expansion: iGSM-style arithmetic graphs, explicit graph/maze traversal, and constraint-satisfaction tasks.

Submitted ablation update 2026-05-25 19:01 CEST:

- Syntax/representation controls are now implemented and submitted: `3661118_[0-17%3]` SFT and dependent `3661119_[0-17%3]` eval cover `terse_nl`, `rule_annotated_nl`, `pseudocode`, `shuffled_logic`, `invalid_logic`, and `shuffled_nl` for train depths `1..25` and seeds `3407..3409`.
- Same target-token budget is submitted as `3661120_[0-5%3]` SFT and `3661121_[0-5%3]` eval. The OLMo tokenizer audit on 512 train-1..25 rows found mean target lengths `1038` for logic and `1454` for `nl_exact`, so the matched-token run uses 10k logic steps and 7140 NL steps.
- Padded-symbol logic length control completed on three seeds as SFT `3672286_[0-2%3]` and eval `3672287_[0-2%3]`. The new `logic_symbol_padded` template preserves the formal proof semantics but expands compact atoms like `Ba` to explicit parser-native calls like `PB(ca)`. A 512-row train-1-to-25 audit gives target/total means `1443/2965` for padded logic vs `1454/2976` for `nl_exact` and `1038/2560` for compact logic, with zero truncation at max length 8192. Three-seed eval is worse than compact logic: OOD correct/joint@16 `0.675/0.206`, depth-50 `0.562/0.094`. Diagnostics point to valid-but-wrong branch tracking rather than a parser bug; the likely cost is that compact one-token atoms such as `Ba` become multi-token calls such as `PB`, `(ca`, `)`.
- Wordified logic length control is complete: SFT `3674875_[0-2%3]` and eval `3674876_[0-2%3]`, with duplicate submission `3674877`/`3674878` canceled immediately. `logic_wordified` keeps constants compact but renders formal predicates with natural state/attribute names such as `Teal(a)`, while proof rules and validation remain formal. A 512-row train-1-to-25 OLMo audit gives target/total means `1470/2991` versus `1454/2975` for `nl_exact`, with zero truncation at 8192. Three-seed eval underperforms compact logic and `nl_exact`: OOD correct/joint@16 `0.508/0.323`, depth-50 correct/joint@16 `0.344/0.094`.
- Shortcut-rich follow-up: initial build `3661122_[0-1%1]` failed in the probe because the old shortcut schema exhausted high-depth state-word/predicate capacity. Dependents `3661123`/`3661124` were canceled. After expanding schema state banks and enabling extended predicate rendering, local probes for shortcut rates `0.5` and `0.8` passed; replacement build `3661135_[0-1%1]` completed. Original SFT rows `3661136_0..2` OOMed under `train.gradient_checkpointing=auto`, so the SFT wrapper now defaults checkpointing to `true`; targeted replacement `3662743_[0-2,6-8%3]` covers failed rows and at-risk pending logic shortcut-0.8 rows. Eval `3661137` was canceled and replaced by `3662744_[0-11%3]`. Shortcut rates `0.5` and `0.8` are complete and report-updated: rate `0.5` gives logic OOD correct/joint@16 `0.906/0.677`, depth-50 `0.833/0.375`, and NL OOD `0.642/0.585`, depth-50 `0.385/0.312`; rate `0.8` gives logic OOD `0.940/0.794`, depth-50 `0.823/0.417`, and NL OOD `0.638/0.565`, depth-50 `0.281/0.146`. Shortcut-rate `0.3` dose-response jobs were submitted on 2026-05-28 as build `3671430`, SFT `3671431_[0-5%3]`, and eval `3671432_[0-5%3]`; SFT is complete, logic eval rows `0..2` are complete and report-ingested with OOD correct/joint@16 `0.892/0.598`, and matched NL rows `3..5` are still running as of 2026-05-30 05:42 CEST.
- Shortcut-kind controls were implemented and submitted on 2026-05-29 as build `3674886_[0-3%2]`, SFT `3674887_[0-23%3]`, and eval `3674888_[0-23%4]`. They keep eval shortcut-neutral and test two different shortcut families at rates `0.5` and `0.8`: `position` makes the gold branch first on shortcut-enabled training examples, while `initial_marker` fixes the gold path's initial marker to `north`. Local probes and tiny materialization smokes passed for both kinds. Build and SFT are complete after replacement row `3682458_22`. As of 2026-06-01 06:35 CEST, eval rows `0..20` are complete and rows `21..23` are running; `21/24` JSONs are report-ingested. Current partials: `position` rate `0.5` logic OOD correct/joint@16 `0.900/0.619`, depth-50 `0.844/0.312`; `position` rate `0.8` logic OOD `0.879/0.650`, depth-50 `0.760/0.323`; matched `nl_exact` rate `0.5` OOD `0.540/0.431`, depth-50 `0.396/0.260`; matched `nl_exact` rate `0.8` OOD `0.513/0.488`, depth-50 `0.396/0.354`; `initial_marker` logic rates `0.5/0.8` are three-seed with OOD `0.883/0.625` and `0.885/0.610`, depth-50 `0.854/0.344` and `0.865/0.344`; `initial_marker` `nl_exact` rate `0.5` is three-seed with OOD correct/translated-joint@16 `0.469/0.421` and depth-50 `0.115/0.094`.
- Hybrid-order full suite: original `3661162_[0-29%4]` SFT and `3661164_[0-29%4]` eval cover `think_formal` and `formal_think` across all main train depths and three seeds. Rows `3661162_0,1` timed out after reaching 10k steps but before final save; originals were canceled and replaced by SFT `3666424_[0-29%4]` with `RESUME_FROM_CHECKPOINT=auto`. Rows `3666424_9..11` and `24..26` then OOMed at train-1-to-20 because auto checkpointing did not trigger until train-1-to-25. The hybrid wrapper now defaults to gradient checkpointing plus expandable CUDA segments; stale eval `3666425` was canceled and replacement SFT/eval `3670782`/`3670783` were submitted on 2026-05-28. Partial eval readout at 2026-06-01 06:35 CEST: `think_formal` is three-seed complete through train-1-to-25, with train-1-to-25 OOD correct/formal-joint/translated-joint@16 `0.573/0.204/0.419` and depth-50 `0.344/0.000/0.135`; `formal_think` train-1-to-5 is three-seed with OOD `0.538/0.120/0.347` and depth-50 `0.323/0.010/0.000`; `formal_think` train-1-to-10 is two-seed with OOD `0.602/0.242/0.363` and depth-50 `0.422/0.000/0.000`. Treat this as partial until the remaining rows finish. Oversight correction 2026-05-30 10:33 CEST: `think_formal` is NL then formal and `formal_think` is formal then NL; `scripts/analysis/build_logic_cot_report.py` now labels those modes correctly and parses `formal_think` outputs.
- Conditioned dual-modality full suite is complete on three seeds for all train ranges. `3661165_[0-14%4]` SFT trains one model per train-depth/seed on separate mode-conditioned logic and NL examples; `3661166_[0-29%4]` eval requests both `conditioned_logic` and `conditioned_nl` from every checkpoint. The report now includes `tables/conditioned_dual_vs_main_by_train_depth.csv` and `figures/ablation_conditioned_dual_vs_main_by_train_depth.pdf`, comparing conditioned logic/NL against the main logic/NL runs at each train level.
- Conditioned dual-modality 50k extension was submitted on 2026-05-29 to measure convergence speed. Because `a100` has a one-day max runtime, it is a resume chain: `3674879` trains to 10k, then `3674880`/`3674881`/`3674882`/`3674883` resume to 20k/30k/40k/50k. Final eval is `3674884_[0-29%4]`; train-1-to-25 checkpoint-curve eval is `3674885_[0-5%3]` over `10000,20000,30000,40000,50000`. As of 2026-06-01 06:35 CEST, 10k, 20k, repaired 30k, and 40k chunks are complete; 50k chunk `3674883` has rows `0..3` complete, rows `4..7` running, and rows `8..14` throttle-pending. Final and checkpoint eval arrays remain dependency-pending with no JSONs.
- Trace-control update 2026-05-30 14:36 CEST: `shuffled_logic` eval rows `3661119_9..11` completed and were added to the report. Three-seed OOD correct/formal-joint@16 is `0.690/0.002`, and depth-50 correct/formal-joint@16 is `0.510/0.000`; raw samples show normal formal wrappers and answer tags but invalid or unparsable higher-depth proof fragments. This is evidence that the control can learn answer patterns while losing valid derivation structure.
- Dedicated Codex oversight for this new ablation wave has `3678051` running and next pass `3679095` begin-time pending, using `scripts/slurm/codex/hfsa_ablation_oversight_2026-05-29.slurm`; the 05:42 CEST pass found no unrecovered severe failures.

## Non-LoRA / Pretraining Follow-Up

The current sweep is LoRA SFT on OLMo-7B. If the formal-CoT scaling trend holds, the next stronger test is to train smaller models from scratch or with full-parameter continuation rather than adapters.

Recommended first non-LoRA wave:

| model size | training mode | purpose |
| ---: | --- | --- |
| 50-200M | from-scratch pretraining on HFSA traces | cheapest clean test of whether formal traces induce an algorithmic prior |
| 300M-1B | from-scratch or continued pretraining | check scaling of the same effect |
| 7B | full fine-tune / non-LoRA continuation if feasible | rule out LoRA bottlenecks on the same base model |

The smaller-model setting should use the same depth curriculum and eval protocol, but with many more tokens and explicit train/eval loss curves because full pretraining is a different regime from 10k-step adapter SFT.

The first implementation is now a single-node pilot, not a production pretraining stack:

```bash
scripts/train_tiny_llama_pretrain.py
scripts/slurm/sweeps/pretrain/hfsa_tiny_llama_scratch_2026-05-24.slurm
scripts/slurm/jobs/posthoc_hfsa_tiny_llama_pretrain_eval_2026-05-24.slurm
```

It uses random-init Llama configs with a Llama3 tokenizer at `50M`, `100M`, and `200M` parameter scale on HFSA train depths `1..10`. The user-requested "50B/100M/200M" wave was interpreted as the earlier planned `50M-200M` scale; a true 50B from-scratch run needs a separate distributed pretraining stack.

Tiny result update 2026-05-25: all six seed-3407 tiny eval rows completed. The trainer learned some train-band answer/format behavior but not valid extrapolating reasoning. Train correct@8 ranges from `0.656` to `0.859`; OOD correct@8 ranges from `0.016` to `0.273`; depth-50 correct@8 is `0.000` for every row; OOD joint@8 and depth-50 joint@8 are also `0.000` for every row. The best answer-only row is 200M logic (`0.859` train correct@8, `0.273` OOD correct@8), but this remains a smoke/mechanism signal rather than a solved small-model result.

Tiny seed update 2026-05-25 18:22 CEST: the missing tiny seeds `3408` and `3409` completed, and their dependent final sparse eval, checkpoint eval, and OOD lm-eval arrays also completed. The three-seed tiny summary keeps the same qualitative result as seed `3407`: logic has higher OOD answer pass@8 than matched `nl_exact` at every size (`0.141/0.036` at 50M, `0.201/0.034` at 100M, `0.240/0.047` at 200M), but OOD joint@8 and depth-50 correct/joint@8 remain `0.000` for every size/template.

Report/plot update 2026-05-30 05:42 CEST: `scripts/analysis/build_logic_cot_report.py` now builds a LaTeX report plus CSV tables and PDF/PNG plots under `analysis/logic_cot_report_2026-05-25/`. It covers main OLMo-7B final plots, @16-only OLMo checkpoint curves split by matched logic/NL train-depth pair, OLMo train-1-to-25 depth-band checkpoint curves, tiny Llama 20k and 100k tables plus depth-band checkpoint curves, bare-format OOD tables, OLMo-32B GSM8K bare results, token-length audit table, partial/complete ablation tables, ablation comparison plots for same-target-token, completed logic length-control including wordified, shortcut-rate, and conditioned-dual controls, architecture comparison tables/figures for OLMo-7B, Qwen-2.5-1.5B, Qwen-2.5-7B, Gemma-3-4B, and OLMo-2-32B short-context, and direct bare-format OOD sample generations. It writes length-control artifacts `tables/logic_length_control_token_match.csv`, `tables/logic_length_control_eval_vs_main.csv`, `tables/logic_wordified_eval_summary.csv`, `tables/logic_wordified_eval_by_seed.csv`, `tables/logic_length_control_depth_curve_vs_main_train25.csv`, and `figures/ablation_logic_length_control_depth_curve_train1to25.pdf`, plus baseline-inclusive shortcut artifacts `tables/shortcut_rate_ablation_vs_main.csv` and `figures/ablation_shortcut_rate_vs_main.pdf`; the latest refresh added the completed shortcut-rate `0.3` logic rows to those shortcut artifacts. It also writes conditioned-dual artifacts `tables/conditioned_dual_vs_main_by_train_depth.csv` and `figures/ablation_conditioned_dual_vs_main_by_train_depth.pdf`. The builder has empty-result-safe ingestion for queued shortcut-kind, conditioned-dual-50k final, and conditioned-dual-50k checkpoint eval roots; after `3674885` completes it will emit `tables/conditioned_dual_50k_checkpoint_summary.csv` and `figures/ablation_conditioned_dual_50k_convergence_train1to25.pdf`. It writes `tables/same_token_budget_exposure_accounting.csv`, which clarifies that the completed "same-token" run matched target-token exposure (`7140` NL steps), while total prompt-plus-target token matching would be about `8600` NL steps and has not been run. A supplemental raw sequence file without manual truncation is written to `analysis/logic_cot_report_2026-05-25/full_generation_sequences_olmo7b_olmo32b_2026-05-28.md`. Because `pdflatex`/`latexmk` are not installed on the current node, the `.tex` report is generated but not compiled here.

Paired-family expansion update 2026-05-28 15:48 CEST: the iGSM subtraction-validation blocker is fixed by treating `-` as an arithmetic token rather than an identifier character in `logic_engine/parser.py`. The paired train-10 wrappers now include `official_igsm`, and the iGSM train-10 chain was submitted as build `3671601_[2]`, SFT `3671602_[4-5%2]`, and eval `3671603_[4-5%2]`. Local smokes for `official_igsm`, `maze_navigation`, and `attribute_constraints` validated through depth 50 before submission. Build `3671601` completed exit `0:0` with a 50k train set plus 1k validation rows/depth through depth 50 and no validation failures; SFT row `3671602_4` completed and row `3671602_5` is running. The full requested paired-family repeat was then submitted as `3672195 -> 3672212 -> 3672213`, with oversight `3672214`: `official_igsm`, `maze_navigation`, and hard `attribute_constraints`; train ranges `1..5/10/15/20/25`; templates `logic,nl_exact`; seeds `3407/3408/3409`; sparse pass@k eval through depth 50. Initial pending SFT/eval/oversight submissions `3672196`/`3672197`/`3672208` were canceled before start after replacing excessive array-id-based startup sleeps with throttle-slot-based sleeps.

Paired-family status update 2026-06-01 02:50 CEST: the full-suite build remains complete with 55 subsets and no missing paths for all three families. SFT has `90/90` final adapters after row-56 replacement `3683070_56` and replacement hard-attribute rows `3682411_84..89` completed. Replacement eval `3682449_[0-89%4]` has released; rows `0..21` are complete, rows `22..25` are running, and rows `26..89` are pending by array throttle. The eval output directory has `22` `official_igsm` pass@k JSONs and sample JSONLs; `maze_navigation` and hard `attribute_constraints` still have no eval JSONs, so full paired-family analysis remains deferred. Diagnostics-only partial iGSM means: logic train-1-to-5/10/15/20 OOD correct@16 `0.312/0.507/0.546/0.536` and internal-joint@16 `0.255/0.377/0.392/0.245`; matched `nl_exact` train-1-to-5/10/15 OOD correct@16 `0.366/0.589/0.618`, with one train-1-to-20 seed at `0.589`, and NL parse/translated validity still `0.000`. A bounded sample audit found intended wrappers and answer extraction, but deeper samples still show grounding, validity, or answer fragility, so paired NL validity claims remain blocked on translator coverage even after eval completion. Paired oversight `3686267` completed cleanly after the 02:36 refresh, and next pass `3686895` is begin-time pending.

Tiny checkpoint eval update 2026-05-25: the first tiny intermediate eval array `3659405` exposed that HF Trainer checkpoint dirs do not contain tokenizer files. The script now stages each checkpoint with tokenizer metadata copied from the corresponding `final/` directory. Replacement `3659415_[0-11%3]` completed cleanly and produced all 12 checkpoint JSONs for 10k and 15k; the report builder was rerun and now includes tiny curves with 10k, 15k, and final/20k points.

Tiny OOD lm-eval update 2026-05-25: original tiny OOD array `3659392` failed with a vLLM CUDA device-side assert under `max_model_len=32768`. The tiny eval script now defaults to `max_model_len=8192`, `max_num_seqs=8`, and smaller GPU memory utilization. Replacement/rerun arrays completed for all three seeds and the report now includes `tables/tiny_llama_ood_lmeval_summary.csv` plus by-seed rows. Three-seed GSM8K EM is near zero: 50M logic/NL `0.0025/0.0000`, 100M logic/NL `0.0010/0.0000`, and 200M logic/NL `0.0056/0.0020`; strict LongBench QA F1/EM is zero for every tiny size/template. Because contexts are truncated to 8192 for these tiny configs, use this as a downstream smoke readout, not a long-context QA claim.

Tiny 100k extension update 2026-05-28 10:01 CEST: all three seeds completed for 50M/100M/200M logic and NL. Final sparse pass@8 remains weak for strict extrapolation: 100k logic OOD correct@8 is `0.112/0.133/0.201` for `50M/100M/200M`, and NL is `0.068/0.258/0.055`; depth-50 joint@8 is `0.000` for every size/template, and only 200M logic has nonzero OOD joint@8 (`0.008`). The 100k checkpoint eval grid is complete with 90 JSONs at `20000,40000,60000,80000,100000`.

## Follow-Up Oversight Notes - 2026-05-24

Qwen 7B logic sparse eval now covers all three seeds for train range `1..20`: mean OOD correct@16/joint@16 is `0.753/0.165`, and mean depth-50 correct@16/joint@16 is `0.656/0.021`. The completed Qwen `logic_train1to10` mean remains `0.618/0.320` OOD correct@16/joint@16, so the current partial Qwen curve is not a monotonic joint-validity replication of the OLMo-7B main result. Wait for `logic_train1to25` and matched `nl_exact` rows before drawing model-family conclusions.

OLMo-32B pilot update: `3656335_[0-1]` completed, but full sparse depth-50 eval is not valid for `allenai/OLMo-2-0325-32B`. Original eval row `3656336_0` failed vLLM config validation at `max_model_len=16384`; forcing the override in replacement `3658461_[0-1]` allowed startup but failed during generation with CUDA position-index asserts once sequences exceeded the model's real 4096-position table. `scripts/slurm/jobs/posthoc_hfsa_model_ablation_olmo32_eval_2026-05-24.slurm` now defaults to a separate short-context slice through depths `{1,2,5,10,12,15}` with `vllm_max_model_len=4096`, `max_new_tokens=2048`, and output subdir `passk_eval/hfsa_model_ablation_olmo2_32b_shortctx_sparse/`. Replacement `3660238_[0-1%1]` completed; both templates are saturated on train/hard-tail correct@16 and format-matched joint@16 in this short slice. OLMo-32B OOD was narrowed to GSM8K-only replacement `3660240_[0-1%1]`, which completed with EM `0.197` for logic and `0.683` for `nl_exact`.

Qwen 7B update 2026-05-25 18:22 CEST: Qwen SFT `3656217_[0-17]` and sparse eval `3656218_[0-17]` completed. Three-seed means for logic train ranges `1..10`, `1..20`, and `1..25` are OOD correct@16/joint@16 `0.618/0.320`, `0.753/0.165`, and `0.906/0.431`; depth-50 correct@16/joint@16 `0.292/0.031`, `0.656/0.021`, and `0.854/0.156`. Matched `nl_exact` means for `1..10`, `1..20`, and `1..25` are OOD correct@16/joint@16 `0.461/0.279`, `0.438/0.339`, and `0.569/0.565`; depth-50 correct@16/joint@16 `0.427/0.000`, `0.333/0.135`, and `0.250/0.229`. This is not a clean monotonic replication of the main OLMo-7B claim: Qwen logic wins answer correctness and depth-50 correctness at deeper train ranges, but `nl_exact_train1to25` has higher joint validity.

Follow-up oversight 2026-05-25 02:44 CEST: Qwen 7B logic sparse eval has all nine expected JSON outputs. Three-seed means for train ranges `1..10`, `1..20`, and `1..25` are OOD correct@16/joint@16 `0.618/0.320`, `0.753/0.165`, and `0.906/0.431`; depth-50 correct@16/joint@16 `0.292/0.031`, `0.656/0.021`, and `0.854/0.156`. This makes Qwen logic correctness improve with train depth, but joint validity is still weaker than the main OLMo-7B `logic_train1to25` result and matched Qwen `nl_exact` rows are not finished.

Follow-up oversight 2026-05-25 06:45 CEST: Qwen 7B `nl_exact_train1to10` sparse eval now has all three seeds. Three-seed means are OOD correct@1/correct@16/joint@16 `0.317/0.461/0.279`, depth-50 correct@16/joint@16 `0.427/0.000`, and OOD joint AUC `0.172`. The matched Qwen `logic_train1to10` means remain OOD correct@1/correct@16/joint@16 `0.249/0.618/0.320`, depth-50 correct@16/joint@16 `0.292/0.031`, and OOD joint AUC `0.215`. Treat this as a partial Qwen architecture signal only; `nl_exact_train1to20/25` eval rows are still running or dependency-pending.

Follow-up oversight 2026-05-25 10:48 CEST: Qwen 7B `nl_exact_train1to20` sparse eval now has seeds `3407` and `3408`. The partial two-seed mean is OOD correct@1/correct@16/joint@16 `0.361/0.576/0.503` and depth-50 correct@16/joint@16 `0.406/0.203`. This is a strong partial NL result relative to Qwen `logic_train1to20` joint validity, but seed `3409` and all `nl_exact_train1to25` rows are still incomplete. Paired maze eval row `3657739_0` failed at chunk `51/56` because a depth-45 prompt had `16400` tokens under `vllm_max_model_len=16384`; row `3657739_1` was canceled before the same expected failure. The paired eval wrapper now defaults `maze_navigation` to `PASSK_VLLM_MAX_MODEL_LEN=32768` and batch `64`; replacement eval `3659556_[0-1%2]` is running.

Follow-up oversight 2026-05-25 18:51 CEST: small-extra SFT `3656323` has rows `0..31` completed or recovered, with rows `32,33` actively training and rows `34,35` still inside the intentional startup stagger. Dependent sparse eval `3656389` and broad OOD eval `3659356` remain dependency-pending on the tail rows. Dense seed-3407 intermediate backfill `3660813` was canceled after detecting the malformed comma-list export described above; replacement `3661090` is running/pending. Active log scans found no Traceback, CUDA OOM, quota/no-space, dependency failure, tokenizer/model-load error, vLLM failure, or new node failure in the non-malformed active jobs.

Follow-up oversight 2026-05-26 02:51 CEST: small-extra SFT `3656323` and retries are complete, so sparse eval `3656389` and broad OOD eval `3659356` are priority-pending rather than dependency-pending. Maze replacement eval `3659556_[0-1]` completed: logic train/OOD/depth-50 correct@16 is `0.750/0.003/0.000` with joint `0.750/0.000/0.000`; `nl_exact` train/OOD/depth-50 correct@16 is `1.000/0.250/0.000`, but NL-to-FOL parse/joint remains `0.000`. Shortcut original rows `3661136_0..2` failed with CUDA OOM, rows `6..8` were canceled before starting, the SFT wrapper now defaults gradient checkpointing to `true`, replacement SFT `3662743_[0-2,6-8%3]` was submitted, and eval `3661137` was replaced by `3662744`.

Follow-up oversight 2026-05-26 09:22 CEST: no new unrecovered severe failures were found in active logs. OLMo-32B short-context eval `3660238_[0-1]` completed and both templates are saturated on the bounded depth-15 slice. Dense backfill `3661090_3` completed `logic_train1to10_seed3407`; `3661090_0` is still writing the `logic_train1to5_seed3407` dense grid. Tiny 100k rows `0..2` completed for seeds 3407/3408, seed-3409 row 2 is running, trace/token-budget/shortcut/hybrid rows show recent progress, and the next oversight pass is `3663541`.

Follow-up oversight 2026-05-26 13:29 CEST: no new unrecovered severe failures were found in the monitored logs. Dense backfill row `3661090_0` completed; rows `6` and `9` are running. Same-target-token rows `3661120_0..2`, trace-control rows `3661118_0,2`, and conditioned-dual rows `3661165_0..4` are complete; trace row `1`, shortcut original rows `4,5`, shortcut replacements `3662743_0..2`, hybrid rows `0..3`, tiny 100k rows `3..5` for all seeds, and broad OOD rows `23..26` are running. The next oversight pass is `3664182`, scheduled for 2026-05-26 17:16 CEST.

Follow-up oversight 2026-05-26 17:27 CEST: no new unrecovered severe failures were found in active logs. Broad OOD `3659356` has 40 completed result JSONs with rows `40..43` running; dense checkpoint eval still has 50 JSONs with `3661090_6,9` and `3664473_0,1` running. Trace-control rows `0..2`, same-target-token rows `0..2`, and conditioned-dual rows `0..5` are complete. Active trace rows `3..5`, token-budget rows `3..5`, shortcut original rows `4,5`, shortcut replacements `0..2`, hybrid rows `0..3`, and tiny 100k rows `3..5` show progress without Traceback/OOM/quota/no-space/tokenizer/vLLM failures. Oversight `3663541` completed, `3664182` is running, and next pass `3664671` is begin-time pending.

Follow-up oversight 2026-05-26 21:27 CEST: no new unrecovered severe failures were found in fresh active logs. Small-extra eval `3656389` has seven Qwen-1.5B logic JSONs with rows `7..10` running; broad OOD `3659356` has 57 result JSONs with rows `57..59` running; dense checkpoint eval has 59 JSONs with `3661090_6,9` and `3664473_0,1` running. Tiny 100k rows `0..3` are complete for all three seeds and rows `4,5` are running. Conditioned-dual rows `0..10` are complete, rows `11..13` are running, and row `14` is pending. Original shortcut row `5` completed; original rows `4,9,10` and replacement rows `0..2` are running. Oversight `3664182` completed, current pass `3664671` is running, and next pass `3665088` is begin-time pending.

Follow-up oversight 2026-05-27 01:28 CEST: a recoverable Gemma eval config failure was found. Original small-extra eval row `3656389_18` failed on node `a0533` because vLLM/Gemma3 could not load image-processor metadata from the merged checkpoint (`preprocessor_config.json` missing). `scripts/merge_lora_checkpoint.py` now best-effort saves `AutoProcessor` metadata during merge, `scripts/slurm/jobs/posthoc_hfsa_model_ablation_small_extra_eval_2026-05-24.slurm` now guards `PASSK_JITTER_SECONDS=0`, and checks passed: `python -m py_compile scripts/merge_lora_checkpoint.py`, `bash -n scripts/slurm/jobs/posthoc_hfsa_model_ablation_small_extra_eval_2026-05-24.slurm`, plus a Gemma3 `AutoProcessor` smoke save. Stale original Gemma eval rows `19..35` were canceled and replacement `3665578_[18-35%4]` was submitted with `FORCE_SFT_MERGE=1`; rows `18..21` are running and have passed the previous processor load point. Qwen-1.5B pass@k now has 16 JSONs; broad OOD has 72 JSONs; dense checkpoint eval has 68 JSONs. Oversight `3665088` is running and next pass `3665575` is begin-time pending.

Follow-up oversight 2026-05-27 05:25 CEST: no new unrecovered severe failures were found in the monitored logs. The severe scan only found the already-recovered shortcut OOM rows plus benign allocator warnings. Qwen-1.5B pass@k is complete at 18 JSONs; Gemma replacement `3665578` has completed rows `18..25`, is running rows `26..29`, and has rows `30..35` pending. Broad OOD has 85 JSONs; dense checkpoint eval has 75 JSONs. Tiny 100k pretraining and its final/checkpoint/OOD eval arrays all completed `0:0`; results remain weak for strict extrapolation. Oversight `3665575` is running and next pass `3666214` is begin-time pending.

Follow-up oversight 2026-05-27 05:36 CEST: targeted dense train-25 eval `3664473_1` completed while this pass was updating docs, raising the dense checkpoint output count to 76 JSONs and completing the matched train-25 dense pair. No replacement or dependency edit was needed.

Follow-up oversight 2026-05-27 05:42 CEST: shortcut replacement SFT `3662743_1` completed while the handoff was being finalized. Replacement shortcut rows `0..2` are now complete, rows `6..8` are running, and original shortcut rows `9..11` are still running; no new shortcut failure was found.

Follow-up oversight 2026-05-27 09:29 CEST: hybrid SFT rows `3661162_0,1` timed out after reaching 10k steps but before final save, and rows `2..5` were likely on the same 24h trajectory. The timeout was treated as recoverable rather than a scientific failure: `train_sft.py` now supports `train.resume_from_checkpoint=auto`, the hybrid wrapper defaults online eval past `max_steps`, focused checks passed, original hybrid SFT/eval `3661162`/`3661164` were canceled, and replacements `3666424`/`3666425` were submitted. Broad OOD `3659356` completed all 90 JSONs; Gemma pass@k replacement has 15/18 JSONs with rows `33..35` running; dense checkpoint eval has 80 JSONs. No other unrecovered severe failures were found in monitored active logs.

Follow-up oversight 2026-05-27 09:37 CEST: final sanity check found same-target-token eval rows `3661121_0..2` completed exit `0:0`, with rows `3..5` running. Conditioned-dual eval row `3661166_3` completed exit `0:0`, with rows `0..2,4` running and rows `5..29` pending. Hybrid replacement rows `3666424_0..2` have logged resume from `checkpoint-9000`; row `3` is running, and remaining replacement rows are pending by throttle. The severe log scan only found known recovered shortcut/hybrid failures and benign tokenizer/vLLM/allocator warnings.

## Submitted / Replacement Jobs

Final replacement job set as of 2026-05-23 21:15 CEST:

| stage | Slurm job | dependency | notes |
| --- | ---: | --- | --- |
| 10k SFT rows `0..6` | `3646736_[0-6]` | none | row 0 skipped due existing final checkpoint; rows 1-6 completed cleanly |
| 10k SFT rows `7..29` | `3647379_[7-29%12]` | none | all later rows completed cleanly; no SFT failures found |
| Old full-grid eval arrays | `3647708`, `3648279`, `3648280`, `3647711`, `3647712` | none | canceled at 2026-05-22 10:47 CEST; partial old outputs retained, but no 10k final JSON completed |
| Sparse final eval, all 30 main 10k rows | `3650951_[0-29%10]` | none | completed; 30/30 JSON files, all tasks exit `0:0` |
| Sparse intermediate eval, seed `3407` subset | `3650952_[0,3,6,9,12,15,18,21,24,27%4]` | none | completed; 30/30 checkpoint JSON files, all tasks exit `0:0` |

## Current Status - 2026-05-23 21:15 CEST

| stage | status | note |
| --- | --- | --- |
| SFT `3646736_0..6`, `3647379_7..29` | completed | all 30 main SFT rows are covered; row 0 skipped due existing checkpoint; all executed rows exit `0:0` |
| Old eval arrays `3647708`, `3648279`, `3648280`, `3647711`, `3647712` | canceled | canceled because they used old full-grid protocol and were still cap-hitting; partial old outputs are not the comparable final result |
| Sparse final eval `3650951` | completed | all 30 main rows finished under the comparable sparse protocol |
| Sparse intermediate eval `3650952` | completed | seed-3407 rows for all train-depth/template groups finished at checkpoints `1000`, `3000`, and `10000` |
| Dense intermediate backfill bad `3660813`, replacement `3661090`; targeted train-25 `3664473` | replacement running/pending | 82 main checkpoint JSONs are present. `3661090_0,3,6,9` completed the full `logic_train1to5/10/15/20_seed3407` dense grids; `3664473_0,1` completed the matched `logic/nl_exact_train1to25_seed3407` dense grids; rows `3661090_15,18` are running and rows `3661090_21,24` are pending |

Operational notes:

- Old greedy eval was a major runtime risk because many completions ran to `max_new_tokens` (`4096` for logic, `6144` for `nl_exact`). A token-length audit indicates these cap hits are degenerate/non-terminating outputs rather than legitimate need for longer answers: logic gold targets are far below `4096` even at depth 50, and logic eval started hitting `4096` already in depth chunk `5..8`; `nl_exact` started hitting `6144` around depths `17..20`, where gold targets are only about `2.2k` tokens.
- Runtime estimate from old logs: old final pass@k rows were trending over the 24h window for several rows. The new sparse protocol completed final rows in roughly `3.3h..8.0h`; the shallow logic rows were the slowest completed rows, while most later logic/NL rows finished in `3.3h..6.3h`.
- Sparse final rows show train-band correct@16 and joint@16 are saturated at `1.000` for every group. The main result is depth-dependent: `nl_exact_train1to5` beats `logic_train1to5` OOD, logic is stronger at train ranges `1..10`, `1..15`, and `1..20`, and `nl_exact_train1to25` catches up/slightly exceeds logic at the deepest train range.
- Sparse seed-3407 intermediate rows now cover all train-depth/template groups at the initial `1000,3000,10000` points. Bad dense backfill job `3660813` produced no new dense JSONs because it only saw `CHECKPOINT_STEPS=1000`; replacement `3661090` is running/pending for the saved `1000,2000,...,10000` grid and targeted job `3664473` filled the matched train-25 pair. Current dense output count is 82 JSONs. Both `logic_train1to25_seed3407` and targeted `nl_exact_train1to25_seed3407` now have the full `1000..10000` grid in the output dir.
- The sparse eval patch is active: vLLM stop string `</answer>` is used with the stop tag retained in output, separate greedy eval is skipped by default, sampled pass@1 provides the @1 curve, and sampled-generation examples are written to JSONL when greedy is skipped.
- Eval stderr repeatedly shows tokenizer/rope warnings (`fix_mistral_regex`, integer rope-scaling fields), but completed rows exit `0:0`; the active merged tokenizer reports `GPT2Tokenizer`. Treat this as nonblocking for the current sparse readout and verify tokenizer round-trip before publication-quality reruns.

Completed sparse-protocol group means:

| template | train depths | OOD correct@1 | OOD correct@16 | OOD joint@16 | OOD joint AUC | depth-50 joint@16 | max depth joint >= 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `logic` | `1..5` | `0.031` | `0.293` | `0.052` | `0.027` | `0.010` | `5` |
| `logic` | `1..10` | `0.248` | `0.727` | `0.362` | `0.265` | `0.010` | `18` |
| `logic` | `1..15` | `0.166` | `0.634` | `0.293` | `0.197` | `0.042` | `20` |
| `logic` | `1..20` | `0.284` | `0.830` | `0.441` | `0.418` | `0.198` | `30` |
| `logic` | `1..25` | `0.513` | `0.921` | `0.710` | `0.711` | `0.417` | `45` |
| `nl_exact` | `1..5` | `0.353` | `0.502` | `0.318` | `0.191` | `0.000` | `18` |
| `nl_exact` | `1..10` | `0.354` | `0.482` | `0.299` | `0.184` | `0.000` | `18` |
| `nl_exact` | `1..15` | `0.191` | `0.322` | `0.137` | `0.044` | `0.000` | `18` |
| `nl_exact` | `1..20` | `0.220` | `0.474` | `0.219` | `0.180` | `0.115` | `25` |
| `nl_exact` | `1..25` | `0.574` | `0.794` | `0.748` | `0.757` | `0.427` | `45` |

Analysis artifacts:

```bash
scripts/analysis/aggregate_hfsa_depth_scaling.py
analysis/hfsa_depth_scaling_2026-05-23/tables/final_group_summary.csv
analysis/hfsa_depth_scaling_2026-05-23/tables/final_group_summary_compact.md
analysis/hfsa_depth_scaling_2026-05-23/figures/final_depth_correct16.png
analysis/hfsa_depth_scaling_2026-05-23/figures/final_depth_joint16.png
analysis/hfsa_depth_scaling_2026-05-23/figures/final_ood_metrics_by_train.png
analysis/hfsa_depth_scaling_2026-05-23/figures/intermediate_seed3407_curves.png
```

Check status:

```bash
squeue -u c107fa12 -o '%.18i %.9P %.34j %.2t %.11M %.6D %R'
sacct -j 3650951,3650952 --format=JobID%30,JobIDRaw,JobName%34,State,Elapsed,ExitCode,Start,End -n -P
for f in logs/hfsa_dscale_eval_3650951_*.out logs/hfsa_dscale_ckpt_eval_3650952_*.out; do [ -f "$f" ] && echo "### $f" && tail -n 20 "$f"; done
```

## Model-Ablation And Pretraining Follow-Up - 2026-05-24

Submitted representative architecture/pretraining follow-ups rather than repeating the full 30-row OLMo grid on every model.

| stage | Slurm job | dependency | notes |
| --- | ---: | --- | --- |
| Qwen SFT | `3656217_[0-17%3]` | none | `Qwen/Qwen2.5-7B`, templates `logic,nl_exact`, train depths `1..10`, `1..20`, `1..25`, seeds `3407..3409` |
| Qwen sparse eval | `3656218_[0-17%3]` | `aftercorr:3656217` | same sparse final protocol as the OLMo readout, output under `passk_eval/hfsa_model_ablation_qwen2p5_7b_sparse/` |
| Qwen-1.5B/Gemma SFT | `3656323_[0-35%4]`, retries `3656359_2` and `3656387_3` | none | completed/recovered; `Qwen/Qwen2.5-1.5B` and `google/gemma-3-4b-pt`, same representative train-depth/template/seed grid; exact `Qwen/Qwen2.5-1B` was not available |
| Qwen-1.5B/Gemma sparse eval | original `3656389_[0-35%4]`, Gemma replacement `3665578_[18-35%4]` | complete | Qwen-1.5B rows `0..17` completed and wrote 18 JSONs. Original Gemma row `18` failed due missing processor metadata in the merged checkpoint; stale original Gemma rows `19..35` were canceled. Replacement `3665578` completed all Gemma rows after the merge fix |
| OLMo-32B SFT pilot | `3656335_[0-1%1]` | none | `allenai/OLMo-2-0325-32B`, train depth `1..20`, seed `3407`, `logic` and `nl_exact`, 4 GPUs |
| OLMo-32B short-context eval | failed `3656336`, failed `3658461`, replacement `3660238_[0-1%1]` | SFT complete | OLMo-2 32B enforces a 4096-position limit, so replacement eval is depths `{1,2,5,10,12,15}` only under `passk_eval/hfsa_model_ablation_olmo2_32b_shortctx_sparse/`; exact OLMo-3 base 32B was not available under the checked IDs |
| tiny Llama scratch pretraining | seed-3407 `3656338_[0-5%3]`, retries `3656360_1`/`3656388_0`; seed-3408 `3659626_[0-5%3]`; seed-3409 `3659630_[0-5%3]` | none | `50M/100M/200M` random-init Llama configs, HFSA `logic` vs `nl_exact`; original seed-3407 rows `0,1` hit cluster `NODE_FAIL`; new seed arrays exclude `a0934` |
| tiny Llama sparse eval | seed-3407 `3656390_[0-5%3]`; seed deps `3659627_[0-5%3]`, `3659631_[0-5%3]` | seed-3407 dependency complete; new deps afterany on `3659626`/`3659630` | lightweight pass@k eval for the scratch-pretrained checkpoints |
| tiny Llama checkpoint eval | seed-3407 replacement `3659415_[0-11%3]`; seed deps `3659628_[0-11%3]`, `3659632_[0-11%3]` | new deps afterany on `3659626`/`3659630` | checkpoint-10000/15000 pass@k curves; script stages tokenizer metadata from final checkpoints |
| tiny Llama 100k scratch extension | train `3660986`, `3660990`, `3660994`; final eval `3660987`, `3660991`, `3660995`; checkpoint eval `3660988`, `3660992`, `3660996`; OOD eval `3660989`, `3660993`, `3660997` | complete | all train, final pass@k, checkpoint pass@k, and OOD lm-eval rows completed exit `0:0`; strict extrapolation remains weak |
| OLMo-7B instruction-control OOD | SFT `3666639_0`; OOD eval `3666640_0` | complete for the initial answer-only OOD suite | UltraChat first-turn SFT with the same OLMo-3-7B LoRA setup and `<question>`/`<answer>` wrapper, but no synthetic reasoning traces. Initial OOD results are complete; the bare-format rerun is included as row `90` of broad array `3667168` and is still pending |
| format-matched OOD CoT pilot | `3667055_[0-3%2]` | completed | OLMo-7B `logic`/`nl_exact` train-1-to-25 seed-3407, bare vs prompted CoT OOD suites, `LM_EVAL_LIMIT=8` |
| full bare OOD rerun, non-tiny | `3667168_[0-90%3]` | running/pending | 66/91 result JSONs complete; the 30-row main OLMo-7B slice is complete and report-updated |
| full bare OOD rerun, tiny | `3667167_[0-17%3]`, `3667169_[0-17%3]` | complete | full `synthrlvl_ood_cot_bare` on tiny 20k and 100k checkpoints, all three seeds; strict EM/F1 remains zero |
| OLMo-32B bare OOD subset | `3667166_[0-1%1]` | complete | GSM8K-only `synthrlvl_gsm8k_cot_bare`; logic EM `0.2335`, NL EM `0.6755`; full LongBench remains invalid under 4096 context |
| trace-control ablations | SFT `3661118_[0-17%3]`; eval `3661119_[0-17%3]`; replacements `3682459_[12,14-17%3]` and `3682460_[5-8%3]` | complete | SFT rows `0..17` completed; eval artifacts are `18/18` after repaired rows. The report now includes all controls, including three-seed `pseudocode`. |
| same target-token budget | SFT `3661120_[0-5%3]`; eval `3661121_[0-5%3]` | complete | all six eval rows complete. Logic OOD correct/joint@16 `0.898/0.335`, depth-50 `0.792/0.125`; token-matched NL OOD `0.554/0.473`, depth-50 `0.344/0.219` |
| shortcut-rate ablation | failed build `3661122`; canceled deps `3661123`/`3661124`; completed replacement build `3661135_[0-1%1]`; original SFT `3661136_[0-11%3]`; replacement SFT `3662743_[0-2,6-8%3]`; replacement eval `3662744_[0-11%3]` | complete for rates `0.5`/`0.8` | both shortcut rates are three-seed complete and report-updated with a baseline-inclusive comparison table/plot |
| shortcut-rate `0.3` extension | build `3671430_[0%1]`; SFT `3671431_[0-5%3]`; eval `3671432_[0-5%3]` | complete | all `0.3` logic and matched NL rows completed and are report-ingested |
| hybrid-order full suite | original SFT `3661162_[0-29%4]`, replacement SFT `3666424_[0-29%4]`, targeted SFT `3670782_[9-11,24-26%3]`; eval `3670783_[0-29%4]` plus replacement `3682461_[13,15-29%4]` | eval running/pending | targeted SFT `3670782` completed exit `0:0`; eval rows `0..17` are now complete via original/replacement rows, while replacement `3682461_18/19/20/21` is running and `22..29` is pending by throttle; three-seed `think_formal` train-1-to-25 and first `formal_think` train-1-to-5 slice are report-ingested |
| conditioned dual-modality full suite | SFT `3661165_[0-14%4]`; eval `3661166_[0-29%4]` | complete | all train levels and seeds completed; all 30 eval rows completed. Report artifacts compare conditioned logic/NL against main logic/NL at each train level |
| wordified length-control logic | SFT `3674875_[0-2%3]`; eval `3674876_[0-2%3]` | complete | all three SFT and eval rows completed; 3 eval JSONs are report-ingested |
| conditioned dual 50k extension | chunks `3674879 -> 3674880 -> 3674881/3682457/3682492 -> 3674882 -> 3674883`; evals `3674884`, `3674885` | running/pending | 10k, 20k, repaired 30k, and 40k chunks complete; 50k `3674883` rows `0..3` are running and rows `4..14` are throttle-pending; final/checkpoint evals remain dependency-pending |
| shortcut-kind controls | build `3674886_[0-3%2]`; SFT `3674887_[0-23%3]`; replacement SFT `3682458_[22%1]`; eval `3674888_[0-23%4]` | eval running/pending | build and SFT complete after replacement row `3682458_22`; eval rows `0..16` are complete, rows `17/18/19/20` are running, and rows `21..23` are throttle-pending. Current `17/24` JSONs are report-ingested. |
| Codex oversight | current `3686268`, next `3686897` | running/pending | autonomous oversight for the 2026-05-29 ablation wave; current pass regenerated/mirrored the report for shortcut-kind `17/24` and paired iGSM `22/30`, with trace-control still `18/18` and hybrid still `18/30`, and made no scheduler edits |

Update 2026-05-30 18:47 CEST: the 2026-05-29 ablation wave is under targeted recovery after several later rows failed or were killed without traceback/OOM/quota/model-load signatures. Replacement arrays are `3682457_[3,6-14%4]` plus `3682492_[5%1]` for conditioned-dual 30k, `3682458_[22%1]` for shortcut-kind SFT, `3682459_[12,14-17%3]` for trace-control eval, `3682460_[5-8%3]` for fixed-translator trace repair, and `3682461_[13,15-29%4]` for hybrid-order eval. Dependencies were rewired for `3674882` and `3674888` to wait on the replacement rows plus the still-running original rows. The report now includes three-seed `think_formal` train-1-to-20 and filters the stale pre-fix `rule_annotated_nl` seed-3409 artifact until the repair row overwrites it.

Update 2026-05-30 22:54 CEST: replacement row `3682492_5` completed cleanly, so `3674882` now waits only on `3682457`. Report-ingested trace-control outputs now include `invalid_logic` seeds `3408/3409`, and hybrid `think_formal` train-1-to-25 remains two-seed partial. The invalid-logic result is surprising but evaluator-sensitive: OOD correct/formal-joint@16 is `0.906/0.544` and depth-50 is `0.734/0.188`, but seed-3409 samples are not grounded-valid despite shallow citation-free validity, and depth-50 sampled rows have zero grounded validity. Hybrid `think_formal` train-1-to-25 is unchanged at OOD correct/formal-joint/translated-joint@16 `0.584/0.188/0.459` and depth-50 `0.344/0.000/0.172`; sample inspection confirmed the intended NL-then-formal surface and weak formal validity. The report was regenerated and mirrored with `64` PDFs and `53` CSVs; local TeX compilation remains unavailable.

Update 2026-05-31 02:35 CEST: trace-control replacements `3682459_12/15` and repairs `3682460_5/6/7` completed, so the report now ingests `15/18` trace-control rows. `invalid_logic` is three-seed with OOD correct/formal-joint@16 `0.892/0.427` and depth-50 `0.750/0.146`, but grounded validity remains zero in inspected samples. Repaired `rule_annotated_nl` is three-seed with OOD correct/translated-joint@16 `0.575/0.485` and depth-50 `0.344/0.146`; samples confirm `[rule: ...]` lines now translate. `pseudocode` is two-seed with OOD correct/translated-joint@16 `0.406/0.334`; samples confirm `step_i: derive "..." using ...` wrappers translate at shallow depths but depth-50 is weak. `shuffled_nl` seed `3407` parses but has translated joint `0.000`, consistent with the intended order negative control. Remaining trace rows are pseudocode seed `3409` and shuffled-NL seeds `3408/3409`. The report was regenerated and mirrored with `64` PDFs and `53` CSVs; local TeX compilation remains unavailable.

Update 2026-05-31 06:35 CEST: trace-control replacements `3682459_16/17` completed, so trace-control artifacts are now `17/18`; only pseudocode seed `3409` (`3682460_8`) remains. `shuffled_nl` is three-seed with OOD correct/translated-joint@16 `0.490/0.000` and depth-50 `0.344/0.000`, with samples confirming parseable but order-invalid NL proof surfaces. Shortcut-kind eval wrote its first `4/24` JSONs: `position` rate `0.5` logic three-seed OOD correct/joint@16 `0.900/0.619`, depth-50 `0.844/0.312`; matched `nl_exact` seed `3407` OOD `0.356/0.300`, depth-50 `0.500/0.438`. Active remaining rows include trace `3682460_8`, shortcut-kind eval `3674888_4..7`, hybrid `3682461_13/15/16/17`, conditioned `3682457_13/14`, and paired SFT/eval rows tracked in `docs/running_experiments.md`. The report was regenerated and mirrored with `64` PDFs and `55` CSVs; local TeX compilation remains unavailable.

Update 2026-05-31 10:45 CEST: trace-control artifacts are complete at `18/18`, with three-seed `pseudocode` OOD correct/translated-joint@16 `0.544/0.479` and depth-50 `0.208/0.104`. Shortcut-kind eval is `7/24`; `position` rate `0.8` logic is now two-seed with OOD correct/joint@16 `0.884/0.647` and depth-50 `0.766/0.328`, while `position` rate `0.5` `nl_exact` is two-seed with OOD `0.459/0.303` and depth-50 `0.422/0.219`. Hybrid `think_formal` train-1-to-25 is now three-seed complete with OOD correct/formal-joint/translated-joint@16 `0.573/0.204/0.419` and depth-50 `0.344/0.000/0.135`. Conditioned 40k row `3674882_2` completed and row `3674882_6` started; no final/checkpoint eval JSONs exist yet. The report was regenerated and mirrored with `64` PDFs and `55` CSVs; local TeX compilation remains unavailable.

Update 2026-05-31 14:38 CEST: shortcut-kind eval is `9/24`; `position` rate `0.8` logic is now three-seed with OOD correct/joint@16 `0.879/0.650` and depth-50 `0.760/0.323`, while `position` rate `0.5` `nl_exact` is three-seed with OOD `0.540/0.431` and depth-50 `0.396/0.260`. Conditioned 40k rows `0..5` and `7` are complete, rows `6/8/9/10` are running, and no final/checkpoint eval JSONs exist yet. New shortcut-kind sample inspection confirmed shortcut-neutral prompts, intended wrappers, normal answer extraction, and fragile deeper validity/grounding. The report was regenerated and mirrored with `64` PDFs and `55` CSVs; local TeX compilation remains unavailable.

Update 2026-05-31 18:35 CEST: shortcut-kind eval is `13/24`; `position` rate `0.8` `nl_exact` is now three-seed with OOD correct/translated-joint@16 `0.513/0.488` and depth-50 `0.396/0.354`, while first-seed `initial_marker` logic rate `0.5` has OOD `0.887/0.569` and depth-50 `0.875/0.219`. Hybrid-order eval is `18/30`; first completed `formal_think` train-1-to-5 slice has OOD correct/formal-joint/translated-joint@16 `0.538/0.120/0.347` and depth-50 `0.323/0.010/0.000`. Conditioned 40k rows `0..8` are complete, rows `9/10/11/12` are running, and no final/checkpoint eval JSONs exist yet. Sample inspection confirmed shortcut-neutral prompts, intended wrappers, normal answer extraction, valid shallow samples, and fragile depth-50 validity/truncation. The report was regenerated and mirrored with `64` PDFs and `55` CSVs; local TeX compilation remains unavailable.

Update 2026-05-31 22:35 CEST: shortcut-kind eval is `15/24`; `initial_marker` logic rate `0.5` is now three-seed with OOD correct/joint@16 `0.883/0.625` and depth-50 `0.854/0.344`. Hybrid-order remains `18/30`, with rows `18..21` running. Conditioned 40k rows `0..11` are complete, rows `12/13/14` are running, and no final/checkpoint eval JSONs exist yet. Sample inspection confirmed shortcut-neutral prompts, intended `<formal>` wrappers, normal answer extraction except one deeper truncated failure, and expected deeper validity/grounding fragility. The report was regenerated and mirrored with `64` PDFs and `55` CSVs; local TeX compilation remains unavailable.

Update 2026-06-01 02:45 CEST: shortcut-kind eval is `17/24`; `initial_marker` `nl_exact` rate `0.5` is now two-seed with OOD correct/translated-joint@16 `0.509/0.481`, depth-50 `0.125/0.109`, and OOD parse@16 `0.991`. Hybrid-order remains `18/30`, with rows `18..21` running. Conditioned 40k is complete, 50k `3674883` rows `0..3` are running, and no final/checkpoint eval JSONs exist yet. Sample inspection confirmed shortcut-neutral prompts, intended `<think>` wrappers, working answer extraction, and depth-50 fragility. The report was regenerated and mirrored with `65` PDFs, `57` CSVs, and `5` Markdown supplements; local TeX compilation remains unavailable.

Oversight update 2026-05-24 15:06 CEST:

- Qwen 7B SFT rows `3656217_0,1,2` completed and rows `3,4,5` are running. Qwen eval rows `3656218_0,1,2` completed; later eval rows remain dependency-pending.
- Qwen 7B `logic_train1to10` sparse eval outputs are complete for seeds `3407..3409` under `${WORK}/synthetic-RLVL/passk_eval/hfsa_model_ablation_qwen2p5_7b_sparse/`. Seed OOD correct@16/joint@16 is `0.591/0.309`, `0.781/0.400`, and `0.481/0.250`; the three-seed mean is `0.618/0.320`. Depth-50 joint@16 is `0.000`, `0.094`, and `0.000`. Do not draw model-family conclusions until the matched `nl_exact` rows complete.
- Qwen-1.5B/Gemma replacement rows `3656359_2` and `3656387_3` completed after the earlier node failures. Original small-extra rows `0,1,4,5,6,7,8` completed; rows `9,10,11,12` are running; remaining rows are pending.
- Tiny Llama node-failed rows are recovered and the dependent eval `3656390_[0-5%3]` completed all six rows. Preliminary OOD correct@8 is `0.219/0.148/0.273` for `50M/100M/200M` logic and `0.086/0.016/0.055` for `50M/100M/200M` `nl_exact`; all six rows have OOD joint@8 and depth-50 correct/joint at `0.0`.
- Paired maze SFT rows `3656309_0,1` failed with CUDA OOM because gradient checkpointing was off. They were resubmitted as `3657088_0,1` with `GRADIENT_CHECKPOINTING=true`; by 15:05 CEST both replacement rows had run past the original OOM window, and replacement eval rows `3657089_0,1` remain dependency-pending.

Oversight update 2026-05-24 18:45 CEST:

- Qwen 7B SFT rows `3656217_0..5` completed; rows `6..8` are running; rows `9..17` remain pending by array limit. Eval rows `3656218_0..3` completed; rows `4,5` are running; later rows remain dependency-pending.
- Qwen 7B `logic_train1to20_seed3407` sparse eval produced OOD correct@16/joint@16 `0.703/0.182` and depth-50 correct@16/joint@16 `0.500/0.000`. Together with the completed `logic_train1to10` rows, Qwen has logic-only signal but no matched `nl_exact` comparison yet.
- Qwen-1.5B/Gemma rows `0,1,4..13` have completed, rows `14..17` are running, and recovered rows `2,3` are complete; eval `3656389` remains dependency-pending.
- OLMo-32B row `3656335_0` is still training at about 8.7h elapsed and has passed multiple online eval windows; row `1` and eval `3656336` remain pending.
- Paired maze retry `3657088_0,1` failed at step 2000 during online generation eval, not during training. The paired SFT script now defaults online eval past `max_steps`; dead eval `3657089` was canceled, and replacements `3657738_[0-1]` plus dependent eval `3657739_[0-1]` were submitted.

Scripts:

```bash
scripts/slurm/sweeps/sft/hfsa_model_ablation_qwen7b_2026-05-24.slurm
scripts/slurm/jobs/posthoc_hfsa_model_ablation_qwen7b_eval_2026-05-24.slurm
scripts/slurm/sweeps/sft/hfsa_model_ablation_small_extra_2026-05-24.slurm
scripts/slurm/jobs/posthoc_hfsa_model_ablation_small_extra_eval_2026-05-24.slurm
scripts/slurm/sweeps/sft/hfsa_model_ablation_olmo32_pilot_2026-05-24.slurm
scripts/slurm/jobs/posthoc_hfsa_model_ablation_olmo32_eval_2026-05-24.slurm
scripts/train_tiny_llama_pretrain.py
scripts/slurm/sweeps/pretrain/hfsa_tiny_llama_scratch_2026-05-24.slurm
scripts/slurm/jobs/posthoc_hfsa_tiny_llama_pretrain_eval_2026-05-24.slurm
scripts/slurm/sweeps/pretrain/hfsa_tiny_llama_scratch_100k_2026-05-25.slurm
scripts/slurm/jobs/posthoc_hfsa_tiny_llama_pretrain_100k_eval_2026-05-25.slurm
scripts/slurm/jobs/posthoc_hfsa_tiny_llama_pretrain_100k_intermediate_eval_2026-05-25.slurm
scripts/slurm/jobs/ood_lm_eval_tiny_llama_100k_2026-05-25.slurm
scripts/slurm/codex/hfsa_followup_oversight_2026-05-24.slurm
```
