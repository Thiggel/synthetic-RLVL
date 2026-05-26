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

The suite contains GSM8K plus context-provided LongBench variants of HotpotQA, 2WikiMultiHopQA, and MuSiQue. Scoring is format-aware: GSM8K extracts from `<answer>...</answer>` before numeric normalization, while the multi-hop QA tasks use strict answer extraction before F1/EM so context-copy text does not receive accidental credit. HotpotQA/2Wiki/MuSiQue now report both QA F1 and exact-match/`qa_exact_match`; GSM8K remains numeric exact-match accuracy.

Pilot wiring jobs `3659344` and strict rerun `3659348` completed. The broad submitted evals are:

| job | scope | note |
| --- | --- | --- |
| `3659356_[0-89%4]` | main OLMo-7B, paired pilots, hard attribute, Qwen-7B, Qwen-1.5B, Gemma-4B | one-GPU LoRA merge/eval; missing checkpoints skip cleanly |
| failed `3659357_[0-1%1]`, replacement `3660240_[0-1%1]` | OLMo-32B pilot | full LongBench OOD failed because OLMo-2 32B enforces a 4096-position limit; replacement is GSM8K-only short-context vLLM eval |
| `3659392_[0-5%3]`, replacement `3659488_[0-5%3]`, EM rerun `3659634_[0-5%3]` | seed-3407 tiny Llama scratch-pretraining checkpoints | direct checkpoint eval; original failed at 32k context, replacement succeeded at 8192 context, EM rerun completed `0:0` and populated new LongBench EM fields |
| `3659629_[0-5%3]`, `3659633_[0-5%3]` | seed-3408/3409 tiny Llama scratch-pretraining checkpoints | dependency-pending on new tiny pretraining arrays `3659626`/`3659630`; separate from the larger-model automatic hook |

The larger-model lm-eval Slurm hook `scripts/slurm/jobs/lm_eval_hfsa_depth_scaling_2026-05-19.slurm` now defaults to this OOD suite so future larger training runs can attach the same downstream eval after training. Tiny scratch-pretraining runs are excluded from the automatic downstream hook.

OOD update 2026-05-26 13:29 CEST: broad array `3659356` has completed rows `0..22`, covering all main OLMo-7B logic rows, all `nl_exact_train1to5/10` rows, and two `nl_exact_train1to15` seeds. Three-seed logic means by train depth `1..5/10/15/20/25` are GSM8K EM `0.056/0.079/0.074/0.086/0.056` and mean strict LongBench F1 `0.395/0.391/0.383/0.408/0.404`. Three-seed `nl_exact` means for `1..5/10` are GSM8K EM `0.493/0.479` and mean LongBench F1 `0.171/0.179`; the two completed `nl_exact_train1to15` seeds have mean GSM8K EM `0.253` and mean LongBench F1 `0.267`. This is a partial downstream transfer readout only: shallow NL is much stronger on GSM8K, while logic is stronger on strict context-QA F1 so far.

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
- Shortcut-rich follow-up: initial build `3661122_[0-1%1]` failed in the probe because the old shortcut schema exhausted high-depth state-word/predicate capacity. Dependents `3661123`/`3661124` were canceled. After expanding schema state banks and enabling extended predicate rendering, local probes for shortcut rates `0.5` and `0.8` passed; replacement build `3661135_[0-1%1]` completed. Original SFT rows `3661136_0..2` OOMed under `train.gradient_checkpointing=auto`, so the SFT wrapper now defaults checkpointing to `true`; targeted replacement `3662743_[0-2,6-8%3]` covers failed rows and at-risk pending logic shortcut-0.8 rows. Eval `3661137` was canceled and replaced by `3662744_[0-11%3]`. The existing shortcut-neutral main grid is the `0.0` baseline.
- Hybrid-order full suite: `3661162_[0-29%4]` SFT and `3661164_[0-29%4]` eval cover `think_formal` and `formal_think` across all main train depths and three seeds.
- Conditioned dual-modality full suite: `3661165_[0-14%4]` SFT trains one model per train-depth/seed on separate mode-conditioned logic and NL examples; `3661166_[0-29%4]` eval requests both `conditioned_logic` and `conditioned_nl` from every checkpoint.

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

Report/plot update 2026-05-25: `scripts/analysis/build_logic_cot_report.py` now builds a LaTeX report plus CSV tables and PDF/PNG plots under `analysis/logic_cot_report_2026-05-25/`. It covers main OLMo-7B final plots, @16-only OLMo checkpoint curves split by matched logic/NL train-depth pair, tiny Llama plots split by model size and by correctness vs joint correct+valid, partial Qwen-7B architecture-ablation plots, and qualitative sample-generation panels. Because `pdflatex`/`latexmk` are not installed on the current node, the `.tex` report is generated but not compiled here.

Tiny checkpoint eval update 2026-05-25: the first tiny intermediate eval array `3659405` exposed that HF Trainer checkpoint dirs do not contain tokenizer files. The script now stages each checkpoint with tokenizer metadata copied from the corresponding `final/` directory. Replacement `3659415_[0-11%3]` completed cleanly and produced all 12 checkpoint JSONs for 10k and 15k; the report builder was rerun and now includes tiny curves with 10k, 15k, and final/20k points.

Tiny OOD lm-eval update 2026-05-25: original tiny OOD array `3659392` failed with a vLLM CUDA device-side assert under `max_model_len=32768`. The tiny eval script now defaults to `max_model_len=8192`, `max_num_seqs=8`, and smaller GPU memory utilization. Replacement/rerun arrays completed for all three seeds and the report now includes `tables/tiny_llama_ood_lmeval_summary.csv` plus by-seed rows. Three-seed GSM8K EM is near zero: 50M logic/NL `0.0025/0.0000`, 100M logic/NL `0.0010/0.0000`, and 200M logic/NL `0.0056/0.0020`; strict LongBench QA F1/EM is zero for every tiny size/template. Because contexts are truncated to 8192 for these tiny configs, use this as a downstream smoke readout, not a long-context QA claim.

Tiny 100k extension update 2026-05-25 18:42 CEST: the 20k tiny results stay in the report for now, and a separate 100k-step extension was submitted for all three seeds. Training arrays are `3660986`, `3660990`, and `3660994`; dependent final sparse eval arrays are `3660987`, `3660991`, and `3660995`; dependent checkpoint eval arrays are `3660988`, `3660992`, and `3660996`; dependent OOD lm-eval arrays are `3660989`, `3660993`, and `3660997`. Run names and output dirs include `100k`, with checkpoint evals planned at `20000,40000,60000,80000,100000`.

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
| Dense intermediate backfill bad `3660813`, replacement `3661090` | replacement running/pending | `3661090_0` and `3661090_3` completed the full `logic_train1to5_seed3407` and `logic_train1to10_seed3407` dense grids; rows `6` and `9` are running, with remaining rows pending by throttle |

Operational notes:

- Old greedy eval was a major runtime risk because many completions ran to `max_new_tokens` (`4096` for logic, `6144` for `nl_exact`). A token-length audit indicates these cap hits are degenerate/non-terminating outputs rather than legitimate need for longer answers: logic gold targets are far below `4096` even at depth 50, and logic eval started hitting `4096` already in depth chunk `5..8`; `nl_exact` started hitting `6144` around depths `17..20`, where gold targets are only about `2.2k` tokens.
- Runtime estimate from old logs: old final pass@k rows were trending over the 24h window for several rows. The new sparse protocol completed final rows in roughly `3.3h..8.0h`; the shallow logic rows were the slowest completed rows, while most later logic/NL rows finished in `3.3h..6.3h`.
- Sparse final rows show train-band correct@16 and joint@16 are saturated at `1.000` for every group. The main result is depth-dependent: `nl_exact_train1to5` beats `logic_train1to5` OOD, logic is stronger at train ranges `1..10`, `1..15`, and `1..20`, and `nl_exact_train1to25` catches up/slightly exceeds logic at the deepest train range.
- Sparse seed-3407 intermediate rows now cover all train-depth/template groups at the initial `1000,3000,10000` points. Bad dense backfill job `3660813` produced no new dense JSONs because it only saw `CHECKPOINT_STEPS=1000`; replacement `3661090` is running/pending for the saved `1000,2000,...,10000` grid and has completed the full `logic_train1to10_seed3407` curve. `logic_train1to25_seed3407` currently improves in OOD joint@16 from `0.500` at checkpoint 1000 to `0.604` at checkpoint 3000 and `0.667` at checkpoint 10000, while OOD correct@16 stays high (`0.938`, `0.979`, `0.917`).
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
| Qwen-1.5B/Gemma sparse eval | `3656389_[0-35%4]` | running/pending by array throttle | first four Qwen-1.5B logic eval rows completed; remaining rows are queued under the `%4` throttle. Replacement for canceled `3656324` and `3656361`; repeated immediate node failure occurred on `a0934`, so the second retry excludes it |
| OLMo-32B SFT pilot | `3656335_[0-1%1]` | none | `allenai/OLMo-2-0325-32B`, train depth `1..20`, seed `3407`, `logic` and `nl_exact`, 4 GPUs |
| OLMo-32B short-context eval | failed `3656336`, failed `3658461`, replacement `3660238_[0-1%1]` | SFT complete | OLMo-2 32B enforces a 4096-position limit, so replacement eval is depths `{1,2,5,10,12,15}` only under `passk_eval/hfsa_model_ablation_olmo2_32b_shortctx_sparse/`; exact OLMo-3 base 32B was not available under the checked IDs |
| tiny Llama scratch pretraining | seed-3407 `3656338_[0-5%3]`, retries `3656360_1`/`3656388_0`; seed-3408 `3659626_[0-5%3]`; seed-3409 `3659630_[0-5%3]` | none | `50M/100M/200M` random-init Llama configs, HFSA `logic` vs `nl_exact`; original seed-3407 rows `0,1` hit cluster `NODE_FAIL`; new seed arrays exclude `a0934` |
| tiny Llama sparse eval | seed-3407 `3656390_[0-5%3]`; seed deps `3659627_[0-5%3]`, `3659631_[0-5%3]` | seed-3407 dependency complete; new deps afterany on `3659626`/`3659630` | lightweight pass@k eval for the scratch-pretrained checkpoints |
| tiny Llama checkpoint eval | seed-3407 replacement `3659415_[0-11%3]`; seed deps `3659628_[0-11%3]`, `3659632_[0-11%3]` | new deps afterany on `3659626`/`3659630` | checkpoint-10000/15000 pass@k curves; script stages tokenizer metadata from final checkpoints |
| tiny Llama 100k scratch extension | train `3660986`, `3660990`, `3660994`; final eval `3660987`, `3660991`, `3660995`; checkpoint eval `3660988`, `3660992`, `3660996`; OOD eval `3660989`, `3660993`, `3660997` | final/checkpoint/OOD evals depend `afterok` on matching seed train array | rows `0..2` are complete for all three seeds; rows `3..5` are running for all three seeds |
| trace-control ablations | SFT `3661118_[0-17%3]`; eval `3661119_[0-17%3]` | eval depends `afterok:3661118` | rows `0` and `2` completed; row `1` is running; remaining rows are pending by array throttle |
| same target-token budget | SFT `3661120_[0-5%3]`; eval `3661121_[0-5%3]` | eval depends `afterok:3661120` | rows `0..2` completed and rows `3..5` are pending; logic 10k steps vs `nl_exact` 7140 steps from the OLMo tokenizer target-length audit |
| shortcut-rate ablation | failed build `3661122`; canceled deps `3661123`/`3661124`; completed replacement build `3661135_[0-1%1]`; original SFT `3661136_[0-11%3]`; replacement SFT `3662743_[0-2,6-8%3]`; replacement eval `3662744_[0-11%3]` | eval depends `afterany:3661136,afterok:3662743` | original logic rows `0..2` OOMed; original row `3` completed, rows `4,5` are running, and rows `9..11` are pending. Replacement rows `0..2` are running and rows `6..8` are pending |
| hybrid-order full suite | SFT `3661162_[0-29%4]`; eval `3661164_[0-29%4]` | eval depends `afterok:3661162` | rows `0..3` are running; remaining rows are pending |
| conditioned dual-modality full suite | SFT `3661165_[0-14%4]`; eval `3661166_[0-29%4]` | eval depends `afterok:3661165` | rows `0..4` completed and rows `5..14` are pending; each checkpoint trains on separate mode-conditioned logic/NL examples, and eval requests both modalities separately |
| Codex oversight | through current `3663541`, next pass `3664182` | none | autonomous `cs exec` oversight for the follow-up wave; `3663541` is running and `3664182` is begin-time pending for 2026-05-26 17:16 CEST |

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
