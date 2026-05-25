# Synthetic-RLVL Current Handoff

Last updated: 2026-05-25 10:48 CEST.

This file is the current operational handoff for the active `synthetic-RLVL` work. Older RL-validity and earlier HFSA snapshots are intentionally not duplicated here; use the archived docs listed near the end if historical detail is needed.

## Active Research State

The active direction is the formal-logic CoT depth-scaling SFT study:

- Dataset: `flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget-depth50`.
- Main grid: `logic` vs `nl_exact`, train depths `1..5`, `1..10`, `1..15`, `1..20`, `1..25`, seeds `3407`, `3408`, `3409`.
- Main training: OLMo-3-1025-7B LoRA SFT, 10k optimizer steps, batch size 1.
- Evaluation: post-hoc pass@k with temporary LoRA merge and vLLM.
- Active eval protocol after the 2026-05-22 runtime patch:
  - final eval uses depths `{1,2,5,10,12,15,18,20,25,30,35,40,45,50}`, `32` prompts/depth, `16` samples/prompt, no separate greedy pass, stop string `</answer>`, output subdir `passk_eval/hfsa_depth_scaling_sparse/`;
  - intermediate eval uses depths `{1,5,10,15,20,25,30,40,50}`, `16` prompts/depth, `16` samples/prompt, no separate greedy pass, stop string `</answer>`, output subdir `passk_eval/hfsa_depth_scaling_intermediate_sparse/`.
  The old full-grid eval arrays were canceled at 2026-05-22 10:47 CEST and replaced by sparse eval arrays.

## Live Slurm Jobs

### SFT Training

| Job | Rows | State | What it is | Current progress / note |
| --- | --- | --- | --- | --- |
| `3646736_0` | 0 | completed | `logic_train1to5_10k_seed3407` | skipped quickly because final checkpoint already existed |
| `3646736_1..6` | 1-6 | completed | remaining early logic rows through `logic_train1to15_seed3407` | completed cleanly, exit `0:0` |
| `3647379_7..29` | 7-29 | completed | later logic and all NL rows | all completed cleanly, exit `0:0`; longest NL `1..25` rows took about 10.8-11.3h |

### Post-Hoc Eval

| Job | Rows | State | What it is | Current progress / note |
| --- | --- | --- | --- | --- |
| `3647708`, `3648279`, `3648280`, `3647711`, `3647712` | old eval arrays | canceled | old full-grid/old-protocol evals | canceled at 2026-05-22 10:47 CEST to avoid spending more GPU on incomparable slow eval |
| `3650951_[0-29%10]` | 0-29 | completed | sparse final eval for all 30 main 10k rows | 30/30 JSON complete; all tasks exit `0:0` |
| `3650952_[0,3,6,9,12,15,18,21,24,27%4]` | seed `3407` rows across all train depths/templates | completed | sparse intermediate eval for correct@1/correct@16 over time | 30/30 checkpoint JSON complete; all tasks exit `0:0` |

### New Follow-Up Jobs Submitted 2026-05-24

| Job | Rows | State at submit/check | What it is | Dependency / note |
| --- | --- | --- | --- | --- |
| `3656210_1` | `attribute_constraints` | completed | materialized audited paired-family train-10 dataset and depth-50 validation set | output under `${WORK}/synthetic-RLVL/datasets/materialized_paired_attribute_constraints_train10_20260524` |
| `3656210_0` | `maze_navigation` | failed | original maze train-10 materialization | exposed a depth-15 room-name bank limit; no training used this output |
| `3656308_0` | `maze_navigation` | completed | resubmitted maze train-10 materialization after generator fix | exit `0:0`; validated every generated row with `--validate-examples -1`; output under `${WORK}/synthetic-RLVL/datasets/materialized_paired_maze_navigation_train10_20260524` |
| `3656211`, `3656213` | paired SFT/eval originals | canceled | original paired pilot dependencies | canceled after `3656210_0` failed |
| `3656309_[0-3%2]`, failed retry `3657088_[0-1%2]`, completed retry `3657738_[0-1%2]` | family x `logic,nl_exact` | completed after retries | paired-family seed-3407 SFT pilot, train depths `1..10` | attribute rows `2,3` completed on the original array; original maze rows `0,1` failed CUDA OOM with gradient checkpointing off; first maze retry `3657088_0,1` trained to step 2000 then OOMed in online generation eval; after disabling default online eval, maze retry `3657738_0,1` completed exit `0:0` |
| `3656310_[2-3%2]`, canceled `3657089_[0-1%2]`, failed/canceled `3657739_[0-1%2]`, replacement `3659556_[0-1%2]` | family x `logic,nl_exact` | attribute complete / maze replacement pending | paired-family sparse pass@k eval | attribute eval rows completed with JSON outputs; maze row `3657739_0` failed at chunk `51/56` because a depth-45 prompt had `16400` tokens under `vllm_max_model_len=16384`; row `3657739_1` was canceled before the same expected failure; wrapper now uses `32768` context and batch `64` for maze eval; replacement `3659556` is pending |
| `3659338` | `attribute_constraints` hard rebuild | completed | hardened train-10 materialization after saturation | exit `0:0` after 31m53s; generator now uses `floor(depth/2)+2` compact slots, larger value banks, recent-window dependencies, and adversarial decoys |
| `3659339_[0-1%2]` | hard attribute `logic,nl_exact` | row `1` complete, row `0` running | seed-3407 SFT replacement for saturated attribute pilot | row `1` (`nl_exact`) completed exit `0:0`; row `0` (`logic`) is still training |
| `3659340_[0-1%2]` | hard attribute `logic,nl_exact` | dependency-pending | sparse pass@k eval for hard attribute replacement | depends on `3659339`; output subdir `passk_eval/paired_attribute_constraints_hard_sparse/` |
| `3656217_[0-17%3]` | Qwen 7B, templates x train depths `1..10,1..20,1..25` x seeds `3407..3409` | rows `0..14` completed, `15..17` running | HFSA architecture ablation using `Qwen/Qwen2.5-7B` LoRA SFT | representative subset, not full 30-row repeat |
| `3656218_[0-17%3]` | same row map as `3656217` | rows `0..13` completed, row `14` running, rows `15..17` dependency-pending | sparse pass@k eval for Qwen ablation | all nine Qwen logic outputs, all three `nl_exact_train1to10` outputs, and `nl_exact_train1to20_seed3407,3408` are complete; outputs to `passk_eval/hfsa_model_ablation_qwen2p5_7b_sparse/` |
| `3656323_[0-35%4]` | Qwen-1.5B + Gemma-4B, templates x train depths `1..10,1..20,1..25` x seeds `3407..3409` | rows `0,1,4..25` completed or recovered; rows `26..29` running; rows `2,3` recovered by retries; rows `30..35` pending by array limit | HFSA architecture ablation using `Qwen/Qwen2.5-1.5B` and `google/gemma-3-4b-pt` | exact `Qwen/Qwen2.5-1B` was not available; "Gemma-4" is implemented as Gemma 3 4B base |
| `3656359_2` | retry of `3656323` row `2` | completed | replacement for node-failed small-extra row 2 | first retry row 3 also hit `NODE_FAIL` on node `a0934` |
| `3656387_3` | retry of `3656323` row `3` | completed | second replacement for small-extra row 3 | submitted with `--exclude=a0934`; exit `0:0` |
| `3656389_[0-35%4]` | same row map as `3656323` | dependency-pending | sparse pass@k eval for Qwen-1.5B/Gemma-4B ablation | replacement for canceled `3656324` and `3656361`; dependency is `afterany:3656323,afterany:3656359,afterok:3656387` |
| `3656335_[0-1%1]` | OLMo-32B pilot, `logic` and `nl_exact`, train depth `1..20`, seed `3407` | row `0` completed, row `1` running | 4-GPU HFSA architecture/scale pilot using `allenai/OLMo-2-0325-32B` | exact OLMo-3 base 32B was not available; `allenai/Olmo-3.1-32B-Think` exists but is a Think model |
| original `3656336_[0-1%1]`, replacement `3658461_[0-1%1]` | same row map as `3656335` | original row `0` failed, original row `1` canceled; replacement dependency-pending | 4-GPU sparse pass@k eval for OLMo-32B pilot | `3656336_0` failed before generation because vLLM rejected `max_model_len=16384` against OLMo-2 config `max_position_embeddings=4096`; eval wrapper now exports `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`; replacement `3658461` depends on `aftercorr:3656335` |
| `3656338_[0-5%3]` | Llama3-style scratch pretraining, sizes `50m,100m,200m` x `logic,nl_exact` | effective six-row wave completed after retries | first tiny from-scratch HFSA pretraining wave | rows `0,1` hit `NODE_FAIL`; recovered by retries `3656388_0` and `3656360_1`; uses Llama3 tokenizer and random-init Llama configs |
| `3656360_1` | retry of `3656338` row `1` | completed | replacement for node-failed pretraining row 1 | first retry row 0 also hit `NODE_FAIL` on node `a0934` |
| `3656388_0` | retry of `3656338` row `0` | completed | second replacement for pretraining row 0 | submitted with `--exclude=a0934`; exit `0:0` |
| `3656390_[0-5%3]` | same row map as `3656338` | completed | sparse pass@k eval for tiny Llama scratch pretraining | six JSON outputs under `passk_eval/hfsa_tiny_llama_pretrain_sparse/`; all tasks exit `0:0` |
| original `3659405`, replacement `3659415_[0-11%3]` | tiny Llama checkpoint eval | completed | sparse pass@k for tiny checkpoints `10000` and `15000` | original `3659405_0..2` failed because Trainer checkpoints lacked tokenizer files; script now stages checkpoint weights with tokenizer metadata from `final/`; replacement produced all 12 JSONs and report was regenerated |
| `3659344_[0-1]`, rerun `3659348_[0-1]` | two Qwen-1.5B runs | completed | OOD lm-eval pilot on GSM8K, HotpotQA, 2WikiMultiHopQA, and MuSiQue | rerun used strict LongBench extraction so passage-copy text without an answer tag cannot get accidental F1 credit |
| `3659356_[0-89%4]` | 90 non-tiny 1-GPU runs | dependency-pending | broad OOD lm-eval for main OLMo-7B, paired pilots, hard attribute, Qwen-7B, Qwen-1.5B, and Gemma-4B runs | dependency `afterany:3656217:3656323:3656359:3656387:3656309:3657738:3659339`; missing checkpoints are skipped instead of failing the array |
| `3659357_[0-1%1]` | OLMo-32B pilot runs | dependency-pending | broad OOD lm-eval for the two OLMo-32B pilot rows | dependency `afterany:3656335`; uses 4-GPU vLLM tensor parallelism |
| original `3659392_[0-5%3]`, replacement `3659488_[0-5%3]` | six tiny scratch-pretrain runs | replacement completed | OOD lm-eval for tiny Llama checkpoints | original failed with vLLM CUDA device-side assert at `max_model_len=32768`; replacement uses `max_model_len=8192`, smaller vLLM batching, and `FORCE_LM_EVAL=1`; all rows exit `0:0` |
| `3656509`, `3656510`, `3657079`, `3657734`, `3658457`, `3658813`, current `3659047`, next `3659552` | oversight | `3659047` running / `3659552` begin-time pending | Codex oversight for this full HFSA follow-up wave | `3657079` canceled dead maze eval `3657089` and submitted `3657738`/`3657739`; `3657734` submitted OLMo eval replacement `3658461`; `3659047` submitted next pass `3659552` and maze eval replacement `3659556` |

New scripts:

```bash
scripts/slurm/jobs/build_paired_followup_train10_2026-05-24.slurm
scripts/slurm/sweeps/sft/paired_followup_train10_seed3407_2026-05-24.slurm
scripts/slurm/jobs/posthoc_paired_followup_train10_eval_2026-05-24.slurm
scripts/slurm/sweeps/sft/hfsa_model_ablation_qwen7b_2026-05-24.slurm
scripts/slurm/jobs/posthoc_hfsa_model_ablation_qwen7b_eval_2026-05-24.slurm
scripts/slurm/sweeps/sft/hfsa_model_ablation_small_extra_2026-05-24.slurm
scripts/slurm/jobs/posthoc_hfsa_model_ablation_small_extra_eval_2026-05-24.slurm
scripts/slurm/sweeps/sft/hfsa_model_ablation_olmo32_pilot_2026-05-24.slurm
scripts/slurm/jobs/posthoc_hfsa_model_ablation_olmo32_eval_2026-05-24.slurm
scripts/train_tiny_llama_pretrain.py
scripts/slurm/sweeps/pretrain/hfsa_tiny_llama_scratch_2026-05-24.slurm
scripts/slurm/jobs/posthoc_hfsa_tiny_llama_pretrain_eval_2026-05-24.slurm
scripts/slurm/jobs/posthoc_hfsa_tiny_llama_pretrain_intermediate_eval_2026-05-25.slurm
scripts/slurm/codex/hfsa_followup_oversight_2026-05-24.slurm
scripts/slurm/jobs/build_attribute_constraints_hard_train10_2026-05-25.slurm
scripts/slurm/sweeps/sft/paired_attribute_constraints_hard_train10_seed3407_2026-05-25.slurm
scripts/slurm/jobs/posthoc_attribute_constraints_hard_eval_2026-05-25.slurm
lm_eval_tasks/synthrlvl_ood/
scripts/analysis/inspect_lm_eval_ood_samples.py
scripts/slurm/jobs/ood_lm_eval_pilot_2026-05-25.slurm
scripts/slurm/jobs/ood_lm_eval_large_1gpu_2026-05-25.slurm
scripts/slurm/jobs/ood_lm_eval_large_olmo32_2026-05-25.slurm
scripts/slurm/jobs/ood_lm_eval_tiny_llama_2026-05-25.slurm
scripts/analysis/build_logic_cot_report.py
```

Implementation notes from this submission wave:

- `synthrlvl/datasets/paired_synthetic.py` now extends both key and room vocabularies deterministically when `maze_navigation` depth exceeds the fixed word banks.
- `synthrlvl/datasets/paired_synthetic.py` also hardens `attribute_constraints` after the first pilot saturated: requested depth maps to `floor(depth/2)+2` compact slots, values are compact `vN` atoms from a larger bank, dependencies use a recent-window DAG rather than a fixed two-step chain, and decoys share one correct prerequisite where possible while remaining logically inapplicable.
- `synthrlvl/eval_loop.py` now honors `VLLM_TENSOR_PARALLEL_SIZE` for vLLM evaluation, needed by the OLMo-32B pilot.
- `scripts/slurm/jobs/posthoc_hfsa_model_ablation_olmo32_eval_2026-05-24.slurm` now exports `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` because vLLM otherwise refuses the intended sparse depth-50 context for `allenai/OLMo-2-0325-32B`, whose config advertises `max_position_embeddings=4096`.
- `scripts/slurm/sweeps/sft/paired_followup_train10_seed3407_2026-05-24.slurm` now defaults online eval to `max_steps + 1` so paired maze recovery uses the separate sparse pass@k eval instead of OOM-prone in-training generation.
- `scontrol requeue` is disabled on this cluster for the node-failed rows, so replacement arrays were submitted and dependent eval arrays were canceled/recreated. Repeated node failures occurred on `a0934`; the second single-row retries exclude that node.
- `scripts/evaluate_lm_eval.py` now has a `synthrlvl_ood` suite, local task include-path support for these OOD configs, and lm-eval unsafe-code/trust-remote-code flags needed by the LongBench-style tasks.
- `lm_eval_tasks/synthrlvl_ood/` defines tag-aware GSM8K/HotpotQA/2Wiki/MuSiQue tasks. GSM8K extracts `<answer>...</answer>` first and then numeric-normalizes; LongBench QA uses strict tag/explicit-answer extraction so copied context without an answer tag receives an empty prediction, avoiding false F1 credit from passage text.
- Tiny Llama scratch eval `3656390` completed all six rows in about 6 minutes per row. Train-band answer pass@8 is nontrivial, but OOD joint@8 and depth-50 correct/joint are `0.0` for every row; treat this as a smoke/mechanism signal, not evidence of solved extrapolation.
- Qwen 7B sparse eval now has all nine logic rows, the matched three `nl_exact_train1to10` rows, and two `nl_exact_train1to20` seeds. Three-seed means for Qwen logic train ranges `1..10`, `1..20`, and `1..25` are OOD correct@16/joint@16 `0.618/0.320`, `0.753/0.165`, and `0.906/0.431`; depth-50 correct@16/joint@16 `0.292/0.031`, `0.656/0.021`, and `0.854/0.156`. The matched Qwen `nl_exact_train1to10` mean is OOD correct@16/joint@16 `0.461/0.279` and depth-50 correct@16/joint@16 `0.427/0.000`. The partial two-seed Qwen `nl_exact_train1to20` mean is OOD correct@16/joint@16 `0.576/0.503` and depth-50 correct@16/joint@16 `0.406/0.203`. Remaining `nl_exact_train1to20/25` rows are incomplete, so do not draw full model-family conclusions yet.
- Paired `attribute_constraints` sparse eval completed for both seed-3407 templates: both are OOD/depth-50 correct@1 and correct@16 `1.000`; logic grounded joint@16 is also `1.000`, while `nl_exact` validity translation is currently `0.000`. This is evidence that the current `attribute_constraints` train-10 pilot is saturated, not a useful hard transfer benchmark. Harden the generator before spending a broad repeat on this family.
- Hard attribute replacement was implemented and submitted as `3659338 -> 3659339 -> 3659340`. Local checks before submission: paired dataset tests pass, depth-12 slot-count regression passes, OLMo token audit gives depth-50 totals around `13.4k` logic / `8.8k` NL, and a depth-50 smoke materialization with full validation succeeds.
- Paired `maze_navigation` SFT retry `3657738_0,1` completed after disabling default online generation eval. Sparse eval `3657739_0` failed at chunk `51/56` because depth-45 prompts exceeded the `16384` vLLM context cap (`16400` tokens); `3657739_1` was canceled before the same expected failure. `scripts/slurm/jobs/posthoc_paired_followup_train10_eval_2026-05-24.slurm` now defaults maze eval to `PASSK_VLLM_MAX_MODEL_LEN=32768` and batch `64`; replacement `3659556_[0-1%2]` is pending.
- `scripts/slurm/codex/hfsa_followup_oversight_2026-05-24.slurm` was submitted as `3656509` to run autonomous Codex checks over the active follow-up chains and update handoff docs or submit targeted repairs if needed. It scheduled the next pass as `3656510`.

### OOD lm-eval Readout - 2026-05-25

The local OOD suite now covers:

- `synthrlvl_gsm8k_tagged`
- `synthrlvl_longbench_hotpotqa_tagged`
- `synthrlvl_longbench_2wikimqa_tagged`
- `synthrlvl_longbench_musique_tagged`

Pilot jobs `3659344` and strict rerun `3659348` completed on two Qwen-1.5B train-1-to-10 checkpoints. Sample inspection showed the extractor correctly pulls answers from learned `<answer>` tags. For LongBench tasks, the strict rerun prevents passage-copy `<think>/<premises>` text from receiving accidental F1 when no final answer tag or explicit answer marker is present.

Strict pilot metrics on three examples/task were intentionally tiny, but useful as wiring checks:

| model | GSM8K EM | GSM8K tag_found | Hotpot F1 | 2Wiki F1 | MuSiQue F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen-1.5B `logic_train1to10_seed3407` | `0.000` | `1.000` | `0.000` | `0.000` | `0.000` |
| Qwen-1.5B `nl_exact_train1to10_seed3407` | `0.000` | `1.000` | `0.000` | `0.000` | `0.000` |

Interpretation: the OOD evaluation and extraction are working; these particular small trained checkpoints are bad on the sampled OOD tasks. Broad OOD arrays are now pending as `3659356` and `3659357`; tiny scratch checkpoints are covered separately by `3659392`.

Tiny Llama scratch pretraining detailed sparse eval:

| size | template | train correct@1 | train correct@8 | OOD correct@1 | OOD correct@8 | depth-50 correct@8 | joint@8 OOD/depth50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50M | logic | `0.207` | `0.797` | `0.036` | `0.219` | `0.000` | `0.000/0.000` |
| 50M | `nl_exact` | `0.135` | `0.656` | `0.014` | `0.086` | `0.000` | `0.000/0.000` |
| 100M | logic | `0.201` | `0.734` | `0.028` | `0.148` | `0.000` | `0.000/0.000` |
| 100M | `nl_exact` | `0.201` | `0.734` | `0.003` | `0.016` | `0.000` | `0.000/0.000` |
| 200M | logic | `0.246` | `0.859` | `0.080` | `0.273` | `0.000` | `0.000/0.000` |
| 200M | `nl_exact` | `0.381` | `0.719` | `0.012` | `0.055` | `0.000` | `0.000/0.000` |

The best tiny row is 200M logic on answer pass@8, but none of the tiny models show strict valid extrapolating reasoning. This is a smoke result: the pretraining path runs and learns some train-band answer/format behavior, but it has not solved OOD/depth-50 reasoning.

### Report Artifacts - 2026-05-25

Analysis/report builder:

```bash
${HPCVAULT}/.venv_rlvl_posttrain/bin/python scripts/analysis/build_logic_cot_report.py
```

Output root:

```bash
analysis/logic_cot_report_2026-05-25/
```

Generated artifacts include:

- LaTeX report source: `analysis/logic_cot_report_2026-05-25/logic_cot_report_2026-05-25.tex`
- Main OLMo final plots: `figures/olmo7b_final_by_train_depth.pdf`, `figures/olmo7b_depth_correct16.pdf`, `figures/olmo7b_depth_joint16.pdf`
- Main OLMo checkpoint plots: `figures/olmo7b_checkpoint_correct_k8_k16.pdf`, `figures/olmo7b_checkpoint_joint_k8_k16.pdf`
- Tiny plots: `figures/tiny_llama_final_bands_correct_joint.pdf`, `figures/tiny_llama_{50m,100m,200m}_depth_correct_joint.pdf`, and checkpoint plots `figures/tiny_llama_checkpoint_correct_k8.pdf`, `figures/tiny_llama_checkpoint_joint_k8.pdf` using 10k, 15k, and final/20k checkpoints
- Partial Qwen plot: `figures/qwen7b_partial_ood_correct_joint.pdf`
- Sample generation panel PDF: `figures/sample_generation_panels.pdf`
- CSV tables under `tables/`
- Tiny OOD lm-eval table: `tables/tiny_llama_ood_lmeval_summary.csv`

The report notes that GSM8K is numeric exact-match accuracy after `<answer>` extraction, while HotpotQA/2Wiki/MuSiQue should be reported as strict-extraction QA F1/EM rather than plain accuracy. `pdflatex`/`latexmk` are not installed on this node, so the LaTeX source was generated but not compiled here. All plot PDFs were generated successfully.

Tiny OOD lm-eval replacement `3659488` completed after reducing the tiny model context to 8192. Results are near-zero as expected for these scratch models: GSM8K EM is `0.000` for 50M/100M, `0.0068` for 200M logic, and `0.0045` for 200M NL; strict HotpotQA/2Wiki/MuSiQue F1 is `0.000` for every tiny row. LongBench contexts are truncated for the tiny 8192-context models, so treat these as smoke/downstream sanity numbers rather than fair long-context QA results.

### Other Visible Jobs Under This Account

These are visible in `squeue` but are not the active `synthetic-RLVL` HFSA handoff target.

| Job | State | Note |
| --- | --- | --- |
| `3656518` | pending | `seqedit_oversight`; unrelated to this handoff |
| `3659266_[0-2]` | running | `tjepa_lstream`; unrelated to this handoff |

## Health Summary

- All 30 main 10k SFT rows are complete. No SFT OOM, quota, or checkpoint-save failure was found. One row skipped because its final checkpoint already existed.
- NL SFT losses are much lower than logic losses at the same train-depth range, so NL is easier to fit under this LoRA setup. Final pass@k shows that this does not translate into uniformly better OOD extrapolation.
- Sparse eval jobs completed under the new protocol. The old protocol produced three 1k-sanity JSON files and six intermediate JSON files, but no full 10k final pass@k JSON before cancellation. The sparse protocol produced 30 final JSON files and 30 intermediate checkpoint JSON files.
- Sparse logs are healthy: no Traceback/OOM/quota/no-space errors seen in `3650951`/`3650952`; all sparse final/intermediate eval tasks exit `0:0`.
- Eval code and Slurm defaults now use sparse explicit depth grids, vLLM stop strings with stop tags kept in output, skipped greedy pass by default, sampled-example JSONL diagnostics when greedy is skipped, and scoring progress logs.
- The new `grounded_valid` / `citation_free_grounded_valid` metrics are not currently interpretable for HFSA logic traces. The prompt does not expose canonical predicate letters, so generated formal traces often choose a semantically equivalent but syntactically different predicate mapping. Use internal citation-free validity and NL-to-FOL validity for the current readout until a semantic/canonicalized grounded verifier is implemented.
- Disk recovered after eval cleanup traps; no active HFSA eval merge dirs are expected.

Approximate disk state at this handoff:

| Path | Size | Note |
| --- | ---: | --- |
| `${WORK}/synthetic-RLVL/runs` | 148G | completed SFT LoRA checkpoints and logs |
| `${WORK}/synthetic-RLVL/tmp` | small/transient | completed sparse evals cleaned their merged checkpoints |
| `${WORK}/synthetic-RLVL/passk_eval` | 80M | sparse final/intermediate JSON/JSONL outputs |

## Sparse Eval Readout

This is the completed comparable sparse-protocol grid. Logic uses citation-free joint validity; `nl_exact` uses translated NL-to-FOL joint validity. Train-band correct@16 and joint@16 are `1.000` for every group below.

| Template | Train depths | OOD correct@1 | OOD correct@16 | OOD joint@16 | OOD joint AUC | Depth-50 joint@16 | Max depth joint >= 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
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

Readout:

- `nl_exact_train1to5` is stronger than `logic_train1to5` OOD, so the formal trace is not automatically better at shallow training depth.
- Logic wins the paired OOD joint@16 comparison at train depths `1..10`, `1..15`, and `1..20`; mean logic-minus-NL OOD joint@16 deltas are `+0.064`, `+0.156`, and `+0.222`.
- `nl_exact_train1to25` catches up and slightly exceeds `logic_train1to25` on OOD joint@16 (`0.748` vs `0.710`) and OOD joint AUC (`0.757` vs `0.711`). At the deepest train range, the claim is no longer "logic strictly dominates"; the cleaner claim is that logic is more sample/depth efficient at intermediate train depths, while matched NL can catch up with enough train-depth coverage.
- Depth-50 joint@16 is similar at train depth `1..25`: `logic=0.417`, `nl_exact=0.427`.
- Sparse seed-3407 intermediate eval now has 30 checkpoint JSON files. For `logic_train1to25_seed3407`, OOD correct@16/joint@16 moves from `0.938/0.500` at checkpoint 1000 to `0.979/0.604` at checkpoint 3000 and `0.917/0.667` at checkpoint 10000.

Analysis artifacts:

| Artifact | Path |
| --- | --- |
| Aggregation script | `scripts/analysis/aggregate_hfsa_depth_scaling.py` |
| Final per-run table | `analysis/hfsa_depth_scaling_2026-05-23/tables/final_run_metrics.csv` |
| Final group summary | `analysis/hfsa_depth_scaling_2026-05-23/tables/final_group_summary.csv` |
| Compact markdown table | `analysis/hfsa_depth_scaling_2026-05-23/tables/final_group_summary_compact.md` |
| Depth curves | `analysis/hfsa_depth_scaling_2026-05-23/figures/final_depth_correct16.png`, `analysis/hfsa_depth_scaling_2026-05-23/figures/final_depth_joint16.png` |
| OOD summary figure | `analysis/hfsa_depth_scaling_2026-05-23/figures/final_ood_metrics_by_train.png` |
| Intermediate curves | `analysis/hfsa_depth_scaling_2026-05-23/figures/intermediate_seed3407_curves.png` |

## Follow-Up Dataset Audit

Broader local materialization/validation audit run on 2026-05-23:

```bash
python -m pytest -q tests/test_paired_synthetic_datasets.py
python scripts/data/build_paired_synthetic_dataset.py --kind <kind> --train-rows 120 --train-max-depth 10 --val-rows-per-depth 4 --val-max-depth 12 --validate-examples -1 ...
```

| Family | Status | Notes |
| --- | --- | --- |
| `attribute_constraints` | passed | audit artifact under `analysis/paired_dataset_audit_2026-05-23/attribute_constraints/` |
| `maze_navigation` | fixed and passed | generator now extends the key vocabulary with deterministic `key_XX` names when depth exceeds the color list; audit artifact under `analysis/paired_dataset_audit_2026-05-23/maze_navigation/` |
| `official_igsm` | blocked | broader audit found invalid proof lines for subtraction substitutions, e.g. `v_J = 13 - v_d`; do not train on this family until proof generation/verifier support is fixed |

Near-term experiment implication:

- Good next synthetic transfer candidates: `maze_navigation` and `attribute_constraints`, after full-size datasets are materialized and pushed/local-root wired.
- Submitted first paired-family pilot wave: train-10 materialization for `maze_navigation` and `attribute_constraints`, followed by seed-3407 `logic`/`nl_exact` SFT and sparse eval.
- `attribute_constraints` train-10 seed-3407 SFT/eval completed for both templates. Both `logic` and `nl_exact` reach OOD/depth-50 correct@1 and correct@16 `1.000`; `logic` also has grounded joint@16 `1.000`. The current family is saturated and should be made harder before broader repeats; the `nl_exact` NL-to-FOL validity metrics are `0.000`, likely because the NL validity translator does not yet cover this paired family, so use correctness rather than NL validity for the first attribute readout.
- `maze_navigation` train-10 SFT is recovered: retry `3657738_[0-1%2]` completed after disabling default online generation eval. Sparse pass@k eval `3657739_[0-1%2]` is running and hitting generation caps in higher-depth chunks, so runtime remains the only active concern so far.
- Do not launch `official_igsm` training yet.
- Submitted first model ablation wave: `Qwen/Qwen2.5-7B` on train depths `1..10`, `1..20`, `1..25`, both templates, seeds `3407..3409`, with dependent sparse eval.
- Submitted additional model ablation wave: `Qwen/Qwen2.5-1.5B`, `google/gemma-3-4b-pt`, and a two-row `allenai/OLMo-2-0325-32B` pilot. Exact `Qwen/Qwen2.5-1B` and an OLMo-3 base 32B model were not available under those names.
- Submitted first tiny from-scratch pretraining wave: random-init Llama configs at `50M/100M/200M` on HFSA logic vs `nl_exact`. This is a pilot implementation inside this repo; serious larger-scale pretraining should still use dedicated distributed pretraining infrastructure.

## Old Completed Eval Readout

Treat this as historical operational signal, not final evidence. These are partial old-protocol sanity/intermediate outputs retained for context after the old eval arrays were canceled.

| Artifact | Band | Correct | Joint valid+correct | Note |
| --- | --- | ---: | ---: | --- |
| 1k `logic_train1to15`, pass@16 | ID `1..15` | `1.000` | `1.000` | final 1k sanity row `3647708_2` |
| 1k `logic_train1to15`, pass@16 | `16..25` | `0.731` | `0.372` | correct samples persist OOD, valid proof quality drops |
| 1k `logic_train1to15`, pass@16 | `26..50` | `0.542` | `0.013` | answer samples exist but valid long proofs mostly collapse |
| 10k `logic_train1to20_seed3408` `checkpoint-1000`, pass@8 | `21..25` | `0.863` | `0.725` | intermediate eval row `3647712_10` |
| 10k `logic_train1to20_seed3408` `checkpoint-3000`, pass@8 | `21..25` | `0.825` | `0.625` | correct@1 improves but pass@8 joint is lower than ckpt-1000 |
| 10k `logic_train1to5_seed3407` `checkpoint-1000`, pass@8 | `6..25` | `0.428` | `0.141` | shallow logic baseline |
| 10k `logic_train1to5_seed3407` `checkpoint-3000`, pass@8 | `6..25` | `0.453` | `0.191` | small improvement over checkpoint 1000 |
| 10k `nl_exact_train1to5_seed3407` `checkpoint-1000`, pass@8 | `6..25` | `0.628` | `0.591` | shallow NL early checkpoint is stronger than shallow logic on this small eval |
| 10k `nl_exact_train1to5_seed3407` `checkpoint-3000`, pass@8 | `6..25` | `0.637` | `0.616` | still stronger than shallow logic on this small eval |

## Important Recent Fixes

| Fix | Files | Status |
| --- | --- | --- |
| Moved W&B data staging out of `$HOME` | `scripts/env.sh` | applied |
| Added greedy/sampled vLLM chunk timing and token-count logging | `synthrlvl/eval_loop.py` | applied |
| Added `--vllm-max-model-len` | `scripts/evaluate_checkpoint_passk.py` | applied |
| Reduced HFSA pass@k eval size and set `vllm_max_model_len=16384` | HFSA eval Slurm scripts | applied |
| Added sparse explicit eval depths, `--skip-greedy`, `--stop-strings`, scoring progress, and sampled diagnostics | `synthrlvl/config.py`, `synthrlvl/eval_loop.py`, `synthrlvl/evaluation/pass_at_k.py`, `scripts/evaluate_checkpoint_passk.py`, HFSA eval Slurm scripts | applied 2026-05-22 10:30 CEST |
| Canceled old eval arrays and submitted sparse replacements | Slurm jobs `3647708`, `3648279`, `3648280`, `3647711`, `3647712`, `3650951`, `3650952` | old arrays canceled; `3650951` final all rows at `%10`, `3650952` seed-3407 intermediate subset at `%4` submitted 2026-05-22 10:47 CEST |
| Replaced bad eval dependencies for rows `0..6` | `3648279`, `3648280` | applied; old `3647709`, `3647710` canceled |
| Added HFSA sparse-grid aggregation and plotting | `scripts/analysis/aggregate_hfsa_depth_scaling.py` | applied 2026-05-23; generated tables/figures under `analysis/hfsa_depth_scaling_2026-05-23/` |
| Fixed maze paired-family key-depth scaling | `synthrlvl/datasets/paired_synthetic.py` | applied 2026-05-23; paired tests pass and depth-12 materialization audit passes |
| Submitted paired-family train-10 pilot and Qwen HFSA model ablation | Slurm jobs `3656210`, `3656211`, `3656213`, `3656217`, `3656218` | submitted 2026-05-24 09:17-09:18 CEST |
| Fixed maze paired-family room-depth scaling | `synthrlvl/datasets/paired_synthetic.py` | applied 2026-05-24 after `3656210_0` failed at depth 15; local smoke materialization through depth 50 passes |
| Added model ablation and pretraining follow-up scripts | Slurm jobs `3656323`, `3656335`, `3656338` plus dependent evals | submitted 2026-05-24; node-failed rows retried as `3656359`, `3656360`, then single-row `a0934`-excluded retries `3656387`, `3656388` |
| Added vLLM tensor-parallel env knob | `synthrlvl/eval_loop.py` | applied 2026-05-24 for OLMo-32B eval; set `VLLM_TENSOR_PARALLEL_SIZE=4` in the OLMo eval Slurm script |
| Recovered paired maze SFT OOM rows | Slurm jobs `3656310_0,1`, `3657088_[0-1%2]`, `3657089_[0-1%2]`, `3657738_[0-1%2]`, `3657739_[0-1%2]` | original paired maze SFT rows `3656309_0,1` failed CUDA OOM at 8192 tokens with gradient checkpointing off; first retry `3657088` fixed that but OOMed during online generation eval at step 2000; dead eval `3657089` was canceled; paired SFT script now defaults online eval past `max_steps`; retry `3657738_0,1` completed exit `0:0` and dependent eval `3657739_0,1` is running |
| Recovered OLMo-32B eval context-load failure | `scripts/slurm/jobs/posthoc_hfsa_model_ablation_olmo32_eval_2026-05-24.slurm`, Slurm jobs `3656336`, `3658461` | original eval row `3656336_0` failed before generation because vLLM rejected `max_model_len=16384`; wrapper now exports `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, `bash -n` passed, stale pending row `3656336_1` was canceled, and replacement eval array `3658461_[0-1%1]` was submitted with `aftercorr:3656335` |

Verification already run after the eval patch:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile synthrlvl/config.py synthrlvl/eval_loop.py synthrlvl/evaluation/pass_at_k.py synthrlvl/grpo_inprocess_eval.py scripts/evaluate_checkpoint_passk.py
bash -n scripts/slurm/jobs/posthoc_hfsa_depth_scaling_merge_eval_2026-05-19.slurm
bash -n scripts/slurm/jobs/posthoc_hfsa_depth_scaling_intermediate_eval_2026-05-19.slurm
bash -n scripts/slurm/jobs/posthoc_hfsa_depth_scaling_1k_merge_eval_2026-05-19.slurm
python -m pytest -q tests/test_pass_at_k.py
```

Result: `2 passed` for the latest targeted pytest run, plus a sparse-config smoke check. Latest paired SFT script check: `bash -n scripts/slurm/sweeps/sft/paired_followup_train10_seed3407_2026-05-24.slurm` passed after disabling default online eval.

Latest OLMo eval wrapper check: `bash -n scripts/slurm/jobs/posthoc_hfsa_model_ablation_olmo32_eval_2026-05-24.slurm` passed after adding `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`.

## Watch Items

| Issue | Why it matters | Suggested next action |
| --- | --- | --- |
| Grounded-validity metric is ill-posed for current HFSA logic outputs | Canonical predicate letters are not in the prompt; generated traces can be semantically correct but use a different formal symbol mapping than the gold trace, making syntactic grounded validity near zero. | Implement semantic/canonicalized grounding using the generated `<predicates>` and `<constants>` mappings, or suppress grounded metrics for HFSA until this is fixed |
| Repeated tokenizer/rope warnings in eval stderr | Logs repeatedly warn about `fix_mistral_regex` and integer rope-scaling fields, but completed rows exit `0:0` and the merged tokenizer reports `GPT2Tokenizer`. | Treat as nonblocking for current eval; verify tokenizer round-trip before publication-quality reruns |
| `official_igsm` broader audit fails | The official-iGSM paired family is not spotless: subtraction substitution proof lines fail validation. | Fix arithmetic proof generation or verifier support before using it for training |
| Paired maze sparse eval is long-running | `maze_navigation` train-depth-10 SFT is now recovered, but sparse eval rows are producing long generations at higher depths and repeatedly hitting generation caps. | Monitor `3657739_[0-1]`; if either row fails, inspect whether it is timeout, vLLM/runtime failure, or scoring rather than resubmitting blindly |
| Hard `attribute_constraints` replacement is live | The original attribute generator was saturated; the hardened replacement needs full materialization/SFT/eval before deciding whether to broaden this family. | Monitor `3659338`, `3659339`, and `3659340`; if eval still saturates, move to higher-arity/multi-query constraints rather than only increasing slot count |
| Broad OOD lm-eval arrays are pending | The OOD suite is newly implemented and pilot-validated, but the full arrays will be more expensive and will include many checkpoints. Tiny original `3659392` failed from vLLM long-context settings and was replaced by `3659488`. | Monitor `3659356`, `3659357`, and tiny replacement `3659488`; inspect sample JSONL and aggregate by training family before using the metrics for claims |
| Small-extra and tiny-pretrain rows hit repeated `NODE_FAIL` | These failures happened before Python produced tracebacks and repeated on node `a0934`; targeted retries have now completed for the failed rows. | Monitor the remaining small-extra original array rows and downstream eval dependency `3656389`; tiny eval `3656390` is complete |
| OLMo-32B eval needs explicit vLLM long-context override | `allenai/OLMo-2-0325-32B` advertises `max_position_embeddings=4096`, but the sparse depth-50 eval uses `vllm_max_model_len=16384`; vLLM fails fast unless the override is set. | Monitor replacement eval `3658461`; if it fails after model load, inspect for real long-context numerical/runtime failures rather than the earlier config validation error |
| Tiny pretraining is a pilot trainer, not a production pretraining stack | It is single-node HF Trainer over materialized HFSA traces, useful for mechanism signal but not for a 50B-scale run. | Inspect first loss curves/checkpoints; use Nanotron or similar before any large-scale pretraining claim |
| Broad non-tiny OOD arrays are pending | Tiny OOD replacement `3659488` is complete, but larger arrays `3659356` and `3659357` are still dependency-pending. | Monitor `3659356` and `3659357`; aggregate OOD results once those arrays write result JSONs |
| Codex oversight uses the `cs` shell alias | Batch jobs source `~/.bash_profile`/`~/.bashrc` and call `cs exec`; if the alias or CLI environment breaks, oversight exits `127`. | Check `logs/hfsa_followup_oversight_3656509.*`; the script was syntax-checked before submission |

## Commands For Next Check

```bash
source ./scripts/env.sh
squeue -u c107fa12 -o '%.18i %.9P %.34j %.2t %.11M %.6D %.24E %R'
sacct -j 3650951,3650952 --format=JobID%30,JobIDRaw,JobName%34,State,Elapsed,ExitCode,Start,End -n -P
sacct -j 3656210,3656308,3656309,3656310,3657088,3657089,3657738,3657739,3659556,3656217,3656218,3656323,3656359,3656387,3656389,3656335,3656336,3658461,3656338,3656360,3656388,3656390,3656509,3656510,3657079,3657734,3658457,3658813,3659047,3659552,3659338,3659339,3659340,3659344,3659348,3659356,3659357,3659392,3659405,3659415,3659488 --format=JobID%30,JobIDRaw,JobName%34,State,Elapsed,ExitCode,Start,End -n -P
for f in logs/hfsa_dscale_eval_3650951_*.out logs/hfsa_dscale_ckpt_eval_3650952_*.out; do [ -f "$f" ] && echo "### $f" && tail -n 20 "$f"; done
for f in logs/build_paired_t10_3656210_*.out logs/build_paired_t10_3656308_*.out logs/sft_pair_t10_3656309_*.out logs/sft_pair_t10_3657088_*.out logs/sft_pair_t10_3657738_*.out logs/pair_t10_eval_3656310_*.out logs/pair_t10_eval_3657089_*.out logs/pair_t10_eval_3657739_*.out logs/pair_t10_eval_3659556_*.out logs/sft_hfsa_qwen7b_3656217_*.out logs/eval_hfsa_qwen7b_3656218_*.out logs/sft_hfsa_extra_3656323_*.out logs/sft_hfsa_extra_3656359_*.out logs/sft_hfsa_extra_3656387_*.out logs/sft_hfsa_olmo32_3656335_*.out logs/eval_hfsa_olmo32_3656336_*.out logs/eval_hfsa_olmo32_3658461_*.out logs/pt_hfsa_llama_3656338_*.out logs/pt_hfsa_llama_3656360_*.out logs/pt_hfsa_llama_3656388_*.out logs/eval_pt_llama_3656390_*.out logs/hfsa_followup_oversight_3656509.* logs/hfsa_followup_oversight_3656510.* logs/hfsa_followup_oversight_3657079.* logs/hfsa_followup_oversight_3657734.* logs/hfsa_followup_oversight_3658457.* logs/hfsa_followup_oversight_3658813.* logs/hfsa_followup_oversight_3659047.* logs/hfsa_followup_oversight_3659552.*; do [ -f "$f" ] && echo "### $f" && tail -n 20 "$f"; done
```

## Pointers

| Document | Purpose |
| --- | --- |
| `docs/formal_logic_cot_research_plan_2026-05-19.md` | active research plan |
| `docs/hfsa_depth_scaling_plan_2026-05-19.md` | active HFSA depth-scaling implementation and eval plan |
| `docs/old_rl_validity_reward_direction_2026-05-19.md` | archived RL-validity direction |
| `docs/paired_synthetic_benchmarks_2026-05-20.md` | future paired benchmark families |
| `docs/materialized_dataset.md` | dataset materialization details |
