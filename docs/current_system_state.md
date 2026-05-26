# Synthetic-RLVL Current Handoff

Last updated: 2026-05-26 15:03 CEST.

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
| bad `3660813_[0,3,6,9,12,15,18,21,24,27%2]`, replacement `3661090_[0,3,6,9,12,15,18,21,24,27%2]` | seed `3407` rows across all train depths/templates | bad array canceled; replacement running/pending | denser sparse intermediate eval for the saved 1k-step checkpoint grid | rows `0` and `3` completed, giving full 1k-grid curves for `logic_train1to5_seed3407` and `logic_train1to10_seed3407`; rows `6` and `9` are running; remaining rows `12,15,18,21,24,27` are pending by throttle |

### New Follow-Up Jobs Submitted 2026-05-24

| Job | Rows | State at submit/check | What it is | Dependency / note |
| --- | --- | --- | --- | --- |
| `3656210_1` | `attribute_constraints` | completed | materialized audited paired-family train-10 dataset and depth-50 validation set | output under `${WORK}/synthetic-RLVL/datasets/materialized_paired_attribute_constraints_train10_20260524` |
| `3656210_0` | `maze_navigation` | failed | original maze train-10 materialization | exposed a depth-15 room-name bank limit; no training used this output |
| `3656308_0` | `maze_navigation` | completed | resubmitted maze train-10 materialization after generator fix | exit `0:0`; validated every generated row with `--validate-examples -1`; output under `${WORK}/synthetic-RLVL/datasets/materialized_paired_maze_navigation_train10_20260524` |
| `3656211`, `3656213` | paired SFT/eval originals | canceled | original paired pilot dependencies | canceled after `3656210_0` failed |
| `3656309_[0-3%2]`, failed retry `3657088_[0-1%2]`, completed retry `3657738_[0-1%2]` | family x `logic,nl_exact` | completed after retries | paired-family seed-3407 SFT pilot, train depths `1..10` | attribute rows `2,3` completed on the original array; original maze rows `0,1` failed CUDA OOM with gradient checkpointing off; first maze retry `3657088_0,1` trained to step 2000 then OOMed in online generation eval; after disabling default online eval, maze retry `3657738_0,1` completed exit `0:0` |
| `3656310_[2-3%2]`, canceled `3657089_[0-1%2]`, failed/canceled `3657739_[0-1%2]`, replacement `3659556_[0-1%2]` | family x `logic,nl_exact` | completed | paired-family sparse pass@k eval | attribute eval rows completed with JSON outputs; maze row `3657739_0` failed at chunk `51/56` because a depth-45 prompt had `16400` tokens under `vllm_max_model_len=16384`; row `3657739_1` was canceled before the same expected failure. The 32k-context replacement `3659556_0,1` completed exit `0:0`; maze `nl_exact` has higher answer-only OOD correct@16 than logic (`0.250` vs `0.003`), but neither row has OOD joint validity and both are `0.000` at depth 50 |
| `3659338` | `attribute_constraints` hard rebuild | completed | hardened train-10 materialization after saturation | exit `0:0` after 31m53s; generator now uses `floor(depth/2)+2` compact slots, larger value banks, recent-window dependencies, and adversarial decoys |
| `3659339_[0-1%2]` | hard attribute `logic,nl_exact` | completed | seed-3407 SFT replacement for saturated attribute pilot | both rows completed exit `0:0`; dependent eval also completed |
| `3659340_[0-1%2]` | hard attribute `logic,nl_exact` | completed | sparse pass@k eval for hard attribute replacement | both rows completed exit `0:0`; hard generator is no longer saturated. Logic gets OOD correct/joint@8 `0.488/0.356` and hard-tail `0.431/0.285`; `nl_exact` gets OOD correct@8 `0.806` and hard-tail correct@8 `0.785`, but NL-to-FOL parse/joint@8 remains `0.000` for this paired family |
| `3656217_[0-17%3]` | Qwen 7B, templates x train depths `1..10,1..20,1..25` x seeds `3407..3409` | completed | HFSA architecture ablation using `Qwen/Qwen2.5-7B` LoRA SFT | all 18 SFT rows completed exit `0:0`; representative subset, not full 30-row repeat |
| `3656218_[0-17%3]` | same row map as `3656217` | completed | sparse pass@k eval for Qwen ablation | all 18 JSON outputs complete under `passk_eval/hfsa_model_ablation_qwen2p5_7b_sparse/`; see Qwen metrics below |
| `3656323_[0-35%4]` | Qwen-1.5B + Gemma-4B, templates x train depths `1..10,1..20,1..25` x seeds `3407..3409` | completed/recovered | HFSA architecture ablation using `Qwen/Qwen2.5-1.5B` and `google/gemma-3-4b-pt` | rows `2,3` recovered by retries; all original tail rows including `34,35` completed exit `0:0`; exact `Qwen/Qwen2.5-1B` was not available; "Gemma-4" is implemented as Gemma 3 4B base |
| `3656359_2` | retry of `3656323` row `2` | completed | replacement for node-failed small-extra row 2 | first retry row 3 also hit `NODE_FAIL` on node `a0934` |
| `3656387_3` | retry of `3656323` row `3` | completed | second replacement for small-extra row 3 | submitted with `--exclude=a0934`; exit `0:0` |
| `3656389_[0-35%4]` | same row map as `3656323` | running/pending | sparse pass@k eval for Qwen-1.5B/Gemma-4B ablation | first four Qwen-1.5B logic eval rows completed and wrote JSON outputs; rows `4..35` remain queued/running under the `%4` array throttle |
| `3656335_[0-1%1]` | OLMo-32B pilot, `logic` and `nl_exact`, train depth `1..20`, seed `3407` | completed | 4-GPU HFSA architecture/scale pilot using `allenai/OLMo-2-0325-32B` | both SFT rows completed exit `0:0`; exact OLMo-3 base 32B was not available; `allenai/Olmo-3.1-32B-Think` exists but is a Think model |
| original `3656336`, failed replacement `3658461`, short-context replacement `3660238_[0-1%1]` | same row map as `3656335` | completed | 4-GPU pass@k eval for OLMo-32B pilot | OLMo-2 32B has `max_position_embeddings=4096`; forcing 16k context caused CUDA position-index asserts in `3658461`. The completed replacement is a bounded short-context slice, depths `{1,2,5,10,12,15}`, `vllm_max_model_len=4096`, `max_new_tokens=2048`, output subdir `passk_eval/hfsa_model_ablation_olmo2_32b_shortctx_sparse/`; both templates are saturated on train/hard-tail correct@16 and format-matched joint@16 in this short slice |
| `3656338_[0-5%3]` | Llama3-style scratch pretraining, sizes `50m,100m,200m` x `logic,nl_exact` | effective six-row seed-3407 wave completed after retries | first tiny from-scratch HFSA pretraining wave | rows `0,1` hit `NODE_FAIL`; recovered by retries `3656388_0` and `3656360_1`; uses Llama3 tokenizer and random-init Llama configs |
| `3656360_1` | retry of `3656338` row `1` | completed | replacement for node-failed pretraining row 1 | first retry row 0 also hit `NODE_FAIL` on node `a0934` |
| `3656388_0` | retry of `3656338` row `0` | completed | second replacement for pretraining row 0 | submitted with `--exclude=a0934`; exit `0:0` |
| `3656390_[0-5%3]` | same row map as `3656338` | completed | sparse pass@k eval for tiny Llama scratch pretraining | six JSON outputs under `passk_eval/hfsa_tiny_llama_pretrain_sparse/`; all tasks exit `0:0` |
| `3659626_[0-5%3]`, `3659630_[0-5%3]` | same tiny row map, seeds `3408` and `3409` | completed | missing tiny scratch-pretraining seeds | all seed-3408/3409 rows completed exit `0:0`, giving three seeds for each tiny size/template |
| original `3659405`, replacement `3659415_[0-11%3]`, new deps `3659628_[0-11%3]`, `3659632_[0-11%3]` | tiny Llama checkpoint eval | completed | sparse pass@k for tiny checkpoints `10000` and `15000` | original `3659405_0..2` failed because Trainer checkpoints lacked tokenizer files; script now stages checkpoint weights with tokenizer metadata from `final/`; seed-3407/3408/3409 checkpoint evals all completed |
| `3660986_[0-5%3]`, `3660990_[0-5%3]`, `3660994_[0-5%3]` | same tiny row map, seeds `3407`, `3408`, `3409` | running/pending by resources | 100k-step tiny scratch-pretraining extension | rows `0..2` completed for all three seeds; rows `3..5` are running for all three seeds. Run names include `100k`, with `max_steps=100000`, `save_steps=20000`, `save_total_limit=6`, and `--exclude=a0934` |
| `3660987_[0-5%3]`, `3660991_[0-5%3]`, `3660995_[0-5%3]` | same tiny 100k row map | dependency-pending | final sparse pass@k for 100k tiny checkpoints | depends `afterok` on the corresponding 100k training seed array; outputs to `passk_eval/hfsa_tiny_llama_pretrain_100k_sparse/` |
| `3660988_[0-29%3]`, `3660992_[0-29%3]`, `3660996_[0-29%3]` | tiny 100k rows x checkpoints `20000,40000,60000,80000,100000` | dependency-pending | checkpoint pass@k curves for 100k tiny checkpoints | depends `afterok` on the corresponding 100k training seed array; outputs to `passk_eval/hfsa_tiny_llama_pretrain_100k_intermediate_sparse/` |
| `3660989_[0-5%3]`, `3660993_[0-5%3]`, `3660997_[0-5%3]` | same tiny 100k row map | dependency-pending | OOD lm-eval for 100k tiny checkpoints | depends `afterok` on the corresponding 100k training seed array; outputs to `lm_eval_results/ood_tiny_llama_100k_2026-05-25/` |
| `3661118_[0-17%3]` | six trace-control templates x seeds `3407..3409` | running/pending | HFSA syntax/representation ablation SFT | rows `0` and `2` completed exit `0:0`; row `1` is still running; rows `3..17` are pending by array throttle. Templates are `terse_nl`, `rule_annotated_nl`, `pseudocode`, `shuffled_logic`, `invalid_logic`, `shuffled_nl`; train depths `1..25`, 10k steps |
| `3661119_[0-17%3]` | same row map as `3661118` | dependency-pending | sparse pass@k eval for trace-control ablations | depends `afterok:3661118`; outputs to `passk_eval/hfsa_ablation_trace_controls_20260525/` |
| `3661120_[0-5%3]` | `logic,nl_exact` x seeds `3407..3409` | running/pending | same target-token budget SFT | rows `0..2` completed exit `0:0`; rows `3..5` are pending. Train depths `1..25`; logic uses 10k steps, NL uses 7140 steps based on a 512-row OLMo tokenizer audit (`1038` vs `1454` mean target tokens) |
| `3661121_[0-5%3]` | same row map as `3661120` | dependency-pending | sparse pass@k eval for same target-token budget ablation | depends `afterok:3661120`; outputs to `passk_eval/hfsa_same_target_token_budget_20260525/` |
| failed `3661122_[0-1%1]` | shortcut rates `0.5`, `0.8` | failed | first shortcut-rich materialization attempt | failed in the pre-materialization probe because shortcut-enabled depth-22..25 examples exhausted the old schema state-word/predicate capacity; no training used this output |
| canceled `3661123_[0-11%3]`, `3661124_[0-11%3]` | shortcut SFT/eval | canceled | dependents of failed shortcut build | canceled after fixing the generator, then replaced by the chain below |
| `3661135_[0-1%1]` | shortcut rates `0.5`, `0.8` | completed | replacement materialization for shortcut-rich HFSA train-1..25 datasets | both rows completed exit `0:0`; local probes for both rates pass after expanding schema state banks and enabling extended predicate rendering; eval remains shortcut-neutral through depth 50 |
| original `3661136_[0-11%3]`, replacement `3662743_[0-2,6-8%3]` | `logic,nl_exact` x shortcut rates `0.5,0.8` x seeds `3407..3409` | original/replacement running | shortcut-rate robustness SFT | original logic shortcut-0.5 rows `0..2` failed OOM; original row `3` completed exit `0:0`, rows `4,5` are running, and rows `9..11` are pending. Replacement `3662743` covers rows `0..2,6..8`; rows `0..2` are running and rows `6..8` are pending, with `GRADIENT_CHECKPOINTING=true` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; baseline shortcut `0.0` is the existing main run |
| canceled `3661137_[0-11%3]`, replacement `3662744_[0-11%3]` | same row map as shortcut SFT | dependency-pending | sparse pass@k eval for shortcut-rate ablation | old eval depended `afterok` on the failed original array and was canceled; replacement depends `afterany:3661136` plus `afterok:3662743` and writes to `passk_eval/hfsa_shortcut_rate_ablation_20260525/` |
| `3661162_[0-29%4]` | hybrid templates x train depths x seeds | running/pending | full-grid hybrid-order SFT | rows `0..3` are running; rows `4..29` are pending. Templates are `think_formal` (NL then logic) and `formal_think` (logic then NL), train depths `1..5/10/15/20/25`, seeds `3407..3409`, one final `<answer>` |
| `3661164_[0-29%4]` | same row map as `3661162` | dependency-pending | sparse pass@k eval for full-grid hybrid-order ablation | depends `afterok:3661162`; outputs to `passk_eval/hfsa_hybrid_order_full_20260525/` |
| `3661165_[0-14%4]` | train depths x seeds | running/pending | conditioned dual-modality SFT | rows `0..4` completed exit `0:0`; rows `5..14` are pending. Each materialized row is duplicated into `conditioned_logic` and `conditioned_nl` targets with a `<reasoning_mode>` prompt tag; train depths `1..5/10/15/20/25`, seeds `3407..3409` |
| `3661166_[0-29%4]` | conditioned checkpoints x eval modality | dependency-pending | sparse pass@k eval for conditioned dual-modality models | depends `afterok:3661165`; evaluates each checkpoint twice, once requesting `conditioned_logic` and once requesting `conditioned_nl`; outputs to `passk_eval/hfsa_conditioned_dual_full_20260525/` |
| `3659344_[0-1]`, rerun `3659348_[0-1]` | two Qwen-1.5B runs | completed | OOD lm-eval pilot on GSM8K, HotpotQA, 2WikiMultiHopQA, and MuSiQue | rerun used strict LongBench extraction so passage-copy text without an answer tag cannot get accidental F1 credit |
| `3659356_[0-89%4]` | 90 non-tiny 1-GPU runs | running/pending | broad OOD lm-eval for main OLMo-7B, paired pilots, hard attribute, Qwen-7B, Qwen-1.5B, and Gemma-4B runs | 31 result directories are complete, including all 30 main OLMo-7B rows; rows `31..34` are running and rows `35..89` are pending by array throttle. Missing checkpoints are skipped instead of failing the array |
| failed `3659357_[0-1%1]`, replacement `3660240_[0-1%1]` | OLMo-32B pilot runs | completed | short-context OOD lm-eval for the two OLMo-32B pilot rows | full LongBench OOD is invalid for OLMo-2 32B's 4096-position config and failed with CUDA position-index asserts. Replacement `3660240` is GSM8K-only at `max_model_len=4096`; GSM8K EM is `0.197` for logic and `0.683` for `nl_exact` |
| original `3659392_[0-5%3]`, replacement `3659488_[0-5%3]`, EM rerun `3659634_[0-5%3]`, new deps `3659629_[0-5%3]`, `3659633_[0-5%3]` | tiny scratch-pretrain OOD eval | completed | OOD lm-eval for tiny Llama checkpoints | original failed with vLLM CUDA device-side assert at `max_model_len=32768`; replacements use `max_model_len=8192`; all three seeds now have GSM8K plus LongBench F1/EM. Tiny GSM8K EM is near zero and all strict LongBench F1/EM means are `0.000` |
| `3656509`, `3656510`, `3657079`, `3657734`, `3658457`, `3658813`, `3659047`, `3659552`, `3660235`, `3661005`, `3662229`, `3662735`, `3663541`, next `3664182` | oversight | `3663541` running / `3664182` begin-time pending | Codex oversight for this full HFSA follow-up wave | current pass found no new unrecovered severe failures and scheduled `3664182` for 2026-05-26 17:16 CEST. Active targets are dense checkpoint backfill `3661090`, 100k tiny pretraining, trace/token-budget SFTs, original/replacement shortcut, hybrid, conditioned ablations, Qwen/Gemma sparse eval, and broad OOD eval |

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
scripts/slurm/sweeps/pretrain/hfsa_tiny_llama_scratch_100k_2026-05-25.slurm
scripts/slurm/jobs/posthoc_hfsa_tiny_llama_pretrain_100k_eval_2026-05-25.slurm
scripts/slurm/jobs/posthoc_hfsa_tiny_llama_pretrain_100k_intermediate_eval_2026-05-25.slurm
scripts/slurm/jobs/ood_lm_eval_tiny_llama_100k_2026-05-25.slurm
scripts/slurm/sweeps/sft/hfsa_ablation_trace_controls_2026-05-25.slurm
scripts/slurm/jobs/posthoc_hfsa_ablation_trace_controls_eval_2026-05-25.slurm
scripts/slurm/sweeps/sft/hfsa_same_target_token_budget_2026-05-25.slurm
scripts/slurm/jobs/posthoc_hfsa_same_target_token_budget_eval_2026-05-25.slurm
scripts/slurm/jobs/build_materialized_hfsa_shortcut_rate_2026-05-25.slurm
scripts/slurm/sweeps/sft/hfsa_shortcut_rate_ablation_2026-05-25.slurm
scripts/slurm/jobs/posthoc_hfsa_shortcut_rate_ablation_eval_2026-05-25.slurm
scripts/slurm/sweeps/sft/hfsa_hybrid_order_full_2026-05-25.slurm
scripts/slurm/jobs/posthoc_hfsa_hybrid_order_full_eval_2026-05-25.slurm
scripts/slurm/sweeps/sft/hfsa_conditioned_dual_full_2026-05-25.slurm
scripts/slurm/jobs/posthoc_hfsa_conditioned_dual_full_eval_2026-05-25.slurm
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
- `synthrlvl/types.py`, `synthrlvl/task.py`, `synthrlvl/metrics.py`, and `synthrlvl/eval_loop.py` now support the trace-control templates `terse_nl`, `rule_annotated_nl`, `pseudocode`, `shuffled_logic`, `invalid_logic`, and `shuffled_nl`. Focused tests pass and the new Slurm wrappers pass `bash -n`.
- `synthetic_dataset.py` and `synthrlvl/natural_logic.py` now support extended predicate names such as `P0(x)` for high-depth shortcut-enabled HFSA schema examples. This fixes the failed shortcut build probe while preserving legacy one-letter predicate rendering for existing datasets.
- `train_sft.py` now treats the new NL-style trace controls as memory-heavy under `train.gradient_checkpointing=auto` at train depth `1..25`, matching the existing `nl_exact` behavior.
- `scripts/slurm/sweeps/sft/hfsa_shortcut_rate_ablation_2026-05-25.slurm` now defaults `train.gradient_checkpointing=true`. The first three logic shortcut rows OOMed under `auto`, because `logic` is not treated as memory-heavy by that heuristic; `bash -n` passed and targeted replacement rows were submitted as `3662743`.
- `synthrlvl/sft_data.py` now supports `conditioned_dual` SFT: it duplicates each materialized row into `conditioned_logic` and `conditioned_nl` examples, each with a `<reasoning_mode>` prompt tag, while post-hoc eval uses the two single-modality conditioned templates separately.
- `scripts/slurm/jobs/posthoc_hfsa_model_ablation_olmo32_eval_2026-05-24.slurm` no longer attempts the depth-50 sparse protocol for `allenai/OLMo-2-0325-32B`. The model advertises and enforces a 4096-position limit: the initial long-context override passed vLLM startup but failed during generation with CUDA position-index asserts. The recovery is a clearly separated short-context slice through depth 15 under `passk_eval/hfsa_model_ablation_olmo2_32b_shortctx_sparse/`.
- `scripts/slurm/sweeps/sft/paired_followup_train10_seed3407_2026-05-24.slurm` now defaults online eval to `max_steps + 1` so paired maze recovery uses the separate sparse pass@k eval instead of OOM-prone in-training generation.
- `scontrol requeue` is disabled on this cluster for the node-failed rows, so replacement arrays were submitted and dependent eval arrays were canceled/recreated. Repeated node failures occurred on `a0934`; the second single-row retries exclude that node.
- `scripts/evaluate_lm_eval.py` now has a `synthrlvl_ood` suite, local task include-path support for these OOD configs, and lm-eval unsafe-code/trust-remote-code flags needed by the LongBench-style tasks.
- `lm_eval_tasks/synthrlvl_ood/` defines tag-aware GSM8K/HotpotQA/2Wiki/MuSiQue tasks. GSM8K extracts `<answer>...</answer>` first and then numeric-normalizes; LongBench QA uses strict tag/explicit-answer extraction so copied context without an answer tag receives an empty prediction, avoiding false F1 credit from passage text. LongBench tasks now report both QA F1 and exact-match/`qa_exact_match`.
- Tiny Llama scratch eval `3656390` completed the six seed-3407 rows in about 6 minutes per row. Train-band answer pass@8 is nontrivial, but OOD joint@8 and depth-50 correct/joint are `0.0` for every row; treat this as a smoke/mechanism signal, not evidence of solved extrapolation. Missing tiny seeds `3408`/`3409` were submitted as `3659626`/`3659630` with dependent final, checkpoint, and OOD evals.
- A 100k-step tiny scratch-pretraining extension was submitted separately on 2026-05-25 as training arrays `3660986`, `3660990`, and `3660994`, with final pass@k eval `3660987`/`3660991`/`3660995`, checkpoint pass@k eval `3660988`/`3660992`/`3660996`, and OOD lm-eval `3660989`/`3660993`/`3660997`. The run names and output directories include `100k`, so the current 20k three-seed report remains unchanged until explicitly regenerated.
- Qwen 7B sparse eval completed all 18 representative rows. Three-seed means for logic train ranges `1..10`, `1..20`, and `1..25` are OOD correct@16/joint@16 `0.618/0.320`, `0.753/0.165`, and `0.906/0.431`; depth-50 correct@16/joint@16 `0.292/0.031`, `0.656/0.021`, and `0.854/0.156`. Matched `nl_exact` means for `1..10`, `1..20`, and `1..25` are OOD correct@16/joint@16 `0.461/0.279`, `0.438/0.339`, and `0.569/0.565`; depth-50 correct@16/joint@16 `0.427/0.000`, `0.333/0.135`, and `0.250/0.229`. This is a mixed architecture readout: Qwen logic wins answer correctness and depth-50 correctness at deeper train ranges, while Qwen `nl_exact_train1to25` has higher joint validity.
- Qwen-1.5B sparse eval has partial outputs only. The completed `logic_train1to10` three-seed rows give OOD correct@16/citation-free-joint@16 `0.691/0.231` and depth-50 correct@16/joint@16 `0.552/0.010`; the single completed `logic_train1to20_seed3407` row gives OOD `0.911/0.203` and depth-50 `0.812/0.000`. Do not interpret this as a modality result until matched `nl_exact` and Gemma rows finish.
- Paired `attribute_constraints` sparse eval completed for both seed-3407 templates: both are OOD/depth-50 correct@1 and correct@16 `1.000`; logic grounded joint@16 is also `1.000`, while `nl_exact` validity translation is currently `0.000`. This is evidence that the current `attribute_constraints` train-10 pilot is saturated, not a useful hard transfer benchmark. Harden the generator before spending a broad repeat on this family.
- Hard attribute replacement was implemented and submitted as `3659338 -> 3659339 -> 3659340`. Local checks before submission: paired dataset tests pass, depth-12 slot-count regression passes, OLMo token audit gives depth-50 totals around `13.4k` logic / `8.8k` NL, and a depth-50 smoke materialization with full validation succeeds.
- Hard attribute replacement eval completed both rows. The hardened family is no longer saturated: logic OOD correct/joint@8 is `0.488/0.356` and hard-tail correct/joint@8 is `0.431/0.285`; `nl_exact` OOD correct@8 is `0.806` and hard-tail correct@8 is `0.785`, but NL-to-FOL parse/joint@8 is `0.000`, so the validity readout for this paired family is blocked by translator coverage.
- Paired `maze_navigation` SFT retry `3657738_0,1` completed after disabling default online generation eval. Sparse eval `3657739_0` failed at chunk `51/56` because depth-45 prompts exceeded the `16384` vLLM context cap (`16400` tokens); `3657739_1` was canceled before the same expected failure. `scripts/slurm/jobs/posthoc_paired_followup_train10_eval_2026-05-24.slurm` now defaults maze eval to `PASSK_VLLM_MAX_MODEL_LEN=32768` and batch `64`; the 32k replacement `3659556_0,1` completed exit `0:0`.
- Maze replacement eval wrote both JSONs under `passk_eval/paired_followup_train10_sparse/`. Logic train/OOD/depth-50 correct@16 is `0.750/0.003/0.000` with joint `0.750/0.000/0.000`; `nl_exact` train/OOD/depth-50 correct@16 is `1.000/0.250/0.000`, but NL-to-FOL parse/joint is `0.000` throughout. This is not evidence of valid maze extrapolation for either substrate.
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

Interpretation: the OOD evaluation and extraction are working. Broad non-OLMo-32B OOD array `3659356` has completed all 30 main OLMo-7B rows. Three-seed logic means by train depth `1..5/10/15/20/25` are GSM8K EM `0.056/0.079/0.074/0.086/0.056` and mean strict LongBench F1 `0.395/0.391/0.383/0.408/0.404`. Matched `nl_exact` means are GSM8K EM `0.493/0.479/0.326/0.411/0.257` and mean strict LongBench F1 `0.171/0.179/0.214/0.185/0.145`. This is a clear downstream tradeoff under the current prompts/extractor: NL transfers much better to GSM8K numeric EM, while logic is substantially stronger on strict context-QA F1/EM. Inspect samples before framing this as a general reasoning-transfer claim. OLMo-32B OOD array `3659357` failed because LongBench contexts exceed OLMo-2 32B's 4096-position limit; replacement `3660240` completed GSM8K-only at `max_model_len=4096` with EM `0.197` for logic and `0.683` for `nl_exact`. Tiny scratch checkpoints are complete for all three seeds; strict LongBench F1/EM means are `0.000` for every tiny size/template and GSM8K EM remains near zero.

OLMo-32B GSM8K sample inspection: the `nl_exact` advantage is not an answer-tag extraction artifact. The logic model has GSM8K `tag_found=1.000`, `extracted_nonempty=1.000`, median generation length `11` characters, and mostly emits terse `<answer>\n<number>` guesses; the NL model has `tag_found=0.995`, median generation length about `524` characters, and emits the learned `<think><premises><proof><conclusion></think><answer>` scaffold. The likely issue is task/manifold mismatch: GSM8K is natural-language arithmetic, and the answer-only OOD prompt gives the logic model no formalized premises or predicate schema to instantiate, while the NL model can reuse a natural-language reasoning scaffold.

Tiny Llama scratch pretraining three-seed sparse eval summary:

| size | template | train correct@8 | OOD correct@8 | depth-50 correct@8 | joint@8 OOD/depth50 |
| --- | --- | ---: | ---: | ---: | ---: |
| 50M | logic | `0.766` | `0.141` | `0.000` | `0.000/0.000` |
| 50M | `nl_exact` | `0.672` | `0.036` | `0.000` | `0.000/0.000` |
| 100M | logic | `0.807` | `0.201` | `0.000` | `0.000/0.000` |
| 100M | `nl_exact` | `0.729` | `0.034` | `0.000` | `0.000/0.000` |
| 200M | logic | `0.854` | `0.240` | `0.000` | `0.000/0.000` |
| 200M | `nl_exact` | `0.719` | `0.047` | `0.000` | `0.000/0.000` |

The best tiny row is 200M logic on answer pass@8, but none of the tiny models show strict valid extrapolating reasoning. This is a smoke result: the pretraining path runs and learns some train-band answer/format behavior, but it has not solved OOD/depth-50 reasoning.

### Report Artifacts - updated 2026-05-26

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
- Main OLMo OOD lm-eval tables: `tables/main_olmo7b_ood_lmeval_summary.csv` and by-seed `tables/main_olmo7b_ood_lmeval_by_seed.csv`
- Main OLMo final plots: `figures/olmo7b_final_by_train_depth.pdf`, `figures/olmo7b_depth_correct16.pdf`, `figures/olmo7b_depth_joint16.pdf`
- Main OLMo checkpoint plots are now split by matched train-depth pair and metric: `figures/olmo7b_checkpoint_train1to{5,10,15,20,25}_{correct16,joint16}.pdf`. These are @16-only; old combined `k8_k16` checkpoint PDFs are no longer referenced by the report.
- Tiny plots are now split by model size and by metric: `figures/tiny_llama_{50m,100m,200m}_bands_{correct,joint}_k8.pdf`, `figures/tiny_llama_{50m,100m,200m}_depth_{correct,joint}_k8.pdf`, and checkpoint plots `figures/tiny_llama_{50m,100m,200m}_checkpoint_{correct,joint}_k8.pdf`.
- Partial Qwen plot: `figures/qwen7b_partial_ood_correct_joint.pdf`
- Sample generation panel PDF: `figures/sample_generation_panels.pdf`
- CSV tables under `tables/`
- Tiny OOD lm-eval tables: `tables/tiny_llama_ood_lmeval_summary.csv` and by-seed `tables/tiny_llama_ood_lmeval_by_seed.csv`

The report notes that GSM8K is numeric exact-match accuracy after `<answer>` extraction, while HotpotQA/2Wiki/MuSiQue should be reported as strict-extraction QA F1/EM rather than plain accuracy. The report builder now aggregates main OLMo OOD and tiny OOD by seed and includes LongBench EM columns. `pdflatex`/`latexmk` are not installed on this node, so the LaTeX source was generated but not compiled here. All plot PDFs were generated successfully.

Checkpoint-curve note: the completed main OLMo SFT runs saved checkpoints every `1000` optimizer steps, not every `500`, and the first sparse intermediate eval only evaluated `1000,3000,10000`. As of the 2026-05-26 report rebuild there are 46 main checkpoint JSONs: full 1k-grid rows for `logic_train1to5_seed3407` and `logic_train1to10_seed3407`, partial dense additions for `logic_train1to15_seed3407` and `logic_train1to20_seed3407`, and the original 1k/3k/10k grid for the remaining six seed-3407 rows. Future reruns need `train.save_steps=500` and enough `save_total_limit` if we want true 500-step curves.

Tiny OOD lm-eval is complete for all three tiny seeds after reducing the tiny model context to 8192. Three-seed GSM8K EM is near-zero: 50M logic/NL `0.0025/0.0000`, 100M logic/NL `0.0010/0.0000`, and 200M logic/NL `0.0056/0.0020`; strict HotpotQA/2Wiki/MuSiQue F1 and EM are `0.000` for every size/template. LongBench contexts are truncated for the tiny 8192-context models, so treat these as smoke/downstream sanity numbers rather than fair long-context QA results.

### Other Visible Jobs Under This Account

These are visible in `squeue` but are not the active `synthetic-RLVL` HFSA handoff target.

| Job | State | Note |
| --- | --- | --- |
| `3661093` | running | `tjepa_lstream_nooracle`; unrelated to this handoff |
| `3663741_[1-2]` | running | `tjepa_lstream`; unrelated to this handoff |

## Health Summary

- All 30 main 10k SFT rows are complete. No SFT OOM, quota, or checkpoint-save failure was found. One row skipped because its final checkpoint already existed.
- NL SFT losses are much lower than logic losses at the same train-depth range, so NL is easier to fit under this LoRA setup. Final pass@k shows that this does not translate into uniformly better OOD extrapolation.
- Sparse eval jobs completed under the new protocol. The old protocol produced three 1k-sanity JSON files and six intermediate JSON files, but no full 10k final pass@k JSON before cancellation. The sparse protocol produced 30 final JSON files and 30 intermediate checkpoint JSON files.
- Sparse logs are healthy: no Traceback/OOM/quota/no-space errors seen in `3650951`/`3650952`; all sparse final/intermediate eval tasks exit `0:0`.
- Eval code and Slurm defaults now use sparse explicit depth grids, vLLM stop strings with stop tags kept in output, skipped greedy pass by default, sampled-example JSONL diagnostics when greedy is skipped, and scoring progress logs.
- The new `grounded_valid` / `citation_free_grounded_valid` metrics are not currently interpretable for HFSA logic traces. The prompt does not expose canonical predicate letters, so generated formal traces often choose a semantically equivalent but syntactically different predicate mapping. Use internal citation-free validity and NL-to-FOL validity for the current readout until a semantic/canonicalized grounded verifier is implemented.
- Disk recovered after eval cleanup traps; the active dense checkpoint backfill currently has job-local merge dirs under `${WORK}/synthetic-RLVL/tmp/intermediate_eval_*`.

Approximate disk state at this handoff:

| Path | Size | Note |
| --- | ---: | --- |
| `${WORK}/synthetic-RLVL/runs` | 331G | completed SFT/pretraining checkpoints and logs |
| `${WORK}/synthetic-RLVL/tmp` | 33G | active dense checkpoint and OOD merge dirs plus transient outputs |
| `${WORK}/synthetic-RLVL/passk_eval` | 178M | sparse final/intermediate JSON/JSONL outputs |
| `${WORK}/synthetic-RLVL/lm_eval_results` | 3.1G | OOD lm-eval JSON and sample outputs |

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
- Sparse seed-3407 intermediate eval still has the 30 checkpoint JSON files from the first pass (`1000,3000,10000`). Bad dense backfill job `3660813` produced no new dense JSONs because Slurm parsed the comma-list export as `CHECKPOINT_STEPS=1000`; replacement `3661090` is filling the saved `1000,2000,...,10000` grid. Completed dense rows: `logic_train1to5_seed3407` has OOD correct@16 range `0.161..0.554`, OOD joint@16 range `0.045..0.098`, and depth-50 joint@16 is nonzero only at checkpoints 7000 and 9000 (`0.062`); `logic_train1to10_seed3407` has OOD correct@16 range `0.396..0.688`, OOD joint@16 range `0.083..0.344`, and depth-50 joint@16 is nonzero only at checkpoint 9000 (`0.062`). Rows `6` and `9` are now running the `logic_train1to15/20_seed3407` dense grids.

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
- `maze_navigation` train-10 SFT/eval is recovered: retry `3657738_[0-1%2]` completed after disabling default online generation eval, first sparse eval replacement `3657739` hit a vLLM context cap at depth 45, and 32k-context replacement `3659556_[0-1%2]` completed. The result is weak: neither substrate has valid OOD/depth-50 joint pass@16, and both are `0.000` correct@16 at depth 50.
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
| Recovered paired maze SFT OOM rows | Slurm jobs `3656310_0,1`, `3657088_[0-1%2]`, `3657089_[0-1%2]`, `3657738_[0-1%2]`, `3657739_[0-1%2]`, `3659556_[0-1%2]` | original paired maze SFT rows `3656309_0,1` failed CUDA OOM at 8192 tokens with gradient checkpointing off; first retry `3657088` fixed that but OOMed during online generation eval at step 2000; dead eval `3657089` was canceled; paired SFT script now defaults online eval past `max_steps`; retry `3657738_0,1` completed exit `0:0`; first dependent eval `3657739` hit the 16k vLLM context cap and was replaced by `3659556` |
| Recovered paired maze eval context cap | `scripts/slurm/jobs/posthoc_paired_followup_train10_eval_2026-05-24.slurm`, Slurm jobs `3657739`, `3659556` | `3657739_0` failed because a depth-45 prompt length `16400` exceeded `vllm_max_model_len=16384`; `3657739_1` was canceled before the same expected failure; maze eval now defaults to `32768` vLLM context and batch `64`; replacement `3659556_[0-1%2]` completed; wrapper also handles `PASSK_JITTER_SECONDS=0` without divide-by-zero and passed `bash -n` |
| Recovered shortcut SFT OOM rows | `scripts/slurm/sweeps/sft/hfsa_shortcut_rate_ablation_2026-05-25.slurm`, Slurm jobs `3661136`, `3662743`, `3662744` | original shortcut rows `3661136_0..2` OOMed with `train.gradient_checkpointing=auto`; wrapper now defaults checkpointing to `true`, at-risk original logic rows `6..8` were canceled before start, replacement SFT `3662743_[0-2,6-8%3]` was submitted with expandable CUDA segments, and old eval `3661137` was replaced by `3662744` |
| Re-scoped OLMo-32B eval to short-context | `scripts/slurm/jobs/posthoc_hfsa_model_ablation_olmo32_eval_2026-05-24.slurm`, Slurm jobs `3656336`, `3658461`, `3660238` | original eval row `3656336_0` failed vLLM config validation at 16k context; forced-long replacement `3658461_[0-1]` then failed with CUDA position-index asserts once generation exceeded OLMo-2 32B's real 4096-position table. The script now defaults to `vllm_max_model_len=4096`, depths `{1,2,5,10,12,15}`, `max_new_tokens=2048`, and `shortctx_sparse` outputs; replacement `3660238_[0-1%1]` was submitted 2026-05-25 14:47 CEST |
| Re-scoped OLMo-32B OOD to GSM8K short-context | `scripts/slurm/jobs/ood_lm_eval_large_olmo32_2026-05-25.slurm`, Slurm jobs `3659357`, `3660240` | full LongBench OOD is not runnable under OLMo-2 32B's 4096-position limit and failed with the same CUDA position-index asserts. Replacement `3660240_[0-1%1]` is GSM8K-only with `max_model_len=4096` and separate output root `ood_large_olmo32_gsm8k_shortctx_2026-05-25/` |

Verification already run after the eval patch:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile synthrlvl/config.py synthrlvl/eval_loop.py synthrlvl/evaluation/pass_at_k.py synthrlvl/grpo_inprocess_eval.py scripts/evaluate_checkpoint_passk.py
bash -n scripts/slurm/jobs/posthoc_hfsa_depth_scaling_merge_eval_2026-05-19.slurm
bash -n scripts/slurm/jobs/posthoc_hfsa_depth_scaling_intermediate_eval_2026-05-19.slurm
bash -n scripts/slurm/jobs/posthoc_hfsa_depth_scaling_1k_merge_eval_2026-05-19.slurm
python -m pytest -q tests/test_pass_at_k.py
```

Result: `2 passed` for the latest targeted pytest run, plus a sparse-config smoke check. Latest wrapper checks: `bash -n scripts/slurm/sweeps/sft/paired_followup_train10_seed3407_2026-05-24.slurm` passed after disabling default online eval, and `bash -n scripts/slurm/sweeps/sft/hfsa_shortcut_rate_ablation_2026-05-25.slurm` passed after forcing shortcut gradient checkpointing on by default.

Latest OLMo wrapper checks: `bash -n scripts/slurm/jobs/posthoc_hfsa_model_ablation_olmo32_eval_2026-05-24.slurm` and `bash -n scripts/slurm/jobs/ood_lm_eval_large_olmo32_2026-05-25.slurm` passed after the short-context re-scope. A dry-run of `scripts/evaluate_lm_eval.py --tasks synthrlvl_gsm8k_tagged --model-arg max_model_len=4096` also produced the expected lm-eval command.

## Watch Items

| Issue | Why it matters | Suggested next action |
| --- | --- | --- |
| Grounded-validity metric is ill-posed for current HFSA logic outputs | Canonical predicate letters are not in the prompt; generated traces can be semantically correct but use a different formal symbol mapping than the gold trace, making syntactic grounded validity near zero. | Implement semantic/canonicalized grounding using the generated `<predicates>` and `<constants>` mappings, or suppress grounded metrics for HFSA until this is fixed |
| Repeated tokenizer/rope warnings in eval stderr | Logs repeatedly warn about `fix_mistral_regex` and integer rope-scaling fields, but completed rows exit `0:0` and the merged tokenizer reports `GPT2Tokenizer`. | Treat as nonblocking for current eval; verify tokenizer round-trip before publication-quality reruns |
| `official_igsm` broader audit fails | The official-iGSM paired family is not spotless: subtraction substitution proof lines fail validation. | Fix arithmetic proof generation or verifier support before using it for training |
| Paired maze train-10 is a weak extrapolation result | The 32k-context replacement eval completed both templates, but both have OOD joint@16 `0.000` and depth-50 correct/joint@16 `0.000/0.000`; `nl_exact` has answer-only OOD correct@16 `0.250` vs logic `0.003`, with NL parse/joint still `0.000`. | Treat this as a negative/diagnostic pilot before broad paired-maze repeats; inspect generated samples before deciding whether to harden prompts, shorten eval traces, or change the family |
| Hard `attribute_constraints` replacement has no valid NL readout yet | The hardened replacement eval completed and is no longer saturated, but NL-to-FOL parse/joint remains `0.000` for this paired family. | Improve paired-family NL validity translation before treating NL joint validity as a scientific result; correctness can still be used as a partial readout |
| Broad OOD lm-eval arrays are running | The OOD suite is newly implemented and pilot-validated, but the full array includes many checkpoints and should be aggregated by family before claims. Tiny original `3659392` failed from vLLM long-context settings and was replaced by `3659488`; seed-3407 EM rerun `3659634` is complete. OLMo-32B full OOD `3659357` is invalid for 4096 context and has GSM8K-only replacement `3660240`. | Monitor running `3659356` plus the tiny 100k OOD deps; inspect sample JSONL and aggregate by training family before using the metrics for claims |
| Small-extra and tiny-pretrain rows hit repeated `NODE_FAIL` | These failures happened before Python produced tracebacks and repeated on node `a0934`; targeted retries have now completed for the failed rows, and small-extra SFT has released eval `3656389`. | Monitor sparse eval `3656389`; first Qwen-1.5B logic rows are complete, but matched NL/Gemma rows are still queued. Tiny seed-3407 eval `3656390` is complete and 100k tiny pretraining is running |
| OLMo-32B cannot support the depth-50 sparse protocol | `allenai/OLMo-2-0325-32B` advertises and enforces `max_position_embeddings=4096`; overriding vLLM's derived limit produced CUDA position-index asserts. | Treat OLMo-32B as a short-context architecture pilot only; do not compare it to depth-50-capable models on long-context extrapolation |
| Tiny pretraining is a pilot trainer, not a production pretraining stack | It is single-node HF Trainer over materialized HFSA traces, useful for mechanism signal but not for a 50B-scale run. | Inspect first loss curves/checkpoints; use Nanotron or similar before any large-scale pretraining claim |
| New ablation arrays are active | Trace controls, same-target-token runs, shortcut-rate runs, hybrid-order runs, and conditioned-dual runs are running, pending, or dependency-pending and have not produced scientific results yet. Shortcut SFT has replacement rows after initial logic OOMs. | Monitor `3661118`/`3661119`, `3661120`/`3661121`, original/replacement shortcut `3661136`/`3662743` and eval `3662744`, `3661162`/`3661164`, and `3661165`/`3661166`; then aggregate outputs under the new pass@k subdirs |
| Codex oversight uses the `cs` shell alias | Batch jobs source `~/.bash_profile`/`~/.bashrc` and call `cs exec`; if the alias or CLI environment breaks, oversight exits `127`. | Check `logs/hfsa_followup_oversight_3656509.*`; the script was syntax-checked before submission |

## Commands For Next Check

```bash
source ./scripts/env.sh
squeue -u c107fa12 -o '%.18i %.9P %.34j %.2t %.11M %.6D %.24E %R'
sacct -j 3650951,3650952,3660813,3661090 --format=JobID%30,JobIDRaw,JobName%34,State,Elapsed,ExitCode,Start,End -n -P
sacct -j 3656210,3656308,3656309,3656310,3657088,3657089,3657738,3657739,3659556,3656217,3656218,3656323,3656359,3656387,3656389,3656335,3656336,3658461,3660238,3656338,3656360,3656388,3656390,3656509,3656510,3657079,3657734,3658457,3658813,3659047,3659552,3660235,3661005,3662229,3662735,3663541,3664182,3659338,3659339,3659340,3659344,3659348,3659356,3659357,3660240,3659392,3659405,3659415,3659488,3659626,3659627,3659628,3659629,3659630,3659631,3659632,3659633,3659634,3660813,3661090,3660986,3660987,3660988,3660989,3660990,3660991,3660992,3660993,3660994,3660995,3660996,3660997,3661118,3661119,3661120,3661121,3661122,3661123,3661124,3661135,3661136,3661137,3662743,3662744,3661162,3661164,3661165,3661166 --format=JobID%30,JobIDRaw,JobName%34,State,Elapsed,ExitCode,Start,End -n -P
for f in logs/hfsa_dscale_eval_3650951_*.out logs/hfsa_dscale_ckpt_eval_3650952_*.out; do [ -f "$f" ] && echo "### $f" && tail -n 20 "$f"; done
for f in logs/build_paired_t10_3656210_*.out logs/build_paired_t10_3656308_*.out logs/sft_pair_t10_3656309_*.out logs/sft_pair_t10_3657088_*.out logs/sft_pair_t10_3657738_*.out logs/pair_t10_eval_3656310_*.out logs/pair_t10_eval_3657089_*.out logs/pair_t10_eval_3657739_*.out logs/pair_t10_eval_3659556_*.out logs/sft_hfsa_qwen7b_3656217_*.out logs/eval_hfsa_qwen7b_3656218_*.out logs/sft_hfsa_extra_3656323_*.out logs/sft_hfsa_extra_3656359_*.out logs/sft_hfsa_extra_3656387_*.out logs/sft_hfsa_olmo32_3656335_*.out logs/eval_hfsa_olmo32_3656336_*.out logs/eval_hfsa_olmo32_3658461_*.out logs/eval_hfsa_olmo32_3660238_*.out logs/ood_lmeval_o32_3659357_*.out logs/ood_lmeval_o32_3660240_*.out logs/pt_hfsa_llama_3656338_*.out logs/pt_hfsa_llama_3656360_*.out logs/pt_hfsa_llama_3656388_*.out logs/eval_pt_llama_3656390_*.out logs/hfsa_followup_oversight_3656509.* logs/hfsa_followup_oversight_3656510.* logs/hfsa_followup_oversight_3657079.* logs/hfsa_followup_oversight_3657734.* logs/hfsa_followup_oversight_3658457.* logs/hfsa_followup_oversight_3658813.* logs/hfsa_followup_oversight_3659047.* logs/hfsa_followup_oversight_3659552.* logs/hfsa_followup_oversight_3660235.* logs/hfsa_followup_oversight_3661005.*; do [ -f "$f" ] && echo "### $f" && tail -n 20 "$f"; done
for f in logs/sft_hfsa_shortcut_3661136_*.out logs/sft_hfsa_shortcut_3661136_*.err logs/sft_hfsa_shortcut_3662743_*.out logs/sft_hfsa_shortcut_3662743_*.err logs/hfsa_shortcut_eval_3662744_*.out logs/hfsa_shortcut_eval_3662744_*.err logs/hfsa_followup_oversight_3662229.* logs/hfsa_followup_oversight_3662735.* logs/hfsa_followup_oversight_3663541.* logs/hfsa_followup_oversight_3664182.*; do [ -f "$f" ] && echo "### $f" && tail -n 20 "$f"; done
```

## Pointers

| Document | Purpose |
| --- | --- |
| `docs/formal_logic_cot_research_plan_2026-05-19.md` | active research plan |
| `docs/hfsa_depth_scaling_plan_2026-05-19.md` | active HFSA depth-scaling implementation and eval plan |
| `docs/old_rl_validity_reward_direction_2026-05-19.md` | archived RL-validity direction |
| `docs/paired_synthetic_benchmarks_2026-05-20.md` | future paired benchmark families |
| `docs/materialized_dataset.md` | dataset materialization details |
