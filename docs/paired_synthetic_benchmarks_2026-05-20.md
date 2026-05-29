# Paired Synthetic Benchmarks: Theory And Construction (2026-05-20)

This note documents the paired natural-language / formal-logic benchmark families used for follow-up experiments on whether formal-logic CoT improves extrapolation and transfer. The design goal is not to vary the proof calculus. All tasks deliberately stay inside a small, verifier-friendly fragment: premise retrieval, conjunction introduction, implication elimination, equality substitution, and the arithmetic `MOD23` rule for official iGSM. The intended variation is the latent algorithm that the model must execute.

The common output format is:

```text
<question> ... </question>
<formal>
<constants> ... </constants>
<predicates> ... </predicates>
<premises> ... </premises>
<proof> ... </proof>
<conclusion> ... </conclusion>
</formal>
<answer> ... </answer>
```

Every benchmark below is designed to be evaluated with grounded validity: generated proof lines are checked against the gold canonical premises and gold conclusion rather than against premises hallucinated by the model. This matters because a model can otherwise create an internally valid but task-irrelevant proof.

## Audit Status - 2026-05-23

A broader local materialization/validation audit was run with every generated example validated by `LogicEngine`:

```bash
python -m pytest -q tests/test_paired_synthetic_datasets.py
python scripts/data/build_paired_synthetic_dataset.py --kind <kind> --train-rows 120 --train-max-depth 10 --val-rows-per-depth 4 --val-max-depth 12 --validate-examples -1 ...
```

| family | status | implication |
| --- | --- | --- |
| `maze_navigation` | passed after a key-vocabulary fix | safe first candidate for follow-up substrate-transfer runs after full-size materialization |
| `attribute_constraints` | passed | safe first candidate for follow-up substrate-transfer runs after full-size materialization |
| `official_igsm` | fixed locally 2026-05-28 | subtraction-substitution proof lines now validate after parser tokenization fix; train-10 build/SFT/eval chain submitted |

Audit artifacts live under:

```bash
analysis/paired_dataset_audit_2026-05-23/
```

Fix update 2026-05-28 12:51 CEST: the iGSM blocker was in the shared term tokenizer, not in the iGSM proof generator. `parse_formula()` strips whitespace before term tokenization, and the old term-token regex allowed `-` inside identifiers, so `v_b - v_d` became the single constant `v_b-v_d`. Equality substitution could not rewrite inside that term. `logic_engine/parser.py` now treats `-` as an arithmetic operator, `tests/test_logic_engine.py` has a regression test for `=E` inside subtraction, and the following checks passed:

```bash
python -m pytest -q tests/test_logic_engine.py tests/test_paired_synthetic_datasets.py
python scripts/data/build_paired_synthetic_dataset.py --kind official_igsm --output-root /tmp/igsm_validation_smoke_20260528 --train-rows 60 --train-max-depth 10 --val-rows-per-depth 2 --val-max-depth 50 --validate-examples -1 --chunk-size 20 --seed 3407
python scripts/data/build_paired_synthetic_dataset.py --kind maze_navigation --output-root /tmp/maze_validation_smoke_20260528 --train-rows 60 --train-max-depth 10 --val-rows-per-depth 2 --val-max-depth 50 --validate-examples -1 --chunk-size 20 --seed 3407
python scripts/data/build_paired_synthetic_dataset.py --kind attribute_constraints --output-root /tmp/attr_validation_smoke_20260528 --train-rows 60 --train-max-depth 10 --val-rows-per-depth 2 --val-max-depth 50 --validate-examples -1 --chunk-size 20 --seed 3407
```

Submitted iGSM train-10 chain after the fix:

| stage | job | note |
| --- | ---: | --- |
| iGSM train-10 materialization | `3671601_[2]` | completed exit `0:0`; wrote `${WORK}/synthetic-RLVL/datasets/materialized_paired_official_igsm_train10_20260528` with 50k train rows and depth-50 validation, no validation failures |
| iGSM seed-3407 SFT | `3671602_[4-5%2]` | both rows completed exit `0:0`; train depth `1..10` |
| iGSM sparse eval | `3671603_[4-5%2]` | both rows completed exit `0:0`; eval depths `1..50`, 32k vLLM context, output under `passk_eval/paired_followup_train10_sparse/` |

Full-suite submission 2026-05-28 15:48 CEST:

| stage | job | scope |
| --- | ---: | --- |
| paired full-suite materialization | `3672195_[0-2%3]` | `official_igsm`, `maze_navigation`, and hard `attribute_constraints`; train ranges `1..5/10/15/20/25`, 50k rows each; validation `val_step_01_1k` through `val_step_50_1k`; every generated row validated |
| paired full-suite SFT | `3672212_[0-89%6]` | `3` families x `5` train ranges x `logic,nl_exact` x seeds `3407/3408/3409`; 10k OLMo-7B LoRA SFT steps; gradient checkpointing on by default |
| paired full-suite sparse eval | `3672213_[0-89%4]` | dependent on SFT; sparse pass@k eval to depth 50, `32` prompts/depth, `16` generations/prompt, output under `passk_eval/paired_full_suite_sparse_20260528/` |
| paired full-suite Codex oversight | `3672214`, `3672448`, `3673399`, `3673729`, `3674556`, `3675380`, `3676517`, `3677238`, next `3677873` | `3672214`, `3672448`, `3673399`, `3673729`, `3674556`, `3675380`, and `3676517` completed; `3677238` is running and `3677873` is begin-time pending as of 2026-05-30 00:07 CEST |

The full-suite roots are:

```bash
${WORK}/synthetic-RLVL/datasets/materialized_paired_official_igsm_full_20260528
${WORK}/synthetic-RLVL/datasets/materialized_paired_maze_navigation_full_20260528
${WORK}/synthetic-RLVL/datasets/materialized_paired_attribute_constraints_hard_full_20260528
```

The submission scripts are:

```bash
scripts/slurm/jobs/build_paired_full_suite_2026-05-28.slurm
scripts/slurm/sweeps/sft/paired_full_suite_2026-05-28.slurm
scripts/slurm/jobs/posthoc_paired_full_suite_eval_2026-05-28.slurm
scripts/slurm/codex/paired_full_suite_oversight_2026-05-28.slurm
```

The first pending SFT/eval/oversight submissions `3672196`/`3672197`/`3672208` were canceled before start after replacing array-id-based startup sleeps with throttle-slot-based sleeps. The active dependent jobs are `3672212`/`3672213`; current oversight state is tracked in the table and latest oversight notes below.

Status update 2026-05-29 07:41 CEST:

- iGSM train-10 chain is complete. Logic gets OOD correct/joint@16 `0.488/0.406` and depth-50 `0.469/0.312`; `nl_exact` gets OOD correct@16 `0.544` and depth-50 correct@16 `0.438`, with NL-to-FOL joint `0.000`.
- Full paired-suite SFT rows `0..31` completed exit `0:0`; rows `32..37` are running and `38..89` are pending by array throttle. Eval `3672213_[0-89%4]` remains dependency-pending on `3672212`, so no full-suite eval JSONs exist yet.
- Paired full-suite oversight passes `3672214`, `3672448`, `3673399`, and `3673729` completed without finding unrecovered severe failures; next pass `3674556` is queued.

Oversight update 2026-05-28 17:02 CEST: build rows `3672195_0`, `3672195_1`, and `3672195_2` were all still running after about 1h14m. SFT `3672212_[0-89%6]` was pending on `afterok:3672195_*`; eval `3672213_[0-89%4]` was pending on `afterok:3672212_*`. Build-log inspection found no Traceback, proof-validation failure, OOM/CUDA OOM, context-length failure, quota/no-space error, dependency failure, node failure, or timeout/cancelled task. The materialized roots were actively being populated; no full-suite manifests were present yet, so no downstream SFT/eval rows were released.

Oversight update 2026-05-28 21:41 CEST: build rows `3672195_0`, `3672195_1`, and `3672195_2` completed exit `0:0`. The three full-suite roots now each have `full_suite_manifest.json` with 55 expected subsets and no missing parquet paths. SFT `3672212_0..5` released; row `0` (`official_igsm`, `logic`, train `1..5`, seed `3407`) reached `checkpoint-5000`, row `1` started optimizer steps, and rows `2..5` were still in normal startup/stagger windows. Eval `3672213_[0-89%4]` remains pending on `afterok:3672212_*`. Log scans found no Traceback, proof-validation failure, OOM/CUDA OOM, context-length failure, quota/no-space issue, dependency failure, node failure, timeout/cancelled task, tokenizer/model-load error, or vLLM failure. No eval JSONs are expected yet.

Oversight update 2026-05-29 01:40 CEST: SFT rows `3672212_0..15` completed exit `0:0`, covering `official_igsm` through `nl_exact_train1to15_seed3407`. Rows `16..21` are running and rows `22..89` remain pending by `JobArrayTaskLimit`; row `16`, row `17`, and row `18` have reached `checkpoint-5000`, while rows `19..21` are in normal startup/early training. Eval `3672213_[0-89%4]` is still pending on `afterok:3672212_*`, and `passk_eval/paired_full_suite_sparse_20260528/` has no pass@k JSONs yet. Log scans found no fatal Traceback, proof-validation failure, OOM/CUDA OOM, context-length/context-cap failure, quota/no-space issue, dependency failure, node failure, timeout/cancelled task, tokenizer/model-load error, or vLLM failure; matches were limited to benign tokenizer warnings, quota-info headers, and standard accelerate OOM-avoidance wording. `3673399` is the running oversight pass and `3673729` is scheduled for 2026-05-29 05:37 CEST.

Oversight update 2026-05-29 07:30 CEST: SFT rows `3672212_0..30` completed exit `0:0`, rows `31..35` are running, and rows `36..89` are pending by array throttle. Eval `3672213_[0-89%4]` remains dependency-pending and no full-suite eval JSONs exist yet. Oversight `3673399` completed exit `0:0`; next oversight `3673729` is queued.

Oversight update 2026-05-29 07:41 CEST: SFT row `3672212_31` completed exit `0:0`, row `3672212_36` released, and current running rows are `32..36`; rows `37..89` remain pending by `JobArrayTaskLimit`. Eval `3672213_[0-89%4]` is still pending on `afterok:3672212_*(unfulfilled)` with zero JSON outputs under `passk_eval/paired_full_suite_sparse_20260528/`. Fatal-log scan found no Traceback, proof-validation failure, actual OOM/CUDA OOM, context-length failure, quota/no-space issue, dependency failure, node failure, timeout/cancelled task, tokenizer/model-load error, or vLLM failure. Full-suite manifests remain complete with 55 expected subsets and no missing parquet paths per family. Oversight `3673729` is running and next oversight `3674556` is pending BeginTime.

Status update 2026-05-29 07:55 CEST: SFT row `3672212_37` released; rows `32..37` are running and rows `38..89` are pending by `JobArrayTaskLimit`. Eval remains dependency-pending with zero JSON outputs. Oversight `3673729` completed exit `0:0`; next oversight `3674556` is pending BeginTime.

Oversight update 2026-05-29 11:42 CEST: build rows `3672195_0..2` remain completed exit `0:0`, and all three full-suite manifests still report 55 subsets with no missing parquet paths. SFT rows `3672212_0..35` completed exit `0:0`; rows `36..41` are running on `maze_navigation` train-1-to-10, with rows `36` and `37` already at `checkpoint-5000`; rows `42..89` are pending by `JobArrayTaskLimit`. Eval `3672213_[0-89%4]` remains dependency-pending on `afterok:3672212_*` and `passk_eval/paired_full_suite_sparse_20260528/` still has zero JSON outputs. Fatal-log scan found no Traceback, proof-validation failure, actual OOM/CUDA OOM, context-length failure, quota/no-space issue, dependency failure, node failure, timeout/cancelled task, tokenizer/model-load error, vLLM failure, idle-GPU symptom, or `DependencyNeverSatisfied`; matches were limited to benign tokenizer warnings, build token-length warnings, quota headers, and standard accelerate OOM-avoidance wording. Oversight `3674556` is running and next oversight `3675380` is scheduled for 2026-05-29 15:39 CEST.

Oversight update 2026-05-29 16:05 CEST: build rows `3672195_0..2` remain completed exit `0:0`; all three full-suite manifests still have 55 subsets and no missing parquet paths. SFT rows `3672212_0..41` completed exit `0:0` and all have final adapter checkpoints. Rows `42..47` are running on `maze_navigation` train-1-to-15 (`logic` and `nl_exact`, seeds `3407..3409`) and are making optimizer-step progress; rows `48..89` remain pending by `JobArrayTaskLimit`. Eval `3672213_[0-89%4]` remains dependency-pending on `afterok:3672212_*(unfulfilled)`, and the full-suite eval output directory has not been created yet, so there are still zero pass@k JSONs. Focused log scan found no Traceback, proof-validation failure, actual OOM/CUDA OOM, context-length failure, quota/no-space issue, dependency failure, node failure, timeout/cancelled task, tokenizer/model-load error, vLLM failure, idle-GPU symptom, or `DependencyNeverSatisfied`; observed matches were benign tokenizer warnings, quota headers, and standard accelerate OOM-avoidance wording. `a100` had idle compatible nodes, but pending paired rows were throttle/dependency-blocked, so no partition edit was made. Oversight `3675380` is running and next oversight `3676517` is begin-time pending.

Oversight update 2026-05-29 20:05 CEST: build rows `3672195_0..2` remain completed exit `0:0`; all three full-suite manifests still have 55 subsets and no missing parquet paths. SFT rows `3672212_0..41` are complete with final adapters; rows `42..47` are running on `maze_navigation` train-1-to-15 (`logic` and `nl_exact`, seeds `3407..3409`) and are making optimizer progress. The latest normalized progress read was row `42` at `5184/10000`, row `43` at `4837/10000`, row `44` at `4632/10000`, row `45` at `4601/10000`, row `46` at `4356/10000`, and row `47` at `4334/10000`; row `42` has written `checkpoint-5000`. Rows `48..89` remain pending by `JobArrayTaskLimit`. Eval `3672213_[0-89%4]` remains dependency-pending on `afterok:3672212_*(unfulfilled)`, and the full-suite eval output directory has not been created yet, so there are still zero pass@k JSONs. Focused SFT log scan found no Traceback, proof-validation failure, actual OOM/CUDA OOM, context-length failure, quota/no-space issue, dependency failure, node failure, timeout/cancelled task, tokenizer/model-load error, vLLM failure, idle-GPU symptom, or `DependencyNeverSatisfied`; matches were limited to benign tokenizer/rope warnings and standard accelerate OOM-avoidance wording. Oversight `3675380` completed exit `0:0`; `3676517` is running and next oversight `3677238` is begin-time pending. No resubmission or partition edit was made.

Oversight update 2026-05-30 00:07 CEST: build rows `3672195_0..2` remain completed exit `0:0`; all three full-suite manifests still have 55 subsets and no missing parquet paths. SFT rows `3672212_0..41` are complete with final adapters; rows `42..47` are still running on `maze_navigation` train-1-to-15 and all have written `checkpoint-5000`. The latest normalized progress read was row `42` at `9139/10000`, row `43` at `8790/10000`, row `44` at `8586/10000`, row `45` at `8744/10000`, row `46` at `8452/10000`, and row `47` at `8489/10000`. Rows `48..89` remain pending by `JobArrayTaskLimit`. Eval `3672213_[0-89%4]` remains dependency-pending on `afterok:3672212_*(unfulfilled)`, and the full-suite eval output directory has not been created yet, so there are still zero pass@k JSONs. Focused SFT fatal-log scan found no Traceback, proof-validation failure, actual OOM/CUDA OOM, context-length failure, quota/no-space issue, dependency failure, node failure, timeout, tokenizer/model-load error, vLLM failure, idle-GPU symptom, or `DependencyNeverSatisfied`; matches in broader scans remain benign tokenizer/rope warnings, quota headers, and standard accelerate OOM-avoidance wording. `a100` had idle nodes, but pending paired rows are throttle/dependency-blocked, so no resubmission or partition edit was made. Oversight `3676517` completed, `3677238` is running, and next oversight `3677873` is begin-time pending.

Follow-up submission on 2026-05-24:

| stage | job | note |
| --- | ---: | --- |
| train-10 materialization | `3656210_1` completed, `3656210_0` failed, replacement `3656308_0` completed | `attribute_constraints` completed; `maze_navigation` exposed a fixed room-word-bank limit at depth 15, was fixed, resubmitted, and completed with every row validated |
| seed-3407 SFT pilot | original `3656211` canceled; `3656309_[2-3]` completed; maze retries `3657088_[0-1]` failed and `3657738_[0-1]` completed | original maze rows `3656309_0,1` failed CUDA OOM at 8192 tokens with gradient checkpointing off; first retry used `GRADIENT_CHECKPOINTING=true` and reached step 2000, then OOMed during online generation eval; after disabling default online eval, `3657738_0,1` completed exit `0:0` |
| sparse eval | original `3656213` canceled; `3656310_[2-3]` completed; `3657089_[0-1]` canceled; `3657739_0` failed, `3657739_1` canceled, replacement `3659556_[0-1]` completed | attribute eval complete; dead maze eval `3657089` was canceled after `3657088` failed; first maze eval replacement exposed a 16k vLLM context cap at depth 45; second replacement used 32k context and smaller batch and completed |

Implementation update: `maze_navigation` now extends both key names and room names deterministically when requested depth exceeds the fixed word banks. A local smoke materialization with validation through depth 50 passed before resubmitting `3656308_0`.

Oversight update 2026-05-24 15:06 CEST: the paired maze materialization completed cleanly as `3656308_0`. The first paired maze SFT attempts then failed during the first training steps with CUDA OOM, not a data-validation error. A token audit over 200 examples per train depth found depth-10 full SFT lengths below 8192 tokens for both `logic` and `nl_exact`, so the recovery kept `data.max_length=8192` and enabled gradient checkpointing instead of truncating. Replacement SFT rows `3657088_0,1` were still running and had cleared the original OOM window by 15:05 CEST; replacement eval rows are `3657089_0,1`.

Oversight update 2026-05-24 18:45 CEST: `attribute_constraints` SFT/eval completed for both templates. Both `logic` and `nl_exact` reach OOD and depth-50 correct@16 `1.000`; `logic` grounded joint@16 is also `1.000`. The `nl_exact` NL-to-FOL validity readout is `0.000`, likely because the translator does not yet support this paired family, so do not use NL validity as a scientific conclusion here. Maze retry `3657088_0,1` failed at step 2000 during online generation eval with CUDA OOM after training had progressed; `3657089` was canceled, `scripts/slurm/sweeps/sft/paired_followup_train10_seed3407_2026-05-24.slurm` was patched to default online eval past `max_steps`, and replacements `3657738_[0-1]` plus dependent eval `3657739_[0-1]` were submitted.

Oversight update 2026-05-25 02:44 CEST: maze retry `3657738_0,1` completed cleanly after default online generation eval was disabled. Dependent sparse eval rows `3657739_0,1` are running; logs show vLLM generation progress and only the already-known tokenizer/rope warnings so far.

Oversight update 2026-05-25 06:45 CEST: maze sparse eval `3657739_0,1` is still running. Log tails show row `0` around sampled chunk `34/56` and row `1` around chunk `32/56`; higher-depth chunks are repeatedly hitting generation caps (`4096` for logic, `6144` for `nl_exact`). No Traceback, CUDA OOM, quota/no-space, or dependency failure was found in the live maze eval logs. If either row fails, inspect for timeout or generation-length pathology before resubmitting.

Oversight update 2026-05-25 10:48 CEST: maze sparse eval row `3657739_0` failed at sampled chunk `51/56` with `ValueError: The decoder prompt (length 16400) is longer than the maximum model length of 16384`. Row `3657739_1` was canceled before reaching the same likely depth-45 prompt cap. `scripts/slurm/jobs/posthoc_paired_followup_train10_eval_2026-05-24.slurm` now defaults `maze_navigation` eval to `PASSK_VLLM_MAX_MODEL_LEN=32768` and batch `64`; `bash -n` passed and replacement eval array `3659556_[0-1%2]` was submitted and is running.

Oversight update 2026-05-25 14:57 CEST: replacement maze eval `3659556_[0-1]` is still running on `a0537` with 32k context. Row `0` was around sampled chunk `59/112`; row `1` was around chunk `53/112`. The logs show long high-depth capped generations and the known tokenizer/rope warnings, but no new Traceback, CUDA OOM, quota/no-space, or dependency failure. The paired eval wrapper now guards `PASSK_JITTER_SECONDS=0`, fixing the harmless divide-by-zero startup warning seen in the current replacement logs; `bash -n scripts/slurm/jobs/posthoc_paired_followup_train10_eval_2026-05-24.slurm` passed.

Oversight update 2026-05-25 18:48 CEST: replacement maze eval `3659556_[0-1]` remains healthy and running on `a0537`. Row `0` has advanced to about sampled chunk `89/112`; row `1` is around chunk `75/112`. Logs still show high-depth capped generations, but no Traceback, CUDA OOM, quota/no-space, dependency failure, tokenizer/model-load error, or vLLM failure was found.

Oversight update 2026-05-25 22:45 CEST: replacement maze eval row `3659556_0` completed exit `0:0` and wrote `passk_eval/paired_followup_train10_sparse/sft_paired_followup_maze_navigation_logic_train1to10_10k_seed3407_passk.json`. The logic row is not a strong maze extrapolation result: train-band correct/joint@16 averages `0.750/0.750`, OOD correct@16 is `0.003`, OOD joint@16 is `0.000`, and depth-50 correct/joint@16 is `0.000/0.000`. Row `3659556_1` is still running around sampled chunk `98/112`; the current log has no Traceback, CUDA OOM, quota/no-space, dependency failure, tokenizer/model-load error, or vLLM failure beyond the known nonfatal `PASSK_JITTER_SECONDS=0` warning from the old submitted script snapshot.

Oversight update 2026-05-26 02:51 CEST: replacement maze eval row `3659556_1` also completed exit `0:0` and wrote `passk_eval/paired_followup_train10_sparse/sft_paired_followup_maze_navigation_nl_exact_train1to10_10k_seed3407_passk.json`. The completed maze train-10 comparison is negative for valid extrapolation: logic train/OOD/depth-50 correct@16 is `0.750/0.003/0.000` with joint `0.750/0.000/0.000`; `nl_exact` train/OOD/depth-50 correct@16 is `1.000/0.250/0.000`, but NL-to-FOL parse/joint@16 is `0.000` throughout. Treat the `nl_exact` answer-only OOD advantage as partial answer correctness, not a valid-proof result.

Saturation update 2026-05-25 08:44 CEST: `attribute_constraints` is saturated in the current train-10 seed-3407 pilot. Both `logic` and `nl_exact` reach train/OOD/hard-tail correct@1 and correct@16 `1.000`, and the logic run also reaches grounded joint@16 `1.000`. Treat this as evidence that this generator version is too easy for a broad follow-up sweep. Good hardening directions are higher-arity constraints, larger value domains, more confusable prerequisite pairs, multiple queried terminal slots, deeper branching dependency DAGs instead of a mostly chain-like recurrence, and stronger distractor rules that share one correct prerequisite but fail only on the other prerequisite or output.

Hardening update 2026-05-25 09:25 CEST: `attribute_constraints` was hardened and resubmitted instead of spending more compute on the saturated version. The generator no longer caps the slot count at six; it maps requested depth to `floor(depth/2)+2` compact slots, uses compact `s0`/`v10` symbols, samples a recent-window dependency DAG instead of a fixed two-previous-slot chain, and adds adversarial decoys that often share one correct prerequisite while failing on the other prerequisite or output. Local checks before submission:

```bash
python -m pytest -q tests/test_paired_synthetic_datasets.py
python scripts/data/build_paired_synthetic_dataset.py --kind attribute_constraints --output-root /tmp/attr_hard_smoke --train-rows 30 --train-max-depth 10 --val-rows-per-depth 2 --val-max-depth 50 --validate-examples -1 --chunk-size 20 --seed 3407
```

The test run passed (`9 passed`), and the smoke materialization validated through depth 50. OLMo-tokenizer audit after hardening: logic depth 10/50 total SFT tokens about `2.6k/13.4k`, NL depth 10/50 about `1.8k/8.8k`. Submitted replacement chain, updated at 09:34 CEST:

| stage | job | note |
| --- | ---: | --- |
| hard train-10 materialization | `3659338` | completed exit `0:0` after 31m53s |
| hard seed-3407 SFT | `3659339_[0-1%2]` | both rows completed exit `0:0` |
| hard sparse eval | `3659340_[0-1%2]` | completed; output under `passk_eval/paired_attribute_constraints_hard_sparse/` |

Oversight update 2026-05-25 14:50 CEST: hard `attribute_constraints` sparse eval has one completed row. `nl_exact` train correct@8 is `1.000`, OOD correct@8 is `0.806`, hard-tail correct@8 is `0.785`, depth-50 correct@8 is `0.000`, and NL-to-FOL joint@8 is `0.000`. The `logic` row is still running, so the hard-generator substrate comparison is not ready yet. Maze replacement eval `3659556_[0-1]` is still running under the 32k context setting; logs show long high-depth capped generations but no new Traceback/OOM/quota failure at the last check.

Oversight update 2026-05-25 18:56 CEST: hard `attribute_constraints` sparse eval completed both rows. The hardened family is no longer saturated: logic OOD correct/joint@8 is `0.488/0.356` and hard-tail correct/joint@8 is `0.431/0.285`; `nl_exact` OOD correct@8 is `0.806` and hard-tail correct@8 is `0.785`, but NL-to-FOL parse/joint remains `0.000`. Use correctness as a partial paired-family readout until translator coverage is improved.

OOD lm-eval update 2026-05-27 11:30 CEST: broad OOD array `3659356` completed all six paired-family rows, and GSM8K was recomputed from sample JSONL after removing raw-trace number fallback. The downstream transfer readout is mixed and task-dependent: maze `logic/nl_exact` gets strict GSM8K EM `0.107/0.586`, but mean strict LongBench F1 `0.403/0.179`; saturated attribute `logic/nl_exact` gets strict GSM8K EM `0.160/0.233`, but LongBench F1 `0.400/0.025`; hard attribute `logic/nl_exact` gets strict GSM8K EM `0.136/0.196`, but LongBench F1 `0.273/0.044`. This mirrors the main HFSA downstream pattern: NL-style traces are stronger on GSM8K arithmetic, while logic-style traces are stronger on strict context-QA under the current prompts/extractor.

## Shared Design Principles

1. Paired trace semantics.
   Each example exposes the same latent derivation in natural language and in formal logic. This makes it possible to compare logic CoT against deterministic natural-language CoT without changing the underlying problem.

2. Grounded verification.
   The verifier checks the proof against the prompt-grounded canonical premises. This prevents self-serving generated premises from receiving validity credit.

3. Distinct latent algorithms.
   The tasks intentionally instantiate different computational structures: arithmetic dependency chains, graph/state traversal, and multi-input constraint propagation.

4. Length extrapolation.
   Each generator has a natural difficulty axis. We can train on short depths and evaluate on longer depths while keeping the same latent algorithm.

5. Distractor pressure.
   Decoy facts and decoy rules are included so that solving requires selecting the applicable rule, not simply copying the first visible relation.

## Benchmark 1: `official_igsm`

Current status: fixed locally and submitted for train-10 seed-3407 SFT/eval as of 2026-05-28. Addition and subtraction chains validate in the local depth-50 smoke after the tokenizer fix above. The full 50k train-10 chain `3671601 -> 3671602 -> 3671603` completed with no validation or runtime failures. A full 3-seed, 5-train-depth paired suite was submitted later the same day as `3672195 -> 3672212 -> 3672213` and is still in SFT.

### What It Tests

`official_igsm` tests arithmetic dependency-chain reasoning. The model must identify and follow a chain of symbolic equations, substitute known values into later equations, evaluate arithmetic modulo 23, and return the queried value. This is different from graph traversal or generic Horn-clause reachability because the state being propagated is a numeric value through variable dependencies.

### Construction

For a requested depth `D`, the generator samples an official iGSM problem from the local `facebookresearch/iGSM` generator with roughly `D` operations. The official problem contains a natural-language prompt, an official solution trace, and an answer. We parse the official solution into equation-chain records:

```text
v_i = expression_i
```

where each expression may depend on earlier variables. The formal premises are exactly these parsed equation relations. The proof then proceeds by:

1. Retrieving an equation premise with `R`.
2. Substituting previously derived numeric variable values with `=E`.
3. Evaluating arithmetic modulo 23 with `MOD23`.

The `<predicates>` block is intentionally empty because iGSM is represented with equations, not predicate atoms.

### Why This Is Included

This task tests whether formal logic traces help with symbolic arithmetic dependency tracking. It is not just number extraction: irrelevant natural-language facts are present, and the queried quantity depends on following the correct equation chain. The extrapolation axis is the number of operations, dependency depth, and distractor equation density.

### Concrete Sequence

A full exact generated sequence is included in the appendix under `official_igsm`. The key formal core is:

```text
<premises>
v_e = 21
v_W = 11 + v_e
</premises>
<proof>
v_e = 21 ; R,1
v_W = 11 + v_e ; R,2
v_W = 11 + 21 ; =E,3,4
v_W = 9 ; MOD23,5
</proof>
<conclusion>
v_W = 9
</conclusion>
<answer>
9
</answer>
```

## Benchmark 2: `maze_navigation`

### What It Tests

`maze_navigation` is now a keyed, constrained graph traversal task. The model must track two pieces of state at every step:

1. Current reachable room.
2. Currently held key.

A door is traversable only when both the room and the required key match. Entering the next room reveals the next key. This makes the task more than plain graph reachability: reachability is state-conditioned.

### Construction

For depth `D`, the generator samples:

- a gold room path `r_0, ..., r_D`,
- a key path `k_0, ..., k_D`,
- blocked decoy doors at every step,
- unreachable treasure-room decoys.

The canonical initial state is:

```text
At0(r_0)
Have0(k_0)
```

At each step `t`, the gold transition is:

```text
Door(r_t,k_t,r_{t+1})
At_t(r_t) & Have_t(k_t) & Door(r_t,k_t,r_{t+1}) -> At_{t+1}(r_{t+1})
Finds(r_{t+1},k_{t+1})
At_{t+1}(r_{t+1}) & Finds(r_{t+1},k_{t+1}) -> Have_{t+1}(k_{t+1})
```

The prompt also contains decoy doors from the same current room that require keys the model does not currently hold. They are structurally plausible but blocked by the key precondition. At the end, several rooms are marked as treasures, and every treasure has a final found rule:

```text
AtD(room) & Treasure(room) -> Found(room)
```

Only the gold terminal room is both reachable at depth `D` and a treasure.

### Why It Is A Constrained Graph Task

Plain graph traversal asks whether a node is reachable along edges. Here, an edge is not merely an edge. It is an action with a precondition:

```text
current room + held key + matching door -> next room
```

The state space is therefore closer to `(room, key)` than just `room`. This tests state-conditioned reachability and update dynamics. The model has to apply the key update rule after each move; otherwise it will choose blocked decoy doors later.

### Why This Is Included

This is intended to test graph/state reasoning rather than arithmetic or slot constraints. It is still expressible in the same formal proof language, which keeps the CoT substrate controlled while changing the latent algorithm. The extrapolation axes are path length, branching factor, number of blocked doors, number of keys, and number of treasure decoys.

### Concrete Sequence

A full exact generated sequence is included in the appendix under `maze_navigation`. The key formal pattern is:

```text
At0(lantern)
Have0(yellow)
Door(lantern,yellow,indigo)
At0(lantern) & Have0(yellow) & Door(lantern,yellow,indigo) -> At1(indigo)
Finds(indigo,purple)
At1(indigo) & Finds(indigo,purple) -> Have1(purple)
...
At3(granite) & Treasure(granite) -> Found(granite)
```

Intuitively, the yellow key opens the first correct door. The next room gives the purple key, which opens the next correct door, and so on. Decoy doors are present, but their required keys are not held at that time.

## Benchmark 3: `attribute_constraints`

### What It Tests

`attribute_constraints` is now a multi-input slot-value constraint-propagation task. The model must maintain several slot values and derive new slot values only when a joint constraint matches two prerequisite slot values.

This is different from the earlier candidate-selection version. There are no `assignment_x` objects, no candidate list, and no Mastermind-style feedback. The task directly asks for the values of the slots.

### Construction

For requested depth `D`, the hardened generator samples a compact slot count:

```text
s0, ..., s_{floor(D/2)+1}
```

and hidden values from a larger compact value bank:

```text
v0, ..., vN
```

The first two slot values are given:

```text
Value(s0,v0)
Value(s1,v1)
```

For each later slot `i >= 2`, the generator samples two prerequisites from already solved slots. One prerequisite is usually the immediately previous slot and the other comes from a recent solved window, which creates a shallow dependency DAG rather than a single fixed recurrence. The generator adds one gold joint constraint:

```text
Constraint(s_a,v_a,s_b,v_b,s_i,v_i)
```

and one implication rule:

```text
Value(s_a,v_a)
& Value(s_b,v_b)
& Constraint(s_a,v_a,s_b,v_b,s_i,v_i)
-> Value(s_i,v_i)
```

The generator also adds decoy constraints with wrong prerequisite values or wrong outputs. Many decoys share one correct prerequisite with the gold rule, so simple lexical matching is not enough, but every decoy remains logically inapplicable because at least one prerequisite value is not known.

The conclusion is a conjunction of all solved slot values:

```text
Value(s0,v0) & ... & Value(sN,vN)
```

and the answer string is the ordered value tuple:

```text
v0-v1-...-vN
```

### Why This Is A Constraint-Propagation Task

The model cannot derive a new slot from one previous value alone. It must keep two prior assignments active and select the matching joint constraint. This tests conjunction-based constraint satisfaction and state propagation over a small dependency DAG. The decoy constraints make simple lexical matching insufficient: many plausible rules mention the same target slot, but only one matches the current known pair.

### Why This Is Included

This task is intended to probe multi-variable constraint propagation. It complements iGSM arithmetic and keyed graph traversal by requiring the model to maintain and compose symbolic assignments rather than numeric values or room/key states. The extrapolation axes are number of slots, value-domain size, number of decoy constraints, and rule arity.

### Concrete Sequence

A full exact generated sequence is included in the appendix under `attribute_constraints`. The key formal core is:

```text
Value(slot_0,orange)
Value(slot_1,black)
Constraint(slot_0,orange,slot_1,black,slot_2,blue)
Value(slot_0,orange) & Value(slot_1,black) & Constraint(slot_0,orange,slot_1,black,slot_2,blue) -> Value(slot_2,blue)
Constraint(slot_1,black,slot_2,blue,slot_3,green)
Value(slot_1,black) & Value(slot_2,blue) & Constraint(slot_1,black,slot_2,blue,slot_3,green) -> Value(slot_3,green)
```

The derived answer is:

```text
orange-black-blue-green
```

## Relationship To The Existing HFSA Dataset

The existing HFSA depth-scaling dataset remains the pure long-chain automaton benchmark. It is useful because it isolates length extrapolation in a very controlled formal system. However, plain maze reachability was too close to HFSA: both could be seen as simple Horn-clause path following. The revised keyed maze introduces action preconditions and key-state updates, while the revised attribute task introduces multi-input symbolic constraints. Together, the benchmark suite now covers:

- HFSA: deterministic automaton execution and length extrapolation.
- official iGSM: arithmetic variable dependency and substitution.
- keyed maze: constrained graph/state reachability.
- attribute constraints: multi-input symbolic constraint propagation.

## Extrapolation Protocol

For each benchmark, the natural experiment is:

- Train on depths `1..5`, `1..10`, `1..15`, `1..20`, `1..25`.
- Evaluate on depths `1..50`.
- Report `correct@1`, `correct@16`, grounded validity, and binned depth curves.
- Compare `logic` vs `nl_exact` traces using matched examples.

Additional extrapolation axes can be added per benchmark:

| benchmark | primary axis | secondary axes |
| --- | --- | --- |
| `official_igsm` | number of operations | distractor facts, arithmetic expression width |
| `maze_navigation` | key-constrained path length | blocked doors, key vocabulary, treasure decoys |
| `attribute_constraints` | slot count | value-domain size, decoy constraints, rule arity |

## Appendix: Exact Generated SFT Sequences

The following sequences were generated programmatically from the current code with seed `17`. They are also available at `tmp/paired_task_examples_compact_2026-05-20.md`.

# official_igsm
answer: 9
validity: format=1.0, internal=1.0, grounded=1.0, citation_free_grounded=1.0
metadata: {'dataset_family': 'official_igsm', 'depth': 2, 'official_n_op': 2, 'official_problem_text': " The number of each Tool Backpack's Diary equals 12 times as much as the sum of each Briefcase Backpack's Colored Paper and each Commuter Backpack's Construction Paper. The number of each Briefcase Backpack's Construction Paper equals 11 more than each Briefcase Backpack's Diary. The number of each Commuter Backpack's Construction Paper equals 1 more than each Briefcase Backpack's Colored Paper. The number of each Briefcase Backpack's Diary equals 21. The number of each Tool Backpack's Colored Paper equals 19 times as much as the difference of each Briefcase Backpack's Construction Paper and each Briefcase Backpack's Diary. The number of each Briefcase Backpack's Colored Paper equals 10. The number of each Commuter Backpack's Colored Paper equals 3 more than each Briefcase Backpack's Construction Paper. How many Construction Paper does Briefcase Backpack have?", 'official_solution_text': " Define Briefcase Backpack's Diary as e; so e = 21. Define Briefcase Backpack's Construction Paper as W; so W = 11 + e = 11 + 21 = 9.", 'official_answer_text': '9', 'equation_chains': [{'original_var': 'e', 'var': 'v_e', 'expr': '21', 'result': 21, 'official_text': 'e = 21'}, {'original_var': 'W', 'var': 'v_W', 'expr': '11 + v_e', 'result': 9, 'official_text': 'W = 11 + e = 11 + 21 = 9'}], 'gold_answer': '9', 'modulus': 23, 'logic_trace_valid': True}
```text
<question>
1. The number of each Tool Backpack's Diary equals 12 times as much as the sum of each Briefcase Backpack's Colored Paper and each Commuter Backpack's Construction Paper.
2. The number of each Briefcase Backpack's Construction Paper equals 11 more than each Briefcase Backpack's Diary.
3. The number of each Commuter Backpack's Construction Paper equals 1 more than each Briefcase Backpack's Colored Paper.
4. The number of each Briefcase Backpack's Diary equals 21.
5. The number of each Tool Backpack's Colored Paper equals 19 times as much as the difference of each Briefcase Backpack's Construction Paper and each Briefcase Backpack's Diary.
6. The number of each Briefcase Backpack's Colored Paper equals 10.
7. The number of each Commuter Backpack's Colored Paper equals 3 more than each Briefcase Backpack's Construction Paper.
How many Construction Paper does Briefcase Backpack have?
</question>

<formal>
<constants>
v_W = official iGSM variable W
v_e = official iGSM variable e
</constants>
<predicates>

</predicates>
<premises>
v_e = 21
v_W = 11 + v_e
</premises>
<proof>
v_e = 21 ; R,1
v_W = 11 + v_e ; R,2
v_W = 11 + 21 ; =E,3,4
v_W = 9 ; MOD23,5
</proof>
<conclusion>
v_W = 9
</conclusion>
</formal>
<answer>
9
</answer>
```

# maze_navigation
answer: granite
validity: format=1.0, internal=1.0, grounded=1.0, citation_free_grounded=1.0
metadata: {'dataset_family': 'maze_navigation', 'task_structure': 'keyed_constrained_graph', 'depth': 3, 'start': 'lantern', 'gold_path': ['lantern', 'indigo', 'pearl', 'granite'], 'key_path': ['yellow', 'purple', 'white', 'green'], 'blocked_edges': [{'step': 0, 'from_room': 'lantern', 'required_key': 'white', 'to_room': 'silver'}, {'step': 0, 'from_room': 'lantern', 'required_key': 'green', 'to_room': 'timber'}, {'step': 0, 'from_room': 'lantern', 'required_key': 'orange', 'to_room': 'aurora'}, {'step': 0, 'from_room': 'lantern', 'required_key': 'blue', 'to_room': 'ochre'}, {'step': 1, 'from_room': 'indigo', 'required_key': 'red', 'to_room': 'heather'}, {'step': 1, 'from_room': 'indigo', 'required_key': 'orange', 'to_room': 'keystone'}, {'step': 1, 'from_room': 'indigo', 'required_key': 'white', 'to_room': 'willow'}, {'step': 1, 'from_room': 'indigo', 'required_key': 'yellow', 'to_room': 'citadel'}, {'step': 2, 'from_room': 'pearl', 'required_key': 'black', 'to_room': 'ruby'}, {'step': 2, 'from_room': 'pearl', 'required_key': 'blue', 'to_room': 'umber'}, {'step': 2, 'from_room': 'pearl', 'required_key': 'green', 'to_room': 'prairie'}, {'step': 2, 'from_room': 'pearl', 'required_key': 'silver', 'to_room': 'estate'}], 'treasure_rooms': ['laurel', 'nectar', 'forest', 'granite', 'linen'], 'unreachable_treasure_rooms': ['forest', 'linen', 'nectar', 'laurel'], 'requires_key_tracking': True, 'solution_rule_for_all_treasures': True, 'gold_answer': 'granite', 'logic_trace_valid': True}
```text
<question>
1. The explorer starts in room lantern.
2. The explorer initially holds the yellow key.
3. There is a door from lantern to silver that requires the white key.
4. If the explorer is in lantern after 0 moves, has the white key, and the matching door leads to silver, then silver is reachable after 1 moves.
5. There is a door from lantern to aurora that requires the orange key.
6. If the explorer is in lantern after 0 moves, has the orange key, and the matching door leads to aurora, then aurora is reachable after 1 moves.
7. There is a door from lantern to indigo that requires the yellow key.
8. If the explorer is in lantern after 0 moves, has the yellow key, and the matching door leads to indigo, then indigo is reachable after 1 moves.
9. Room indigo contains the purple key.
10. If the explorer reaches indigo after 1 moves and indigo contains the purple key, then the explorer has the purple key after 1 moves.
11. There is a door from lantern to timber that requires the green key.
12. If the explorer is in lantern after 0 moves, has the green key, and the matching door leads to timber, then timber is reachable after 1 moves.
13. There is a door from lantern to ochre that requires the blue key.
14. If the explorer is in lantern after 0 moves, has the blue key, and the matching door leads to ochre, then ochre is reachable after 1 moves.
15. There is a door from indigo to heather that requires the red key.
16. If the explorer is in indigo after 1 moves, has the red key, and the matching door leads to heather, then heather is reachable after 2 moves.
17. There is a door from indigo to citadel that requires the yellow key.
18. If the explorer is in indigo after 1 moves, has the yellow key, and the matching door leads to citadel, then citadel is reachable after 2 moves.
19. There is a door from indigo to pearl that requires the purple key.
20. If the explorer is in indigo after 1 moves, has the purple key, and the matching door leads to pearl, then pearl is reachable after 2 moves.
21. Room pearl contains the white key.
22. If the explorer reaches pearl after 2 moves and pearl contains the white key, then the explorer has the white key after 2 moves.
23. There is a door from indigo to keystone that requires the orange key.
24. If the explorer is in indigo after 1 moves, has the orange key, and the matching door leads to keystone, then keystone is reachable after 2 moves.
25. There is a door from indigo to willow that requires the white key.
26. If the explorer is in indigo after 1 moves, has the white key, and the matching door leads to willow, then willow is reachable after 2 moves.
27. There is a door from pearl to prairie that requires the green key.
28. If the explorer is in pearl after 2 moves, has the green key, and the matching door leads to prairie, then prairie is reachable after 3 moves.
29. There is a door from pearl to estate that requires the silver key.
30. If the explorer is in pearl after 2 moves, has the silver key, and the matching door leads to estate, then estate is reachable after 3 moves.
31. There is a door from pearl to umber that requires the blue key.
32. If the explorer is in pearl after 2 moves, has the blue key, and the matching door leads to umber, then umber is reachable after 3 moves.
33. There is a door from pearl to granite that requires the white key.
34. If the explorer is in pearl after 2 moves, has the white key, and the matching door leads to granite, then granite is reachable after 3 moves.
35. Room granite contains the green key.
36. If the explorer reaches granite after 3 moves and granite contains the green key, then the explorer has the green key after 3 moves.
37. There is a door from pearl to ruby that requires the black key.
38. If the explorer is in pearl after 2 moves, has the black key, and the matching door leads to ruby, then ruby is reachable after 3 moves.
39. Room laurel contains a marked treasure.
40. Room nectar contains a marked treasure.
41. Room forest contains a marked treasure.
42. Room granite contains a marked treasure.
43. Room linen contains a marked treasure.
44. If room laurel is reachable after exactly 3 key-constrained moves and contains a treasure, then the treasure in laurel is found.
45. If room nectar is reachable after exactly 3 key-constrained moves and contains a treasure, then the treasure in nectar is found.
46. If room forest is reachable after exactly 3 key-constrained moves and contains a treasure, then the treasure in forest is found.
47. If room granite is reachable after exactly 3 key-constrained moves and contains a treasure, then the treasure in granite is found.
48. If room linen is reachable after exactly 3 key-constrained moves and contains a treasure, then the treasure in linen is found.
The rooms form a locked maze. The explorer may use only doors whose key they currently hold, and entering a room may reveal the next key. Which marked treasure room is reachable after exactly 3 moves?
</question>

<formal>
<constants>
aurora = maze room aurora
citadel = maze room citadel
estate = maze room estate
forest = maze room forest
granite = maze room granite
heather = maze room heather
indigo = maze room indigo
keystone = maze room keystone
lantern = maze room lantern
laurel = maze room laurel
linen = maze room linen
nectar = maze room nectar
ochre = maze room ochre
pearl = maze room pearl
prairie = maze room prairie
ruby = maze room ruby
silver = maze room silver
timber = maze room timber
umber = maze room umber
willow = maze room willow
black = maze key black
blue = maze key blue
green = maze key green
orange = maze key orange
purple = maze key purple
red = maze key red
silver = maze key silver
white = maze key white
yellow = maze key yellow
</constants>
<predicates>
AtN(x): the explorer can be at room x after N moves
HaveN(x): the explorer has key x after N moves
Door(x,y,z): there is a door from room x to room z requiring key y
Finds(x,y): room x contains key y
Treasure(x): room x contains a marked treasure
Found(x): the reachable marked treasure is in room x
</predicates>
<premises>
At0(lantern)
Have0(yellow)
Door(lantern,white,silver)
At0(lantern) & Have0(white) & Door(lantern,white,silver) -> At1(silver)
Door(lantern,orange,aurora)
At0(lantern) & Have0(orange) & Door(lantern,orange,aurora) -> At1(aurora)
Door(lantern,yellow,indigo)
At0(lantern) & Have0(yellow) & Door(lantern,yellow,indigo) -> At1(indigo)
Finds(indigo,purple)
At1(indigo) & Finds(indigo,purple) -> Have1(purple)
Door(lantern,green,timber)
At0(lantern) & Have0(green) & Door(lantern,green,timber) -> At1(timber)
Door(lantern,blue,ochre)
At0(lantern) & Have0(blue) & Door(lantern,blue,ochre) -> At1(ochre)
Door(indigo,red,heather)
At1(indigo) & Have1(red) & Door(indigo,red,heather) -> At2(heather)
Door(indigo,yellow,citadel)
At1(indigo) & Have1(yellow) & Door(indigo,yellow,citadel) -> At2(citadel)
Door(indigo,purple,pearl)
At1(indigo) & Have1(purple) & Door(indigo,purple,pearl) -> At2(pearl)
Finds(pearl,white)
At2(pearl) & Finds(pearl,white) -> Have2(white)
Door(indigo,orange,keystone)
At1(indigo) & Have1(orange) & Door(indigo,orange,keystone) -> At2(keystone)
Door(indigo,white,willow)
At1(indigo) & Have1(white) & Door(indigo,white,willow) -> At2(willow)
Door(pearl,green,prairie)
At2(pearl) & Have2(green) & Door(pearl,green,prairie) -> At3(prairie)
Door(pearl,silver,estate)
At2(pearl) & Have2(silver) & Door(pearl,silver,estate) -> At3(estate)
Door(pearl,blue,umber)
At2(pearl) & Have2(blue) & Door(pearl,blue,umber) -> At3(umber)
Door(pearl,white,granite)
At2(pearl) & Have2(white) & Door(pearl,white,granite) -> At3(granite)
Finds(granite,green)
At3(granite) & Finds(granite,green) -> Have3(green)
Door(pearl,black,ruby)
At2(pearl) & Have2(black) & Door(pearl,black,ruby) -> At3(ruby)
Treasure(laurel)
Treasure(nectar)
Treasure(forest)
Treasure(granite)
Treasure(linen)
At3(laurel) & Treasure(laurel) -> Found(laurel)
At3(nectar) & Treasure(nectar) -> Found(nectar)
At3(forest) & Treasure(forest) -> Found(forest)
At3(granite) & Treasure(granite) -> Found(granite)
At3(linen) & Treasure(linen) -> Found(linen)
</premises>
<proof>
At0(lantern) ; R,1
Have0(yellow) ; R,2
Door(lantern,yellow,indigo) ; R,7
At0(lantern) & Have0(yellow) ; ∧I,49,50
At0(lantern) & Have0(yellow) & Door(lantern,yellow,indigo) ; ∧I,52,51
At1(indigo) ; ->E,8,53
Finds(indigo,purple) ; R,9
At1(indigo) & Finds(indigo,purple) ; ∧I,54,55
Have1(purple) ; ->E,10,56
Door(indigo,purple,pearl) ; R,19
At1(indigo) & Have1(purple) ; ∧I,54,57
At1(indigo) & Have1(purple) & Door(indigo,purple,pearl) ; ∧I,59,58
At2(pearl) ; ->E,20,60
Finds(pearl,white) ; R,21
At2(pearl) & Finds(pearl,white) ; ∧I,61,62
Have2(white) ; ->E,22,63
Door(pearl,white,granite) ; R,33
At2(pearl) & Have2(white) ; ∧I,61,64
At2(pearl) & Have2(white) & Door(pearl,white,granite) ; ∧I,66,65
At3(granite) ; ->E,34,67
Finds(granite,green) ; R,35
At3(granite) & Finds(granite,green) ; ∧I,68,69
Have3(green) ; ->E,36,70
Treasure(granite) ; R,42
At3(granite) & Treasure(granite) ; ∧I,68,72
Found(granite) ; ->E,47,73
</proof>
<conclusion>
Found(granite)
</conclusion>
</formal>
<answer>
granite
</answer>
```

# attribute_constraints
answer: orange-black-blue-green
validity: format=1.0, internal=1.0, grounded=1.0, citation_free_grounded=1.0
metadata: {'dataset_family': 'attribute_constraints', 'task_structure': 'multi_input_slot_constraint_dag', 'depth': 4, 'slot_count': 4, 'base_slot_count': 2, 'palette': ['red', 'blue', 'green', 'yellow', 'white', 'black', 'orange'], 'slots': [{'slot': 'slot_0', 'value': 'orange'}, {'slot': 'slot_1', 'value': 'black'}, {'slot': 'slot_2', 'value': 'blue'}, {'slot': 'slot_3', 'value': 'green'}], 'constraints': [{'target_index': 2, 'dep_a': 0, 'dep_b': 1, 'slot_a': 'slot_0', 'value_a': 'orange', 'slot_b': 'slot_1', 'value_b': 'black', 'target_slot': 'slot_2', 'target_value': 'blue'}, {'target_index': 3, 'dep_a': 1, 'dep_b': 2, 'slot_a': 'slot_1', 'value_a': 'black', 'slot_b': 'slot_2', 'value_b': 'blue', 'target_slot': 'slot_3', 'target_value': 'green'}], 'decoy_constraints': [{'target_index': 2, 'slot_a': 'slot_0', 'value_a': 'green', 'slot_b': 'slot_1', 'value_b': 'blue', 'target_slot': 'slot_2', 'target_value': 'yellow'}, {'target_index': 2, 'slot_a': 'slot_0', 'value_a': 'red', 'slot_b': 'slot_1', 'value_b': 'black', 'target_slot': 'slot_2', 'target_value': 'white'}, {'target_index': 2, 'slot_a': 'slot_0', 'value_a': 'orange', 'slot_b': 'slot_1', 'value_b': 'red', 'target_slot': 'slot_2', 'target_value': 'orange'}, {'target_index': 2, 'slot_a': 'slot_0', 'value_a': 'yellow', 'slot_b': 'slot_1', 'value_b': 'green', 'target_slot': 'slot_2', 'target_value': 'yellow'}, {'target_index': 3, 'slot_a': 'slot_1', 'value_a': 'yellow', 'slot_b': 'slot_2', 'value_b': 'white', 'target_slot': 'slot_3', 'target_value': 'black'}, {'target_index': 3, 'slot_a': 'slot_1', 'value_a': 'black', 'slot_b': 'slot_2', 'value_b': 'green', 'target_slot': 'slot_3', 'target_value': 'yellow'}, {'target_index': 3, 'slot_a': 'slot_1', 'value_a': 'white', 'slot_b': 'slot_2', 'value_b': 'green', 'target_slot': 'slot_3', 'target_value': 'orange'}, {'target_index': 3, 'slot_a': 'slot_1', 'value_a': 'black', 'slot_b': 'slot_2', 'value_b': 'orange', 'target_slot': 'slot_3', 'target_value': 'blue'}], 'gold_answer': 'orange-black-blue-green', 'logic_trace_valid': True, 'grounded_validity_supported': True}
```text
<question>
1. slot_0 has value orange.
2. slot_1 has value black.
3. The joint constraint says: if slot_0 is orange and slot_1 is black, then slot_2 is blue.
4. If both prerequisite slot values hold and the matching joint constraint is present, then slot_2 has blue.
5. A decoy joint constraint says: if slot_0 is green and slot_1 is blue, then slot_2 is yellow.
6. If the decoy prerequisite values held and the decoy constraint applied, then slot_2 would be yellow.
7. A decoy joint constraint says: if slot_0 is red and slot_1 is black, then slot_2 is white.
8. If the decoy prerequisite values held and the decoy constraint applied, then slot_2 would be white.
9. A decoy joint constraint says: if slot_0 is orange and slot_1 is red, then slot_2 is orange.
10. If the decoy prerequisite values held and the decoy constraint applied, then slot_2 would be orange.
11. A decoy joint constraint says: if slot_0 is yellow and slot_1 is green, then slot_2 is yellow.
12. If the decoy prerequisite values held and the decoy constraint applied, then slot_2 would be yellow.
13. The joint constraint says: if slot_1 is black and slot_2 is blue, then slot_3 is green.
14. If both prerequisite slot values hold and the matching joint constraint is present, then slot_3 has green.
15. A decoy joint constraint says: if slot_1 is yellow and slot_2 is white, then slot_3 is black.
16. If the decoy prerequisite values held and the decoy constraint applied, then slot_3 would be black.
17. A decoy joint constraint says: if slot_1 is black and slot_2 is green, then slot_3 is yellow.
18. If the decoy prerequisite values held and the decoy constraint applied, then slot_3 would be yellow.
19. A decoy joint constraint says: if slot_1 is white and slot_2 is green, then slot_3 is orange.
20. If the decoy prerequisite values held and the decoy constraint applied, then slot_3 would be orange.
21. A decoy joint constraint says: if slot_1 is black and slot_2 is orange, then slot_3 is blue.
22. If the decoy prerequisite values held and the decoy constraint applied, then slot_3 would be blue.
Starting from the given slot values, apply the joint constraints. Which values fill all slots?
</question>

<formal>
<constants>
slot_0 = attribute slot 0
slot_1 = attribute slot 1
slot_2 = attribute slot 2
slot_3 = attribute slot 3
black = attribute value black
blue = attribute value blue
green = attribute value green
orange = attribute value orange
red = attribute value red
white = attribute value white
yellow = attribute value yellow
</constants>
<predicates>
Value(x,y): slot x has value y
Constraint(x,y,z,w,u,v): values y and w at slots x and z jointly force value v at slot u
</predicates>
<premises>
Value(slot_0,orange)
Value(slot_1,black)
Constraint(slot_0,orange,slot_1,black,slot_2,blue)
Value(slot_0,orange) & Value(slot_1,black) & Constraint(slot_0,orange,slot_1,black,slot_2,blue) -> Value(slot_2,blue)
Constraint(slot_0,green,slot_1,blue,slot_2,yellow)
Value(slot_0,green) & Value(slot_1,blue) & Constraint(slot_0,green,slot_1,blue,slot_2,yellow) -> Value(slot_2,yellow)
Constraint(slot_0,red,slot_1,black,slot_2,white)
Value(slot_0,red) & Value(slot_1,black) & Constraint(slot_0,red,slot_1,black,slot_2,white) -> Value(slot_2,white)
Constraint(slot_0,orange,slot_1,red,slot_2,orange)
Value(slot_0,orange) & Value(slot_1,red) & Constraint(slot_0,orange,slot_1,red,slot_2,orange) -> Value(slot_2,orange)
Constraint(slot_0,yellow,slot_1,green,slot_2,yellow)
Value(slot_0,yellow) & Value(slot_1,green) & Constraint(slot_0,yellow,slot_1,green,slot_2,yellow) -> Value(slot_2,yellow)
Constraint(slot_1,black,slot_2,blue,slot_3,green)
Value(slot_1,black) & Value(slot_2,blue) & Constraint(slot_1,black,slot_2,blue,slot_3,green) -> Value(slot_3,green)
Constraint(slot_1,yellow,slot_2,white,slot_3,black)
Value(slot_1,yellow) & Value(slot_2,white) & Constraint(slot_1,yellow,slot_2,white,slot_3,black) -> Value(slot_3,black)
Constraint(slot_1,black,slot_2,green,slot_3,yellow)
Value(slot_1,black) & Value(slot_2,green) & Constraint(slot_1,black,slot_2,green,slot_3,yellow) -> Value(slot_3,yellow)
Constraint(slot_1,white,slot_2,green,slot_3,orange)
Value(slot_1,white) & Value(slot_2,green) & Constraint(slot_1,white,slot_2,green,slot_3,orange) -> Value(slot_3,orange)
Constraint(slot_1,black,slot_2,orange,slot_3,blue)
Value(slot_1,black) & Value(slot_2,orange) & Constraint(slot_1,black,slot_2,orange,slot_3,blue) -> Value(slot_3,blue)
</premises>
<proof>
Value(slot_0,orange) ; R,1
Value(slot_1,black) ; R,2
Constraint(slot_0,orange,slot_1,black,slot_2,blue) ; R,3
Value(slot_0,orange) & Value(slot_1,black) ; ∧I,23,24
Value(slot_0,orange) & Value(slot_1,black) & Constraint(slot_0,orange,slot_1,black,slot_2,blue) ; ∧I,26,25
Value(slot_2,blue) ; ->E,4,27
Constraint(slot_1,black,slot_2,blue,slot_3,green) ; R,13
Value(slot_1,black) & Value(slot_2,blue) ; ∧I,24,28
Value(slot_1,black) & Value(slot_2,blue) & Constraint(slot_1,black,slot_2,blue,slot_3,green) ; ∧I,30,29
Value(slot_3,green) ; ->E,14,31
Value(slot_0,orange) & Value(slot_1,black) ; ∧I,23,24
Value(slot_0,orange) & Value(slot_1,black) & Value(slot_2,blue) ; ∧I,33,28
Value(slot_0,orange) & Value(slot_1,black) & Value(slot_2,blue) & Value(slot_3,green) ; ∧I,34,32
</proof>
<conclusion>
Value(slot_0,orange) & Value(slot_1,black) & Value(slot_2,blue) & Value(slot_3,green)
</conclusion>
</formal>
<answer>
orange-black-blue-green
</answer>
```
