# Hard-v3 Pass@k Analysis (2026-05-03)

Inputs:
- Metrics: `/home/atuin/c107fa/c107fa12/synthetic-RLVL/passk_eval/hard_v3/*_passk.json`
- Representative sample logs: `/home/atuin/c107fa/c107fa12/synthetic-RLVL/passk_eval/hard_v3/*_samples.jsonl`
- Analysis script: `scripts/analysis/plot_hard_v3_passk_ablation.py`
- Outputs: `plots/hard_v3_passk/`

All five hard-v3 reward schemas completed for seeds `3407`, `3408`, and `3409` at `global_step_1500`.

## Generated Outputs

- Band pass@k plots for all hard-v3 ablations: `plots/hard_v3_passk/hard_v3_*_{correct,valid,joint,valid_given_correct,invalid_but_correct,format,syntactic}.{png,pdf}`
- Step-wise greedy curves: `plots/hard_v3_passk/hard_v3_greedy_steps_*.{png,pdf}`
- Step-wise sampled curves: `plots/hard_v3_passk/hard_v3_sampled_steps_*_k{1,8,64}.{png,pdf}`
- Mean/std summary table: `plots/hard_v3_passk/hard_v3_passk_summary.tsv`
- Per-seed table: `plots/hard_v3_passk/hard_v3_passk_per_seed.tsv`
- Step tables:
  - `plots/hard_v3_passk/hard_v3_greedy_step_summary.tsv`
  - `plots/hard_v3_passk/hard_v3_sampled_step_summary.tsv`
- Sample inspection report: `plots/hard_v3_passk/hard_v3_sample_report.md`

## Main Result

Hard-v3 is not solved in the sense we care about. Answer correctness remains high, but formal proof validity collapses for long OOD chains. The model often reaches the right answer through invalid formal reasoning.

Validity rewards improve pass@1 validity and joint correctness+validity, especially OOD. They do not materially improve answer correctness, which is already near saturated.

At high k, the no-validity baseline is still competitive or slightly better on valid/joint pass@64, likely because sampling 64 completions gives enough chances to find a valid proof even without an explicit validity reward. The important signal is therefore pass@1 and low-k pass@k, not only pass@64.

## Band-Level Findings

ID steps `1-10`:
- Correctness is saturated for every reward: pass@1 is about `0.999`, pass@64 is `1.000`.
- Validity reward matters at pass@1: baseline valid pass@1 is `0.734`, while validity-reward variants are about `0.799-0.803`.
- At pass@64, all rewards are saturated at valid/joint `0.995`.

OOD steps `11-20`:
- Correct pass@1 is almost identical: about `0.919-0.922`.
- Valid pass@1 improves from `0.328` for baseline to `0.365-0.373` with validity rewards.
- Joint pass@1 improves from `0.324` for baseline to `0.361-0.369`.
- Best OOD pass@1 joint is additive binary validity: `correct + valid + 0.1 format` at `0.369`.
- At pass@64, baseline is slightly best on valid/joint:
  - baseline joint pass@64: `0.938`
  - best validity-reward joint pass@64: about `0.923`

Hard tail steps `15-20`:
- Correct pass@1 remains high: about `0.898-0.901`.
- Valid pass@1 improves from `0.219` baseline to `0.232-0.242` with validity rewards.
- Joint pass@1 improves from `0.215` baseline to `0.227-0.237`.
- Best hard-tail pass@1 joint is multiplicative line validity: `correct * line-valid + 0.1 format` at `0.237`, but the difference from additive binary validity is small.
- At pass@64, baseline again has the best valid/joint:
  - baseline joint pass@64: `0.939`
  - validity-reward variants: `0.903-0.911`

## Step-Level Findings

Step 10:
- Correctness is `1.000` for all rewards.
- Sampled valid pass@1:
  - baseline: `0.633`
  - validity variants: `0.723`
- This is the clearest in-distribution benefit of validity reward.

Step 15:
- Sampled correct pass@1 is still high: `0.959-0.965`.
- Sampled valid pass@1:
  - baseline: `0.416`
  - best validity variant: `0.467`
- Invalid-but-correct remains very high: roughly `0.50-0.55`.

Step 20:
- Sampled correct pass@1 is still high: `0.763-0.776`.
- Sampled valid pass@1 is very low:
  - baseline: `0.056`
  - validity variants: `0.057-0.063`
- Sampled joint pass@1 is only `0.052-0.058`.
- Invalid-but-correct is dominant: about `0.71-0.72`.
- At pass@64, correctness reaches `1.000` for all schemas, but joint pass@64 ranges only from `0.683` to `0.817`.

## Reward Comparisons

Baseline `correct + 0.1 format`:
- Strongest high-k joint/valid performance.
- Worst pass@1 validity and joint performance.
- Highest invalid-but-correct rate at low k.

Additive binary validity `correct + valid + 0.1 format`:
- Best OOD pass@1 joint and valid-given-correct.
- Good default if the objective is to improve greedy/low-k formal proof quality.
- Does not improve answer correctness.

Multiplicative binary validity `correct * valid + 0.1 format`:
- Similar to additive binary validity.
- Slightly better in some step-20 sampled metrics, but not consistently better than additive binary.

Additive line-valid `correct + line-valid + 0.1 format`:
- Does not clearly beat binary validity.
- Slightly weaker on OOD/hard-tail low-k joint than binary validity.

Multiplicative line-valid `correct * line-valid + 0.1 format`:
- Best hard-tail pass@1 joint by a small margin.
- Not better at pass@64.

## Sample Inspection

The saved sample JSONL files contain only the first `collect_samples=8` greedy rows from each eval, and because eval records are ordered by step, all saved examples are step-1 generations. They are all clean and valid. They are useful as sanity checks that the models produce the expected `<formal>...</formal><answer>...</answer>` format, but they are not useful for long-step failure analysis.

This has been fixed for future evals: sample logging now round-robins over validation steps instead of taking the first `collect_samples` rows. With `collect_samples=8` over steps 1-20, future files will include steps 1-8; with larger values they will cover the OOD tail.

To inspect long-step failures qualitatively, rerun a targeted sample eval with `--step-min 20 --step-max 20 --collect-samples` high enough, or change the sampler logging policy to stratify by step.

## Interpretation

Validity reward works in the narrow sense: it reduces invalid-but-correct behavior at pass@1 and improves formal validity on ID/OOD steps. However, it does not produce the desired transfer from formal validity to answer correctness because correctness is already high and because the model can often guess or shortcut the final answer while producing invalid proof traces.

For scientific reporting, emphasize:
- `joint_pass@k` and `valid_given_correct@k`, not only `correct_pass@k`.
- low-k metrics, especially pass@1/pass@8, because pass@64 can hide reward-shaping differences.
- step-wise curves, especially step 15 and step 20, because band averages obscure the sharp validity collapse.
