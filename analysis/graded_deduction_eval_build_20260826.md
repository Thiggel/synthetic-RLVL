# Graded-depth NL deduction eval — build notes (2026-08-26)

New lm_eval task family `synthrlvl_deduction_graded`: per-depth deduction accuracy,
plain prose QA, no trained scaffold (no <formal>/Solution:/Final answer: envelope).

## Data

Output root: `/home/vault/c107fa/c107fa12/synthetic-RLVL/datasets/graded_deduction_eval_20260826/`
(manifest.json records everything; seed everywhere: **20260826**)

### Tier A — ProofWriter OWA test (external NL deduction)
- Raw data had been deleted; re-downloaded the official zip
  (aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip) to
  `/home/atuin/c107fa/c107fa12/synthetic-RLVL/data/raw/proofwriter/` ($WORK, not vault/hpc-home).
- Source: OWA/depth-{0,1,2,3,5}/meta-test.jsonl (TEST splits — never ingested by
  build_real_logic_corpus.py, which used train+dev only).
- Pooled all five dirs, bucketed each question by its **QDep** annotation
  (a depth-N dir contains QDep<N questions), kept buckets {0,1,2,3,5},
  shuffled with random.Random(20260826+depth), capped at 500/bucket.
- Fields: {context (theory prose), question, answer in {true,false,unknown}, depth, source_id, source_dir}.
- Gold answer distribution (d: true/false/unknown): 0: 95/131/274, 1: 133/122/245,
  2: 192/190/118, 3: 199/214/87, 5: 237/240/23.

### Tier B — deep BranchProof prose QA (held out, freshly generated)
- Generator: `synthetic_dataset.LogicDatasetGenerator(DatasetConfig(...))`, mirroring the
  branchproof_unique_v2_20260710 manifest: difficulty=hard_fsa_schema, distractor_ratio=0.5,
  branching_factor=4, shortcut_rate=0.0, require_unique_solution=True — but with **seed 20260826**
  (training-era materializations, incl. the band15/band25 sources of the mixdepth SFT traces
  in reasoning_mixture_20260821, used seed 3407), so no collision with any training data.
- 200 problems per depth in {5,10,15,20,25}, indices 0..199.
- Rendering: nl_exact premise prose only (premises_nl, unnumbered, one per line, same
  transformation as the nl_exact template's _join_unnumbered), then question_nl, no
  proof/derivation, no training envelope. answer = ex.answer (unique, e.g. a state word).

## Code

- Build script: `scripts/data/build_graded_deduction_eval.py` (argparse; deterministic;
  prints per-depth counts + prompt token-length stats using the doc_to_text functions
  and the Qwen2.5 tokenizer from the control checkpoint).
- Task yamls: `lm_eval_tasks/synthrlvl_ood/synthrlvl_deduction_pw_d{0,1,2,3,5}.yaml` and
  `synthrlvl_deduction_bp_d{5,10,15,20,25}.yaml` (dataset_path: json + data_files pointing
  at the vault jsonls; generate_until; until: "\n"; greedy; max_gen_toks 16 (PW) / 64 (BP)).
- New utils (appended to `lm_eval_tasks/synthrlvl_ood/utils.py`, existing functions untouched):
  - doc_to_text_deduction_pw: context + "Question: ..." + one-word True/False/Unknown
    instruction, ending "Answer:".
  - process_deduction_pw: first true/false/unknown token in the response, case-insensitive.
  - doc_to_text_deduction_bp: premises + question + "Give only the final answer.\nAnswer:".
  - process_deduction_bp: first response line, normalize_answer (lowercase, strip
    punctuation/articles), exact match vs gold.
- Smoke slurm: `scripts/slurm/jobs/graded_deduction_smoke_2026-08-26.slurm`.

## Token-length stats (Qwen2.5 tokenizer, prompt only; guard vs silent truncation)

| file | n | p50 | p99 | max |
|---|---|---|---|---|
| proofwriter_owa_d0 | 500 | 136 | 265 | 278 |
| proofwriter_owa_d1 | 500 | 141 | 262 | 298 |
| proofwriter_owa_d2 | 500 | 144 | 264 | 305 |
| proofwriter_owa_d3 | 500 | 147 | 271 | 290 |
| proofwriter_owa_d5 | 500 | 162 | 277 | 305 |
| branchproof_nl_d5 | 200 | 630 | 652 | 655 |
| branchproof_nl_d10 | 200 | 1247 | 1277 | 1283 |
| branchproof_nl_d15 | 200 | 1948 | 1983 | 1994 |
| branchproof_nl_d20 | 200 | 2650 | 2682 | 2695 |
| branchproof_nl_d25 | 200 | 3352 | 3380 | 3392 |

Max prompt 3392 + max_gen_toks 64 + chat template (~30) << 8192: comfortable margin.

## Smoke test (job 4107554, COMPLETED)

Checkpoint: post_sft_reasoning_mixture_20260821/qwen25_7b_mixdepth_control_seed3407/final,
vllm, --limit 8, --apply-chat-template, max_model_len 8192.
Results: `/home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results/graded_deduction_smoke_20260826/`

exact_match (n=8 each; extracted_nonempty = 1.0 on all 10 tasks):
- PW: d0 0.500, d1 0.875, d2 0.500, d3 0.375, d5 0.500
- BP: d5 0.000, d10 0.000, d15 0.000, d20 0.000, d25 0.000

Raw-sample inspection (per analysis discipline): responses are clean single tokens/words
("True"; "north"), extraction works. PW d0 misses are the model answering "True" on
gold-unknown items (d0 bucket is 55% unknown — OWA unknown-calibration failure, not an
evaluator artifact). BP 0/8: the control model (no synthetic traces) answers with a
plausible attribute word without reasoning; gold is a different state word — consistent
with ~chance (~1/19) for a no-CoT direct answer. Expected for the control condition;
the graded readout across trained conditions is the point of the eval. Worth re-checking
against a band25 nl_exact checkpoint before interpreting BP levels.

## Caveats / decisions

- $WORK on alex is /home/atuin (not woody); raw ProofWriter and nothing else went there.
  hpc home is over soft quota — nothing was written there beyond repo code.
- Tokenizer load emits a transformers "incorrect regex / fix_mistral_regex" warning for the
  Qwen checkpoint tokenizer; benign for length measurement.
- BP question wording is the generator's own question_nl ("Which state applies to cN?");
  answers are single state words, matched after normalize_answer.
- Nothing committed to git (per instruction).

## Follow-up (2026-08-26): CoT BP variants + full matrix

- Added `lm_eval_tasks/synthrlvl_ood/synthrlvl_deduction_bp_cot_d{5,10,15,20,25}.yaml`:
  same jsonls, CoT-prompted ("Reason step by step, then give the final answer on its own
  last line as \"Answer: <answer>\""), max_gen_toks 2048 (compromise: d25 latent proofs run
  ~3.4k rendered-with-proof tokens, so 2048 of reasoning may truncate the deepest chains —
  noted in the yaml comments), until: [] (natural EOS only). Token budget verified:
  max prompt 3392 + 2048 gen + chat template << 8192.
- New utils (appended, existing untouched): `doc_to_text_deduction_bp_cot`,
  `process_deduction_bp_cot` (LAST "Answer:"/"Final answer:" line, else last non-empty
  line; normalize_answer match; also reports tag_found). Unit-checked: multi-answer
  response extracts the last one.
- Full matrix submitted: **job 4107575** (array 0-9%4),
  `scripts/slurm/jobs/graded_deduction_full_2026-08-26.slurm` — all 15 tasks, no limit,
  10 mixdepth checkpoints. Results per model:
  - /home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results/qwen25_mixdepth_graded_deduction_20260826/qwen25_7b_mixdepth_control_seed3407
  - /home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results/qwen25_mixdepth_graded_deduction_20260826/qwen25_7b_mixdepth_control_seed3408
  - /home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results/qwen25_mixdepth_graded_deduction_20260826/qwen25_7b_mixdepth_logic_band15_seed3407
  - /home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results/qwen25_mixdepth_graded_deduction_20260826/qwen25_7b_mixdepth_logic_band15_seed3408
  - /home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results/qwen25_mixdepth_graded_deduction_20260826/qwen25_7b_mixdepth_nl_exact_band15_seed3407
  - /home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results/qwen25_mixdepth_graded_deduction_20260826/qwen25_7b_mixdepth_nl_exact_band15_seed3408
  - /home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results/qwen25_mixdepth_graded_deduction_20260826/qwen25_7b_mixdepth_logic_band25_seed3407
  - /home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results/qwen25_mixdepth_graded_deduction_20260826/qwen25_7b_mixdepth_logic_band25_seed3408
  - /home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results/qwen25_mixdepth_graded_deduction_20260826/qwen25_7b_mixdepth_nl_exact_band25_seed3407
  - /home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results/qwen25_mixdepth_graded_deduction_20260826/qwen25_7b_mixdepth_nl_exact_band25_seed3408
