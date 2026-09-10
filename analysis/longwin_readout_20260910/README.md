# Long-window readout, gruenau, 2026-09-10 (evening session)

## 1. ProofWriter: where the gain actually lives

Script `scripts/analysis/proofwriter_error_analysis.py`, full output in
`analysis/pw_error_analysis_20260910.txt`, both instruction-tuning seeds, 2,500
items each, joined to ProofWriter's own question metadata (strategy, QLen).

Polarity of the claim x gold label, decidable items, seed 3407 / 3408:

| cell | n | Control | LongDoc | Formal | English | Condensed |
|---|---|---|---|---|---|---|
| negated claim, gold false | 785 | .345 / .348 | .363 / .357 | .531 / .540 | .680 / .676 | .494 / .510 |
| negated claim, gold true | 67 | .940 / .940 | .955 / .940 | .910 / .925 | .910 / .896 | .925 / .925 |
| positive claim, gold false | 112 | .857 / .875 | .866 / .857 | .875 / .875 | .875 / .884 | .902 / .902 |
| positive claim, gold true | 789 | .715 / .696 | .707 / .705 | .679 / .674 | .698 / .701 | .650 / .621 |

The entire gain is in one cell: a negated claim whose positive form is
derivable (ProofWriter strategy `inv-proof`). Within that cell the gain is flat
across inference depth: English minus Control is +.33, +.30, +.37, +.35, +.31
at depths 0, 1, 2, 3, 5 (seed 3407; seed 3408 within .03 of each). At depth 0
the fact is stated verbatim in the context ("The dog is red." / claim "The dog
is not red.") and Control answers correctly only 53 percent of the time;
English 85 percent. Positive claims do not move at any depth; at depth 5 the
positive/true cell is slightly worse for every derivation arm (seed 3407
-.06 English, p = .003; seed 3408 -.01 n.s.).

Gold-Unknown items (OWA): the arms shift predictions from "true" toward
"false" (Control 448 true / 236 false / 63 unknown; English 331 / 374 / 42).
No arm learns "unknown".

The training derivations contain no negation at all (0 of 2,000 band-25
traces in either notation contain "not" or a negation symbol) and no
true/false questions; the answer is an attribute word. So the model did not
learn negation from the data. The consistent reading is a changed decision
rule: a claim is accepted only if it matches a derived fact, otherwise it is
rejected. Control's failure mode is negation-blindness after Dolci SFT, and
the derivation arms remove it. LongDoc removes none of it (p = .62 / 1.00).

Consequences for the paper:
- the "depth profile" in the current draft (+4.6 at d0 rising to +12.6) is an
  artefact of class mix across depth buckets, not a depth effect; replace it
  with the 2x2 above;
- the ProofWriter result is a decision-rule and negation-robustness result and
  should be described as one; BranchProof-CoT carries the multi-step claim;
- the MCC / balanced-accuracy argument still holds (gold-true accuracy is
  preserved while gold-false doubles), so it is not a threshold shift, but it
  is also not depth-scaling deduction.

## 2. FOLIO paired statistics (seed 3407, n=203)

`scripts/analysis/folio_gpqa_rigor.py`. Both derivation arms +4.9 points;
English vs Control McNemar p = .031 (14 gained, 4 lost), bootstrap
[+.010, +.089]; Formal p = .053 (16 / 6), [+.005, +.094]; LongDoc p = .80.
English and Formal tie exactly (5 / 5 flips). Unlike ProofWriter the FOLIO
gain is on gold-false items of both polarities (negated/false .46 -> .63
English; positive/false .34 -> .47 Formal), but cells have 24-38 items.
Seed 3408 pending (gruenau jobs 1904/1907/1910/1913/1916).

The GPQA samples stored under `qwen25_longwin_folio_gpqa_20260910` are the
void 8-token run (p50 response 5 words); the rigor script skips them. Fresh
GPQA comes from the gruenau matrix.

## 3. Probes submitted on gruenau (results pending)

- `deduction_mc` (jobs 1933-1945): ProofWriter scored by choice
  log-likelihood. Gives a threshold-free AUC between gold-true and gold-false
  items per arm; if the arms only moved the threshold, AUC is flat.
- `deduction_cot` (jobs 1946-1958): ProofWriter with a derive-then-answer
  prompt, 1,024 tokens. Shows whether the arms write a derivation and whether
  the negated/false gain comes with an explicit contradiction.
- Both for the five instruction-tuned arms x two seeds and the three
  midtrained bases (no chat template).

## 4. Infrastructure findings

- The notation SFT on gruenau11 (job 1844_0) diverged at optimizer step 3
  (loss 1.3e8, then 0.0 with grad_norm nan for 778 steps) on torch 2.9.0, the
  version the handoff believed safe. The run "completed" and wrote a
  checkpoint of NaN weights. `train_instruction_sft.py` now carries a
  `DivergenceGuard` callback that raises on a non-finite or >50 loss.
  gruenau11 went DOWN (not responding) at 21:09. Smoke test on gruenau9
  (2x A100, job 1924, 12 steps) is running to separate H100 numerics from the
  gruenau software stack.
- `gruenau_full_eval.slurm`: (a) `.complete` is an empty file, the skip test
  used `-s` and never fired; (b) the free-memory wait read the node's first GPU
  instead of the allocated one; (c) the deduction yamls hardcode alex's vault
  path, so the job now rewrites the task directory into scratch. The graded
  deduction dataset (14 MB) and the ProofWriter per-item samples of both seeds
  were copied from alex to `~/rlvl_data/datasets/` and
  `~/rlvl_data/lm_eval_results/alex_mirror/`.
