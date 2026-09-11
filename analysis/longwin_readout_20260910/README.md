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

## 5. Cross-cluster check (`scripts/analysis/cross_cluster_check.py`)

Standard suite, seed 3407, logic/nl arms: gruenau reproduces alex to within
vLLM batching noise (log-likelihood tasks identical, BBH subtasks within 1-2
points, aggregate BBH +0.0025). The first gruenau multihop bundles did NOT
reproduce (control HotpotQA F1 .36 vs .56 on alex, tagged near zero): the job
ran at max_model_len 8192 while alex used 32768, so lm-eval left-truncated
the LongBench contexts. Those bundles are quarantined under
`lm_eval_results/quarantine/gruenau_it_multihop_20260910_len8192_truncated`
and the suite was resubmitted at 32768 (jobs listed in the session log).
Rule: every gruenau suite must pass this check on seed 3407 before any of its
numbers, including the base-model ones, enter the paper.

## 6. Scrambled-derivation control (alex, submitted 2026-09-10 ~22:40)

Rendering `nl_scrambled` in `scripts/data/build_longwin_trace_jsonls.py`:
the controlled-English document with the lines of each `<proof>` block
shuffled (seeded), premises, conclusion and answer untouched, so length,
vocabulary and token statistics are identical to `nl_exact_band25` and only
the deductive order is destroyed. Chain: build 4212870 (72k parquet of
2026-08-26, packs `nanosets_longwin_20260826/nl_scrambled_band25`) -> audit
4212871 (`LONGWIN_AUDIT_ARMS=nl_scrambled_band25`, template sft_nl_exact) ->
midtrain passes 4212872-74 (`q25_longwin_scr`, singleton, 10 percent
replacement, same wrapper as the accepted five arms, CONDITION
`nl_scrambled_band25`). Post-SFT wrapper index 5 is wired; eval wrappers take
the arm name. Pre-registered prediction: if the ProofWriter negated/false gain
survives scrambling, it is surface structure (a decision-rule change induced
by any derivation-shaped text); if it drops to LongDoc's level, it requires
valid deductive order. Everything on alex waits behind the 24-GPU cap held by
the p30 arms.

## 7. H100 divergence isolated to the H100 node (smoke 1988, gruenau9)

Same script, same venv (torch 2.9.0+cu128), same converted base and formal
mixture, 2x A100 with FSDP CPU offload, window 4096, per-step logging: six
optimizer steps at lr 5e-6 give losses 0.73, 0.84, 0.78, 0.80, 0.88, 0.77
with grad norms 1.7-4.6. The gruenau11 H100 run exploded at step 3 (loss
1.3e8) at the same learning rate. So the recipe and the gruenau software
stack are fine on A100; the fault is specific to the H100 node or its
driver/kernels, and gruenau11 is currently DOWN anyway. Offload runs at
515 s/step (about 4.7 days for 781 steps), so the notation SFT cannot be
run on gruenau9 this way; it needs gruenau11 back (and then a smoke there
first) or alex GPUs once the p30 arms release the cap. The alex notation
array 4208280 is queued for exactly that.

## 8. MC log-likelihood probe: mostly offset removal, a little discrimination

Choice log-likelihood over True/False/Unknown, margin = ll(True) - ll(False),
AUC between gold-true and gold-false decidable items (seed 3407 / 3408):

| arm | AUC pooled | AUC within negated | AUC within positive | P(argmax=True) |
|---|---|---|---|---|
| Control | .662 / .654 | .872 / .869 | .860 / .858 | .64 / .63 |
| LongDoc | .663 / .664 | .876 / .877 | .867 / .867 | .64 / .64 |
| Formal | .711 / .707 | .889 / .889 | .868 / .868 | .55 / .54 |
| English | .770 / - | .898 / - | .873 / - | .48 / - |

Control already ranks items well within each polarity; its pooled AUC is low
because negated claims carry a wholesale shift of the margin toward True.
The derivation arms mostly remove that shift (pooled AUC +.05 Formal, +.11
English) and add a small within-polarity discrimination gain on negated
claims (+.02/+.03) and almost none on positive claims (+.01). LongDoc
changes nothing. So the ProofWriter effect is a calibration-of-negation
effect with a minor discrimination component, and it should be described
that way. (vLLM returns bf16-quantised logprobs on a 1/8 grid; 6 percent of
items tie True/False, handled by mid-ranks.)

Base models (no chat template), same probe: Control base pooled AUC .755,
within-negated .818, within-positive .821, P(True) .59; English base .782 /
.875 / .856 / .64. Two things follow. (a) Instruction tuning is what creates
Control's negation offset: its pooled AUC falls from .755 to .662 through
Dolci SFT, while the English-midtrained model keeps .78 -> .77. (b) Before
instruction tuning the midtrained English model has a real within-polarity
discrimination advantage (+.06 negated, +.035 positive) that Dolci SFT
compresses to +.03 / +.01. So the derivations do improve the base model's
deduction signal, and the instruction-tuning stage both erodes that and
introduces the negation-blindness that the derivation arms then resist.

## 9. CoT probe: the answer-only gain does not survive a written chain

Derive-then-answer prompt, 1,024 tokens, same 2,500 items (seed 3407 / 3408):

| arm | acc | marker found | acc given marker | neg/false | pos/true |
|---|---|---|---|---|---|
| Control | .480 / .484 | .82 / .83 | .584 / .580 | .504 / .502 | .630 / .615 |
| LongDoc | .558 / .549 | .89 / .91 | .617 / .603 | .589 / .575 | .667 / .650 |
| Formal | .454 / .454 | .76 / .75 | .596 / .599 | .506 / .508 | .579 / .573 |
| English | .507 / - | .81 / - | .620 / - | .552 / - | .620 / - |

Once the model writes a chain, Control's negated/false accuracy rises from
.35 to .50 on its own, Formal is level with Control, English keeps a +.05
edge, and LongDoc is the best arm. Conditional on the chain terminating with
a marker, all arms sit at .58-.62. The differences that remain are mostly
chain termination (Formal loops and runs past 1,024 tokens on a quarter of
items; loop rate 8.4 vs 7.2 percent, mean 266 vs 231 words), not deduction.
So the ProofWriter effect is specific to the answer-only regime: it is a
change in the direct-answer decision rule, and a written chain replaces
that rule with whatever the chain concludes. This is the honest framing for
the paper, alongside the base-model discrimination gain of section 8.

## 10. Perturbation probes (answer-only prompt; seed 3407 / 3408)

Flip (claim polarity toggled, theory unchanged): fraction of decidable items
answered identically under both polarities, i.e. negation-blind:
Control .376 / .376, LongDoc .376 / .380, Formal .285 / .280 (English pending).
On the 785 negated/false items, the positive version is answered correctly by
every arm (Control .68, Formal .68), so the arms differ only on the negated
version: Control says "true" to both, Formal says "true"/"false".

Ablate (the proof's premises removed, claim unchanged, gold unknown): the
arms' answers do depend on the premises. On the negated/false items Formal's
P(false) falls from .53 to .27 once the supporting facts are gone (Control
.35 -> .20, LongDoc .36 -> .19); overall 44 percent of Formal's answers
change under ablation vs 42 percent for Control. So the extra "false" is not
a closed-world default applied blindly; it is conditioned on the premises
that make the negated claim false. No arm answers "unknown" (.19-.24).

Taken together with sections 8 and 9: the derivation arms make the
direct-answer decision premise-grounded and negation-aware; they do not add
depth-scaling deduction, and the effect is bypassed when a chain is written.

English and Condensed (seed 3407 / 3408): negation-blind rate English .242 /
.250, Condensed .289 / -; ablation P(false) on negated/false items English
.67 -> .46 / .68 -> .46, Condensed .49 -> .29. English keeps the largest
residual "false" after the premises are gone (.46 vs Formal .27, Control
.20), so the arm with the biggest ProofWriter gain is also the one whose
extra "false" is least premise-conditioned: about two thirds of its gain
over Control on this cell survives ablation. English's gain is therefore
part premise-grounded, part a shifted default toward "false" for negated
claims; Formal's smaller gain is more premise-grounded.

## 11. GPQA-Diamond at 4,096 tokens (seed 3407, n=198) and FOLIO rerun noise

GPQA (fresh, the void 8-token run discarded): Control .348, LongDoc .308
(p = .35), English .278 (22 gained / 36 lost, McNemar p = .087, bootstrap
[-.146, .000]). Formal and Condensed pending, seed 3408 pending. A 7-point
drop on 198 items is borderline and must be replicated before it is called
a cost, but it is in the same direction as the replicated LogiQA loss and
should be reported with it if seed 3408 agrees.

FOLIO rerun on gruenau for Control and English (same checkpoints as the
16:43 alex-side numbers): English minus Control is now +.034 (13/6,
p = .17) against +.049 (14/4, p = .031) before. Three items of vLLM batching
nondeterminism move FOLIO across the .05 line, so FOLIO cannot carry a
significance claim on its own at n=203; report it as directionally
consistent and pool both seeds.

Seed 3408 (n=203 / 198): FOLIO Control .552, LongDoc .581 (+.030, p = .24),
English .586 (+.034, p = .12), Condensed .581 (+.030, p = .21); GPQA Control
.338, LongDoc .374 (+.035), English .318 (-.020, p = .67), Condensed .308
(-.030, p = .47). Verdicts: (a) the seed-3407 English GPQA drop does not
replicate; GPQA is flat within noise for every arm, so it is the intended
safety check and nothing more. (b) FOLIO: English is +.034 in both seeds
but LongDoc is +.030 at seed 3408 and +.010 at 3407, so FOLIO does not
separate derivations from long documents; report the pooled English gain
(+.034 / +.034) as directionally consistent, not as evidence of transfer.

## 12. Midtrained bases under the chat-style prompts, and the native-format probe

Base models with the chat-style graded prompts (no template): ProofWriter
Control .36/.34/.51/.51/.58 vs Formal .49/.36/.48/.48/.49 at d0/1/2/3/5;
BranchProof-CoT Formal .10/.12/.06/.06/.04 vs Control .04/.03/.03/.02/.04,
and 0 of 200 Formal-base responses at d15 contain a <formal>, <premises> or
FOL line: the base drifts into unrelated pretraining-style text. So the
chat-style prompt cannot tell whether the notation survived midtraining. New
suite `deduction_native` (jobs 2005-2012): BranchProof d5-d25 rendered as the
training document ("<question>\n1. ...\nWhich state applies to cN?\n</question>\n\n"),
generation until </answer>, 4,600 tokens, metrics exact_match, answer_tag,
formal_block, think_block, proof_block. Run on the three bases and, for
contrast, the five instruction-tuned arms. Stored generations feed
`synthrlvl.metrics.OutputEvaluator` for the checker result.

## 13. The ProofWriter gain does not exist before instruction tuning

Midtrained bases, answer-only prompt, no chat template (d0/d1/d2/d3/d5):
Control .362/.342/.508/.506/.576, Formal .490/.356/.476/.484/.494, English
.440/.350/.490/.504/.530. Apart from depth 0 the derivation bases are at or
below the control base; base negation-blind rates are equal (Control .404,
Formal .406). Together with section 8 (Control's pooled AUC falls .755 ->
.662 through Dolci SFT while the derivation arms hold theirs) the mechanism
is an interaction: Dolci instruction tuning makes the control negation-blind
on direct answers, and prior derivation midtraining immunises against that.
The paper should say "derivation midtraining protects deductive
calibration through instruction tuning", not "derivation midtraining
teaches ProofWriter". The base-model discrimination advantage in the MC
probe (+.05/.06 within polarity) is the one thing that is present before SFT.

## 14. Midtrained bases on multihop QA (no chat template)

The stock standard-prompt F1 for the derivation bases reads ~0 (Formal .002,
English .016 on HotpotQA) against Control .542. That is termination, not
ability: a base has no end-of-turn token, and the derivation bases keep
writing "Question: ... Answer: ..." continuations after a correct first
line, while the control base happens to stop. Scoring the first non-empty
line (`scripts/analysis/base_multihop_firstline.py`): HotpotQA Control .542,
Formal .568, English .568; 2Wiki .349 / .357 / .362; MuSiQue .306 / .273 /
.269. Tagged prompt (stock scorer, EM): HotpotQA .395 / .460 / .435, 2Wiki
.240 / .275 / .265, MuSiQue .125 / .150 / .150. So at the base stage the
derivation arms are level with or slightly above control on multihop QA;
nothing is damaged before instruction tuning either. Report the base table
with the first-line rescoring and say why.

## 15. THE MISSING LINK: the midtrained English base writes checker-valid derivations

Native document format (`synthrlvl_deduction_bp_native_d*`, generation until
</answer>, 4,600 tokens), midtrained bases, no chat template:

| base | d5 | d10 | d15 | d20 | d25 (4.6k budget) |
|---|---|---|---|---|---|
| Control | .000 (no derivation; answers "north" 189/200) | .000 | .000 | .000 | .000 |
| English | **1.000** | **1.000** | **1.000** | **1.000** | .000 (98.5% cut before </answer>) |
| Formal | pending (job 2006) | | | | |

`scripts/analysis/check_native_derivations.py` re-derives every proof line
by forward chaining over the item's own facts and rules (Horn fragment:
"cK is X" is valid iff it is a premise or the head of a rule whose
antecedents are already established). English base, 200 items per depth:
has_proof 1.000, every line valid 1.000, valid prefix 1.000, conclusion
derivable 1.000, conclusion equals last proof line 1.000, answer correct
1.000, at d5/d10/d15/d20; mean proof length 11/21/31/41 lines, i.e. exactly
2 lines per depth step plus the seed facts, with 5-18 percent of lines
restating premises (the trained rendering does that too).

Contamination: none of the 600 evaluated theories (d5/d15/d25) appears in
the 72,000 midtrain documents (full-context match 0; 77 share a three-rule
prefix, which the small rule vocabulary makes unavoidable).

Instruction-tuned English under the same format: answer tag 99 percent, no
derivation block at all, 23-25 percent correct. So the derivation ability
is fully present after midtraining and fully removed by Dolci instruction
tuning, which is the strongest real-model statement in the paper and
replaces the marker-count argument. Depth 25 needs a longer budget because
the English rendering restates the ~3.4k-token premises: rerun as
`deduction_native_long` (12,000 tokens, max_model_len 16,384, jobs 2015/2016),
to be reported with the caveat that it exceeds the 8,192 midtrain window.
