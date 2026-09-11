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

## 16. Alex state at 2026-09-11 ~04:30

- 30-percent sweep: `nl_exact_band25_p30` reached 2385 (`2p5b` alias
  present); logic and condensed stopped at 2250 at the walltime, pass 2
  pending on priority. Post-SFT for the finished English arm submitted as
  4213747 (`qwen25_longwin_p30_post_sft_2026-09-09.slurm --array=1`).
- The unconditioned notation SFT array 4208280 is RUNNING on alex a100
  (tasks 0 and 1 since ~00:20, loss 0.50-0.52, learning rate in the decay
  phase, healthy), so the notation contrast will come from alex, not from
  the failed gruenau11 run.
- Scrambled-derivation chain resubmitted after a renderer bug (nl_scrambled
  was passed to the TemplateName enum): build 4213748 -> audit 4213749 ->
  midtrain passes 4213750-52 (`q25_longwin_scr`).

Formal on FOLIO (both seeds, gruenau): .606 / .606, +.049 (16/6, p = .053)
and +.054 (15/4, p = .019) over Control; GPQA .333 / .313, -.015 / -.025
(n.s.). Formal is therefore the arm with the most consistent FOLIO gain
(English +.034 / +.034, LongDoc +.010 / +.030), the opposite ordering to
ProofWriter. Pooled over seeds the FOLIO ordering is Formal > English >
LongDoc > Control, with Formal's gain twice LongDoc's; still small, but it is
the one out-of-family signal where the formal notation leads.

Depth 25 with the 12,000-token budget (max_model_len 16,384, i.e. beyond the
8,192 midtrain window), 200 items each, checker as above:

| base | correct | derivation present | every line valid | conclusion derivable | proof lines | words |
|---|---|---|---|---|---|---|
| Formal | **1.000** | 1.000 | .995 | 1.000 | 51 | 1,159 |
| English | .985 | 1.000 | .985 | .985 | 51 | 2,172 |

Both midtrained bases produce 51-line, checker-valid derivations at depth 25
in their own notation; the formal base is perfect and about half the length.
This is the real-model analogue of the synthetic depth-25 crossover (formal
edge at maximal depth), on n=200 the +.015 is suggestive only. The formal
lines are checked by mapping "P(cK) ; rule" through the model's own
<predicates> block back to "cK is X".

## 17. Final cross-cluster check and file index

533 overlapping cells between alex and gruenau (standard, multihop at 32768,
deduction; five arms, seed 3407): mean |diff| .0027, max .050 on one
200-item BBH subtask. Every gruenau number used above passes this check.

Outputs in this directory: `gruenau_matrix_tables.md` (every suite x kind x
seed), `pw_probes.txt` (MC / CoT / perturbation), `folio_gpqa_rigor.txt`,
`native_derivation_check.txt`, `cross_cluster_check.txt`, `pw_error_analysis`
in `analysis/pw_error_analysis_20260910.txt`. Still pending when written:
Formal base native d5-d20 (job 2019); the alex chains (p30 English post-SFT
4213747, notation SFT 4208280, scrambled midtrain 4213750-52, p30
logic/condensed pass 2).

## 18. What the paper should now say (proposed)

1. Program-generated derivations as 10 percent of a 2.5B-token midtrain
   give a base model that writes checker-valid derivations to depth 25 in
   the trained notation (Formal 200/200, English 197/200), with zero
   contamination; the control base writes none. Dolci instruction tuning
   removes the behaviour completely. This replaces the marker-count claim.
2. After instruction tuning, the ProofWriter gain is real, replicated,
   prior-invariant, and confined to negated claims whose positive form is
   derivable; it is flat in depth, absent before instruction tuning, and
   is best described as protection of deductive calibration through
   instruction tuning (Control's true/false AUC .755 -> .662 through SFT;
   derivation arms hold theirs). Under a written chain the arms converge.
3. Notation: English leads on ProofWriter and sampled BranchProof after SFT;
   Formal leads on FOLIO (+.05 both seeds) and at depth 25 in the base.
   The dose test (p30) and the scrambled control decide the remaining
   causal questions and are in flight.
4. Costs: LogiQA -2 to -3 replicated; GPQA, GSM8K, MATH, MMLU, multihop flat
   in both the base and instruction-tuned tables.

Formal base, native format, 4,600-token budget (job 2019): correct
.990 / 1.000 / 1.000 / .995 at d5/d10/d15/d20, every line checker-valid
1.000 / 1.000 / 1.000 / .995, conclusion derivable likewise, 11/21/31/41
proof lines. The d25 row of the 4,600-budget runs is VOID for both bases:
native-format prompts at depth 25 exceed 8,192 - 4,600 tokens, so lm-eval
left-truncated the context (the checker then rejects the first proof line,
valid_prefix .001, and the answer is right only 24 percent). The 12,000-token
run at max_model_len 16,384 (section 15) is the depth-25 measurement.

Final native-format table (midtrained bases, checker-valid derivation AND
correct answer, n=200 per cell):

| base | d5 | d10 | d15 | d20 | d25 (16k window) |
|---|---|---|---|---|---|
| Control | .000 | .000 | .000 | .000 | .000 |
| Formal | .990 | 1.000 | 1.000 | .995 | 1.000 |
| English | 1.000 | 1.000 | 1.000 | 1.000 | .985 |

## 19. Notation SFT on the control base (alex array 4208280, tasks 0/1; evaluated on gruenau)

Control midtrain base + Dolci with 10k band-25 traces replacing 10 percent
of the instruction mixture (formal or English), seed 3407, no derivation
midtraining. Compared with the derivation-midtrained arms (+ plain Dolci):

| model | PW d3 | PW d5 | BP-CoT d5 | BP-CoT d25 | MC AUC pooled | MC AUC within-neg | negation-blind | FOLIO |
|---|---|---|---|---|---|---|---|---|
| Control | .444 | .452 | .045 | .070 | .662 | .872 | .395 | .552 |
| English midtrain | .570 | .570 | .515 | .310 | .770 | .898 | .242 | .601 |
| Formal midtrain | .500 | .506 | .285 | .185 | .711 | .889 | .285 | .606 |
| Ctl + English SFT traces | .524 | .526 | **.995** | **.870** | .739 | .876 | (pending) | (pending) |
| Ctl + Formal SFT traces | .558 | .546 | .475 | .260 | .746 | .864 | **.181** | .581 |

Three consequences. (1) The answer-only ProofWriter gain is reproduced by
10k traces at instruction-tuning time (+.09 to +.11 at d3-d5) without any
derivation midtraining, and the formal traces do it as well as English
ones; this is the offset-removal component (pooled AUC up, within-polarity
AUC unchanged at .86). (2) What midtraining adds beyond that is the small
within-polarity discrimination gain (English midtrain .898 vs .876) and the
FOLIO edge (.60 vs .58). (3) Near transfer to BranchProof-CoT is dominated
by the SFT-time English traces (.995 at d5, .870 at d25, versus .515/.310
for the midtrained English arm), and formal SFT traces transfer far less to
the chat CoT format (.475/.260), reproducing the mixdepth finding that
prose beats formal at instruction-tuning time. So for "does midtraining on
derivations matter", the honest answer from this pair is: for ProofWriter
calibration, no more than instruction-tuning traces; for the base model's
own derivation ability (section 15), yes, and only midtraining gives that.
The matched-notation runs (derivation base + same-notation SFT, tasks 2/3)
will show whether the two stack.

ProofWriter CoT for the notation-SFT models: Ctl+Formal traces .472
(marker .78, loop 9 percent), Ctl+English traces .395 (marker .64, loop 18
percent, 352 words); conditional on a terminated chain both sit at
.60-.62 like every other arm. The English-trace SFT model is the most
brittle under an open-ended chain (the mixdepth scaffold-brittleness
pattern), while being by far the best at in-format BranchProof-CoT. Under a
written chain, nothing separates the arms except termination.

## 20. Correction to sections 15 and 19: SFT-time traces also restore derivation writing

Control midtrain + Dolci with 10k English traces, prompted in the native
document format: has_proof 1.000, every line checker-valid 1.000, correct
1.000 at d5/d10/d15/d20 (d25 at the 4.6k budget is void as before; the
12k-budget run is submitted). So checker-valid derivation writing is a
property of whichever training stage last contained derivations, not of
midtraining as such: derivation midtraining gives it to the base, Dolci-only
instruction tuning erases it, and instruction tuning that keeps 10 percent
traces gives it (back) to a base that never saw a derivation. The claim to
make is therefore: (a) program-generated derivations teach a 7B model to
write checker-valid derivations to depth 25 at either stage, with zero
teacher involvement and zero contamination; (b) whether they are retained is
decided by the instruction-tuning mixture; (c) the ProofWriter calibration
effect likewise appears from traces at either stage. What is left that is
specific to midtraining: the base-model discrimination gain (section 8), the
FOLIO edge of the Formal arm, and any dose or scrambled-control effect still
in flight. Section 15's sentence "only midtraining gives that" is withdrawn.

Depth 25, 12k budget, notation-SFT models on the control midtrain: Ctl +
English traces .995 correct / .995 every line valid; Ctl + formal traces
.980 / .990 (51 lines each). Same level as the midtrained bases (English
.985, Formal 1.000).

Standard suite for the notation-SFT models (seed 3407): GSM8K .788 / .784,
BBH .689 / .691, MMLU .687 / .692, LogiQA acc_norm .367 / .367, ARC .549 /
.561, WinoGrande .690 / .681 (formal / English traces). So the replicated
LogiQA loss (-1.4 to -1.7 for the derivation midtrains, both seeds) is
specific to derivation MIDTRAINING; traces at instruction-tuning time cost
nothing on LogiQA. WinoGrande is 1 to 2 points lower for the SFT-trace
models, within its ~1.3-point standard error.

## 21. The derivation scaffold generalises to other content (qualitative, then measured)

`scripts/analysis/native_format_transfer_probe.py` wraps GSM8K, ProofWriter
d3 and FOLIO items in the exact midtraining document format. Greedy
continuations (6 items each):

- English base: writes `<think><premises>..<proof>..<conclusion>` on all 18
  items and an `<answer>` on 14. On GSM8K the proof is a numbered arithmetic
  chain ("6. 3 + 4 = 7 ... 11. 9 * 2 = 18", answer 18, correct). On
  ProofWriter it restates the theory as premises and chains; on FOLIO it
  does the same in prose.
- Formal base: opens `<formal>` on all 18 items. On ProofWriter it produces a
  faithful first-order formalisation (constants, predicates incl. negated
  ones, premises like `H(r,d) -> D(d)`, `->E` steps), on FOLIO a plausible
  but loose one, on GSM8K a nonsensical predicate soup (arithmetic does not
  fit its grammar).
- Control base + formal SFT traces: GSM8K answered in plain prose inside
  `<answer>`; ProofWriter and FOLIO formalised as the Formal base does.

So the document format elicits the trained scaffold on arbitrary content,
and the English scaffold is the one that survives contact with arithmetic.
Measured next: full GSM8K test (1,319), ProofWriter d0-d5 (2,500) and
FOLIO (203) in the document format, greedy, for the two bases and the two
notation-SFT models (`scripts/analysis/native_format_eval.py`).

Multihop for the notation-SFT models (seed 3407, 32k context): standard
prompt flat (HotpotQA .551 / .553 vs Control .557; 2Wiki .378 / .373;
MuSiQue .285 / .307). Tagged prompt: Ctl + formal traces is the best model
on all three (2Wiki .305, HotpotQA .410, MuSiQue .195 vs Control .040 /
.280 / .180) while Ctl + English traces collapses on two (HotpotQA .110,
MuSiQue .040). Same asymmetry as the mixdepth campaign: English traces
make the elicited scaffold brittle under greedy decoding, formal traces do
not, even though English traces transfer better to the in-format CoT task.

Ctl + formal SFT traces, native format: correct 1.000 / 1.000 / .990 / .995
at d5-d20 (every line valid 1.000 / 1.000 / .995 / .995), .980 at d25 with
the 12k budget. Complete native-format table (checker-valid and correct):

| model | d5 | d10 | d15 | d20 | d25 |
|---|---|---|---|---|---|
| Control base | .000 | .000 | .000 | .000 | .000 |
| Formal base | .990 | 1.000 | 1.000 | .995 | 1.000 |
| English base | 1.000 | 1.000 | 1.000 | 1.000 | .985 |
| Ctl + formal SFT traces | 1.000 | 1.000 | .990 | .995 | .980 |
| Ctl + English SFT traces | 1.000 | 1.000 | 1.000 | 1.000 | .995 |
| any arm + plain Dolci SFT | no derivation written | | | | |

## 22. Document-format evaluation of real benchmarks (full sets, greedy, 3,000 tokens)

`scripts/analysis/native_format_eval.py`: GSM8K test (1,319), ProofWriter
d0-d5 (2,500), FOLIO (203) rendered as midtraining documents. Accuracy,
with (derivation written, answer tag reached):

| task | Control base | English base | Ctl + formal SFT traces | Ctl + English SFT traces |
|---|---|---|---|---|
| GSM8K | .386 (.00, .75) | .637 (.93, .83) | .669 (.00, 1.00) | .539 (.98, .70) |
| ProofWriter d0 | .430 (.00, .57) | .286 (1.00, .59) | .504 (.84, 1.00) | .550 (1.00, .97) |
| ProofWriter d2 | .214 (.00, .56) | .400 (1.00, .58) | .622 (.98, .99) | .566 (1.00, .96) |
| ProofWriter d3 | .150 (.00, .52) | .400 (1.00, .60) | .660 (.99, .99) | .536 (1.00, .96) |
| ProofWriter d5 | .082 (.00, .62) | .366 (1.00, .49) | .744 (.99, .99) | .510 (1.00, .94) |
| FOLIO | .335 (.00, .39) | .424 (1.00, .74) | .557 (.19, .99) | .557 (1.00, .94) |
| PW two-class accuracy given an extracted answer | .158 | .794 | .822 | .687 |

Formal base pending (job 2060). Reading: (1) the English base applies its
scaffold to everything and reaches .637 on GSM8K from a prompt it never saw
(its 8-shot standard score is .826), and .794 two-class on ProofWriter when
it finishes; its weakness is termination (answer tag on 49-62 percent of
ProofWriter items within 3,000 tokens). (2) The formal-trace SFT model,
when it writes a formal derivation, is the strongest ProofWriter model we
have: .744 at d5 with the answer tag on 99 percent, two-class .822, gold-
false .871, gold-true .761; but it is closed-world (unknown .06) and loses
the negated/true cell (.51). On GSM8K it drops the scaffold and answers in
prose (.669). (3) English SFT traces write long GSM8K derivations (623
words) that terminate less often (.70) and score lower (.539). So in the
one format that elicits the derivation, formal derivations plus a checker
would be the deployment story; and on ProofWriter the formal derivation
beats every chat-prompt arm by 15-20 points at depth 3-5.

## 23. Sampled derivations with checker selection (parse@k), first model

`scripts/analysis/sample_native_derivations.py` (n=16, T=0.8, top-p .95,
16k window, 9,000 tokens) + `native_passk_analysis.py`. parse@k = answer of
the first of k samples whose derivation the checker accepts in full, no
gold used. Ctl + formal SFT traces:

| depth | greedy | pass@1 / @4 / @16 | maj@1 / @4 / @16 | parse@1 / @4 / @16 | valid samples | acc given valid |
|---|---|---|---|---|---|---|
| 5 | 1.000 | .995 / 1.000 / 1.000 | .995 / .995 / 1.000 | .995 / .995 / .995 | .917 | .999 |
| 10 | .995 | .995 / 1.000 / 1.000 | .995 / .995 / 1.000 | .995 / .995 / .995 | .682 | .998 |
| 15 | .990 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / .995 | .995 / 1.000 / 1.000 | .728 | .998 |
| 20 | .400 | .995 / 1.000 / 1.000 | .995 / .995 / .985 | .990 / .995 / .995 | .618 | .991 |
| 25 | .910 | .985 / 1.000 / 1.000 | .985 / .990 / .990 | .980 / .990 / .990 | .630 | .992 |

Two facts. A checker-valid derivation implies the right answer (.99+), so
parse@1 equals pass@16 and the checker is a near-perfect gold-free
selector. And for this SFT model the decision to derive at all is a
first-token coin flip: only 62-92 percent of samples open with `<formal>`
(the rest jump straight to `<answer>`), and greedy at d20 derives on 22
percent of items (accuracy .400), whereas the two midtrained bases and the
English-trace SFT model open with a derivation on 100 percent of greedy and
sampled outputs. Formal traces at SFT time are a less firmly anchored
scaffold than English ones; sampling plus the checker removes the problem.

## 24. Matched-notation SFT: midtraining and SFT traces do NOT stack

Formal midtrain + formal SFT traces (alex 4208280_2), seed 3407, against
the two single-stage models:

| model | PW d3 / d5 | BP-CoT d5 / d25 | MC AUC pooled / within-neg | CoT acc | FOLIO |
|---|---|---|---|---|---|
| Formal midtrain + plain Dolci | .500 / .506 | .285 / .185 | .711 / .889 | .454 | .606 |
| Control midtrain + formal SFT traces | .558 / .546 | .475 / .260 | .746 / .864 | .472 | .581 |
| Formal midtrain + formal SFT traces | .536 / .522 | .285 / .205 | .714 / .884 | .461 | (pending) |

The stacked model sits between the two single-stage ones on ProofWriter and
at the level of the plain-Dolci formal arm everywhere else: no additive
effect. The ceiling is set by whichever stage last supplied derivations, not
by their sum. Same for the probes (pooled AUC .714, negation-blind and
ablation rows within noise of the plain formal arm).

## 25. Both matched-notation pairs: no stacking, in either notation

Seed 3407, chat prompting:

| model | PW d3 | PW d5 | BP-CoT d5 | BP-CoT d25 | FOLIO |
|---|---|---|---|---|---|
| MT-Formal + plain Dolci | .500 | .506 | .285 | .185 | .606 |
| Ctl + formal SFT traces | .558 | .546 | .475 | .260 | .581 |
| MT-Formal + formal SFT traces | .536 | .522 | .285 | .205 | .606 |
| MT-English + plain Dolci | .570 | .570 | .515 | .295 | .591 |
| Ctl + English SFT traces | .524 | .526 | .995 | .870 | .542 |
| MT-English + English SFT traces | .562 | .562 | .620 | .305 | (pending) |

Neither pair exceeds the better of its two single-stage parents on any cell.
The English pair is the sharper case: SFT traces alone give .995 / .870 on
in-format BranchProof-CoT, and adding them on top of English midtraining
gives only .620 / .305, i.e. prior midtraining SUPPRESSES the in-format
near-transfer that the traces alone produce. Stacking is not additive and
can be subtractive.

## 26. The checker dividend does not transfer to GSM8K

`scripts/analysis/checkers.py:check_gsm8k` verifies every `a op b = c` line
of a document-format GSM8K derivation. Applied to the greedy runs (1,319
items each):

| model | items with arithmetic lines | all lines check out | accuracy overall | accuracy when the arithmetic checks out | when it does not |
|---|---|---|---|---|---|
| Control base | .23 | .68 | .386 | .798 | .810 |
| English base | .48 | .66 | .637 | .817 | .784 |
| Ctl + formal SFT traces | .55 | .66 | .669 | .816 | .762 |
| Ctl + English SFT traces | .52 | .61 | .539 | .798 | .714 |

Arithmetic validity separates right from wrong answers by 3 to 8 points,
against 99 points on BranchProof (valid derivation implies correct answer
.99). Two reasons: only half the derivations state their arithmetic as
checkable equations, and a wrong GSM8K answer usually comes from a wrong
plan with correct arithmetic, which a step checker cannot see. So the
verifier dividend is a property of domains where the checker is complete
with respect to the task, not of derivation writing in general. State this
as the boundary of the verifiability claim.

The ProofWriter English-derivation checker in the same module is NOT usable:
it parses 23-36 percent of the model's proof lines and its own closure
agrees with the gold label on as few as 4 of 40 items at depth 5. Its
rejections measure the parser, not the model. Do not report parse@k on
ProofWriter until it handles the relational and quantified surface forms.
