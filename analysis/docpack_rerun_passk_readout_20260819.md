# Docpack 2.5B rerun: sampled pass@k / maj@k readout

Date: 2026-08-19. Array `4043881_[0-5%3]`, harness
`scripts/analysis/evaluate_real_passk.py` (1 greedy + n=16 samples, same
sampler/seed as the accepted 5B readout). All six per-row audits
`accepted: true`. Analysis re-run through the SAME committed scripts used for
the accepted readout -- `passk_boot.py`, `passk_cond.py`, `passk_cond2.py` --
with the results root made overridable via `PASSK_ROOT` (defaults unchanged),
so the rerun and the 5B run sit on identical methodology.

## Question under test

The greedy rerun readout (analysis/docpack_rerun_multihop_readout_20260819.md)
was null. Remaining question, as at 5B: does the document-preserving objective
shift the SAMPLED solution distribution for the formal condition, visible in
pass@k but not greedy -- as it does synthetically (0.75 -> 0.97 under
sampling+verification)?

## Verdict: NO -- and more strongly negative than at 5B.

**1. Untagged reasoning is flat at k=16.** GSM8K pass@16 logic-control +0.0023
[-0.0053,+0.0099] and logic-nl -0.0023 [-0.0099,+0.0053]; MATH-500 pass@16
+0.0000 [-0.0380,+0.0360] and -0.0180 [-0.0520,+0.0180]. The only significant
untagged pass@1 effects are small and mixed in sign (GSM8K logic ABOVE both,
+0.0146*/+0.0178*; MATH-500 logic BELOW control -0.0091*).

**2. Tagged pass@16 shows no logic advantage anywhere; one significant deficit.**
2wiki strict logic-control -0.0550 [-0.1000,-0.0100]*; hotpotqa -0.0050 n.s.;
musique -0.0500 [-0.1000,+0.0000] n.s. Under fallback scoring all tagged
pass@16 contrasts are null. At 5B, logic led at low k and converged by k=16;
here logic does not lead at k=16 at all.

**3. Removing BOTH compliance layers does not reveal a logic gain -- it reveals
a synthetic-replacement deficit.** Layer 1 (EM|tag): logic-control -0.0231 n.s.
(2wiki), -0.0337* (hotpotqa), -0.0206* (musique). Layer 2 (EM|good, excluding
degenerate tag content): logic-control -0.0244* (2wiki), -0.0214* (hotpotqa),
-0.0096 n.s. (musique). Critically the SAME direction holds for nl_exact:
nl-control EM|tag -0.0386* (hotpotqa), -0.0211* (musique). Both synthetic
conditions sit at or below control on conditional accuracy, so this is a
synthetic-replacement effect, NOT a representation (logic-vs-NL) effect.

**4. The degenerate-tag artifact REVERSED relative to 5B.** Degenerate tagged
samples (literal `...` echoed from the instruction template):
hotpotqa control/logic/nl = 94/216/279; musique 80/298/324. At 5B the artifact
belonged to control (e.g. 2wiki 457 control vs 309 logic). Under the
document-preserving objective the synthetic conditions echo the `...`
scaffolding 2-4x MORE than control. Raw-sample inspection confirms the
extracted content is the literal `...`.

**5. Compliance ordering also flipped.** Tag rates control/logic/nl:
2wiki 0.691/0.652/0.839; hotpotqa 0.749/0.772/0.837; musique 0.677/0.775/0.753.
nl_exact now has the highest tag rate on 2 of 3 tasks, and on 2wiki logic
(0.652) is BELOW control (0.691). At 5B both synthetic conditions were
strongly above control (~0.84 vs ~0.50) with logic slightly ahead of NL.

## Implications

- The preregistered transfer gate remains NOT passed. Document-preserving
  packing does not rescue formal-logic transfer; the 5B null was not an
  artifact of mid-document splitting (44-48% of proof docs were split).
- The honest positive statement from 5B -- "synthetic replacement strongly
  improves instructed answer-format compliance" -- WEAKENS here: compliance
  gains are smaller, not consistently logic-favouring, and are partly an
  artifact of echoing template scaffolding.
- Combined with the flat ten-task macro (0.5815/0.5835/0.5847), the honest
  summary is that 5% synthetic replacement under the document-preserving
  objective is benchmark-neutral and produces no reasoning transfer in either
  representation.

## Limitations

- Single training run per condition; uncertainty is example-level only.
- Rerun corpus capped at depths 1-14; the deep regime where logic wins
  synthetically is not represented. This remains the strongest live
  explanation for the transfer null and is NOT ruled out by this experiment.
- Many comparisons; per the 5B standard, isolated significants require
  corroboration across both contrasts. The logic-below-control conditional
  deficits ARE corroborated across two tasks and mirrored by nl_exact, which is
  why they are read as a synthetic-replacement effect rather than noise.
- The control arm's midtrain terminal retained model weights only (narrowed
  base gate); model weights verified byte-identical in manifest to logic.
