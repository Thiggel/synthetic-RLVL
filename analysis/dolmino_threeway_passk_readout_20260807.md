# Real-benchmark pass@k / maj@k readout — three-way Dolmino post-SFT checkpoints

Date: 2026-08-07. Array `3962322_[0-5%3]` (resubmission of 3962260, which failed at launch because `WORK` was unset in a non-login submission shell; job script now defaults `WORK`/`HPCVAULT`, commit d045e4f). Harness: `scripts/analysis/evaluate_real_passk.py` (commit 87695bf) — exact accepted prompts replayed through vLLM, 1 greedy + n=16 samples (T=0.8, top_p=0.95, seed 20260806), accepted scorers reused (CPU parity-verified). All six per-row audit JSONs `accepted: true`. Rows took 30–60 min each on one A100-80GB. Full tables: `summary_passk.md` in `$HPCVAULT/synthetic-RLVL/lm_eval_results/qwen25_dolmino_post_sft_passk_20260806/`.

## Question under test

The greedy three-way readout (2026-08-06) was null-for-reasoning: tagged multi-hop gaps decomposed entirely into `<answer>`-tag compliance. Remaining preregistered question: does formal midtraining shift the *sampled solution distribution* — visible in pass@k/maj@k but not greedy — as it does in the synthetic domain (where sampling+verification lifts the formal condition from 0.75 to 0.97)?

## Verdict: NO. The sampled distribution confirms and sharpens the compliance decomposition.

**1. Untagged reasoning tasks are flat at every k.** GSM8K pass@16 0.9795/0.9780/0.9803 (control/logic/nl_exact); MATH-500 pass@16 0.5380/0.5300/0.5440. Paired bootstrap deltas (10k resamples, paired by doc): all CIs cover zero at k=16. The only significant untagged delta is logic *below* control on GSM8K sampled pass@1 (−0.0127 [−0.0189,−0.0064]).

**2. Tagged-task gaps at low k close as k grows.** Logic leads at pass@1–8 on all three tagged tasks, but at pass@16 conditions converge: 2wiki 0.575/0.575/0.580; hotpotqa 0.610/0.620/0.605. Oracle best-F1@16 is flat everywhere (2wiki 0.676/0.674/0.669; hotpotqa 0.739/0.751/0.741; musique 0.495/0.541/0.520). The support of the sampled distribution is essentially identical; sixteen tries give the less-compliant control model enough chances to emit one well-formed answer.

**3. The per-draw gaps are compliance, in two layers.** Layer 1: tag presence — control's sampled extraction-failure rate is 12–18% on tagged tasks vs 7–9% (logic/nl). Layer 2 (new): *degenerate tag content* — control frequently emits the literal `...` from the instruction template inside its tags (e.g. 2wiki: 457 degenerate tagged samples for control vs 309 logic), which scores as tagged-but-wrong and therefore survives both strict EM|tag conditioning and the first-line fallback extractor. Conditioning on tag_found alone leaves significant logic advantages (EM|tag logic−control +0.017…+0.048 across tagged tasks); additionally excluding degenerate extractions collapses them: EM|good logic−control 2wiki +0.014 [−0.010,+0.039], hotpotqa +0.018 [−0.001,+0.038], musique +0.011 [−0.004,+0.027] — all null. Raw-sample inspection confirms both patterns (wrong-entity errors shared across conditions; control echoing `<answer>...` scaffolding).

**4. Residual signals are isolated and not corroborated.** Two of ~12 comparisons remain significant after degenerate exclusion: musique strict pass@16 logic−control +0.060 [+0.015,+0.105] (also fallback +0.050) — but the matching logic−nl_exact delta is null (+0.020 [−0.015,+0.055]); and hotpotqa EM|good logic−nl_exact +0.015 [+0.004,+0.027] — not corroborated vs control. At this comparison count these are within multiple-testing expectations; neither supports a coherent "formal beats both" distribution claim.

**5. What the synthetic data does buy at low sampling budgets:** higher harvestable accuracy via compliance (maj@16 strict 2wiki 0.210/0.295/0.260), which shrinks under compliance-robust scoring. Honest positive statement unchanged from the greedy readout: 5% synthetic replacement is benchmark-neutral and strongly improves instructed answer-format compliance (logic ≈ nl, both ≫ control), now shown to hold across the whole sampling distribution.

## Consistency and implications

The synthetic-domain sampling advantage (pass@k ≫ greedy for the formal condition) does **not** transfer under the current flat-window packing objective, consistent with the split-document diagnosis (44–48% of derivation docs split mid-proof). The preregistered transfer gate remains NOT passed; the document-preserving/compact-objective pilot stays the P0 science gate. The harness makes the sampled readout cheap (~30–60 min/row), so the rerun checkpoints get the same analysis.

Paper: add to the transfer section that the null is robust across the sampled distribution (pass@1–16, maj@k, oracle best-F1@16) — the compliance decomposition now has a second layer (degenerate tag echo) that strict-vs-fallback alone does not catch. Do not present low-k tagged gaps as reasoning gains.

## Provenance

Bootstrap/conditional scripts: /tmp/passk_boot.py, /tmp/passk_cond.py, /tmp/passk_cond2.py on alex (paired by doc_id, 10k resamples, seeds 20260807–9); to be re-run via a committed in-repo script before any number is quoted in the camera-ready (same rule as the greedy fallback rescorer).
