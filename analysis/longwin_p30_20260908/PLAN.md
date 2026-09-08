# 30-percent-share midtrain sweep (submitted 2026-09-08)

## Why this sweep exists

The clean sampled readout of the 10-percent long-window arms (2026-09-08,
`lm_eval_results/qwen25_longwin_passk_20260908_eosfix`) put English at or
above Formal on every decoding rule: pooled pass@16 control/longdoc/logic/nl
= 57.0/61.4/81.4/86.4, nl - logic = +5.0 [2.1, 7.9]. The earlier readout that
showed the opposite sign was an artifact (the sampler did not stop on
<|im_end|>; see docs/project_log.md 2026-09-08). The paper's remaining claim
about notation, "the formal advantage is a property of dense, deep
supervision", is therefore asserted without midtraining evidence. This sweep
tests the dose half of it directly, on the same base, pipeline and
evaluation.

Two facts constrain the design:

1. On Qwen2.5-7B the controlled study itself shows no formal advantage at
   D=25 (pass@1 69.9 vs 74.9, pass@16 77.3 vs 76.0). The prior for this
   sweep is therefore a null, and a null is publishable: it removes the dose
   hypothesis from the paper. An OLMo-3-7B midtrain (the base where the
   effect exists) is not possible on this stack: the nanotron checkout
   implements only Llama and Qwen.
2. Deeper chains are not possible at 8192: depth-25 formal documents already
   reach 7,727 tokens.

## Design

Three arms, everything identical to the accepted 10-percent wrapper
(`nanotron_qwen25_longwin_midtrain_2026-08-26.slurm`: 2,385 steps x 128 x
8192 = 2,500,853,760 tokens, LR 1e-5, seq 8192, document-preserving packer,
decoded-batch audit gate) except the replaced share, 30 percent of loss
tokens instead of 10:

| arm | rendering | docs needed at 30% (mean tokens) | corpus | epochs |
|---|---|---|---|---|
| logic_band25 | standard formal (theory stated twice: NL prompt + FOL) | ~195k (3,856) | 520k | ~0.37 |
| nl_exact_band25 | controlled English (theory once) | ~192k (3,907) | 520k | ~0.37 |
| condensed_logic_band25 | formal only, theory once | ~500k (1,500) | 520k | ~0.96 |

Corpus: 520,000 latent proofs, seed 20260908, depths 1-25 round-robin,
branching 4, distractor 0.5, hard_fsa_schema, rendered three ways from the
same parquet (`datasets/branchproof_unique_v2_longwin_p30_20260908`). 520k
was chosen so that no arm sees any document twice; in the 10-percent sweep
the condensed arm traversed its 72k proofs 2.31 times, which confounded
rendering with exposure. Here every document is seen at most once. The
arms still differ in the number of distinct proofs seen (condensed about
2.6x more, because FOL is about 1.5x more compact than controlled English
and the standard formal document states the theory twice). That is a
property of the rendering under a fixed token budget, and it is reported
as such.

No control and no longdoc arm: the 10-percent control is the same base
checkpoint and the same token budget, so it remains the reference; the
general-benchmark suite will show whether 30 percent replacement costs
anything.

## Jobs

- corpus build: 4201577 (`longwin_build_band25_2026-08-26.slurm` with
  LONGWIN_ROWS=520000, seed 20260908, p30 roots). The 72k build took 21 min,
  so 520k should take about 2.5 h inside the 12 h walltime.
- audit at target ratio 0.30: 4201578 (`longwin_docpack_audit_p30_2026-09-08.slurm`),
  afterok the build; writes `analysis/longwin_p30_20260908/audit_docpack_*.json`
- midtrain: `nanotron_qwen25_longwin_p30_midtrain_2026-09-08.slurm`, job name
  q25_longwin_p30, singleton-serialized, three passes per arm (a 2,385-step
  run needs about 2 x 24 h; the grid script resumes from the newest complete
  checkpoint and exits immediately once the final checkpoint exists):
  condensed 4201579/80/81, nl_exact 4201582/83/84, logic 4201585/86/87.
  Run roots: `$WORK/synthetic-RLVL/nanotron_longwin_midtrain/qwen25_7b_longwin_<arm>_p30_2p5b_s8192`.
- after the midtrains: post-SFT (`qwen25_longwin_post_sft_2026-08-26.slurm`,
  seed 3407) and the same readouts as the 10-percent arms (graded deduction,
  clean pass@k, downstream std/multihop), all seed-parameterised scripts.
  NOT yet submitted; they need the run-tag mapping for the _p30 names.

Expected wall time: about 6 days of serialized midtraining plus about 1 day
of post-SFT and evals, queue permitting.

## What was cancelled and why

- RL sweep on the 10-percent midtrain arms (`rl_longwin_band25_2026-09-07.slurm`):
  never submitted beyond the smoke; cancelled 2026-09-08. After instruction
  tuning none of the arms writes a derivation, so the two validity rewards
  have nothing to check, and the correctness-only reward would start from a
  model whose formal coverage is below its English coverage. The config
  repairs (eval block, update_weights_bucket_megabytes=4096) are kept.
- RL on the controlled-study D=25 adapters: not started. The adapters were
  deleted in the 2026-08 vault cleanup and are not on the Hub; rerunning
  the SFT (two ~12 h jobs) before any RL step did not fit the deadline.

## Predictions, written before the results

- English >= Formal on ProofWriter and on near-transfer greedy at 30
  percent, as at 10 percent (the Qwen control predicts no formal advantage).
- If the dose hypothesis is right, Formal pass@16 should close the 5-point
  gap or reverse it at 30 percent; if it stays or widens, the hypothesis is
  dropped from the paper.
- Condensed at 30 percent with 0.96 epochs: its 10-percent lead over English
  on near transfer should persist if it came from seeing more distinct
  proofs, and shrink if it came from repetition.

## Audit result (2026-09-08, after the build)

Build 4201577 completed in 2h06 (520,000 rows, three renderings, three
docpacks). Audit 4201578 passed all four gates for every arm at target ratio
0.30 (zero_overlength, decoded_batch, padding_loss_mask, exact_mixture):

| arm | proof_weight | realised loss-token ratio | epochs over the 520k docs | windows | real tokens |
|---|---|---|---|---|---|
| logic_band25 | 0.3097 | 0.30000 | 0.369 | 256,247 | 2,099,431,671 |
| nl_exact_band25 | 0.3098 | 0.30000 | 0.364 | 259,766 | 2,128,262,838 |
| condensed_logic_band25 | 0.3018 | 0.30000 | 0.959 | 96,124 | 787,543,932 |

Rendered lengths match the 72k corpus (logic p50 3,749, max 7,718, 0 percent
over the window). No arm repeats a document. The first midtrain pass
(4201579, condensed) is released from its dependency and waits on priority.
