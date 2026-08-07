# Dolmino 5B Three-Way Post-SFT Readout (2026-08-06)

Date: 2026-08-06 ~16:30 CEST. Author: Claude oversight pass (interactive session).

Status: all six bundles (control/logic/nl_exact x standard/multihop) report
`accepted: true` in `production_audit.json` / `multihop_audit.json`. The formal
standard and multi-hop consumer rows `3950715_1` (as 3950718) and `3950716_1`
(as 3950720) completed `0:0` today at 14:24 and 13:53 CEST. This document is
the first full three-way aggregation, computed directly from the accepted
artifacts under
`/home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results/qwen25_dolmino_post_sft_20260804/`.
Raw-generation inspection was performed for the load-bearing cell (control
tagged failures); a broader cross-task generation review remains open for the
scheduled oversight pass.

## Ten-task standard suite (post-SFT, per prereg PRIMARY_METRICS; MATH-500 via
answer-prefix math-verify sidecar)

| metric | control | logic | nl_exact | logic-ctl | logic-nl |
| --- | --- | --- | --- | --- | --- |
| gsm8k | 0.7976 | 0.8067 | 0.8074 | +0.0091 | -0.0008 |
| math500 (sidecar) | 0.1060 | 0.0900 | 0.0900 | -0.0160 | +0.0000 |
| arc_challenge | 0.5853 | 0.5631 | 0.5666 | -0.0222 | -0.0034 |
| hellaswag | 0.7371 | 0.7374 | 0.7362 | +0.0003 | +0.0012 |
| winogrande | 0.7024 | 0.6985 | 0.7017 | -0.0039 | -0.0032 |
| piqa | 0.7982 | 0.8036 | 0.8030 | +0.0054 | +0.0005 |
| agieval_logiqa_en | 0.3441 | 0.3425 | 0.3118 | -0.0015 | +0.0307 |
| bbh | 0.6758 | 0.6890 | 0.6864 | +0.0132 | +0.0026 |
| mmlu | 0.6825 | 0.6795 | 0.6825 | -0.0030 | -0.0030 |
| mmlu_pro | 0.4738 | 0.4761 | 0.4764 | +0.0023 | -0.0002 |
| **MACRO10** | **0.5903** | **0.5886** | **0.5862** | -0.0016 | +0.0025 |

The primary aggregate is flat across conditions: no formal gain, no material
regression (5% synthetic replacement of Dolmino is benchmark-neutral at 5B).

## Multi-hop (LongBench Hotpot/2Wiki/MuSiQue, 32,768-token window, n=200/task)

Raw qa_f1 macros:

| protocol | control | logic | nl_exact |
| --- | --- | --- | --- |
| standard | 0.4134 | 0.4089 | 0.4127 |
| tagged | 0.2352 | 0.3124 | 0.2868 |

Paired example-level bootstrap (10k resamples, pooled n=600):

- tagged logic-control: +0.0773 [+0.0459, +0.1084]
- tagged logic-nl:      +0.0256 [+0.0066, +0.0446]
- tagged nl-control:    +0.0517 [+0.0225, +0.0812]
- standard: all pairwise pooled deltas within [-0.005, +0.001], all n.s.

## Decomposition: the tagged gap is answer-format compliance, not reasoning

1. Restricted to examples where both conditions produce a nonempty extracted
   answer, all tagged deltas collapse: logic-nl +0.0075 [-0.0077, +0.0231];
   logic-control -0.0094 [-0.0379, +0.0182]; nl-control -0.0251 n.s.
2. Extraction rates differ strongly: tag_found control ~0.50, nl ~0.77-0.85,
   logic ~0.83-0.85.
3. Raw-generation inspection of control tagged failures (hotpot): the model
   outputs the correct bare answer (e.g. `Pleiospilos`, `Long Island`) without
   the instructed `<answer>...</answer>` wrapper and is scored F1=0. The
   prompt instructs "Put only the final answer in <answer>...</answer>."
4. Rescoring tagged responses with a fallback extractor (bare first line when
   no tag) removes the effect entirely: fallback macros control/logic/nl =
   0.3390 / 0.3535 / 0.3431; pooled logic-control +0.0145 [-0.0106, +0.0402];
   logic-nl +0.0105 [-0.0058, +0.0274]; nl-control +0.0041 n.s.

## Gate verdict (per docs/iclr_2027_execution_plan_2026-08-04.md)

- Primary aggregate: formal does NOT exceed control or NL (flat).
- Preregistered multi-hop macro: formal exceeds both on the tagged protocol
  numerically, but the interpretation gate (generation inspection) shows the
  aggregate difference is an extraction/compliance artifact, not reasoning.
  Under compliance-corrected scoring the three-way comparison is null.
- Verdict: the positive formal-transfer gate is NOT passed at 5B. Per the
  Week-3 stop rule, do not launch the percentage grid and do not continue to
  10B on this recipe; spend compute on the one diagnosis supported by raw
  generations - the document-preserving/compact-objective pilot already
  preregistered as P0 (44-48% of proof documents exceeded the 4,096-token
  training window and were split arbitrarily by the fixed-stream loader).

What survives as a positive, honest transfer statement:

- 5% synthetic proof-trace replacement (either representation) is
  benchmark-neutral on the ten-task macro after identical instruction SFT.
- Both synthetic conditions substantially improve instructed answer-format
  compliance on long-context multi-hop prompts (tag rates 0.50 -> ~0.84);
  logic slightly exceeds NL on compliance. This is a response-control
  transfer, consistent with the p15 diagnosis, and should be reported as
  such - not as a reasoning gain.

## Analysis assumptions / limitations

- Single training run per condition; uncertainty is example-level only.
- Multi-hop n=200 per task per protocol (LongBench limit); the ten-task
  standard suite is full-set.
- The fallback rescorer is a session-local reanalysis
  (first-line-if-no-tag; SQuAD-style token F1 over gold list); it should be
  reimplemented in-repo before the number is used in the paper.
- Sample inspection covered control-vs-logic tagged hotpot failures; the
  full cross-task, cross-condition review per the analysis discipline should
  be completed by the next oversight pass before the verdict is quoted in
  the paper.

## 2026-08-07 sampled pass@k / maj@k follow-up

The preregistered sampled readout `3962322_[0-5%3]` completed after the
initial aggregation. It replays the same accepted prompts with one greedy
generation and 16 T=0.8 samples (top-p 0.95, seed 20260806). Every one of the
six full runs passes its coverage/RoPE/prompt-truncation audit. Generated
summary: `$HPCVAULT/synthetic-RLVL/lm_eval_results/qwen25_dolmino_post_sft_passk_20260806/summary_passk.md`.

| macro | control | logic | nl_exact |
| --- | ---: | ---: | ---: |
| standard greedy | 0.4461 | 0.4339 | 0.4324 |
| standard pass@16 | 0.7588 | 0.7540 | 0.7621 |
| standard maj@16 | 0.5243 | 0.5145 | 0.5174 |
| tagged multi-hop pass@16 | 0.5150 | 0.5383 | 0.5283 |
| fallback multi-hop pass@16 | 0.5317 | 0.5417 | 0.5317 |
| sampled multi-hop tag rate | 0.6445 | 0.8346 | 0.8440 |

Correct and incorrect sampled generations were inspected for all five tasks
and all three conditions; prompt hashes match across conditions. The samples
include the already-diagnosed answer/prompt continuation fragments (for
example, repeated `Answer:` or `<answer>` instructions) and the strict
multi-hop ordering follows wrapper compliance. Consequently this one-run
follow-up supplies no robust sampled-reasoning advantage and leaves the
null-for-reasoning stop decision unchanged. It is a diagnostic supplement,
not an ICLR result table.
