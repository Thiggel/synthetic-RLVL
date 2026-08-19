# Docpack 2.5B rerun: three-way multi-hop readout

Pooled n per condition/protocol: 600

| protocol | control | logic | nl_exact |
| --- | --- | --- | --- |
| standard | 0.4212 | 0.4161 | 0.4191 |
| tagged | 0.2968 | 0.2837 | 0.3274 |
| tagged_fallback | 0.3898 | 0.3793 | 0.3755 |

| condition | tag_found rate |
| --- | --- |
| control | 0.6283 |
| logic | 0.7600 |
| nl_exact | 0.8133 |

Paired example-level bootstrap (10000 resamples, seed 3407):

| protocol | contrast | delta | ci_low | ci_high | significant |
| --- | --- | --- | --- | --- | --- |
| standard | logic-control | -0.0050 | -0.0249 | +0.0144 | no |
| standard | logic-nl_exact | -0.0030 | -0.0198 | +0.0130 | no |
| standard | nl_exact-control | -0.0021 | -0.0211 | +0.0168 | no |
| tagged | logic-control | -0.0131 | -0.0410 | +0.0147 | no |
| tagged | logic-nl_exact | -0.0437 | -0.0682 | -0.0186 | yes |
| tagged | nl_exact-control | +0.0307 | +0.0017 | +0.0596 | yes |
| tagged_fallback | logic-control | -0.0105 | -0.0360 | +0.0153 | no |
| tagged_fallback | logic-nl_exact | +0.0039 | -0.0170 | +0.0256 | no |
| tagged_fallback | nl_exact-control | -0.0143 | -0.0403 | +0.0114 | no |

## Raw-generation inspection (load-bearing cell, hotpotqa tagged)

Per-condition, n=200: no-tag counts control/logic/nl_exact = 58/40/29, of which
22/25/12 score strict-0 but recover qa_f1 > 0.5 under the fallback extractor.
The failure mode is identical to the accepted 5B readout: the model emits the
correct bare answer without the instructed wrapper and is scored 0.
Representative case (all three conditions): gold `Miller v. California`,
generation `Miller v. California`, strict qa_f1 = 0.

## Verdict

The document-preserving objective does NOT rescue formal-logic transfer.

- Standard protocol: flat, all pairwise contrasts n.s.
- Tagged raw: logic-control -0.0131 n.s.; logic is SIGNIFICANTLY BELOW NL
  (-0.0437 [-0.0682, -0.0186]); nl-control +0.0307 significant.
- Tagged compliance-corrected (fallback): ALL pairwise contrasts n.s.
  (0.3898 / 0.3793 / 0.3755).

The tagged logic-control gap seen at 5B (+0.0773, significant) is GONE in the
document-preserving rerun (-0.0131, n.s.), while the compliance-corrected null
reproduces. Compliance transfer persists (tag rate 0.6283 -> 0.7600 / 0.8133),
but the ordering has flipped relative to 5B: NL now has the highest tag rate,
where at 5B logic (~0.83-0.85) slightly exceeded NL (~0.77-0.85).

This closes the P0 diagnosis NEGATIVELY: the 5B null was not an artifact of
mid-document splitting. Fixing the loader so proof documents are packed whole
does not produce a formal-logic reasoning gain.

## Limitations

- Greedy/pass@1 only; the sampled pass@k readout (4043881) has NOT run. Per
  standing practice, greedy systematically understates the logic condition, so
  this verdict is provisional until pass@k lands.
- The ten-task standard macro is NOT included: logic and nl_exact standard
  evals were still running at the time of writing.
- Single training run per condition; uncertainty is example-level only.
- Rerun corpus is capped at depths 1-14; the deep regime where logic wins
  synthetically is not represented.
- The control arm's midtrain terminal retained model weights only, so its base
  checkpoint passed a narrowed gate (state-shard expectations set to 0). Model
  weights were verified byte-identical in manifest to the accepted logic tree.
