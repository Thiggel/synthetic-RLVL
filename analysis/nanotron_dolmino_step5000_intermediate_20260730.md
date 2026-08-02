# Dolmino step-5000 intermediate readout

Date: 2026-07-30; matched-NL gate added 2026-08-03

This is a matched, limited (`100` examples per task) direct readout of the
accepted control and formal-5% Nanotron checkpoints at step 5000
(`2,621,440,000` consumed tokens). It is not the terminal 5B result.

## Artifact gate

- Slurm array `3913651_[0-1%2]` completed inference for both rows.
- Both local HF conversions passed file, Qwen2.5 RoPE-`1000000`, CUDA-load,
  and finite-forward checks.
- Each standard bundle has 105 leaf sample files and 10,600 retained rows.
  Each multi-hop bundle has six task files and 600 retained rows.
- Formal row 1 initially exited nonzero because MATH-500 sample 67 answered
  `$10\%$` for gold `10`: stock MATH normalization removes `\%`, but the
  symbolic sidecar did not. The sidecar now applies the same terminal-percent
  normalization before `math_verify`; focused tests pass (`19 passed`).
  CPU-only forced rescoring and production re-audit accept both retained
  bundles with zero lost stock-exact positives. No GPU inference was rerun.

## Limited reviewer-suite result

| Task | Control | Formal 5% | Delta |
| --- | ---: | ---: | ---: |
| GSM8K | 0.7400 | 0.8500 | +0.1100 |
| ARC-Challenge | 0.5600 | 0.5300 | -0.0300 |
| HellaSwag | 0.6600 | 0.7000 | +0.0400 |
| WinoGrande | 0.6900 | 0.7600 | +0.0700 |
| PIQA | 0.8200 | 0.8200 | +0.0000 |
| LogiQA | 0.4200 | 0.4700 | +0.0500 |
| BBH | 0.6167 | 0.6819 | +0.0652 |
| MMLU | 0.6844 | 0.7246 | +0.0402 |
| MMLU-Pro | 0.4614 | 0.4743 | +0.0129 |
| MATH-500 sidecar | 0.2800 | 0.3300 | +0.0500 |
| Ten-task macro | 0.5932 | 0.6341 | +0.0408 |

This one-checkpoint, limit-100 macro is encouraging but provisional and is not
a trigger for broader training. Correct and incorrect GSM8K, MATH-500, BBH,
and MMLU-Pro generations were inspected in both conditions. Prompts contain no
next-document marker. BBH invalid extraction changes from `6.30%` to `5.44%`;
MMLU-Pro changes from `2.50%` to `4.86%`. Neither condition generated the
previous p15 `You are an AI assistant` next-document marker in these families.

## Multi-hop raw-generation finding

Stock short-answer QA-F1 macro changes from `0.3171` to `0.1093`, while the
strict tagged macro changes from `0.3104` to `0.3348`. The stock decline is a
response-control artifact, not clean evidence of worse multi-hop reasoning:
formal-5% generations continue into a new `Question:`/`Answer:` record in
`91%`, `94%`, and `97%` of 2Wiki, HotpotQA, and MuSiQue rows, compared with
`33%`, `28%`, and `49%` for control. Tagged prompts suppress that continuation
and keep `<answer>` opening coverage at `100%` (control MuSiQue is `99%`).

Manual review covered correct, partial, and incorrect outputs for all six
condition/protocol cells. It confirmed intact prompts and targets, ordinary
wrong entity selection, and the stock continuation artifact. No learned
`<formal>` or `<think>` opening and no next-document assistant marker appeared
in these limited multi-hop rows.

## Extended generation audit

The July 31 audit inspected all 1,200 retained multi-hop generations and all
text generations in the 10,600-row standard bundles. The formal checkpoint
does not emit formal syntax or the injected `Solution:`, `Context:`,
`Derivation:`, `Conclusion:`, or `Final answer:` envelope on these tasks.
Those markers occur in zero of the 600 formal multi-hop generations and are
also effectively absent from GSM8K, BBH, MATH-500, and MMLU-Pro. The observed
format change is instead a stopping-boundary failure after a bare answer.

### Multi-hop results by benchmark

| Benchmark | Protocol | Control F1 | Formal F1 | Delta |
| --- | --- | ---: | ---: | ---: |
| 2WikiMultiHopQA | stock | 0.3624 | 0.1271 | -0.2353 |
| HotpotQA | stock | 0.3670 | 0.1454 | -0.2216 |
| MuSiQue | stock | 0.2218 | 0.0556 | -0.1663 |
| 2WikiMultiHopQA | tagged | 0.3295 | 0.3402 | +0.0107 |
| HotpotQA | tagged | 0.4186 | 0.4774 | +0.0588 |
| MuSiQue | tagged | 0.1830 | 0.1867 | +0.0037 |

The stock task has no generation stop string and relies on model EOS after a
short answer. Formal QA-record continuation rates are `91%`, `94%`, and `97%`
for 2Wiki, HotpotQA, and MuSiQue, versus `33%`, `28%`, and `49%` for control.
This happens independently of answer quality: among exact first answers, the
formal continuation rates remain `87.5%`, `92.9%`, and `100%`.

Scoring only the first generated line gives:

| Benchmark | Control first-head F1 | Formal first-head F1 | Delta |
| --- | ---: | ---: | ---: |
| 2WikiMultiHopQA | 0.4072 | 0.3689 | -0.0383 |
| HotpotQA | 0.4887 | 0.5541 | +0.0654 |
| MuSiQue | 0.2719 | 0.2168 | -0.0551 |
| Macro | 0.3893 | 0.3799 | -0.0094 |

Thus formatting explains most of the catastrophic stock-score drop, but it
does not reveal a broad hidden multi-hop gain. HotpotQA improves under both
first-head and tagged scoring; 2Wiki and MuSiQue are flat-to-lower at this
sample size. The tagged macro gain of `+0.0244` is driven mainly by HotpotQA.

Representative failures make the scorer interaction explicit:

- HotpotQA formal: `Charles L. Clifford` is the exact answer, followed by a
  new generated `Question:`. Stock F1 is only `0.25`; the matched tagged
  generation is `<answer>Charles L. Clifford`.
- 2Wiki formal: `no` is the exact answer, followed by two new QA records.
  Stock F1 is `0.095`.
- MuSiQue formal: `1912` is the exact answer, followed by a new unrelated QA
  record. Stock F1 is `0.118`; the tagged generation is `<answer>1912`.

The retained tagged strings normally omit the literal closing tag because
`</answer>` is configured as a generation stop string and lm-eval removes the
matched stop text. This is not evidence that the model failed to close every
tag.

### Other generated tasks

The standard suite shows real answer improvements without learned-envelope
leakage. Formal corrects representative arithmetic, ordering, and
multiple-choice reasoning errors, including a GSM8K house-profit calculation,
a BBH five-object ordering, a BBH multistep arithmetic calculation, and
MMLU-Pro math and computer-science items. Aggregate generated-task changes are
GSM8K `0.7400 -> 0.8500`, BBH `0.6167 -> 0.6819`, MATH-500 sidecar
`0.2800 -> 0.3300`, and MMLU-Pro `0.4614 -> 0.4743`.

There is nevertheless a second response-control problem in the MMLU-Pro tail.
Invalid extracted choices increase from `35/1400` (`2.50%`) to `68/1400`
(`4.86%`). Invalid generations are long repetitive responses: their median
length is about 5.5K characters under formal training, compared with about
0.24K for valid formal responses. The formal MMLU-Pro response-length p95
increases from 1,186 to 2,113 characters. BBH p95 also increases from 1,454
to 1,969 characters, although BBH accuracy and empty-extraction counts
improve. This is a tail-risk increase, not uniform verbosity.

ARC-Challenge, HellaSwag, WinoGrande, PIQA, and MMLU are scored through
multiple-choice likelihood rather than free-form answer extraction. In
particular, the MMLU change `0.6844 -> 0.7246` cannot be explained by output
formatting. The limited standard-suite gain therefore contains a competence
signal, even though this single checkpoint and limit-100 sample cannot
establish a final transfer claim.

### Likely mechanism

The current proof source uses the same modality-neutral outer envelope for
formal and NL records, but the implemented envelope is more elaborate than
the originally proposed minimal format:

```text
{full problem and premise list}

Solution:
Context:
{declarations and a second copy of the premises}

Derivation:
{trace}

Conclusion:
{conclusion}

Final answer: {answer}
```

It applies causal-LM loss to the entire document. Formal records average about
3,816 tokens (`550,000,813 / 144,136`); matched NL records average about
3,875 tokens (`550,000,176 / 141,932`), while the exported Dolmino records
average about 456 tokens (`5,100,006,129 / 11,195,395`). Many proof chunks
therefore contain premise/declaration continuation but no answer boundary.
The intervention teaches EOS after `Final answer:`, not after the bare
`Answer:` used by stock LongBench. Replacing 5% of Dolmino tokens with these
long documents also reduces global document-boundary density by roughly 4.4%.

This is a plausible indirect cause of poorer stopping, but it does not yet
identify formal symbols as the cause. Formal and NL source records have nearly
the same length and exactly the same outer envelope. The matched NL
step-5000 evaluation is therefore the decisive control:

- if NL shows the same continuation increase, the cause is the long
  full-document objective/envelope;
- if NL retains control-like stopping, the formal solution contents or their
  gradients are implicated;
- if both improve after identical answer-only calibration, the issue is
  downstream response alignment rather than lost reasoning ability.

## Matched-NL causal gate

The exact matched-NL step-5000 readout completed as `3942598_2` on 2026-08-02
and passed the same local-HF, RoPE-`1000000`, retained-sample, multi-hop,
standard-suite, and schema-v4 MATH sidecar gates. The bundle contains 600
multi-hop rows and 10,600 standard-suite rows.

The ten-task macro is `0.6327`, compared with `0.5932` for control and
`0.6341` for formal. Formal-minus-NL is only `+0.0014` at this limited
readout. The per-task matched-NL values are GSM8K `0.8300`, ARC-Challenge
`0.5500`, HellaSwag `0.6900`, WinoGrande `0.7600`, PIQA `0.8200`, LogiQA
`0.4700`, BBH `0.6789`, MMLU `0.7235`, MMLU-Pro `0.4743`, and the MATH-500
sidecar `0.3300`. Thus the encouraging control-to-intervention competence
change is shared by formal and matched NL at this checkpoint; it is not a
formal-modality advantage.

The stopping diagnostic is equally decisive. Matched-NL stock continuation
rates are `93%/94%/96%` on 2Wiki/HotpotQA/MuSiQue, versus formal
`91%/94%/97%` and control `33%/28%/49%`. Among exact first answers, matched-NL
continuation remains `90.6%/95.0%/100%`. Its stock QA-F1 macro is `0.1067`,
tagged QA-F1 macro is `0.3420`, and first-answer macro is `0.3863`; the formal
values are `0.1093/0.3348/0.3799`. This near identity supports the shared
long-document/full-loss envelope and boundary-density mechanism, not formal
syntax, as the cause of bare-answer continuation.

Raw review covered correct and incorrect GSM8K, MATH-500, BBH, MMLU-Pro, and
stock/tagged multi-hop generations. Prompts and extraction match the intended
protocol. Correct rows contain ordinary task reasoning; incorrect rows contain
genuine arithmetic, entity-selection, or option-selection errors. Tagged
outputs obey the answer boundary, while stock rows often continue into a new
QA record. BBH invalid extraction is `5.56%` under matched NL, close to formal
`5.44%` and below control `6.30%`. MMLU-Pro invalid extraction is `5.00%`,
close to formal `4.86%` and above control `2.50%`; the invalid rows are long
repetitive tails. No `You are an AI assistant` next-document marker appears in
matched-NL BBH or MMLU-Pro generations.

## Decision

Retain this as an intermediate optimization/readout diagnostic. Do not update
the official preprint or launch a broader mixture grid from it. The matched-NL
gate has now shown that both the competence gain and stopping regression are
shared intervention effects at step 5000, not evidence for a formal-modality
advantage. Continue the preregistered control/formal/NL 5B gate and require
terminal checkpoint, direct, post-midtraining readout, and raw-generation
audits before a scientific transfer claim. Use identical answer-only and
neutral reasoning calibration for all three terminal checkpoints. For a later
pilot, compare the current full-document objective with a minimal envelope and
a proof-focused loss that masks copied problem/premise tokens.
