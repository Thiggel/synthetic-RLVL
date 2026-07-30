# Dolmino step-5000 intermediate readout

Date: 2026-07-30

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

## Decision

Retain this as an intermediate optimization/readout diagnostic. Do not update
the official preprint or launch a broader mixture grid from it. Continue the
preregistered control/formal/NL 5B gate and require terminal checkpoint,
direct, post-midtraining readout, and raw-generation audits before a scientific
transfer claim.
