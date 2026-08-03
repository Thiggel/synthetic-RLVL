# Dolmino 5B Terminal Direct Readout: Control and Formal Partial

Date: 2026-08-03

This is a provisional two-condition readout at exact step 9,537
(`5,000,134,656` tokens). The matched natural-language endpoint is not yet
available, so these numbers must not be used as the final three-way result.

## Acceptance

Jobs `3944016_0/1` completed `0:0`. Both checkpoints passed the complete
TP4/DP2 Nanotron structure and accounting gate, local HF conversion and finite
forward pass, RoPE/context checks, retained-sample audits, and schema-v4 MATH
symbolic scoring. Each standard run retained 10,600 rows over 105 leaf tasks;
each multi-hop run retained 600 rows.

## Limit-100 standard suite

The macro uses the same ten primary metrics as the step-5,000 readout, with
the post-hoc symbolic MATH score replacing stock string exact match.

| Task | Control | Formal | Formal - control |
| --- | ---: | ---: | ---: |
| GSM8K | 0.8200 | 0.8000 | -0.0200 |
| ARC-Challenge | 0.6000 | 0.5600 | -0.0400 |
| HellaSwag | 0.6500 | 0.6800 | +0.0300 |
| WinoGrande | 0.7700 | 0.7700 | 0.0000 |
| PIQA | 0.8400 | 0.8200 | -0.0200 |
| LogiQA | 0.4100 | 0.3900 | -0.0200 |
| BBH | 0.6756 | 0.6685 | -0.0070 |
| MMLU | 0.7081 | 0.7026 | -0.0054 |
| MMLU-Pro | 0.4429 | 0.4479 | +0.0050 |
| MATH-500 symbolic | 0.3300 | 0.3300 | 0.0000 |
| **Macro** | **0.6246** | **0.6169** | **-0.0077** |

The terminal standard-suite result does not show an aggregate formal gain.
This limited evaluation has roughly 100 examples per leaf task and should be
reported with its sampling uncertainty rather than interpreted from the
0.77-point macro difference alone.

## Multi-hop QA

| Prompt/scorer | Control | Formal | Formal - control |
| --- | ---: | ---: | ---: |
| 2Wiki standard QA-F1 | 0.1078 | 0.1386 | +0.0308 |
| HotpotQA standard QA-F1 | 0.1145 | 0.1165 | +0.0020 |
| MuSiQue standard QA-F1 | 0.0708 | 0.0788 | +0.0081 |
| 2Wiki tagged QA-F1 | 0.2811 | 0.2865 | +0.0054 |
| HotpotQA tagged QA-F1 | 0.4404 | 0.4937 | +0.0533 |
| MuSiQue tagged QA-F1 | 0.1810 | 0.2210 | +0.0399 |

Formal improves all six multi-hop cells. Tagged exact match also changes
from `0.23/0.33/0.11` to `0.24/0.40/0.14` for
2Wiki/HotpotQA/MuSiQue. Tag emission is `0.99/0.95/0.91` for control and
`1.00/1.00/1.00` for formal.

## Raw-generation audit

Representative GSM8K generations from both models are coherent and end in
the expected `####` answer. Representative MATH outputs are answer-first and
the symbolic sidecar rescues mathematically equivalent answers without losing
any stock-exact positives; both conditions score `33/100`.

The standard multi-hop prompt remains a poor interface readout. Many outputs
give one bare answer and then continue with a new `Question:`/`Answer:` pair,
for example both control and formal continue after the first 2Wiki answer.
This continuation contaminates whole-response QA-F1 even when the first answer
is useful. The explicitly tagged variant supplies a reliable extraction
boundary and is therefore the cleaner current comparison. It is still a
prompt intervention, so both variants should remain visible.

## Current interpretation and gate

- Terminal formal is slightly below control on the limited ten-task macro.
- Terminal formal is consistently above control on the multi-hop set,
  especially tagged HotpotQA and MuSiQue.
- These are not yet a complete treatment comparison. Freeze interpretation
  until terminal NL receives the identical readout.
- Keep direct evaluation separate from the post-SFT interface experiment.
  The latter tests whether a matched instruction stage exposes retained
  capability while restoring response control.

Artifact root:
`$HPCVAULT/synthetic-RLVL/lm_eval_results/qwen25_dolmino_terminal_5b_20260803`
