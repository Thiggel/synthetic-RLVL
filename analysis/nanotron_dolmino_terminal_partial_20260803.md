# Dolmino 5B Terminal Direct Readout: Control, Formal, and Matched NL

Date: 2026-08-05

This is the complete three-condition direct readout at exact step 9,537
(`5,000,134,656` tokens). It remains separate from the identical post-SFT
readout, which is the primary modern-interface comparison.

## Acceptance

Jobs `3944016_0/1` and `3944017_2` completed `0:0`. All three checkpoints
passed the complete
TP4/DP2 Nanotron structure and accounting gate, local HF conversion and finite
forward pass, RoPE/context checks, retained-sample audits, and schema-v4 MATH
symbolic scoring. Each standard run retained 10,600 rows over 105 leaf tasks;
each multi-hop run retained 600 rows. The matched-NL terminal checkpoint has
645 nonempty files and exact `4,750,127,104 + 250,007,552` Dolmino/NL token
accounting.

## Limit-100 standard suite

The macro uses the same ten primary metrics as the step-5,000 readout, with
the post-hoc symbolic MATH score replacing stock string exact match. As in the
existing control/formal table, ARC-Challenge and HellaSwag use normalized
accuracy rather than raw accuracy.

| Task | Control | Formal | Matched NL | Formal - control | NL - control |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K | 0.8200 | 0.8000 | 0.8300 | -0.0200 | +0.0100 |
| ARC-Challenge | 0.6000 | 0.5600 | 0.5600 | -0.0400 | -0.0400 |
| HellaSwag | 0.6500 | 0.6800 | 0.6600 | +0.0300 | +0.0100 |
| WinoGrande | 0.7700 | 0.7700 | 0.7500 | 0.0000 | -0.0200 |
| PIQA | 0.8400 | 0.8200 | 0.8100 | -0.0200 | -0.0300 |
| LogiQA | 0.4100 | 0.3900 | 0.4000 | -0.0200 | -0.0100 |
| BBH | 0.6756 | 0.6685 | 0.6707 | -0.0070 | -0.0048 |
| MMLU | 0.7081 | 0.7026 | 0.7026 | -0.0054 | -0.0054 |
| MMLU-Pro | 0.4429 | 0.4479 | 0.4557 | +0.0050 | +0.0129 |
| MATH-500 symbolic | 0.3300 | 0.3300 | 0.3100 | 0.0000 | -0.0200 |
| **Macro** | **0.6246** | **0.6169** | **0.6149** | **-0.0077** | **-0.0097** |

The terminal standard-suite result shows neither a formal nor matched-NL
aggregate gain over control.
This limited evaluation has roughly 100 examples per leaf task and should be
reported with its sampling uncertainty rather than interpreted from sub-point
macro differences alone.

## Multi-hop QA

| Prompt/scorer | Control | Formal | Matched NL |
| --- | ---: | ---: | ---: |
| 2Wiki standard QA-F1 | 0.1078 | 0.1386 | 0.1257 |
| HotpotQA standard QA-F1 | 0.1145 | 0.1165 | 0.1155 |
| MuSiQue standard QA-F1 | 0.0708 | 0.0788 | 0.0784 |
| 2Wiki tagged QA-F1 | 0.2811 | 0.2865 | 0.2697 |
| HotpotQA tagged QA-F1 | 0.4404 | 0.4937 | 0.4548 |
| MuSiQue tagged QA-F1 | 0.1810 | 0.2210 | 0.2173 |

Formal improves all six multi-hop cells over control, while matched NL lies
between them in five of six cells and below both on tagged 2Wiki. Tagged exact
match is `0.23/0.33/0.11`, `0.24/0.40/0.14`, and `0.22/0.36/0.14` for
control/formal/NL across 2Wiki/HotpotQA/MuSiQue. NL tag emission is
`1.00/1.00/0.99`.

## Raw-generation audit

Representative GSM8K generations from all three models are coherent and end in
the expected `####` answer. Representative MATH outputs are answer-first and
the symbolic sidecar rescues mathematically equivalent answers without losing
any stock-exact positives; control/formal/NL score `33/33/31` out of 100.

The standard multi-hop prompt remains a poor interface readout. Many outputs
give one bare answer and then continue with a new `Question:`/`Answer:` pair,
for example both control and formal continue after the first 2Wiki answer.
Matched NL reproduces the same stopping failure: `95/95/95%` of stock
2Wiki/HotpotQA/MuSiQue generations contain another `Question:` marker. This
continuation contaminates whole-response QA-F1 even when the first answer
is useful. The explicitly tagged variant supplies a reliable extraction
boundary and is therefore the cleaner current comparison. It is still a
prompt intervention, so both variants should remain visible.

Across the complete standard retained samples, BBH invalid extraction is
`3.44/3.70/3.70%` and MMLU-Pro invalid extraction is
`3.93/3.86/3.21%` for control/formal/NL. No response or prompt in the audited
GSM8K, MATH, BBH, or MMLU-Pro families contains the assistant-preamble
next-document marker. The longest MMLU-Pro generations remain large
(`12,132/11,160/11,148` characters), but the formal and NL treatments do not
increase the invalid-extraction rate at this terminal direct readout.

## Current interpretation and gate

- Terminal formal and matched NL are both slightly below control on the
  limited ten-task macro and are nearly identical to each other.
- Formal is consistently above control on the multi-hop set, especially tagged
  HotpotQA and MuSiQue, but matched NL largely moves in the same direction.
- The direct readout therefore does not show a broad formal-specific transfer
  advantage and retains a shared long-document stopping regression.
- Keep direct evaluation separate from the post-SFT interface experiment.
  The latter tests whether a matched instruction stage exposes retained
  capability while restoring response control.

Artifact root:
`$HPCVAULT/synthetic-RLVL/lm_eval_results/qwen25_dolmino_terminal_5b_20260803`
