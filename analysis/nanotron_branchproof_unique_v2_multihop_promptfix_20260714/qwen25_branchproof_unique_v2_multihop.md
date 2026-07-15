# Qwen2.5 BranchProof-v2 Multi-Hop QA

Context-provided LongBench evaluation; this does not test retrieval or proof validity.

The standard-short-answer rows reproduce the stock LongBench prompt and 32-token decoding. The strict-tagged rows test transfer to the synthetic `<answer>...</answer>` response contract.

| condition | mode | protocol | mean QA F1 | mean exact match | tag found |
| --- | --- | --- | ---: | ---: | ---: |
| control | direct | standard_short_answer | 0.189 | -- | -- |
| control | direct | strict_tagged | 0.312 | 0.235 | 0.983 |
| control | instruction | standard_short_answer | 0.097 | -- | -- |
| control | instruction | strict_tagged | 0.254 | 0.182 | 0.550 |
| nl_p15 | direct | standard_short_answer | 0.238 | -- | -- |
| nl_p15 | direct | strict_tagged | 0.069 | 0.057 | 0.293 |
| nl_p15 | instruction | standard_short_answer | 0.085 | -- | -- |
| nl_p15 | instruction | strict_tagged | 0.302 | 0.222 | 0.747 |
| logic_p15 | direct | standard_short_answer | 0.250 | -- | -- |
| logic_p15 | direct | strict_tagged | 0.003 | 0.002 | 0.012 |
| logic_p15 | instruction | standard_short_answer | 0.100 | -- | -- |
| logic_p15 | instruction | strict_tagged | 0.294 | 0.208 | 0.743 |

## Raw-generation audit

The direct strict-tagged prompt mostly triggers the learned continuation substrate: logic generations open `<formal>` in 98.5--99.0% of rows and natural-language generations open `<think>` in 97.0--99.0%. The 64-token diagnostic therefore usually ends before a usable answer. Instruction SFT removes those openings, but the 32-token stock protocol remains strongly cap-limited and frequently contains continuation artifacts. These rows measure response control as well as QA.

A diagnostic rescore truncates only obvious generated continuation after the first answer span. Averaged over the three direct standard tasks, control/logic/NL QA-F1 changes from 0.189/0.250/0.238 under stock scoring to 0.349/0.361/0.367 under this answer-head sensitivity check. The apparent stock gains therefore mostly collapse after continuation is removed and are not clean evidence of reasoning transfer.
