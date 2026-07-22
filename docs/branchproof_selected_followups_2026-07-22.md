# Selected Corrected BranchProof Follow-ups

Date: 2026-07-22

## Scope

The exhaustive corrected report matrix was pruned after mapping it against the
official preprint's claims. The retained protocol finishes the full corrected
OLMo-7B baseline and then evaluates only the controls needed to distinguish the
trace-substrate claim from length, shortcut, architecture, mixture, and model-
capacity explanations.

All selected follow-ups depend on successful baseline aggregate `3857769`.
The selected matrix has 48 GPU rows: 45 evaluations and three OLMo-3-32B
conditioned-dual SFT rows. A programmatic audit against
`scripts/experiments/branchproof_unique_v2_report_matrix.py` passed for every
index and verified that every selected checkpoint except the intentional three
32B conditioned rows already has a nonempty final adapter.

| Claim/control | Exact rows | Job | Rows |
| --- | --- | --- | ---: |
| Length/syntax | surface eval `0..2` symbol-padded logic, `6..8` terse NL, `24..26` target-token-matched NL; all train-1-to-25, three seeds | `3881774` | 9 eval |
| Shortcut robustness | shortcut eval `12..17`; schema shortcut rate `0.8`, logic/NL, three seeds, shortcut-neutral eval | `3881775` | 6 eval |
| Same-datapoint hybrid | hybrid eval `12..14` NL-then-formal and `27..29` formal-then-NL; train-1-to-25, three seeds | `3881776` | 6 eval |
| Conditioned dual at 7B | conditioned-10k eval `24..29`; formal/NL modes, train-1-to-25, three seeds | `3881777` | 6 eval |
| Independent architecture | architecture eval `24..26,33..35`; Qwen2.5-7B logic/NL, train-1-to-25, three seeds | `3881778` | 6 eval |
| Capacity-conditioned SFT | large train `12..14`; OLMo-3-32B conditioned dual, train-1-to-25, three seeds | `3881779` | 3 SFT |
| Capacity single-modality baselines | large eval `0..5`; OLMo-3-32B logic/NL, train-1-to-25, three seeds | `3881780` | 6 eval |
| Capacity conditioned readout | large eval `12..17`; OLMo-3-32B conditioned formal/NL modes, train-1-to-25, three seeds | `3881781` | 6 eval |

The capacity-conditioned eval also depends on successful rows
`3881779_12/13/14`. The full corrected baseline, corrected Tiny-model results,
and Attribute Constraints are retained separately and are not duplicated here.

After cancellation, incomplete nonselected checkpoint payloads were removed
only from six conditioned-50k and three batch-size roots with no final adapter
and no live/dependent job reference. This reclaimed about 22.5 GiB while
preserving run metadata, all completed finals, and every selected artifact.

## Deliberately Dropped

- Full batch-size matrix.
- Conditioned-50k convergence grid.
- Nonselected shortcut rates and shortcut kinds.
- Wordified, rule-annotated, pseudocode, shuffled, and invalid-trace surfaces.
- Hybrid and conditioned train-depth sweeps below train depth 25.
- Qwen2.5-1.5B/Gemma full architecture depth sweeps.
- Qwen3-32B corrected rerun.

These were useful historical diagnostics but are not necessary for the active
preprint claims. Completed artifacts remain available; cancellation does not
promote old ambiguous-generator results back into evidence.
