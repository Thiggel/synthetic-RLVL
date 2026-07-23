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

Update 20:15 CEST: baseline aggregate `3857769` passed and released the wave.
Surface row `3881774_0` started on verified A100-80GB hardware with clean
merge/vLLM startup; other selected rows remain scheduler/dependency pending.

Update 2026-07-23 01:17 CEST: eleven selected eval rows are running cleanly at
sampled chunks `55..89/112`. Conditioned OLMo-3-32B SFT `3881779_12` exposed
an operational defect: its measured roughly 31-hour runtime exceeded the
24-hour allocation, while the stored 20,000-step checkpoint interval could
not save during a 10,000-step run. The `large` wrapper now retains two states
at 250-step intervals. Original `3881779` was canceled and replaced exactly by
`3883534_[12-14%1]`; conditioned eval `3881781` depends on the replacement.

Update 2026-07-23 07:16 CEST: surface `3881774_0/2` and conditioned-7B
`3881777_25` completed `0:0` and pass row-level production audits plus
representative raw review. Each has 448 prompts, 16 generations, all 14
depths, 576 retained rows with 448 sampled rows, complete chunk logs, and no
credited validity-diagnostic contradiction. Symbol-padded formal retains
answer-correct invalid long traces and depth-50 repetition/truncation;
conditioned NL has clean translated-valid shallow/OOD examples but `32/32`
retained depth-50 format failures after long copying. These partials are not
family evidence. Eight rows remain active, and replacement `3883534` remains
A100-80 pending with no stale checkpoint/final in its first run root.

Update 2026-07-23 08:27 CEST: surface `3881774_1`, shortcut
`3881775_12..14`, and conditioned-7B `3881777_24` also completed `0:0`,
bringing selected eval completion to `8/45`. These five newer rows remain
provisional until production artifact/invariant gates and representative raw
review are recorded. Hybrid `3881776_12..14` is healthy at approximately
`99/99/98` of `112` sampled chunks. No selected row has failed, and no
partial-family result is eligible for the report.

Update 2026-07-23 13:24 CEST: all eleven completed selected rows pass their
production and representative raw-generation gates. Surface symbol-padded
formal `3881774_0..2` and NL-then-formal hybrid `3881776_12..14` are complete
three-seed families. Surface OOD answer/joint pass@1 is `0.673 +/- 0.175` /
`0.626 +/- 0.176`; hybrid is `0.002 +/- 0.001` / `0.000`. The hybrid fits
depth 25 but copies both trace surfaces and reaches the shared cap without an
answer OOD. These two controls are now report evidence. Shortcut formal
`3881775_12..14` and the conditioned-7B seed-3407 pair
`3881777_24/25` remain partial. OLMo-3-32B logic eval `3881780_0` is healthy
on four A100-80GB GPUs, while conditioned-32B replacement SFT `3883534`
remains pending.

Update 2026-07-23 19:17 CEST: OLMo-3-32B logic row `3881780_0`, raw job
`3883993`, completed in `06:34:03` on four A100-80GB GPUs and passed the same
production/invariant/raw-generation gate. It is perfect in the retained
sample through depth 25 and then shows ordinary wrong-branch, malformed, and
long-generation failures. One-seed OOD greedy/pass@1/pass@16 answer is
`0.738/0.641/0.994`, with citation-free joint `0.491/0.504/0.919`. This is
only the first of six matched single-modal 32B rows, so it does not open a
family claim or report refresh. Selected acceptance is now `12/45`;
`3881780_1` and three conditioned-7B rows continue without fatal signatures.

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
