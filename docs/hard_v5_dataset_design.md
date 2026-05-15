# Hard-v5 Dataset Design

Status: implemented on 2026-05-03.

## Goal

Hard-v5 is designed to make answer correctness difficult to learn from a shallow shortcut while keeping the target proof format short enough for SFT and GRPO. It isolates the reward signal we care about:

- `correctness`: final answer string matches the gold state.
- `citation_free_validity`: the proof formulas are derivable in order from the premises, ignoring explicit line citations.
- `line_valid_fraction`: fraction of proof lines whose formulas are derivable in order.

The target proofs intentionally omit citations. This removes citation bookkeeping as the dominant failure mode while preserving faithful logic checking through the logic engine.

## Algorithm

For depth `d`, the generator samples a sequence of `d + 1` normal state words, for example `willow -> ivory -> teal -> violet`. It uses parser-safe one-letter predicates in the formal block and maps them to normal words in `<predicates>`.

Each example has:

- Compact constants: `a = a`, `b = b`, ... . The logic parser treats `s-z` as variables, so hard-v5 uses constants `a-r` and wraps for depths above 17.
- Initial facts: state of the first entity and an `active` marker.
- At every step, two locally plausible branches:
  - true branch: `active(source) & state_i(source) -> state_{i+1}(target)`
  - shortcut branch: `dormant(source) & state_i(source) -> wrong_state(target)`
- Active propagation for non-final steps: `state_{i+1}(target) -> active(target)`
- Gold proof lines with no citations, for example `Bb ; ->E`.

Training materialization uses `shortcut_rate=0.8`: in 80% of examples, the true branch is listed first, so a model can learn a brittle "take the first branch" heuristic. Validation uses `shortcut_rate=0.0`: the shortcut branch is listed first and fails unless the model tracks the active marker and constructs a valid chain.

## Why This Is Scientifically Useful

The intended comparison is whether validity reward helps the model abandon the shortcut faster than answer-only reward. The no-validity baseline can sometimes get high training correctness by selecting the first branch. On validation, that shortcut is adversarially flipped. A citation-free validity reward directly rewards the invariant rule: only the branch whose marker is actually derivable should be used.

This mirrors real shortcut-learning settings:

- Geirhos et al. describe shortcut learning as models using decision rules that work on standard benchmarks but fail under distribution shift: https://www.nature.com/articles/s42256-020-00257-z
- Gururangan et al. show annotation artifacts in NLI that allow high apparent performance without solving the intended premise-hypothesis reasoning problem: https://arxiv.org/abs/1803.02324
- McCoy, Pavlick, and Linzen's HANS benchmark shows NLI models can rely on syntactic heuristics that fail on controlled counterexamples: https://arxiv.org/abs/1902.01007
- Jia and Liang show reading-comprehension models can fail when adversarial distractor sentences break shallow matching heuristics: https://arxiv.org/abs/1707.07328

Hard-v5 is the same experimental pattern in a fully controlled formal-logic environment: the train shortcut is measurable, the eval shortcut is flipped to zero, and proof validity is mechanically checkable.

## Example SFT Datapoint

This is an exact generated depth-3 example from `DatasetConfig(depth=3, difficulty="hard_v5", shortcut_rate=0.8, seed=3407)`.

Prompt:

```text
<question>
1. a is willow.
2. a is active.
3. If a is dormant and a is willow, then b is teal.
4. If a is active and a is willow, then b is ivory.
5. If b is ivory, then b is active.
6. If b is dormant and b is ivory, then c is violet.
7. If b is active and b is ivory, then c is teal.
8. If c is teal, then c is active.
9. If c is dormant and c is teal, then d is hazel.
10. If c is active and c is teal, then d is violet.
Which state applies to d?
</question>
```

Target:

```text
<formal>
<constants>
a = a
b = b
c = c
d = d
</constants>
<predicates>
Ax: x is willow
Bx: x is ivory
Cx: x is teal
Dx: x is violet
Ex: x is hazel
Fx: x is active
Gx: x is dormant
</predicates>
<premises>
Aa
Fa
Ga & Aa -> Cb
Fa & Aa -> Bb
Bb -> Fb
Gb & Bb -> Dc
Fb & Bb -> Cc
Cc -> Fc
Gc & Cc -> Ed
Fc & Cc -> Dd
</premises>
<proof>
Aa ; R
Fa ; R
Bb ; ->E
Fb ; ->E
Cc ; ->E
Fc ; ->E
Dd ; ->E
</proof>
<conclusion>
Dd
</conclusion>
</formal>
<answer>
violet
</answer>
```

The shortcut branch is first in every step in this example, but it requires `dormant`, which is never derivable. The gold proof follows the active branch and is valid under citation-free proof analysis.

## Dataset Splits

HF repo: `flaitenberger/LogicalReasoning-hard-v5`

Planned materialized configs:

- `train_up_to_3_1k`: SFT, depths `1..3`, 1,000 rows, `shortcut_rate=0.8`.
- `train_up_to_15_120k`: GRPO, depths `1..15`, 120,000 rows, `shortcut_rate=0.8`.
- `val_step_01_1k` through `val_step_20_1k`: validation, fixed depth per split, `shortcut_rate=0.0`.

## Rewards

Hard-v5 reward ablation uses three seeds and five schemas:

- `correct_plus_0p1_format`
- `correct_plus_citation_free_valid_plus_0p1_format`
- `correct_times_citation_free_valid_plus_0p1_format`
- `correct_plus_citation_free_line_valid_plus_0p1_format`
- `correct_times_citation_free_line_valid_plus_0p1_format`

Primary metrics are `correct_pass@k`, `citation_free_valid_pass@k`, `citation_free_joint_pass@k`, and `citation_free_valid_given_correct@k`, split by step and by train/OOD/hard-tail bands.
