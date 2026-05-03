# Hard-v2 Logic Dataset

Hard-v2 is an adversarial variant of the synthetic logic dataset. It preserves the old `standard` generator and is enabled explicitly with `task.difficulty=hard_v2` or `DatasetConfig(difficulty="hard_v2")`.

## What Changes

Hard-v2 makes shortcutting less reliable by adding:

- Branching rules: each gold source predicate has several plausible non-gold outgoing rules.
- Near-miss rules: rules differ from gold by entity, antecedent, or missing side support.
- Wrong-entity chains: full-looking chains for non-query entities.
- Missing-support conjunctions: conjunction rules where one antecedent is absent for the query entity.
- Answer decoys: wrong queried-family values are made salient for other entities.
- Side subproofs: conjunction side facts may be derived through short subchains instead of appearing only as direct premises.

The gold proof is still generated and validated by the same logic engine. Metadata records the hard counts under `metadata["hard_counts"]`.

## Preset

`hard_v2` currently means:

```yaml
task:
  difficulty: hard_v2
  branching_factor: 4
  decoy_chains: 3
  near_miss_ratio: 0.75
  side_chain_depth: 2
  entity_decoy_ratio: 1.0
  answer_decoy_ratio: 1.0
```

Hard-v2 prompts are much longer than standard prompts. Use the hard configs:

- `conf/sft_hard_v2.yaml`
- `conf/posttrain_grpo_hard_v2.yaml`

These raise sequence budgets and use the Hugging Face dataset repo by default:

```text
flaitenberger/LogicalReasoning-hard-v2
```

The Slurm build job still writes a local staging copy to `$WORK/synthetic-RLVL/datasets/materialized_logic_hard_v2` before pushing.

## Build Data

```bash
source ./scripts/env.sh
$HPCVAULT/.venv_rlvl_posttrain/bin/python -m synthrlvl.datasets.materialize \
  --output-root "$WORK/synthetic-RLVL/datasets/materialized_logic_hard_v2" \
  --difficulty hard_v2 \
  --train-up-to-5-rows 1000000 \
  --train-up-to-10-rows 1000000 \
  --val-rows-per-step 1000
```

Slurm wrapper, including HF upload:

```bash
HF_REPO_ID=flaitenberger/LogicalReasoning-hard-v2 PUSH_TO_HUB=1 sbatch scripts/slurm/jobs/build_materialized_hard_v2.slurm
```

Current build/push job: `3565609`.

## Reward Sweep

Planned hard-v2 reward schemas:

- `correct_plus_0p1_format`
- `correct_plus_valid_plus_0p1_format`
- `correct_times_valid_plus_0p1_format`
- `correct_plus_line_valid_plus_0p1_format`
- `correct_times_line_valid_plus_0p1_format`

Slurm wrapper after hard-v2 SFT checkpoints are merged. The hard-v2 configs load `data.materialized.dataset_id=flaitenberger/LogicalReasoning-hard-v2`, so training uses the HF variant rather than local disk:

```bash
sbatch --array=0-14%2 scripts/slurm/sweeps/posttrain_hard_v2_reward_ablation.slurm
```

The script expects merged SFT checkpoints at:

```text
$WORK/synthetic-RLVL/tmp/merged_sft_hard_v2_seed{3407,3408,3409}
```

## Evaluation Notes

Pass@k now also logs:

- `correct_given_valid@k`
- `invalid_but_correct@k`
- `valid_but_wrong@k`

Online GRPO eval is patched to force a final validation call at `global_steps = grpo.train_steps`, so the 1500-step point should appear in training curves.
