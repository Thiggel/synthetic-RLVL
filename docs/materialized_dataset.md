# Materialized Synthetic Dataset

This repo now supports using prebuilt parquet subsets instead of on-the-fly sample generation.

## Subsets

Training:
- `train_up_to_5_1m`: 1,000,000 rows mixed over depths `1..5`.
- `train_up_to_10_1m`: 1,000,000 rows mixed over depths `1..10`.

Validation:
- `val_step_01_1k` ... `val_step_20_1k`: each contains 1,000 rows at fixed depth.

## Build Locally

```bash
source ./scripts/env.sh
python -m synthrlvl.datasets.materialize \
  --output-root "$WORK/synthetic-RLVL/datasets/materialized_logic"
```

## Optional Push To Hugging Face

```bash
source ./scripts/env.sh
python -m synthrlvl.datasets.materialize \
  --output-root "$WORK/synthetic-RLVL/datasets/materialized_logic" \
  --push-to-hub \
  --hf-repo-id "<org-or-user>/<dataset-name>"
```

## Train With Materialized Data

SFT:
```bash
python train_sft.py \
  data.source=materialized \
  data.materialized.local_root="$WORK/synthetic-RLVL/datasets/materialized_logic"
```

GRPO:
```bash
python posttrain_grpo_verl.py \
  data.source=materialized \
  data.materialized.local_root="$WORK/synthetic-RLVL/datasets/materialized_logic"
```

You can also use HF-hosted subsets with:
- `data.materialized.dataset_id=<repo_id>`
- `data.materialized.local_root=null`
