# synthetic-RLVL

Minimal synthetic logic stack for:
- deterministic dataset generation
- logic proof validation
- SFT with prompt masking
- RL post-training (VERL + GRPO) with format/correctness/validity rewards
- Slurm jobs and array sweeps

## Quick Start

```bash
python -m pytest -q
```

## SFT

Default protocol (train on 1-5 steps, validate on 1-10, eval curves on 1-20):

```bash
python train_sft.py run_name=sft_logic_seed3407 seed=3407
```

Key overrides:
- `task.template={logic,natural,logic_natural,natural_logic,nl_exact,formal_think,think_formal}`
- `task.prefill={none,gold,line_reward}`
- `task.distractor_ratio=0.5`

## Post-Train RL

Default protocol (train on 1-10 steps, validate on 1-20):

```bash
export SFT_CHECKPOINT=/path/to/your/sft/checkpoint
python posttrain_grpo_verl.py run_name=rl_logic_seed3407 seed=3407
```

`posttrain_grpo_verl.py` rejects base-model-only runs and requires an SFT checkpoint.

Reward ablations (`reward.schema`):
- `correct_plus_0p1_format`
- `indicator_correct_and_format`
- `correct_plus_valid_plus_0p1_format`
- `correct_plus_0p75_valid_plus_0p1_format`
- `correct_plus_0p5_valid_plus_0p1_format`
- `correct_plus_0p25_valid_plus_0p1_format`
- `indicator_all`

## Slurm Jobs

Materialized dataset build + push:

```bash
HF_REPO_ID=<org-or-user>/<dataset-name> sbatch scripts/slurm/jobs/build_materialized_dataset_push.slurm
```

Sweeps use native Slurm arrays:

```bash
sbatch scripts/slurm/sweeps/sft/batch_size.slurm
sbatch scripts/slurm/sweeps/posttrain_reward_ablation.slurm
```

All scripts source `scripts/env.sh` and default to `\$HPCVAULT/.venv_rlvl_posttrain`, falling back to `\$WORK/.venv`.

## Batch-Size Fit Probe

Measured one-GPU fit for OLMo-3-7B style SFT setup:

```bash
python scripts/probe_sft_batch_fit.py --model /home/atuin/c107fa/c107fa12/RLVL/finetune/olmo3-7b-logic-lora-full-1ep-lr5e-4-seed101-v4-merged --seq-len 2048
```

Result: only per-device batch size `1` fits on 1x A100 80GB; larger global batch sizes require gradient accumulation.

## Plot Mean/Std Curves (W&B)

```bash
python scripts/analysis/plot_wandb_group.py \
  --entity <wandb_entity> \
  --project synthetic-rlvl \
  --group <group_name> \
  --step-min 1 --step-max 20 \
  --out-dir plots
```

## Notes

- Distractor ratio is configurable and defaults to `0.5` everywhere.
- Dataset proofs are engine-valid by construction and covered by tests.
- Runtime/cache/proxy bootstrap details: `docs/runtime_env.md`
- Slurm organization and submission examples: `docs/slurm_layout.md`
- Materialized parquet dataset workflow: `docs/materialized_dataset.md`
