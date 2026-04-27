# Slurm Layout

The repository uses direct Slurm scripts instead of generated command text files.

## Structure

- `scripts/slurm/jobs/`: single-run jobs (for example dataset build/push).
- `scripts/slurm/sweeps/`: Slurm-array jobs for sweep dimensions.
- `scripts/slurm/sweeps/sft/`: SFT hyperparameter sweeps.

## Submit Examples

```bash
HF_REPO_ID=<org-or-user>/<dataset-name> sbatch scripts/slurm/jobs/build_materialized_dataset_push.slurm

sbatch scripts/slurm/sweeps/sft/batch_size.slurm
sbatch scripts/slurm/sweeps/posttrain_reward_ablation.slurm
sbatch scripts/slurm/sweeps/posttrain_merged_num_prompts_ablation.slurm
sbatch scripts/slurm/sweeps/posttrain_merged_num_rollouts_ablation.slurm
```

## Posttrain Note

Current GRPO sweeps expect pre-merged per-seed checkpoints at:

- `$WORK/synthetic-RLVL/tmp/merged_sft_lr1e-4_seed3407`
- `$WORK/synthetic-RLVL/tmp/merged_sft_lr1e-4_seed3408`
- `$WORK/synthetic-RLVL/tmp/merged_sft_lr1e-4_seed3409`

See `docs/posttrain_status_2026-04-18.md` for current job IDs and exact array mapping.

## Runtime Convention

Every script:

1. `cd` to `SLURM_SUBMIT_DIR`.
2. `source ./scripts/env.sh`.
3. Activates `\$HPCVAULT/.venv_rlvl_posttrain` if available, otherwise `\$WORK/.venv`.

This keeps submission transparent and avoids wrapper bash scripts or generated command files.
