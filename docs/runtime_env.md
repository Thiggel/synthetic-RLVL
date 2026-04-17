# Runtime Environment Bootstrap

Use `scripts/env.sh` to standardize module loading and environment variables across local shells and Slurm jobs.

What it does:
- Loads `cuda/12.8.1` (best-effort) via `module load`.
- Loads local `.env` if present.
- Sets proxy defaults (`http_proxy`, `https_proxy`, `no_proxy`).
- Sets cache paths for HF, datasets, transformers, vLLM, torch, triton, uv, wandb.
- Sets `HF_HUB_DISABLE_XET=1` by default.
- Creates all required cache/work/tmp directories.
- Clears `ROCR_VISIBLE_DEVICES` if `CUDA_VISIBLE_DEVICES` is set.

Usage:
```bash
source ./scripts/env.sh
```

Slurm integration:
- All scripts under `scripts/slurm/jobs/*.slurm` and `scripts/slurm/sweeps/**/*.slurm` source `scripts/env.sh`.
- Jobs request `--partition=a100 --constraint=a100_80 --gres=gpu:a100:1`.

Venv note:
- `\$HPCVAULT/.venv_rlvl_posttrain` can run both GRPO and SFT in this repo.
- If you want one environment, point both job submissions to that venv path.
