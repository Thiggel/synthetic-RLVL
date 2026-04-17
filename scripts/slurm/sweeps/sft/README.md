SFT sweeps (3 seeds each, grouped in W&B):

- `batch_size.slurm`
  - batch size in `{4, 8, 16}` (per-device batch size)
  - train samples in `{40000, 80000, 160000}`
  - fixed `lr=3e-5`
  - one epoch max (`train.num_epochs=1`, `train.max_steps=-1`)
  - fixed `10000` optimizer steps per run via `train_samples = bsz * 10000`
  - validation every `1000` steps (10 validation runs)
  - array: `0-8` (`3 configs x 3 seeds`)

- `lr.slurm`
  - learning rate in `{1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 5e-4}`
  - fixed `per_device_batch_size=1`, `train_samples=10000`
  - one epoch max (`train.num_epochs=1`, `train.max_steps=-1`)
  - array: `0-20` (`7 configs x 3 seeds`)

- `nl_exact_lr1e4.slurm`
  - template: `task.template=nl_exact` (`<think>` with NL premises/proof/conclusion)
  - fixed `lr=1e-4`, `per_device_batch_size=1`, `train_samples=10000`
  - array: `0-2` (3 seeds)

- `formal_then_nl_lr1e4.slurm`
  - template: `task.template=formal_think` (`<formal>` then `<think>`, `<answer>` at end)
  - fixed `lr=1e-4`, `per_device_batch_size=1`, `train_samples=10000`
  - array: `0-2` (3 seeds)

- `nl_then_formal_lr1e4.slurm`
  - template: `task.template=think_formal` (`<think>` then `<formal>`, `<answer>` at end)
  - fixed `lr=1e-4`, `per_device_batch_size=1`, `train_samples=10000`
  - array: `0-2` (3 seeds)

- `logic_cot_lr1e4.slurm`
  - template: `task.template=logic` (formal proof + answer)
  - fixed `lr=1e-4`, `per_device_batch_size=1`, `train_samples=10000`
  - array: `0-2` (3 seeds)

- `formal_then_nl_lr_coarse.slurm`
  - template: `task.template=formal_think`
  - learning rate in `{5e-5, 1e-4, 5e-4}`
  - fixed `per_device_batch_size=1`, `train_samples=10000`
  - array: `0-8` (`3 lrs x 3 seeds`)

- `nl_then_formal_lr_coarse.slurm`
  - template: `task.template=think_formal`
  - learning rate in `{5e-5, 1e-4, 5e-4}`
  - fixed `per_device_batch_size=1`, `train_samples=10000`
  - array: `0-8` (`3 lrs x 3 seeds`)

Submit examples:

```bash
sbatch scripts/slurm/sweeps/sft/batch_size.slurm
sbatch scripts/slurm/sweeps/sft/lr.slurm
sbatch scripts/slurm/sweeps/sft/nl_exact_lr1e4.slurm
sbatch scripts/slurm/sweeps/sft/formal_then_nl_lr1e4.slurm
sbatch scripts/slurm/sweeps/sft/nl_then_formal_lr1e4.slurm
sbatch scripts/slurm/sweeps/sft/logic_cot_lr1e4.slurm
sbatch scripts/slurm/sweeps/sft/formal_then_nl_lr_coarse.slurm
sbatch scripts/slurm/sweeps/sft/nl_then_formal_lr_coarse.slurm
```
