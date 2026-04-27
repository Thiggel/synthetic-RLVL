# Constrained Proof-Line Evaluation

This is an optional post-hoc eval mode. It does not affect training or the default pass@k eval unless explicitly enabled.

## What It Does

For each constrained completion, the model first generates its own `<formal>` prefix up to `<proof>`. Then the proof is generated line by line.

At each proof line, we sample a fixed number of candidate lines:

```text
N = constrained_candidates_per_line
```

The logic engine ranks candidates as:

```text
random/unparseable < syntactic < valid < valid-and-novel
```

The best candidate is accepted, appended to the proof, and the next line is generated the same way. After the proof is closed, the model generates the remaining suffix normally (`<conclusion>`, `</formal>`, `<answer>`).

## Important Notation

`N` is the number of candidate proof lines sampled per proof-line decision.

`k` in `valid_pass@k`, `correct_pass@k`, etc. is the usual pass@k over completed constrained generations for the same prompt.

So a constrained eval has two sampling axes:

```text
N = candidates per generated proof line
k = number of completed constrained solutions considered for pass@k
```

Example: `constrained_num_generations=8` and `constrained_candidates_per_line=8` means we create 8 full constrained completions per prompt; each proof line inside each completion chooses from 8 candidate lines.

## Metrics

Constrained metrics are logged separately under:

```text
synthetic_constrained_sampled/...
```

Main metrics:

```text
syntactic_pass@k
format_pass@k
correct_pass@k
valid_pass@k
joint_pass@k
valid_given_correct@k
```

`syntactic_pass@k` means the proof parses syntactically under the logic engine, not merely that XML tags exist.

## CLI Usage

```bash
source ./scripts/env.sh
$HPCVAULT/.venv_rlvl_posttrain/bin/python scripts/evaluate_checkpoint_passk.py \
  --checkpoint /path/to/merged_checkpoint \
  --config conf/posttrain_grpo.yaml \
  --profile grpo \
  --backend vllm \
  --disable-external \
  --constrained-enabled \
  --constrained-num-generations 8 \
  --constrained-candidates-per-line 8 \
  --constrained-k-values 1,2,4,8 \
  --output /tmp/passk_constrained.json \
  --samples-output /tmp/passk_constrained_samples.jsonl
```

## Slurm Usage

The existing post-hoc eval script supports constrained eval through env vars:

```bash
CONSTRAINED_PASSK_ENABLED=1 \
CONSTRAINED_PASSK_NUM_GENERATIONS=8 \
CONSTRAINED_PASSK_CANDIDATES_PER_LINE=8 \
CONSTRAINED_PASSK_K_VALUES=1,2,4,8 \
sbatch scripts/slurm/jobs/posthoc_merge_eval_passk_2026-04-24.slurm
```

## Scientific Use

To test whether validity rewards help, compare reward-design checkpoints with identical constrained settings:

- same prompts
- same `constrained_num_generations`
- same `constrained_candidates_per_line`
- same temperature and max proof lines
- same final checkpoint step

Compare validity-reward runs against no-validity baselines on constrained and unconstrained pass@k, especially OOD depths `11-20` and hard tail `15-20`.
