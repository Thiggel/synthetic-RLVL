from __future__ import annotations

import subprocess
from pathlib import Path
import pytest


def _env():
    import os

    env = dict(os.environ)
    env["WANDB_MODE"] = "disabled"
    return env


pytestmark = pytest.mark.skipif(
    __import__("os").environ.get("RUN_E2E_SMOKE", "0") != "1",
    reason="Set RUN_E2E_SMOKE=1 to run slow end-to-end smoke tests.",
)


def test_slurm_script_layout(tmp_path: Path):
    assert Path("scripts/slurm/jobs/targeted_sft_external_eval.slurm").exists()
    assert Path("scripts/slurm/jobs/targeted_grpo_vllm_startup.slurm").exists()
    assert Path("scripts/slurm/sweeps/sft_lr.slurm").exists()
    assert Path("scripts/slurm/sweeps/posttrain_reward_ablation.slurm").exists()


def test_sft_smoke_tiny_model(tmp_path: Path):
    out_dir = tmp_path / "sft"
    cmd = [
        "python",
        "train_sft.py",
        "model.name=sshleifer/tiny-gpt2",
        "model.bf16=false",
        "model.lora.enabled=false",
        "data.train_samples=8",
        "data.eval_samples=4",
        "data.max_length=256",
        "train.max_steps=1",
        "train.eval_steps=1",
        "train.save_steps=1",
        "train.logging_steps=1",
        "validation.eval_steps=1",
        "validation.step_min=1",
        "validation.step_max=1",
        "validation.samples_per_step=2",
        "validation.max_new_tokens=32",
        "task.train_min_step=1",
        "task.train_max_step=2",
        "task.val_min_step=1",
        "task.val_max_step=2",
        "logging.report_to=[]",
        f"output_dir={out_dir}",
        "run_name=test_sft_smoke",
    ]
    subprocess.run(cmd, check=True, env=_env())
    assert (out_dir / "final").exists()


def test_posttrain_smoke_tiny_model(tmp_path: Path):
    out_dir = tmp_path / "rl"
    cmd = [
        "python",
        "posttrain_grpo_verl.py",
        "model.path=sshleifer/tiny-gpt2",
        "grpo.train_steps=1",
        "grpo.num_prompts=1",
        "grpo.num_rollouts=1",
        "grpo.max_prompt_length=128",
        "grpo.max_response_length=64",
        "validation.eval_every=1",
        "validation.save_every=1",
        "eval.enabled=false",
        "task.train_min_step=1",
        "task.train_max_step=2",
        "task.val_min_step=1",
        "task.val_max_step=2",
        "data.train_samples=8",
        "data.val_samples=4",
        "system.ray_cpus=2",
        "logging.project=synthetic-rlvl-test",
        "logging.entity=null",
        f"output_dir={out_dir}",
        "run_name=test_rl_smoke",
    ]
    subprocess.run(cmd, check=True, env=_env())
    assert (out_dir / "final").exists()
