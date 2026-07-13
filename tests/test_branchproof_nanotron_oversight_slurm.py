from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "slurm"
    / "codex"
    / "branchproof_nanotron_oversight_2026-07-11.slurm"
)


def test_oversight_does_not_consume_experiment_gpu_quota() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "#SBATCH --partition=a100mig" in text
    assert "#SBATCH --gres" not in text


def test_oversight_schedules_successor_before_codex() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.index("  sbatch \\") < text.index('cs exec "')
