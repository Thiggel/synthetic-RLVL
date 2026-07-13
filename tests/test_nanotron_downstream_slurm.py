from pathlib import Path


JOBS = Path(__file__).parents[1] / "scripts" / "slurm" / "jobs"


def test_remote_model_jobs_disable_unreliable_hf_transfer() -> None:
    for name in (
        "nanotron_qwen25_downstream_eval_2026-06-24.slurm",
        "nanotron_qwen25_instruction_sft_2026-06-24.slurm",
    ):
        text = (JOBS / name).read_text(encoding="utf-8")
        assert "export HF_HUB_ENABLE_HF_TRANSFER=0" in text


def test_downstream_aggregate_does_not_request_a_gpu() -> None:
    text = (JOBS / "aggregate_nanotron_downstream_pilot_2026-07-11.slurm").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --partition=a100mig" in text
    assert "#SBATCH --gres" not in text
