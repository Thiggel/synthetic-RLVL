from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "slurm"
    / "jobs"
    / "nanotron_qwen25_push_hf_2026-06-24.slurm"
)


def test_converter_runs_under_single_rank_torchrun() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert (
        "torchrun --standalone --nproc_per_node=1 "
        "scripts/nanotron/convert_qwen2_nanotron_to_hf.py"
    ) in text
    assert "python scripts/nanotron/convert_qwen2_nanotron_to_hf.py" not in text


def test_verifier_uses_downstream_environment() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert (
        '"${HPCVAULT}/.venv_rlvl_posttrain/bin/python" '
        "scripts/nanotron/verify_qwen2_hf_checkpoint.py"
    ) in text
