from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_tiny_llama_pretrain.py"
SPEC = importlib.util.spec_from_file_location("train_tiny_llama_pretrain", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_resolve_resume_checkpoint_uses_latest_complete_checkpoint(tmp_path: Path) -> None:
    for step in (20_000, 40_000):
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    (tmp_path / "checkpoint-60000").mkdir()

    assert MODULE._resolve_resume_checkpoint(tmp_path, "auto") == str(tmp_path / "checkpoint-40000")
    assert MODULE._resolve_resume_checkpoint(tmp_path, None) is None
    assert MODULE._resolve_resume_checkpoint(tmp_path, "/tmp/manual") == "/tmp/manual"


def test_unique_train_budget_accepts_exactly_one_pass() -> None:
    assert MODULE._validate_unique_train_budget(
        dataset_rows=100_000,
        max_steps=6_250,
        per_device_batch_size=2,
        grad_accum=8,
    ) == 100_000


def test_unique_train_budget_rejects_row_reuse() -> None:
    try:
        MODULE._validate_unique_train_budget(
            dataset_rows=50_000,
            max_steps=100_000,
            per_device_batch_size=2,
            grad_accum=8,
        )
    except ValueError as exc:
        assert "1600000 rows" in str(exc)
        assert "50000" in str(exc)
    else:
        raise AssertionError("expected repeated-data training budget to fail")
