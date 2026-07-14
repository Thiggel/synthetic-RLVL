from pathlib import Path

from synthrlvl.eval_loop import UnifiedEvaluator


def test_resolve_checkpoint_paths_accepts_separate_tokenizer(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-1250"
    tokenizer = tmp_path / "final"
    checkpoint.mkdir()
    tokenizer.mkdir()

    model_path, tokenizer_path, adapter_dir = UnifiedEvaluator()._resolve_checkpoint_paths(
        checkpoint,
        tokenizer,
    )

    assert model_path == str(checkpoint.resolve())
    assert tokenizer_path == str(tokenizer.resolve())
    assert adapter_dir is None
