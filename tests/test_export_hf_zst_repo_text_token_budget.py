import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "data" / "export_hf_zst_repo_text_token_budget.py"
SPEC = importlib.util.spec_from_file_location("export_hf_zst_repo_text_token_budget", SCRIPT)
assert SPEC and SPEC.loader
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def test_download_shard_retries_transient_failure(monkeypatch, tmp_path: Path):
    calls = []
    target = tmp_path / "shard.jsonl.zst"

    def fake_download(dataset, repo_file, repo_type):
        calls.append((dataset, repo_file, repo_type))
        if len(calls) < 3:
            raise RuntimeError("transient")
        return str(target)

    delays = []
    monkeypatch.setattr(exporter, "hf_hub_download", fake_download)
    monkeypatch.setattr(exporter, "sleep", delays.append)

    assert exporter._download_shard("dataset", "data/shard.jsonl.zst") == target
    assert len(calls) == 3
    assert delays == [2, 4]


def test_download_shard_raises_after_last_attempt(monkeypatch):
    calls = []

    def fake_download(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("persistent")

    monkeypatch.setattr(exporter, "hf_hub_download", fake_download)
    monkeypatch.setattr(exporter, "sleep", lambda _: None)

    try:
        exporter._download_shard("dataset", "data/shard.jsonl.zst", attempts=3)
    except RuntimeError as error:
        assert str(error) == "persistent"
    else:
        raise AssertionError("persistent download failure was not raised")
    assert len(calls) == 3
