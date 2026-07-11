from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[1] / "scripts" / "nanotron" / "verify_training_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("verify_nanotron_training_checkpoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _checkpoint(tmp_path: Path) -> Path:
    root = tmp_path / "4096"
    for name in ("config.yaml", "model_config.json"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("ok\n", encoding="utf-8")
    metadata = {
        "tp": 2,
        "dp": 2,
        "metas": {
            "last_train_step": 4,
            "consumed_train_samples": 8,
            "consumed_tokens_total": 32,
            "data_stages": [
                {
                    "sequence_length": 4,
                    "consumed_train_samples": 8,
                    "consumed_tokens_per_dataset_folder": {"normal": 24, "proof": 8},
                }
            ],
        },
    }
    (root / "checkpoint_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for directory, count in (("model", 2), ("lr_scheduler", 2), ("random", 4)):
        for index in range(count):
            path = root / directory / f"{index}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
    for index in range(2):
        path = root / "optimizer" / f"optimizer_{index}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"xxxx")
    return root


def _audit(root: Path) -> dict:
    return MODULE.audit_checkpoint(
        root,
        expected_step=4,
        sequence_length=4,
        global_batch_samples=2,
        tp=2,
        dp=2,
        model_files=2,
        optimizer_shards=2,
        lr_scheduler_shards=2,
        rng_shards=4,
        min_optimizer_shard_bytes=4,
    )


def test_accepts_complete_checkpoint(tmp_path: Path) -> None:
    result = _audit(_checkpoint(tmp_path))
    assert result["status"] == "accepted"
    assert result["expected_tokens"] == 32
    assert result["file_counts"] == {
        "model": 2,
        "optimizer": 2,
        "lr_scheduler": 2,
        "random": 4,
    }


def test_rejects_bad_offsets_and_empty_file(tmp_path: Path) -> None:
    root = _checkpoint(tmp_path)
    metadata_path = root / "checkpoint_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["metas"]["last_train_step"] = 3
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    (root / "model" / "0.pt").write_bytes(b"")

    result = _audit(root)
    assert result["status"] == "rejected"
    assert any("last_train_step" in error for error in result["errors"])
    assert any("zero-byte" in error for error in result["errors"])


def test_rejects_unequal_optimizer_shards_and_bad_mixture_total(tmp_path: Path) -> None:
    root = _checkpoint(tmp_path)
    (root / "optimizer" / "optimizer_1.pt").write_bytes(b"xxxxx")
    metadata_path = root / "checkpoint_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["metas"]["data_stages"][0]["consumed_tokens_per_dataset_folder"]["proof"] = 7
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    result = _audit(root)
    assert result["status"] == "rejected"
    assert any("optimizer shard sizes differ" in error for error in result["errors"])
    assert any("data-stage tokens" in error for error in result["errors"])
