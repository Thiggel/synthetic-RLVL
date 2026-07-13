from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[1] / "scripts" / "nanotron" / "verify_qwen2_hf_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("verify_qwen2_hf_checkpoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "tokenizer_config.json").write_text("{}")
    (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"a")
    (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"b")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "lm_head.weight": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    return checkpoint


def test_accepts_complete_shard_manifest(tmp_path: Path) -> None:
    report = MODULE.audit_local_files(_checkpoint(tmp_path))
    assert report["accepted"]
    assert report["weight_shards"] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]


def test_rejects_missing_weight_shard(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    (checkpoint / "model-00002-of-00002.safetensors").unlink()
    report = MODULE.audit_local_files(checkpoint)
    assert not report["accepted"]
    assert "missing or empty weight shard: model-00002-of-00002.safetensors" in report["errors"]


def test_rejects_transformers_5_only_rope_config(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "rope_parameters": {"rope_theta": 1_000_000.0, "rope_type": "default"},
            }
        ),
        encoding="utf-8",
    )
    report = MODULE.audit_model_config(checkpoint)
    assert not report["accepted"]
    assert "rope_theta=None, expected 1000000.0" in report["errors"]


def test_accepts_dual_format_rope_config(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "rope_theta": 1_000_000.0,
                "rope_parameters": {"rope_theta": 1_000_000.0, "rope_type": "default"},
            }
        ),
        encoding="utf-8",
    )
    report = MODULE.audit_model_config(checkpoint)
    assert report["accepted"]
    assert report["loaded_rope_theta"] == 1_000_000.0
