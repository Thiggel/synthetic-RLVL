#!/usr/bin/env python3
"""Verify that a Nanotron training checkpoint is complete and resumable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--global-batch-samples", type=int, default=128)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--dp", type=int, default=2)
    parser.add_argument("--model-files", type=int, default=625)
    parser.add_argument("--optimizer-shards", type=int, default=4)
    parser.add_argument("--lr-scheduler-shards", type=int, default=4)
    parser.add_argument("--rng-shards", type=int, default=8)
    parser.add_argument("--min-optimizer-shard-bytes", type=int, default=20_000_000_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def audit_checkpoint(
    checkpoint: Path,
    *,
    expected_step: int,
    sequence_length: int,
    global_batch_samples: int,
    tp: int,
    dp: int,
    model_files: int,
    optimizer_shards: int,
    lr_scheduler_shards: int,
    rng_shards: int,
    min_optimizer_shard_bytes: int,
) -> dict[str, Any]:
    errors: list[str] = []
    required_files = ("checkpoint_metadata.json", "config.yaml", "model_config.json")
    for name in required_files:
        path = checkpoint / name
        _require(path.is_file() and path.stat().st_size > 0, f"missing or empty {name}", errors)

    all_files = [path for path in checkpoint.rglob("*") if path.is_file()]
    empty_files = [str(path.relative_to(checkpoint)) for path in all_files if path.stat().st_size == 0]
    _require(not empty_files, f"zero-byte files: {empty_files}", errors)

    groups = {
        "model": [path for path in (checkpoint / "model").rglob("*") if path.is_file()],
        "optimizer": sorted((checkpoint / "optimizer").glob("optimizer_*.pt")),
        "lr_scheduler": [
            path for path in (checkpoint / "lr_scheduler").rglob("*") if path.is_file()
        ],
        "random": [path for path in (checkpoint / "random").rglob("*") if path.is_file()],
    }
    expected_counts = {
        "model": model_files,
        "optimizer": optimizer_shards,
        "lr_scheduler": lr_scheduler_shards,
        "random": rng_shards,
    }
    for name, expected_count in expected_counts.items():
        _require(
            len(groups[name]) == expected_count,
            f"{name} file count={len(groups[name])}, expected {expected_count}",
            errors,
        )

    optimizer_sizes = [path.stat().st_size for path in groups["optimizer"]]
    _require(
        not optimizer_sizes or min(optimizer_sizes) >= min_optimizer_shard_bytes,
        f"optimizer shard below {min_optimizer_shard_bytes} bytes: {optimizer_sizes}",
        errors,
    )
    _require(
        len(set(optimizer_sizes)) <= 1,
        f"optimizer shard sizes differ: {optimizer_sizes}",
        errors,
    )

    metadata: dict[str, Any] = {}
    metadata_path = checkpoint / "checkpoint_metadata.json"
    if metadata_path.is_file() and metadata_path.stat().st_size:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"invalid checkpoint metadata: {exc}")

    expected_samples = expected_step * global_batch_samples
    expected_tokens = expected_samples * sequence_length
    _require(metadata.get("tp") == tp, f"tp={metadata.get('tp')!r}, expected {tp}", errors)
    _require(metadata.get("dp") == dp, f"dp={metadata.get('dp')!r}, expected {dp}", errors)
    metas = metadata.get("metas") if isinstance(metadata.get("metas"), dict) else {}
    _require(
        metas.get("last_train_step") == expected_step,
        f"last_train_step={metas.get('last_train_step')!r}, expected {expected_step}",
        errors,
    )
    _require(
        metas.get("consumed_train_samples") == expected_samples,
        f"consumed_train_samples={metas.get('consumed_train_samples')!r}, expected {expected_samples}",
        errors,
    )
    _require(
        metas.get("consumed_tokens_total") == expected_tokens,
        f"consumed_tokens_total={metas.get('consumed_tokens_total')!r}, expected {expected_tokens}",
        errors,
    )

    data_stages = metas.get("data_stages") if isinstance(metas.get("data_stages"), list) else []
    _require(bool(data_stages), "metadata has no data stages", errors)
    stage_samples = 0
    stage_tokens = 0
    for index, stage in enumerate(data_stages):
        if not isinstance(stage, dict):
            errors.append(f"data stage {index} is not an object")
            continue
        _require(
            stage.get("sequence_length") == sequence_length,
            f"data stage {index} sequence_length={stage.get('sequence_length')!r}, expected {sequence_length}",
            errors,
        )
        samples = stage.get("consumed_train_samples")
        if isinstance(samples, int) and not isinstance(samples, bool):
            stage_samples += samples
        else:
            errors.append(f"data stage {index} has invalid consumed_train_samples={samples!r}")
        per_dataset = stage.get("consumed_tokens_per_dataset_folder")
        if isinstance(per_dataset, dict) and all(
            isinstance(value, int) and not isinstance(value, bool) for value in per_dataset.values()
        ):
            stage_tokens += sum(per_dataset.values())
        else:
            errors.append(f"data stage {index} has invalid per-dataset token accounting")
    _require(stage_samples == expected_samples, f"data-stage samples={stage_samples}, expected {expected_samples}", errors)
    _require(stage_tokens == expected_tokens, f"data-stage tokens={stage_tokens}, expected {expected_tokens}", errors)

    return {
        "status": "accepted" if not errors else "rejected",
        "checkpoint": str(checkpoint),
        "expected_step": expected_step,
        "expected_samples": expected_samples,
        "expected_tokens": expected_tokens,
        "file_count": len(all_files),
        "file_counts": {name: len(paths) for name, paths in groups.items()},
        "optimizer_sizes": optimizer_sizes,
        "zero_byte_files": empty_files,
        "metadata": metadata,
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    result = audit_checkpoint(
        args.checkpoint,
        expected_step=args.expected_step,
        sequence_length=args.sequence_length,
        global_batch_samples=args.global_batch_samples,
        tp=args.tp,
        dp=args.dp,
        model_files=args.model_files,
        optimizer_shards=args.optimizer_shards,
        lr_scheduler_shards=args.lr_scheduler_shards,
        rng_shards=args.rng_shards,
        min_optimizer_shard_bytes=args.min_optimizer_shard_bytes,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if result["status"] != "accepted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
