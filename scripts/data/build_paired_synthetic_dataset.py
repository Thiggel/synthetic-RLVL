#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthrlvl.datasets import (
    PAIRED_DATASET_KINDS,
    PairedGeneratorConfig,
    PairedSyntheticGenerator,
    validate_logic_example,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build materialized paired natural/logic synthetic datasets.")
    parser.add_argument("--kind", choices=list(PAIRED_DATASET_KINDS), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-subset", default=None)
    parser.add_argument("--train-rows", type=int, default=10_000)
    parser.add_argument("--train-max-depth", type=int, default=10)
    parser.add_argument("--val-rows-per-depth", type=int, default=128)
    parser.add_argument("--val-max-depth", type=int, default=50)
    parser.add_argument("--val-subset-template", default="val_step_{step:02d}_1k")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--branching-factor", type=int, default=4)
    parser.add_argument("--distractor-ratio", type=float, default=0.25)
    parser.add_argument("--max-operand", type=int, default=20)
    parser.add_argument("--max-multiplier", type=int, default=6)
    parser.add_argument("--operation-tokens", default="+,-,*")
    parser.add_argument("--modulus", type=int, default=None)
    parser.add_argument("--official-igsm-repo-path", default=None)
    parser.add_argument("--official-igsm-max-edge", type=int, default=None)
    parser.add_argument("--official-igsm-perm-level", type=int, default=5)
    parser.add_argument("--official-igsm-detail-level", type=int, default=0)
    parser.add_argument("--official-igsm-p-format", default="pq")
    parser.add_argument("--candidate-count", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument(
        "--validate-examples",
        type=int,
        default=100,
        help="Validate this many examples per subset with LogicEngine. Use -1 for all, 0 to disable.",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hf-repo-id", default=None)
    parser.add_argument("--hf-private", action="store_true")
    return parser.parse_args()


def _subset_path(output_root: Path, subset: str) -> Path:
    return output_root / subset / "train.parquet"


def _records(
    *,
    kind: str,
    min_depth: int,
    max_depth: int,
    rows: int,
    seed: int,
    start_index: int,
    args: argparse.Namespace,
) -> Iterator[dict[str, Any]]:
    depths = list(range(int(min_depth), int(max_depth) + 1))
    generators = {
        depth: PairedSyntheticGenerator(
            PairedGeneratorConfig(
                kind=kind,  # type: ignore[arg-type]
                depth=depth,
                seed=int(seed) + depth,
                branching_factor=int(args.branching_factor),
                distractor_ratio=float(args.distractor_ratio),
                max_operand=int(args.max_operand),
                max_multiplier=int(args.max_multiplier),
                operation_tokens=tuple(part.strip() for part in str(args.operation_tokens).split(",") if part.strip()),
                modulus=args.modulus,
                official_igsm_repo_path=args.official_igsm_repo_path,
                official_igsm_max_edge=args.official_igsm_max_edge,
                official_igsm_perm_level=int(args.official_igsm_perm_level),
                official_igsm_detail_level=int(args.official_igsm_detail_level),
                official_igsm_p_format=str(args.official_igsm_p_format),
                candidate_count=int(args.candidate_count),
            )
        )
        for depth in depths
    }
    counters = {depth: 0 for depth in depths}
    for idx in range(int(rows)):
        depth = depths[idx % len(depths)]
        local_index = start_index + counters[depth]
        counters[depth] += 1
        example = generators[depth].generate(local_index)
        row = example.to_dict()
        row["depth"] = int(depth)
        row["record_index"] = int(local_index)
        row["dataset_kind"] = str(kind)
        yield row


def _write_subset(
    *,
    output_root: Path,
    subset: str,
    rows: Iterator[dict[str, Any]],
    chunk_size: int,
    validate_examples: int,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_file = _subset_path(output_root, subset)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    chunk: list[dict[str, Any]] = []
    count = 0
    validation_failures: list[str] = []
    for row in rows:
        if validate_examples < 0 or count < validate_examples:
            from synthetic_dataset import LogicExample

            example = LogicExample(
                constants=list(row["constants"]),
                predicates=list(row["predicates"]),
                premises_fol=list(row["premises_fol"]),
                premises_nl=list(row["premises_nl"]),
                proof_fol=list(row["proof_fol"]),
                proof_nl=list(row["proof_nl"]),
                question_fol=str(row["question_fol"]),
                question_nl=str(row["question_nl"]),
                answer=str(row["answer"]),
                metadata=dict(row.get("metadata", {})),
            )
            validation = validate_logic_example(example)
            if not validation.ok:
                validation_failures.append(f"record={count} error={validation.error} lines={list(validation.line_errors)[:3]}")
                if len(validation_failures) >= 5:
                    raise RuntimeError(f"Logic validation failed for subset {subset}: {validation_failures}")
        chunk.append(row)
        count += 1
        if len(chunk) >= int(chunk_size):
            table = pa.Table.from_pylist(chunk)
            if writer is None:
                writer = pq.ParquetWriter(str(out_file), table.schema, compression="zstd")
            writer.write_table(table)
            chunk = []
    if chunk:
        table = pa.Table.from_pylist(chunk)
        if writer is None:
            writer = pq.ParquetWriter(str(out_file), table.schema, compression="zstd")
        writer.write_table(table)
    if writer is not None:
        writer.close()
    if validation_failures:
        raise RuntimeError(f"Logic validation failed for subset {subset}: {validation_failures}")
    return {"subset": subset, "rows": count, "path": str(out_file), "validation_failures": validation_failures}


def _push_to_hub(output_root: Path, subsets: list[str], *, repo_id: str, private: bool) -> None:
    from datasets import Dataset

    for subset in subsets:
        parquet_file = _subset_path(output_root, subset)
        if not parquet_file.exists():
            continue
        ds = Dataset.from_parquet(str(parquet_file))
        ds.push_to_hub(repo_id=repo_id, config_name=subset, split="train", private=private)


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    train_subset = args.train_subset or f"train_{args.kind}_up_to_{args.train_max_depth}_{args.train_rows}"
    summaries: list[dict[str, Any]] = []

    if int(args.train_rows) > 0:
        summaries.append(
            _write_subset(
                output_root=output_root,
                subset=train_subset,
                rows=_records(
                    kind=args.kind,
                    min_depth=1,
                    max_depth=int(args.train_max_depth),
                    rows=int(args.train_rows),
                    seed=int(args.seed),
                    start_index=0,
                    args=args,
                ),
                chunk_size=int(args.chunk_size),
                validate_examples=int(args.validate_examples),
            )
        )

    for step in range(1, int(args.val_max_depth) + 1):
        subset = str(args.val_subset_template).format(step=step, kind=args.kind, rows=int(args.val_rows_per_depth))
        summaries.append(
            _write_subset(
                output_root=output_root,
                subset=subset,
                rows=_records(
                    kind=args.kind,
                    min_depth=step,
                    max_depth=step,
                    rows=int(args.val_rows_per_depth),
                    seed=int(args.seed) + 1_000_000,
                    start_index=0,
                    args=args,
                ),
                chunk_size=int(args.chunk_size),
                validate_examples=int(args.validate_examples),
            )
        )

    manifest = {
        "kind": args.kind,
        "train_subset": train_subset,
        "train_rows": int(args.train_rows),
        "train_max_depth": int(args.train_max_depth),
        "val_rows_per_depth": int(args.val_rows_per_depth),
        "val_max_depth": int(args.val_max_depth),
        "seed": int(args.seed),
        "branching_factor": int(args.branching_factor),
        "distractor_ratio": float(args.distractor_ratio),
        "official_igsm_repo_path": args.official_igsm_repo_path,
        "official_igsm_max_edge": args.official_igsm_max_edge,
        "official_igsm_perm_level": int(args.official_igsm_perm_level),
        "official_igsm_detail_level": int(args.official_igsm_detail_level),
        "official_igsm_p_format": str(args.official_igsm_p_format),
        "candidate_count": int(args.candidate_count),
        "subsets": summaries,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)

    if args.push_to_hub:
        if not args.hf_repo_id:
            raise ValueError("--hf-repo-id is required with --push-to-hub")
        _push_to_hub(output_root, [row["subset"] for row in summaries], repo_id=str(args.hf_repo_id), private=bool(args.hf_private))


if __name__ == "__main__":
    main()
