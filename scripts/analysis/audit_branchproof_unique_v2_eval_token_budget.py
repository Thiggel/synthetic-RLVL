#!/usr/bin/env python3
"""Audit corrected BranchProof gold targets against evaluation token limits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthrlvl.task import task_sample_from_materialized_row
from synthrlvl.types import PrefillMode, StepRange, TaskConfig, TemplateName


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", default="allenai/Olmo-3-1025-7B")
    parser.add_argument(
        "--steps",
        default="1,2,5,10,12,15,18,20,25,30,35,40,45,50",
    )
    parser.add_argument("--generation-cap", type=int, default=7168)
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _task_config(template: TemplateName) -> TaskConfig:
    return TaskConfig(
        template=template,
        prefill=PrefillMode.NONE,
        distractor_ratio=0.5,
        train_steps=StepRange(1, 25),
        val_steps=StepRange(1, 50),
        seed=3407,
        difficulty="hard_fsa_schema",
        branching_factor=4,
        require_unique_solution=True,
    )


def _quantile(ordered: list[int], q: float) -> int:
    return ordered[round(q * (len(ordered) - 1))]


def summarize(lengths: list[int], limit: int) -> dict[str, int | float]:
    if not lengths:
        raise AssertionError("Cannot summarize empty token lengths")
    ordered = sorted(lengths)
    count_over = sum(length > limit for length in ordered)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": _quantile(ordered, 0.50),
        "p95": _quantile(ordered, 0.95),
        "p99": _quantile(ordered, 0.99),
        "max": ordered[-1],
        "limit": limit,
        "headroom": limit - ordered[-1],
        "over_limit_count": count_over,
        "over_limit_rate": count_over / len(ordered),
    }


def main() -> None:
    args = parse_args()
    steps = [int(value) for value in args.steps.split(",") if value]
    if not steps:
        raise SystemExit("--steps must contain at least one depth")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    result: dict[str, Any] = {
        "accepted": False,
        "data_root": str(args.data_root),
        "tokenizer": args.tokenizer,
        "steps": steps,
        "generation_cap": args.generation_cap,
        "context_limit": args.context_limit,
        "templates": {},
    }
    failures: list[str] = []

    for template in (TemplateName.LOGIC, TemplateName.NL_EXACT):
        per_depth: dict[str, Any] = {}
        all_targets: list[int] = []
        all_totals: list[int] = []
        config = _task_config(template)
        for step in steps:
            path = args.data_root / f"val_step_{step:02d}_1k" / "train.parquet"
            rows = pq.read_table(path).to_pylist()
            samples = [task_sample_from_materialized_row(row, cfg=config) for row in rows]
            target_lengths = [
                len(token_ids)
                for token_ids in tokenizer(
                    [sample.target for sample in samples], add_special_tokens=False
                ).input_ids
            ]
            total_lengths = [
                len(token_ids)
                for token_ids in tokenizer(
                    [sample.prompt + sample.target for sample in samples],
                    add_special_tokens=False,
                ).input_ids
            ]
            target_summary = summarize(target_lengths, args.generation_cap)
            total_summary = summarize(total_lengths, args.context_limit)
            if target_summary["over_limit_count"]:
                failures.append(
                    f"{template.value} depth {step}: "
                    f"{target_summary['over_limit_count']} targets exceed {args.generation_cap}"
                )
            if total_summary["over_limit_count"]:
                failures.append(
                    f"{template.value} depth {step}: "
                    f"{total_summary['over_limit_count']} sequences exceed {args.context_limit}"
                )
            per_depth[str(step)] = {
                "target_tokens": target_summary,
                "prompt_plus_target_tokens": total_summary,
            }
            all_targets.extend(target_lengths)
            all_totals.extend(total_lengths)

        result["templates"][template.value] = {
            "records": len(all_targets),
            "target_tokens": summarize(all_targets, args.generation_cap),
            "prompt_plus_target_tokens": summarize(all_totals, args.context_limit),
            "by_depth": per_depth,
        }

    result["failures"] = failures
    result["accepted"] = not failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit("BranchProof evaluation token-budget audit failed")


if __name__ == "__main__":
    main()
