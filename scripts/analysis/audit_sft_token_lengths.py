#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthrlvl.datasets import MaterializedSyntheticDataset
from synthrlvl.task import task_sample_from_materialized_row
from synthrlvl.types import PrefillMode, StepRange, TaskConfig, TemplateName


def _pct(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return float(ordered[idx])


def _mean(values: list[int]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _task_cfg(template: str, train_max: int, seed: int) -> TaskConfig:
    return TaskConfig(
        template=TemplateName(template),
        prefill=PrefillMode.NONE,
        distractor_ratio=0.5,
        train_steps=StepRange(1, int(train_max)),
        val_steps=StepRange(1, 50),
        seed=int(seed),
        difficulty="hard_fsa_schema",
        branching_factor=4,
        require_unique_solution=True,
    )


def _summarize(rows: list[dict], *, tokenizer, template: str, train_max: int, max_length: int, seed: int) -> dict[str, object]:
    cfg = _task_cfg(template, train_max, seed)
    prompt_lengths: list[int] = []
    trace_lengths: list[int] = []
    target_lengths: list[int] = []
    total_lengths: list[int] = []
    depths: list[int] = []
    truncated = 0
    for row in rows:
        sample = task_sample_from_materialized_row(row, cfg=cfg)
        prompt_ids = tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(sample.target, add_special_tokens=False)["input_ids"]
        trace_text = sample.target.split("<answer>", 1)[0]
        trace_ids = tokenizer(trace_text, add_special_tokens=False)["input_ids"]
        total = len(prompt_ids) + len(target_ids) + 1
        prompt_lengths.append(len(prompt_ids))
        trace_lengths.append(len(trace_ids))
        target_lengths.append(len(target_ids))
        total_lengths.append(total)
        depths.append(int(row["depth"]))
        truncated += int(total > max_length)
    return {
        "template": template,
        "train_max": int(train_max),
        "n": len(rows),
        "mean_depth": _mean(depths),
        "prompt_mean": _mean(prompt_lengths),
        "trace_mean": _mean(trace_lengths),
        "trace_p50": _pct(trace_lengths, 0.50),
        "trace_p95": _pct(trace_lengths, 0.95),
        "target_mean": _mean(target_lengths),
        "target_p50": _pct(target_lengths, 0.50),
        "target_p95": _pct(target_lengths, 0.95),
        "total_mean": _mean(total_lengths),
        "total_p50": _pct(total_lengths, 0.50),
        "total_p95": _pct(total_lengths, 0.95),
        "truncation_rate_at_max_length": truncated / max(1, len(rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SFT prompt/target token lengths for materialized HFSA rows.")
    parser.add_argument("--dataset-id", default="flaitenberger/LogicalReasoning-hard-fsa-schema-fixedtarget-depth50")
    parser.add_argument("--local-root", default=None)
    parser.add_argument("--tokenizer", default="allenai/Olmo-3-1025-7B")
    parser.add_argument("--train-maxes", nargs="+", type=int, default=[5, 10, 15, 20, 25])
    parser.add_argument("--templates", nargs="+", default=["logic", "nl_exact"])
    parser.add_argument("--sample-limit", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    mat = MaterializedSyntheticDataset()
    summaries: list[dict[str, object]] = []
    for train_max in args.train_maxes:
        subset = f"train_fixedtarget_up_to_{train_max}_50k"
        rows = mat.load_rows(
            subset=subset,
            dataset_id=args.dataset_id,
            local_root=args.local_root,
            split="train",
            limit=int(args.sample_limit),
        )
        for template in args.templates:
            summaries.append(
                _summarize(
                    rows,
                    tokenizer=tokenizer,
                    template=template,
                    train_max=int(train_max),
                    max_length=int(args.max_length),
                    seed=int(args.seed),
                )
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"wrote {output}")
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
