#!/usr/bin/env python3
"""Measure non-oracle verifier selection on retained pass@k generations."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METRICS = (
    "random_correct",
    "random_valid",
    "random_joint",
    "first_valid_correct",
    "first_valid_valid",
    "first_valid_joint",
    "max_line_correct",
    "max_line_valid",
    "max_line_joint",
    "valid_coverage",
    "oracle_correct",
    "oracle_joint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-k", type=int, default=16)
    parser.add_argument("--train-max", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _modality(path: Path) -> str:
    name = path.name
    if "_nl_exact_" in name:
        return "nl_exact"
    if "_logic_" in name:
        return "logic"
    raise ValueError(f"cannot infer modality from {path}")


def _seed(path: Path) -> int:
    match = re.search(r"_seed(\d+)", path.name)
    if match is None:
        raise ValueError(f"cannot infer seed from {path}")
    return int(match.group(1))


def _validity(row: dict[str, Any], modality: str) -> tuple[bool, float]:
    if modality == "logic":
        return (
            bool(row.get("citation_free_valid")),
            float(row.get("citation_free_line_valid_fraction", 0.0)),
        )
    return (
        bool(row.get("nl_logic_citation_free_valid")),
        float(row.get("nl_logic_line_valid_fraction", 0.0)),
    )


def _bands(step: int, train_max: int) -> Iterable[str]:
    yield "all"
    yield "id" if step <= train_max else "ood"
    if step == 50:
        yield "depth50"


def analyze_file(path: Path, expected_k: int, train_max: int) -> dict[str, Any]:
    modality = _modality(path)
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") != "synthetic_sampled":
                continue
            step = row.get("step")
            prompt = row.get("prompt")
            if not isinstance(step, int) or not isinstance(prompt, str):
                raise ValueError(f"invalid sampled row at {path}:{line_number}")
            groups[(step, prompt)].append(row)

    if not groups:
        raise ValueError(f"no synthetic_sampled rows in {path}")

    band_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (step, _prompt), candidates in groups.items():
        if len(candidates) != expected_k:
            raise ValueError(
                f"{path}: step {step} prompt has {len(candidates)} samples, "
                f"expected {expected_k}"
            )
        scored = []
        for candidate in candidates:
            valid, line_fraction = _validity(candidate, modality)
            scored.append(
                {
                    "correct": bool(candidate.get("correct")),
                    "valid": valid,
                    "line_fraction": line_fraction,
                }
            )
        random_choice = scored[0]
        valid_choices = [candidate for candidate in scored if candidate["valid"]]
        first_valid = valid_choices[0] if valid_choices else random_choice
        max_line = max(scored, key=lambda candidate: candidate["line_fraction"])
        metrics = {
            "random_correct": float(random_choice["correct"]),
            "random_valid": float(random_choice["valid"]),
            "random_joint": float(random_choice["correct"] and random_choice["valid"]),
            "first_valid_correct": float(first_valid["correct"]),
            "first_valid_valid": float(first_valid["valid"]),
            "first_valid_joint": float(first_valid["correct"] and first_valid["valid"]),
            "max_line_correct": float(max_line["correct"]),
            "max_line_valid": float(max_line["valid"]),
            "max_line_joint": float(max_line["correct"] and max_line["valid"]),
            "valid_coverage": float(bool(valid_choices)),
            "oracle_correct": float(any(candidate["correct"] for candidate in scored)),
            "oracle_joint": float(
                any(candidate["correct"] and candidate["valid"] for candidate in scored)
            ),
        }
        for band in _bands(step, train_max):
            band_rows[band].append(metrics)

    metrics_by_band = {
        band: {
            metric: sum(row[metric] for row in rows) / len(rows)
            for metric in METRICS
        }
        for band, rows in sorted(band_rows.items())
    }
    return {
        "path": str(path),
        "modality": modality,
        "seed": _seed(path),
        "prompt_groups": len(groups),
        "sampled_rows": sum(len(rows) for rows in groups.values()),
        "metrics_by_band": metrics_by_band,
    }


def summarize(files: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for file_result in files:
        for band, metrics in file_result["metrics_by_band"].items():
            grouped[(file_result["modality"], band)].append(metrics)

    summary: dict[str, dict[str, Any]] = defaultdict(dict)
    for (modality, band), rows in sorted(grouped.items()):
        summary[modality][band] = {
            "n_seeds": len(rows),
            **{
                metric: {
                    "mean": statistics.fmean(row[metric] for row in rows),
                    "std": statistics.pstdev(row[metric] for row in rows),
                }
                for metric in METRICS
            },
        }
    return dict(summary)


def main() -> None:
    args = parse_args()
    if args.expected_k <= 0:
        raise SystemExit("--expected-k must be positive")
    files = [
        analyze_file(path, expected_k=args.expected_k, train_max=args.train_max)
        for path in args.samples
    ]
    payload = {
        "schema_version": 1,
        "selection_uses_gold_answer": False,
        "expected_k": args.expected_k,
        "train_max": args.train_max,
        "files": files,
        "summary": summarize(files),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
