#!/usr/bin/env python3
"""Build a reviewable qualitative audit for the corrected BranchProof grid."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEPTHS = (1, 2, 5, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50)
TEMPLATES = ("logic", "nl_exact")
TRAIN_MAXES = (5, 10, 15, 20, 25)
SEEDS = (3407, 3408, 3409)
SAMPLE_RE = re.compile(
    r"sft_branchproof_unique_v2_(?P<template>logic|nl_exact)_"
    r"train1to(?P<train>\d+)_10k_seed(?P<seed>\d+)_samples\.jsonl$"
)
CHUNK_RE = re.compile(
    r"^\[syntheval\] sampled vLLM chunk (?P<index>\d+)/(?P<total>\d+) "
    r"done in [0-9.]+s \(\d+ output tokens, max=(?P<maximum>\d+)\)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--eval-array-job-id", default="3834582")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--expected-sampled-rows", type=int, default=896)
    parser.add_argument("--expected-rows-per-depth", type=int, default=64)
    parser.add_argument("--generation-cap", type=int, default=7168)
    return parser.parse_args()


def _array_index(template: str, train_max: int, seed: int) -> int:
    return TEMPLATES.index(template) * 15 + TRAIN_MAXES.index(train_max) * 3 + SEEDS.index(seed)


def _read_rows(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            errors.append(f"{path.name}:{line_number}: blank line")
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}:{line_number}: invalid JSON: {exc}")
            continue
        if not isinstance(row, dict):
            errors.append(f"{path.name}:{line_number}: row is not an object")
            continue
        if row.get("source") != "synthetic_sampled":
            continue
        row = dict(row)
        row["_line_number"] = line_number
        rows.append(row)
    return rows


def _cap_chunks(path: Path, generation_cap: int, errors: list[str]) -> list[int]:
    if not path.is_file():
        errors.append(f"missing eval log: {path}")
        return []
    chunks: list[int] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = CHUNK_RE.match(line.strip())
        if match and int(match.group("maximum")) == generation_cap:
            chunks.append(int(match.group("index")))
    return chunks


def _is_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0.0


def _is_unit_metric(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= float(value) <= 1.0
    )


def _excerpt(row: dict[str, Any], *, path: Path, label: str) -> dict[str, Any]:
    generation = row.get("generation")
    text = generation if isinstance(generation, str) else ""
    return {
        "label": label,
        "file": str(path),
        "line_number": row.get("_line_number"),
        "step": row.get("step"),
        "sample_index": row.get("sample_index"),
        "gold_answer": row.get("gold_answer"),
        "correct": row.get("correct"),
        "format_ok": row.get("format_ok"),
        "citation_free_valid": row.get("citation_free_valid"),
        "nl_logic_citation_free_valid": row.get("nl_logic_citation_free_valid"),
        "generation_chars": len(text),
        "generation_head": text[:600],
        "generation_tail": text[-600:] if len(text) > 600 else "",
    }


def _select_slice(rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    validity_key = (
        "citation_free_valid" if "_logic_" in path.name else "nl_logic_citation_free_valid"
    )
    predicates = (
        ("correct_valid", lambda row: _is_positive(row.get("correct")) and _is_positive(row.get(validity_key))),
        ("correct_invalid", lambda row: _is_positive(row.get("correct")) and not _is_positive(row.get(validity_key))),
        ("incorrect", lambda row: not _is_positive(row.get("correct"))),
    )
    selected: list[dict[str, Any]] = []
    for label, predicate in predicates:
        match = next((row for row in rows if predicate(row)), None)
        if match is not None:
            selected.append(_excerpt(match, path=path, label=label))
    if not selected and rows:
        selected.append(_excerpt(rows[0], path=path, label="fallback"))
    return selected


def audit_grid(
    final_dir: Path,
    log_dir: Path,
    *,
    eval_array_job_id: str,
    expected_sampled_rows: int,
    expected_rows_per_depth: int,
    generation_cap: int,
) -> dict[str, Any]:
    errors: list[str] = []
    expected_grid = {
        (template, train_max, seed)
        for template in TEMPLATES
        for train_max in TRAIN_MAXES
        for seed in SEEDS
    }
    paths: dict[tuple[str, int, int], Path] = {}
    for path in sorted(final_dir.glob("*_samples.jsonl")):
        match = SAMPLE_RE.match(path.name)
        if match is None:
            continue
        key = (match.group("template"), int(match.group("train")), int(match.group("seed")))
        if key in paths:
            errors.append(f"duplicate sample artifact for {key}: {path}")
        paths[key] = path
    for key in sorted(expected_grid - paths.keys()):
        errors.append(f"missing sample artifact for {key}")
    for key in sorted(paths.keys() - expected_grid):
        errors.append(f"unexpected sample artifact for {key}: {paths[key]}")

    runs: list[dict[str, Any]] = []
    coverage: Counter[str] = Counter()
    for template, train_max, seed in sorted(expected_grid):
        path = paths.get((template, train_max, seed))
        if path is None:
            continue
        rows = _read_rows(path, errors)
        if len(rows) != expected_sampled_rows:
            errors.append(
                f"{path.name}: sampled rows={len(rows)}, expected {expected_sampled_rows}"
            )
        by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            step = row.get("step")
            if isinstance(step, int):
                by_depth[step].append(row)
        for depth in DEPTHS:
            if len(by_depth[depth]) != expected_rows_per_depth:
                errors.append(
                    f"{path.name}: depth {depth} rows={len(by_depth[depth])}, "
                    f"expected {expected_rows_per_depth}"
                )

        ood_depth = next(depth for depth in DEPTHS if depth > train_max)
        slices = {
            "shallow": 1,
            "train_edge": train_max,
            "ood_edge": ood_depth,
            "depth50": 50,
        }
        selections: dict[str, list[dict[str, Any]]] = {}
        validity_key = (
            "citation_free_valid" if template == "logic" else "nl_logic_citation_free_valid"
        )
        category_counts = Counter()
        for row_number, row in enumerate(rows):
            for metric_name in ("correct", "format_ok", validity_key):
                if not _is_unit_metric(row.get(metric_name)):
                    errors.append(
                        f"{path.name}: sampled row {row_number} has invalid "
                        f"{metric_name}={row.get(metric_name)!r}"
                    )
            correct = _is_positive(row.get("correct"))
            valid = _is_positive(row.get(validity_key))
            category_counts["correct" if correct else "incorrect"] += 1
            category_counts["valid" if valid else "invalid"] += 1
            category_counts["format_ok" if _is_positive(row.get("format_ok")) else "format_failure"] += 1
        coverage.update(f"{template}:{name}" for name, count in category_counts.items() if count)
        for name, depth in slices.items():
            selections[name] = _select_slice(by_depth[depth], path)

        array_index = _array_index(template, train_max, seed)
        log_path = log_dir / f"eval_bp_unique_{eval_array_job_id}_{array_index}.out"
        cap_chunks = _cap_chunks(log_path, generation_cap, errors)
        cap_examples: list[dict[str, Any]] = []
        for chunk in cap_chunks[:3]:
            start = (chunk - 1) * 8
            candidates = rows[start : start + 8]
            if candidates:
                longest = max(
                    candidates,
                    key=lambda row: len(row.get("generation", ""))
                    if isinstance(row.get("generation"), str)
                    else 0,
                )
                cap_examples.append(
                    _excerpt(
                        longest,
                        path=path,
                        label=f"cap_hit_chunk_{chunk}_longest_retained",
                    )
                )
        if cap_examples:
            coverage[f"{template}:cap_hit_chunk"] += 1

        runs.append(
            {
                "template": template,
                "train_max": train_max,
                "seed": seed,
                "sample_file": str(path),
                "eval_log": str(log_path),
                "sampled_rows": len(rows),
                "category_counts": dict(sorted(category_counts.items())),
                "cap_hit_chunks": cap_chunks,
                "slice_selections": selections,
                "cap_hit_examples": cap_examples,
            }
        )

    return {
        "accepted": not errors,
        "errors": errors,
        "expected_grid_size": len(expected_grid),
        "observed_grid_size": len(paths),
        "coverage": dict(sorted(coverage.items())),
        "runs": runs,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Corrected BranchProof qualitative audit",
        "",
        f"Structural acceptance: **{report['accepted']}**.",
        "",
        "Each run contributes shallow, train-edge, first-OOD, and depth-50 selections. "
        "Labels distinguish correct+valid, correct+invalid, and incorrect retained generations "
        "when those cases exist. For each generation chunk whose observed maximum reached the "
        "configured cap, the audit references the longest of the retained generations; that sample "
        "is a cap-hit diagnostic, not proof that it was the exact generation that reached the cap.",
        "",
        "| modality | train max | seed | correct | incorrect | valid | invalid | cap chunks |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report["runs"]:
        counts = run["category_counts"]
        lines.append(
            f"| `{run['template']}` | {run['train_max']} | {run['seed']} | "
            f"{counts.get('correct', 0)} | {counts.get('incorrect', 0)} | "
            f"{counts.get('valid', 0)} | {counts.get('invalid', 0)} | "
            f"{len(run['cap_hit_chunks'])} |"
        )
    lines.extend(["", "## Selected generations", ""])
    for run in report["runs"]:
        lines.append(
            f"### {run['template']} train 1..{run['train_max']} seed {run['seed']}"
        )
        for slice_name, examples in run["slice_selections"].items():
            for example in examples:
                lines.append(
                    f"- `{slice_name}/{example['label']}`: depth {example['step']}, "
                    f"line {example['line_number']}, correct={example['correct']}, "
                    f"format={example['format_ok']}, chars={example['generation_chars']}"
                )
        for example in run["cap_hit_examples"]:
            lines.append(
                f"- `{example['label']}`: depth {example['step']}, line "
                f"{example['line_number']}, correct={example['correct']}, "
                f"format={example['format_ok']}, chars={example['generation_chars']}"
            )
        lines.append("")
    if report["errors"]:
        lines.extend(["## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = audit_grid(
        args.final_dir,
        args.log_dir,
        eval_array_job_id=args.eval_array_job_id,
        expected_sampled_rows=args.expected_sampled_rows,
        expected_rows_per_depth=args.expected_rows_per_depth,
        generation_cap=args.generation_cap,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(report, args.output_markdown)
    if not report["accepted"]:
        raise SystemExit(f"BranchProof qualitative grid audit failed; see {args.output_json}")
    print(json.dumps({key: report[key] for key in ("accepted", "coverage")}, indent=2))


if __name__ == "__main__":
    main()
