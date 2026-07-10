#!/usr/bin/env python3
"""Structurally audit the corrected BranchProof pilot evaluation."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


QUESTION_RE = re.compile(r"<question>\s*(.*?)\s*</question>", re.DOTALL)
CONSTANT_RE = re.compile(r"\bc(\d+)\b")
GREEDY_METRICS = (
    "syntactic",
    "format",
    "correct",
    "valid",
    "citation_free_valid",
    "grounded_valid",
    "citation_free_grounded_valid",
)
SAMPLED_METRICS = (
    "syntactic_pass",
    "format_pass",
    "correct_pass",
    "valid_pass",
    "joint_pass",
    "citation_free_valid_pass",
    "citation_free_joint_pass",
    "grounded_valid_pass",
    "grounded_joint_pass",
    "citation_free_grounded_valid_pass",
    "citation_free_grounded_joint_pass",
)


def _csv_ints(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--steps",
        type=_csv_ints,
        default=_csv_ints("1,2,5,10,12,15,18,20,25,30,35,40,45,50"),
    )
    parser.add_argument("--k-values", type=_csv_ints, default=_csv_ints("1,2,4,8,16"))
    parser.add_argument("--samples-per-step", type=int, default=32)
    parser.add_argument("--generations-per-prompt", type=int, default=16)
    parser.add_argument("--expected-retained-samples", type=int, default=128)
    parser.add_argument("--train-max", type=int, default=25)
    return parser.parse_args()


def _check_unit_metric(metrics: dict[str, Any], key: str, errors: list[str]) -> float | None:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"missing or non-numeric metric: {key}")
        return None
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        errors.append(f"metric outside [0, 1]: {key}={value}")
        return None
    return value


def _audit_metrics(
    payload: dict[str, Any],
    *,
    steps: Iterable[int],
    k_values: Iterable[int],
    samples_per_step: int,
    generations_per_prompt: int,
    train_max: int,
    errors: list[str],
) -> dict[str, Any]:
    if payload.get("profile") != "sft":
        errors.append(f"unexpected evaluation profile: {payload.get('profile')!r}")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, str) or "branchproof_unique_v2" not in checkpoint:
        errors.append(f"unexpected checkpoint path: {checkpoint!r}")

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("metrics payload is not an object")
        return {"metric_count": 0}

    expected_prompts = len(list(steps)) * samples_per_step
    if metrics.get("posthoc/prompts") != expected_prompts:
        errors.append(
            f"posthoc/prompts={metrics.get('posthoc/prompts')!r}, expected {expected_prompts}"
        )
    if metrics.get("posthoc/sampled_generations_per_prompt") != generations_per_prompt:
        errors.append(
            "posthoc/sampled_generations_per_prompt="
            f"{metrics.get('posthoc/sampled_generations_per_prompt')!r}, "
            f"expected {generations_per_prompt}"
        )

    primary: dict[str, dict[str, float]] = defaultdict(dict)
    sampled: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for step in steps:
        for metric_name in GREEDY_METRICS:
            key = f"synthetic/step_{step}/{metric_name}"
            value = _check_unit_metric(metrics, key, errors)
            if value is not None:
                primary[str(step)][metric_name] = value
        for metric_name in SAMPLED_METRICS:
            previous: float | None = None
            for k in k_values:
                key = f"synthetic_sampled/step_{step}/{metric_name}@{k}"
                value = _check_unit_metric(metrics, key, errors)
                if value is not None:
                    sampled[str(step)][metric_name][str(k)] = value
                    if previous is not None and value + 1e-9 < previous:
                        errors.append(f"non-monotonic pass@k metric: {key}={value} < {previous}")
                    previous = value

    train_steps = [step for step in steps if step <= train_max]
    for metric_name in ("syntactic", "format", "correct"):
        values = [primary[str(step)].get(metric_name, 0.0) for step in train_steps]
        if not any(value > 0.0 for value in values):
            errors.append(f"all train-band greedy {metric_name} metrics are zero")

    return {
        "metric_count": len(metrics),
        "greedy": dict(primary),
        "sampled": {step: dict(values) for step, values in sampled.items()},
    }


def _read_samples(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                errors.append(f"blank sample line: {line_number}")
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"invalid sample JSON at line {line_number}: {exc}")
                continue
            if not isinstance(row, dict):
                errors.append(f"sample line {line_number} is not an object")
                continue
            rows.append(row)
    return rows


def _audit_samples(
    rows: list[dict[str, Any]],
    *,
    steps: Iterable[int],
    expected_retained_samples: int,
    errors: list[str],
) -> dict[str, Any]:
    expected_steps = set(steps)
    if len(rows) != expected_retained_samples:
        errors.append(f"retained sample count={len(rows)}, expected {expected_retained_samples}")

    counts: Counter[int] = Counter()
    nonempty_generations = 0
    constant_failures = 0
    by_step: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    representatives: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        step = row.get("step")
        if not isinstance(step, int) or step not in expected_steps:
            errors.append(f"sample {index} has unexpected step: {step!r}")
            continue
        counts[step] += 1
        if row.get("source") != "synthetic":
            errors.append(f"sample {index} has unexpected source: {row.get('source')!r}")

        prompt = row.get("prompt")
        generation = row.get("generation")
        gold_answer = row.get("gold_answer")
        if not isinstance(prompt, str) or not prompt.strip():
            errors.append(f"sample {index} has an empty prompt")
            continue
        if not isinstance(gold_answer, str) or not gold_answer.strip():
            errors.append(f"sample {index} has an empty gold answer")
        if isinstance(generation, str) and generation.strip():
            nonempty_generations += 1
            by_step[step]["nonempty_generation"] += 1

        question = QUESTION_RE.search(prompt)
        constants = [] if question is None else sorted({int(value) for value in CONSTANT_RE.findall(question.group(1))})
        expected_constants = list(range(step + 1))
        if constants != expected_constants:
            constant_failures += 1
            errors.append(
                f"sample {index} step {step} constants={constants[:4]}...{constants[-4:]}, "
                f"expected c0..c{step}"
            )
        else:
            by_step[step]["fresh_constants"] += 1

        for field in ("syntactic", "format_ok", "correct", "valid"):
            value = row.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                by_step[step][field] += 1
        representatives.setdefault(
            str(step),
            {
                "gold_answer": gold_answer,
                "prompt_head": prompt[:500],
                "generation_head": generation[:800] if isinstance(generation, str) else generation,
                "syntactic": row.get("syntactic"),
                "format_ok": row.get("format_ok"),
                "correct": row.get("correct"),
                "valid": row.get("valid"),
            },
        )

    missing_steps = sorted(expected_steps - counts.keys())
    if missing_steps:
        errors.append(f"retained samples do not cover steps: {missing_steps}")
    if rows and nonempty_generations == 0:
        errors.append("all retained generations are empty")

    return {
        "sample_count": len(rows),
        "step_counts": {str(step): counts[step] for step in sorted(counts)},
        "nonempty_generation_count": nonempty_generations,
        "fresh_constant_failure_count": constant_failures,
        "by_step": {str(step): dict(by_step[step]) for step in sorted(by_step)},
        "representative_samples": representatives,
    }


def audit_artifacts(
    metrics_path: Path,
    samples_path: Path,
    *,
    steps: list[int],
    k_values: list[int],
    samples_per_step: int,
    generations_per_prompt: int,
    expected_retained_samples: int,
    train_max: int,
) -> dict[str, Any]:
    errors: list[str] = []
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = _read_samples(samples_path, errors)
    report = {
        "metrics_path": str(metrics_path),
        "samples_path": str(samples_path),
        "expected_steps": steps,
        "expected_k_values": k_values,
        "metrics_audit": _audit_metrics(
            payload,
            steps=steps,
            k_values=k_values,
            samples_per_step=samples_per_step,
            generations_per_prompt=generations_per_prompt,
            train_max=train_max,
            errors=errors,
        ),
        "samples_audit": _audit_samples(
            rows,
            steps=steps,
            expected_retained_samples=expected_retained_samples,
            errors=errors,
        ),
    }
    report["errors"] = errors
    report["accepted"] = not errors
    return report


def main() -> None:
    args = parse_args()
    report = audit_artifacts(
        args.metrics,
        args.samples,
        steps=args.steps,
        k_values=args.k_values,
        samples_per_step=args.samples_per_step,
        generations_per_prompt=args.generations_per_prompt,
        expected_retained_samples=args.expected_retained_samples,
        train_max=args.train_max,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["accepted"]:
        raise SystemExit("BranchProof pilot evaluation audit failed; see " + str(args.output))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
