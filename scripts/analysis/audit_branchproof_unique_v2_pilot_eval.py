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
CHUNK_DONE_RE = re.compile(
    r"^\[syntheval\] (?P<mode>greedy|sampled) vLLM chunk "
    r"(?P<index>\d+)/(?P<total>\d+) done in (?P<seconds>[0-9.]+)s "
    r"\((?P<tokens>\d+) output tokens, max=(?P<maximum>\d+)\)$"
)
GREEDY_METRICS = (
    "syntactic",
    "format",
    "correct",
    "valid",
    "citation_free_valid",
    "grounded_valid",
    "citation_free_grounded_valid",
)
NL_GREEDY_METRICS = (
    "nl_logic_parse",
    "nl_logic_citation_free_valid",
    "nl_logic_joint",
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
NL_SAMPLED_METRICS = (
    "nl_logic_parse_pass",
    "nl_logic_citation_free_valid_pass",
    "nl_logic_joint_pass",
)
SAMPLE_METRICS = (
    "syntactic",
    "format_ok",
    "correct",
    "valid",
    "citation_free_valid",
    "nl_logic_parse",
    "nl_logic_citation_free_valid",
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
    parser.add_argument("--eval-log", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--generation-cap", type=int, default=7168)
    parser.add_argument("--greedy-optional", action="store_true")
    parser.add_argument("--expected-sample-source", default="synthetic")
    parser.add_argument("--sample-source-filter")
    parser.add_argument("--expected-total-samples", type=int)
    parser.add_argument("--expected-samples-per-step", type=int)
    parser.add_argument("--expected-unique-prompts-per-step", type=int)
    parser.add_argument("--expected-sample-indices", type=_csv_ints)
    parser.add_argument("--skip-fresh-constant-check", action="store_true")
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
    greedy_required: bool,
    errors: list[str],
) -> dict[str, Any]:
    if payload.get("profile") != "sft":
        errors.append(f"unexpected evaluation profile: {payload.get('profile')!r}")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, str) or "branchproof_unique_v2" not in checkpoint:
        errors.append(f"unexpected checkpoint path: {checkpoint!r}")
    is_nl = isinstance(checkpoint, str) and "_nl_exact_" in checkpoint
    greedy_metrics = GREEDY_METRICS + (NL_GREEDY_METRICS if is_nl else ())
    sampled_metrics = SAMPLED_METRICS + (NL_SAMPLED_METRICS if is_nl else ())

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
        if greedy_required:
            for metric_name in greedy_metrics:
                key = f"synthetic/step_{step}/{metric_name}"
                value = _check_unit_metric(metrics, key, errors)
                if value is not None:
                    primary[str(step)][metric_name] = value
        for metric_name in sampled_metrics:
            previous: float | None = None
            for k in k_values:
                key = f"synthetic_sampled/step_{step}/{metric_name}@{k}"
                value = _check_unit_metric(metrics, key, errors)
                if value is not None:
                    sampled[str(step)][metric_name][str(k)] = value
                    if previous is not None and value + 1e-9 < previous:
                        errors.append(f"non-monotonic pass@k metric: {key}={value} < {previous}")
                    previous = value

    if greedy_required:
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


def _audit_eval_log(
    path: Path,
    *,
    expected_greedy_chunks: int,
    expected_sampled_chunks: int,
    generation_cap: int,
    errors: list[str],
) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"evaluation log does not exist: {path}")
        return {"path": str(path), "exists": False}

    records: dict[str, list[dict[str, int | float]]] = {"greedy": [], "sampled": []}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = CHUNK_DONE_RE.match(line.strip())
        if match is None:
            continue
        records[match.group("mode")].append(
            {
                "index": int(match.group("index")),
                "total": int(match.group("total")),
                "seconds": float(match.group("seconds")),
                "output_tokens": int(match.group("tokens")),
                "max_output_tokens": int(match.group("maximum")),
            }
        )

    result: dict[str, Any] = {"path": str(path), "exists": True}
    for mode, expected_count in (
        ("greedy", expected_greedy_chunks),
        ("sampled", expected_sampled_chunks),
    ):
        mode_records = records[mode]
        indices = [int(record["index"]) for record in mode_records]
        totals = {int(record["total"]) for record in mode_records}
        expected_indices = list(range(1, expected_count + 1))
        if indices != expected_indices:
            errors.append(
                f"{mode} completed chunk indices={indices}, expected {expected_indices}"
            )
        if totals != {expected_count}:
            errors.append(
                f"{mode} logged chunk totals={sorted(totals)}, expected [{expected_count}]"
            )

        elapsed = sum(float(record["seconds"]) for record in mode_records)
        output_tokens = sum(int(record["output_tokens"]) for record in mode_records)
        max_output_tokens = max(
            (int(record["max_output_tokens"]) for record in mode_records), default=0
        )
        if max_output_tokens > generation_cap:
            errors.append(
                f"{mode} max output tokens={max_output_tokens} exceeds cap {generation_cap}"
            )
        result[mode] = {
            "expected_chunks": expected_count,
            "completed_chunks": len(mode_records),
            "elapsed_seconds": elapsed,
            "output_tokens": output_tokens,
            "tokens_per_second": output_tokens / elapsed if elapsed else 0.0,
            "max_output_tokens": max_output_tokens,
            "generation_cap": generation_cap,
            "cap_hit_chunk_count": sum(
                int(record["max_output_tokens"]) == generation_cap
                for record in mode_records
            ),
            "chunks": mode_records,
        }
    return result


def _audit_samples(
    rows: list[dict[str, Any]],
    *,
    steps: Iterable[int],
    expected_retained_samples: int,
    expected_source: str,
    expected_samples_per_step: int | None,
    expected_unique_prompts_per_step: int | None,
    expected_sample_indices: list[int] | None,
    require_fresh_constants: bool,
    errors: list[str],
) -> dict[str, Any]:
    expected_steps = set(steps)
    if len(rows) != expected_retained_samples:
        errors.append(f"retained sample count={len(rows)}, expected {expected_retained_samples}")

    counts: Counter[int] = Counter()
    nonempty_generations = 0
    constant_failures = 0
    by_step: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sample_indices_by_step: dict[int, Counter[int]] = defaultdict(Counter)
    prompts_by_step: dict[int, set[str]] = defaultdict(set)
    representatives: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        step = row.get("step")
        if not isinstance(step, int) or step not in expected_steps:
            errors.append(f"sample {index} has unexpected step: {step!r}")
            continue
        counts[step] += 1
        if row.get("source") != expected_source:
            errors.append(f"sample {index} has unexpected source: {row.get('source')!r}")

        prompt = row.get("prompt")
        generation = row.get("generation")
        gold_answer = row.get("gold_answer")
        if not isinstance(prompt, str) or not prompt.strip():
            errors.append(f"sample {index} has an empty prompt")
            continue
        if not isinstance(gold_answer, str) or not gold_answer.strip():
            errors.append(f"sample {index} has an empty gold answer")
        prompts_by_step[step].add(prompt)
        if expected_sample_indices is not None:
            sample_index = row.get("sample_index")
            if not isinstance(sample_index, int):
                errors.append(f"sample {index} has invalid sample_index: {sample_index!r}")
            else:
                sample_indices_by_step[step][sample_index] += 1
        if isinstance(generation, str) and generation.strip():
            nonempty_generations += 1
            by_step[step]["nonempty_generation"] += 1

        if require_fresh_constants:
            question = QUESTION_RE.search(prompt)
            constants = [] if question is None else sorted(
                {int(value) for value in CONSTANT_RE.findall(question.group(1))}
            )
            expected_constants = list(range(step + 1))
            if constants != expected_constants:
                constant_failures += 1
                errors.append(
                    f"sample {index} step {step} constants={constants[:4]}...{constants[-4:]}, "
                    f"expected c0..c{step}"
                )
            else:
                by_step[step]["fresh_constants"] += 1

        for field in SAMPLE_METRICS:
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                errors.append(f"sample {index} has invalid {field}: {value!r}")
            elif value > 0:
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
                "citation_free_valid": row.get("citation_free_valid"),
                "nl_logic_parse": row.get("nl_logic_parse"),
                "nl_logic_citation_free_valid": row.get("nl_logic_citation_free_valid"),
            },
        )

    missing_steps = sorted(expected_steps - counts.keys())
    if missing_steps:
        errors.append(f"retained samples do not cover steps: {missing_steps}")
    if rows and nonempty_generations == 0:
        errors.append("all retained generations are empty")
    if expected_samples_per_step is not None:
        for step in sorted(expected_steps):
            if counts[step] != expected_samples_per_step:
                errors.append(
                    f"step {step} retained samples={counts[step]}, "
                    f"expected {expected_samples_per_step}"
                )
    if expected_unique_prompts_per_step is not None:
        for step in sorted(expected_steps):
            observed = len(prompts_by_step[step])
            if observed != expected_unique_prompts_per_step:
                errors.append(
                    f"step {step} unique prompts={observed}, "
                    f"expected {expected_unique_prompts_per_step}"
                )
    if expected_sample_indices is not None:
        expected_index_set = set(expected_sample_indices)
        expected_count_per_index = (
            expected_samples_per_step // len(expected_sample_indices)
            if expected_samples_per_step is not None and expected_sample_indices
            else None
        )
        for step in sorted(expected_steps):
            observed = sample_indices_by_step[step]
            if set(observed) != expected_index_set:
                errors.append(
                    f"step {step} sample indices={sorted(observed)}, "
                    f"expected {sorted(expected_index_set)}"
                )
            if expected_count_per_index is not None:
                for sample_index in expected_sample_indices:
                    if observed[sample_index] != expected_count_per_index:
                        errors.append(
                            f"step {step} sample_index {sample_index} count="
                            f"{observed[sample_index]}, expected {expected_count_per_index}"
                        )

    return {
        "sample_count": len(rows),
        "step_counts": {str(step): counts[step] for step in sorted(counts)},
        "nonempty_generation_count": nonempty_generations,
        "fresh_constant_failure_count": constant_failures,
        "source": expected_source,
        "unique_prompt_counts": {
            str(step): len(prompts_by_step[step]) for step in sorted(prompts_by_step)
        },
        "sample_index_counts": {
            str(step): dict(sorted(sample_indices_by_step[step].items()))
            for step in sorted(sample_indices_by_step)
        },
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
    eval_log_path: Path | None = None,
    batch_size: int = 64,
    generation_cap: int = 7168,
    greedy_required: bool = True,
    expected_sample_source: str = "synthetic",
    sample_source_filter: str | None = None,
    expected_total_samples: int | None = None,
    expected_samples_per_step: int | None = None,
    expected_unique_prompts_per_step: int | None = None,
    expected_sample_indices: list[int] | None = None,
    require_fresh_constants: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    all_rows = _read_samples(samples_path, errors)
    if expected_total_samples is not None and len(all_rows) != expected_total_samples:
        errors.append(
            f"total retained sample count={len(all_rows)}, expected {expected_total_samples}"
        )
    rows = (
        [row for row in all_rows if row.get("source") == sample_source_filter]
        if sample_source_filter is not None
        else all_rows
    )
    prompt_count = len(steps) * samples_per_step
    sampled_prompt_batch_size = max(1, batch_size // generations_per_prompt)
    report = {
        "metrics_path": str(metrics_path),
        "samples_path": str(samples_path),
        "total_sample_count": len(all_rows),
        "sample_source_filter": sample_source_filter,
        "expected_steps": steps,
        "expected_k_values": k_values,
        "metrics_audit": _audit_metrics(
            payload,
            steps=steps,
            k_values=k_values,
            samples_per_step=samples_per_step,
            generations_per_prompt=generations_per_prompt,
            train_max=train_max,
            greedy_required=greedy_required,
            errors=errors,
        ),
        "samples_audit": _audit_samples(
            rows,
            steps=steps,
            expected_retained_samples=expected_retained_samples,
            expected_source=expected_sample_source,
            expected_samples_per_step=expected_samples_per_step,
            expected_unique_prompts_per_step=expected_unique_prompts_per_step,
            expected_sample_indices=expected_sample_indices,
            require_fresh_constants=require_fresh_constants,
            errors=errors,
        ),
        "generation_log_audit": (
            _audit_eval_log(
                eval_log_path,
                expected_greedy_chunks=math.ceil(prompt_count / batch_size),
                expected_sampled_chunks=math.ceil(prompt_count / sampled_prompt_batch_size),
                generation_cap=generation_cap,
                errors=errors,
            )
            if eval_log_path is not None
            else None
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
        eval_log_path=args.eval_log,
        batch_size=args.batch_size,
        generation_cap=args.generation_cap,
        greedy_required=not args.greedy_optional,
        expected_sample_source=args.expected_sample_source,
        sample_source_filter=args.sample_source_filter,
        expected_total_samples=args.expected_total_samples,
        expected_samples_per_step=args.expected_samples_per_step,
        expected_unique_prompts_per_step=args.expected_unique_prompts_per_step,
        expected_sample_indices=args.expected_sample_indices,
        require_fresh_constants=not args.skip_fresh_constant_check,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["accepted"]:
        raise SystemExit("BranchProof pilot evaluation audit failed; see " + str(args.output))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
