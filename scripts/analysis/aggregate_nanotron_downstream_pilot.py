#!/usr/bin/env python3
"""Aggregate the matched control/logic/NL Nanotron downstream pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_nanotron_downstream_eval import (
    FINAL_TASKS,
    MATH_POSTHOC_METRIC,
    MATH_POSTHOC_SIDECAR,
    PRIMARY_METRICS,
    audit_run,
)


CONDITIONS = ("control", "logic", "nl_exact")
BRANCHES = ("direct", "instruction")
RUN_TAGS = {
    "control": "qwen25_7b_midtrain_control_p0_4p3b",
    "logic": "qwen25_7b_midtrain_logic_p15_bp_unique_v2_4p3b",
    "nl_exact": "qwen25_7b_midtrain_nl_exact_p15_bp_unique_v2_4p3b",
}
TARGETED_METRICS = {
    "mmlu_formal_logic": "acc,none",
    "bbh_cot_fewshot_boolean_expressions": "exact_match,get-answer",
    "bbh_cot_fewshot_formal_fallacies": "exact_match,get-answer",
    "bbh_cot_fewshot_logical_deduction_three_objects": "exact_match,get-answer",
    "bbh_cot_fewshot_logical_deduction_five_objects": "exact_match,get-answer",
    "bbh_cot_fewshot_logical_deduction_seven_objects": "exact_match,get-answer",
}
MACROS = {
    "all_primary": tuple(FINAL_TASKS),
    "reasoning_core": (
        "gsm8k",
        "hendrycks_math500",
        "agieval_logiqa_en",
        "bbh",
        "mmlu_pro",
    ),
    "general_multiple_choice": (
        "arc_challenge",
        "hellaswag",
        "winogrande",
        "piqa",
        "mmlu",
    ),
    "logic_targeted": (
        "agieval_logiqa_en",
        "mmlu_formal_logic",
        "bbh_cot_fewshot_formal_fallacies",
        "bbh_cot_fewshot_logical_deduction_three_objects",
        "bbh_cot_fewshot_logical_deduction_five_objects",
        "bbh_cot_fewshot_logical_deduction_seven_objects",
    ),
}
QUALITATIVE_TASKS = (
    "gsm8k",
    "agieval_logiqa_en",
    "mmlu_formal_logic",
    "bbh_cot_fewshot_logical_deduction_three_objects",
    "mmlu_pro_computer_science",
)
GENERATION_FILTERS = {
    "gsm8k": "flexible-extract",
    "math500": "none",
    "bbh": "get-answer",
    "mmlu_pro": "custom-extract",
}
NEXT_DOCUMENT_MARKER = "You are an AI assistant that helps people find information."


@dataclass(frozen=True)
class Bundle:
    condition: str
    branch: str
    run_dir: Path
    payload: dict[str, Any]
    math_posthoc: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=8192)
    return parser.parse_args()


def _section(payload: dict[str, Any], task: str) -> dict[str, Any] | None:
    section = payload.get("results", {}).get(task)
    if not isinstance(section, dict):
        section = payload.get("groups", {}).get(task)
    return section if isinstance(section, dict) else None


def _finite_metric(payload: dict[str, Any], task: str, metric: str) -> float:
    section = _section(payload, task)
    value = None if section is None else section.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"missing numeric metric {task}/{metric}")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"invalid metric {task}/{metric}={result}")
    return result


def _stderr(payload: dict[str, Any], task: str, metric: str) -> float | None:
    section = _section(payload, task)
    name, filter_name = metric.split(",", 1)
    value = None if section is None else section.get(f"{name}_stderr,{filter_name}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0.0 else None


def _result_payload(run_dir: Path) -> dict[str, Any]:
    result_files = sorted(run_dir.rglob("results_*.json"))
    if len(result_files) != 1:
        raise ValueError(f"{run_dir}: expected one result JSON, found {len(result_files)}")
    return json.loads(result_files[0].read_text(encoding="utf-8"))


def _math_posthoc_payload(run_dir: Path) -> dict[str, Any]:
    path = run_dir / MATH_POSTHOC_SIDECAR
    if not path.is_file():
        raise ValueError(f"{run_dir}: missing MATH-500 post-hoc sidecar {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("accepted") or payload.get("scorer") != MATH_POSTHOC_METRIC:
        raise ValueError(f"{run_dir}: invalid MATH-500 post-hoc sidecar")
    return payload


def load_bundles(root: Path, *, checkpoint_step: int, validate: bool = True) -> list[Bundle]:
    bundles: list[Bundle] = []
    for condition in CONDITIONS:
        for branch in BRANCHES:
            run_dir = root / f"{RUN_TAGS[condition]}_step{checkpoint_step}_{branch}"
            if validate:
                report = audit_run(
                    run_dir,
                    mode=branch,
                    expected_tasks=list(FINAL_TASKS),
                    require_full=True,
                )
                if not report["accepted"]:
                    raise ValueError(
                        f"production audit failed for {condition}/{branch}: {report['errors']}"
                    )
            bundles.append(
                Bundle(
                    condition=condition,
                    branch=branch,
                    run_dir=run_dir,
                    payload=_result_payload(run_dir),
                    math_posthoc=_math_posthoc_payload(run_dir),
                )
            )
    return bundles


def task_rows(bundles: list[Bundle]) -> list[dict[str, Any]]:
    metric_map = dict(PRIMARY_METRICS)
    metric_map.update(TARGETED_METRICS)
    raw: list[dict[str, Any]] = []
    for bundle in bundles:
        for task, metric in metric_map.items():
            if task == "hendrycks_math500":
                value = bundle.math_posthoc.get("accuracy")
                stderr = bundle.math_posthoc.get("stderr")
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1.0
                ):
                    raise ValueError(f"invalid post-hoc MATH-500 accuracy: {value!r}")
                if (
                    stderr is not None
                    and (
                        isinstance(stderr, bool)
                        or not isinstance(stderr, (int, float))
                        or not math.isfinite(float(stderr))
                        or float(stderr) < 0.0
                    )
                ):
                    raise ValueError(f"invalid post-hoc MATH-500 stderr: {stderr!r}")
                stock_value = _finite_metric(bundle.payload, task, metric)
                stock_stderr = _stderr(bundle.payload, task, metric)
                output_metric = MATH_POSTHOC_METRIC
            else:
                value = _finite_metric(bundle.payload, task, metric)
                stderr = _stderr(bundle.payload, task, metric)
                stock_value = None
                stock_stderr = None
                output_metric = metric
            raw.append(
                {
                    "condition": bundle.condition,
                    "branch": bundle.branch,
                    "task": task,
                    "metric": output_metric,
                    "value": float(value),
                    "stderr": None if stderr is None else float(stderr),
                    "stock_metric": metric if task == "hendrycks_math500" else None,
                    "stock_value": stock_value,
                    "stock_stderr": stock_stderr,
                }
            )
    control = {
        (row["branch"], row["task"]): float(row["value"])
        for row in raw
        if row["condition"] == "control"
    }
    for row in raw:
        row["delta_vs_control"] = float(row["value"]) - control[(row["branch"], row["task"])]
    by_condition_task = {
        (str(row["condition"]), str(row["branch"]), str(row["task"])): float(row["value"])
        for row in raw
    }
    for row in raw:
        condition = str(row["condition"])
        task = str(row["task"])
        row["instruction_minus_direct"] = (
            by_condition_task[(condition, "instruction", task)]
            - by_condition_task[(condition, "direct", task)]
        )
    return raw


def macro_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run_task = {
        (str(row["condition"]), str(row["branch"]), str(row["task"])): float(row["value"])
        for row in rows
    }
    output: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for branch in BRANCHES:
            for macro, tasks in MACROS.items():
                values = [by_run_task[(condition, branch, task)] for task in tasks]
                output.append(
                    {
                        "condition": condition,
                        "branch": branch,
                        "macro": macro,
                        "task_count": len(tasks),
                        "value": mean(values),
                    }
                )
    by_key = {
        (str(row["condition"]), str(row["branch"]), str(row["macro"])): float(row["value"])
        for row in output
    }
    for row in output:
        condition = str(row["condition"])
        branch = str(row["branch"])
        macro = str(row["macro"])
        row["delta_vs_control"] = float(row["value"]) - by_key[("control", branch, macro)]
        row["instruction_minus_direct"] = (
            by_key[(condition, "instruction", macro)] - by_key[(condition, "direct", macro)]
        )
    return output


def _sample_files(run_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in run_dir.rglob("samples_*.jsonl"):
        task = path.name.removeprefix("samples_").split("_202", 1)[0]
        files[task] = path
    return files


def _prompt(row: dict[str, Any]) -> str:
    arguments = row.get("arguments")
    if not isinstance(arguments, dict):
        return ""
    for value in arguments.values():
        if not isinstance(value, dict):
            continue
        for key, candidate in value.items():
            if str(key).startswith("arg_") and isinstance(candidate, str):
                return candidate
    return ""


def _first_response(row: dict[str, Any]) -> str:
    responses = row.get("resps")
    if (
        isinstance(responses, list)
        and responses
        and isinstance(responses[0], list)
        and responses[0]
        and isinstance(responses[0][0], str)
    ):
        return responses[0][0]
    return ""


def _first_filtered_response(row: dict[str, Any]) -> str:
    responses = row.get("filtered_resps")
    if isinstance(responses, list) and responses and isinstance(responses[0], str):
        return responses[0]
    return ""


def _generation_family(task: str) -> str | None:
    if task == "gsm8k":
        return "gsm8k"
    if task == "hendrycks_math500":
        return "math500"
    if task.startswith("bbh_cot_fewshot_"):
        return "bbh"
    if task.startswith("mmlu_pro_"):
        return "mmlu_pro"
    return None


def generation_diagnostic_rows(bundles: list[Bundle]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for bundle in bundles:
        grouped: dict[str, list[tuple[str, str, str]]] = {
            family: [] for family in GENERATION_FILTERS
        }
        for task, path in _sample_files(bundle.run_dir).items():
            family = _generation_family(task)
            if family is None:
                continue
            expected_filter = GENERATION_FILTERS[family]
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("filter") != expected_filter:
                        continue
                    grouped[family].append(
                        (
                            _first_response(row),
                            _first_filtered_response(row),
                            _prompt(row),
                        )
                    )

        for family, rows in grouped.items():
            if not rows:
                raise ValueError(
                    f"missing generation diagnostics for {bundle.condition}/{bundle.branch}/{family}"
                )
            lengths = sorted(len(response) for response, _, _ in rows)
            invalid_count = sum(
                extracted.strip().lower() in {"", "[invalid]"}
                for _, extracted, _ in rows
            )
            marker_count = sum(
                NEXT_DOCUMENT_MARKER in response for response, _, _ in rows
            )
            prompt_marker_count = sum(
                NEXT_DOCUMENT_MARKER in prompt for _, _, prompt in rows
            )
            count = len(rows)
            output.append(
                {
                    "condition": bundle.condition,
                    "branch": bundle.branch,
                    "family": family,
                    "row_count": count,
                    "invalid_extraction_count": invalid_count,
                    "invalid_extraction_rate": invalid_count / count,
                    "next_document_marker_count": marker_count,
                    "next_document_marker_rate": marker_count / count,
                    "prompt_marker_count": prompt_marker_count,
                    "prompt_marker_rate": prompt_marker_count / count,
                    "response_chars_mean": mean(lengths),
                    "response_chars_p50": median(lengths),
                    "response_chars_p95": lengths[math.ceil(0.95 * count) - 1],
                    "response_chars_max": lengths[-1],
                }
            )
    return output


def _qualitative_metric(task: str) -> tuple[str, str]:
    metric = PRIMARY_METRICS.get(task) or TARGETED_METRICS.get(task)
    if metric is None and task.startswith("mmlu_pro_"):
        metric = PRIMARY_METRICS["mmlu_pro"]
    if metric is None:
        raise ValueError(f"no qualitative metric declared for {task}")
    metric_name, filter_name = metric.split(",", 1)
    return metric_name, filter_name


def _qualitative_score(row: dict[str, Any], metric_name: str) -> float | None:
    value = row.get(metric_name)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def qualitative_rows(bundles: list[Bundle]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for bundle in bundles:
        sample_files = _sample_files(bundle.run_dir)
        for task in QUALITATIVE_TASKS:
            path = sample_files.get(task)
            if path is None:
                raise ValueError(f"missing qualitative sample file {bundle.condition}/{bundle.branch}/{task}")
            metric_name, filter_name = _qualitative_metric(task)
            candidates: dict[str, tuple[int, dict[str, Any]]] = {}
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("filter") != filter_name:
                        continue
                    score = _qualitative_score(row, metric_name)
                    label = "correct" if score is not None and score > 0.0 else "incorrect"
                    candidates.setdefault(label, (line_number, row))
                    if len(candidates) == 2:
                        break
            for label, (line_number, row) in sorted(candidates.items()):
                response = json.dumps(row.get("filtered_resps"), ensure_ascii=True)
                target = str(row.get("target", ""))
                prompt = _prompt(row)
                selected.append(
                    {
                        "condition": bundle.condition,
                        "branch": bundle.branch,
                        "task": task,
                        "label": label,
                        "sample_file": str(path),
                        "line_number": line_number,
                        "doc_id": row.get("doc_id"),
                        "metric": f"{metric_name},{filter_name}",
                        "score": _qualitative_score(row, metric_name),
                        "prompt_head": prompt[:1000],
                        "target_head": target[:500],
                        "response_head": response[:800],
                    }
                )
    return selected


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    tasks: list[dict[str, Any]],
    macros: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    qualitative: list[dict[str, Any]],
) -> None:
    lines = [
        "# Corrected Nanotron p15 downstream comparison",
        "",
        "Each condition is one continuation-training run. Task-level stderr values come from "
        "lm-eval, except MATH-500, which uses answer-prefix symbolic equivalence. The stock "
        "MATH exact score remains in the CSV as a format-sensitive diagnostic. Macro rows are "
        "unweighted task means and do not estimate training-seed variance.",
        "",
        "| condition | branch | macro | score | delta vs control | instruction - direct |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in macros:
        lines.append(
            f"| `{row['condition']}` | `{row['branch']}` | `{row['macro']}` | "
            f"{float(row['value']):.4f} | {float(row['delta_vs_control']):+.4f} | "
            f"{float(row['instruction_minus_direct']):+.4f} |"
        )
    targeted = {"agieval_logiqa_en"} | set(TARGETED_METRICS)
    lines.extend(
        [
            "",
            "## Targeted task results",
            "",
            "| condition | branch | task | score | stderr | delta vs control | instruction - direct |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in tasks:
        if row["task"] not in targeted:
            continue
        stderr = "N/A" if row["stderr"] is None else f"{float(row['stderr']):.4f}"
        lines.append(
            f"| `{row['condition']}` | `{row['branch']}` | `{row['task']}` | "
            f"{float(row['value']):.4f} | {stderr} | "
            f"{float(row['delta_vs_control']):+.4f} | "
            f"{float(row['instruction_minus_direct']):+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Generation diagnostics",
            "",
            "The next-document marker is a literal continuation into the Qwen assistant "
            "preamble after an answer. Character lengths are diagnostics, not token counts.",
            "",
            "| condition | branch | family | rows | invalid extraction | response marker | prompt marker | chars p50 | chars p95 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in diagnostics:
        lines.append(
            f"| `{row['condition']}` | `{row['branch']}` | `{row['family']}` | "
            f"{row['row_count']} | {float(row['invalid_extraction_rate']):.4f} | "
            f"{float(row['next_document_marker_rate']):.4f} | "
            f"{float(row['prompt_marker_rate']):.4f} | "
            f"{float(row['response_chars_p50']):.0f} | {row['response_chars_p95']} |"
        )
    lines.extend(["", "## Qualitative index", ""])
    for row in qualitative:
        lines.append(
            f"- `{row['condition']}/{row['branch']}/{row['task']}/{row['label']}`: "
            f"line {row['line_number']}, doc {row['doc_id']}, score={row['score']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    bundles = load_bundles(args.root, checkpoint_step=args.checkpoint_step)
    tasks = task_rows(bundles)
    macros = macro_rows(tasks)
    diagnostics = generation_diagnostic_rows(bundles)
    qualitative = qualitative_rows(bundles)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_task.csv", tasks)
    _write_csv(args.output_dir / "macro_summary.csv", macros)
    _write_csv(args.output_dir / "generation_diagnostics.csv", diagnostics)
    (args.output_dir / "qualitative_samples.json").write_text(
        json.dumps(qualitative, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(args.output_dir / "summary.md", tasks, macros, diagnostics, qualitative)
    manifest = {
        "accepted": True,
        "root": str(args.root),
        "checkpoint_step": args.checkpoint_step,
        "conditions": list(CONDITIONS),
        "branches": list(BRANCHES),
        "primary_tasks": list(FINAL_TASKS),
        "math500_primary_metric": MATH_POSTHOC_METRIC,
        "math500_stock_metric": PRIMARY_METRICS["hendrycks_math500"],
        "targeted_tasks": list(TARGETED_METRICS),
        "macros": {name: list(tasks) for name, tasks in MACROS.items()},
        "bundle_count": len(bundles),
        "task_row_count": len(tasks),
        "generation_diagnostic_row_count": len(diagnostics),
        "qualitative_row_count": len(qualitative),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
