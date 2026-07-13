#!/usr/bin/env python3
"""Audit a production Nanotron downstream lm-eval artifact bundle."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rescore_math500 import METRIC_NAME as MATH_POSTHOC_METRIC
from rescore_math500 import SIDECAR_NAME as MATH_POSTHOC_SIDECAR
from rescore_math500 import ensure_sidecar


FINAL_TASKS = (
    "gsm8k",
    "hendrycks_math500",
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "piqa",
    "agieval_logiqa_en",
    "bbh",
    "mmlu",
    "mmlu_pro",
)

PRIMARY_METRICS = {
    "gsm8k": "exact_match,flexible-extract",
    "hendrycks_math500": "exact_match,none",
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "winogrande": "acc,none",
    "piqa": "acc_norm,none",
    "agieval_logiqa_en": "acc_norm,none",
    "bbh": "exact_match,get-answer",
    "mmlu": "acc,none",
    "mmlu_formal_logic": "acc,none",
    "mmlu_pro": "exact_match,custom-extract",
}


def _split_tasks(raw: str) -> list[str]:
    return [part for part in raw.replace(":", ",").replace(" ", ",").split(",") if part]


def _first_prompt(sample_path: Path) -> str:
    with sample_path.open("r", encoding="utf-8") as handle:
        row = json.loads(next(line for line in handle if line.strip()))
    arguments = row.get("arguments") or {}
    if not isinstance(arguments, dict):
        return ""
    for value in arguments.values():
        if not isinstance(value, dict):
            continue
        for key, candidate in value.items():
            if str(key).startswith("arg_") and isinstance(candidate, str):
                return candidate
    return ""


def _metric_value(payload: dict[str, Any], task: str, metric: str) -> float | None:
    section = payload.get("results", {}).get(task)
    if not isinstance(section, dict):
        section = payload.get("groups", {}).get(task)
    if not isinstance(section, dict):
        return None
    value = section.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


def audit_run(
    run_dir: Path,
    *,
    mode: str,
    expected_tasks: list[str],
    require_full: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    result_files = sorted(run_dir.rglob("results_*.json"))
    sample_files = sorted(run_dir.rglob("samples_*.jsonl"))
    command_path = run_dir / "command.json"

    if len(result_files) != 1:
        errors.append(f"expected exactly one result JSON, found {len(result_files)}")
        payload: dict[str, Any] = {}
    else:
        try:
            payload = json.loads(result_files[0].read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"could not read result JSON: {exc}")
            payload = {}

    if not command_path.is_file():
        errors.append(f"missing command metadata: {command_path}")
        command: dict[str, Any] = {}
    else:
        try:
            command = json.loads(command_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"could not read command metadata: {exc}")
            command = {}

    command_tasks = command.get("tasks")
    if not isinstance(command_tasks, list) or sorted(command_tasks) != sorted(
        expected_tasks
    ):
        errors.append(f"command tasks={command_tasks!r}, expected {expected_tasks!r}")

    command_args = command.get("cmd") if isinstance(command.get("cmd"), list) else []
    expects_chat = mode in {"chat", "instruction"}
    has_chat_flag = "--apply_chat_template" in command_args
    if has_chat_flag != expects_chat:
        errors.append(
            f"chat-template command flag={has_chat_flag}, expected {expects_chat} for mode={mode}"
        )

    combined_keys = set(payload.get("results", {})) | set(payload.get("groups", {}))
    missing_tasks = sorted(set(expected_tasks) - combined_keys)
    if missing_tasks:
        errors.append(f"missing required result/group keys: {missing_tasks}")

    chat_template = payload.get("chat_template")
    if bool(chat_template) != expects_chat:
        errors.append(
            f"stored chat template present={bool(chat_template)}, expected {expects_chat}"
        )

    limit = payload.get("config", {}).get("limit")
    if require_full and limit is not None:
        errors.append(f"production result has lm-eval limit={limit!r}")
    if require_full and "--limit" in command_args:
        errors.append("production command contains --limit")

    n_samples = payload.get("n-samples")
    if not isinstance(n_samples, dict) or not n_samples:
        errors.append("missing n-samples metadata")
        n_samples = {}

    samples_by_task: dict[str, list[Path]] = {}
    for path in sample_files:
        stem = path.name.removeprefix("samples_").split("_202", 1)[0]
        samples_by_task.setdefault(stem, []).append(path)
        if path.stat().st_size == 0:
            errors.append(f"empty sample file: {path}")

    sample_rows = 0
    for task, counts in n_samples.items():
        if not isinstance(counts, dict):
            errors.append(f"invalid n-samples entry for {task}: {counts!r}")
            continue
        original = counts.get("original")
        effective = counts.get("effective")
        if not isinstance(original, int) or not isinstance(effective, int) or effective <= 0:
            errors.append(f"invalid sample counts for {task}: {counts!r}")
            continue
        if require_full and effective != original:
            errors.append(f"incomplete sample count for {task}: {effective}/{original}")
        matching = samples_by_task.get(task, [])
        if len(matching) != 1:
            errors.append(f"expected one sample file for {task}, found {len(matching)}")
            continue
        document_ids = set()
        rows = 0
        with matching[0].open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows += 1
                row = json.loads(line)
                document_ids.add(row.get("doc_id"))
        sample_rows += rows
        if len(document_ids) != effective:
            errors.append(
                f"unique sample documents for {task}: {len(document_ids)}, expected {effective}"
            )

    math_posthoc: dict[str, Any] | None = None
    math_counts = n_samples.get("hendrycks_math500")
    math_expected = math_counts.get("effective") if isinstance(math_counts, dict) else None
    if isinstance(math_expected, int) and math_expected > 0:
        try:
            math_posthoc = ensure_sidecar(run_dir, expected_count=math_expected)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"could not build MATH-500 post-hoc sidecar: {exc}")
        else:
            if not math_posthoc.get("accepted"):
                errors.append(
                    f"MATH-500 post-hoc scorer rejected bundle: {math_posthoc.get('errors')}"
                )
            if math_posthoc.get("row_count") != math_expected:
                errors.append(
                    "MATH-500 post-hoc row count "
                    f"{math_posthoc.get('row_count')} != expected {math_expected}"
                )
            value = math_posthoc.get("accuracy")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                errors.append(f"invalid MATH-500 post-hoc accuracy: {value!r}")
    else:
        errors.append("missing effective sample count for MATH-500 post-hoc scoring")

    for task, metric in PRIMARY_METRICS.items():
        if _metric_value(payload, task, metric) is None:
            errors.append(f"missing or invalid primary metric: {task}/{metric}")

    gsm8k_samples = samples_by_task.get("gsm8k", [])
    if len(gsm8k_samples) == 1:
        prompt = _first_prompt(gsm8k_samples[0])
        has_qwen_chat = "<|im_start|>user" in prompt and "<|im_start|>assistant" in prompt
        if has_qwen_chat != expects_chat:
            errors.append(
                f"retained prompt has Qwen chat rendering={has_qwen_chat}, expected {expects_chat}"
            )

    report = {
        "accepted": not errors,
        "run_dir": str(run_dir),
        "mode": mode,
        "require_full": require_full,
        "result_file": str(result_files[0]) if len(result_files) == 1 else None,
        "expected_tasks": expected_tasks,
        "result_or_group_keys": sorted(combined_keys),
        "leaf_task_count": len(n_samples),
        "sample_file_count": len(sample_files),
        "sample_row_count": sample_rows,
        "chat_template_applied": bool(chat_template),
        "math500_posthoc": (
            {
                key: math_posthoc.get(key)
                for key in (
                    "accepted",
                    "scorer",
                    "row_count",
                    "correct_count",
                    "accuracy",
                    "stderr",
                    "stock_exact_correct_count",
                    "stock_exact_accuracy",
                    "rescued_count",
                    "lost_stock_exact_count",
                    "sample_sha256",
                )
            }
            if math_posthoc is not None
            else None
        ),
        "math500_posthoc_metric": MATH_POSTHOC_METRIC,
        "math500_posthoc_sidecar": str(run_dir / MATH_POSTHOC_SIDECAR),
        "errors": errors,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("direct", "chat", "instruction"), required=True)
    parser.add_argument("--tasks", default=",".join(FINAL_TASKS))
    parser.add_argument("--allow-limit", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or args.run_dir / "production_audit.json"
    report = audit_run(
        args.run_dir,
        mode=args.mode,
        expected_tasks=_split_tasks(args.tasks),
        require_full=not args.allow_limit,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
