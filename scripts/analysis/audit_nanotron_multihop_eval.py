#!/usr/bin/env python3
"""Audit supplemental context-provided multi-hop QA lm-eval bundles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_TASKS = (
    "synthrlvl_longbench_hotpotqa_tagged",
    "synthrlvl_longbench_2wikimqa_tagged",
    "synthrlvl_longbench_musique_tagged",
    "synthrlvl_longbench_hotpotqa_standard",
    "synthrlvl_longbench_2wikimqa_standard",
    "synthrlvl_longbench_musique_standard",
)
MINIMUM_MODEL_LENGTH = 32_768
STOCK_INSTRUCTION = "Answer the question based on the given passages."
STOCK_PASSAGE_HEADER = "The following are given passages."


def _split_tasks(raw: str) -> list[str]:
    return [part for part in raw.replace(":", ",").replace(" ", ",").split(",") if part]


def _metric(payload: dict[str, Any], task: str, name: str) -> float | None:
    section = payload.get("results", {}).get(task)
    if not isinstance(section, dict):
        return None
    value = section.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


def _generation_request(row: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    arguments = row.get("arguments")
    if not isinstance(arguments, dict):
        return None, None
    request = arguments.get("gen_args_0")
    if not isinstance(request, dict):
        return None, None
    prompt = request.get("arg_0")
    kwargs = request.get("arg_1")
    return (prompt if isinstance(prompt, str) else None, kwargs if isinstance(kwargs, dict) else None)


def audit(run_dir: Path, *, mode: str, expected_tasks: list[str], require_full: bool) -> dict[str, Any]:
    errors: list[str] = []
    result_files = sorted(run_dir.rglob("results_*.json"))
    command_path = run_dir / "command.json"
    if len(result_files) != 1:
        errors.append(f"expected one result JSON, found {len(result_files)}")
        payload: dict[str, Any] = {}
    else:
        payload = json.loads(result_files[0].read_text(encoding="utf-8"))
    command = json.loads(command_path.read_text(encoding="utf-8")) if command_path.is_file() else {}
    if not command:
        errors.append("missing command metadata")

    command_tasks = command.get("tasks")
    if not isinstance(command_tasks, list) or sorted(command_tasks) != sorted(expected_tasks):
        errors.append(f"command tasks={command_tasks!r}, expected={expected_tasks!r}")
    command_args = command.get("cmd") if isinstance(command.get("cmd"), list) else []
    if "--include_path" not in command_args:
        errors.append("local task include path is absent from the command")
    model_lengths = [
        int(value.split("=", 1)[1])
        for value in command_args
        if isinstance(value, str) and value.startswith("max_model_len=")
    ]
    if model_lengths != [MINIMUM_MODEL_LENGTH]:
        errors.append(
            f"max_model_len={model_lengths!r}, expected exactly [{MINIMUM_MODEL_LENGTH}] "
            "to retain every measured LongBench prompt"
        )
    expects_chat = mode == "instruction"
    if ("--apply_chat_template" in command_args) != expects_chat:
        errors.append(f"chat-template flag does not match mode={mode}")
    if bool(payload.get("chat_template")) != expects_chat:
        errors.append(f"stored chat template does not match mode={mode}")
    if require_full and (payload.get("config", {}).get("limit") is not None or "--limit" in command_args):
        errors.append("production bundle uses an lm-eval limit")

    result_keys = set(payload.get("results", {}))
    missing = sorted(set(expected_tasks) - result_keys)
    if missing:
        errors.append(f"missing result tasks: {missing}")

    n_samples = payload.get("n-samples", {})
    sample_files = sorted(run_dir.rglob("samples_*.jsonl"))
    samples_by_task: dict[str, list[Path]] = {}
    for path in sample_files:
        stem = path.name.removeprefix("samples_").split("_202", 1)[0]
        samples_by_task.setdefault(stem, []).append(path)

    sample_rows = 0
    task_metrics: dict[str, dict[str, float | None]] = {}
    for task in expected_tasks:
        protocol = "strict_tagged" if task.endswith("_tagged") else "standard_short_answer"
        counts = n_samples.get(task)
        if not isinstance(counts, dict):
            errors.append(f"missing sample counts for {task}")
            continue
        original, effective = counts.get("original"), counts.get("effective")
        if not isinstance(original, int) or not isinstance(effective, int) or effective <= 0:
            errors.append(f"invalid sample counts for {task}: {counts!r}")
            continue
        if require_full and original != effective:
            errors.append(f"incomplete task {task}: {effective}/{original}")
        matching = samples_by_task.get(task, [])
        if len(matching) != 1:
            errors.append(f"expected one sample file for {task}, found {len(matching)}")
        else:
            rows = [json.loads(line) for line in matching[0].read_text(encoding="utf-8").splitlines() if line.strip()]
            sample_rows += len(rows)
            if len(rows) != effective or len({row.get("doc_id") for row in rows}) != effective:
                errors.append(f"sample coverage mismatch for {task}")
            prompt_errors: set[str] = set()
            expected_max_tokens = 64 if protocol == "strict_tagged" else 32
            for row in rows:
                prompt, kwargs = _generation_request(row)
                if prompt is None or kwargs is None:
                    prompt_errors.add("missing retained generation request")
                    continue
                if kwargs.get("max_gen_toks") != expected_max_tokens:
                    prompt_errors.add(
                        f"max_gen_toks={kwargs.get('max_gen_toks')!r}, expected {expected_max_tokens}"
                    )
                if "Question: Question:" in prompt:
                    prompt_errors.add("duplicated question prefix")
                if protocol == "strict_tagged":
                    passage_marker = "\nPassages:\n"
                    passage_body = prompt.split(passage_marker, 1)[1] if passage_marker in prompt else ""
                    if not passage_body:
                        prompt_errors.add("missing tagged passage block")
                    elif passage_body.lstrip().startswith(STOCK_INSTRUCTION):
                        prompt_errors.add("embedded stock wrapper remains inside tagged prompt")
                else:
                    passage_marker = f"{STOCK_PASSAGE_HEADER}\n"
                    passage_body = prompt.split(passage_marker, 1)[1] if passage_marker in prompt else ""
                    if not prompt.startswith(STOCK_INSTRUCTION) or not passage_body:
                        prompt_errors.add("missing stock prompt wrapper")
                    elif passage_body.lstrip().startswith(STOCK_INSTRUCTION):
                        prompt_errors.add("embedded stock wrapper remains inside standard prompt")
            errors.extend(f"{task}: {error}" for error in sorted(prompt_errors))
        metrics = {
            name: _metric(payload, task, name)
            for name in ("qa_f1_score,none", "qa_exact_match,none", "tag_found,none", "extracted_nonempty,none")
        }
        task_metrics[task] = metrics
        if metrics["qa_f1_score,none"] is None:
            errors.append(f"missing QA F1 for {task}")
        if protocol == "strict_tagged" and (
            metrics["tag_found,none"] is None or metrics["extracted_nonempty,none"] is None
        ):
            errors.append(f"missing extraction diagnostics for {task}")

    return {
        "accepted": not errors,
        "run_dir": str(run_dir),
        "mode": mode,
        "require_full": require_full,
        "expected_tasks": expected_tasks,
        "result_file": str(result_files[0]) if len(result_files) == 1 else None,
        "sample_file_count": len(sample_files),
        "sample_row_count": sample_rows,
        "max_model_len": model_lengths[0] if len(model_lengths) == 1 else None,
        "task_metrics": task_metrics,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("direct", "instruction"), required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--allow-limit", action="store_true")
    args = parser.parse_args()
    report = audit(
        args.run_dir,
        mode=args.mode,
        expected_tasks=_split_tasks(args.tasks),
        require_full=not args.allow_limit,
    )
    output = args.run_dir / "multihop_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
