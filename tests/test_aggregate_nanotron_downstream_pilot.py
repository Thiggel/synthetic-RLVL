from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analysis" / "aggregate_nanotron_downstream_pilot.py"
SPEC = importlib.util.spec_from_file_location("aggregate_nanotron_downstream_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _payload(value: float) -> dict:
    results = {}
    groups = {}
    metrics = dict(MODULE.PRIMARY_METRICS)
    metrics.update(MODULE.TARGETED_METRICS)
    for task, metric in metrics.items():
        section = {metric: value, metric.replace(",", "_stderr,", 1): 0.01}
        (groups if task in {"bbh", "mmlu", "mmlu_pro"} else results)[task] = section
    return {"results": results, "groups": groups}


def _math_posthoc(value: float) -> dict:
    return {
        "accepted": True,
        "scorer": MODULE.MATH_POSTHOC_METRIC,
        "accuracy": value,
        "stderr": 0.02,
    }


def test_task_and_macro_deltas() -> None:
    values = {"control": 0.4, "logic": 0.6, "nl_exact": 0.5}
    bundles = [
        MODULE.Bundle(
            condition,
            branch,
            Path(f"/{condition}/{branch}"),
            _payload(value),
            _math_posthoc(value),
        )
        for condition, value in values.items()
        for branch in MODULE.BRANCHES
    ]

    tasks = MODULE.task_rows(bundles)
    macros = MODULE.macro_rows(tasks)

    logic_direct = next(
        row
        for row in macros
        if row["condition"] == "logic"
        and row["branch"] == "direct"
        and row["macro"] == "logic_targeted"
    )
    assert logic_direct["value"] == 0.6
    assert abs(logic_direct["delta_vs_control"] - 0.2) < 1e-9
    assert logic_direct["instruction_minus_direct"] == 0.0
    expected_tasks = set(MODULE.PRIMARY_METRICS) | set(MODULE.TARGETED_METRICS)
    assert len(tasks) == 6 * len(expected_tasks)
    logic_task = next(
        row
        for row in tasks
        if row["condition"] == "logic"
        and row["branch"] == "instruction"
        and row["task"] == "mmlu_formal_logic"
    )
    assert logic_task["instruction_minus_direct"] == 0.0


def test_rejects_missing_primary_metric() -> None:
    payload = _payload(0.5)
    del payload["results"]["gsm8k"][MODULE.PRIMARY_METRICS["gsm8k"]]
    bundle = MODULE.Bundle(
        "control",
        "direct",
        Path("/control/direct"),
        payload,
        _math_posthoc(0.5),
    )
    try:
        MODULE.task_rows([bundle])
    except ValueError as exc:
        assert "gsm8k" in str(exc)
    else:
        raise AssertionError("missing metric was accepted")


def test_qualitative_metric_uses_declared_filter() -> None:
    assert MODULE._qualitative_metric("gsm8k") == ("exact_match", "flexible-extract")
    assert MODULE._qualitative_metric("mmlu_pro_computer_science") == (
        "exact_match",
        "custom-extract",
    )
