from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[1] / "scripts" / "analysis" / "aggregate_hfsa_depth_scaling.py"
SPEC = importlib.util.spec_from_file_location("aggregate_hfsa_depth_scaling", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

FINAL_RE = MODULE.FINAL_RE
RunRecord = MODULE.RunRecord
final_records_complete = MODULE.final_records_complete


def _complete_metrics(template: str, *, greedy_ood: float, sampled_offset: float) -> dict[str, float]:
    valid = "citation_free_valid" if template == "logic" else "nl_logic_citation_free_valid"
    valid_pass = MODULE.valid_metric(template)
    joint_pass = MODULE.joint_metric(template)
    metrics: dict[str, float] = {
        "posthoc/prompts": len(MODULE.DEPTHS) * 32,
        "posthoc/sampled_generations_per_prompt": 16,
    }
    for depth in MODULE.DEPTHS:
        metrics[f"synthetic/step_{depth}/correct"] = 0.9 if depth <= 25 else greedy_ood
        metrics[f"synthetic/step_{depth}/{valid}"] = 0.8 if depth <= 25 else 0.5
        for index, k in enumerate(MODULE.K_VALUES):
            correct = min(1.0, sampled_offset + 0.1 * index)
            joint = max(0.0, correct - 0.1)
            metrics[f"synthetic_sampled/step_{depth}/correct_pass@{k}"] = correct
            metrics[f"synthetic_sampled/step_{depth}/{valid_pass}@{k}"] = correct
            metrics[f"synthetic_sampled/step_{depth}/{joint_pass}@{k}"] = joint
    for band in ("train", "ood", "hard_tail"):
        for index, k in enumerate(MODULE.K_VALUES):
            correct = min(1.0, sampled_offset + 0.1 * index)
            joint = max(0.0, correct - 0.1)
            metrics[f"synthetic_sampled/band_{band}/correct_pass@{k}"] = correct
            metrics[f"synthetic_sampled/band_{band}/{valid_pass}@{k}"] = correct
            metrics[f"synthetic_sampled/band_{band}/{joint_pass}@{k}"] = joint
    return metrics


def test_final_pattern_accepts_old_and_corrected_branchproof_names() -> None:
    names = (
        "sft_hfsa_depth_scaling_logic_train1to25_10k_seed3407_passk.json",
        "sft_branchproof_unique_v2_nl_exact_train1to10_10k_seed3409_passk.json",
    )
    assert all(FINAL_RE.match(name) for name in names)


def test_strict_grid_reports_missing_metrics() -> None:
    record = RunRecord(
        path=Path("sft_branchproof_unique_v2_logic_train1to5_10k_seed3407_passk.json"),
        template="logic",
        train_max=5,
        seed=3407,
        checkpoint_step=None,
        elapsed_seconds=1.0,
        metrics={},
    )
    problems = final_records_complete([record], strict_metrics=True)
    assert any("posthoc/prompts" in problem for problem in problems)
    assert any("synthetic/step_1/correct" in problem for problem in problems)


def test_grid_reports_duplicate_rows() -> None:
    record = RunRecord(
        path=Path("duplicate.json"),
        template="logic",
        train_max=5,
        seed=3407,
        checkpoint_step=None,
        elapsed_seconds=1.0,
        metrics={},
    )
    problems = final_records_complete([record, record])
    assert "duplicate final ('logic', 5, 3407)" in problems


def test_exports_greedy_and_all_sampling_budgets() -> None:
    logic = RunRecord(
        path=Path("logic.json"),
        template="logic",
        train_max=25,
        seed=3407,
        checkpoint_step=None,
        elapsed_seconds=3600.0,
        metrics=_complete_metrics("logic", greedy_ood=0.6, sampled_offset=0.5),
    )
    nl = RunRecord(
        path=Path("nl.json"),
        template="nl_exact",
        train_max=25,
        seed=3407,
        checkpoint_step=None,
        elapsed_seconds=3600.0,
        metrics=_complete_metrics("nl_exact", greedy_ood=0.4, sampled_offset=0.4),
    )

    run = MODULE.final_run_rows([logic])[0]
    assert run["greedy_ood_correct"] == 0.6
    assert run["ood_correct1"] == 0.5
    assert run["ood_joint4"] == 0.6
    assert run["ood_correct16"] == 0.9

    depth = MODULE.final_depth_rows([logic])[0]
    assert depth["greedy_correct"] == 0.9
    assert all(f"joint{k}" in depth for k in MODULE.K_VALUES)

    summary = MODULE.group_summary_rows([logic])[0]
    assert summary["greedy_ood_correct_mean"] == 0.6
    assert abs(summary["ood_joint8_mean"] - 0.7) < 1e-9

    delta = MODULE.paired_delta_rows([logic, nl])[0]
    assert abs(delta["delta_greedy_ood_correct"] - 0.2) < 1e-9
    assert abs(delta["delta_ood_correct4"] - 0.1) < 1e-9
