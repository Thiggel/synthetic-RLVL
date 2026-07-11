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
