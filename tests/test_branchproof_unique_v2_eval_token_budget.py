from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "analysis"
    / "audit_branchproof_unique_v2_eval_token_budget.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_branchproof_unique_v2_eval_token_budget", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_summarize_reports_quantiles_and_headroom():
    summary = AUDIT.summarize([1, 2, 3, 4, 10], limit=8)

    assert summary == {
        "count": 5,
        "min": 1,
        "p50": 3,
        "p95": 10,
        "p99": 10,
        "max": 10,
        "limit": 8,
        "headroom": -2,
        "over_limit_count": 1,
        "over_limit_rate": 0.2,
    }


def test_summarize_rejects_empty_input():
    with pytest.raises(AssertionError, match="empty token lengths"):
        AUDIT.summarize([], limit=8)
